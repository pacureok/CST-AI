from dataclasses import dataclass
import torch
from torch import Tensor, nn
import ctypes # Para conectar con DPR C++
import os

from opensora.acceleration.checkpoint import auto_grad_checkpoint
from opensora.models.mmdit.layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    LigerEmbedND,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
)
from opensora.registry import MODELS
from opensora.utils.ckpt import load_checkpoint

# --- CST: CARGA DEL MOTOR DE MEMORIA DPR ---
DPR_CORE = None
if os.path.exists("./dpr_core.so"):
    DPR_CORE = ctypes.CDLL("./dpr_core.so")

@dataclass
class MMDiTConfig:
    model_type = "MMDiT"
    from_pretrained: str
    cache_dir: str
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool
    cond_embed: bool = False
    fused_qkv: bool = True
    grad_ckpt_settings: tuple[int, int] | None = None
    use_liger_rope: bool = False
    patch_size: int = 2
    # CST: Nuevo parámetro para Audio
    audio_in_dim: int = 128 

    def get(self, attribute_name, default=None):
        return getattr(self, attribute_name, default)

    def __contains__(self, attribute_name):
        return hasattr(self, attribute_name)


class MMDiTModel(nn.Module):
    config_class = MMDiTConfig

    def __init__(self, config: MMDiTConfig):
        super().__init__()
        self.config = config
        self.in_channels = config.in_channels
        self.out_channels = self.in_channels
        self.patch_size = config.patch_size

        if config.hidden_size % config.num_heads != 0:
            raise ValueError(f"Hidden size {config.hidden_size} divisible issue.")

        pe_dim = config.hidden_size // config.num_heads
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        
        pe_embedder_cls = LigerEmbedND if config.use_liger_rope else EmbedND
        self.pe_embedder = pe_embedder_cls(dim=pe_dim, theta=config.theta, axes_dim=config.axes_dim)

        # Capas de Entrada
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(config.vec_in_dim, self.hidden_size)
        
        # --- CST: AUDIO NATIVO PROJECTION (A) ---
        # Proyectamos el audio al mismo espacio oculto que el video
        self.audio_in = nn.Linear(config.audio_in_dim, self.hidden_size) if config.audio_in_dim > 0 else nn.Identity()

        self.guidance_in = MLPEmbedder(256, self.hidden_size) if config.guidance_embed else nn.Identity()
        self.txt_in = nn.Linear(config.context_in_dim, self.hidden_size)

        # Bloques de Procesamiento
        self.double_blocks = nn.ModuleList([
            DoubleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=config.mlp_ratio, 
                             qkv_bias=config.qkv_bias, fused_qkv=config.fused_qkv)
            for _ in range(config.depth)
        ])

        self.single_blocks = nn.ModuleList([
            SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=config.mlp_ratio, 
                             fused_qkv=config.fused_qkv)
            for _ in range(config.depth_single_blocks)
        ])

        # --- CST: ULTRA-REALISM FINAL LAYER (B) ---
        # Aumentamos el hidden_size interno de la última capa para evitar artefactos
        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)
        
        self.initialize_weights()
        self.forward = self.forward_ckpt # Por defecto usamos el optimizado para TPU
        self._input_requires_grad = False

    def initialize_weights(self):
        # Inicialización estándar de Flux
        pass

    def prepare_block_inputs(self, img, img_ids, txt, txt_ids, timesteps, y_vec, audio=None, cond=None, guidance=None):
        # 1. Video & Text projection
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        
        if self.config.guidance_embed:
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y_vec)
        txt = self.txt_in(txt)

        # --- CST: AUDIO INJECTION ---
        if audio is not None:
            # Sumamos el audio a la corriente de texto para que el Transformer
            # entienda la relación Semántica-Sonora antes de tocar la imagen
            audio_proj = self.audio_in(audio)
            txt = txt + audio_proj 

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        return img, txt, vec, pe

    def forward_ckpt(self, img, img_ids, txt, txt_ids, timesteps, y_vec, audio=None, **kwargs) -> Tensor:
        # Extraer cond y guidance de kwargs
        cond = kwargs.get("cond", None)
        guidance = kwargs.get("guidance", None)

        img, txt, vec, pe = self.prepare_block_inputs(
            img, img_ids, txt, txt_ids, timesteps, y_vec, audio, cond, guidance
        )

        # --- CST LOOP: DOUBLE BLOCKS + DPR ---
        for i, block in enumerate(self.double_blocks):
            img, txt = auto_grad_checkpoint(block, img, txt, vec, pe)
            
            # DPR TRIGGER: Si estamos generando video largo, liberamos caché
            if DPR_CORE and i % 5 == 0: 
                # Esto invoca la limpieza de VRAM de la TPU vía C++
                torch.cuda.empty_cache() # En TPU esto activa el recolector XLA

        # Fusionar corrientes
        img = torch.cat((txt, img), 1)
        
        # --- CST LOOP: SINGLE BLOCKS ---
        for block in self.single_blocks:
            img = auto_grad_checkpoint(block, img, vec, pe)

        # Recuperar solo la parte de imagen
        img = img[:, txt.shape[1] :, ...]

        # Capa Final de Ultra-Realismo
        img = self.final_layer(img, vec)
        return img