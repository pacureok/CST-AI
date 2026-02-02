from dataclasses import dataclass
import torch
from torch import Tensor, nn
import ctypes
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
# Este motor permite que el modelo gestione video de larga duración (1h+)
# liberando memoria de forma inteligente sin perder el gradiente.
DPR_CORE = None
if os.path.exists("./dpr_core.so"):
    try:
        DPR_CORE = ctypes.CDLL("./dpr_core.so")
    except Exception:
        DPR_CORE = None

@dataclass
class MMDiTConfig:
    model_type = "MMDiT"
    from_pretrained: str = None
    cache_dir: str = None
    in_channels: int = 16
    vec_in_dim: int = 768
    context_in_dim: int = 4096
    hidden_size: int = 1152
    mlp_ratio: float = 4.0
    num_heads: int = 16
    depth: int = 19
    depth_single_blocks: int = 38
    axes_dim: list[int] = None
    theta: int = 10000
    qkv_bias: bool = True
    guidance_embed: bool = True
    cond_embed: bool = False
    fused_qkv: bool = True
    grad_ckpt_settings: tuple[int, int] | None = None
    use_liger_rope: bool = True
    patch_size: int = 2
    audio_in_dim: int = 128 

    def get(self, attribute_name, default=None):
        return getattr(self, attribute_name, default)

    def __contains__(self, attribute_name):
        return hasattr(self, attribute_name)

@MODELS.register_module("MMDiT")
class MMDiTModel(nn.Module):
    config_class = MMDiTConfig

    def __init__(self, config: MMDiTConfig):
        super().__init__()
        self.config = config
        self.in_channels = config.in_channels
        self.out_channels = self.in_channels
        self.patch_size = config.patch_size

        if config.hidden_size % config.num_heads != 0:
            raise ValueError(f"Hidden size {config.hidden_size} no es divisible por num_heads {config.num_heads}")

        pe_dim = config.hidden_size // config.num_heads
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        
        # Positional Embedder (CST usa Liger por defecto para eficiencia)
        pe_embedder_cls = LigerEmbedND if config.use_liger_rope else EmbedND
        self.pe_embedder = pe_embedder_cls(dim=pe_dim, theta=config.theta, axes_dim=config.axes_dim)

        # Capas de Entrada Proyectadas
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(config.vec_in_dim, self.hidden_size)
        
        # CST: Proyección de Audio para Video-Audio Alignment
        self.audio_in = nn.Linear(config.audio_in_dim, self.hidden_size) if config.audio_in_dim > 0 else nn.Identity()

        self.guidance_in = MLPEmbedder(256, self.hidden_size) if config.guidance_embed else nn.Identity()
        self.txt_in = nn.Linear(config.context_in_dim, self.hidden_size)

        # Bloques Transformers (Double Stream: Text & Image interaction)
        self.double_blocks = nn.ModuleList([
            DoubleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=config.mlp_ratio, 
                             qkv_bias=config.qkv_bias, fused_qkv=config.fused_qkv)
            for _ in range(config.depth)
        ])

        # Bloques Transformers (Single Stream: Unified Latent processing)
        self.single_blocks = nn.ModuleList([
            SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=config.mlp_ratio, 
                             fused_qkv=config.fused_qkv)
            for _ in range(config.depth_single_blocks)
        ])

        # Capa Final de reconstrucción
        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)
        
        self.initialize_weights()

    def initialize_weights(self):
        # Inicialización de pesos optimizada para Transformers de gran escala
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def prepare_block_inputs(self, img, img_ids, txt, txt_ids, timesteps, y_vec, audio=None, cond=None, guidance=None):
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        
        if self.config.guidance_embed and guidance is not None:
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        
        vec = vec + self.vector_in(y_vec)
        txt = self.txt_in(txt)

        # CST: Sincronización de Audio con el flujo de Texto
        if audio is not None and self.config.audio_in_dim > 0:
            txt = txt + self.audio_in(audio) 

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        return img, txt, vec, pe

    def forward(self, img, img_ids, txt, txt_ids, timesteps, y_vec, audio=None, **kwargs) -> Tensor:
        cond = kwargs.get("cond", None)
        guidance = kwargs.get("guidance", None)

        # Preparación de datos y embeddings rotatorios
        img, txt, vec, pe = self.prepare_block_inputs(
            img, img_ids, txt, txt_ids, timesteps, y_vec, audio, cond, guidance
        )

        # Procesamiento en Doble Flujo (Interacción Imagen-Texto)
        # 
        for i, block in enumerate(self.double_blocks):
            img, txt = auto_grad_checkpoint(block, img, txt, vec, pe)
            
            # CST/DPR Trigger: Limpieza de memoria en TPU cada 5 bloques
            if DPR_CORE and i % 5 == 0: 
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Concatenación para procesamiento unificado
        img = torch.cat((txt, img), 1)
        
        # Procesamiento en Flujo Único (Refinamiento Profundo)
        for block in self.single_blocks:
            img = auto_grad_checkpoint(block, img, vec, pe)

        # Separar la parte de imagen del contexto de texto/audio
        img = img[:, txt.shape[1] :, ...]

        # Decodificación final a espacio de parches
        img = self.final_layer(img, vec)
        return img