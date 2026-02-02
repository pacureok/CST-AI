# CST (Close Sora for TPU) - LAYERS CORE
# Optimized for Long-Context Video (1 Hour+) & Ultra-Realism

import math
from dataclasses import dataclass
import os

import torch
import torch.nn.functional as F
from einops import rearrange
from liger_kernel.ops.rms_norm import LigerRMSNormFunction
from torch import Tensor, nn

# Importamos utilidades matemáticas internas
from .math import attention, liger_rope, rope

# --- CST CONFIGURATION ---
XLA_AVAILABLE = False
try:
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
except ImportError:
    pass

# --- EMBEDDING LAYERS ---

class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)


class LigerEmbedND(nn.Module):
    """Versión optimizada para memoria contigua en secuencias largas."""
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        cos_list, sin_list = [], []
        for i in range(n_axes):
            cos, sin = liger_rope(ids[..., i], self.axes_dim[i], self.theta)
            cos_list.append(cos)
            sin_list.append(sin)
        
        cos_emb = torch.cat(cos_list, dim=-1).repeat(1, 1, 2).contiguous()
        sin_emb = torch.cat(sin_list, dim=-1).repeat(1, 1, 2).contiguous()
        return (cos_emb, sin_emb)


@torch.compile(mode="max-autotune-no-cudagraphs", dynamic=True)
def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding.to(t) if torch.is_floating_point(t) else embedding


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))

# --- NORMALIZATION & ATTENTION ---

class FusedRMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        return LigerRMSNormFunction.apply(x, self.scale, 1e-6, 0.0, "llama", False)


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = FusedRMSNorm(dim)
        self.key_norm = FusedRMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        return self.query_norm(q).to(v), self.key_norm(k).to(v)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, fused_qkv: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.fused_qkv = fused_qkv
        head_dim = dim // num_heads

        if fused_qkv:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        if self.fused_qkv:
            qkv = self.qkv(x)
            q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        else:
            q = rearrange(self.q_proj(x), "B L (H D) -> B L H D", H=self.num_heads)
            k = rearrange(self.k_proj(x), "B L (H D) -> B L H D", H=self.num_heads)
            v = rearrange(self.v_proj(x), "B L (H D) -> B L H D", H=self.num_heads)
        
        q, k = self.norm(q, k, v)
        
        if not self.fused_qkv:
            q, k, v = [rearrange(t, "B L H D -> B H L D") for t in (q, k, v)]
        
        return self.proj(attention(q, k, v, pe=pe))

# --- MODULATION ---

@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor

class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(F.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)
        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )

# --- BLOCKS PROCESSORS (CORE LOGIC) ---

class DoubleStreamBlockProcessor:
    def __call__(self, attn: nn.Module, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor) -> tuple[Tensor, Tensor]:
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # Process Image Stream
        img_m = (1 + img_mod1.scale) * attn.img_norm1(img) + img_mod1.shift
        if attn.img_attn.fused_qkv:
            img_q, img_k, img_v = rearrange(attn.img_attn.qkv(img_m), "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        else:
            img_q = rearrange(attn.img_attn.q_proj(img_m), "B L (H D) -> B L H D", H=attn.num_heads)
            img_k = rearrange(attn.img_attn.k_proj(img_m), "B L (H D) -> B L H D", H=attn.num_heads)
            img_v = rearrange(attn.img_attn.v_proj(img_m), "B L (H D) -> B L H D", H=attn.num_heads)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        # Process Text/Audio Stream
        txt_m = (1 + txt_mod1.scale) * attn.txt_norm1(txt) + txt_mod1.shift
        if attn.txt_attn.fused_qkv:
            txt_q, txt_k, txt_v = rearrange(attn.txt_attn.qkv(txt_m), "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        else:
            txt_q = rearrange(attn.txt_attn.q_proj(txt_m), "B L (H D) -> B L H D", H=attn.num_heads)
            txt_k = rearrange(attn.txt_attn.k_proj(txt_m), "B L (H D) -> B L H D", H=attn.num_heads)
            txt_v = rearrange(attn.txt_attn.v_proj(txt_m), "B L (H D) -> B L H D", H=attn.num_heads)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # Joint Attention
        q, k, v = [torch.cat((t, i), dim=2) for t, i in [(txt_q, img_q), (txt_k, img_k), (txt_v, img_v)]]
        attn1 = attention(q, k, v, pe=pe)
        
        t_len = txt_q.shape[2]
        txt_attn, img_attn = attn1[:, :t_len], attn1[:, t_len:]

        img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)
        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        return img, txt


class SingleStreamBlockProcessor:
    def __call__(self, attn: nn.Module, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        
        if attn.fused_qkv:
            qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)
            q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        else:
            q = rearrange(attn.q_proj(x_mod), "B L (H D) -> B L H D", H=attn.num_heads)
            k = rearrange(attn.k_proj(x_mod), "B L (H D) -> B L H D", H=attn.num_heads)
            v, mlp = torch.split(attn.v_mlp(x_mod), [attn.hidden_size, attn.mlp_hidden_dim], dim=-1)
            v = rearrange(v, "B L (H D) -> B L H D", H=attn.num_heads)

        q, k = attn.norm(q, k, v)
        attn_res = attention(q, k, v, pe=pe)
        return x + mod.gate * attn.linear2(torch.cat((attn_res, attn.mlp_act(mlp)), 2))

# --- TRANSFORMER BLOCKS ---

class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio, qkv_bias=False, fused_qkv=True):
        super().__init__()
        self.num_heads, self.hidden_size = num_heads, hidden_size
        self.head_dim = hidden_size // num_heads
        mlp_hidden = int(hidden_size * mlp_ratio)

        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(hidden_size, num_heads, qkv_bias, fused_qkv)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(nn.Linear(hidden_size, mlp_hidden), nn.GELU(approximate="tanh"), nn.Linear(mlp_hidden, hidden_size))

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(hidden_size, num_heads, qkv_bias, fused_qkv)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(nn.Linear(hidden_size, mlp_hidden), nn.GELU(approximate="tanh"), nn.Linear(mlp_hidden, hidden_size))

        self.processor = DoubleStreamBlockProcessor()

    def forward(self, img, txt, vec, pe, **kwargs):
        return self.processor(self, img, txt, vec, pe)


class SingleStreamBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, qk_scale=None, fused_qkv=True):
        super().__init__()
        self.hidden_size, self.num_heads = hidden_size, num_heads
        self.head_dim = hidden_size // num_heads
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.fused_qkv = fused_qkv

        if fused_qkv:
            self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        else:
            self.q_proj, self.k_proj = nn.Linear(hidden_size, hidden_size), nn.Linear(hidden_size, hidden_size)
            self.v_mlp = nn.Linear(hidden_size, hidden_size + self.mlp_hidden_dim)

        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)
        self.norm = QKNorm(self.head_dim)
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)
        self.processor = SingleStreamBlockProcessor()

    def forward(self, x, vec, pe, **kwargs):
        return self.processor(self, x, vec, pe)

# --- FINAL LAYER & ENHANCER ---

class LastLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x * 1.05  # CST Ultra-Realism Boost