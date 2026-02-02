import torch
from einops import rearrange
from flash_attn import flash_attn_func as flash_attn_func_v2
from liger_kernel.ops.rope import LigerRopeFunction
from torch import Tensor
from typing import Tuple, Union

# Intentar importar Flash Attention v3 para hardware H100+
try:
    from flash_attn_interface import flash_attn_func as flash_attn_func_v3
    SUPPORT_FA3 = True
except ImportError:
    SUPPORT_FA3 = False

# --- CORE ATTENTION ---

def flash_attn_func(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """Selecciona la mejor versión de Flash Attention disponible."""
    if SUPPORT_FA3:
        # FA3 devuelve una tupla, el primer elemento es el output
        return flash_attn_func_v3(q, k, v)[0]
    return flash_attn_func_v2(q, k, v)

def attention(q: Tensor, k: Tensor, v: Tensor, pe: Union[Tensor, Tuple[Tensor, Tensor]]) -> Tensor:
    """
    Motor de atención principal.
    Aplica RoPE (Positional Embeddings) y luego Flash Attention.
    """
    # 1. Aplicar Positional Embeddings (RoPE)
    if isinstance(pe, torch.Tensor):
        # Implementación estándar (e.g., para inferencia clásica)
        q, k = apply_rope(q, k, pe)
    else:
        # Implementación Liger (CST) optimizada para memoria contigua
        cos, sin = pe
        q, k = LigerRopeFunction.apply(q, k, cos, sin)
    
    # 2. Reordenar para Flash Attention (B L H D)
    q, k, v = [rearrange(x, "B H L D -> B L H D") for x in (q, k, v)]
    
    # 3. Calcular Atención Eficiente
    x = flash_attn_func(q, k, v)
    
    # 4. Proyectar de vuelta al espacio del modelo
    x = rearrange(x, "B L H D -> B L (H D)")
    return x

# --- ROTARY POSITIONAL EMBEDDINGS (RoPE) ---

def liger_rope(pos: Tensor, dim: int, theta: int) -> Tuple[Tensor, Tensor]:
    """Genera cosenos y senos para Liger Rope (Eficiencia en entrenamiento)."""
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)  # Producto exterior
    return (out.cos(), out.sin())

def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    """Genera matriz de rotación completa para RoPE estándar."""
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    # Crea la matriz de rotación [[cos, -sin], [sin, cos]]
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()

def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> Tuple[Tensor, Tensor]:
    """Aplica la rotación compleja a los tensores Q y K."""
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    
    # Rotación compleja manual: (a+bi)(c+di)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

# --- UTILIDADES DE REORDENAMIENTO ---

def rearrange_tensor(tensor: Tensor) -> Tensor:
    """Reorganiza la dimensión D para intercalar componentes reales e imaginarios."""
    B, H, L, D = tensor.shape
    if D % 2 != 0:
        raise ValueError("La última dimensión D debe ser par.")

    half_D = D // 2
    indices = torch.empty(D, dtype=torch.long, device=tensor.device)
    indices[:half_D] = torch.arange(0, D, 2, device=tensor.device)
    indices[half_D:] = torch.arange(1, D, 2, device=tensor.device)

    return tensor.index_select(dim=-1, index=indices)

def reverse_rearrange_tensor(tensor: Tensor) -> Tensor:
    """Restaura el orden original de la dimensión D."""
    B, H, L, D = tensor.shape
    if D % 2 != 0:
        raise ValueError("La última dimensión D debe ser par.")

    half_D = D // 2
    reverse_indices = torch.empty(D, dtype=torch.long, device=tensor.device)
    reverse_indices[::2] = torch.arange(half_D, device=tensor.device)
    reverse_indices[1::2] = torch.arange(half_D, D, device=tensor.device)

    return tensor.index_select(dim=-1, index=reverse_indices)