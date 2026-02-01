import torch
import torch.nn as nn

# Este es el puente que convierte pedidos de GPU en operaciones XLA
def flash_attn_func(q, k, v, dropout_p=0.0, software_fallback=True, **kwargs):
    # Redirigimos a la implementación nativa de PyTorch optimizada para TPU
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, dropout_p=dropout_p
    )

class FlashAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, q, k, v, **kwargs):
        return flash_attn_func(q, k, v)

# Mocking de sub-módulos para engañar a Open-Sora
flash_attn_varlen_func = flash_attn_func