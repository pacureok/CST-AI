import torch
def flash_attn_func(q, k, v, dropout_p=0.0, **kwargs):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
