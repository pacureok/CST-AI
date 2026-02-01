import torch
def memory_efficient_attention(q, k, v, **kwargs):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)
