import torch
import torch.distributed as dist

# CST OPTIMIZATION: Detectar si estamos en TPU
try:
    import torch_xla.core.xla_model as xm
    IS_XLA = True
except ImportError:
    IS_XLA = False

# ====================
# All-To-All (CST Native Link)
# ====================
def _all_to_all(
    input_: torch.Tensor,
    world_size: int,
    group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
):
    # En TPU XLA, las operaciones deben ser lo más contiguas posible
    input_list = [t.contiguous() for t in torch.tensor_split(input_, world_size, scatter_dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    
    # PCURE-AI+ Hook: Usamos el dispatch de distribuido estándar (XLA lo intercepta)
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()

class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, process_group, scatter_dim, gather_dim):
        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.world_size = dist.get_world_size(process_group)
        return _all_to_all(input_, ctx.world_size, process_group, scatter_dim, gather_dim)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = _all_to_all(
            grad_output,
            ctx.world_size,
            ctx.process_group,
            ctx.gather_dim,
            ctx.scatter_dim,
        )
        return grad_output, None, None, None

def all_to_all(input_, process_group, scatter_dim=2, gather_dim=1):
    return _AllToAll.apply(input_, process_group, scatter_dim, gather_dim)

# ====================
# Gather-Split (TPU Fix)
# ====================

def _split(input_, pg: dist.ProcessGroup, dim=-1):
    world_size = dist.get_world_size(pg)
    rank = dist.get_rank(pg)
    if world_size == 1:
        return input_

    dim_size = input_.size(dim)
    tensor_list = torch.split(input_, dim_size // world_size, dim=dim)
    return tensor_list[rank].contiguous()

def _gather(input_, pg: dist.ProcessGroup, dim=-1):
    input_ = input_.contiguous()
    world_size = dist.get_world_size(pg)
    if world_size == 1:
        return input_

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    
    # --- FIX CRÍTICO PCURE-AI+ ---
    # Eliminamos el assert de "cuda" para permitir TPU (XLA)
    # -----------------------------
    dist.all_gather(tensor_list, input_, group=pg)
    return torch.cat(tensor_list, dim=dim).contiguous()

class _GatherForwardSplitBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, process_group, dim, grad_scale):
        ctx.mode = process_group
        ctx.dim = dim
        ctx.grad_scale = grad_scale
        return _gather(input_, process_group, dim)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.grad_scale == "up":
            grad_output = grad_output * dist.get_world_size(ctx.mode)
        elif ctx.grad_scale == "down":
            grad_output = grad_output / dist.get_world_size(ctx.mode)
        return _split(grad_output, ctx.mode, ctx.dim), None, None, None

class _SplitForwardGatherBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, process_group, dim, grad_scale):
        ctx.mode = process_group
        ctx.dim = dim
        ctx.grad_scale = grad_scale
        return _split(input_, process_group, dim)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.grad_scale == "up":
            grad_output = grad_output * dist.get_world_size(ctx.mode)
        elif ctx.grad_scale == "down":
            grad_output = grad_output / dist.get_world_size(ctx.mode)
        return _gather(grad_output, ctx.mode, ctx.dim), None, None, None

def split_forward_gather_backward(input_, process_group, dim, grad_scale=1.0):
    return _SplitForwardGatherBackward.apply(input_, process_group, dim, grad_scale)

def gather_forward_split_backward(input_, process_group, dim, grad_scale=None):
    return _GatherForwardSplitBackward.apply(input_, process_group, dim, grad_scale)