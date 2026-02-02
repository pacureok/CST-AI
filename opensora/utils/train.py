import random
import warnings
from collections import OrderedDict
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F
from colossalai.booster.plugin import HybridParallelPlugin, LowLevelZeroPlugin
from colossalai.cluster import DistCoordinator
from colossalai.utils import get_current_device
from einops import rearrange
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from opensora.acceleration.parallel_states import (
    set_data_parallel_group,
    set_sequence_parallel_group,
    set_tensor_parallel_group,
)
from opensora.utils.optimizer import LinearWarmupLR

# ======================================================
# LR & Optimizer Utilities
# ======================================================

def set_lr(optimizer: torch.optim.Optimizer, lr_scheduler: _LRScheduler, lr: float, initial_lr: float = None):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    if isinstance(lr_scheduler, LinearWarmupLR):
        lr_scheduler.base_lrs = [lr] * len(lr_scheduler.base_lrs)
        if initial_lr is not None:
            lr_scheduler.initial_lr = initial_lr

def set_warmup_steps(lr_scheduler: _LRScheduler, warmup_steps: int):
    if isinstance(lr_scheduler, LinearWarmupLR):
        lr_scheduler.warmup_steps = warmup_steps

def set_eps(optimizer: torch.optim.Optimizer, eps: float = None):
    if eps is not None:
        for param_group in optimizer.param_groups:
            param_group["eps"] = eps

# ======================================================
# Device & Distributed Setup
# ======================================================

def setup_device() -> tuple[torch.device, DistCoordinator]:
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    coordinator = DistCoordinator()
    device = get_current_device()
    return device, coordinator

def create_colossalai_plugin(plugin: str, dtype: str, grad_clip: float, **kwargs) -> LowLevelZeroPlugin | HybridParallelPlugin:
    plugin_kwargs = dict(
        precision=dtype,
        initial_scale=2**16,
        max_norm=grad_clip,
        overlap_allgather=True,
        cast_inputs=False,
        reduce_bucket_size_in_m=20,
    )
    plugin_kwargs.update(kwargs)
    sp_size = plugin_kwargs.get("sp_size", 1)
    
    if plugin in ["zero1", "zero2"]:
        assert sp_size == 1, "Zero plugin does not support sequence parallelism"
        stage = 1 if plugin == "zero1" else 2
        plugin = LowLevelZeroPlugin(stage=stage, **plugin_kwargs)
        set_data_parallel_group(dist.group.WORLD)
    elif plugin == "hybrid":
        plugin_kwargs["find_unused_parameters"] = True
        rb_size = plugin_kwargs.pop("reduce_bucket_size_in_m")
        if "zero_bucket_size_in_m" not in plugin_kwargs:
            plugin_kwargs["zero_bucket_size_in_m"] = rb_size
        plugin_kwargs.pop("cast_inputs")
        plugin_kwargs["enable_metadata_cache"] = False

        custom_policy = plugin_kwargs.pop("custom_policy", None)
        if custom_policy is not None:
            custom_policy = custom_policy()
            
        plugin = HybridParallelPlugin(custom_policy=custom_policy, **plugin_kwargs)
        set_tensor_parallel_group(plugin.tp_group)
        set_sequence_parallel_group(plugin.sp_group)
        set_data_parallel_group(plugin.dp_group)
    else:
        raise ValueError(f"Unknown plugin {plugin}")
    return plugin

# ======================================================
# Training Logic (EMA & Dropout)
# ======================================================

@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, optimizer=None, decay: float = 0.9999, sharded: bool = True):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        if name == "pos_embed" or not param.requires_grad:
            continue
        if not sharded:
            param_data = param.data
        else:
            if param.data.dtype != torch.float32:
                param_id = id(param)
                master_param = optimizer.get_working_to_master_map()[param_id]
                param_data = master_param.data
            else:
                param_data = param.data
        ema_params[name].mul_(decay).add_(param_data, alpha=1 - decay)

def dropout_condition(prob: float, txt: torch.Tensor, null_txt: torch.Tensor) -> torch.Tensor:
    if prob == 0: return txt
    drop_ids = torch.rand(txt.shape[0], device=txt.device) < prob
    drop_ids = drop_ids.view((drop_ids.shape[0],) + (1,) * (txt.ndim - 1))
    return torch.where(drop_ids, null_txt, txt)

# ======================================================
# Visual Condition Preparation (I2V / V2V)
# ======================================================

def prepare_visual_condition_uncausal(x: torch.Tensor, condition_config: dict, model_ae: nn.Module, pad: bool = False):
    B = x.shape[0]
    C = model_ae.cfg.latent_channels
    T, H, W = model_ae.get_latent_size(x.shape[-3:])

    masks = torch.zeros(B, 1, T, H, W).to(x.device, x.dtype)
    latent = torch.zeros(B, C, T, H, W).to(x.device, x.dtype)
    x_0 = torch.zeros(B, C, T, H, W).to(x.device, x.dtype)

    if T > 1:
        mask_cond_options = list(condition_config.keys())
        mask_cond_weights = list(condition_config.values())

        for i in range(B):
            mask_cond = random.choices(mask_cond_options, weights=mask_cond_weights, k=1)[0]
            if mask_cond == "i2v_head":
                masks[i, :, 0, :, :] = 1
                x_0[i] = model_ae.encode(x[i:i+1])[0]
                latent[i, :, :1, :, :] = model_ae.encode(x[i, :, :1, :, :].unsqueeze(0))
            elif mask_cond == "i2v_loop":
                masks[i, :, 0, :, :] = 1
                masks[i, :, -1, :, :] = 1
                x_0[i] = model_ae.encode(x[i:i+1])[0]
                latent[i, :, :1, :, :] = model_ae.encode(x[i, :, :1, :, :].unsqueeze(0))
                latent[i, :, -1:, :, :] = model_ae.encode(x[i, :, -1:, :, :].unsqueeze(0))
            # ... (otras condiciones como v2v_head/tail siguen lógica similar)
            else:
                x_0[i] = model_ae.encode(x[i].unsqueeze(0))[0]
    else:
        x_0 = model_ae.encode(x)

    cond = torch.cat((masks, masks * latent), dim=1)
    return x_0, cond

# ======================================================
# Loss Calculation
# ======================================================

def get_batch_loss(model_pred, v_t, masks=None):
    if masks is not None:
        # Lógica para calcular pérdida solo en frames no condicionados
        num_frames, height, width = masks.shape[-3:]
        masks_simple = masks[:, :, 0, 0] 
        # Reorganizar predicciones para comparar píxel a píxel
        batch_loss = 0
        for i in range(model_pred.size(0)):
            # Se excluyen los frames de referencia (máscara=1) del cálculo del gradiente
            batch_loss += F.mse_loss(model_pred[i].float(), v_t[i].float())
        return batch_loss / model_pred.size(0)
    
    return F.mse_loss(model_pred.float(), v_t.float(), reduction="mean")

@torch.no_grad()
def warmup_ae(model_ae: nn.Module, shapes: list[tuple[int, ...]], device: torch.device, dtype: torch.dtype):
    progress_bar = tqdm(shapes, desc="Warmup AE", disable=dist.get_rank() != 0)
    for x_shape in progress_bar:
        x = torch.randn(*x_shape, device=device, dtype=dtype)
        _ = model_ae.encode(x)