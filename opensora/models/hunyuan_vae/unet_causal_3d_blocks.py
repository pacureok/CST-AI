# CST (Close Sora for TPU) - VAE BLOCKS CORE
# Copyright 2026 Pcure-AI+. All rights reserved.
#
# LICENSE: COMMERCIAL ATTRIBUTION LICENSE (CAL)
# Optimized for Google Cloud TPU v5 / Kaggle TPU v3-8
# Integration: PCURE-AI+ Native Engine (C++/ASM)

import ctypes
import os
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.activations import get_activation
from diffusers.models.attention_processor import Attention
from diffusers.utils import logging
from einops import rearrange
from torch import nn

from opensora.acceleration.checkpoint import auto_grad_checkpoint
from opensora.models.vae.utils import ChannelChunkConv3d, get_conv3d_n_chunks

# ----------------------------------------------------------------
# PCURE-AI+ NATIVE ENGINE LINK
# ----------------------------------------------------------------
logger = logging.get_logger("CST_ENGINE")

try:
    LIB_PATH = "/kaggle/working/CST-AI/MODELS/libtentpu.so"
    if os.path.exists(LIB_PATH):
        pcure_native = ctypes.CDLL(LIB_PATH)
        # Definimos la interfaz C++
        pcure_native.process_video_tensor.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        pcure_native.get_engine_version.restype = ctypes.c_char_p
        NATIVE_ACTIVE = True
        logger.info(f"PCURE-AI+ Native Linked: {pcure_native.get_engine_version().decode()}")
    else:
        NATIVE_ACTIVE = False
except Exception as e:
    logger.warning(f"Native engine fallback: {e}")
    NATIVE_ACTIVE = False

def apply_pcure_native_refinement(tensor: torch.Tensor):
    """Aplica sharpening de bajo nivel vía C++/ASM directamente en el puntero de la TPU."""
    if NATIVE_ACTIVE and tensor.dtype == torch.float32:
        # Aseguramos que el tensor esté en memoria contigua para C++
        tensor_cont = tensor.contiguous()
        ptr = tensor_cont.data_ptr()
        size = tensor_cont.numel()
        pcure_native.process_video_tensor(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float)), size)
        return tensor_cont
    return tensor

# ----------------------------------------------------------------
# CST OPTIMIZATION: HELPER FUNCTIONS
# ----------------------------------------------------------------
INTERPOLATE_NUMEL_LIMIT = 2**31 - 1

def chunk_nearest_interpolate(x: torch.Tensor, scale_factor):
    limit = INTERPOLATE_NUMEL_LIMIT // np.prod(scale_factor)
    n_chunks = get_conv3d_n_chunks(x.numel(), x.size(1), limit)
    x_chunks = x.chunk(n_chunks, dim=1)
    x_chunks = [F.interpolate(x_chunk, scale_factor=scale_factor, mode="nearest") for x_chunk in x_chunks]
    return torch.cat(x_chunks, dim=1)

def prepare_causal_attention_mask(n_frame: int, n_hw: int, dtype, device, batch_size: int = None):
    seq_len = n_frame * n_hw
    row_idx = torch.arange(seq_len, device=device).view(-1, 1)
    col_idx = torch.arange(seq_len, device=device).view(1, -1)
    i_frame = row_idx // n_hw
    visibility_limit = (i_frame + 1) * n_hw
    
    mask = torch.where(
        col_idx < visibility_limit,
        torch.tensor(0.0, dtype=dtype, device=device),
        torch.tensor(float("-inf"), dtype=dtype, device=device)
    )
    if batch_size is not None:
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask

# ----------------------------------------------------------------
# CST LAYERS (CAUSAL OPS)
# ----------------------------------------------------------------

class CausalConv3d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, stride=1, dilation=1, pad_mode="replicate", **kwargs):
        super().__init__()
        self.pad_mode = pad_mode
        padding = (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size - 1, 0)
        self.time_causal_padding = padding
        self.conv = ChannelChunkConv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def forward(self, x):
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        return self.conv(x)

class UpsampleCausal3D(nn.Module):
    def __init__(self, channels: int, out_channels: Optional[int] = None, kernel_size: int = 3, bias=True, upsample_factor=(2, 2, 2)):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.upsample_factor = upsample_factor
        self.conv = CausalConv3d(self.channels, self.out_channels, kernel_size=kernel_size, bias=bias)

    def forward(self, input_tensor: torch.FloatTensor) -> torch.FloatTensor:
        dtype = input_tensor.dtype
        hidden_states = input_tensor.to(torch.float32) if dtype == torch.bfloat16 else input_tensor
        
        T = hidden_states.size(2)
        first_h, other_h = hidden_states.split((1, T - 1), dim=2)
        
        if T > 1: other_h = chunk_nearest_interpolate(other_h, scale_factor=self.upsample_factor)
        first_h = chunk_nearest_interpolate(first_h.squeeze(2), scale_factor=self.upsample_factor[1:]).unsqueeze(2)
        
        hidden_states = torch.cat((first_h, other_h), dim=2) if T > 1 else first_h
        return self.conv(hidden_states.to(dtype))

class DownsampleCausal3D(nn.Module):
    def __init__(self, channels: int, kernel_size=3, bias=True, stride=2):
        super().__init__()
        self.conv = CausalConv3d(channels, channels, kernel_size=kernel_size, stride=stride, bias=bias)

    def forward(self, x): return self.conv(x)

class ResnetBlockCausal3D(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, dropout=0.0, groups=32, groups_out=None, eps=1e-6, non_linearity="swish", output_scale_factor=1.0, use_in_shortcut=None, pre_norm=True):
        super().__init__()
        out_channels = out_channels or in_channels
        self.norm1 = nn.GroupNorm(groups, in_channels, eps=eps)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3)
        self.norm2 = nn.GroupNorm(groups_out or groups, out_channels, eps=eps)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3)
        self.nonlinearity = get_activation(non_linearity)
        self.output_scale_factor = output_scale_factor
        
        self.conv_shortcut = CausalConv3d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x, audio_cond=None):
        h = self.conv2(self.dropout(self.nonlinearity(self.norm2(self.conv1(self.nonlinearity(self.norm1(x)))))))
        x = self.conv_shortcut(x) if self.conv_shortcut else x
        return (x + h) / self.output_scale_factor

class UNetMidBlockCausal3D(nn.Module):
    def __init__(self, in_channels, dropout=0.0, num_layers=1, resnet_eps=1e-6, resnet_act_fn="swish", resnet_groups=32, add_attention=True, attention_head_dim=1, output_scale_factor=1.0):
        super().__init__()
        self.resnets = nn.ModuleList([ResnetBlockCausal3D(in_channels=in_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor)])
        self.attentions = nn.ModuleList()
        for _ in range(num_layers):
            self.attentions.append(Attention(in_channels, heads=in_channels // attention_head_dim, dim_head=attention_head_dim, eps=resnet_eps, residual_connection=True, upcast_softmax=True) if add_attention else None)
            self.resnets.append(ResnetBlockCausal3D(in_channels=in_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor))

    def forward(self, x, mask=None, audio_cond=None):
        x = self.resnets[0](x, audio_cond)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn:
                B, C, T, H, W = x.shape
                x = rearrange(x, "b c f h w -> b (f h w) c")
                x = attn(x, attention_mask=mask)
                x = rearrange(x, "b (f h w) c -> b c f h w", f=T, h=H, w=W)
            x = resnet(x, audio_cond)
        return x

class DownEncoderBlockCausal3D(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=1, add_downsample=True, downsample_stride=2):
        super().__init__()
        self.resnets = nn.ModuleList([ResnetBlockCausal3D(in_channels=in_channels if i==0 else out_channels, out_channels=out_channels) for i in range(num_layers)])
        self.downsamplers = nn.ModuleList([DownsampleCausal3D(out_channels, stride=downsample_stride)]) if add_downsample else None

    def forward(self, x, audio_cond=None):
        for resnet in self.resnets: x = auto_grad_checkpoint(resnet, x, audio_cond=audio_cond)
        if self.downsamplers:
            for ds in self.downsamplers: x = auto_grad_checkpoint(ds, x)
        return x

class UpDecoderBlockCausal3D(nn.Module):
    def __init__(self, in_channels, out_channels, resolution_idx=None, num_layers=1, add_upsample=True, upsample_scale_factor=(2, 2, 2)):
        super().__init__()
        self.resnets = nn.ModuleList([ResnetBlockCausal3D(in_channels=in_channels if i==0 else out_channels, out_channels=out_channels) for i in range(num_layers)])
        self.upsamplers = nn.ModuleList([UpsampleCausal3D(out_channels, upsample_factor=upsample_scale_factor)]) if add_upsample else None
        self.resolution_idx = resolution_idx

    def forward(self, x, audio_cond=None):
        for resnet in self.resnets: x = auto_grad_checkpoint(resnet, x, audio_cond=audio_cond)
        if self.upsamplers:
            for us in self.upsamplers: x = auto_grad_checkpoint(us, x)
        
        # --- CST ULTRA-REALISM ENGINE + PCURE NATIVE ASM ---
        if self.resolution_idx is not None and self.resolution_idx < 2:
            # Paso A: Refinamiento estadístico (XLA)
            avg = torch.mean(x, dim=(3, 4), keepdim=True)
            x = x + 0.05 * (x - avg)
            
            # Paso B: Refinamiento por Hardware (C++/ASM)
            x = apply_pcure_native_refinement(x)

        return x