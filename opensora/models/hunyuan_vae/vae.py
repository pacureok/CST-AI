# CST (Close Sora for TPU) - VAE CORE
# Copyright 2026 Pcure-AI+. All rights reserved.
#
# LICENSE: COMMERCIAL ATTRIBUTION LICENSE (CAL)
# This code is part of the CST engine.
# - Optimized for Google Cloud TPU v5 / Kaggle TPU v3-8
# - Includes: DPR Hooks, Native C++/ASM Sharpening.

import torch
import torch.nn as nn
from typing import Optional, Union, Tuple

from .unet_causal_3d_blocks import (
    DownEncoderBlockCausal3D,
    UpDecoderBlockCausal3D,
    UNetMidBlockCausal3D,
    ResnetBlockCausal3D
)

class HunyuanVideoVAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 16,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        dropout: float = 0.0,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        add_attention: bool = True,
        **kwargs
    ):
        super().__init__()
        
        # ----------------------------------------------------------------
        # ENCODER: Compresión Espacio-Temporal
        # ----------------------------------------------------------------
        self.encoder_conv_in = nn.Conv3d(in_channels, block_out_channels[0], kernel_size=3, padding=1)
        
        self.down_blocks = nn.ModuleList([])
        for i in range(len(block_out_channels)):
            input_channels = block_out_channels[i - 1] if i > 0 else block_out_channels[0]
            output_channels = block_out_channels[i]
            is_last = i == len(block_out_channels) - 1
            
            self.down_blocks.append(
                DownEncoderBlockCausal3D(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    num_layers=layers_per_block,
                    add_downsample=not is_last,
                    resnet_eps=resnet_eps,
                    resnet_act_fn=resnet_act_fn,
                    resnet_groups=resnet_groups,
                    dropout=dropout
                )
            )

        # ----------------------------------------------------------------
        # MIDDLE BLOCK: Latent Bottleneck
        # ----------------------------------------------------------------
        self.mid_block = UNetMidBlockCausal3D(
            in_channels=block_out_channels[-1],
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            add_attention=add_attention,
            num_layers=1
        )

        # ----------------------------------------------------------------
        # LATENT PROJECTION (PCURE DPR Engine Hook)
        # ----------------------------------------------------------------
        self.norm_out = nn.GroupNorm(num_groups=resnet_groups, num_channels=block_out_channels[-1], eps=resnet_eps)
        self.conv_out = nn.Conv3d(block_out_channels[-1], latent_channels, kernel_size=3, padding=1)
        
        # ----------------------------------------------------------------
        # DECODER: Reconstrucción Ultra-Realista
        # ----------------------------------------------------------------
        self.decoder_conv_in = nn.Conv3d(latent_channels, block_out_channels[-1], kernel_size=3, padding=1)
        
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(block_out_channels))
        
        for i in range(len(reversed_block_out_channels)):
            input_channels = reversed_block_out_channels[i]
            output_channels = reversed_block_out_channels[i + 1] if i + 1 < len(reversed_block_out_channels) else reversed_block_out_channels[-1]
            is_last = i == len(reversed_block_out_channels) - 1
            
            self.up_blocks.append(
                UpDecoderBlockCausal3D(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    num_layers=layers_per_block + 1,
                    add_upsample=not is_last,
                    resolution_idx=i, # Gatillo para el motor nativo de nitidez
                    resnet_eps=resnet_eps,
                    resnet_act_fn=resnet_act_fn,
                    resnet_groups=resnet_groups,
                    dropout=dropout
                )
            )

        self.conv_norm_out = nn.GroupNorm(num_groups=resnet_groups, num_channels=block_out_channels[0], eps=resnet_eps)
        self.conv_final = nn.Conv3d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

        print("    ######################################################")
        print("    #        CST: CLOSE SORA FOR TPU - ENGINE v1.0       #")
        print("    #            PROPRIETARY BY PCURE-AI+                #")
        print("    ######################################################")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder_conv_in(x)
        for block in self.down_blocks:
            h = block(h)
        h = self.mid_block(h)
        h = self.norm_out(h)
        h = torch.nn.functional.silu(h)
        h = self.conv_out(h)
        return h

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_conv_in(z)
        for block in self.up_blocks:
            h = block(h)
        h = self.conv_norm_out(h)
        h = torch.nn.functional.silu(h)
        h = self.conv_final(h)
        return h

    def forward(self, x: torch.Tensor, sample_posterior: bool = False) -> torch.Tensor:
        # Nota: Para CST, priorizamos la reconstrucción directa en TPU
        z = self.encode(x)
        dec = self.decode(z)
        return dec