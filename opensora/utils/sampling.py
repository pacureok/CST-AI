import math
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

import torch
from einops import rearrange, repeat
from mmengine.config import Config
from peft import PeftModel
from torch import Tensor, nn

from opensora.datasets.aspect import get_image_size
from opensora.models.mmdit.model import MMDiTModel
from opensora.models.text.conditioner import HFEmbedder
from opensora.registry import MODELS, build_module
from opensora.utils.inference import (
    SamplingMethod,
    collect_references_batch,
    prepare_inference_condition,
)

# ======================================================
# Sampling Options
# ======================================================

@dataclass
class SamplingOption:
    width: int | None = None
    height: int | None = None
    resolution: str | None = None
    aspect_ratio: str | None = None
    num_frames: int = 1
    num_steps: int = 50
    guidance: float = 4.0
    text_osci: bool = False
    guidance_img: float | None = None
    image_osci: bool = False
    scale_temporal_osci: bool = False
    seed: int | None = None
    shift: bool = True
    method: str | SamplingMethod = SamplingMethod.I2V
    temporal_reduction: int = 1
    is_causal_vae: bool = False
    flow_shift: float | None = None

def sanitize_sampling_option(sampling_option: SamplingOption) -> SamplingOption:
    if sampling_option.resolution is not None or sampling_option.aspect_ratio is not None:
        assert sampling_option.resolution is not None and sampling_option.aspect_ratio is not None
        resolution = sampling_option.resolution
        aspect_ratio = sampling_option.aspect_ratio
        height, width = get_image_size(resolution, aspect_ratio, training=False)
    else:
        assert sampling_option.height is not None and sampling_option.width is not None
        height, width = sampling_option.height, sampling_option.width

    height = (height // 16 + (1 if height % 16 else 0)) * 16
    width = (width // 16 + (1 if width % 16 else 0)) * 16
    replace_dict = dict(height=height, width=width)

    if isinstance(sampling_option.method, str):
        method = SamplingMethod(sampling_option.method)
        replace_dict["method"] = method

    return replace(sampling_option, **replace_dict)

def get_oscillation_gs(guidance_scale: float, i: int, force_num=10):
    if i < force_num or (i >= force_num and i % 2 == 0):
        gs = guidance_scale
    else:
        gs = 1.0
    return gs

# ======================================================
# Denoising Logic
# ======================================================

class Denoiser(ABC):
    @abstractmethod
    def denoise(self, model: MMDiTModel, **kwargs) -> Tensor:
        pass

    @abstractmethod
    def prepare_guidance(self, text: list[str], **kwargs) -> dict[str, Tensor]:
        pass

class I2VDenoiser(Denoiser):
    def denoise(self, model: MMDiTModel, **kwargs) -> Tensor:
        img = kwargs.pop("img")
        timesteps = kwargs.pop("timesteps")
        guidance = kwargs.pop("guidance")
        guidance_img = kwargs.pop("guidance_img")
        masks = kwargs.pop("masks")
        masked_ref = kwargs.pop("masked_ref")
        patch_size = kwargs.pop("patch_size", 2)

        guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
        
        for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
            b, c, t, w, h = masked_ref.size()
            cond = pack(torch.cat((masks, masked_ref), dim=1), patch_size=patch_size)
            kwargs["cond"] = torch.cat([cond, cond, torch.zeros_like(cond)], dim=0)

            cond_x = img[: len(img) // 3]
            img = torch.cat([cond_x, cond_x, cond_x], dim=0)
            
            pred = model(img=img, **kwargs, timesteps=t_vec, guidance=guidance_vec)

            text_gs = get_oscillation_gs(guidance, i) if kwargs.get("text_osci") else guidance
            image_gs = guidance_img
            
            cond, uncond, uncond_2 = pred.chunk(3, dim=0)
            pred = uncond_2 + image_gs * (uncond - uncond_2) + text_gs * (cond - uncond)
            pred = torch.cat([pred, pred, pred], dim=0)

            img = img + (t_prev - t_curr) * pred

        return img[: len(img) // 3]

    def prepare_guidance(self, text: list[str], **kwargs) -> tuple[list[str], dict[str, Tensor]]:
        neg = kwargs.get("neg") or [""] * len(text)
        return text + neg + neg, {"guidance_img": kwargs.pop("guidance_img")}

class DistilledDenoiser(Denoiser):
    def denoise(self, model: MMDiTModel, **kwargs) -> Tensor:
        img, timesteps, guidance = kwargs.pop("img"), kwargs.pop("timesteps"), kwargs.pop("guidance")
        guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
        for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
            pred = model(img=img, **kwargs, timesteps=t_vec, guidance=guidance_vec)
            img = img + (t_prev - t_curr) * pred
        return img

    def prepare_guidance(self, text: list[str], **kwargs) -> tuple[list[str], dict[str, Tensor]]:
        return text, {}

SamplingMethodDict = {
    SamplingMethod.I2V: I2VDenoiser(),
    SamplingMethod.DISTILLED: DistilledDenoiser(),
}

# ======================================================
# Scheduler & Utilities
# ======================================================

def time_shift(alpha: float, t: Tensor) -> Tensor:
    return alpha * t / (1 + (alpha - 1) * t)

def get_schedule(num_steps, image_seq_len, num_frames, shift=True, **kwargs) -> list[float]:
    timesteps = torch.linspace(1, 0, num_steps + 1)
    if shift:
        # Lógica de ajuste de escala temporal/espacial
        shift_alpha = kwargs.get("shift_alpha", 1.0) * math.sqrt(num_frames)
        timesteps = time_shift(shift_alpha, timesteps)
    return timesteps.tolist()

def pack(x: Tensor, patch_size: int = 2) -> Tensor:
    return rearrange(x, "b c t (h ph) (w pw) -> b (t h w) (c ph pw)", ph=patch_size, pw=patch_size)

def unpack(x: Tensor, height: int, width: int, num_frames: int, patch_size: int = 2) -> Tensor:
    D = int(os.environ.get("AE_SPATIAL_COMPRESSION", 16))
    return rearrange(x, "b (t h w) (c ph pw) -> b c t (h ph) (w pw)", 
                     h=math.ceil(height/D), w=math.ceil(width/D), t=num_frames, ph=patch_size, pw=patch_size)

# ======================================================
# Main API Function
# ======================================================

def prepare_api(model, model_ae, model_t5, model_clip, optional_models):
    @torch.inference_mode()
    def api_fn(opt: SamplingOption, text=None, **kwargs):
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        seed = opt.seed or random.randint(0, 2**32 - 1)
        
        # 1. Generar ruido inicial
        z = torch.randn(len(text), 16, opt.num_frames, opt.height//16, opt.width//16, 
                        device=device, dtype=dtype, generator=torch.manual_seed(seed))
        
        denoiser = SamplingMethodDict[opt.method]
        
        # 2. Programar pasos de tiempo (Schedule)
        timesteps = get_schedule(opt.num_steps, (z.shape[-1]*z.shape[-2]), opt.num_frames, shift=opt.shift)

        # 3. Preparar Embeddings de texto y condiciones
        text_aug, add_inp = denoiser.prepare_guidance(text=text, **kwargs)
        # (Aquí se llamarían a los encoders T5/CLIP - simplificado para brevedad)
        
        # 4. Bucle de Denoising
        x = denoiser.denoise(model, img=pack(z), timesteps=timesteps, guidance=opt.guidance, **add_inp)
        
        # 5. Decodificación VAE (De latente a Píxeles)
        x_unpacked = unpack(x, opt.height, opt.width, opt.num_frames)
        return model_ae.decode(x_unpacked)

    return api_fn