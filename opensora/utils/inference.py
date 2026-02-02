import copy
import os
import re
from enum import Enum

import torch
from torch import nn

from opensora.datasets import save_sample
from opensora.datasets.aspect import get_image_size
from opensora.datasets.utils import read_from_path, rescale_image_by_path
from opensora.utils.logger import log_message
from opensora.utils.prompt_refine import refine_prompts


class SamplingMethod(Enum):
    I2V = "i2v"  # Para generación de video Open-Sora
    DISTILLED = "distill"  # Para generación de imágenes FLUX


def create_tmp_csv(save_dir: str, prompt: str, ref: str = None, create=True) -> str:
    """Crea un CSV temporal con el prompt para procesar en lote."""
    tmp_file = os.path.join(save_dir, "prompt.csv")
    if not create:
        return tmp_file
    with open(tmp_file, "w", encoding="utf-8") as f:
        if ref is not None:
            f.write(f'text,ref\n"{prompt}","{ref}"')
        else:
            f.write(f'text\n"{prompt}"')
    return tmp_file


def modify_option_to_t2i(sampling_option, distilled: bool = False, img_resolution: str = "1080px"):
    """Modifica opciones de video para generar una imagen simple (num_frames=1)."""
    sampling_option_t2i = copy.copy(sampling_option)
    if distilled:
        sampling_option_t2i.method = SamplingMethod.DISTILLED
    sampling_option_t2i.num_frames = 1
    sampling_option_t2i.height, sampling_option_t2i.width = get_image_size(img_resolution, sampling_option.aspect_ratio)
    sampling_option_t2i.guidance = 4.0
    sampling_option_t2i.resized_resolution = sampling_option.resolution
    return sampling_option_t2i


def get_save_path_name(save_dir, sub_dir, save_prefix="", name=None, fallback_name=None, index=None, num_sample_pos=None, prompt_as_path=False, prompt=None):
    """Genera la ruta de guardado final del video."""
    if prompt_as_path:
        cleaned_prompt = prompt.strip(".")
        fname = f"{cleaned_prompt}-{num_sample_pos}"
    else:
        if name is not None:
            fname = save_prefix + name
        else:
            fname = f"{save_prefix + fallback_name}_{index:04d}"
        if num_sample_pos > 0:
            fname += f"_{num_sample_pos}"
    return os.path.join(save_dir, sub_dir, fname)


def get_names_from_path(path):
    filename = os.path.basename(path)
    name, _ = os.path.splitext(filename)
    return name


def process_and_save(x: torch.Tensor, batch: dict, cfg: dict, sub_dir: str, generate_sampling_option, epoch: int, start_index: int, saving: bool = True):
    """Procesa los tensores resultantes y los guarda como video/imagen."""
    fallback_name = cfg.dataset.data_path.split("/")[-1].split(".")[0]
    prompt_as_path = cfg.get("prompt_as_path", False)
    fps_save = cfg.get("fps_save", 16)
    save_dir = cfg.save_dir

    names = batch["name"] if "name" in batch else [None] * len(x)
    indices = batch["index"] if "index" in batch else [None] * len(x)
    if "index" in batch:
        indices = [idx + start_index for idx in indices]
    prompts = batch["text"]

    ret_names = []
    is_image = generate_sampling_option.num_frames == 1
    for img, name, index, prompt in zip(x, names, indices, prompts):
        save_path = get_save_path_name(save_dir, sub_dir, save_prefix=cfg.get("save_prefix", ""), name=name, fallback_name=fallback_name, index=index, num_sample_pos=epoch, prompt_as_path=prompt_as_path, prompt=prompt)
        ret_name = get_names_from_path(save_path)
        ret_names.append(ret_name)

        if saving:
            # Guardar el prompt en un .txt junto al video
            with open(save_path + ".txt", "w", encoding="utf-8") as f:
                f.write(prompt)
            # Guardar la muestra (video o imagen)
            save_sample(img, save_path=save_path, fps=fps_save)

            # Reescalar si es necesario (T2I2V)
            if cfg.get("use_t2i2v", False) and is_image and generate_sampling_option.resolution != generate_sampling_option.resized_resolution:
                log_message("Rescaling image to %s...", generate_sampling_option.resized_resolution)
                height, width = get_image_size(generate_sampling_option.resized_resolution, generate_sampling_option.aspect_ratio)
                rescale_image_by_path(save_path + ".png", width, height)
    return ret_names


def add_fps_info_to_text(text: list[str], fps: int = 16):
    """Añade información de FPS al prompt para mejorar la consistencia temporal."""
    mod_text = []
    for item in text:
        item = item.strip()
        if not item.endswith("."): item += "."
        if not re.search(r"\d+ FPS\.$", item):
            item = item + f" {fps} FPS."
        mod_text.append(item)
    return mod_text


def collect_references_batch(reference_paths, cond_type, model_ae, image_size, is_causal=False):
    """Codifica imágenes/videos de referencia usando el AutoEncoder."""
    refs_x = []
    device = next(model_ae.parameters()).device
    dtype = next(model_ae.parameters()).dtype
    for reference_path in reference_paths:
        if reference_path == "":
            refs_x.append(None)
            continue
        ref_path = reference_path.split(";")
        ref = []
        
        # Lógica para I2V (Image to Video) o V2V (Video to Video)
        if "v2v" in cond_type or "i2v" in cond_type:
            r = read_from_path(ref_path[0], image_size, transform_name="resize_crop")
            # Selección de frames según el tipo de condición (head, tail, loop)
            if "head" in cond_type: r = r[:, :1]
            elif "tail" in cond_type: r = r[:, -1:]
            
            r_x = model_ae.encode(r.unsqueeze(0).to(device, dtype))
            ref.append(r_x.squeeze(0))
        refs_x.append(ref)
    return refs_x


def prepare_inference_condition(z, mask_cond, ref_list=None, causal=True):
    """Prepara las máscaras y latentes condicionados para el modelo."""
    B, C, T, H, W = z.shape
    masks = torch.zeros(B, 1, T, H, W)
    masked_z = torch.zeros(B, C, T, H, W)

    if ref_list is None: return masks.to(z.device, z.dtype), masked_z.to(z.device, z.dtype)

    for i in range(B):
        ref = ref_list[i]
        if ref is not None and T > 1:
            if mask_cond == "i2v_head":
                masks[i, :, 0, :, :] = 1
                masked_z[i, :, 0, :, :] = ref[0][:, 0, :, :]
            elif mask_cond == "i2v_loop":
                masks[i, :, 0, :, :] = 1
                masks[i, :, -1, :, :] = 1
                masked_z[i, :, 0, :, :] = ref[0][:, 0, :, :]
                masked_z[i, :, -1, :, :] = ref[-1][:, -1, :, :]
    
    return masks.to(z.device, z.dtype), masked_z.to(z.device, z.dtype)