# PCURE-AI+ / CST: Optimized Inference for TPU
# Copyright 2026 Pcure-AI+. All rights reserved.

import os
import time
import warnings
from pprint import pformat

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
# Eliminamos dist tradicional para evitar bloqueos de CUDA
import torch.distributed as dist 
from tqdm import tqdm

# Manejo de Hardware TPU
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    HAS_XLA = True
except ImportError:
    HAS_XLA = False

# Importaciones locales modificadas
from opensora.acceleration.parallel_states import get_data_parallel_group
from opensora.datasets.dataloader import prepare_dataloader
from opensora.registry import DATASETS, build_module
from opensora.utils.config import parse_alias, parse_configs
from opensora.utils.inference import (
    add_fps_info_to_text,
    add_motion_score_to_text,
    create_tmp_csv,
    modify_option_to_t2i,
    process_and_save,
)
from opensora.utils.logger import create_logger, is_main_process
from opensora.utils.misc import to_torch_dtype
from opensora.utils.prompt_refine import refine_prompts
from opensora.utils.sampling import (
    SamplingOption,
    prepare_api,
    prepare_models,
    sanitize_sampling_option,
)

# Bypass de ColossalAI: Usamos funciones nativas de CST
def set_seed(seed):
    torch.manual_seed(seed)
    if HAS_XLA: xm.set_rng_state(seed)

@torch.inference_mode()
def main():
    # ======================================================
    # 1. Configs & Runtime Variables (TPU Optimized)
    # ======================================================
    torch.set_grad_enabled(False)
    cfg = parse_configs()
    cfg = parse_alias(cfg)

    # Forzar dispositivo a TPU si está disponible
    device = xm.xla_device() if HAS_XLA else "cpu"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    
    seed = cfg.get("seed", 1024)
    if seed is not None:
        set_seed(seed)

    logger = create_logger()
    logger.info("CST Engine Active. Device: %s, Dtype: %s", device, dtype)
    
    # El 'booster' se ignora en TPU ya que usamos el motor nativo libtentpu.so
    is_saving_process = is_main_process() 

    # ======================================================
    # 2. Build Dataset and Dataloader
    # ======================================================
    logger.info("Building dataset...")
    save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)

    if cfg.get("prompt"):
        cfg.dataset.data_path = create_tmp_csv(save_dir, cfg.prompt, cfg.get("ref", None), create=is_main_process())
    
    # Sincronización XLA en lugar de dist.barrier()
    if HAS_XLA: xm.rendezvous("dataset_init")
    
    dataset = build_module(cfg.dataset, DATASETS)
    logger.info("Dataset contains %s samples.", len(dataset))

    dataloader_args = dict(
        dataset=dataset,
        batch_size=cfg.get("batch_size", 1),
        num_workers=cfg.get("num_workers", 4),
        seed=cfg.get("seed", 1024),
        shuffle=False,
        drop_last=False,
        pin_memory=False, # Pin memory puede dar error en TPU
        process_group=get_data_parallel_group(),
    )
    dataloader, _ = prepare_dataloader(**dataloader_args)

    sampling_option = SamplingOption(**cfg.sampling_option)
    sampling_option = sanitize_sampling_option(sampling_option)
    
    fps_save = cfg.get("fps_save", 16)
    num_sample = cfg.get("num_sample", 1)
    sub_dir = f"video_{cfg.sampling_option.resolution}"
    os.makedirs(os.path.join(save_dir, sub_dir), exist_ok=True)

    # ======================================================
    # 3. Build Model (PCURE-AI+ Bridge)
    # ======================================================
    logger.info("Building models on TPU...")
    
    # Cargamos el modelo directamente al dispositivo XLA
    model, model_ae, model_t5, model_clip, optional_models = prepare_models(
        cfg, device, dtype, offload_model=False 
    )

    # Vinculamos la API de muestreo
    api_fn = prepare_api(model, model_ae, model_t5, model_clip, optional_models)

    # ======================================================
    # 4. Inference loop
    # ======================================================
    for epoch in range(num_sample):
        dataloader_iter = iter(dataloader)
        with tqdm(
            enumerate(dataloader_iter, start=0),
            desc="CST Inference Progress",
            disable=not is_main_process(),
            total=len(dataloader),
        ) as pbar:
            for _, batch in pbar:
                original_text = batch.pop("text")
                batch["text"] = original_text
                
                # Refinamiento de prompt (opcional)
                if cfg.get("prompt_refine", False):
                    batch["text"] = refine_prompts(original_text, type="t2v")
                
                batch["text"] = add_fps_info_to_text(batch.pop("text"), fps=fps_save)

                logger.info("PCURE Engine: Generating video...")
                
                # Ejecución de Inferencia
                x = api_fn(
                    sampling_option,
                    "t2v",
                    seed=sampling_option.seed + epoch if sampling_option.seed else None,
                    channel=cfg["model"]["in_channels"],
                    **batch,
                ).cpu() # Movemos a CPU solo para guardar

                if is_saving_process:
                    process_and_save(x, batch, cfg, sub_dir, sampling_option, epoch, 0)
                
                if HAS_XLA: xm.mark_step() # Vital para ejecutar el grafo en TPU

    logger.info("Inference finished. CST Engine out.")

if __name__ == "__main__":
    main()