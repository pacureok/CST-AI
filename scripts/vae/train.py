import gc
import os
import random
import subprocess
import warnings
from contextlib import nullcontext
from copy import deepcopy
from pprint import pformat

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
gc.disable()

import torch
import torch.distributed as dist
from colossalai.booster import Booster
from colossalai.utils import set_seed
from torch.profiler import ProfilerActivity, profile, schedule
from tqdm import tqdm

import wandb
from opensora.acceleration.checkpoint import set_grad_checkpoint
from opensora.acceleration.parallel_states import get_data_parallel_group
from opensora.datasets.dataloader import prepare_dataloader
from opensora.datasets.pin_memory_cache import PinMemoryCache
from opensora.models.vae.losses import DiscriminatorLoss, GeneratorLoss, VAELoss
from opensora.registry import DATASETS, MODELS, build_module
from opensora.utils.ckpt import CheckpointIO, model_sharding, record_model_param_shape, rm_checkpoints
from opensora.utils.config import config_to_name, create_experiment_workspace, parse_configs
from opensora.utils.logger import create_logger
from opensora.utils.misc import (
    Timer,
    all_reduce_sum,
    create_tensorboard_writer,
    is_log_process,
    log_model_params,
    to_torch_dtype,
)
from opensora.utils.optimizer import create_lr_scheduler, create_optimizer
from opensora.utils.train import create_colossalai_plugin, set_lr, set_warmup_steps, setup_device, update_ema

# Optimización de cómputo
torch.backends.cudnn.benchmark = True

# Configuración del Profiler
WAIT, WARMUP, ACTIVE = 1, 10, 20
my_schedule = schedule(wait=WAIT, warmup=WARMUP, active=ACTIVE)

def main():
    # ======================================================
    # 1. Configuración y Variables de Entorno
    # ======================================================
    cfg = parse_configs()
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    device, coordinator = setup_device()
    checkpoint_io = CheckpointIO()
    set_seed(cfg.get("seed", 1024))
    
    PinMemoryCache.force_dtype = dtype
    PinMemoryCache.pre_alloc_numels = cfg.get("pin_memory_cache_pre_alloc_numels", None)

    # Inicializar ColossalAI Booster
    plugin_type = cfg.get("plugin", "zero2")
    plugin_config = cfg.get("plugin_config", {})
    plugin = create_colossalai_plugin(
        plugin=plugin_type, dtype=cfg.get("dtype", "bf16"), 
        grad_clip=cfg.get("grad_clip", 0), **plugin_config
    ) if plugin_type != "none" else None
    booster = Booster(plugin=plugin)

    # Workspace y Logging
    exp_name, exp_dir = create_experiment_workspace(
        cfg.get("outputs", "./outputs"),
        model_name=config_to_name(cfg),
        config=cfg.to_dict(),
    )
    
    logger = create_logger(exp_dir)
    tb_writer = create_tensorboard_writer(exp_dir) if coordinator.is_master() else None
    
    if coordinator.is_master() and cfg.get("wandb", False):
        wandb.init(project=cfg.get("wandb_project", "Open-Sora"), name=exp_name, config=cfg.to_dict(), dir=exp_dir)

    # ======================================================
    # 2. Dataset y Dataloader
    # ======================================================
    logger.info("Building dataset...")
    dataset = build_module(cfg.dataset, DATASETS)
    
    dataloader, sampler = prepare_dataloader(
        dataset=dataset,
        batch_size=cfg.get("batch_size", None),
        num_workers=cfg.get("num_workers", 4),
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        process_group=get_data_parallel_group(),
        bucket_config=cfg.get("bucket_config", None),
    )
    num_steps_per_epoch = len(dataloader)

    # ======================================================
    # 3. Construcción de Modelos (VAE + Discriminador)
    # ======================================================
    logger.info("Building PCURE-AI+ VAE Models...")
    
    # El modelo VAE (CST Engine Ready)
    model = build_module(cfg.model, MODELS, device_map=device, torch_dtype=dtype).train()
    if cfg.get("grad_checkpoint", False): set_grad_checkpoint(model)
    
    vae_loss_fn = VAELoss(**cfg.vae_loss_config, device=device, dtype=dtype)

    # EMA Model para estabilidad en realismo
    ema = deepcopy(model).cpu().eval().requires_grad_(False) if cfg.get("ema_decay") else None
    ema_shape_dict = record_model_param_shape(ema) if ema else None

    # Discriminador (Vital para el Parche B de Micro-textura)
    use_discriminator = cfg.get("discriminator") is not None
    if use_discriminator:
        discriminator = build_module(cfg.discriminator, MODELS).to(device, dtype).train()
        gen_loss_fn = GeneratorLoss(**cfg.gen_loss_config)
        disc_loss_fn = DiscriminatorLoss(**cfg.disc_loss_config)

    # Optimizadores
    optimizer = create_optimizer(model, cfg.optim)
    lr_scheduler = create_lr_scheduler(optimizer, num_steps_per_epoch, cfg.get("epochs", 1000), **cfg.lr_scheduler)

    if use_discriminator:
        disc_optimizer = create_optimizer(discriminator, cfg.optim_discriminator)
        disc_lr_scheduler = create_lr_scheduler(disc_optimizer, num_steps_per_epoch, cfg.get("epochs", 1000), **cfg.disc_lr_scheduler)

    # =======================================================
    # 4. Inyección Booster (Distribución de carga)
    # =======================================================
    model, optimizer, _, dataloader, lr_scheduler = booster.boost(model, optimizer, lr_scheduler, dataloader)
    if use_discriminator:
        discriminator, disc_optimizer, _, _, disc_lr_scheduler = booster.boost(discriminator, disc_optimizer, disc_lr_scheduler)

    # =======================================================
    # 5. Bucle de Entrenamiento
    # =======================================================
    start_epoch, start_step = 0, 0
    running_loss = {k: 0.0 for k in ["all", "nll", "nll_rec", "nll_per", "kl", "gen", "disc"]}
    accumulation_steps = int(cfg.get("accumulation_steps", 1))

    for epoch in range(start_epoch, cfg.get("epochs", 1000)):
        sampler.set_epoch(epoch)
        dataiter = iter(dataloader)
        
        with tqdm(enumerate(dataiter), desc=f"Epoch {epoch}", total=num_steps_per_epoch, disable=not coordinator.is_master()) as pbar:
            for step, batch in pbar:
                x = batch["video"].to(device, dtype, non_blocking=True)
                global_step = epoch * num_steps_per_epoch + step
                actual_step = (global_step + 1) // accumulation_steps

                # --- FORWARD VAE ---
                x_rec, posterior, z = model(x)
                
                # --- LOSS GENERADOR ---
                ret = vae_loss_fn(x, x_rec, posterior)
                vae_loss = ret["nll_loss"] + ret["kl_loss"]
                
                if use_discriminator:
                    discriminator.requires_grad_(False)
                    fake_logits = discriminator(x_rec.contiguous())
                    # El realismo B se apoya en get_last_layer() para el balance de gradientes
                    gen_loss, _ = gen_loss_fn(fake_logits, ret["nll_loss"], model.module.get_last_layer(), actual_step)
                    vae_loss += gen_loss
                    discriminator.requires_grad_(True)

                # --- BACKWARD GENERADOR ---
                booster.backward(vae_loss / accumulation_steps, optimizer)
                
                if (step + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    if lr_scheduler: lr_scheduler.step(actual_step)
                    if ema: update_ema(ema, model.unwrap(), optimizer, cfg.get("ema_decay", 0.999))

                # --- LOSS & BACKWARD DISCRIMINADOR ---
                if use_discriminator:
                    real_logits = discriminator(x.detach().contiguous())
                    fake_logits_disc = discriminator(x_rec.detach().contiguous())
                    disc_loss = disc_loss_fn(real_logits, fake_logits_disc, actual_step)
                    
                    booster.backward(disc_loss / accumulation_steps, disc_optimizer)
                    
                    if (step + 1) % accumulation_steps == 0:
                        disc_optimizer.step()
                        disc_optimizer.zero_grad()
                        if disc_lr_scheduler: disc_lr_scheduler.step(actual_step)

                # --- LOGGING & CHECKPOINTING ---
                if coordinator.is_master() and (step + 1) % cfg.get("log_every", 10) == 0:
                    pbar.set_postfix({"loss": f"{vae_loss.item():.4f}"})
                    if cfg.get("wandb"): wandb.log({"total_loss": vae_loss.item()}, step=actual_step)

                if (step + 1) % cfg.get("ckpt_every", 500) == 0:
                    checkpoint_io.save(booster, exp_dir, model=model, ema=ema, optimizer=optimizer, epoch=epoch, step=step)

    logger.info("Training PCURE-AI+ Complete.")

if __name__ == "__main__":
    main()