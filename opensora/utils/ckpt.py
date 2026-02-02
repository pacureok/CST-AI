import functools
import json
import operator
import os
import re
import shutil
from glob import glob
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from colossalai.booster import Booster
from colossalai.checkpoint_io import GeneralCheckpointIO
from colossalai.utils.safetensors import save as async_save
from colossalai.zero.low_level import LowLevelZeroOptimizer
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from tensornvme.async_file_io import AsyncFileWriter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from opensora.acceleration.parallel_states import get_data_parallel_group
from .logger import log_message

# Configuración de entorno para HuggingFace y TensorNVME
hf_endpoint = os.environ.get("HF_ENDPOINT")
if hf_endpoint is None:
    hf_endpoint = "https://huggingface.co"
os.environ["TENSORNVME_DEBUG"] = "1"

# ==========================================
# FUNCIONES DE CARGA (LOADING)
# ==========================================

def load_from_hf_hub(repo_path: str, cache_dir: str = None) -> str:
    repo_id = "/".join(repo_path.split("/")[:-1])
    repo_file = repo_path.split("/")[-1]
    ckpt_path = hf_hub_download(repo_id=repo_id, filename=repo_file, cache_dir=cache_dir)
    return ckpt_path

def load_from_sharded_state_dict(model: nn.Module, ckpt_path: str, model_name: str = "model", strict=False):
    ckpt_io = GeneralCheckpointIO()
    ckpt_io.load_model(model, os.path.join(ckpt_path, model_name), strict=strict)

def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 or len(unexpected) > 0:
        if len(missing) > 0:
            log_message(f"Missing keys: {len(missing)}\n\t" + "\n\t".join(missing[:5]) + "...")
        if len(unexpected) > 0:
            log_message(f"Unexpected keys: {len(unexpected)}\n\t" + "\n\t".join(unexpected[:5]) + "...")
    else:
        log_message("Model loaded successfully")

def load_checkpoint(
    model: nn.Module,
    path: str,
    cache_dir: str = None,
    device_map: torch.device | str = "cpu",
    cai_model_name: str = "model",
    strict: bool = False,
    rename_keys: dict = None,
) -> nn.Module:
    if not os.path.exists(path):
        log_message(f"Checkpoint not found at {path}, downloading from HF Hub...")
        path = load_from_hf_hub(path, cache_dir)
    
    log_message(f"Loading checkpoint from {path}")
    if path.endswith(".safetensors"):
        ckpt = load_file(path, device='cpu')
        if rename_keys:
            renamed_ckpt = {}
            for old_key, v in ckpt.items():
                new_key = old_key
                for old_p, new_p in rename_keys.items():
                    if old_p in old_key:
                        new_key = old_key.replace(old_p, new_p)
                renamed_ckpt[new_key] = v
            ckpt = renamed_ckpt
        missing, unexpected = model.load_state_dict(ckpt, strict=strict)
        print_load_warning(missing, unexpected)
    elif path.endswith((".pt", ".pth")):
        ckpt = torch.load(path, map_location=device_map)
        missing, unexpected = model.load_state_dict(ckpt, strict=strict)
        print_load_warning(missing, unexpected)
    else:
        load_from_sharded_state_dict(model, path, model_name=cai_model_name, strict=strict)
    return model

# ==========================================
# GESTIÓN DE MEMORIA Y ESPACIO (CLEANING)
# ==========================================

def rm_checkpoints(save_dir: str, keep_n_latest: int = 0):
    """
    Elimina checkpoints antiguos para ahorrar espacio en disco.
    Crítico para instancias de Cloud Shell con límites de almacenamiento.
    """
    if keep_n_latest <= 0 or (dist.is_initialized() and dist.get_rank() != 0):
        return
    
    files = glob(os.path.join(save_dir, "epoch*-global_step*"))
    # Ordenar por época y paso global usando Regex
    files = sorted(
        files, 
        key=lambda s: tuple(map(int, re.search(r"epoch(\d+)-global_step(\d+)", s).groups())), 
        reverse=True
    )
    
    to_remove = files[keep_n_latest:]
    for f in to_remove:
        log_message(f"Removing old checkpoint: {f}")
        for item in glob(os.path.join(f, "*")):
            if os.path.isdir(item) and os.path.basename(item) != "eval":
                shutil.rmtree(item)
            elif os.path.isfile(item):
                os.remove(item)

# ==========================================
# LÓGICA DE DISTRIBUCIÓN (SHARDING/GATHERING)
# ==========================================

def model_sharding(model: torch.nn.Module, device: torch.device = None):
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    for _, param in model.named_parameters():
        padding_size = (world_size - param.numel() % world_size) % world_size
        padding_param = torch.nn.functional.pad(param.data.view(-1), [0, padding_size])
        param.data = padding_param.split(padding_param.numel() // world_size)[global_rank].to(device)

def model_gathering(model: torch.nn.Module, model_shape_dict: dict, pinned_state_dict: dict):
    global_size = dist.get_world_size()
    for name, param in model.named_parameters():
        all_params = [torch.empty_like(param.data) for _ in range(global_size)]
        dist.all_gather(all_params, param.data, group=dist.group.WORLD)
        if dist.get_rank() == 0:
            gathered = torch.cat(all_params)[:functools.reduce(operator.mul, model_shape_dict[name])]
            pinned_state_dict[name].copy_(gathered.view(model_shape_dict[name]))

# ==========================================
# CLASE PRINCIPAL: CHECKPOINT IO
# ==========================================

class CheckpointIO:
    def __init__(self, n_write_entries: int = 32):
        self.writer = None
        self.pinned_state_dict = None
        self.master_pinned_state_dict = None

    def save(self, booster, save_dir, model=None, ema=None, optimizer=None, **kwargs) -> str:
        actual_step = kwargs.get('actual_update_step', 0)
        save_path = os.path.join(save_dir, f"epoch{kwargs.get('epoch', 0)}-global_step{actual_step}")
        
        if model:
            os.makedirs(os.path.join(save_path, "model"), exist_ok=True)
            booster.save_model(model, os.path.join(save_path, "model"), shard=True, use_safetensors=True)
        
        if optimizer:
            booster.save_optimizer(optimizer, os.path.join(save_path, "optimizer"), shard=True)
            
        if dist.get_rank() == 0:
            save_json(kwargs, os.path.join(save_path, "running_states.json"))
            if ema:
                # Guardado asíncrono de EMA para no bloquear la TPU
                self.writer = async_save(os.path.join(save_path, "ema.safetensors"), ema.state_dict())
        
        dist.barrier()
        return save_path

    def load(self, booster, load_dir, model=None, ema=None, optimizer=None, strict=False):
        states = load_json(os.path.join(load_dir, "running_states.json"))
        if model:
            booster.load_model(model, os.path.join(load_dir, "model"), strict=strict)
        if ema:
            path = os.path.join(load_dir, "ema.safetensors")
            ema.load_state_dict(load_file(path) if path.endswith("safetensors") else torch.load(path))
        if optimizer:
            booster.load_optimizer(optimizer, os.path.join(load_dir, "optimizer"))
        return states["epoch"], states["step"]

# Funciones auxiliares de JSON
def load_json(path): return json.load(open(path, "r"))
def save_json(data, path): json.dump(data, open(path, "w"), indent=4)