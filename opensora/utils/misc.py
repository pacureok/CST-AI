import os
import time
from collections import OrderedDict
from collections.abc import Sequence
from contextlib import nullcontext

import numpy as np
import psutil
import torch
import torch.distributed as dist
import torch.nn as nn
from colossalai.cluster.dist_coordinator import DistCoordinator
from torch.utils.tensorboard import SummaryWriter

from opensora.acceleration.parallel_states import get_data_parallel_group
from .logger import log_message


def create_tensorboard_writer(exp_dir: str) -> SummaryWriter:
    """Crea un escritor de Tensorboard para visualizar métricas."""
    tensorboard_dir = f"{exp_dir}/tensorboard"
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(tensorboard_dir)
    return writer


# ======================================================
# Gestión de Memoria (VRAM y RAM)
# ======================================================

GIGABYTE = 1024**3

def log_cuda_memory(stage: str = None):
    """Registra el uso actual de memoria CUDA (VRAM)."""
    text = "CUDA memory usage"
    if stage is not None:
        text += f" at {stage}"
    log_message(text + ": %.1f GB", torch.cuda.memory_allocated() / GIGABYTE)


def log_cuda_max_memory(stage: str = None):
    """Registra el pico máximo de memoria CUDA alcanzado."""
    torch.cuda.synchronize()
    max_memory_allocated = torch.cuda.max_memory_allocated()
    max_memory_reserved = torch.cuda.max_memory_reserved()
    log_message("CUDA max memory allocated at " + stage + ": %.1f GB", max_memory_allocated / GIGABYTE)
    log_message("CUDA max memory reserved at " + stage + ": %.1f GB", max_memory_reserved / GIGABYTE)


def get_process_mem():
    """Obtiene el uso de memoria RAM del proceso actual."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / GIGABYTE


def print_mem(prefix: str = ""):
    """Imprime el estado de memoria del sistema y del proceso."""
    rank = dist.get_rank()
    print(
        f"[{rank}] {prefix} process memory: {get_process_mem():.2f} GB, total memory: {psutil.virtual_memory().used / GIGABYTE:.2f} GB",
        flush=True,
    )

# ======================================================
# Análisis de Parámetros del Modelo
# ======================================================

def get_model_numel(model: torch.nn.Module) -> tuple[int, int]:
    """Cuenta el número total y entrenable de parámetros."""
    num_params = sum(p.numel() for p in model.parameters())
    num_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params, num_params_trainable


def log_model_params(model: nn.Module):
    """Registra de forma legible el tamaño del modelo."""
    num_params, num_params_trainable = get_model_numel(model)
    model_name = model.__class__.__name__
    log_message(f"[{model_name}] Total Parameters: {format_numel_str(num_params)}")
    log_message(f"[{model_name}] Trainable Parameters: {format_numel_str(num_params_trainable)}")


def format_numel_str(numel: int) -> str:
    """Formatea números grandes a K, M o B (Billones)."""
    if numel >= 1024**3: return f"{numel / 1024**3:.2f} B"
    elif numel >= 1024**2: return f"{numel / 1024**2:.2f} M"
    elif numel >= 1024: return f"{numel / 1024:.2f} K"
    return f"{numel}"

# ======================================================
# Herramientas de Perfilado (Profiling)
# ======================================================

class Timer:
    """Cronómetro para medir bloques de código con sincronización CUDA."""
    def __init__(self, name, log=False, barrier=False, coordinator=None):
        self.name, self.log, self.barrier, self.coordinator = name, log, barrier, coordinator

    def __enter__(self):
        torch.cuda.synchronize()
        if self.barrier: dist.barrier()
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        torch.cuda.synchronize()
        self.end_time = time.time()
        if self.log:
            print(f"Elapsed time for {self.name}: {self.end_time - self.start_time:.2f} s")

class NsysProfiler:
    """Integra NVIDIA Nsight Systems para análisis profundo de kernels CUDA."""
    def __init__(self, warmup_steps=0, num_steps=1, enabled=True):
        self.warmup_steps, self.num_steps, self.enabled = warmup_steps, num_steps, enabled
        self.current_step = 0

    def step(self):
        if not self.enabled: return
        self.current_step += 1
        if self.current_step == self.warmup_steps:
            torch.cuda.cudart().cudaProfilerStart()
        elif self.current_step >= self.warmup_steps + self.num_steps:
            torch.cuda.cudart().cudaProfilerStop()

# ======================================================
# Utilidades de Conversión y Reducción
# ======================================================

def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Promedia un tensor entre todos los núcleos distribuidos."""
    dist.all_reduce(tensor=tensor, group=get_data_parallel_group())
    tensor.div_(dist.get_world_size(group=get_data_parallel_group()))
    return tensor

def to_torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    """Mapea cadenas de texto a tipos de datos de PyTorch."""
    mapping = {
        "fp32": torch.float32, "float32": torch.float32,
        "fp16": torch.float16, "half": torch.float16,
        "bf16": torch.bfloat16,
    }
    return mapping.get(dtype, dtype) if isinstance(dtype, str) else dtype