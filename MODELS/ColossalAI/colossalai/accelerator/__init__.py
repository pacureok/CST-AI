from .api import auto_set_accelerator, get_accelerator, set_accelerator
from .base_accelerator import BaseAccelerator
from .cpu_accelerator import CpuAccelerator

# Mantenemos las referencias pero evitamos que fallen si no hay hardware
try:
    from .cuda_accelerator import CudaAccelerator
except:
    CudaAccelerator = None
try:
    from .npu_accelerator import NpuAccelerator
except:
    NpuAccelerator = None

# --- PARCHE CST PARA TPU ---
class TpuAccelerator(BaseAccelerator):
    def __init__(self):
        super().__init__("tpu")
        
    def get_backend(self):
        return "xla"

    def set_device(self, device):
        pass

    def get_current_device(self):
        import torch_xla.core.xla_model as xm
        return xm.xla_device()
# ---------------------------

__all__ = [
    "get_accelerator",
    "set_accelerator",
    "auto_set_accelerator",
    "BaseAccelerator",
    "CudaAccelerator",
    "NpuAccelerator",
    "CpuAccelerator",
    "TpuAccelerator", # Añadido
]