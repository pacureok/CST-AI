#!/usr/bin/env python
from collections import OrderedDict
from typing import Union

# Importamos la base y las existentes
from .base_accelerator import BaseAccelerator
from .cpu_accelerator import CpuAccelerator

# Intentamos importar las demás, pero con fallback para evitar errores de compilación
try:
    from .cuda_accelerator import CudaAccelerator
except:
    CudaAccelerator = None
try:
    from .npu_accelerator import NpuAccelerator
except:
    NpuAccelerator = None

# --- INYECCIÓN DE CLASE TPU ---
# Definimos TpuAccelerator aquí mismo si no quieres crear un archivo extra
class TpuAccelerator(BaseAccelerator):
    def __init__(self):
        super().__init__("tpu")
    
    def is_available(self):
        try:
            import torch_xla.core.xla_model as xm
            return True
        except:
            return False

    def get_backend(self):
        return "xla"

    def set_device(self, device):
        pass

    def get_current_device(self):
        import torch_xla.core.xla_model as xm
        return xm.xla_device()
# ------------------------------

__all__ = ["set_accelerator", "auto_set_accelerator", "get_accelerator"]

_ACCELERATOR = None

# MODIFICACIÓN CRÍTICA: Ponemos 'tpu' al principio de la prioridad
_ACCELERATOR_MAPPING = OrderedDict()
_ACCELERATOR_MAPPING['tpu'] = TpuAccelerator
if CudaAccelerator: _ACCELERATOR_MAPPING['cuda'] = CudaAccelerator
if NpuAccelerator: _ACCELERATOR_MAPPING['npu'] = NpuAccelerator
_ACCELERATOR_MAPPING['cpu'] = CpuAccelerator


def set_accelerator(accelerator: Union[str, BaseAccelerator]) -> None:
    global _ACCELERATOR
    if isinstance(accelerator, str):
        _ACCELERATOR = _ACCELERATOR_MAPPING[accelerator]()
    elif isinstance(accelerator, BaseAccelerator):
        _ACCELERATOR = accelerator
    else:
        raise TypeError("accelerator must be either a string or an instance of BaseAccelerator")


def auto_set_accelerator() -> None:
    global _ACCELERATOR
    # Este bucle ahora encontrará 'tpu' primero
    for accelerator_name, accelerator_cls in _ACCELERATOR_MAPPING.items():
        if accelerator_cls is None: continue
        try:
            accelerator = accelerator_cls()
            # Si es TPU y está disponible (vía torch_xla), la seleccionamos
            if accelerator_name == "tpu" and accelerator.is_available():
                _ACCELERATOR = accelerator
                break
            if accelerator_name == "cpu" or (hasattr(accelerator, 'is_available') and accelerator.is_available()):
                _ACCELERATOR = accelerator
                break
        except:
            pass

    if _ACCELERATOR is None:
        raise RuntimeError("No accelerator is available (CST-TPU Patch failed).")


def get_accelerator() -> BaseAccelerator:
    global _ACCELERATOR
    if _ACCELERATOR is None:
        auto_set_accelerator()
    return _ACCELERATOR