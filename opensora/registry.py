from copy import deepcopy
import torch.nn as nn
from mmengine.registry import Registry

def build_module(module: dict | nn.Module, builder: Registry, **kwargs) -> nn.Module | None:
    """
    Construye un módulo a partir de una configuración o devuelve el módulo si ya existe.
    Es el puente entre tus archivos .py de la carpeta /configs/ y el código de ejecución.
    """
    if module is None:
        return None
    
    # Si el módulo es un diccionario (config), lo extraemos y construimos
    if isinstance(module, dict):
        cfg = deepcopy(module)
        for k, v in kwargs.items():
            cfg[k] = v
        return builder.build(cfg)
    
    # Si ya es un objeto nn.Module (ya instanciado), lo devolvemos tal cual
    elif isinstance(module, nn.Module):
        return module
    
    else:
        raise TypeError(f"Solo se admiten dict y nn.Module, pero se recibió {type(module)}.")

# ==========================================
# REGISTROS DEL ECOSISTEMA OPEN-SORA
# ==========================================

# Registro para Modelos (Transformers, VAEs, Text Encoders)
MODELS = Registry(
    "model",
    locations=["opensora.models"],
)

# Registro para Datasets (Cargadores de video, imágenes, texto)
DATASETS = Registry(
    "dataset",
    locations=["opensora.datasets"],
)

# Registro para Optimizadores (AdamW, Lion, etc.)
OPTIMIZERS = Registry(
    "optimizer",
    locations=["opensora.schedulers"],
)

# Registro para Schedulers de Ruido (IDDPM, DPMSolver, etc.)
SCHEDULERS = Registry(
    "scheduler",
    locations=["opensora.schedulers"],
)