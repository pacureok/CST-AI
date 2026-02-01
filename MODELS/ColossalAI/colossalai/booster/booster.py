from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader

from colossalai.logging import get_dist_logger

SUPPORT_PEFT = False
try:
    import peft
    SUPPORT_PEFT = True
except ImportError:
    pass

import colossalai.interface.pretrained as pretrained_utils
from colossalai.checkpoint_io import GeneralCheckpointIO
from colossalai.interface import ModelWrapper, OptimizerWrapper
from colossalai.quantization import BnbQuantizationConfig

from .accelerator import get_accelerator # Modificado para usar tu API parcheada
from .mixed_precision import MixedPrecision, mixed_precision_factory
from .plugin import Plugin
from .plugin.pp_plugin_base import PipelinePluginBase

__all__ = ["Booster"]


class Booster:
    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        mixed_precision: Optional[Union[MixedPrecision, str]] = None,
        plugin: Optional[Plugin] = None,
    ) -> None:
        if plugin is not None:
            assert isinstance(
                plugin, Plugin
            ), f"Expected the argument plugin to be an instance of Plugin, but got {type(plugin)}."
        self.plugin = plugin
        self.logger = get_dist_logger()

        # --- PARCHE CST: Lógica de selección de Acelerador TPU ---
        if self.plugin and self.plugin.control_device():
            self.accelerator = None
            if device is not None:
                self.logger.warning(
                    "The plugin will control the accelerator, so the device argument will be ignored.", ranks=[0]
                )
        else:
            # Si estamos en Kaggle/TPU y device es None o 'cuda', forzamos el acelerador parcheado
            if device is None or device == 'cuda':
                self.accelerator = get_accelerator() # Retornará TpuAccelerator por tu parche en api.py
            else:
                from .accelerator import Accelerator
                self.accelerator = Accelerator(device)
        # -------------------------------------------------------

        # Configuración de precisión (BF16 es nativo en TPU)
        if self.plugin and self.plugin.control_precision():
            self.mixed_precision = None
        elif mixed_precision is None:
            # Sugerencia: En TPU v5e, podrías querer forzar 'bf16' por defecto aquí
            self.mixed_precision = None
        else:
            if isinstance(mixed_precision, str):
                self.mixed_precision = mixed_precision_factory(mixed_precision)
            elif isinstance(mixed_precision, MixedPrecision):
                self.mixed_precision = mixed_precision
            else:
                raise ValueError(f"Mixed_precision must be str or MixedPrecision, got {type(mixed_precision)}")

        if self.plugin is not None and self.plugin.control_checkpoint_io():
            self.checkpoint_io = self.plugin.get_checkpoint_io()
        else:
            self.checkpoint_io = GeneralCheckpointIO()

    def boost(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        criterion: Optional[Callable] = None,
        dataloader: Optional[DataLoader] = None,
        lr_scheduler: Optional[LRScheduler] = None,
    ) -> List[Union[nn.Module, Optimizer, LRScheduler, DataLoader]]:
        
        pretrained_path = pretrained_utils.get_pretrained_path(model)
        
        if self.plugin:
            model, optimizer, criterion, dataloader, lr_scheduler = self.plugin.configure(
                model, optimizer, criterion, dataloader, lr_scheduler
            )

        # Configuración del modelo para el acelerador (Mueve el modelo a XLA)
        if self.plugin and not self.plugin.control_device():
            model = self.accelerator.configure_model(model)
        elif not self.plugin:
            # Si no hay plugin, usamos nuestro acelerador de TPU para mover el modelo
            import torch_xla.core.xla_model as xm
            model = model.to(xm.xla_device())

        if self.mixed_precision and (self.plugin is None or not self.plugin.control_precision()):
            model, optimizer, criterion = self.mixed_precision.configure(model, optimizer, criterion)

        if pretrained_path:
            self.load_model(model, pretrained_path)
            orig_model = model.unwrap() if isinstance(model, ModelWrapper) else model
            pretrained_utils.set_pretrained_path(orig_model, None)

        return model, optimizer, criterion, dataloader, lr_scheduler

    def backward(self, loss: torch.Tensor, optimizer: Optimizer) -> None:
        # En TPU/XLA, el backward es estándar, pero xm.optimizer_step manejará la sincronización
        optimizer.backward(loss)
    
    # ... (Resto de métodos como execute_pipeline y load/save se mantienen igual)