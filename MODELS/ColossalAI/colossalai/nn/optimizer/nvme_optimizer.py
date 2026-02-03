import math
import os
import tempfile
from typing import Callable, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

class NVMeOptimizer(torch.optim.Optimizer):
    """
    MODIFICADO POR PACUR AI+: Sistema de compatibilidad universal.
    Si tensornvme no está presente, el offload se desactiva automáticamente.
    """

    def __init__(
        self, params, defaults: dict, nvme_offload_fraction: float = 0.0, offload_dir: Optional[str] = None
    ) -> None:
        assert 0.0 <= nvme_offload_fraction <= 1.0
        super().__init__(params, defaults)
        self.nvme_offload_fraction = float(nvme_offload_fraction)
        
        # Bypass de seguridad para Pacur AI+
        self.offload_dir = None
        self.offloader = None
        
        if self.nvme_offload_fraction > 0.0:
            try:
                from tensornvme import DiskOffloader
                from tensornvme._C import get_backends
                
                self.offload_dir = offload_dir or tempfile.mkdtemp()
                backend = "uring" if "uring" in get_backends() else "aio"
                self.offloader = DiskOffloader(self.offload_dir, 8, backend=backend)
                print(f"🚀 NVMe Offloader activado en: {self.offload_dir}")
            except (ModuleNotFoundError, ImportError):
                # En lugar de raise, simplemente desactivamos la fracción
                print("⚠️  Pacur AI+ Aviso: tensornvme no detectado. Continuando en modo RAM/TPU.")
                self.nvme_offload_fraction = 0.0
                self.offload_dir = None
                self.offloader = None
        
        self.is_on_nvme: Dict[Parameter, bool] = {}
        self.offloaded_numel: int = 0
        self.total_numel: Optional[int] = None
        self.can_offload_numel: Optional[int] = None
        self.prefetch_params: List[Parameter] = []
        self.param_to_prefetch_idx: Dict[Parameter, int] = {}

    def _get_numel(self) -> int:
        numel = 0
        for group in self.param_groups:
            for p in group["params"]:
                try:
                    numel += p.storage().size()
                except:
                    numel += p.numel()
        return numel

    def _post_state_init(self, param: Parameter) -> None:
        if self.offloader is None:
            self.is_on_nvme[param] = False
            return
            
        numel = param.storage().size()
        if (param.device.type == "cpu" and numel + self.offloaded_numel <= self.can_offload_numel):
            self.is_on_nvme[param] = True
            self.offloaded_numel += numel
        else:
            self.is_on_nvme[param] = False

    def _setup_prefetch_params(self) -> List[Parameter]:
        if self.offloader is None:
            return
        assert len(self.prefetch_params) == 0 and len(self.param_to_prefetch_idx) == 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if len(self.state[p]) > 0 and self.is_on_nvme[p]:
                    self.param_to_prefetch_idx[p] = len(self.prefetch_params)
                    self.prefetch_params.append(p)

    def _pre_step(self, *state_keys: str) -> None:
        if self.total_numel is None:
            self.total_numel = self._get_numel()
            self.can_offload_numel = math.floor(self.total_numel * self.nvme_offload_fraction)
        
        self._setup_prefetch_params()
        if self.offloader is None or len(self.prefetch_params) == 0:
            return
        state = self.state[self.prefetch_params[0]]
        for key in state_keys:
            self.offloader.async_read(state[key])

    def _pre_update(self, param: Parameter, *state_keys: str) -> None:
        if self.offloader is None or param not in self.param_to_prefetch_idx:
            return
        self.offloader.sync_read_events()
        idx = self.param_to_prefetch_idx[param]
        if idx + 1 < len(self.prefetch_params):
            state = self.state[self.prefetch_params[idx + 1]]
            for key in state_keys:
                self.offloader.async_read(state[key])

    def _post_update(self, param: Parameter, *state_keys: str) -> None:
        if self.offloader is None:
            return
        self.offloader.sync_write_events()
        if self.is_on_nvme[param]:
            state = self.state[param]
            for key in state_keys:
                self.offloader.async_write(state[key])

    def _post_step(self) -> None:
        if self.offloader is not None:
            self.offloader.synchronize()
            self.prefetch_params.clear()
            self.param_to_prefetch_idx.clear()

    def step(self, closure: Optional[Callable[[], float]] = ...) -> Optional[float]:
        raise NotImplementedError

    def state_dict(self) -> dict:
        return super().state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)

    def __del__(self) -> None:
        if getattr(self, "offloader", None) is not None:
            del self.offloader
            if self.offload_dir and os.path.exists(self.offload_dir):
                try:
                    import shutil
                    shutil.rmtree(self.offload_dir)
                except:
                    pass
