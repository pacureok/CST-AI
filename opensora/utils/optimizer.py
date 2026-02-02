import torch
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.nn.optimizer import HybridAdam
from torch.optim.lr_scheduler import _LRScheduler


def create_optimizer(
    model: torch.nn.Module,
    optimizer_config: dict,
) -> torch.optim.Optimizer:
    """
    Crea un optimizador para el modelo.
    Utiliza HybridAdam por defecto, que es altamente eficiente en TPUs/GPUs
    al combinar operaciones en CPU y acelerador.
    """
    optimizer_config = optimizer_config.copy()
    optimizer_name = optimizer_config.pop("cls", "HybridAdam")
    
    if optimizer_name == "HybridAdam":
        optimizer_cls = HybridAdam
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Solo optimizamos los parámetros que requieren gradiente
    optimizer = optimizer_cls(
        filter(lambda p: p.requires_grad, model.parameters()),
        **optimizer_config,
    )
    return optimizer


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    num_steps_per_epoch: int,
    epochs: int = 1000,
    warmup_steps: int | None = None,
    use_cosine_scheduler: bool = False,
    initial_lr: float = 1e-6,
) -> _LRScheduler | None:
    """
    Crea un planificador de tasa de aprendizaje (Learning Rate).
    Soporta calentamiento lineal (Warmup) y decaimiento de coseno.
    """
    if warmup_steps is None and not use_cosine_scheduler:
        lr_scheduler = None
    elif use_cosine_scheduler:
        # Decaimiento de coseno: suave y efectivo para modelos grandes
        lr_scheduler = CosineAnnealingWarmupLR(
            optimizer,
            total_steps=num_steps_per_epoch * epochs,
            warmup_steps=warmup_steps,
        )
    else:
        # Calentamiento lineal por defecto
        lr_scheduler = LinearWarmupLR(
            optimizer, 
            initial_lr=initial_lr, 
            warmup_steps=warmup_steps
        )

    return lr_scheduler


class LinearWarmupLR(_LRScheduler):
    """
    Implementación de calentamiento lineal de la tasa de aprendizaje.
    Sube gradualmente el LR desde un valor inicial bajo hasta el objetivo
    durante los primeros 'warmup_steps'.
    """

    def __init__(self, optimizer, initial_lr=0, warmup_steps: int = 0, last_epoch: int = -1):
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Incremento progresivo por paso
            return [
                self.initial_lr + (self.last_epoch + 1) / (self.warmup_steps + 1) * (lr - self.initial_lr)
                for lr in self.base_lrs
            ]
        else:
            # Una vez terminado el warmup, mantenemos el LR base
            return self.base_lrs