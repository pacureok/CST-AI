import colossalai
import torch
import torch.distributed as dist
from colossalai.booster import Booster
from colossalai.cluster import DistCoordinator

from opensora.acceleration.parallel_states import (
    get_sequence_parallel_group,
    get_tensor_parallel_group,
    set_sequence_parallel_group,
)
from opensora.models.hunyuan_vae.policy import HunyuanVaePolicy
from opensora.models.mmdit.distributed import MMDiTPolicy
from opensora.utils.logger import is_distributed
from opensora.utils.train import create_colossalai_plugin

from .logger import log_message


def set_group_size(plugin_config: dict):
    """
    Establece el tamaño de grupo para paralelismo de tensor (TP) o secuencia (SP).
    Para TPU, esto asegura que los 8 núcleos se dividan correctamente.
    """
    tp_size = int(plugin_config.get("tp_size", 1))
    sp_size = int(plugin_config.get("sp_size", 1))
    
    # Regla: No se permite TP y SP simultáneos en esta implementación híbrida
    if tp_size > 1:
        assert sp_size == 1
        # Asegura que no intentemos usar más núcleos de los que existen (ej. 8 en TPU v3-8)
        device_count = torch.cuda.device_count() if torch.cuda.is_available() else 8
        plugin_config["tp_size"] = tp_size = min(tp_size, device_count)
        log_message(f"Using TP with size {tp_size}")
        
    if sp_size > 1:
        assert tp_size == 1
        device_count = torch.cuda.device_count() if torch.cuda.is_available() else 8
        plugin_config["sp_size"] = sp_size = min(sp_size, device_count)
        log_message(f"Using SP with size {sp_size}")


def init_inference_environment():
    """
    Inicializa el entorno distribuido para la generación de video.
    """
    if is_distributed():
        colossalai.launch_from_torch({})
        coordinator = DistCoordinator()
        # Si hay múltiples núcleos, activamos SP para permitir videos largos (Ring Attention)
        enable_sequence_parallelism = coordinator.world_size > 1
        if enable_sequence_parallelism:
            set_sequence_parallel_group(dist.group.WORLD)


def get_booster(cfg: dict, ae: bool = False):
    """
    Crea el acelerador Booster basado en la configuración.
    ae=True aplica políticas para el Autoencoder, ae=False para el Transformer (MMDiT).
    """
    suffix = "_ae" if ae else ""
    # Selecciona la política de distribución de pesos según el componente
    policy = HunyuanVaePolicy if ae else MMDiTPolicy

    plugin_type = cfg.get(f"plugin{suffix}", "zero2")
    plugin_config = cfg.get(f"plugin_config{suffix}", {})
    plugin_kwargs = {}
    booster = None
    
    if plugin_type == "hybrid":
        set_group_size(plugin_config)
        plugin_kwargs = dict(custom_policy=policy)

        # Creamos el plugin de aceleración (ZeRO + TP/SP)
        plugin = create_colossalai_plugin(
            plugin=plugin_type,
            dtype=cfg.get("dtype", "bf16"), # bf16 es nativo en TPU
            grad_clip=cfg.get("grad_clip", 0),
            **plugin_config,
            **plugin_kwargs,
        )
        booster = Booster(plugin=plugin)
    else:
        # Fallback para entrenamientos simples sin hibridación
        plugin = create_colossalai_plugin(plugin=plugin_type)
        booster = Booster(plugin=plugin)
        
    return booster


def get_is_saving_process(cfg: dict):
    """
    Determina si el proceso actual es el 'Rank 0' encargado de guardar checkpoints.
    Esto evita que los 8 núcleos de la TPU escriban en el mismo archivo simultáneamente.
    """
    plugin_type = cfg.get("plugin", "zero2")
    plugin_config = cfg.get("plugin_config", {})
    
    # Es el proceso de guardado si:
    # 1. No es un plugin híbrido (single process)
    # 2. Es el líder del grupo de Paralelismo de Tensores
    # 3. Es el líder del grupo de Paralelismo de Secuencia
    is_saving_process = (
        plugin_type != "hybrid"
        or (plugin_config.get("tp_size", 1) > 1 and dist.get_rank(get_tensor_parallel_group()) == 0)
        or (plugin_config.get("sp_size", 1) > 1 and dist.get_rank(get_sequence_parallel_group()) == 0)
    )
