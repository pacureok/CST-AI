import argparse
import ast
import json
import os
from datetime import datetime

import torch
from mmengine.config import Config

from .logger import is_distributed, is_main_process


def parse_args() -> tuple[str, argparse.Namespace]:
    """
    Parsea el argumento principal del archivo de configuración.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Ruta al archivo de configuración del modelo")
    args, unknown_args = parser.parse_known_args()
    return args.config, unknown_args


def read_config(config_path: str) -> Config:
    """
    Lee el archivo de configuración usando mmengine.
    """
    cfg = Config.fromfile(config_path)
    return cfg


def parse_configs() -> Config:
    """
    Función principal que orquestra la lectura y el mezclado de argumentos.
    """
    config, args = parse_args()
    cfg = read_config(config)
    cfg = merge_args(cfg, args)
    cfg.config_path = config

    # Configuración estricta para la compresión espacial del AutoEncoder
    if cfg.get("ae_spatial_compression", None) is not None:
        os.environ["AE_SPATIAL_COMPRESSION"] = str(cfg.ae_spatial_compression)
    return cfg


def merge_args(cfg: Config, args: list) -> Config:
    """
    Mezcla los argumentos de la línea de comandos con el objeto Config.
    Permite el uso de sintaxis de punto para claves anidadas (ej: --model.patch_size 1).
    """
    for k, v in zip(args[::2], args[1::2]):
        assert k.startswith("--"), f"Argumento inválido: {k}"
        k = k[2:].replace("-", "_")
        k_split = k.split(".")
        target = cfg
        
        # Navegación en diccionarios anidados
        for key in k_split[:-1]:
            assert key in target, f"La clave '{key}' no se encuentra en la configuración"
            target = target[key]
        
        # Conversión automática de tipos basada en el valor original o inferencia
        if v.lower() == "none":
            v = None
        elif k_split[-1] in target:
            v_type = type(target[k_split[-1]])
            if v_type == bool:
                v = auto_convert(v)
            else:
                v = v_type(v)
        else:
            v = auto_convert(v)
        
        target[k_split[-1]] = v
    return cfg


def auto_convert(value: str) -> int | float | bool | list | dict | None:
    """
    Convierte cadenas a tipos de Python (int, float, bool, list, dict).
    """
    if value == "":
        return value
    if value.lower() == "none":
        return None

    lower_value = value.lower()
    if lower_value == "true":
        return True
    elif lower_value == "false":
        return False

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        pass

    return value


def sync_string(value: str):
    """
    Sincroniza una cadena de texto entre todos los procesos distribuidos.
    Vital para que todos los núcleos de la TPU usen el mismo nombre de carpeta.
    """
    if not is_distributed():
        return value
    
    bytes_value = value.encode("utf-8")
    max_len = 256
    # Se asume entorno CUDA/TPU para el tensor de bytes
    bytes_tensor = torch.zeros(max_len, dtype=torch.uint8).cuda()
    bytes_tensor[: len(bytes_value)] = torch.tensor(
        list(bytes_value), dtype=torch.uint8
    )
    torch.distributed.broadcast(bytes_tensor, 0)
    synced_value = bytes_tensor.cpu().numpy().tobytes().decode("utf-8").rstrip("\x00")
    return synced_value


def create_experiment_workspace(
    output_dir: str, model_name: str = None, config: dict = None, exp_name: str = None
) -> tuple[str, str]:
    """
    Crea el directorio de trabajo para el experimento y guarda el config.txt.
    """
    if exp_name is None:
        # Generar índice basado en tiempo y sincronizarlo
        experiment_index = datetime.now().strftime("%y%m%d_%H%M%S")
        experiment_index = sync_string(experiment_index)
        
        model_name_suffix = (
            "-" + model_name.replace("/", "-") if model_name is not None else ""
        )
        exp_name = f"{experiment_index}{model_name_suffix}"
    
    exp_dir = f"{output_dir}/{exp_name}"
    
    if is_main_process():
        os.makedirs(exp_dir, exist_ok=True)
        # Guardar la configuración final para reproducibilidad
        with open(f"{exp_dir}/config.txt", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

    return exp_name, exp_dir


def config_to_name(cfg: Config) -> str:
    """Genera un nombre legible a partir de la ruta del archivo de configuración."""
    filename = cfg._filename
    return filename.replace("configs/", "").replace(".py", "").replace("/", "_")


def parse_alias(cfg: Config) -> Config:
    """
    Mapea alias simplificados a las rutas de configuración profundas.
    Permite usar --resolution en lugar de --sampling_option.resolution.
    """
    aliases = {
        "resolution": ("sampling_option", "resolution"),
        "guidance": ("sampling_option", "guidance"),
        "guidance_img": ("sampling_option", "guidance_img"),
        "num_steps": ("sampling_option", "num_steps"),
        "num_frames": ("sampling_option", "num_frames"),
        "aspect_ratio": ("sampling_option", "aspect_ratio"),
    }
    
    for alias, (target_key, sub_key) in aliases.items():
        if cfg.get(alias, None) is not None:
            # Forzar tipo float para guías y int para pasos/frames
            val = cfg.get(alias)
            if "guidance" in alias:
                val = float(val)
            elif "num_" in alias:
                val = int(val)
            cfg[target_key][sub_key] = val
            
    if cfg.get("ckpt_path", None) is not None:
        cfg.model.from_pretrained = cfg.ckpt_path
        
    return cfg