import logging
import os

import torch.distributed as dist


def is_distributed() -> bool:
    """
    Verifica si el código se está ejecutando en un entorno distribuido (múltiples GPUs/TPUs).
    """
    return os.environ.get("WORLD_SIZE", None) is not None


def is_main_process() -> bool:
    """
    Verifica si el proceso actual es el principal (Rank 0).
    """
    return not is_distributed() or dist.get_rank() == 0


def get_world_size() -> int:
    """
    Obtiene el número total de procesos/núcleos en ejecución.
    """
    if is_distributed():
        return dist.get_world_size()
    else:
        return 1


def create_logger(logging_dir: str = None) -> logging.Logger:
    """
    Crea un logger que escribe en un archivo y en la terminal.
    Solo el proceso principal (Rank 0) realiza el registro real.
    """
    if is_main_process():
        additional_args = dict()
        if logging_dir is not None:
            # Configura salida doble: Consola + Archivo log.txt
            additional_args["handlers"] = [
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ]
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s", # Color azul para la fecha
            datefmt="%Y-%m-%d %H:%M:%S",
            **additional_args,
        )
        logger = logging.getLogger(__name__)
        if logging_dir is not None:
            logger.info("Directorio del experimento creado en: %s", logging_dir)
    else:
        # Los procesos esclavos usan un NullHandler (no imprimen nada)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def log_message(*args, level: str = "info"):
    """
    Envía un mensaje al logger centralizado.
    """
    logger = logging.getLogger(__name__)
    if level == "info":
        logger.info(*args)
    elif level == "warning":
        logger.warning(*args)
    elif level == "error":
        logger.error(*args)
    elif level == "print":
        print(*args)
    else:
        raise ValueError(f"Nivel de logging inválido: {level}")