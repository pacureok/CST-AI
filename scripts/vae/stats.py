from pprint import pformat
import colossalai
import torch
from colossalai.utils import get_current_device, set_seed
from tqdm import tqdm

from opensora.acceleration.parallel_states import get_data_parallel_group
from opensora.datasets.dataloader import prepare_dataloader
from opensora.registry import DATASETS, MODELS, build_module
from opensora.utils.config import parse_configs
from opensora.utils.logger import create_logger, is_distributed, is_main_process
from opensora.utils.misc import log_cuda_max_memory, log_model_params, to_torch_dtype

@torch.inference_mode()
def main():
    torch.set_grad_enabled(False)
    
    # 1. Configuración y Entorno Runtime
    cfg = parse_configs()
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if is_distributed():
        colossalai.launch_from_torch({})
        device = get_current_device()
    set_seed(cfg.get("seed", 1024))

    logger = create_logger()
    logger.info("PCURE-AI+ Stats Configuration:\n %s", pformat(cfg.to_dict()))
    verbose = cfg.get("verbose", 1)

    # 2. Construcción del Modelo VAE
    if cfg.get("ckpt_path", None) is not None:
        cfg.model.from_pretrained = cfg.ckpt_path
    
    logger.info("Building PCURE VAE for latent analysis...")
    model = build_module(cfg.model, MODELS, device_map=device, torch_dtype=dtype).eval()
    log_model_params(model)

    # 3. Preparación de Dataset y Dataloader
    logger.info("Loading dataset samples...")
    dataset = build_module(cfg.dataset, DATASETS)
    
    dataloader_args = dict(
        dataset=dataset,
        batch_size=cfg.get("batch_size", 1), # Se recomienda batch_size bajo para stats precisas
        num_workers=cfg.get("num_workers", 4),
        seed=cfg.get("seed", 1024),
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        process_group=get_data_parallel_group(),
    )

    # Ajuste de resolución para la evaluación
    if cfg.get("eval_setting", None) is not None:
        num_frames = int(cfg.eval_setting.split("x")[0])
        resolution = str(cfg.eval_setting.split("x")[-1])
        bucket_config = {f"{resolution}px_ar1:1": {num_frames: (1.0, 1)}}
    else:
        bucket_config = cfg.get("bucket_config", None)

    dataloader, _ = prepare_dataloader(
        bucket_config=bucket_config,
        num_bucket_build_workers=cfg.get("num_bucket_build_workers", 1),
        **dataloader_args,
    )
    
    dataiter = iter(dataloader)
    num_steps = len(dataloader)

    # 4. Inferencia y Cálculo de Estadísticas (Welford's Algorithm para estabilidad)
    num_samples = 0
    running_sum = 0.0
    running_var = 0.0

    

    with tqdm(enumerate(dataiter), disable=not is_main_process() or verbose < 1, total=num_steps) as pbar:
        for _, batch in pbar:
            x = batch["video"].to(device, dtype)  # Input: [B, C, T, H, W]

            # Obtenemos los latentes del Encoder modificado
            # z suele ser de forma [B, D, T', H', W']
            z = model.encode(x)
            
            # En VAEs como SDXL o Hunyuan, z puede ser una distribución; extraemos el promedio
            if hasattr(z, 'latent_dist'):
                z = z.latent_dist.sample()
            elif isinstance(z, (list, tuple)):
                z = z[0]

            # Actualización de estadísticas
            batch_mean = z.mean().item()
            num_samples += 1
            
            # Cálculo de media acumulada: 
            # $\mu_{n} = \mu_{n-1} + \frac{x_n - \mu_{n-1}}{n}$
            running_sum += batch_mean
            shift = running_sum / num_samples
            
            # Cálculo de varianza acumulada
            running_var += (z - shift).pow(2).mean().item()
            scale = (running_var / num_samples) ** 0.5
            
            pbar.set_postfix({"latent_mean": f"{shift:.4f}", "latent_std": f"{scale:.4f}"})

    logger.info("--- PCURE-AI+ Final Latent Stats ---")
    logger.info("Final Mean (Shift): %.6f", shift)
    logger.info("Final Std (Scale): %.6f", scale)
    log_cuda_max_memory("inference")

if __name__ == "__main__":
    main()