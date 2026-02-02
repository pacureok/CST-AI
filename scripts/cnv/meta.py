import argparse
import os
import numpy as np
import pandas as pd
from pandarallel import pandarallel
from torchvision.io.video import read_video
from tqdm import tqdm

def set_parallel(num_workers: int = None) -> callable:
    """Configura el procesamiento paralelo para acelerar el escaneo de videos."""
    if num_workers == 0:
        return lambda x, *args, **kwargs: x.progress_apply(*args, **kwargs)
    else:
        # Inicializa pandarallel para usar múltiples núcleos de CPU
        nb_workers = num_workers if num_workers is not None else os.cpu_count()
        pandarallel.initialize(progress_bar=True, nb_workers=nb_workers, use_memory_fs=False)
        return lambda x, *args, **kwargs: x.parallel_apply(*args, **kwargs)

def get_video_info(path: str) -> pd.Series:
    """Extrae metadatos críticos de cada archivo de video."""
    try:
        # pts_unit="sec" es vital para la precisión del tiempo en el Parche B
        vframes, _, vinfo = read_video(path, pts_unit="sec", output_format="TCHW")
        
        if vframes.shape[0] == 0:
            raise ValueError("Video vacío o corrupto")
            
        num_frames, C, height, width = vframes.shape
        fps = round(float(vinfo.get("video_fps", 0)), 3)
        aspect_ratio = height / width if width > 0 else np.nan
        resolution = height * width

        return pd.Series(
            [height, width, fps, num_frames, aspect_ratio, resolution, "valid"],
            index=["height", "width", "fps", "num_frames", "aspect_ratio", "resolution", "status"],
        )
    except Exception as e:
        # Retorna serie con error para no detener el proceso completo
        return pd.Series([0, 0, 0, 0, 0, 0, f"error: {str(e)}"], 
                         index=["height", "width", "fps", "num_frames", "aspect_ratio", "resolution", "status"])

def parse_args():
    parser = argparse.ArgumentParser(description="PCURE-AI+ Meta Data Extractor")
    parser.add_argument("--input", type=str, required=True, help="Ruta al CSV con columna 'path'")
    parser.add_argument("--output", type=str, required=True, help="Ruta de salida para el CSV procesado")
    parser.add_argument("--num_workers", type=int, default=None, help="Número de hilos (default: max CPU)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.input):
        print(f"[!] Error: El archivo de entrada {args.input} no existe.")
        return

    # 1. Cargar Dataset
    df = pd.read_csv(args.input)
    if "path" not in df.columns:
        print("[!] Error: El CSV debe contener una columna llamada 'path'.")
        return

    print(f"[*] Procesando {len(df)} videos para PCURE-AI+...")
    
    # 2. Configurar Paralelismo
    tqdm.pandas()
    apply_func = set_parallel(args.num_workers)

    # 3. Ejecutar Extracción
    # El uso de parallel_apply distribuye el costo de decodificación de video
    result = apply_func(df["path"], get_video_info)
    
    # 4. Integrar resultados y Guardar
    for col in result.columns:
        df[col] = result[col]
    
    # Filtrar automáticamente videos corruptos para el entrenamiento
    clean_df = df[df["status"] == "valid"].drop(columns=["status"])
    clean_df.to_csv(args.output, index=False)
    
    print(f"\n[✅] Metadatos listos en: {args.output}")
    print(f"[*] Videos válidos: {len(clean_df)} | Errores: {len(df) - len(clean_df)}")

if __name__ == "__main__":
    main()