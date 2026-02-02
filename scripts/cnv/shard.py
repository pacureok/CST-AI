import os
import pandas as pd
from tqdm import tqdm

try:
    import dask.dataframe as dd
    SUPPORT_DASK = True
except ImportError:
    SUPPORT_DASK = False

def shard_parquet(input_path, k):
    """
    Divide un archivo Parquet en K fragmentos para entrenamiento distribuido.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"El archivo {input_path} no existe.")

    print(f"[*] Cargando dataset desde {input_path}...")
    
    # Uso de Dask para archivos que superan la memoria RAM disponible
    if SUPPORT_DASK:
        df = dd.read_parquet(input_path).compute()
    else:
        df = pd.read_parquet(input_path)

    # Eliminamos columnas técnicas que no se usan durante el muestreo del entrenamiento
    # para reducir el peso de los fragmentos (shards).
    columns_to_remove = [
        "num_frames",
        "height",
        "width",
        "aspect_ratio",
        "fps",
        "resolution",
    ]
    df = df.drop(columns=[col for col in columns_to_remove if col in df.columns])

    total_rows = len(df)
    rows_per_shard = (total_rows + k - 1) // k 

    # Organizar salida en carpeta dedicada
    base_dir = os.path.dirname(input_path)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_dir = os.path.join(base_dir, base_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"[*] Creando {k} shards en: {output_dir}")

    # Proceso de fragmentación
    for i in tqdm(range(k)):
        start_idx = i * rows_per_shard
        end_idx = min(start_idx + rows_per_shard, total_rows)

        shard_df = df.iloc[start_idx:end_idx]
        if shard_df.empty:
            continue

        # Formato de nombre 00001.parquet, 00002.parquet...
        shard_file_name = f"{i + 1:05d}.parquet"
        shard_path = os.path.join(output_dir, shard_file_name)

        shard_df.to_parquet(shard_path, index=False)

    print(f"[✅] Fragmentación completada. {total_rows} registros distribuidos.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PCURE-AI+ Sharding Tool")
    parser.add_argument("input_path", type=str, help="Ruta al archivo .parquet maestro")
    parser.add_argument("k", type=int, nargs='?', default=100, help="Número de fragmentos (shards)")

    args = parser.parse_args()
    shard_parquet(args.input_path, args.k)