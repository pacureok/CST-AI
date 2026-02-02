# ======================================================
# PCURE-AI+ Config: Video DC-AE (Deep Compression)
# ======================================================

dtype = "bf16"
batch_size = 1
seed = 42
save_dir = "samples/video_dc_ae"

# ======================================================
# Dataset & Loading
# ======================================================
dataset = dict(
    type="video_text",
    transform_name="resize_crop",
    fps_max=16,
    data_path="datasets/pexels_45k_necessary.csv",
)

bucket_config = {
    "512px_ar1:1": {96: (1.0, 1)}, # 96 frames es múltiplo de la compresión temporal (4)
}

num_workers = 24
num_bucket_build_workers = 16
prefetch_factor = 4

# ======================================================
# Model Architecture: DC-AE (Symmetry and Precision)
# ======================================================
model = dict(
    type="dc_ae",
    model_name="dc-ae-f32t4c128", # F32: Spatial Downsample | T4: Temporal | C128: Channels
    from_pretrained="./ckpts/F32T4C128_AE.safetensors",
    from_scratch=True,
    
    # Configuración de Tiling Avanzada
    use_spatial_tiling=True,
    use_temporal_tiling=True,
    spatial_tile_size=256,     # Tamaño del bloque procesado en GPU
    temporal_tile_size=32,     # Segmentos de tiempo para evitar jittering
    tile_overlap_factor=0.25,  # 25% de solapamiento para evitar costuras visuales
)