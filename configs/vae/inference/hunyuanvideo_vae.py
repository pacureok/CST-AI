# ======================================================
# PCURE-AI+ Inference Config: HunyuanVideo VAE
# ======================================================

# Precisión de cómputo (bf16 es ideal para mantener el rango dinámico del Parche B)
dtype = "bf16"
batch_size = 1
seed = 42
save_dir = "samples/hunyuanvideo_vae"

# Aceleración distribuida (Zero2 para eficiencia de memoria en múltiples GPUs)
plugin = "zero2"

# ======================================================
# Dataset & Dataloader
# ======================================================
dataset = dict(
    type="video_text",
    transform_name="resize_crop",
    fps_max=16,
    data_path="datasets/pexels_45k_necessary.csv",
)

# Configuración de Buckets para evitar distorsiones en el aspecto
bucket_config = {
    "512px_ar1:1": {97: (1.0, 1)},  # Formato cuadrado para validación técnica
}

num_workers = 24
num_bucket_build_workers = 16
prefetch_factor = 4

# ======================================================
# Model Architecture (CST Engine Ready)
# ======================================================
model = dict(
    type="hunyuan_vae",
    from_pretrained="./ckpts/hunyuan_vae.safetensors",
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    latent_channels=16,      # Alta densidad de información latente
    scale_factor=0.476986,   # Factor de normalización estándar de Hunyuan
    shift_factor=0,
    
    # Tiling: Esencial para generar resoluciones 720p/1080p en hardware comercial
    use_spatial_tiling=True,  
    use_temporal_tiling=True,
    
    # Compresión temporal: 4 frames de video = 1 frame latente
    time_compression_ratio=4,
)