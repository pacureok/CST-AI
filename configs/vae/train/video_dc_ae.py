# ======================================================
# PCURE-AI+ Training Config: Video DC-AE Base
# ======================================================

# ============================
# 1. Model Configuration
# ============================
model = dict(
    type="dc_ae",
    model_name="dc-ae-f32t4c128", # F32 (Espacial), T4 (Temporal), C128 (Canales latentes)
    from_scratch=True,             # Inicia entrenamiento desde cero para adaptar a Pexels
    from_pretrained=None,
)

# ============================
# 2. Data Configuration
# ============================
dataset = dict(
    type="video_text",
    transform_name="resize_crop",
    data_path="datasets/pexels_45k_necessary.csv",
    fps_max=24,                    # Estándar cinemático para PCURE
)

# Configuración de Buckets (Resolución Progresiva)
bucket_config = {
    "256px_ar1:1": {32: (1.0, 1)}, # 32 frames a 256x256: ideal para estabilidad inicial
}

num_bucket_build_workers = 64
num_workers = 12
prefetch_factor = 2

# ============================
# 3. Training Strategy
# ============================
# Mezcla de imágenes y videos para mantener fidelidad en texturas estáticas
mixed_strategy = "mixed_video_image"
mixed_image_ratio = 0.2  # 20% imágenes, 80% videos

# Optimizador de alto rendimiento
optim = dict(
    cls="HybridAdam",
    lr=5e-5,
    eps=1e-8,
    weight_decay=0.0,
    adamw_mode=True,
    betas=(0.9, 0.98),
)

lr_scheduler = dict(warmup_steps=0)
update_warmup_steps = True

# Precisión y Paralelismo
dtype = "bf16"
plugin = "zero2"
plugin_config = dict(
    reduce_bucket_size_in_m=128,
    overlap_allgather=False,
)

# Control de gradientes y memoria
grad_clip = 1.0
grad_checkpoint = False # Cambiar a True si subes a 512px o 720p
pin_memory_cache_pre_alloc_numels = [50 * 1024 * 1024] * num_workers * prefetch_factor

# ============================
# 4. Logs & Checkpoints
# ============================
seed = 42
outputs = "outputs/dc_ae_pcure"
epochs = 100
log_every = 10
ckpt_every = 3000
keep_n_latest = 50
ema_decay = 0.99
wandb_project = "pcure-dcae-train"

# ============================
# 5. Loss Configuration
# ============================
vae_loss_config = dict(
    perceptual_loss_weight=0.5, # Mantiene la nitidez visual (LPIPS)
    kl_loss_weight=0,            # 0 para AE puro, >0 si buscas Variacional (VAE)
)