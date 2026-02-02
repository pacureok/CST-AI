# ======================================================
# PCURE-AI+ Foundation: FLUX Image Training Config
# ======================================================

# 1. Dataset & Quality Metrics
dataset = dict(
    type="video_text",
    transform_name="resize_crop",
    fps_max=24,
    vmaf=True,  # Carga puntuaciones de calidad visual para filtrar el mejor contenido
)

# 2. Bucketing Estratégico (Solo Imágenes: frame=1)
bucket_config = {
    "256px": {1: (1.0, 50)},   # Batch size grande para estabilización
    "768px": {1: (0.5, 11)},   # Alta resolución media
    "1024px": {1: (0.5, 7)},   # Máximo detalle para Parche B
}

# 3. Model Architecture (FLUX Engine)
grad_ckpt_settings = (8, 100)
model = dict(
    type="flux",
    from_pretrained=None,
    strict_load=False,
    guidance_embed=False,
    fused_qkv=False,
    use_liger_rope=True,  # Optimización de RoPE para secuencias largas
    grad_ckpt_settings=grad_ckpt_settings,
    
    # Dimensiones masivas para ultra-realismo
    in_channels=64,
    vec_in_dim=768,       # CLIP guidance
    context_in_dim=4096,  # T5 XXL guidance
    hidden_size=3072,
    mlp_ratio=4.0,
    num_heads=24,
    depth=19,
    depth_single_blocks=38,
    axes_dim=[16, 56, 56],
    theta=10_000,
    qkv_bias=True,
)

# Probabilidad de Dropout para mejorar el seguimiento de prompts (CFG)
dropout_ratio = {
    "t5": 0.31622777,
    "clip": 0.31622777,
}

# 4. AutoEncoder (Hunyuan VAE por defecto para Stage 0)
ae = dict(
    type="hunyuan_vae",
    from_pretrained="./ckpts/hunyuan_vae.safetensors",
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    latent_channels=16,
    use_spatial_tiling=True,
    use_temporal_tiling=False, # Desactivado para imágenes estáticas
)
is_causal_vae = True

# 5. Text Embedders (Dual Encoder Setup)
t5 = dict(
    type="text_embedder",
    from_pretrained="google/t5-v1_1-xxl",
    cache_dir="/mnt/ddn/sora/tmp_load/huggingface/hub/",
    max_length=512,
    shardformer=True, # Fragmentación para ahorrar VRAM
)
clip = dict(
    type="text_embedder",
    from_pretrained="openai/clip-vit-large-patch14",
    cache_dir="/mnt/ddn/sora/tmp_load/huggingface/hub/",
    max_length=77,
)

# 6. Optimization Settings
lr = 1e-5
eps = 1e-15
optim = dict(
    cls="HybridAdam",
    lr=lr,
    eps=eps,
    weight_decay=0.0,
    adamw_mode=True,
)
warmup_steps = 0
update_warmup_steps = True
grad_clip = 1.0
accumulation_steps = 1
ema_decay = None

# 7. Acceleration & Hardware
dtype = "bf16"
plugin = "zero2"
grad_checkpoint = True
num_workers = 12
prefetch_factor = 2
num_bucket_build_workers = 64

# Configuración de memoria pre-asignada para evitar fragmentación
pin_memory_cache_pre_alloc_numels = [(260 + 20) * 1024 * 1024] * 24 + [
    (34 + 20) * 1024 * 1024
] * 4

# 8. Logs & Checkpoints
seed = 42
outputs = "outputs"
epochs = 1000
log_every = 10
ckpt_every = 100
keep_n_latest = 20
wandb_project = "mmdit"
save_master_weights = True
load_master_weights = True