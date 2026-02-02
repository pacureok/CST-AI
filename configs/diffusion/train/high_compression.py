# ======================================================
# PCURE-AI+ Training Config: High Compression Mode
# ======================================================

# Hereda parámetros base de imagen (T5, optimizadores básicos, etc.)
_base_ = ["image.py"]

# ============================
# 1. Bucketing Progresivo (768px)
# ============================
# Optimizamos el batch size multiplicador según la longitud del video
bucket_config = {
    "_delete_": True,
    "768px": {
        1: (1.0, 20),    # Imágenes: Entrenamiento de textura pura
        16: (1.0, 8),    # Clips ultra-cortos
        20: (1.0, 8),
        24: (1.0, 8),
        28: (1.0, 8),
        32: (1.0, 8),
        36: (1.0, 4),
        40: (1.0, 4),
        44: (1.0, 4),
        48: (1.0, 4),
        52: (1.0, 4),
        56: (1.0, 4),
        60: (1.0, 4),
        64: (1.0, 4),    # ~2.6 segundos
        68: (1.0, 3),
        72: (1.0, 3),
        76: (1.0, 3),
        80: (1.0, 3),
        84: (1.0, 3),
        88: (1.0, 3),
        92: (1.0, 3),
        96: (1.0, 3),    # 4 segundos
        100: (1.0, 2),
        104: (1.0, 2),
        108: (1.0, 2),
        112: (1.0, 2),
        116: (1.0, 2),
        120: (1.0, 2),
        124: (1.0, 2),
        128: (1.0, 2),   # ~5.3 segundos reales (o hasta 30s según compresión temporal)
    },
}

# ============================
# 2. Model & Condition Settings
# ============================
# Habilitamos tanto Text-to-Video como Image-to-Video
condition_config = dict(
    t2v=1,
    i2v_head=7,
)

# Ajustes de arquitectura STDiT (Transformer de Video)
grad_ckpt_settings = (100, 100)
patch_size = 1 # Resolución máxima de atención por parche latente

model = dict(
    from_pretrained=None,
    grad_ckpt_settings=grad_ckpt_settings,
    in_channels=128,    # Emparejado con los 128 canales del DC-AE
    cond_embed=True,    # Embeddings de condicionamiento activados
    patch_size=patch_size,
)

# ============================
# 3. AutoEncoder (DC-AE) Integration
# ============================
ae = dict(
    _delete_=True,
    type="dc_ae",
    model_name="dc-ae-f32t4c128",
    from_pretrained="./ckpts/F32T4C128_AE.safetensors",
    from_scratch=True,
    scaling_factor=0.493,      # Normalización estadística de los latentes
    use_spatial_tiling=True,
    use_temporal_tiling=True,
    spatial_tile_size=256,
    temporal_tile_size=32,
    tile_overlap_factor=0.25,
)

# Atributos específicos de la compresión de video
is_causal_vae = False
ae_spatial_compression = 32 # El DC-AE reduce la resolución espacial en 32x

# ============================
# 4. Optimizer & Schedule
# ============================
ckpt_every = 250
lr = 3e-5
optim = dict(lr=lr)