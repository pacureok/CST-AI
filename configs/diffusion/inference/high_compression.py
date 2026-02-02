# ======================================================
# PCURE-AI+ High Compression Inference: DC-AE Engine
# ======================================================

_base_ = ["t2i2v_768px.py"]

# Desactivamos el paralelismo para inferencia directa en un solo dispositivo/chip
plugin = None
plugin_config = None
plugin_ae = None
plugin_config_ae = None

# ============================
# Model Settings (DC-AE Compatible)
# ============================
patch_size = 1
model = dict(
    from_pretrained="./ckpts/Open_Sora_v2_Video_DC_AE.safetensors",
    in_channels=128,    # Emparejado con los canales del DC-AE
    cond_embed=True,
    patch_size=patch_size,
)

# ============================
# AutoEncoder Settings (F32T4C128)
# ============================
ae = dict(
    _delete_=True,
    type="dc_ae",
    from_scratch=True,
    model_name="dc-ae-f32t4c128",
    from_pretrained="./ckpts/F32T4C128_AE.safetensors",
    use_spatial_tiling=True,
    use_temporal_tiling=True,
    spatial_tile_size=256,
    temporal_tile_size=32,
    tile_overlap_factor=0.25,
)

# Factor de compresión: 32x espacial, 4x temporal
ae_spatial_compression = 32

# ============================
# Sampling Option
# ============================
sampling_option = dict(
    num_frames=128,  # Generación de clip estándar (~5 segundos)
)