# ======================================================
# PCURE-AI+ Inference Config: 256px Base Mode
# ======================================================

# 1. Configuración General de Salida
save_dir = "samples"    # Directorio donde se guardarán los .mp4 resultantes
seed = 42               # Semilla global para reproducibilidad
batch_size = 1
dtype = "bf16"          # Formato optimizado para núcleos TPU/GPU modernos

# 2. Configuración de Condicionamiento
cond_type = "t2v"       # Por defecto: Text-to-Video
# Opciones disponibles: t2v, i2v_head, i2v_tail, i2v_loop, v2v_head_half, v2v_tail_half

dataset = dict(type="text")

# 3. Parámetros de Muestreo (Sampling)
# Aquí se define la "estética" y duración del video
sampling_option = dict(
    resolution="256px",
    aspect_ratio="16:9",      # Formato panorámico
    num_frames=129,           # ~5.3 segundos a 24 fps
    num_steps=50,             # Pasos de eliminación de ruido
    shift=True,               # Aplica shift de tiempo para mejor flujo
    temporal_reduction=4,
    is_causal_vae=True,
    
    # Guía de Clasificación (Classifier-Free Guidance)
    guidance=7.5,             # Fuerza del prompt de texto
    guidance_img=3.0,         # Fuerza de la imagen (usado en I2V)
    
    # Técnicas de Oscilación para suavizar parpadeos (Flickering)
    text_osci=True,
    image_osci=True,
    scale_temporal_osci=True,
    
    method="i2v",             # Método de muestreo interno
    seed=None,                # Semilla aleatoria para el ruido latente (z)
)

motion_score = "4"            # Intensidad del movimiento (1-7)
fps_save = 24                 # Frames por segundo del archivo final

# 4. Arquitectura del Modelo (FLUX Engine)
model = dict(
    type="flux",
    from_pretrained="./ckpts/Open_Sora_v2.safetensors",
    guidance_embed=False,
    fused_qkv=False,
    use_liger_rope=True,      # Optimización de atención rotatoria para TPU
    
    # Parámetros del Transformer
    in_channels=64,
    vec_in_dim=768,
    context_in_dim=4096,      # Dimensión para T5-XXL
    hidden_size=3072,
    mlp_ratio=4.0,
    num_heads=24,
    depth=19,
    depth_single_blocks=38,
    axes_dim=[16, 56, 56],
    theta=10_000,
    qkv_bias=True,
    cond_embed=True,
)

# 5. Componentes de Decodificación y Texto
ae = dict(
    type="hunyuan_vae",       # Autoencoder estándar para Stage 1
    from_pretrained="./ckpts/hunyuan_vae.safetensors",
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    latent_channels=16,
    use_spatial_tiling=True,
    use_temporal_tiling=False,
)

t5 = dict(
    type="text_embedder",
    from_pretrained="./ckpts/google/t5-v1_1-xxl",
    max_length=512,
    shardformer=True,         # Permite cargar el modelo XXL en fragmentos
)

clip = dict(
    type="text_embedder",
    from_pretrained="./ckpts/openai/clip-vit-large-patch14",
    max_length=77,
)