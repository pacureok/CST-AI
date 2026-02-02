# ======================================================
# PCURE-AI+ Training Config: DC-AE Adversarial (GAN Mode)
# ======================================================
_base_ = ["video_dc_ae.py"]

# 1. Configuración del Discriminador 3D (El "Crítico")
discriminator = dict(
    type="N_Layer_discriminator_3D",
    from_pretrained=None,
    input_nc=3,         # RGB
    n_layers=5,         # Profundidad para capturar micro-detalles y macro-movimientos
    conv_cls="conv3d"   # Esencial para evaluar la fluidez temporal entre frames
)

# 2. Programación del Aprendizaje
disc_lr_scheduler = dict(warmup_steps=0)

# 3. Balance de Pérdidas (Loss Functions)
gen_loss_config = dict(
    gen_start=0,        # El generador empieza a intentar engañar al disc. desde el paso 0
    disc_weight=0.05,   # Peso de la pérdida adversarial (ajusta para evitar colapso de color)
)

disc_loss_config = dict(
    disc_start=0,
    disc_loss_type="hinge", # Hinge loss es más estable que la clásica bce para video
)

# 4. Optimizador Híbrido (Optimizado para hardware de alto rendimiento)
optim_discriminator = dict(
    cls="HybridAdam",
    lr=1e-4,
    eps=1e-8,
    weight_decay=0.0,
    adamw_mode=True,
    betas=(0.9, 0.98),
)

# 5. Optimización de Memoria VRAM
grad_checkpoint = True
model = dict(
    # Desactiva el grad_ckpt en el discriminador si es necesario para estabilidad
    disc_off_grad_ckpt = True, 
)