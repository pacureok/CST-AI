# ======================================================
# PCURE-AI+ Stage 1: Motion Training (Temporal Learning)
# ======================================================

# Hereda la base de FLUX y los codificadores de texto
_base_ = ["image.py"]

# Optimización de carga: False permite un acceso más rápido si tienes suficiente RAM
dataset = dict(memory_efficient=False)

# Configuración de Checkpointing de Gradientes
grad_ckpt_settings = (8, 100)

# ============================
# Bucketing Progresivo
# ============================
bucket_config = {
    "_delete_": True,
    
    # Entrenamiento de Video a 256px: Progresión de 1 a 129 frames
    "256px": {
        1: (1.0, 45),    # Ancla de imagen (Batch alto)
        5: (1.0, 12),    # Comienzo del movimiento
        9: (1.0, 12),
        13: (1.0, 12),
        17: (1.0, 12),
        21: (1.0, 12),
        25: (1.0, 12),
        29: (1.0, 12),
        33: (1.0, 12),   # Clips de ~1.5 seg
        37: (1.0, 6),
        41: (1.0, 6),
        45: (1.0, 6),
        49: (1.0, 6),
        53: (1.0, 6),
        57: (1.0, 6),
        61: (1.0, 6),
        65: (1.0, 6),
        69: (1.0, 4),    # ~3 seg
        73: (1.0, 4),
        77: (1.0, 4),
        81: (1.0, 4),
        85: (1.0, 4),
        89: (1.0, 4),
        93: (1.0, 4),
        97: (1.0, 4),    # ~4 seg
        101: (1.0, 3),
        105: (1.0, 3),
        109: (1.0, 3),
        113: (1.0, 3),
        117: (1.0, 3),
        121: (1.0, 3),
        125: (1.0, 3),
        129: (1.0, 3),   # Máxima duración Stage 1 (~5.3 seg a 24fps)
    },
    
    # Mantenimiento de Calidad Fotográfica (Solo imágenes)
    "768px": {
        1: (0.5, 13),
    },
    "1024px": {
        1: (0.5, 7),
    },
}

# ============================
# Model & Optimization
# ============================
model = dict(grad_ckpt_settings=grad_ckpt_settings)

lr = 5e-5            # Aumentamos ligeramente el LR para aprender dinámicas de movimiento
optim = dict(lr=lr)

ckpt_every = 2000    # Intervalo de guardado más largo para evitar cuellos de botella de I/O
keep_n_latest = 20