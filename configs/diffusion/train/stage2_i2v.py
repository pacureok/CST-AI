# ======================================================
# PCURE-AI+ Stage 2: High-Resolution I2V (TPU Optimized)
# ======================================================

_base_ = ["stage2.py"]

# 1. Model Configuration
model = dict(cond_embed=True)

# Optimización de Memoria para TPU: Buffer de gradientes de 25GB
# Crucial para manejar el throughput masivo de los núcleos MXU
grad_ckpt_buffer_size = 25 * 1024**3

# 2. I2V Conditioning Weights
condition_config = dict(
    t2v=1,
    i2v_head=5,  # Mantenemos prioridad alta en el primer frame para evitar flickering
    i2v_loop=1,
    i2v_tail=1,
)

is_causal_vae = True

# 3. Bucket Config (Estrategia de Batching para TPU)
# Los batch sizes aquí son significativamente más altos para aprovechar la arquitectura HBM de la TPU
bucket_config = {
    "_delete_": True,
    "256px": {
        1: (1.0, 195),   # Batch masivo para estabilización de gradientes
        5: (1.0, 80),
        33: (1.0, 80),
        65: (1.0, 40),
        97: (1.0, 28),
        129: (1.0, 23),  # Secuencias largas a baja resolución
    },
    "768px": {
        1: (0.5, 38),    # Referencia de alta calidad (Parche B)
        # Progresión temporal en alta resolución
        5: (0.5, 10), 13: (0.5, 10), 21: (0.5, 10), 29: (0.5, 10), 33: (0.5, 10),
        37: (0.5, 5),  45: (0.5, 5),  53: (0.5, 5),  61: (0.5, 5),  65: (0.5, 5),
        69: (0.5, 3),  77: (0.5, 3),  85: (0.5, 3),  93: (0.5, 3),  97: (0.5, 3),
        101: (0.5, 2), 109: (0.5, 2), 117: (0.5, 2), 125: (0.5, 2), 129: (0.5, 2),
    },
}