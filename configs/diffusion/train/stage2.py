# ======================================================
# PCURE-AI+ Stage 2: Full Resolution Training
# ============================

_base_ = ["image.py"]

# Optimización extrema de Checkpointing
grad_ckpt_settings = (100, 100)

# ============================
# Distributed Training (TPU/GPU Hybrid)
# ============================
plugin = "hybrid"
plugin_config = dict(
    tp_size=1,                         # Tensor Parallelism
    pp_size=1,                         # Pipeline Parallelism
    sp_size=4,                         # Sequence Parallelism (Crítico para video largo)
    sequence_parallelism_mode="ring_attn", 
    enable_sequence_parallelism=True,
    static_graph=True,                 # Acelera la compilación XLA en TPU
    zero_stage=2,                      # ZeRO-2 para optimización de memoria
)

# ============================
# Bucketing Config: HD Focus
# ============================
bucket_config = {
    "_delete_": True,
    "256px": {
        1: (1.0, 130),
        5: (1.0, 14), 33: (1.0, 14), 65: (1.0, 10), 97: (1.0, 7), 129: (1.0, 6),
    },
    "768px": {
        1: (1.0, 38),   # Calidad fotográfica base
        5: (1.0, 6),    13: (1.0, 6),  21: (1.0, 6),  29: (1.0, 6),  33: (1.0, 6),
        37: (1.0, 4),   45: (1.0, 4),  53: (1.0, 4),  61: (1.0, 4),  65: (1.0, 4),
        69: (1.0, 3),   77: (1.0, 3),  85: (1.0, 3),  93: (1.0, 3),  97: (1.0, 3),
        101: (1.0, 2),  109: (1.0, 2), 113: (1.0, 2), 121: (1.0, 2), 129: (1.0, 2),
    },
}

# ============================
# Training Settings
# ============================
model = dict(grad_ckpt_settings=grad_ckpt_settings)
lr = 5e-5
optim = dict(lr=lr)

ckpt_every = 200     # Guardado frecuente: los fallos en Stage 2 son costosos
keep_n_latest = 20