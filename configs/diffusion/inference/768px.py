# ======================================================
# PCURE-AI+ Inference Config: 768px High-Definition
# ======================================================

_base_ = [
    "256px.py",       # Hereda la arquitectura FLUX y los text-embedders
    "plugins/sp.py",  # ACTIVA: Sequence Parallelism (sp_size=8 + Ring Attention)
]

# Sobrescribimos la resolución para el motor de muestreo
sampling_option = dict(
    resolution="768px",
)