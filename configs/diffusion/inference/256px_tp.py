# ======================================================
# PCURE-AI+ Inference: 256px Fast-Track (TPU Optimized)
# ======================================================

_base_ = [
    "256px.py",       # Configura dimensiones, arquitectura STDiT y pesos del modelo
    "plugins/tp.py",  # Aplica tp_size=8 para fragmentar el modelo en la TPU
]