# ======================================================
# PCURE-AI+ T2I2V: 768px Cinema Mode (High-Fidelity)
# ======================================================

_base_ = [
    "768px.py",          # Activa 768px, SP (sp_size=8) y Ring Attention
    "plugins/t2i2v.py",  # Activa el generador de imagen base FLUX.1-dev
]