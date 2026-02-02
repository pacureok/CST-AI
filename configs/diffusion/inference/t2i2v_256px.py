# ======================================================
# PCURE-AI+ T2I2V: 256px Inference Mode
# ======================================================

_base_ = [
    "256px.py",          # Define parámetros de muestreo y arquitectura de video
    "plugins/t2i2v.py",  # Activa el modelo FLUX para la generación de la imagen base
]