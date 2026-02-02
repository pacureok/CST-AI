# ======================================================
# PCURE-AI+ Stage 1: Demo & Base Training Config
# ======================================================

# Hereda la arquitectura del modelo, optimizadores y datasets de stage1.py
_base_ = ["stage1.py"]

# Configuración de Buckets: Define grupos por resolución y número de frames
bucket_config = {
    "_delete_": True,  # Ignora la configuración de buckets del archivo base
    "256px": {
        # frames: (probabilidad, batch_size_multiplicador)
        1: (1.0, 1),    # Entrenamiento con Imágenes (Realismo estático)
        33: (1.0, 1),   # Videos cortos (Movimiento básico)
        97: (1.0, 1),   # Videos estándar (~4 segundos a 24fps)
        129: (1.0, 1),  # Videos largos (Coherencia temporal extendida)
    },
}