# ======================================================
# PCURE-AI+ Stage 1: Image-to-Video (I2V) Training
# ======================================================

# Hereda toda la configuración de Stage 1 (STDiT architecture, datasets, etc.)
_base_ = ["stage1.py"]

# 1. Activación de Embeddings de Condicionamiento
model = dict(cond_embed=True)

# 2. Configuración de Pesos para el Entrenamiento I2V
# Aquí se define qué tan fuerte debe "prestar atención" el modelo a la imagen de referencia
condition_config = dict(
    t2v=1,        # Peso base para Text-to-Video
    i2v_head=5,   # PRIORIDAD ALTA: Usar la imagen como el primer frame del video
    i2v_loop=1,   # Conexión interna de la imagen a través de la secuencia
    i2v_tail=1,   # Usar la imagen como el último frame (útil para loops perfectos)
)

# 3. Parámetros de Optimización conservadores para estabilidad
lr = 1e-5
optim = dict(lr=lr)