# ======================================================
# PCURE-AI+ Inference Plugin: Pure Tensor Parallelism (TP)
# ======================================================

# 1. Configuración del Modelo de Difusión (Transformer)
# Se utiliza para fragmentar las capas del modelo entre los 8 núcleos de la TPU.
plugin = "hybrid"
plugin_config = dict(
    tp_size=8,                         # Divide los pesos del modelo en 8 fragmentos
    pp_size=1,                         # Sin Pipeline Parallelism (mantenemos latencia baja)
    sp_size=1,                         # Desactivamos Sequence Parallelism
    zero_stage=2,                      # Optimización de estados de memoria ZeRO-2
    enable_sequence_parallelism=False,
    static_graph=True,                 # Crítico para la aceleración XLA en TPU
    overlap_allgather=False,           # Prioriza la precisión de sincronización
)

# 2. Configuración del Autoencoder (VAE / DC-AE)
# Vital para decodificar las texturas del Parche B en alta resolución.
plugin_ae = "hybrid"
plugin_config_ae = dict(
    tp_size=8,                         # Los 8 núcleos procesan el tensor espacial en paralelo
    pp_size=1, 
    sp_size=1, 
    zero_stage=2, 
    overlap_allgather=False,
)