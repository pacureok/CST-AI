# ======================================================
# PCURE-AI+ Inference Plugin: Distributed Optimization
# ======================================================

# Configuración para el Transformer (Difusión)
plugin = "hybrid"
plugin_config = dict(
    tp_size=1, 
    pp_size=1, 
    sp_size=8,                         # Foco en Paralelismo de Secuencia (Temporal)
    sequence_parallelism_mode="ring_attn", 
    enable_sequence_parallelism=True,
    static_graph=True,                 # Crítico para evitar re-compilaciones en TPU
    zero_stage=2, 
    overlap_allgather=False,
)

# Configuración para el Autoencoder (VAE/DC-AE)
plugin_ae = "hybrid"
plugin_config_ae = dict(
    tp_size=8,                         # Foco en Paralelismo de Tensores (Espacial)
    pp_size=1, 
    sp_size=1, 
    zero_stage=2, 
    overlap_allgather=False,
)