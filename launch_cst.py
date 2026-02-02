import colossalai
from colossalai.shardformer import ShardConfig, ShardFormer
from opensora.models.mmdit.model import MMDiTModel, MMDiTConfig
from opensora.models.mmdit.policy import HunyuanVaePolicy

def launch_training():
    # 1. Configuración de fragmentación (Tensor Parallelism)
    shard_config = ShardConfig(
        tensor_parallel_size=4, # Dividimos el modelo en 4 TPUs/GPUs
        enable_gradient_checkpointing=True,
        preprocess_checkpoint=True
    )

    # 2. Inicializar Modelo CST
    config = MMDiTConfig(hidden_size=1152, depth=19, axes_dim=[1, 64, 64])
    model = MMDiTModel(config)

    # 3. Aplicar Política de Ultra-Realismo (Shardformer)
    shardformer = ShardFormer(shard_config=shard_config)
    model, _ = shardformer.optimize(model, HunyuanVaePolicy())

    print("✅ Sistema CST inicializado con éxito para Long-Context.")
    return model

if __name__ == "__main__":
    launch_training()