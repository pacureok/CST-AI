import os
import sys

# Get the absolute path of the MODELS directory
# Esto asegura que el motor encuentre las rutas sin importar dónde se ejecute
MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Definir la ruta específica del código fuente de ColossalAI
# Tu estructura es MODELS/ColossalAI/colossalai
COLOSSAL_ROOT = os.path.join(MODELS_DIR, "ColossalAI")

# 2. Inyectar las rutas en el sistema con prioridad máxima
# Insertamos al inicio (índice 0) para que ignore versiones de pip
if COLOSSAL_ROOT not in sys.path:
    sys.path.insert(0, COLOSSAL_ROOT)
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

# 3. Registro del Motor Propietario Pcure-AI+
__version__ = "1.2.0"
__engine__ = "CST-TPU (DPR Enabled)"

def init_cst_env():
    """
    Inicializa el entorno de hardware para el motor CST.
    Verifica que las dependencias parcheadas para TPU estén accesibles.
    """
    try:
        import colossalai
        print(f"[💎] CST-INIT: ColossalAI Engine (TPU Patch) vinculado con éxito.")
    except ImportError:
        print("[⚠️] CST-WARNING: ColossalAI no detectado en MODELS/ColossalAI. Verifique la estructura de carpetas.")

    # Mocks de seguridad para evitar fallos por hardware de NVIDIA
    import torch
    if not hasattr(torch.cuda, 'is_available'):
        torch.cuda.is_available = lambda: False

# Ejecutar inicialización al importar
init_cst_env()