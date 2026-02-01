import sys
import os

# Definir CAITPU (ColossalAI for TPU)
caitpu_path = os.path.join(os.path.dirname(__file__), "ColossalAI")
sys.path.insert(0, caitpu_path)
try:
    import colossalai as caitpu
    sys.modules['CAITPU'] = caitpu
    print("[💎] CAITPU: ColossalAI for TPU cargado.")
except:
    pass

# Definir TENTPU (Triton for TPU)
tentpu_path = os.path.join(os.path.dirname(__file__), "triton")
sys.path.insert(0, tentpu_path)
try:
    import triton as tentpu
    sys.modules['TENTPU'] = tentpu
except:
    pass
