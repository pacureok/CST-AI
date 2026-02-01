"""isort:skip_file"""
import sys
from types import ModuleType

# --- INYECCIÓN DE OPERACIONES CORE (FIX DEFINITIVO) ---
def constexpr(x): return x
class constexpr_type:
    def __init__(self, x): self.x = x

# Mocks para manipulación de frames de video
def view(x, *args, **kwargs): return x
def reshape(x, *args, **kwargs): return x
def cast(x, *args, **kwargs): return x

# --- ESTRUCTURA DE IMPORTACIÓN ---
from . import math
from . import extra
from . import core

# Asegurar que constexpr sea visible a nivel triton.language
this = sys.modules[__name__]
setattr(this, 'constexpr', constexpr)
setattr(this, 'view', view)
setattr(this, 'reshape', reshape)

# Importaciones estándar de Triton (se mantienen por compatibilidad)
try:
    from .standard import *
    from .core import *
    from .math import *
    from .random import *
except ImportError:
    # Si los archivos hijos no existen, definimos tipos básicos para TPU
    float32 = "float32"
    bfloat16 = "bfloat16"
    int32 = "int32"

__all__ = [
    "constexpr", "view", "reshape", "cast", "float32", "bfloat16",
    "math", "extra", "core", "cdiv", "next_power_of_2"
]

# Fix para el error de 'extra' y 'math'
from . import math as math_mod
from . import extra as extra_mod
math = math_mod
extra = extra_mod
