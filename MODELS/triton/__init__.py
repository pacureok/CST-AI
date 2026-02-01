"""isort:skip_file"""
__version__ = '3.6.0'
import os
import sys

# Clases base para compatibilidad con TPU
class Config:
    def __init__(self, kwargs=None, num_warps=4, num_stages=2):
        self.kwargs = kwargs if kwargs else {}
        self.num_warps = num_warps
        self.num_stages = num_stages

def jit(fn=None, **kwargs):
    if fn is None: return lambda f: f
    return fn

def constexpr_function(fn): return fn

class CompilationError(Exception): pass
class OutOfResources(Exception): pass

# Importación segura de submódulos
try:
    from . import language
    from . import runtime
    from . import compiler
except ImportError:
    pass

# Funciones de utilidad para el Kernel de Video
@constexpr_function
def cdiv(x: int, y: int): return (x + y - 1) // y

@constexpr_function
def next_power_of_2(n: int):
    if n <= 0: return 1
    n -= 1; n |= n >> 1; n |= n >> 2; n |= n >> 4; n |= n >> 8; n |= n >> 16; n |= n >> 32; n += 1
    return n

# Atajos críticos
autotune = heuristics = JITFunction = KernelInterface = object
