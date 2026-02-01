# CST-Mock: Identidad de versión para engañar a PyTorch Inductor
__version__ = '3.0.0' 

from . import language
from . import compiler
from . import runtime

class Config:
    def __init__(self, *args, **kwargs): pass

def jit(x): return x

HAS_WARP_SPEC = False
