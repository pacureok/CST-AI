from . import autotuner
from .autotuner import OutOfResources, autotune, Heuristics

def triton_cache_dir():
    return '/tmp/triton'

class JITFunction:
    def __init__(self, *args, **kwargs): pass
    def __call__(self, *args, **kwargs): return None

def reinterpret_view(*args, **kwargs):
    return None

__all__ = ["autotuner", "OutOfResources", "autotune", "Heuristics", "triton_cache_dir", "JITFunction", "reinterpret_view"]
