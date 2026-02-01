"""isort:skip_file"""
__version__ = '3.6.0'
import os
import sys

try:
    from .runtime import autotune, Config, heuristics, JITFunction, KernelInterface, reinterpret, TensorWrapper, OutOfResources, InterpreterError, MockTensor
    from .runtime.jit import constexpr_function, jit
    from .compiler import compile, CompilationError
except ImportError:
    class Config:
        def __init__(self, kwargs=None, num_warps=4, num_stages=2):
            self.kwargs = kwargs if kwargs else {}; self.num_warps = num_warps; self.num_stages = num_stages
    def jit(fn): return fn
    def constexpr_function(fn): return fn
    class CompilationError(Exception): pass
    autotune = heuristics = JITFunction = KernelInterface = object

try:
    from . import language
    must_use_result = language.core.must_use_result
except ImportError:
    from types import ModuleType
    language = ModuleType('language'); language.core = ModuleType('core')
    language.core.must_use_result = lambda x: x; must_use_result = language.core.must_use_result

@constexpr_function
def cdiv(x: int, y: int): return (x + y - 1) // y

@constexpr_function
def next_power_of_2(n: int):
    if n <= 0: return 1
    n -= 1; n |= n >> 1; n |= n >> 2; n |= n >> 4; n |= n >> 8; n |= n >> 16; n |= n >> 32; n += 1
    return n
