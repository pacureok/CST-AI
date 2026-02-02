import torch
class CompiledKernel:
    def __init__(self, *args, **kwargs): pass
    def __getitem__(self, *args): return self
    def __call__(self, *args, **kwargs): return None
def compile(*args, **kwargs): return CompiledKernel()
def jit(*args, **kwargs): return lambda f: f
