def jit(*args, **kwargs):
    # Decorador que simplemente devuelve la función original
    return lambda x: x

def autotune(*args, **kwargs):
    return lambda x: x

class Config:
    def __init__(self, *args, **kwargs): pass