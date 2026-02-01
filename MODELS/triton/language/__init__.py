import sys

# Definimos constexpr como una clase que simplemente devuelve lo que recibe
class constexpr:
    def __init__(self, value):
        self.value = value
    def __repr__(self):
        return str(self.value)

# Añadimos constantes matemáticas básicas que Inductor suele buscar
def log2(x): return x # Mock simple
def exp(x): return x

# Aseguramos que 'core' y 'extra' sigan vinculados
from . import core
from . import extra

# Inyectamos los submódulos en el registro de Python para evitar AttributeError
sys.modules['triton.language.core'] = core
sys.modules['triton.language.extra'] = extra
