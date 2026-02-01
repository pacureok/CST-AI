from . import core
from . import extra
import sys

# Definimos los mocks para que sean accesibles como atributos
class MathMock:
    pass

math = MathMock()

# Inyectamos los módulos en el registro global de Python
sys.modules['triton.language.core'] = core
sys.modules['triton.language.extra'] = extra
