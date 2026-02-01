from . import extra
import sys

# Simulamos submódulos que PyTorch Inductor busca desesperadamente
class MathMock:
    pass

math = MathMock()

# Agregamos 'extra' como atributo del módulo para evitar el AttributeError
sys.modules['triton.language.extra'] = extra
