import sys
from types import ModuleType

# --- PCURE-AI+ BYPASS START ---
def mock_triton():
    if 'triton' not in sys.modules:
        triton = ModuleType('triton')
        triton.Config = type('Config', (), {'__init__': lambda self, **k: None})
        triton.jit = lambda fn=None, **k: (lambda x: x) if fn is None else fn
        
        # Crear sub-módulos necesarios
        for sub in ['language', 'language.core', 'compiler', 'runtime']:
            m = ModuleType(f'triton.{sub}')
            if sub == 'language.core': m.view = lambda x, *a: x
            if sub == 'runtime': m.triton_cache_dir = lambda: '/tmp'
            sys.modules[f'triton.{sub}'] = m
            setattr(triton, sub.split('.')[-1], m)
            
        sys.modules['triton'] = triton
        print("[💎] PCURE-AI+: Triton Bypass inyectado en HunyuanVAE")

mock_triton()
# --- PCURE-AI+ BYPASS END ---