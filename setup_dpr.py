import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Buscamos el archivo C++ automáticamente
core_path = "opensora/models/mmdit/cst_dpr_core.cpp"

if os.path.exists(core_path):
    setup(
        name='cst_dpr_core',
        ext_modules=[
            CUDAExtension('cst_dpr_core', [core_path])
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )
else:
    print(f"[!] Error: No se encontró el core en {core_path}")