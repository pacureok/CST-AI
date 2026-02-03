from setuptools import setup, Extension, find_packages
import os

# --- LÓGICA DE INSTALACIÓN PACUR ---
# Verificamos si estamos en entorno TPU para ajustar la compilación
extra_args = ['-O3', '-fPIC', '-std=c++11']
if os.environ.get('TPU_NAME'):
    extra_args += ['-DTPU_ACCELERATION'] # Flag para tu código C++

setup(
    name="CST-AI",
    version="1.2.1", # Subimos versión por la integración Open Sora
    packages=find_packages(),
    
    # Módulo de alto rendimiento original
    ext_modules=[
        Extension(
            'cst_tpu_core', 
            sources=['src/cst_core.cpp'], 
            extra_compile_args=extra_args
        )
    ],
    
    # --- NUEVAS DEPENDENCIAS DE OPEN SORA & TPU ---
    install_requires=[
        "torch",
        "torch-xla",
        "diffusers>=0.24.0",
        "transformers",
        "accelerate",
        "huggingface_hub",
        "timm",        # Necesario para DiT (Open Sora)
        "omegaconf",   # Para los archivos de configuración v2
        "einops",      # Procesamiento de tensores avanzado
    ],
    
    # Para que puedas ejecutar 'pacur-run' en la consola
    entry_points={
        'console_scripts': [
            'pacur-setup=scripts.setup_assets:main',
        ],
    },
    
    include_package_data=True,
    zip_safe=False,
)
