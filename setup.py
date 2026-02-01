from setuptools import setup, Extension, find_packages
import os

# 1. Definición de la extensión C++ (CST-CORE)
# Este binario gestionará la memoria HBM de la TPU directamente.
cst_extension = Extension(
    'cst_tpu_core',
    sources=['src/cst_core.cpp'],
    extra_compile_args=['-O3', '-fPIC', '-shared', '-std=c++11'],
)

# 2. Configuración Maestra del Paquete
setup(
    name="CST-AI",
    version="1.2.0",
    author="PCURE-AI+",
    description="Motor de IA Híbrido para TPU v5e (TENTPU & CAITPU)",
    packages=find_packages(),
    ext_modules=[cst_extension],
    python_requires='>=3.10',
    install_requires=[
        'torch',
        'numpy',
    ],
    # Incluimos archivos que no son de python (como el binario .so)
    include_package_data=True,
    zip_safe=False,
)