import os
from setuptools import setup, find_packages, Extension

# Configuración del Motor C++ Nativo
cst_extension = Extension(
    'cst_tpu_core',
    sources=['src/cst_core.cpp'],
    extra_compile_args=['-O3', '-fPIC', '-shared'],
)

setup(
    name="CST-AI",
    version="1.2.0",
    description="Motor Sora-Close optimizado para Google TPU (TENTPU & CAITPU)",
    author="PCURE-AI+",
    packages=find_packages(),
    ext_modules=[cst_extension],
    install_requires=[
        'torch',
        'numpy',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
    ],
)