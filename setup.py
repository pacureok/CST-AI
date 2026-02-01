from setuptools import setup, Extension, find_packages

# 1. Definición del Kernel Nativo (TENTPU Core)
# Este binario es el que intercepta las llamadas de NVIDIA para redirigirlas a la TPU.
cst_extension = Extension(
    'cst_tpu_core',
    sources=['src/cst_core.cpp'],
    extra_compile_args=['-O3', '-fPIC', '-std=c++11'],
    # No incluimos '-shared' aquí porque setuptools lo añade automáticamente
)

# 2. Configuración de Instalación
setup(
    # El nombre y versión se heredan del pyproject.toml, 
    # pero los mantenemos aquí para compatibilidad con pip install -e
    name="CST-AI",
    version="1.2.0",
    
    # Buscamos los paquetes en MODELS y src como se ve en tu imagen
    packages=find_packages(where="."),
    
    # Vinculamos la extensión de C++
    ext_modules=[cst_extension],
    
    # Aseguramos que los archivos no-python se incluyan
    include_package_data=True,
    zip_safe=False,
)