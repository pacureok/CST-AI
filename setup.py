from setuptools import setup, Extension, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "cst_tpu_core",
        ["src/cst_core.cpp"],
        extra_compile_args=['-O3', '-std=c++11'],
    ),
]

setup(
    name="CST-AI",
    version="1.2.1",
    author="Pacure-AI+",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    install_requires=[
        "torch",
        "torch-xla",
        "diffusers",
        "transformers",
        "accelerate",
        "huggingface_hub",
        "einops",
    ],
    zip_safe=False,
)
