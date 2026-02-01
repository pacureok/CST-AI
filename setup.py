from setuptools import setup, find_packages, Extension
import os

tpu_extension = Extension(
    name="cst_tpu_core",
    sources=["src/cst_core.cpp"],
    libraries=[],
    extra_compile_args=["-O3", "-std=c++17"],
)

setup(
    name="CST-AI",
    version="1.2.0",
    packages=find_packages(),
    ext_modules=[tpu_extension],
    install_requires=["torch", "torch_xla", "mmengine", "einops"],
    description="AI Engine for Google TPU (Python/C++)",
)
