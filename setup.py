# /home/oficialpacureok/Open-Sora/setup.py
from setuptools import setup, find_packages

setup(
    name="cst_ai_tpu",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "librosa",
        "soundfile",
        "numpy",
        "tqdm",
    ],
    # Eliminamos las referencias a NVIDIA CUDA para evitar conflictos en TPU
    classifiers=[
        "Programming Language :: Python :: 3",
        "Environment :: Cloud Shell",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)