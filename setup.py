from setuptools import setup, Extension, find_packages
setup(
    name="CST-AI",
    version="1.2.0",
    packages=find_packages(),
    ext_modules=[Extension('cst_tpu_core', sources=['src/cst_core.cpp'], extra_compile_args=['-O3', '-fPIC', '-std=c++11'])],
    include_package_data=True,
    zip_safe=False,
)
