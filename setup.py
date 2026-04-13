import os
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

__version__ = "2.0.1"

# Identifica la directory radice del progetto
project_root = os.path.dirname(os.path.abspath(__file__))


ext_modules = [
    Pybind11Extension(
        "machine_learning_module",
        [os.path.join(project_root, "pybinding/machine_learning_module.cpp")], # PERCORSO ASSOLUTO
        include_dirs=[
            os.path.join(project_root, "include"),
            "/usr/include/eigen3", 
        ],
        extra_objects=[os.path.join(project_root, "build/lib/libml_library.a")],
        extra_compile_args=["-std=c++17", "-O3", "-fPIC"],
    ),
]


setup(
    name="ml_library",
    version=__version__,
    author="Maurizio Penna",
    author_email="mauriziopenna@gmail.com",
    description="Machine Learning Library in C++ with Python bindings",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.21.0",
    ],
)