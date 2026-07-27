import os
import sys
from pathlib import Path
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Directory radice del progetto
BASE_DIR = Path(__file__).parent.resolve()

def get_relative_sources(source_dir):
    """
    Raccoglie tutti i sorgenti .cpp e garantisce 
    che siano espressi con percorsi RELATIVI a BASE_DIR.
    """
    sources = []
    target_path = BASE_DIR / source_dir
    for p in target_path.rglob("*.cpp"):
        # Convertiamo il percorso assoluto in relativo alla root del progetto
        rel_p = p.relative_to(BASE_DIR)
        sources.append(str(rel_p))
    return sources

# Recupera la lista dei file di binding in modo relativo
binding_sources = get_relative_sources("pybinding/bindings")

# Configurazione Include / Include Eigen / Libreria C++ compilata
include_dirs = [
    str(BASE_DIR / "include"),
    str(BASE_DIR / "pybinding"),
    str(BASE_DIR / "pybinding" / "bindings"),
    "/usr/include/eigen3",
]

library_dirs = [
    str(BASE_DIR / "build" / "lib"),
]

extra_objects = [
    str(BASE_DIR / "build" / "lib" / "libml_library.a"),
]

ext_modules = [
    Pybind11Extension(
        "ml_core",
        sources=binding_sources,  # <--- Ora sono tutti percorsi relativi
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        extra_objects=extra_objects,
        cxx_std=17,
        extra_compile_args=["-O3", "-fPIC", "-Wall", "-Wextra"],
        extra_link_args=["-Wl,-rpath,$ORIGIN/../build/lib"],
    ),
]

setup(
    name="ml_core",
    version="2.0.1",
    author="Maurizio Penna",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    project_urls={
        "Source": "https://github.com/mauriziopenna/ai-devel",
        "Documentation": "https://github.com/mauriziopenna/ai-devel/wiki",
    },
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)