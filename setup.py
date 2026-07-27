#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

__version__ = "3.0.0"

# Identifica la directory radice del progetto
project_root = Path(__file__).parent.absolute()

# ============================================================================
# FUNZIONE PER TROVARE EIGEN3
# ============================================================================
def find_eigen3():
    """Trova il percorso di Eigen3 in diverse possibili posizioni."""
    possible_paths = [
        "/usr/include/eigen3",
        "/usr/local/include/eigen3",
        "/opt/homebrew/include/eigen3",  # macOS Homebrew
        str(project_root / "include/eigen3"),  # Locale
    ]
    
    # Prova con pkg-config
    try:
        result = subprocess.run(
            ["pkg-config", "--cflags", "eigen3"],
            capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            # Estrai il percorso -I
            for part in result.stdout.split():
                if part.startswith("-I"):
                    path = part[2:]
                    if os.path.exists(path):
                        return path
    except FileNotFoundError:
        pass
    
    # Cerca nelle posizioni comuni
    for path in possible_paths:
        eigen_path = Path(path)
        if eigen_path.exists() and (eigen_path / "Eigen/Core").exists():
            return str(eigen_path)
    
    # Se non trovato, solleva un errore
    raise RuntimeError(
        "Eigen3 not found. Please install eigen3:\n"
        "  Ubuntu/Debian: sudo apt install libeigen3-dev\n"
        "  macOS: brew install eigen\n"
        "  Fedora: sudo dnf install eigen3-devel"
    )

# ============================================================================
# FUNZIONE PER TROVARE LA LIBRERIA ML
# ============================================================================
def find_ml_library():
    """Trova la libreria ml_library compilata (statica o dinamica)."""
    build_dir = project_root / "build"
    
    # Cerca libreria statica (.a)
    static_lib = build_dir / "lib" / "libml_library.a"
    if static_lib.exists():
        return str(static_lib)
    
    # Cerca libreria dinamica (.so)
    dynamic_lib = build_dir / "lib" / "libml_library.so"
    if dynamic_lib.exists():
        return str(dynamic_lib)
    
    # Se non trovata, avvisa ma non blocca (potrebbe essere linkata dinamicamente)
    print("⚠️  Warning: libml_library not found in build/lib/")
    print("   Run './build.sh' first to compile the C++ library")
    return None

# ============================================================================
# COMPILA LA LIBRERIA C++ SE NECESSARIA (opzionale)
# ============================================================================
def ensure_cpp_library():
    """Assicura che la libreria C++ sia compilata prima del binding."""
    lib_file = project_root / "build" / "lib" / "libml_library.a"
    if lib_file.exists():
        return True
    
    print("🔨 C++ library not found. Compiling...")
    try:
        # Esegui build.sh
        subprocess.run(
            ["./build.sh", "--no-bindings"],  # Assumendo che build.sh supporti questa opzione
            cwd=project_root,
            check=True,
            capture_output=False
        )
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to compile C++ library")
        print("   Please run './build.sh' manually first")
        return False
    except FileNotFoundError:
        print("⚠️  ./build.sh not found. Assuming library is pre-compiled")
        return True

# ============================================================================
# CONFIGURAZIONE DEI FILE DI BUILD
# ============================================================================
def get_extensions():
    """Configura le estensioni pybind11."""
    
    # Trova Eigen3
    eigen_include = find_eigen3()
    print(f"📐 Using Eigen3 from: {eigen_include}")
    
    # Trova la libreria ML
    ml_library = find_ml_library()
    if ml_library:
        print(f"📚 Using ML library: {ml_library}")
    
    extra_objects = [ml_library] if ml_library and Path(ml_library).exists() else []
    
    ext_modules = [
        Pybind11Extension(
            "ml_core",
            sources=[
                str(project_root / "pybinding" / "ml_core.cpp"),
            ],
            include_dirs=[
                str(project_root / "include"),
                eigen_include,
            ],
            library_dirs=[
                str(project_root / "build" / "lib"),
            ],
            libraries=[],
            extra_objects=extra_objects,
            extra_compile_args=[
                "-std=c++17",
                "-O3",
                "-fPIC",
                "-Wall",
                "-Wextra",
            ],
            extra_link_args=[
                "-Wl,-rpath,$ORIGIN/../build/lib",  # Per trovare la .so in runtime
            ] if sys.platform != "win32" else [],
        ),
    ]
    
    return ext_modules


# ============================================================================
# LEGGI README
# ============================================================================
def read_readme():
    readme_path = project_root / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Machine Learning Library in C++ with Python bindings"


# ============================================================================
# SETUP PRINCIPALE
# ============================================================================
def main():
    # Opzionale: compila automaticamente la libreria C++
    # ensure_cpp_library()  # Decommentare se si vuole la compilazione automatica
    
    ext_modules = get_extensions()
    
    setup(
        name="ml_library",
        version=__version__,
        author="Maurizio Penna",
        author_email="mauriziopenna@gmail.com",
        description="Machine Learning Library in C++ with Python bindings",
        long_description=read_readme(),
        long_description_content_type="text/markdown",
        ext_modules=ext_modules,
        cmdclass={"build_ext": build_ext},
        zip_safe=False,
        python_requires=">=3.7",
        install_requires=[
            "numpy>=1.21.0",
            "pybind11>=2.10.0",
        ],
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: C++",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Scientific/Engineering :: Mathematics",
        ],
        project_urls={
            "Source": "https://github.com/mauriziopenna/ai-devel",
            "Documentation": "https://github.com/mauriziopenna/ai-devel/wiki",
        },
    )


if __name__ == "__main__":
    main()