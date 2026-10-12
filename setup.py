from importlib.machinery import EXTENSION_SUFFIXES
import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


BASE_DIR = Path(__file__).parent.resolve()


class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    def build_extension(self, extension):
        cmake = shutil.which("cmake")
        if cmake is None:
            raise RuntimeError("CMake 3.15 or newer is required to build ml_core")

        build_dir = Path(self.build_temp).resolve() / extension.name
        output_dir = Path(self.get_ext_fullpath(extension.name)).resolve().parent
        build_dir.mkdir(parents=True, exist_ok=True)

        configure_command = [
            cmake,
            "-S",
            str(BASE_DIR),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={build_dir / 'install'}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            "-DBUILD_PYTHON_BINDINGS=ON",
            "-DBUILD_TESTS=OFF",
            "-DBUILD_EXAMPLES=OFF",
        ]
        build_environment = os.environ.copy()
        build_environment.pop("PYENV_ROOT", None)
        subprocess.run(
            configure_command,
            check=True,
            cwd=BASE_DIR,
            env=build_environment,
        )

        build_command = [
            cmake,
            "--build",
            str(build_dir),
            "--target",
            extension.name,
            "--config",
            "Release",
        ]
        if self.parallel:
            build_command.extend(["--parallel", str(self.parallel)])
        subprocess.run(build_command, check=True, cwd=BASE_DIR, env=build_environment)

        candidates = [
            path
            for output_root in (build_dir / "lib", build_dir / "bin")
            if output_root.exists()
            for path in output_root.rglob(f"{extension.name}*")
            if path.is_file() and path.name.endswith(tuple(EXTENSION_SUFFIXES))
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected one built extension for {extension.name}, found {len(candidates)}"
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidates[0], output_dir / candidates[0].name)


setup(
    name="ml_core",
    version="3.0.0",
    author="Maurizio Penna",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    project_urls={
        "Source": "https://github.com/codinglab42/machineLearning",
        "Documentation": "https://github.com/codinglab42/machineLearning/tree/main/docs",
    },
    ext_modules=[CMakeExtension("ml_core")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
