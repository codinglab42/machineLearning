import sys
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUILD_ROOT = PROJECT_ROOT / "build"
EXTENSION_DIRECTORIES = [
    BUILD_ROOT / "lib",
    BUILD_ROOT / "pybinding",
    *BUILD_ROOT.glob("lib.*"),
    *(build_dir / "lib" for build_dir in BUILD_ROOT.glob("temp.*/ml_core")),
]
extensions = [
    path
    for directory in EXTENSION_DIRECTORIES
    if directory.is_dir()
    for path in directory.glob("ml_core*")
    if path.is_file() and path.name.endswith(tuple(EXTENSION_SUFFIXES))
]

if not extensions:
    raise ImportError(
        "No local ml_core extension found. Build it with "
        "`python setup.py build_ext --inplace` before running the Python tests."
    )

extension = max(extensions, key=lambda path: path.stat().st_mtime_ns)
sys.path.insert(0, str(extension.parent))