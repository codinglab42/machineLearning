import sys
import os

# Ottieni il percorso assoluto della cartella 'tests/python'
current_dir = os.path.dirname(os.path.abspath(__file__))

# Se la struttura è:
# machineLearning/
# ├── build/            <-- La libreria è qui
# └── tests/
#     └── python/
#         └── conftest.py

# Dobbiamo risalire di DUE livelli per arrivare alla radice 'machineLearning'
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
build_path = os.path.join(project_root, 'build', 'lib')

if build_path not in sys.path:
    sys.path.insert(0, build_path)

print(f"\n[DEBUG] Percorso progetto: {project_root}")
print(f"[DEBUG] Cercando la libreria in: {build_path}")