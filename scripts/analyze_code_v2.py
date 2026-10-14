#!/usr/bin/env python3
# scripts/analyze_code_v2.py

import os
import re
from pathlib import Path

def extract_method_names(filepath):
    """Estrae correttamente i nomi dei metodi da un file C++"""
    methods = set()
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Pattern per metodi: Tipo NomeClasse::nomeMetodo(parametri)
    pattern = r'(\w+(?:<[^>]+>)?)\s+(\w+)::(\w+)\s*\([^)]*\)'
    matches = re.findall(pattern, content)
    for match in matches:
        methods.add(match[2])  # il nome del metodo
    
    # Pattern per metodi virtuali puri: virtual tipo nomeMetodo(parametri) = 0;
    pattern2 = r'virtual\s+\w+(?:<[^>]+>)?\s+(\w+)\s*\([^)]*\)\s*=\s*0'
    matches2 = re.findall(pattern2, content)
    for match in matches2:
        methods.add(match)
    
    return methods

def main():
    root = Path("/home/devel/ai-lab/projects/ai-devel/machineLearning")
    
    print("🔍 ANALISI METODI IMPLEMENTATI VS DICHIARATI\n")
    
    # Header files
    headers = list(root.glob("include/**/*.h"))
    srcs = list(root.glob("src/**/*.cpp"))
    
    declared = {}
    implemented = {}
    
    for h in headers:
        methods = extract_method_names(h)
        if methods:
            declared[h.relative_to(root)] = methods
    
    for s in srcs:
        methods = extract_method_names(s)
        if methods:
            implemented[s.relative_to(root)] = methods
    
    print(f"Header analizzati: {len(declared)}")
    print(f"Sorgenti analizzati: {len(implemented)}")
    
    # Trova metodi dichiarati ma non implementati
    all_declared = set()
    for methods in declared.values():
        all_declared.update(methods)
    
    all_implemented = set()
    for methods in implemented.values():
        all_implemented.update(methods)
    
    missing = all_declared - all_implemented
    
    print(f"\n⚠️ METODI DICHIARATI MA NON IMPLEMENTATI: {len(missing)}")
    for m in sorted(missing)[:30]:
        print(f"   - {m}")
    
    # Trova layer non usati
    nn_file = root / "src" / "models" / "neural_network.cpp"
    if nn_file.exists():
        with open(nn_file, 'r') as f:
            nn_content = f.read()
        
        print("\n🔧 LAYER IN NEURAL NETWORK:")
        for layer in ['simple_rnn', 'lstm', 'gru', 'conv2d', 'pooling', 'flatten', 'dropout', 'batch_norm']:
            if f"add_{layer}_layer" in nn_content:
                print(f"   ✅ {layer}_layer: integrato")
            else:
                print(f"   ❌ {layer}_layer: NON integrato")

if __name__ == "__main__":
    main()