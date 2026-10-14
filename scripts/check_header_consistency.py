#!/usr/bin/env python3
"""
Verifica che i metodi dichiarati negli header siano implementati nei .cpp
"""

import re
from pathlib import Path

def extract_declared_methods(header_file):
    """Estrae metodi dichiarati nell'header"""
    methods = set()
    with open(header_file, 'r') as f:
        content = f.read()
    
    # Pattern per dichiarazioni di metodo
    pattern = r'(\w+(?:<[^>]+>)?)\s+(\w+)\s*\([^)]*\)\s*(?:const)?\s*(?:override)?\s*;'
    matches = re.findall(pattern, content)
    for match in matches:
        methods.add(match[1])  # nome metodo
    
    # Pattern per metodi virtuali puri
    pattern2 = r'virtual\s+\w+(?:<[^>]+>)?\s+(\w+)\s*\([^)]*\)\s*=\s*0'
    matches2 = re.findall(pattern2, content)
    for match in matches2:
        methods.add(match)
    
    return methods

def extract_implemented_methods(src_file):
    """Estrae metodi implementati nel .cpp"""
    methods = set()
    with open(src_file, 'r') as f:
        content = f.read()
    
    # Pattern per implementazioni
    pattern = r'(\w+)::(\w+)\s*\('
    matches = re.findall(pattern, content)
    for match in matches:
        methods.add(match[1])
    
    return methods

def main():
    root = Path("/home/devel/ai-lab/projects/ai-devel/machineLearning")
    
    print("🔍 ANALISI COERENZA HEADER/IMPLEMENTAZIONE")
    print("="*60)
    
    # Mappa header -> src
    header_src_map = {
        'models/neural_network.h': 'models/neural_network.cpp',
        'models/linear_regression.h': 'models/linear_regression.cpp',
        'models/logistic_regression.h': 'models/logistic_regression.cpp',
        'components/layers/dense_layer.h': 'components/layers/dense_layer.cpp',
        'components/loss/binary_cross_entropy_loss.h': 'components/loss/binary_cross_entropy_loss.cpp',
        'components/optimizers/adam_optimizer.h': 'components/optimizers/adam_optimizer.cpp',
    }
    
    issues = []
    
    for header_rel, src_rel in header_src_map.items():
        header = root / "include" / header_rel
        src = root / "src" / src_rel
        
        if not header.exists() or not src.exists():
            continue
        
        declared = extract_declared_methods(header)
        implemented = extract_implemented_methods(src)
        
        missing = declared - implemented
        extra = implemented - declared
        
        if missing:
            issues.append(f"\n❌ {header_rel}")
            for m in missing:
                issues.append(f"   Dichiarato ma non implementato: {m}")
        
        if extra:
            issues.append(f"\n⚠️ {header_rel}")
            for m in extra:
                issues.append(f"   Implementato ma non dichiarato: {m}")
    
    if issues:
        print("\n".join(issues))
    else:
        print("\n✅ Tutti i metodi dichiarati sono implementati!")

if __name__ == "__main__":
    main()