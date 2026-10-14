#!/usr/bin/env python3
"""
Analizza quali componenti sono testati e quali no
"""

import re
from pathlib import Path
from collections import defaultdict

def extract_tested_methods(test_file):
    """Estrae i metodi testati da un file di test"""
    tested = set()
    with open(test_file, 'r') as f:
        content = f.read()
    
    # Cerca pattern come TEST_F(ClassName, MethodName)
    pattern = r'TEST_F\s*\([^,]+,\s*([^)]+)\)'
    matches = re.findall(pattern, content)
    tested.update(matches)
    
    # Cerca chiamate a metodi specifici
    patterns = [
        r'(\w+)\.fit\(',
        r'(\w+)\.predict\(',
        r'(\w+)\.score\(',
        r'(\w+)\.set_loss_function\(',
        r'(\w+)\.set_regularizer\(',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, content)
        tested.update([f"{m}_method" for m in matches])
    
    return tested

def main():
    root = Path("/home/devel/ai-lab/projects/ai-devel/machineLearning")
    
    print("🔍 ANALISI COPERTURA TEST")
    print("="*50)
    
    # Componenti principali
    components = {
        'NeuralNetwork': ['fit', 'predict', 'score', 'predict_proba', 'save', 'load'],
        'LinearRegression': ['fit', 'predict', 'score', 'save', 'load'],
        'LogisticRegression': ['fit', 'predict', 'score', 'predict_class'],
        'LossFunctions': ['binary_crossentropy', 'mse', 'mae', 'huber'],
        'Optimizers': ['SGD', 'Adam', 'Momentum'],
        'Regularizers': ['L1', 'L2', 'ElasticNet'],
    }
    
    # Trova tutti i test
    test_files = list(root.glob("tests/cpp/**/*.cpp"))
    all_tested = set()
    for test_file in test_files:
        all_tested.update(extract_tested_methods(test_file))
    
    # Verifica copertura per componente
    for component, methods in components.items():
        print(f"\n📦 {component}:")
        for method in methods:
            tested = any(method.lower() in t.lower() for t in all_tested)
            status = "✅" if tested else "❌"
            print(f"   {status} {method}")

if __name__ == "__main__":
    main()