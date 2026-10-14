#!/usr/bin/env python3
"""
Trova funzioni implementate ma mai chiamate
"""

import re
from pathlib import Path
from collections import defaultdict

def find_function_calls(filepath):
    """Trova tutte le chiamate a funzioni in un file"""
    calls = set()
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Pattern per chiamate di funzione
    pattern = r'(\w+)\s*\('
    matches = re.findall(pattern, content)
    
    # Filtra parole chiave C++
    keywords = {'if', 'for', 'while', 'switch', 'return', 'sizeof', 'new', 'delete'}
    for match in matches:
        if match not in keywords and not match[0].isdigit():
            calls.add(match)
    
    return calls

def find_function_definitions(filepath):
    """Trova tutte le definizioni di funzione in un file"""
    definitions = set()
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Pattern per definizioni: Tipo Classe::metodo(parametri)
    pattern = r'(\w+)::(\w+)\s*\('
    matches = re.findall(pattern, content)
    for match in matches:
        definitions.add(f"{match[0]}::{match[1]}")
    
    return definitions

def main():
    root = Path("/home/devel/ai-lab/projects/ai-devel/machineLearning")
    
    print("🔍 ANALISI CODICE MORTO")
    print("="*50)
    
    # Trova tutte le definizioni
    all_definitions = set()
    for src in root.glob("src/**/*.cpp"):
        all_definitions.update(find_function_definitions(src))
    
    # Trova tutte le chiamate
    all_calls = set()
    for src in root.glob("src/**/*.cpp"):
        all_calls.update(find_function_calls(src))
    for test in root.glob("tests/**/*.cpp"):
        all_calls.update(find_function_calls(test))
    
    # Trova funzioni definite ma mai chiamate
    dead_code = all_definitions - all_calls
    
    print(f"\n📊 Statistiche:")
    print(f"   Funzioni definite: {len(all_definitions)}")
    print(f"   Funzioni chiamate: {len(all_calls)}")
    print(f"   Potenziale codice morto: {len(dead_code)}")
    
    if dead_code:
        print("\n⚠️ FUNZIONI DEFINITE MA MAI CHIAMATE:")
        for func in sorted(dead_code)[:20]:
            print(f"   - {func}")

if __name__ == "__main__":
    main()