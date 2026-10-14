#!/usr/bin/env python3
"""
Analisi della coerenza del codice della libreria ML
Identifica:
- Metodi dichiarati ma non implementati
- Metodi implementati ma non utilizzati
- Include inutilizzati
- Componenti non integrati
"""

import os
import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Set, Dict, List, Tuple

class CodeAnalyzer:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.include_dir = self.root_dir / "include"
        self.src_dir = self.root_dir / "src"
        self.test_dir = self.root_dir / "tests"
        
        # Per tracciare dichiarazioni e implementazioni
        self.declarations: Dict[str, Set[str]] = defaultdict(set)  # file -> metodi
        self.implementations: Dict[str, Set[str]] = defaultdict(set)  # file -> metodi
        self.calls: Dict[str, Set[str]] = defaultdict(set)  # file -> chiamate
        self.includes: Dict[str, Set[str]] = defaultdict(set)  # file -> include
        
    def extract_method_name(self, line: str) -> str:
        """Estrae il nome del metodo da una dichiarazione o implementazione"""
        # Pattern per metodi: return_type ClassName::method_name(params)
        pattern = r'\w+(?:::)?(\w+)\s*\('
        match = re.search(pattern, line)
        if match:
            return match.group(1)
        return None
    
    def extract_include(self, line: str) -> str:
        """Estrae il nome del file incluso"""
        match = re.search(r'#include\s+[<"]([^>"]+)[>"]', line)
        if match:
            return match.group(1)
        return None
    
    def analyze_headers(self):
        """Analizza tutti i file header per trovare dichiarazioni di metodi"""
        print("\n" + "="*60)
        print("1. ANALISI HEADER (.h)")
        print("="*60)
        
        for header in self.include_dir.rglob("*.h"):
            rel_path = header.relative_to(self.include_dir)
            print(f"\n📄 {rel_path}")
            
            with open(header, 'r') as f:
                content = f.read()
                
            # Trova tutte le dichiarazioni di metodi (non implementazioni inline)
            lines = content.split('\n')
            methods_found = []
            
            for i, line in enumerate(lines):
                # Cerca dichiarazioni di metodo (terminano con ;)
                if '(' in line and ')' in line and ';' in line:
                    # Esclude costruttori/destruttori
                    if '~' not in line and 'operator' not in line:
                        method = self.extract_method_name(line)
                        if method and method not in ['if', 'for', 'while']:
                            methods_found.append(method)
                            self.declarations[str(rel_path)].add(method)
            
            if methods_found:
                print(f"   Metodi dichiarati: {', '.join(methods_found[:10])}")
                if len(methods_found) > 10:
                    print(f"   ... e altri {len(methods_found)-10}")
            else:
                print("   Nessun metodo pubblico trovato")
    
    def analyze_sources(self):
        """Analizza tutti i file sorgente per trovare implementazioni"""
        print("\n" + "="*60)
        print("2. ANALISI SORGENTI (.cpp)")
        print("="*60)
        
        for src in self.src_dir.rglob("*.cpp"):
            rel_path = src.relative_to(self.src_dir)
            print(f"\n📄 {rel_path}")
            
            with open(src, 'r') as f:
                content = f.read()
                
            lines = content.split('\n')
            methods_found = []
            
            for line in lines:
                # Cerca implementazioni: ClassName::methodName
                if '::' in line and '(' in line:
                    method = self.extract_method_name(line)
                    if method and method not in ['if', 'for', 'while']:
                        methods_found.append(method)
                        self.implementations[str(rel_path)].add(method)
                        
                # Cerca chiamate a metodi
                call_match = re.search(r'(\w+)\s*\([^)]*\)', line)
                if call_match:
                    call = call_match.group(1)
                    if call not in ['if', 'for', 'while', 'return', 'sizeof']:
                        self.calls[str(rel_path)].add(call)
                
                # Cerca include
                include = self.extract_include(line)
                if include:
                    self.includes[str(rel_path)].add(include)
            
            if methods_found:
                print(f"   Metodi implementati: {', '.join(methods_found[:10])}")
                if len(methods_found) > 10:
                    print(f"   ... e altri {len(methods_found)-10}")
    
    def analyze_tests(self):
        """Analizza i test per vedere cosa viene effettivamente testato"""
        print("\n" + "="*60)
        print("3. ANALISI TEST")
        print("="*60)
        
        tested_methods = set()
        
        for test in self.test_dir.rglob("*.cpp"):
            rel_path = test.relative_to(self.test_dir)
            print(f"\n🧪 {rel_path}")
            
            with open(test, 'r') as f:
                content = f.read()
            
            # Cerca chiamate a metodi nei test
            lines = content.split('\n')
            test_calls = []
            
            for line in lines:
                # Cerca pattern come obj.method( o Class::method
                matches = re.findall(r'(\w+)\s*\([^)]*\)', line)
                for match in matches:
                    if match not in ['TEST_F', 'TEST', 'EXPECT', 'ASSERT', 'std']:
                        test_calls.append(match)
                        tested_methods.add(match)
            
            if test_calls:
                print(f"   Metodi testati: {', '.join(set(test_calls[:10]))}")
        
        return tested_methods
    
    def find_inconsistencies(self):
        """Trova incoerenze tra dichiarazioni e implementazioni"""
        print("\n" + "="*60)
        print("4. INCOERENZE")
        print("="*60)
        
        # Trova metodi dichiarati ma non implementati
        all_declared = set()
        all_implemented = set()
        
        for methods in self.declarations.values():
            all_declared.update(methods)
        
        for methods in self.implementations.values():
            all_implemented.update(methods)
        
        missing_impl = all_declared - all_implemented
        if missing_impl:
            print(f"\n⚠️  METODI DICHIARATI MA NON IMPLEMENTATI: {len(missing_impl)}")
            for method in sorted(missing_impl)[:20]:
                print(f"   - {method}")
        
        # Trova include inutilizzati (semplice euristica)
        print(f"\n📦 INCLUDE PIÙ COMUNI:")
        all_includes = defaultdict(int)
        for includes in self.includes.values():
            for inc in includes:
                all_includes[inc] += 1
        
        for inc, count in sorted(all_includes.items(), key=lambda x: -x[1])[:15]:
            print(f"   {inc}: {count} file")
    
    def find_unintegrated_components(self):
        """Trova componenti implementati ma non integrati"""
        print("\n" + "="*60)
        print("5. COMPONENTI NON INTEGRATI")
        print("="*60)
        
        # Layer non utilizzati in NeuralNetwork
        nn_file = self.src_dir / "models" / "neural_network.cpp"
        if nn_file.exists():
            with open(nn_file, 'r') as f:
                nn_content = f.read()
            
            layers = ['conv2d', 'pooling', 'flatten', 'dropout', 'batch_norm', 
                     'simple_rnn', 'lstm', 'gru']
            
            print("\n🔧 LAYER NELLA NEURAL NETWORK:")
            for layer in layers:
                if f"add_{layer}_layer" in nn_content:
                    print(f"   ✅ {layer}_layer: integrato")
                else:
                    print(f"   ❌ {layer}_layer: NON integrato")
        
        # Optimizer non utilizzati
        optimizers = ['MomentumOptimizer', 'RMSpropOptimizer', 'AdagradOptimizer']
        print("\n⚡ OPTIMIZER:")
        for opt in optimizers:
            found = False
            for src in self.src_dir.rglob("*.cpp"):
                with open(src, 'r') as f:
                    if opt in f.read():
                        found = True
                        break
            if found:
                print(f"   ✅ {opt}: utilizzato")
            else:
                print(f"   ❌ {opt}: NON utilizzato")
    
    def generate_report(self):
        """Genera report completo"""
        print("\n" + "="*60)
        print("RIEPILOGO FINALE")
        print("="*60)
        
        print(f"\n📊 STATISTICHE:")
        print(f"   Header analizzati: {len(self.declarations)}")
        print(f"   Sorgenti analizzati: {len(self.implementations)}")
        print(f"   Metodi dichiarati: {sum(len(v) for v in self.declarations.values())}")
        print(f"   Metodi implementati: {sum(len(v) for v in self.implementations.values())}")
        
        # Salva report su file
        report_path = self.root_dir / "code_analysis_report.txt"
        with open(report_path, 'w') as f:
            f.write("=== REPORT ANALISI CODICE ML LIBRARY ===\n\n")
            f.write(f"Header analizzati: {len(self.declarations)}\n")
            f.write(f"Sorgenti analizzati: {len(self.implementations)}\n\n")
            
            f.write("=== METODI DICHIARATI MA NON IMPLEMENTATI ===\n")
            all_declared = set()
            for methods in self.declarations.values():
                all_declared.update(methods)
            all_implemented = set()
            for methods in self.implementations.values():
                all_implemented.update(methods)
            missing = all_declared - all_implemented
            for m in sorted(missing):
                f.write(f"  {m}\n")
        
        print(f"\n📄 Report salvato in: {report_path}")

def main():
    # Trova la directory root (dove si trova lo script)
    script_dir = Path(__file__).parent.parent
    root_dir = script_dir
    
    print("🔍 ANALISI COERENZA CODICE ML LIBRARY")
    print(f"📁 Root directory: {root_dir}")
    
    analyzer = CodeAnalyzer(str(root_dir))
    analyzer.analyze_headers()
    analyzer.analyze_sources()
    tested = analyzer.analyze_tests()
    analyzer.find_inconsistencies()
    analyzer.find_unintegrated_components()
    analyzer.generate_report()

if __name__ == "__main__":
    main()