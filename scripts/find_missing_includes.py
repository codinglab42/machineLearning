#!/usr/bin/env python3
"""
Trova potenziali include mancanti analizzando i simboli usati
"""

import re
import os
from pathlib import Path
from collections import defaultdict

def find_symbols_in_file(filepath):
    """Trova simboli che potrebbero aver bisogno di include"""
    symbols = set()
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Cerca pattern che indicano uso di classi/tipi
    patterns = [
        r'(Eigen::MatrixXd|Eigen::VectorXd|Eigen::VectorXi)',
        r'(std::unique_ptr|std::shared_ptr|std::make_unique)',
        r'(ML_CHECK_\w+|ML_THROW_\w+)',
        r'(loss::Loss|models::Optimizer|layers::DenseLayer)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, content)
        symbols.update(matches)
    
    return symbols

def check_includes(filepath):
    """Verifica se gli include necessari sono presenti"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    needed_includes = {
        'Eigen::MatrixXd': '#include <Eigen/Dense>',
        'std::unique_ptr': '#include <memory>',
        'ML_CHECK_': '#include "exceptions/exception_macros.h"',
        'loss::Loss': '#include "components/loss/loss.h"',
    }
    
    missing = []
    for symbol, include in needed_includes.items():
        if symbol in content and include not in content:
            missing.append((symbol, include))
    
    return missing

def main():
    root = Path("/home/devel/ai-lab/projects/ai-devel/machineLearning")
    
    print("🔍 ANALISI INCLUDE MANCANTI")
    print("="*50)
    
    for src in root.glob("src/**/*.cpp"):
        missing = check_includes(src)
        if missing:
            print(f"\n📄 {src.relative_to(root)}")
            for symbol, include in missing:
                print(f"   ⚠️  Usa '{symbol}' ma manca {include}")

if __name__ == "__main__":
    main()