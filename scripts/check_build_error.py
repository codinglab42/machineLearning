#!/usr/bin/env python3
"""
Analizza gli errori di compilazione per trovare pattern ricorrenti
"""

import re
import sys
from collections import Counter

def analyze_build_log(log_file):
    errors = []
    with open(log_file, 'r') as f:
        for line in f:
            if 'error:' in line:
                # Estrai tipo di errore
                if 'undefined reference' in line:
                    errors.append('undefined_reference')
                elif 'no matching function' in line:
                    errors.append('no_matching_function')
                elif 'was not declared' in line:
                    errors.append('not_declared')
                elif 'redefinition' in line:
                    errors.append('redefinition')
                elif 'multiple definition' in line:
                    errors.append('multiple_definition')
                elif 'does not name a type' in line:
                    errors.append('missing_include')
                else:
                    errors.append('other')
    
    print("🔍 ANALISI ERRORI DI COMPILAZIONE")
    print("="*50)
    for error_type, count in Counter(errors).most_common():
        print(f"  {error_type}: {count}")
    
    return errors

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_build_log(sys.argv[1])
    else:
        print("Usage: python check_build_errors.py build_log.txt")