#!/usr/bin/env python3
"""
Script per aggiornare automaticamente i file di test dei layer
Aggiunge i nuovi metodi virtuali ai MockLayer
"""

import os
import re
from pathlib import Path

class LayerTestUpdater:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.test_dir = self.root_dir / "tests" / "cpp" / "unit" / "layers"
        
        # Nuovi metodi da aggiungere al MockLayer
        self.new_methods = '''
    MOCK_METHOD(Eigen::MatrixXd, get_weights_gradient, (), (const, override));
    MOCK_METHOD(Eigen::VectorXd, get_bias_gradient, (), (const, override));
    MOCK_METHOD(bool, get_use_bias, (), (const, override));'''
        
        # Pattern per trovare la fine dei MOCK_METHOD nel MockLayer
        self.mock_end_pattern = r'(MOCK_METHOD\([^;]+;[\s]*)+'
        
    def find_test_files(self):
        """Trova tutti i file di test dei layer"""
        test_files = []
        if self.test_dir.exists():
            for file in self.test_dir.glob("*.cpp"):
                # Cerca file che contengono MockLayer
                with open(file, 'r') as f:
                    content = f.read()
                    if 'MockLayer' in content and 'MOCK_METHOD' in content:
                        test_files.append(file)
        return test_files
    
    def check_needs_update(self, filepath: Path) -> bool:
        """Verifica se il file ha bisogno di aggiornamento"""
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Verifica se i nuovi metodi sono già presenti
        has_weights_gradient = 'get_weights_gradient' in content
        has_bias_gradient = 'get_bias_gradient' in content
        has_use_bias = 'get_use_bias' in content
        
        return not (has_weights_gradient and has_bias_gradient and has_use_bias)
    
    def find_mock_layer_class(self, content: str) -> tuple:
        """Trova la classe MockLayer e la posizione per inserire i metodi"""
        # Cerca la definizione della classe MockLayer
        class_pattern = r'class\s+MockLayer\s*:\s*public\s+Layer\s*\{'
        class_match = re.search(class_pattern, content)
        
        if not class_match:
            return None, None
        
        class_start = class_match.end()
        
        # Cerca la sezione public (di solito dopo la dichiarazione della classe)
        # I MOCK_METHOD sono di solito nella sezione public
        public_pattern = r'public\s*:'
        public_match = re.search(public_pattern, content[class_start:])
        
        if public_match:
            start_pos = class_start + public_match.end()
        else:
            start_pos = class_start
        
        # Cerca la fine della sezione dei MOCK_METHOD
        # Di solito c'è un'altra sezione (protected, private) o la fine della classe
        end_pattern = r'(protected:|private:|};)'
        end_match = re.search(end_pattern, content[start_pos:])
        
        if end_match:
            insert_pos = start_pos + end_match.start()
        else:
            # Se non trova, cerca la fine della classe
            brace_count = 0
            for i in range(start_pos, len(content)):
                if content[i] == '{':
                    brace_count += 1
                elif content[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        insert_pos = i
                        break
            else:
                return None, None
        
        return start_pos, insert_pos
    
    def update_test_file(self, filepath: Path) -> bool:
        """Aggiorna un singolo file di test"""
        print(f"   🔧 Aggiornamento {filepath.name}...")
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Trova la posizione per inserire i nuovi metodi
        start_pos, insert_pos = self.find_mock_layer_class(content)
        
        if insert_pos is None:
            print(f"      ❌ Non trovata la classe MockLayer o posizione per inserire")
            return False
        
        # Verifica se i metodi sono già presenti
        if 'get_weights_gradient' in content[start_pos:insert_pos]:
            print(f"      ⏭️  Metodi già presenti, salto...")
            return False
        
        # Inserisci i nuovi metodi
        new_content = content[:insert_pos] + self.new_methods + content[insert_pos:]
        
        with open(filepath, 'w') as f:
            f.write(new_content)
        
        print(f"      ✅ {filepath.name} aggiornato")
        return True
    
    def update_all(self):
        """Aggiorna tutti i file di test"""
        print("=" * 70)
        print("🔧 AGGIORNAMENTO TEST DEI LAYER")
        print("=" * 70)
        
        test_files = self.find_test_files()
        
        if not test_files:
            print("   Nessun file di test dei layer trovato!")
            return
        
        print(f"\n📁 File di test trovati ({len(test_files)}):")
        for f in test_files:
            print(f"   - {f.name}")
        
        print("\n" + "=" * 70)
        print("🔍 VERIFICA E AGGIORNAMENTO")
        print("=" * 70)
        
        updated_count = 0
        for filepath in test_files:
            print(f"\n📄 {filepath.name}")
            
            if self.check_needs_update(filepath):
                if self.update_test_file(filepath):
                    updated_count += 1
            else:
                print(f"   ✅ Già aggiornato")
        
        print("\n" + "=" * 70)
        print(f"✅ Aggiornamento completato! ({updated_count} file aggiornati)")
        print("=" * 70)

def main():
    script_dir = Path(__file__).parent.parent
    root_dir = script_dir
    
    print(f"📁 Root directory: {root_dir}\n")
    
    updater = LayerTestUpdater(str(root_dir))
    updater.update_all()

if __name__ == "__main__":
    main()