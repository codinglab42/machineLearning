#!/usr/bin/env python3
"""
Analisi coerenza header (include/) vs implementazioni (src/) per progetti C++.
Gestisce: template, namespace annidati, tipi di ritorno multiriga, argomenti multiriga.
"""

import os
import re
import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Set, List, Tuple


# ──────────────────────────────────────────────
# Colori ANSI per output leggibile nel terminale
# ──────────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"
    GRAY   = "\033[90m"
    BLUE   = "\033[94m"

def ok(msg):    print(f"  {C.GREEN}✔{C.RESET}  {msg}")
def warn(msg):  print(f"  {C.YELLOW}⚠{C.RESET}  {msg}")
def err(msg):   print(f"  {C.RED}✘{C.RESET}  {msg}")
def info(msg):  print(f"  {C.CYAN}→{C.RESET}  {msg}")
def dim(msg):   print(f"  {C.GRAY}{msg}{C.RESET}")


# ──────────────────────────────────────────────
# Struttura dati per ogni classe trovata
# ──────────────────────────────────────────────
@dataclass
class ClassInfo:
    name: str
    header_file: str = ""
    source_files: Set[str] = field(default_factory=set)
    methods_declared: Set[str] = field(default_factory=set)
    methods_implemented: Set[str] = field(default_factory=set)

    @property
    def missing(self) -> Set[str]:
        return self.methods_declared - self.methods_implemented

    @property
    def extra(self) -> Set[str]:
        """Metodi implementati ma non dichiarati nell'header (es. metodi privati/inline)."""
        return self.methods_implemented - self.methods_declared


# ──────────────────────────────────────────────
# Utilità: rimuove commenti C/C++ dal sorgente
# ──────────────────────────────────────────────
def strip_comments(text: str) -> str:
    # Rimuove commenti /* ... */ e // ...
    text = re.sub(r'/\*.*?\*/', ' ', text, flags=re.DOTALL)
    text = re.sub(r'//[^\n]*', '', text)
    return text


# ──────────────────────────────────────────────
# Estrai classi e metodi dagli header
# ──────────────────────────────────────────────

# Metodo dichiarato (termina con ; oppure = 0; oppure = default; oppure = delete;)
# Esclude costruttori/distruttori se vuoi, ma qui li include per completezza.
METHOD_DECL = re.compile(
    r'''
    (?:
        (?:virtual|static|explicit|inline|constexpr|[[nodiscard]]\s*)?  # qualificatori
        \s*
    )*
    (?:[\w:<>,*&\s]+?\s+)?          # tipo di ritorno (opzionale, es. void, std::string)
    (\w+)                            # nome metodo (gruppo 1)
    \s*\(                            # parentesi aperta
    [^)]*                            # parametri (singola riga; vedi nota sotto)
    \)
    \s*(?:const)?\s*(?:noexcept)?\s*(?:override)?\s*(?:final)?
    \s*(?:=\s*(?:0|default|delete))?
    \s*;
    ''',
    re.VERBOSE | re.MULTILINE
)


def extract_class_body(content: str, class_name: str) -> str | None:
    """Ritorna il corpo della classe (tra le prime { }) oppure None."""
    # Cerca 'class ClassName' o 'struct ClassName' con eventuale ereditarietà
    pattern = rf'\b(?:class|struct)\s+{re.escape(class_name)}\b[^{{]*\{{'
    m = re.search(pattern, content)
    if not m:
        return None

    start = m.end()
    depth = 1
    i = start
    while i < len(content) and depth > 0:
        if content[i] == '{':
            depth += 1
        elif content[i] == '}':
            depth -= 1
        i += 1
    return content[start : i - 1]


def extract_declared_methods(header_path: Path, class_name: str) -> Set[str]:
    """Estrae i nomi dei metodi dichiarati per una classe in un header."""
    methods: Set[str] = set()
    try:
        content = strip_comments(header_path.read_text(encoding='utf-8', errors='replace'))
        body = extract_class_body(content, class_name)
        if body is None:
            return methods

        for m in METHOD_DECL.finditer(body):
            name = m.group(1)
            # Escludi costruttori/distruttori e keyword che matchano per errore
            if name in ('if', 'for', 'while', 'switch', 'return', 'class',
                        'struct', 'namespace', 'operator'):
                continue
            if name == class_name or name == f'~{class_name}':
                continue
            methods.add(name)
    except Exception as e:
        warn(f"Header {header_path.name}: {e}")
    return methods


# ──────────────────────────────────────────────
# Estrai implementazioni dai file .cpp
# ──────────────────────────────────────────────

# Pattern robusto per ClassName::method( anche con:
#   - namespace annidati  (A::B::method)
#   - template            (MyClass<T>::method)
#   - tipo di ritorno     su riga precedente
#   - argomenti multiriga (il corpo { è trovato dopo)
IMPL_PATTERN = re.compile(
    r'''
    # Classe (con possibili namespace e template prima di ::)
    (?:[\w]+(?:\s*<[^>]*>)?\s*::)+   # uno o più "Foo::" o "Foo<T>::"
    (\w+)                              # nome metodo (gruppo 1)
    \s*\(                              # parentesi argomenti
    ''',
    re.VERBOSE | re.MULTILINE
)

# Per estrarre anche il nome della classe (l'ultimo :: prima del metodo)
IMPL_CLASS_METHOD = re.compile(
    r'((?:[\w]+(?:\s*<[^>]*>)?\s*::)+)(\w+)\s*\(',
    re.MULTILINE
)


def extract_implementations(src_path: Path) -> Dict[str, Set[str]]:
    """
    Ritorna un dict {class_name: {method1, method2, ...}} per tutte le
    implementazioni ClassName::method trovate nel file .cpp.
    Gestisce namespace annidati prendendo solo l'ultima componente come classe.
    """
    result: Dict[str, Set[str]] = defaultdict(set)
    try:
        content = strip_comments(src_path.read_text(encoding='utf-8', errors='replace'))

        for m in IMPL_CLASS_METHOD.finditer(content):
            qualifier = m.group(1)   # es. "AI::NeuralNet::" o "MyClass::"
            method    = m.group(2)   # es. "forward"

            # Prende il nome della classe: ultima parola prima di ::
            # "AI::NeuralNet::" → ["AI", "NeuralNet"] → "NeuralNet"
            parts = [p.strip() for p in qualifier.split('::') if p.strip()]
            # Rimuove eventuali parametri template dal nome classe
            class_name = re.sub(r'<.*>', '', parts[-1]).strip()

            if not class_name or not method:
                continue
            if method in ('if', 'for', 'while', 'switch', 'return'):
                continue

            result[class_name].add(method)
    except Exception as e:
        warn(f"Sorgente {src_path.name}: {e}")
    return result


# ──────────────────────────────────────────────
# Analizzatore principale
# ──────────────────────────────────────────────
class Analyzer:
    def __init__(self, root_dir: str):
        self.root = Path(root_dir).resolve()
        self.classes: Dict[str, ClassInfo] = {}

    # ── Scansione header ──────────────────────
    def scan_headers(self):
        include_dir = self.root / "include"
        if not include_dir.exists():
            err(f"Directory 'include/' non trovata in {self.root}")
            sys.exit(1)

        class_pattern = re.compile(r'\b(?:class|struct)\s+(\w+)\b')

        for header in sorted(include_dir.rglob("*.h")) + sorted(include_dir.rglob("*.hpp")):
            try:
                content = strip_comments(header.read_text(encoding='utf-8', errors='replace'))
            except Exception as e:
                warn(f"Impossibile leggere {header}: {e}")
                continue

            for m in class_pattern.finditer(content):
                class_name = m.group(1)
                # Filtra keyword C++ che non sono classi
                if class_name in ('if', 'else', 'for', 'while', 'switch',
                                  'return', 'public', 'private', 'protected',
                                  'virtual', 'override', 'final', 'typename',
                                  'template', 'namespace', 'using', 'typedef'):
                    continue

                if class_name not in self.classes:
                    self.classes[class_name] = ClassInfo(name=class_name)

                cls = self.classes[class_name]
                if not cls.header_file:
                    cls.header_file = str(header.relative_to(self.root))

                methods = extract_declared_methods(header, class_name)
                cls.methods_declared.update(methods)

    # ── Scansione sorgenti ────────────────────
    def scan_sources(self):
        src_dir = self.root / "src"
        if not src_dir.exists():
            err(f"Directory 'src/' non trovata in {self.root}")
            sys.exit(1)

        for src in sorted(src_dir.rglob("*.cpp")) + sorted(src_dir.rglob("*.cc")):
            rel = str(src.relative_to(self.root))
            implementations = extract_implementations(src)

            for class_name, methods in implementations.items():
                if class_name in self.classes:
                    self.classes[class_name].source_files.add(rel)
                    self.classes[class_name].methods_implemented.update(methods)

    # ── Report su terminale ───────────────────
    def print_report(self):
        print(f"\n{C.BOLD}{C.BLUE}{'═'*70}{C.RESET}")
        print(f"{C.BOLD}  REPORT COERENZA  ·  Header vs Implementazioni{C.RESET}")
        print(f"{C.BOLD}{C.BLUE}{'═'*70}{C.RESET}\n")

        classes_ok     = []
        classes_warn   = []  # metodi mancanti
        classes_nosrc  = []  # nessun sorgente trovato

        for class_name, cls in sorted(self.classes.items()):
            if not cls.methods_declared:
                continue  # classe senza metodi rilevati (es. struct dati)
            if not cls.source_files:
                classes_nosrc.append(cls)
            elif cls.missing:
                classes_warn.append(cls)
            else:
                classes_ok.append(cls)

        # ── Classi OK ──
        if classes_ok:
            print(f"{C.BOLD}  ✅ Classi complete ({len(classes_ok)}){C.RESET}")
            for cls in classes_ok:
                print(f"     {C.GREEN}{cls.name}{C.RESET}  "
                      f"{C.GRAY}({len(cls.methods_declared)} metodi · {cls.header_file}){C.RESET}")
            print()

        # ── Classi senza sorgente ──
        if classes_nosrc:
            print(f"{C.BOLD}  ❓ Sorgente non trovato ({len(classes_nosrc)}){C.RESET}")
            for cls in classes_nosrc:
                print(f"     {C.YELLOW}{cls.name}{C.RESET}  "
                      f"{C.GRAY}(header: {cls.header_file}){C.RESET}")
                if cls.methods_declared:
                    dim(f"     metodi dichiarati: {', '.join(sorted(cls.methods_declared))}")
            print()

        # ── Classi con metodi mancanti ──
        if classes_warn:
            print(f"{C.BOLD}  ⚠️  Implementazioni mancanti ({len(classes_warn)} classi){C.RESET}")
            for cls in classes_warn:
                print(f"\n     {C.RED}{cls.name}{C.RESET}  "
                      f"{C.GRAY}(header: {cls.header_file}){C.RESET}")
                if cls.source_files:
                    dim(f"     sorgente: {', '.join(sorted(cls.source_files))}")
                for m in sorted(cls.missing):
                    print(f"       {C.RED}✘{C.RESET}  {m}")
                if cls.extra:
                    dim(f"     implementati ma non dichiarati: {', '.join(sorted(cls.extra))}")
            print()

        # ── Riepilogo numerico ──
        total_decl  = sum(len(c.methods_declared)  for c in self.classes.values())
        total_impl  = sum(len(c.methods_implemented & c.methods_declared) for c in self.classes.values())
        total_miss  = sum(len(c.missing)            for c in self.classes.values())

        print(f"{C.BOLD}{C.BLUE}{'─'*70}{C.RESET}")
        print(f"  Classi totali         : {C.BOLD}{len(self.classes)}{C.RESET}")
        print(f"  Metodi dichiarati     : {C.BOLD}{total_decl}{C.RESET}")
        print(f"  Metodi implementati   : {C.BOLD}{C.GREEN}{total_impl}{C.RESET}")
        if total_miss:
            print(f"  Metodi mancanti       : {C.BOLD}{C.RED}{total_miss}{C.RESET}")
        else:
            print(f"  Metodi mancanti       : {C.BOLD}{C.GREEN}0  — tutto OK!{C.RESET}")
        print(f"{C.BOLD}{C.BLUE}{'═'*70}{C.RESET}\n")

    # ── Salva report su file (no ANSI) ────────
    def save_report(self, output_path: Path):
        lines: List[str] = []
        lines.append("=" * 70)
        lines.append("REPORT COERENZA HEADER vs IMPLEMENTAZIONI")
        lines.append("=" * 70)

        for class_name, cls in sorted(self.classes.items()):
            if not cls.methods_declared:
                continue
            lines.append(f"\n[{class_name}]")
            lines.append(f"  Header  : {cls.header_file}")
            lines.append(f"  Sorgente: {', '.join(sorted(cls.source_files)) or 'NON TROVATO'}")
            lines.append(f"  Dichiarati  ({len(cls.methods_declared)}): {', '.join(sorted(cls.methods_declared))}")
            lines.append(f"  Implementati({len(cls.methods_implemented & cls.methods_declared)}): "
                         f"{', '.join(sorted(cls.methods_implemented & cls.methods_declared))}")
            if cls.missing:
                lines.append(f"  !! MANCANTI: {', '.join(sorted(cls.missing))}")
            else:
                lines.append(f"  OK - tutti i metodi implementati")

        total_decl = sum(len(c.methods_declared) for c in self.classes.values())
        total_miss = sum(len(c.missing)          for c in self.classes.values())

        lines.append("\n" + "=" * 70)
        lines.append("RIEPILOGO")
        lines.append("=" * 70)
        lines.append(f"Classi          : {len(self.classes)}")
        lines.append(f"Metodi dichiarati: {total_decl}")
        lines.append(f"Metodi mancanti  : {total_miss}")

        output_path.write_text('\n'.join(lines), encoding='utf-8')
        print(f"  📄 Report salvato in: {C.CYAN}{output_path}{C.RESET}\n")

    # ── Entry point ───────────────────────────
    def run(self):
        print(f"\n{C.BOLD}📁 Root: {self.root}{C.RESET}")

        print(f"\n{C.CYAN}[1/2]{C.RESET} Scansione header in include/ ...")
        self.scan_headers()
        info(f"{len(self.classes)} classi trovate")

        print(f"\n{C.CYAN}[2/2]{C.RESET} Scansione implementazioni in src/ ...")
        self.scan_sources()

        self.print_report()
        self.save_report(self.root / "coherence_report.txt")


# ──────────────────────────────────────────────
def main():
    # Accetta root opzionale come argomento da CLI
    if len(sys.argv) > 1:
        root = sys.argv[1]
    else:
        root = Path(__file__).parent.parent  # default: due livelli sopra lo script

    Analyzer(str(root)).run()


if __name__ == "__main__":
    main()