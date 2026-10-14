#!/usr/bin/env python3
"""
Elenca tutti i file in include/, src/, pybinding/ (ricorsivamente)
e li salva in un unico file di testo.
Uso: python list_test_files.py [root_dir]
"""

import sys
from pathlib import Path
from datetime import datetime

DIRS_TO_SCAN = ["pybinding"]

def main():
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).parent.parent.resolve()
    output_file = root / "file_test_list.txt"

    lines = []
    lines.append(f"FILE LIST — {root}")
    lines.append(f"Generato il: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)

    total = 0

    for dir_name in DIRS_TO_SCAN:
        target = root / dir_name
        if not target.exists():
            lines.append(f"\n[{dir_name}/]  ← directory non trovata, saltata")
            continue

        files = sorted(target.rglob("*"))
        files = [f for f in files if f.is_file()]

        lines.append(f"\n[{dir_name}/]  ({len(files)} file)")
        lines.append("-" * 70)

        for f in files:
            rel = f.relative_to(root)
            lines.append(f"\n  ┌─ {rel}")
            lines.append(f"  │  {f}".replace(str(root), ""))
            lines.append("  ├" + "─" * 67)
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
                for line in content.splitlines():
                    lines.append(f"  │  {line}")
            except Exception as e:
                lines.append(f"  │  [errore lettura: {e}]")
            lines.append("  └" + "─" * 67)

        total += len(files)

    lines.append("\n" + "=" * 70)
    lines.append(f"TOTALE: {total} file in {len(DIRS_TO_SCAN)} directory")

    output_file.write_text("\n".join(lines), encoding="utf-8")

    print(f"✔  {total} file trovati")
    print(f"✔  Salvato in: {output_file}")

if __name__ == "__main__":
    main()