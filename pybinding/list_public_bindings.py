#!/usr/bin/env python3
"""Esporta in JSON l'API pubblica effettivamente esposta da ``ml_core``.

Uso:
    python pybinding/list_public_bindings.py
    python pybinding/list_public_bindings.py --output public_bindings.json

Il modulo deve essere installato oppure presente in build/pybinding o
build/lib rispetto alla radice del repository.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def load_ml_core() -> ModuleType:
    """Importa ml_core anche quando lo script è eseguito dalla repository."""
    project_root = Path(__file__).resolve().parents[1]
    for candidate in (project_root / "build" / "pybinding", project_root / "build" / "lib"):
        if candidate.is_dir():
            sys.path.insert(0, str(candidate))
    return importlib.import_module("ml_core")


def describe_callable(value: Any) -> dict[str, str]:
    """Restituisce nome e documentazione di una funzione/metodo binding."""
    return {
        "kind": "callable",
        "doc": inspect.getdoc(value) or "",
    }


def describe_class(value: type[Any]) -> dict[str, Any]:
    """Elenca API pubblica, inclusi metodi ereditati, enum e proprietà."""
    members: dict[str, dict[str, str]] = {}
    for name, member in inspect.getmembers(value):
        if name.startswith("_"):
            continue

        if callable(member):
            members[name] = describe_callable(member)
        elif isinstance(member, property):
            members[name] = {
                "kind": "property",
                "doc": inspect.getdoc(member) or "",
            }
        else:
            members[name] = {
                "kind": "attribute",
                "type": type(member).__name__,
                "doc": inspect.getdoc(member) or "",
            }

    result: dict[str, Any] = {
        "kind": "class",
        "doc": inspect.getdoc(value) or "",
        "members": members,
    }
    enum_members = getattr(value, "__members__", None)
    if enum_members is not None:
        result["enum_values"] = list(enum_members)
    return result


def build_inventory(module: ModuleType) -> dict[str, Any]:
    """Crea una descrizione JSON dei nomi pubblici definiti nel modulo."""
    exports: dict[str, dict[str, Any]] = {}
    for name, value in inspect.getmembers(module):
        if name.startswith("_"):
            continue

        if inspect.isclass(value):
            exports[name] = describe_class(value)
        elif callable(value):
            exports[name] = describe_callable(value)
        else:
            exports[name] = {
                "kind": "value",
                "type": type(value).__name__,
                "value": repr(value),
            }

    return {
        "module": module.__name__,
        "version": getattr(module, "__version__", None),
        "exports": exports,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="salva l'inventario JSON nel file indicato invece di stamparlo",
    )
    return parser.parse_args()


def main() -> None:
    inventory = build_inventory(load_ml_core())
    rendered = json.dumps(inventory, indent=2, ensure_ascii=False) + "\n"
    args = parse_args()
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Inventario salvato in {args.output}")


if __name__ == "__main__":
    main()
