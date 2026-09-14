#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

TARGET = "_ERP_PROJECT_ROOT"


def states_from_text(text: str, filename: str):
    tree = ast.parse(text, filename=filename)
    loads, stores = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == TARGET:
            if isinstance(node.ctx, ast.Load):
                loads.append(getattr(node, "lineno", -1))
            elif isinstance(node.ctx, (ast.Store, ast.Del)):
                stores.append(getattr(node, "lineno", -1))
    return loads, stores


def notebook_source(path: Path) -> str:
    nb = json.loads(path.read_text(encoding="utf-8"))
    return "\n\n".join(
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    code_root = root / "code"
    failures = []
    checked = 0

    for path in sorted(code_root.rglob("*.py")):
        loads, stores = states_from_text(path.read_text(encoding="utf-8"), str(path))
        if loads:
            checked += 1
            if not stores:
                failures.append((path.relative_to(root), loads))

    for path in sorted(code_root.rglob("*.ipynb")):
        text = notebook_source(path)
        loads, stores = states_from_text(text, str(path))
        if loads:
            checked += 1
            if not stores:
                failures.append((path.relative_to(root), loads))

    print("=" * 100)
    print("ACTIVE-CODE PORTABILITY VALIDATION")
    print("=" * 100)
    print(f"Files/notebooks using project-root helper : {checked}")
    print(f"Undefined active items                   : {len(failures)}")

    if failures:
        for rel, lines in failures:
            print(f"- {rel}: use line(s) {lines}")
        raise SystemExit(2)

    print("Status                                   : PASS_PORTABILITY_ALL_CODE")


if __name__ == "__main__":
    main()
