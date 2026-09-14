#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import subprocess
import sys
from pathlib import Path

STDLIB = set(getattr(sys, "stdlib_module_names", set()))

def package_files(package: Path):
    files = []
    for sub in ("src", "tests"):
        d = package / sub
        if d.is_dir():
            files.extend(sorted(d.rglob("*.py")))
    # Stage-level workflow directories (for example data preparation and
    # report generation) intentionally do not use a src/tests package layout.
    # Audit every Python file there rather than silently skipping the route.
    if not files:
        files = sorted(package.rglob("*.py"))
    return files

def imports_for(files):
    local_modules = {p.stem for p in files}
    imports = set()
    for p in files:
        try:
            tree = ast.parse(
                p.read_text(encoding="utf-8", errors="replace"),
                filename=str(p),
            )
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    imports.add(node.module.split(".")[0])

    return sorted(
        x for x in imports
        if x not in STDLIB
        and x not in local_modules
        and not x.startswith("_")
    )

def availability(python_bin: str, modules):
    code = """
import importlib.util, json, sys
mods = json.loads(sys.argv[1])
out = {}
for m in mods:
    try:
        out[m] = importlib.util.find_spec(m) is not None
    except Exception:
        out[m] = False
print(json.dumps(out))
"""
    p = subprocess.run(
        [python_bin, "-c", code, json.dumps(modules)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(p.stdout)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--torch-python", required=True)
    ap.add_argument("--cluster-python", required=True)
    ap.add_argument(
        "--routing-csv",
        default="environment/environment_routing.csv",
    )
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    routing_path = root / args.routing_csv

    if not routing_path.is_file():
        raise FileNotFoundError(routing_path)

    with routing_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    failures = []
    checked = 0

    print("=" * 110)
    print("PUBLIC ENVIRONMENT ROUTING VALIDATION")
    print("=" * 110)

    for row in rows:
        prefix = row["package_prefix"].strip()
        route = row["declared_environment"].strip().upper()
        package = root / prefix

        # Some prefixes such as code/04_forecasting are not package-style
        # src/tests modules. They are covered by explicit workflow preflight.
        if not package.exists():
            failures.append((prefix, route, ["PACKAGE_PATH_MISSING"]))
            continue

        files = package_files(package)
        if not files:
            print(f"{prefix}")
            print(f"  route   : {route}")
            print("  imports : explicit workflow stage; no package src/tests audit")
            print("  status  : SKIP_STATIC_IMPORT_AUDIT")
            print()
            continue

        modules = imports_for(files)
        python_bin = (
            args.torch_python if route == "TORCH"
            else args.cluster_python
        )
        avail = availability(python_bin, modules)
        missing = [m for m in modules if not avail.get(m, False)]

        print(prefix)
        print(f"  route   : {route}")
        print(f"  python  : {python_bin}")
        print(f"  imports : {', '.join(modules) if modules else '(stdlib/local only)'}")
        print(f"  missing : {', '.join(missing) if missing else 'NONE'}")
        print(f"  status  : {'FAIL' if missing else 'PASS'}")
        print()

        checked += 1
        if missing:
            failures.append((prefix, route, missing))

    print("=" * 110)
    print(f"Packages statically audited : {checked}")
    print(f"Routing failures            : {len(failures)}")

    if failures:
        print("\nFAILED ROUTES:")
        for prefix, route, missing in failures:
            print(f"- {prefix} -> {route}: {', '.join(missing)}")
        sys.exit(2)

    print("Status                      : PASS_ENVIRONMENT_ROUTING")

if __name__ == "__main__":
    main()
