#!/usr/bin/env python3
"""Run all packaged non-training self-tests in their declared environments."""
from __future__ import annotations
import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys


def load_routes(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return [(r["package_prefix"].strip().rstrip("/"), r["declared_environment"].strip().upper()) for r in rows]


def route_for(test: Path, root: Path, routes):
    rel = test.relative_to(root).as_posix()
    matches = [(prefix, route) for prefix, route in routes if rel == prefix or rel.startswith(prefix + "/")]
    if not matches:
        raise RuntimeError(f"No declared environment route for self-test: {rel}")
    return max(matches, key=lambda x: len(x[0]))[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--torch-python", required=True)
    ap.add_argument("--cluster-python", required=True)
    ap.add_argument("--routing-csv", default="environment/environment_routing.csv")
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    routes = load_routes(root / args.routing_csv)
    tests = sorted(p for p in (root / "code").rglob("*.py") if "tests" in p.parts)
    if not tests:
        raise RuntimeError("No packaged self-tests discovered.")

    print("=" * 100)
    print("PACKAGED SELF-TESTS")
    print("=" * 100)
    failures = []
    for test in tests:
        route = route_for(test, root, routes)
        python_bin = args.torch_python if route == "TORCH" else args.cluster_python
        env = os.environ.copy()
        env["ERP_PROJECT_ROOT"] = str(root)
        print(f"[{route}] {test.relative_to(root)}")
        proc = subprocess.run([python_bin, str(test)], cwd=root, env=env, text=True)
        if proc.returncode != 0:
            failures.append((test.relative_to(root).as_posix(), route, proc.returncode))

    print("=" * 100)
    print(f"Self-tests discovered : {len(tests)}")
    print(f"Self-tests failed     : {len(failures)}")
    if failures:
        for rel, route, rc in failures:
            print(f"FAIL {route} rc={rc}: {rel}")
        return 2
    print("Status                : PASS_PACKAGED_SELFTESTS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
