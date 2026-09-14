#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve()
MODULE = HERE.parents[1] / "src/group_heterogeneity_group_heterogeneity_practical_significance.py"
spec = importlib.util.spec_from_file_location("group_heterogeneity", MODULE)
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

# Four perfectly separated groups must have eta-squared close to one.
y = np.repeat([0.0, 1.0, 2.0, 3.0], 10)
g = np.repeat([1, 2, 3, 4], 10)
e = mod.one_way_effect(y, g)
assert e["eta_squared"] > 0.99

# Identical group distributions must have zero between-group variation.
y2 = np.tile(np.arange(10, dtype=float), 4)
g2 = np.repeat([1, 2, 3, 4], 10)
e2 = mod.one_way_effect(y2, g2)
assert abs(e2["eta_squared"]) < 1e-12

q = mod.bh_adjust([0.01, 0.04, 0.03, 0.20])
assert np.all((q >= 0) & (q <= 1))
assert q[0] <= q[1]

print("GROUP HETEROGENEITY SELFTEST: PASS")
