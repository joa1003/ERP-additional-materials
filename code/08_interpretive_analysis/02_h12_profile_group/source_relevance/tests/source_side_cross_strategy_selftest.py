#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src' / 'source_side_cross_strategy_appendix_source_side_cross_strategy_diagnostics.py'
spec = importlib.util.spec_from_file_location('source_side_cross_strategymod', SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

x = np.array([1,2,3,4,5,6,7,8], float)
y = np.array([8,7,6,5,4,3,2,1], float)
g = np.array([1,1,1,1,2,2,2,2])
assert abs(stats.spearmanr(x,y).statistic + 1) < 1e-12
assert np.isfinite(stats.pearsonr(mod.rank_residual(x,g), mod.rank_residual(y,g)).statistic)

p = np.array([0.001, 0.01, 0.03, 0.2])
q = mod.bh_adjust(p)
assert np.all(q >= p - 1e-15)
assert np.all((q >= 0) & (q <= 1))

r = stats.rankdata(x, method='average')
rc = mod.centered_rank(x, g)
for grp in np.unique(g):
    assert abs(rc[g == grp].mean()) < 1e-12
assert abs(mod.correlation_fast(r, stats.rankdata(y, method='average')) + 1) < 1e-12

print('SOURCE_SIDE_CROSS_STRATEGY SYNTHETIC SELFTEST: PASS')
