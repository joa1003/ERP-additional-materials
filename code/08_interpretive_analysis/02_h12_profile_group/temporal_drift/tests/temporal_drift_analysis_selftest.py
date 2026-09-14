#!/usr/bin/env python3
import numpy as np


def safe_relative(absolute, comparator):
    absolute = np.asarray(absolute, float)
    comparator = np.asarray(comparator, float)
    out = np.full(np.broadcast_shapes(absolute.shape, comparator.shape), np.nan, dtype=float)
    np.divide(100.0 * absolute, comparator, out=out, where=np.abs(comparator) > 1e-12)
    return out


def bh(values):
    p = np.asarray(values, float)
    n = len(p)
    order = np.argsort(p)
    out = np.empty(n)
    previous = 1.0
    for rank, idx in reversed(list(enumerate(order, start=1))):
        previous = min(previous, p[idx] * n / rank)
        out[idx] = previous
    return np.clip(out, 0, 1)

D = np.array([2.0, 4.0])
T = np.array([1.5, 4.5])
L = np.array([3.0, 5.0])
F = np.array([1.0, 3.0])
source = L - D
fine = D - T
full = L - F
direct_full = F - D
fine_limited = L - T
full_fine = T - F
assert np.allclose(fine_limited, source + fine)
assert np.allclose(direct_full, source - full)
assert np.allclose(full_fine, full - fine_limited)
assert np.allclose(safe_relative(source, L), [100 / 3, 20])
assert np.isnan(safe_relative([1.0], [0.0]))[0]
q = bh([0.01, 0.02, 0.5])
assert len(q) == 3 and np.all((q >= 0) & (q <= 1))
print("TEMPORAL DRIFT ANALYSIS SELFTEST: PASS")
