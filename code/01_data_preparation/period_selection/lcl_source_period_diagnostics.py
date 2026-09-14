"""Reproduce Appendix A.1 source-period coverage diagnostics from raw LCL data."""
from pathlib import Path
import os
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("ERP_PROJECT_ROOT", ".")).expanduser().resolve()
RAW = ROOT / "data/raw/LCL/csv/data_collection/data_tables/consumption_n.csv"
OUT = ROOT / "outputs/tables/period_selection/lcl_candidate_period_comparison.csv"

PERIODS = [
    ("2011 Nov-Dec", "2011-11-23 09:00", "2011-12-31 23:30"),
    ("2012 Full Year", "2012-01-01 00:00", "2012-12-31 23:30"),
    ("2013 Full Year", "2013-01-01 00:00", "2013-12-31 23:30"),
    ("2012-2013 Combined", "2012-01-01 00:00", "2013-12-31 23:30"),
    ("2014 (Jan-Feb)", "2014-01-01 00:00", "2014-02-28 00:00"),
    ("2013-2014 Combined", "2013-01-01 00:00", "2014-02-28 00:00"),
]
EXPECTED_TIMESTAMPS = [1854, 17568, 17520, 35088, 2785, 20305]


def main() -> None:
    if not RAW.exists():
        raise FileNotFoundError(f"Missing raw LCL file: {RAW}")

    header = pd.read_csv(RAW, nrows=0)
    households = [c for c in header.columns if c != "GMT"]
    if len(households) != 4173:
        raise AssertionError(f"Expected 4,173 LCL household columns; found {len(households):,}.")

    timestamps = pd.to_datetime(pd.read_csv(RAW, usecols=["GMT"])["GMT"], errors="raise")
    masks = []
    for (label, start, end), expected in zip(PERIODS, EXPECTED_TIMESTAMPS):
        mask = (timestamps >= pd.Timestamp(start)) & (timestamps <= pd.Timestamp(end))
        if int(mask.sum()) != expected:
            raise AssertionError(f"{label}: expected {expected:,} timestamps, found {int(mask.sum()):,}.")
        masks.append(mask.to_numpy())

    valid_counts = [np.zeros(len(households), dtype=np.int64) for _ in PERIODS]
    offset = 0
    for chunk in pd.read_csv(RAW, usecols=["GMT"] + households, chunksize=2000, low_memory=False):
        n = len(chunk)
        block = chunk[households].apply(pd.to_numeric, errors="coerce")
        finite = block.notna().to_numpy()
        for i, mask in enumerate(masks):
            local = mask[offset:offset+n]
            if local.any():
                valid_counts[i] += finite[local].sum(axis=0)
        offset += n

    rows = []
    for i, ((label, start, end), expected) in enumerate(zip(PERIODS, EXPECTED_TIMESTAMPS)):
        coverage = valid_counts[i] / expected
        rows.append({
            "candidate_window": label,
            "start": start,
            "end": end,
            "timestamps": expected,
            "mean_coverage_pct": 100 * float(np.mean(coverage)),
            "median_coverage_pct": 100 * float(np.median(coverage)),
            "p25_coverage_pct": 100 * float(np.quantile(coverage, 0.25)),
            "p75_coverage_pct": 100 * float(np.quantile(coverage, 0.75)),
            "minimum_coverage_pct": 100 * float(np.min(coverage)),
            "households_ge_90pct": int((coverage >= 0.90).sum()),
            "households_lt_50pct": int((coverage < 0.50).sum()),
            "overall_missing_rate_pct": 100 * (1 - float(valid_counts[i].sum()) / (expected * len(households))),
        })

    result = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False)
    print(result.to_string(index=False))
    print(f"\nWrote: {OUT}")


if __name__ == "__main__":
    main()
