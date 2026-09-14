#!/usr/bin/env python3
"""Build the h=12 household load-range inputs used by the reported decomposition.

The producer reads the formal h=12 prediction files for the four strategies and four
seeds, verifies that strategy rows share the same household/timestamp/actual-load
support within each seed, attaches the fixed source-defined profile-group assignment,
and aggregates errors within the reported actual-load ranges.

No forecasting model is trained or modified here. The load ranges are post-hoc
interpretive summaries of the already generated h=12 test predictions.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

LEAD = 12
EXPECTED_METERS = 929
EXPECTED_GROUP_COUNTS = {1: 141, 2: 325, 3: 324, 4: 139}
SEEDS = [42, 123, 2026, 31415]
STRATEGIES = [
    "direct_transfer",
    "fine_tuning",
    "cer_scratch_limited",
    "cer_scratch_full",
]
BINS = [
    "actual_eq_0",
    "actual_gt_0_le_0_1",
    "actual_gt_0_1_le_0_5",
    "actual_gt_0_5_le_1_0",
    "actual_gt_1_0",
]

PEAK_QUANTILES = {
    "top_10pct": 0.90,
    "top_5pct": 0.95,
    "top_1pct": 0.99,
}
GROUP_DESCRIPTIONS = {
    1: "Morning–evening double peak",
    2: "Early evening peak",
    3: "Late evening peak",
    4: "Late night peak",
}


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def norm_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def subset_label(actual: pd.Series) -> pd.Series:
    values = pd.to_numeric(actual, errors="raise").to_numpy(float)
    out = np.empty(len(values), dtype=object)
    out[values == 0.0] = "actual_eq_0"
    out[(values > 0.0) & (values <= 0.1)] = "actual_gt_0_le_0_1"
    out[(values > 0.1) & (values <= 0.5)] = "actual_gt_0_1_le_0_5"
    out[(values > 0.5) & (values <= 1.0)] = "actual_gt_0_5_le_1_0"
    out[values > 1.0] = "actual_gt_1_0"
    ensure(pd.Series(out).isin(BINS).all(), "At least one actual-load row was not assigned to a reported range")
    return pd.Series(out, index=actual.index, dtype="object")


def prediction_path(root: Path, strategy: str, seed: int) -> Path:
    return root / "outputs/tables/pooled_strategy_evaluation_formal" / f"h{LEAD}" / strategy / f"seed_{seed}" / "test_predictions.parquet"


def load_assignments(root: Path) -> pd.DataFrame:
    path = root / "outputs/tables/target_profile_assignment_cer_assignment/target_profile_assignment_cer_source_defined_assignments.csv"
    ensure(path.is_file(), f"Missing fixed target profile assignment: {path}")
    frame = pd.read_csv(path, usecols=["meter_id", "assigned_group"])
    frame["meter_id"] = norm_id(frame["meter_id"])
    frame["assigned_group"] = pd.to_numeric(frame["assigned_group"], errors="raise").astype(int)
    ensure(len(frame) == EXPECTED_METERS and frame["meter_id"].nunique() == EXPECTED_METERS,
           "Target assignment must contain exactly 929 unique households")
    counts = frame["assigned_group"].value_counts().sort_index().to_dict()
    ensure(counts == EXPECTED_GROUP_COUNTS, f"Unexpected target group counts: {counts}")
    frame["group_description"] = frame["assigned_group"].map(GROUP_DESCRIPTIONS)
    return frame


def load_prediction(path: Path) -> pd.DataFrame:
    ensure(path.is_file(), f"Missing formal prediction file: {path}")
    columns = [
        "entity_id", "target_timestamp", "actual_kwh", "predicted_kwh",
        "absolute_error_kwh", "squared_error_kwh2", "smape_percent",
    ]
    frame = pd.read_parquet(path, columns=columns)
    missing = sorted(set(columns) - set(frame.columns))
    ensure(not missing, f"Prediction file missing columns {missing}: {path}")
    frame["entity_id"] = norm_id(frame["entity_id"])
    frame["target_timestamp"] = pd.to_datetime(frame["target_timestamp"], errors="raise")
    for column in columns[2:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    ensure(frame["entity_id"].nunique() == EXPECTED_METERS, f"Expected 929 households in {path}")
    ensure(np.isfinite(frame[["actual_kwh", "predicted_kwh", "absolute_error_kwh", "squared_error_kwh2", "smape_percent"]].to_numpy(float)).all(),
           f"Non-finite prediction values in {path}")
    ensure((frame["actual_kwh"] >= 0).all(), f"Negative actual load found in formal h=12 predictions: {path}")
    return frame


def verify_support(reference: pd.DataFrame, other: pd.DataFrame, label: str) -> None:
    keys = ["entity_id", "target_timestamp"]
    left = reference[keys + ["actual_kwh"]].sort_values(keys).reset_index(drop=True)
    right = other[keys + ["actual_kwh"]].sort_values(keys).reset_index(drop=True)
    ensure(len(left) == len(right), f"Prediction-row count differs across strategies for {label}")
    ensure(left[keys].equals(right[keys]), f"Household/timestamp support differs across strategies for {label}")
    ensure(np.allclose(left["actual_kwh"].to_numpy(float), right["actual_kwh"].to_numpy(float), rtol=0, atol=1e-7),
           f"Actual load differs across strategies for {label}")


def summarise_run(frame: pd.DataFrame, assignments: pd.DataFrame, strategy: str, seed: int) -> pd.DataFrame:
    work = frame.merge(assignments, left_on="entity_id", right_on="meter_id", how="inner", validate="many_to_one")
    ensure(len(work) == len(frame), f"Some prediction rows lacked fixed profile-group assignment for {strategy}/seed {seed}")
    work["subset"] = subset_label(work["actual_kwh"])

    def summarise(part: pd.DataFrame, subset: str, meter_meta: pd.Series) -> dict:
        n = len(part)
        abs_sum = float(part["absolute_error_kwh"].sum()) if n else 0.0
        sq_sum = float(part["squared_error_kwh2"].sum()) if n else 0.0
        smape_sum = float(part["smape_percent"].sum()) if n else 0.0
        return {
            "strategy": strategy,
            "seed": seed,
            "lead": LEAD,
            "meter_id": str(meter_meta["meter_id"]),
            "assigned_group": int(meter_meta["assigned_group"]),
            "group_description": str(meter_meta["group_description"]),
            "subset": subset,
            "observations": int(n),
            "mae_kwh": abs_sum / n if n else np.nan,
            "rmse_kwh": math.sqrt(sq_sum / n) if n else np.nan,
            "smape_percent": smape_sum / n if n else np.nan,
            "absolute_error_sum_kwh": abs_sum,
            "squared_error_sum_kwh2": sq_sum,
        }

    rows: list[dict] = []
    for _, meter_part in work.groupby("meter_id", sort=True):
        meta = meter_part.iloc[0]
        rows.append(summarise(meter_part, "all", meta))
        for raw_bin in BINS:
            rows.append(summarise(meter_part[meter_part["subset"].eq(raw_bin)], raw_bin, meta))
    return pd.DataFrame(rows)


def derive_peak_thresholds(reference: pd.DataFrame) -> dict[str, float]:
    """Derive the formal global h=12 peak thresholds once from common actual values."""
    values = pd.to_numeric(reference["actual_kwh"], errors="raise").to_numpy(dtype=float)
    ensure(np.isfinite(values).all(), "Non-finite actual values while deriving peak thresholds")

    thresholds = {
        label: float(np.quantile(values, q, method="linear"))
        for label, q in PEAK_QUANTILES.items()
    }

    ensure(
        thresholds["top_10pct"] <= thresholds["top_5pct"] <= thresholds["top_1pct"],
        f"Peak thresholds are not ordered: {thresholds}",
    )
    return thresholds


def build_meter_raw(
    reference: pd.DataFrame,
    assignments: pd.DataFrame,
    peak_thresholds: dict[str, float],
) -> pd.DataFrame:
    work = reference.merge(assignments, left_on="entity_id", right_on="meter_id", how="inner", validate="many_to_one")
    work["subset"] = subset_label(work["actual_kwh"])
    rows = []
    for meter_id, part in work.groupby("meter_id", sort=True):
        total = len(part)
        row = {
            "meter_id": str(meter_id),
            "assigned_group": int(part["assigned_group"].iloc[0]),
            "group_description": str(part["group_description"].iloc[0]),
            "all_observations": int(total),
            "test_actual_mean_kwh": float(part["actual_kwh"].mean()),
            "test_actual_std_kwh_ddof0": float(part["actual_kwh"].std(ddof=0)),
        }
        for raw_bin in BINS:
            n = int(part["subset"].eq(raw_bin).sum())
            row[f"{raw_bin}_observations"] = n
            row[f"{raw_bin}_row_share"] = n / total

        actual = part["actual_kwh"].to_numpy(dtype=float)
        for peak_name, threshold in peak_thresholds.items():
            n = int((actual >= threshold).sum())
            row[f"{peak_name}_observations"] = n
            row[f"{peak_name}_row_share"] = n / total

        rows.append(row)
    out = pd.DataFrame(rows)
    ensure(len(out) == EXPECTED_METERS, "Raw-load household summary must contain 929 rows")
    share_cols = [f"{b}_row_share" for b in BINS]
    ensure(
        np.allclose(out[share_cols].sum(axis=1), 1.0, atol=1e-12),
        "Reported load-range shares do not sum to one",
    )

    peak_share_cols = [
        "top_10pct_row_share",
        "top_5pct_row_share",
        "top_1pct_row_share",
    ]
    ensure(
        ((out[peak_share_cols] >= 0.0) & (out[peak_share_cols] <= 1.0)).all().all(),
        "Peak row shares must lie in [0, 1]",
    )
    ensure(
        (
            (out["top_1pct_row_share"] <= out["top_5pct_row_share"])
            & (out["top_5pct_row_share"] <= out["top_10pct_row_share"])
        ).all(),
        "Peak row shares are not properly nested",
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    assignments = load_assignments(root)

    all_summaries: list[pd.DataFrame] = []
    reference_by_seed: dict[int, pd.DataFrame] = {}
    for seed in SEEDS:
        reference = load_prediction(prediction_path(root, STRATEGIES[0], seed))
        reference_by_seed[seed] = reference
        for strategy in STRATEGIES:
            frame = reference if strategy == STRATEGIES[0] else load_prediction(prediction_path(root, strategy, seed))
            if strategy != STRATEGIES[0]:
                verify_support(reference, frame, f"seed={seed}, strategy={strategy}")
            all_summaries.append(summarise_run(frame, assignments, strategy, seed))

    all_runs = pd.concat(all_summaries, ignore_index=True)
    expected_rows = len(STRATEGIES) * len(SEEDS) * EXPECTED_METERS * (len(BINS) + 1)
    ensure(len(all_runs) == expected_rows, f"Expected {expected_rows:,} aggregated rows; found {len(all_runs):,}")
    ensure(not all_runs.duplicated(["strategy", "seed", "meter_id", "subset"]).any(), "Duplicate aggregated load-range rows")

    # The actual-load composition is identical across formal seeds by construction.
    # Derive the global peak thresholds once from the common formal h=12 actual values.
    peak_thresholds = derive_peak_thresholds(reference_by_seed[SEEDS[0]])

    raw_frames = [
        build_meter_raw(reference_by_seed[seed], assignments, peak_thresholds)
        for seed in SEEDS
    ]
    baseline = raw_frames[0].sort_values("meter_id").reset_index(drop=True)
    comparison_columns = [
        "meter_id",
        "assigned_group",
        "all_observations",
        *[f"{b}_observations" for b in BINS],
        *[f"{p}_observations" for p in PEAK_QUANTILES],
    ]
    for seed, frame in zip(SEEDS[1:], raw_frames[1:]):
        check = frame.sort_values("meter_id").reset_index(drop=True)
        ensure(baseline[comparison_columns].equals(check[comparison_columns]),
               f"Actual-load composition differs across formal seeds; first mismatch at seed {seed}")

    out_dir = root / "outputs/tables/raw_load_structure_h12_profile_group_peak_decomposition"
    meta_dir = root / "outputs/metadata/raw_load_structure_h12_profile_group_peak_decomposition"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    all_path = out_dir / "raw_load_structure_h12_meter_subset_strategy_metrics_all_runs.parquet"
    raw_path = out_dir / "raw_load_structure_h12_meter_raw_load_structure.csv"
    all_runs.to_parquet(all_path, index=False)
    baseline.to_csv(raw_path, index=False)

    status = {
        "analysis_id": "raw_load_structure",
        "status": "COMPLETE_PASS",
        "lead": LEAD,
        "households": EXPECTED_METERS,
        "strategies": STRATEGIES,
        "seeds": SEEDS,
        "descriptive_load_ranges": BINS,
        "peak_thresholds_kwh": peak_thresholds,
        "aggregated_rows": int(len(all_runs)),
        "training_performed": False,
        "predictions_modified": False,
        "profile_groups_modified": False,
    }
    (meta_dir / "raw_load_structure_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print("RAW LOAD STRUCTURE: PASS")
    print(f"Households: {EXPECTED_METERS}")
    print(f"Aggregated rows: {len(all_runs):,}")
    print(f"Tables: {out_dir}")


if __name__ == "__main__":
    main()
