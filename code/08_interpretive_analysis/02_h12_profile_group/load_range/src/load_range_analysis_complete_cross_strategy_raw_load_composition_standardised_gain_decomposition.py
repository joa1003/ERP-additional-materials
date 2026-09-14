#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
import os as _erp_os
_ERP_PROJECT_ROOT = Path(_erp_os.environ.get("ERP_PROJECT_ROOT", ".")).expanduser().resolve()
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ANALYSIS_ID = "load_range_analysis"
ANALYSIS_SLUG = "load_range_analysis_complete_cross_strategy_raw_load_composition_standardised_gain_decomposition"
EXPECTED_METERS = 929
EXPECTED_GROUP_COUNTS = {1: 141, 2: 325, 3: 324, 4: 139}
FORMAL_SEEDS = [42, 123, 2026, 31415]
LEAD = 12
RANDOM_SEED = 20260728
PERMUTATIONS = 1999
BOOTSTRAPS = 1000
TOLERANCE = 1e-9
REPRODUCTION_TOLERANCE = {"mae": 1e-6, "rmse": 1e-6, "smape": 1e-4}

GROUP_DESCRIPTIONS = {
    1: "Morning–evening double peak",
    2: "Early evening peak",
    3: "Late evening peak",
    4: "Late night peak",
}

STRATEGIES = [
    "direct_transfer",
    "fine_tuning",
    "cer_scratch_limited",
    "cer_scratch_full",
]
STRATEGY_LABELS = {
    "direct_transfer": "Direct Transfer",
    "fine_tuning": "Fine Tuning",
    "cer_scratch_limited": "CER Scratch Limited",
    "cer_scratch_full": "CER Scratch Full",
}

# Gain is comparator error minus focal-strategy error. Positive means focal is better.
COMPARISONS = [
    {
        "comparison_id": "source_transfer",
        "comparison_label": "Source Transfer Gain",
        "focal_strategy": "direct_transfer",
        "comparator_strategy": "cer_scratch_limited",
        "interpretive_role": "primary_lock_controlled",
    },
    {
        "comparison_id": "fine_tuning",
        "comparison_label": "Fine-Tuning Gain",
        "focal_strategy": "fine_tuning",
        "comparator_strategy": "direct_transfer",
        "interpretive_role": "primary_lock_controlled",
    },
    {
        "comparison_id": "full_data",
        "comparison_label": "Full-Data Gain",
        "focal_strategy": "cer_scratch_full",
        "comparator_strategy": "cer_scratch_limited",
        "interpretive_role": "primary_lock_controlled",
    },
    {
        "comparison_id": "direct_vs_full",
        "comparison_label": "Direct-versus-Full Gain",
        "focal_strategy": "direct_transfer",
        "comparator_strategy": "cer_scratch_full",
        "interpretive_role": "primary_lock_controlled",
    },
    {
        "comparison_id": "fine_tuning_vs_limited",
        "comparison_label": "Fine-Tuning-versus-Limited Gain",
        "focal_strategy": "fine_tuning",
        "comparator_strategy": "cer_scratch_limited",
        "interpretive_role": "supplementary_strategy_closure",
    },
    {
        "comparison_id": "full_vs_fine_tuning",
        "comparison_label": "Full-versus-Fine-Tuning Gain",
        "focal_strategy": "cer_scratch_full",
        "comparator_strategy": "fine_tuning",
        "interpretive_role": "supplementary_strategy_closure",
    },
]
COMPARISON_BY_ID = {x["comparison_id"]: x for x in COMPARISONS}

DESCRIPTIVE_BINS = [
    "actual_eq_0",
    "actual_gt_0_le_0_1",
    "actual_gt_0_1_le_0_5",
    "actual_gt_0_5_le_1_0",
    "actual_gt_1_0",
]
DESCRIPTIVE_LABELS = {
    "actual_eq_0": "Actual = 0",
    "actual_gt_0_le_0_1": "0 < actual ≤ 0.1",
    "actual_gt_0_1_le_0_5": "0.1 < actual ≤ 0.5",
    "actual_gt_0_5_le_1_0": "0.5 < actual ≤ 1.0",
    "actual_gt_1_0": "Actual > 1.0",
}
PRIMARY_BINS = [
    "actual_le_0_1",
    "actual_gt_0_1_le_0_5",
    "actual_gt_0_5_le_1_0",
    "actual_gt_1_0",
]
PRIMARY_LABELS = {
    "actual_le_0_1": "Actual ≤ 0.1",
    "actual_gt_0_1_le_0_5": "0.1 < actual ≤ 0.5",
    "actual_gt_0_5_le_1_0": "0.5 < actual ≤ 1.0",
    "actual_gt_1_0": "Actual > 1.0",
}
ADDITIVE_METRICS = {
    "mae": {"unit": "kWh", "strategy_column": "mae_kwh"},
    "mse": {"unit": "kWh²", "strategy_column": "mse_kwh2"},
    "smape": {"unit": "percentage points", "strategy_column": "smape_percent"},
}
FORMAL_METRICS = {
    "mae": {"unit": "kWh"},
    "rmse": {"unit": "kWh"},
    "smape": {"unit": "percentage points"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load range analysis complete cross-strategy raw-load composition-standardised gain decomposition"
    )
    parser.add_argument("--root", type=Path, default=Path(str(_ERP_PROJECT_ROOT)))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--input-all-runs", type=Path, default=None)
    parser.add_argument("--input-meter-raw", type=Path, default=None)
    parser.add_argument("--input-profile_group_analysis", type=Path, default=None)
    parser.add_argument("--bootstraps", type=int, default=BOOTSTRAPS)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    return parser.parse_args()


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def normalise_meter_id(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="raise").astype(int)


def load_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def input_paths(
    root: Path,
    override_all_runs: Path | None = None,
    override_meter_raw: Path | None = None,
    override_profile_group_analysis: Path | None = None,
) -> dict[str, Path]:
    return {
        "canonical_lock": root / "documentation/reproducibility_design_contract.md",
        "all_runs": override_all_runs or root / "outputs/tables/raw_load_structure_h12_profile_group_peak_decomposition/raw_load_structure_h12_meter_subset_strategy_metrics_all_runs.parquet",
        "meter_raw": override_meter_raw or root / "outputs/tables/raw_load_structure_h12_profile_group_peak_decomposition/raw_load_structure_h12_meter_raw_load_structure.csv",
        "profile_group_analysis_gains": override_profile_group_analysis or root / "outputs/tables/profile_group_analysis_profile_group_analysis/profile_group_analysis_native_meter_gains_with_group.csv",
    }


def required_profile_group_analysis_columns() -> set[str]:
    return {
        "entity_id", "lead", "criterion", "assigned_group", "group_description", "seed_count",
        "source_transfer_gain_mean", "fine_tuning_gain_mean", "full_data_gain_mean", "direct_vs_full_gain_mean",
        "direct_transfer_value_mean", "fine_tuning_value_mean", "cer_scratch_limited_value_mean", "cer_scratch_full_value_mean",
    }


def check_inputs(
    root: Path,
    override_all_runs: Path | None = None,
    override_meter_raw: Path | None = None,
    override_profile_group_analysis: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    paths = input_paths(root, override_all_runs, override_meter_raw, override_profile_group_analysis)
    for name, path in paths.items():
        ensure(path.is_file(), f"Missing {name} input: {path}")

    raw = pd.read_csv(paths["meter_raw"])
    raw["meter_id"] = normalise_meter_id(raw["meter_id"])
    ensure(len(raw) == EXPECTED_METERS, f"Raw load structure raw structure must contain {EXPECTED_METERS} rows")
    ensure(raw["meter_id"].nunique() == EXPECTED_METERS, "Raw load structure raw structure meter IDs are not unique")
    counts = raw["assigned_group"].value_counts().sort_index().to_dict()
    ensure(counts == EXPECTED_GROUP_COUNTS, f"Unexpected group counts: {counts}")
    for raw_bin in DESCRIPTIVE_BINS:
        ensure(
            {f"{raw_bin}_observations", f"{raw_bin}_row_share"}.issubset(raw.columns),
            f"Missing raw-bin columns for {raw_bin}",
        )
    ensure("all_observations" in raw.columns, "Raw load structure raw structure is missing all_observations")
    five_share_columns = [f"{raw_bin}_row_share" for raw_bin in DESCRIPTIVE_BINS]
    ensure(
        np.allclose(raw[five_share_columns].sum(axis=1), 1.0, atol=1e-12),
        "Five descriptive raw-bin shares do not sum to one per meter",
    )

    gains = pd.read_csv(paths["profile_group_analysis_gains"])
    ensure(required_profile_group_analysis_columns().issubset(gains.columns), f"Profile group analysis input missing columns: {sorted(required_profile_group_analysis_columns() - set(gains.columns))}")
    gains["entity_id"] = normalise_meter_id(gains["entity_id"])
    ensure(len(gains) == EXPECTED_METERS * 3 * 3, f"Expected 8,361 Profile group analysis rows, found {len(gains):,}")
    ensure(set(gains["lead"]) == {1, 12, 48}, f"Profile group analysis lead set mismatch: {sorted(gains['lead'].unique())}")
    ensure(set(gains["criterion"]) == {"mae", "rmse", "smape"}, "Profile group analysis criterion set mismatch")
    h12 = gains[gains["lead"].eq(LEAD)].copy()
    ensure(len(h12) == EXPECTED_METERS * 3, f"Expected 2,787 h=12 Profile group analysis rows, found {len(h12):,}")
    ensure(set(h12["entity_id"]) == set(raw["meter_id"]), "Profile group analysis and Raw load structure meter sets differ")
    ensure(h12["seed_count"].eq(4).all(), "Profile group analysis h=12 rows do not all contain four seeds")

    all_runs = load_frame(paths["all_runs"])
    all_runs["meter_id"] = normalise_meter_id(all_runs["meter_id"])
    required_all_runs = {
        "strategy", "seed", "lead", "meter_id", "subset", "observations",
        "mae_kwh", "rmse_kwh", "smape_percent", "absolute_error_sum_kwh", "squared_error_sum_kwh2",
    }
    ensure(required_all_runs.issubset(all_runs.columns), f"Raw load structure all-runs input missing columns: {sorted(required_all_runs - set(all_runs.columns))}")
    selected = all_runs[
        all_runs["strategy"].isin(STRATEGIES)
        & all_runs["seed"].isin(FORMAL_SEEDS)
        & all_runs["lead"].eq(LEAD)
        & all_runs["subset"].isin(DESCRIPTIVE_BINS + ["all"])
    ].copy()
    expected_rows = len(STRATEGIES) * len(FORMAL_SEEDS) * EXPECTED_METERS * (len(DESCRIPTIVE_BINS) + 1)
    ensure(len(selected) == expected_rows, f"Expected {expected_rows:,} selected all-runs rows, found {len(selected):,}")
    ensure(set(selected["strategy"]) == set(STRATEGIES), f"Strategy set mismatch: {sorted(selected['strategy'].unique())}")
    ensure(set(selected["seed"]) == set(FORMAL_SEEDS), f"Seed set mismatch: {sorted(selected['seed'].unique())}")
    ensure(set(selected["meter_id"]) == set(raw["meter_id"]), "Raw load structure all-runs meter set differs")
    key_columns = ["strategy", "seed", "meter_id", "subset"]
    ensure(not selected.duplicated(key_columns).any(), "Duplicate strategy-seed-meter-subset rows exist")
    complete_cells = selected.groupby(["strategy", "seed", "subset"])["meter_id"].nunique()
    ensure(complete_cells.eq(EXPECTED_METERS).all(), "Incomplete strategy-seed-subset meter coverage")

    observation_check = selected.pivot_table(
        index=["seed", "meter_id", "subset"], columns="strategy", values="observations", aggfunc="first"
    )
    ensure(not observation_check.isna().any().any(), "Missing strategy cells in observation-alignment audit")
    ensure((observation_check.max(axis=1) == observation_check.min(axis=1)).all(), "Observation counts differ across strategies")

    return raw, h12, selected, paths


def weighted(values: Iterable[float], weights: Iterable[float]) -> float:
    values_array = np.asarray(list(values), dtype=float)
    weights_array = np.asarray(list(weights), dtype=float)
    mask = np.isfinite(values_array) & np.isfinite(weights_array) & (weights_array > 0)
    if not mask.any():
        return math.nan
    return float(np.sum(values_array[mask] * weights_array[mask]) / np.sum(weights_array[mask]))


def safe_relative_gain(gain: float | np.ndarray | pd.Series, comparator_error: float | np.ndarray | pd.Series):
    gain_array = np.asarray(gain, dtype=float)
    denominator = np.asarray(comparator_error, dtype=float)
    result = np.full(np.broadcast_shapes(gain_array.shape, denominator.shape), np.nan, dtype=float)
    gain_b = np.broadcast_to(gain_array, result.shape)
    denominator_b = np.broadcast_to(denominator, result.shape)
    valid = np.isfinite(gain_b) & np.isfinite(denominator_b) & (denominator_b != 0)
    result[valid] = 100.0 * gain_b[valid] / denominator_b[valid]
    if result.ndim == 0:
        return float(result)
    return result


def summary_stats(values: pd.Series) -> dict[str, float | int]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(x) == 0:
        return {
            "meters": 0, "mean": math.nan, "sample_std": math.nan, "median": math.nan,
            "q25": math.nan, "q75": math.nan, "minimum": math.nan, "maximum": math.nan,
        }
    return {
        "meters": int(len(x)),
        "mean": float(np.mean(x)),
        "sample_std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "median": float(np.median(x)),
        "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)),
        "minimum": float(np.min(x)),
        "maximum": float(np.max(x)),
    }


def eta_omega(values: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    grand = values.mean()
    ss_total = float(np.sum((values - grand) ** 2))
    ss_between = 0.0
    ss_within = 0.0
    unique_groups = np.unique(groups)
    for group in unique_groups:
        x = values[groups == group]
        ss_between += len(x) * (x.mean() - grand) ** 2
        ss_within += float(np.sum((x - x.mean()) ** 2))
    eta = ss_between / ss_total if ss_total > 0 else 0.0
    df_between = len(unique_groups) - 1
    df_within = len(values) - len(unique_groups)
    ms_within = ss_within / df_within if df_within > 0 else 0.0
    denominator = ss_total + ms_within
    omega = (ss_between - df_between * ms_within) / denominator if denominator > 0 else 0.0
    return float(eta), float(max(0.0, omega))


def permutation_p(values: np.ndarray, groups: np.ndarray, permutations: int, rng: np.random.Generator) -> float:
    observed = eta_omega(values, groups)[0]
    exceed = 0
    for _ in range(permutations):
        if eta_omega(values, rng.permutation(groups))[0] >= observed - 1e-15:
            exceed += 1
    return float((exceed + 1) / (permutations + 1))


def bh_adjust(p_values: pd.Series) -> pd.Series:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    output = np.empty(len(p), dtype=float)
    previous = 1.0
    for rank_index in range(len(p) - 1, -1, -1):
        index = order[rank_index]
        rank = rank_index + 1
        value = min(previous, p[index] * len(p) / rank)
        output[index] = value
        previous = value
    return pd.Series(output, index=p_values.index)


def combine_primary_bins(all_runs: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    descriptive_to_primary = {
        "actual_eq_0": "actual_le_0_1",
        "actual_gt_0_le_0_1": "actual_le_0_1",
        "actual_gt_0_1_le_0_5": "actual_gt_0_1_le_0_5",
        "actual_gt_0_5_le_1_0": "actual_gt_0_5_le_1_0",
        "actual_gt_1_0": "actual_gt_1_0",
    }
    source = all_runs[all_runs["subset"].isin(DESCRIPTIVE_BINS)].copy()
    source["raw_bin"] = source["subset"].map(descriptive_to_primary)
    source["smape_weighted_sum"] = (
        source["smape_percent"].fillna(0.0) * source["observations"].astype(float)
    )
    grouped = (
        source.groupby(["strategy", "seed", "meter_id", "raw_bin"], sort=True, observed=True)
        .agg(
            observations=("observations", "sum"),
            absolute_error_sum_kwh=("absolute_error_sum_kwh", "sum"),
            squared_error_sum_kwh2=("squared_error_sum_kwh2", "sum"),
            smape_weighted_sum=("smape_weighted_sum", "sum"),
        )
        .reset_index()
    )
    metadata = raw[["meter_id", "assigned_group", "group_description", "all_observations"]].copy()
    result = grouped.merge(metadata, on="meter_id", how="left", validate="many_to_one")
    result["strategy_label"] = result["strategy"].map(STRATEGY_LABELS)
    result["lead"] = LEAD
    result["raw_bin_label"] = result["raw_bin"].map(PRIMARY_LABELS)
    result["row_share"] = result["observations"] / result["all_observations"]
    positive = result["observations"] > 0
    result["mae_kwh"] = np.where(
        positive, result["absolute_error_sum_kwh"] / result["observations"], np.nan
    )
    result["mse_kwh2"] = np.where(
        positive, result["squared_error_sum_kwh2"] / result["observations"], np.nan
    )
    result["rmse_kwh"] = np.sqrt(result["mse_kwh2"])
    result["smape_percent"] = np.where(
        positive, result["smape_weighted_sum"] / result["observations"], np.nan
    )
    result = result[[
        "strategy", "strategy_label", "seed", "lead", "meter_id", "assigned_group",
        "group_description", "raw_bin", "raw_bin_label", "observations",
        "all_observations", "row_share", "mae_kwh", "mse_kwh2", "rmse_kwh",
        "smape_percent",
    ]].sort_values(["strategy", "seed", "meter_id", "raw_bin"]).reset_index(drop=True)
    expected = len(STRATEGIES) * len(FORMAL_SEEDS) * EXPECTED_METERS * len(PRIMARY_BINS)
    ensure(len(result) == expected, f"Unexpected combined-bin row count: {len(result):,}")
    share_check = result[
        result["strategy"].eq(STRATEGIES[0]) & result["seed"].eq(FORMAL_SEEDS[0])
    ].groupby("meter_id")["row_share"].sum()
    ensure(np.allclose(share_check, 1.0, atol=1e-12), "Four primary-bin shares do not sum to one")
    return result


def four_seed_strategy_bin(primary: pd.DataFrame) -> pd.DataFrame:
    result = (
        primary.groupby(
            ["strategy", "strategy_label", "meter_id", "assigned_group", "group_description", "raw_bin", "raw_bin_label"],
            sort=True,
        )
        .agg(
            observations=("observations", "first"),
            all_observations=("all_observations", "first"),
            seed_count=("seed", "nunique"),
            row_share=("row_share", "mean"),
            mae_kwh=("mae_kwh", "mean"),
            mse_kwh2=("mse_kwh2", "mean"),
            rmse_kwh=("rmse_kwh", "mean"),
            smape_percent=("smape_percent", "mean"),
        )
        .reset_index()
    )
    ensure(len(result) == len(STRATEGIES) * EXPECTED_METERS * len(PRIMARY_BINS), "Unexpected four-seed strategy-bin count")
    ensure(result["seed_count"].eq(4).all(), "Not every strategy-meter-bin cell contains four seeds")
    observation_check = result.pivot_table(
        index=["meter_id", "raw_bin"], columns="strategy", values="observations", aggfunc="first"
    )
    ensure((observation_check.max(axis=1) == observation_check.min(axis=1)).all(), "Four-seed bin observations differ across strategies")
    return result


def build_meter_bin_pairwise_gains(strategy_bin: pd.DataFrame) -> pd.DataFrame:
    keys = ["meter_id", "assigned_group", "group_description", "raw_bin", "raw_bin_label", "observations", "all_observations", "row_share"]
    rows: list[pd.DataFrame] = []
    for comparison in COMPARISONS:
        focal = strategy_bin[strategy_bin["strategy"].eq(comparison["focal_strategy"])][keys + ["mae_kwh", "mse_kwh2", "rmse_kwh", "smape_percent"]].copy()
        comparator = strategy_bin[strategy_bin["strategy"].eq(comparison["comparator_strategy"])][keys + ["mae_kwh", "mse_kwh2", "rmse_kwh", "smape_percent"]].copy()
        focal = focal.rename(columns={x: f"{x}__focal" for x in ["mae_kwh", "mse_kwh2", "rmse_kwh", "smape_percent"]})
        comparator = comparator.rename(columns={x: f"{x}__comparator" for x in ["mae_kwh", "mse_kwh2", "rmse_kwh", "smape_percent"]})
        merged = focal.merge(comparator, on=keys, how="outer", validate="one_to_one")
        merged.insert(0, "comparison_id", comparison["comparison_id"])
        merged.insert(1, "comparison_label", comparison["comparison_label"])
        merged.insert(2, "interpretive_role", comparison["interpretive_role"])
        merged.insert(3, "focal_strategy", comparison["focal_strategy"])
        merged.insert(4, "comparator_strategy", comparison["comparator_strategy"])
        merged["gain_mae"] = merged["mae_kwh__comparator"] - merged["mae_kwh__focal"]
        merged["gain_mse"] = merged["mse_kwh2__comparator"] - merged["mse_kwh2__focal"]
        merged["gain_rmse_nonadditive_bin"] = merged["rmse_kwh__comparator"] - merged["rmse_kwh__focal"]
        merged["gain_smape"] = merged["smape_percent__comparator"] - merged["smape_percent__focal"]
        rows.append(merged)
    result = pd.concat(rows, ignore_index=True)
    expected = len(COMPARISONS) * EXPECTED_METERS * len(PRIMARY_BINS)
    ensure(len(result) == expected, f"Unexpected meter-bin pairwise row count: {len(result):,}")
    return result


def build_all_row_outputs(all_runs: pd.DataFrame, raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_lookup = raw.set_index("meter_id")
    all_rows = all_runs[all_runs["subset"].eq("all") & all_runs["strategy"].isin(STRATEGIES)].copy()
    all_rows["mse_kwh2"] = all_rows["squared_error_sum_kwh2"] / all_rows["observations"]
    strategy = (
        all_rows.groupby(["strategy", "meter_id"], sort=True)
        .agg(
            seed_count=("seed", "nunique"),
            observations=("observations", "first"),
            mae_kwh=("mae_kwh", "mean"),
            mse_kwh2=("mse_kwh2", "mean"),
            rmse_kwh=("rmse_kwh", "mean"),
            smape_percent=("smape_percent", "mean"),
        )
        .reset_index()
    )
    strategy["strategy_label"] = strategy["strategy"].map(STRATEGY_LABELS)
    strategy["assigned_group"] = strategy["meter_id"].map(raw_lookup["assigned_group"])
    strategy["group_description"] = strategy["meter_id"].map(raw_lookup["group_description"])
    ensure(len(strategy) == len(STRATEGIES) * EXPECTED_METERS, "Unexpected all-row strategy count")
    ensure(strategy["seed_count"].eq(4).all(), "All-row strategy values do not all contain four seeds")

    metric_columns = {"mae": "mae_kwh", "rmse": "rmse_kwh", "smape": "smape_percent"}
    gain_rows: list[dict[str, object]] = []
    strategy_index = strategy.set_index(["strategy", "meter_id"])
    for comparison in COMPARISONS:
        for meter_id in sorted(raw_lookup.index):
            for metric, column in metric_columns.items():
                focal_error = float(strategy_index.loc[(comparison["focal_strategy"], meter_id), column])
                comparator_error = float(strategy_index.loc[(comparison["comparator_strategy"], meter_id), column])
                gain = comparator_error - focal_error
                gain_rows.append({
                    "comparison_id": comparison["comparison_id"],
                    "comparison_label": comparison["comparison_label"],
                    "interpretive_role": comparison["interpretive_role"],
                    "focal_strategy": comparison["focal_strategy"],
                    "comparator_strategy": comparison["comparator_strategy"],
                    "meter_id": int(meter_id),
                    "assigned_group": int(raw_lookup.loc[meter_id, "assigned_group"]),
                    "group_description": raw_lookup.loc[meter_id, "group_description"],
                    "metric": metric,
                    "unit": FORMAL_METRICS[metric]["unit"],
                    "focal_error": focal_error,
                    "comparator_error": comparator_error,
                    "absolute_gain": gain,
                    "relative_gain_pct": safe_relative_gain(gain, comparator_error),
                    "positive_gain": bool(gain > 0),
                })
    gains = pd.DataFrame(gain_rows)
    ensure(len(gains) == len(COMPARISONS) * EXPECTED_METERS * len(FORMAL_METRICS), "Unexpected all-row pairwise gain count")

    summaries: list[dict[str, object]] = []
    for comparison in COMPARISONS:
        for group in range(1, 5):
            for metric in FORMAL_METRICS:
                q = gains[
                    gains["comparison_id"].eq(comparison["comparison_id"])
                    & gains["assigned_group"].eq(group)
                    & gains["metric"].eq(metric)
                ]
                relative = q["relative_gain_pct"].dropna()
                summaries.append({
                    "comparison_id": comparison["comparison_id"],
                    "comparison_label": comparison["comparison_label"],
                    "interpretive_role": comparison["interpretive_role"],
                    "focal_strategy": comparison["focal_strategy"],
                    "comparator_strategy": comparison["comparator_strategy"],
                    "assigned_group": group,
                    "group_description": GROUP_DESCRIPTIONS[group],
                    "metric": metric,
                    "unit": FORMAL_METRICS[metric]["unit"],
                    "meters": len(q),
                    "absolute_gain_mean": float(q["absolute_gain"].mean()),
                    "absolute_gain_median": float(q["absolute_gain"].median()),
                    "absolute_gain_sample_std": float(q["absolute_gain"].std(ddof=1)),
                    "relative_gain_defined_meters": len(relative),
                    "relative_gain_mean_pct": float(relative.mean()) if len(relative) else math.nan,
                    "relative_gain_median_pct": float(relative.median()) if len(relative) else math.nan,
                    "positive_gain_meters": int(q["positive_gain"].sum()),
                    "positive_gain_share": float(q["positive_gain"].mean()),
                })
    return strategy, gains, pd.DataFrame(summaries)


def raw_distribution_outputs(raw: pd.DataFrame, permutations: int, bootstraps: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    overall = {raw_bin: float(raw[f"{raw_bin}_row_share"].mean()) for raw_bin in DESCRIPTIVE_BINS}
    populations = [("Overall CER", raw)] + [(f"G{group}", raw[raw["assigned_group"].eq(group)]) for group in range(1, 5)]
    for population, part in populations:
        group = 0 if population == "Overall CER" else int(population[1:])
        for raw_bin in DESCRIPTIVE_BINS:
            rows.append({
                "population": population,
                "assigned_group": group,
                "raw_bin": raw_bin,
                "raw_bin_label": DESCRIPTIVE_LABELS[raw_bin],
                "reference_overall_share": overall[raw_bin],
                **summary_stats(part[f"{raw_bin}_row_share"]),
            })
    summary = pd.DataFrame(rows)

    rng = np.random.default_rng(RANDOM_SEED)
    effects: list[dict[str, object]] = []
    values_group = raw["assigned_group"].to_numpy(int)
    for raw_bin in DESCRIPTIVE_BINS:
        values = raw[f"{raw_bin}_row_share"].to_numpy(float)
        eta, omega = eta_omega(values, values_group)
        p_value = permutation_p(values, values_group, permutations, rng)
        bootstrap_eta: list[float] = []
        for _ in range(bootstraps):
            indexes: list[int] = []
            for group in range(1, 5):
                positions = np.flatnonzero(values_group == group)
                indexes.extend(rng.choice(positions, size=len(positions), replace=True))
            indexes_array = np.asarray(indexes)
            bootstrap_eta.append(eta_omega(values[indexes_array], values_group[indexes_array])[0])
        effects.append({
            "raw_bin": raw_bin,
            "raw_bin_label": DESCRIPTIVE_LABELS[raw_bin],
            "eta_squared": eta,
            "omega_squared": omega,
            "bootstrap_eta_ci_low": float(np.quantile(bootstrap_eta, 0.025)),
            "bootstrap_eta_ci_high": float(np.quantile(bootstrap_eta, 0.975)),
            "permutation_p": p_value,
            "permutations": permutations,
            "bootstraps": bootstraps,
        })
    effects_frame = pd.DataFrame(effects)
    effects_frame["bh_q_value"] = bh_adjust(effects_frame["permutation_p"])

    pair_rows: list[dict[str, object]] = []
    for raw_bin in DESCRIPTIVE_BINS:
        for group_a in range(1, 5):
            for group_b in range(group_a + 1, 5):
                values_a = raw.loc[raw["assigned_group"].eq(group_a), f"{raw_bin}_row_share"].to_numpy(float)
                values_b = raw.loc[raw["assigned_group"].eq(group_b), f"{raw_bin}_row_share"].to_numpy(float)
                pair_rows.append({
                    "raw_bin": raw_bin,
                    "raw_bin_label": DESCRIPTIVE_LABELS[raw_bin],
                    "group_a": group_a,
                    "group_b": group_b,
                    "mean_a": float(values_a.mean()),
                    "mean_b": float(values_b.mean()),
                    "mean_difference_b_minus_a": float(values_b.mean() - values_a.mean()),
                })
    return summary, effects_frame, pd.DataFrame(pair_rows)


def strategy_error_components(strategy_bin: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    populations = [(0, "Overall CER", strategy_bin)] + [
        (group, f"G{group}", strategy_bin[strategy_bin["assigned_group"].eq(group)]) for group in range(1, 5)
    ]
    rows: list[dict[str, object]] = []
    for group, population, part in populations:
        for strategy in STRATEGIES:
            strategy_part = part[part["strategy"].eq(strategy)]
            for raw_bin in PRIMARY_BINS:
                q = strategy_part[strategy_part["raw_bin"].eq(raw_bin)]
                exposure = float(q["row_share"].mean())
                for metric, specification in ADDITIVE_METRICS.items():
                    conditional_error = weighted(q[specification["strategy_column"]], q["row_share"])
                    rows.append({
                        "assigned_group": group,
                        "population": population,
                        "group_description": "Overall CER" if group == 0 else GROUP_DESCRIPTIONS[group],
                        "strategy": strategy,
                        "strategy_label": STRATEGY_LABELS[strategy],
                        "raw_bin": raw_bin,
                        "raw_bin_label": PRIMARY_LABELS[raw_bin],
                        "metric": metric,
                        "unit": specification["unit"],
                        "mean_meter_row_share": exposure,
                        "exposure_weighted_conditional_error": conditional_error,
                        "observed_error_contribution": exposure * conditional_error,
                        "meters_with_bin": int((q["observations"] > 0).sum()),
                        "total_meters": int(q["meter_id"].nunique()),
                    })
    components = pd.DataFrame(rows)
    reference = components[components["assigned_group"].eq(0)].set_index(["strategy", "raw_bin", "metric"])
    totals: list[dict[str, object]] = []
    for group in range(1, 5):
        group_part = components[components["assigned_group"].eq(group)]
        for strategy in STRATEGIES:
            for metric, specification in ADDITIVE_METRICS.items():
                q = group_part[group_part["strategy"].eq(strategy) & group_part["metric"].eq(metric)].set_index("raw_bin")
                observed = float(q["observed_error_contribution"].sum())
                standardised = float(sum(
                    reference.loc[(strategy, raw_bin, metric), "mean_meter_row_share"]
                    * q.loc[raw_bin, "exposure_weighted_conditional_error"]
                    for raw_bin in PRIMARY_BINS
                ))
                composition_only = float(sum(
                    q.loc[raw_bin, "mean_meter_row_share"]
                    * reference.loc[(strategy, raw_bin, metric), "exposure_weighted_conditional_error"]
                    for raw_bin in PRIMARY_BINS
                ))
                totals.append({
                    "assigned_group": group,
                    "group_description": GROUP_DESCRIPTIONS[group],
                    "strategy": strategy,
                    "strategy_label": STRATEGY_LABELS[strategy],
                    "metric": metric,
                    "unit": specification["unit"],
                    "observed_group_error": observed,
                    "common_composition_standardised_error": standardised,
                    "composition_only_expected_error": composition_only,
                    "standardisation_change": standardised - observed,
                })
    totals_frame = pd.DataFrame(totals)

    rmse_rows: list[dict[str, object]] = []
    mse_rows = totals_frame[totals_frame["metric"].eq("mse")]
    for row in mse_rows.itertuples(index=False):
        rmse_rows.append({
            "assigned_group": row.assigned_group,
            "group_description": row.group_description,
            "strategy": row.strategy,
            "strategy_label": row.strategy_label,
            "metric": "rmse",
            "unit": "kWh",
            "observed_rmse_derived_from_group_mse": math.sqrt(max(0.0, row.observed_group_error)),
            "common_composition_standardised_rmse": math.sqrt(max(0.0, row.common_composition_standardised_error)),
            "composition_only_expected_rmse": math.sqrt(max(0.0, row.composition_only_expected_error)),
            "standardisation_change": math.sqrt(max(0.0, row.common_composition_standardised_error)) - math.sqrt(max(0.0, row.observed_group_error)),
            "scope": "derived from additive group MSE mechanism; not mean meter-level RMSE",
        })
    return components, totals_frame, pd.DataFrame(rmse_rows)


def gain_components(pairwise_bin: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    populations = [(0, "Overall CER", pairwise_bin)] + [
        (group, f"G{group}", pairwise_bin[pairwise_bin["assigned_group"].eq(group)]) for group in range(1, 5)
    ]
    rows: list[dict[str, object]] = []
    for group, population, part in populations:
        for comparison in COMPARISONS:
            comparison_part = part[part["comparison_id"].eq(comparison["comparison_id"])]
            for raw_bin in PRIMARY_BINS:
                q = comparison_part[comparison_part["raw_bin"].eq(raw_bin)]
                exposure = float(q["row_share"].mean())
                for metric, gain_column in [("mae", "gain_mae"), ("mse", "gain_mse"), ("smape", "gain_smape")]:
                    conditional_gain = weighted(q[gain_column], q["row_share"])
                    rows.append({
                        "comparison_id": comparison["comparison_id"],
                        "comparison_label": comparison["comparison_label"],
                        "interpretive_role": comparison["interpretive_role"],
                        "focal_strategy": comparison["focal_strategy"],
                        "comparator_strategy": comparison["comparator_strategy"],
                        "assigned_group": group,
                        "population": population,
                        "group_description": "Overall CER" if group == 0 else GROUP_DESCRIPTIONS[group],
                        "raw_bin": raw_bin,
                        "raw_bin_label": PRIMARY_LABELS[raw_bin],
                        "metric": metric,
                        "unit": ADDITIVE_METRICS[metric]["unit"],
                        "mean_meter_row_share": exposure,
                        "exposure_weighted_conditional_gain": conditional_gain,
                        "observed_gain_contribution": exposure * conditional_gain,
                        "meters_with_bin": int((q["observations"] > 0).sum()),
                        "total_meters": int(q["meter_id"].nunique()),
                    })
    components = pd.DataFrame(rows)
    reference = components[components["assigned_group"].eq(0)].set_index(["comparison_id", "raw_bin", "metric"])
    totals: list[dict[str, object]] = []
    for comparison in COMPARISONS:
        for group in range(1, 5):
            group_part = components[
                components["comparison_id"].eq(comparison["comparison_id"])
                & components["assigned_group"].eq(group)
            ]
            for metric in ADDITIVE_METRICS:
                q = group_part[group_part["metric"].eq(metric)].set_index("raw_bin")
                observed = float(q["observed_gain_contribution"].sum())
                standardised = float(sum(
                    reference.loc[(comparison["comparison_id"], raw_bin, metric), "mean_meter_row_share"]
                    * q.loc[raw_bin, "exposure_weighted_conditional_gain"]
                    for raw_bin in PRIMARY_BINS
                ))
                composition_only = float(sum(
                    q.loc[raw_bin, "mean_meter_row_share"]
                    * reference.loc[(comparison["comparison_id"], raw_bin, metric), "exposure_weighted_conditional_gain"]
                    for raw_bin in PRIMARY_BINS
                ))
                totals.append({
                    "comparison_id": comparison["comparison_id"],
                    "comparison_label": comparison["comparison_label"],
                    "interpretive_role": comparison["interpretive_role"],
                    "focal_strategy": comparison["focal_strategy"],
                    "comparator_strategy": comparison["comparator_strategy"],
                    "assigned_group": group,
                    "group_description": GROUP_DESCRIPTIONS[group],
                    "metric": metric,
                    "unit": ADDITIVE_METRICS[metric]["unit"],
                    "observed_group_gain": observed,
                    "common_composition_standardised_gain": standardised,
                    "composition_only_expected_gain": composition_only,
                    "standardisation_change": standardised - observed,
                })
    totals_frame = pd.DataFrame(totals)
    return components, totals_frame, derive_rmse_gain_from_strategy_errors(totals_frame=None)


def derive_rmse_gain_from_strategy_errors(
    totals_frame: pd.DataFrame | None,
    strategy_rmse: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if strategy_rmse is None:
        return pd.DataFrame()
    indexed = strategy_rmse.set_index(["assigned_group", "strategy"])
    rows: list[dict[str, object]] = []
    for comparison in COMPARISONS:
        for group in range(1, 5):
            focal = indexed.loc[(group, comparison["focal_strategy"])]
            comparator = indexed.loc[(group, comparison["comparator_strategy"])]
            rows.append({
                "comparison_id": comparison["comparison_id"],
                "comparison_label": comparison["comparison_label"],
                "interpretive_role": comparison["interpretive_role"],
                "focal_strategy": comparison["focal_strategy"],
                "comparator_strategy": comparison["comparator_strategy"],
                "assigned_group": group,
                "group_description": GROUP_DESCRIPTIONS[group],
                "metric": "rmse",
                "unit": "kWh",
                "observed_group_gain": float(comparator["observed_rmse_derived_from_group_mse"] - focal["observed_rmse_derived_from_group_mse"]),
                "common_composition_standardised_gain": float(comparator["common_composition_standardised_rmse"] - focal["common_composition_standardised_rmse"]),
                "composition_only_expected_gain": float(comparator["composition_only_expected_rmse"] - focal["composition_only_expected_rmse"]),
                "standardisation_change": float(
                    (comparator["common_composition_standardised_rmse"] - focal["common_composition_standardised_rmse"])
                    - (comparator["observed_rmse_derived_from_group_mse"] - focal["observed_rmse_derived_from_group_mse"])
                ),
                "scope": "derived from additive group MSE mechanism; not mean meter-level RMSE gain",
            })
    return pd.DataFrame(rows)


def pairwise_group_decomposition(gain_component_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for comparison in COMPARISONS:
        comparison_part = gain_component_frame[gain_component_frame["comparison_id"].eq(comparison["comparison_id"])]
        for metric in ADDITIVE_METRICS:
            for group_a in range(1, 5):
                for group_b in range(group_a + 1, 5):
                    a = comparison_part[
                        comparison_part["assigned_group"].eq(group_a) & comparison_part["metric"].eq(metric)
                    ].set_index("raw_bin")
                    b = comparison_part[
                        comparison_part["assigned_group"].eq(group_b) & comparison_part["metric"].eq(metric)
                    ].set_index("raw_bin")
                    composition = 0.0
                    conditional = 0.0
                    for raw_bin in PRIMARY_BINS:
                        p_a = float(a.loc[raw_bin, "mean_meter_row_share"])
                        p_b = float(b.loc[raw_bin, "mean_meter_row_share"])
                        d_a = float(a.loc[raw_bin, "exposure_weighted_conditional_gain"])
                        d_b = float(b.loc[raw_bin, "exposure_weighted_conditional_gain"])
                        composition += (p_b - p_a) * (d_a + d_b) / 2.0
                        conditional += (d_b - d_a) * (p_a + p_b) / 2.0
                    total = float(b["observed_gain_contribution"].sum() - a["observed_gain_contribution"].sum())
                    ensure(abs(total - composition - conditional) <= 1e-10, "Pairwise composition-performance identity failed")
                    denominator = abs(composition) + abs(conditional)
                    rows.append({
                        "comparison_id": comparison["comparison_id"],
                        "comparison_label": comparison["comparison_label"],
                        "interpretive_role": comparison["interpretive_role"],
                        "focal_strategy": comparison["focal_strategy"],
                        "comparator_strategy": comparison["comparator_strategy"],
                        "metric": metric,
                        "unit": ADDITIVE_METRICS[metric]["unit"],
                        "group_a": group_a,
                        "group_b": group_b,
                        "contrast": f"G{group_b} - G{group_a}",
                        "observed_gain_difference": total,
                        "composition_component": composition,
                        "conditional_performance_component": conditional,
                        "reconstruction_difference": total - composition - conditional,
                        "composition_share_of_absolute_difference": abs(composition) / denominator if denominator > 0 else math.nan,
                        "conditional_share_of_absolute_difference": abs(conditional) / denominator if denominator > 0 else math.nan,
                    })
    return pd.DataFrame(rows)


def complete_support_outputs(
    strategy_bin: pd.DataFrame,
    all_row_gains: pd.DataFrame,
    permutations: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = strategy_bin[strategy_bin["strategy"].eq(STRATEGIES[0])]
    pivot = base.pivot_table(
        index=["meter_id", "assigned_group", "group_description"], columns="raw_bin", values="observations", aggfunc="first"
    )
    complete_ids = pivot[(pivot[PRIMARY_BINS] > 0).all(axis=1)].reset_index()["meter_id"]
    ensure(len(complete_ids) == 845, f"Expected 845 complete-support meters, found {len(complete_ids)}")

    reference_shares = base.groupby("raw_bin")["row_share"].mean().to_dict()
    complete = strategy_bin[strategy_bin["meter_id"].isin(set(complete_ids))].copy()
    strategy_rows: list[dict[str, object]] = []
    for (strategy, meter_id, group, description), part in complete.groupby(
        ["strategy", "meter_id", "assigned_group", "group_description"], sort=True
    ):
        indexed = part.set_index("raw_bin")
        mae = sum(reference_shares[raw_bin] * float(indexed.loc[raw_bin, "mae_kwh"]) for raw_bin in PRIMARY_BINS)
        mse = sum(reference_shares[raw_bin] * float(indexed.loc[raw_bin, "mse_kwh2"]) for raw_bin in PRIMARY_BINS)
        smape = sum(reference_shares[raw_bin] * float(indexed.loc[raw_bin, "smape_percent"]) for raw_bin in PRIMARY_BINS)
        strategy_rows.append({
            "meter_id": meter_id,
            "assigned_group": group,
            "group_description": description,
            "strategy": strategy,
            "strategy_label": STRATEGY_LABELS[strategy],
            "standardised_error_mae": mae,
            "standardised_error_mse": mse,
            "standardised_error_rmse": math.sqrt(max(0.0, mse)),
            "standardised_error_smape": smape,
        })
    strategy_standardised = pd.DataFrame(strategy_rows)
    ensure(len(strategy_standardised) == 845 * len(STRATEGIES), "Unexpected complete-support strategy-error count")

    strategy_index = strategy_standardised.set_index(["strategy", "meter_id"])
    gain_rows: list[dict[str, object]] = []
    for comparison in COMPARISONS:
        for meter_id in sorted(complete_ids):
            group = int(strategy_index.loc[(comparison["focal_strategy"], meter_id), "assigned_group"])
            description = strategy_index.loc[(comparison["focal_strategy"], meter_id), "group_description"]
            for metric in FORMAL_METRICS:
                focal_error = float(strategy_index.loc[(comparison["focal_strategy"], meter_id), f"standardised_error_{metric}"])
                comparator_error = float(strategy_index.loc[(comparison["comparator_strategy"], meter_id), f"standardised_error_{metric}"])
                gain = comparator_error - focal_error
                gain_rows.append({
                    "comparison_id": comparison["comparison_id"],
                    "comparison_label": comparison["comparison_label"],
                    "interpretive_role": comparison["interpretive_role"],
                    "focal_strategy": comparison["focal_strategy"],
                    "comparator_strategy": comparison["comparator_strategy"],
                    "meter_id": meter_id,
                    "assigned_group": group,
                    "group_description": description,
                    "metric": metric,
                    "unit": FORMAL_METRICS[metric]["unit"],
                    "standardised_focal_error": focal_error,
                    "standardised_comparator_error": comparator_error,
                    "standardised_absolute_gain": gain,
                    "standardised_relative_gain_pct": safe_relative_gain(gain, comparator_error),
                    "positive_standardised_gain": bool(gain > 0),
                })
    standardised_gains = pd.DataFrame(gain_rows)

    effect_rows: list[dict[str, object]] = []
    for comparison_index, comparison in enumerate(COMPARISONS):
        rng = np.random.default_rng(RANDOM_SEED + 19 + comparison_index * 1000)
        for metric in FORMAL_METRICS:
            q = standardised_gains[
                standardised_gains["comparison_id"].eq(comparison["comparison_id"])
                & standardised_gains["metric"].eq(metric)
            ]
            values = q["standardised_absolute_gain"].to_numpy(float)
            groups = q["assigned_group"].to_numpy(int)
            eta, omega = eta_omega(values, groups)
            p_value = permutation_p(values, groups, permutations, rng)
            effect_rows.append({
                "comparison_id": comparison["comparison_id"],
                "comparison_label": comparison["comparison_label"],
                "interpretive_role": comparison["interpretive_role"],
                "metric": metric,
                "complete_support_meters": len(q),
                "group_counts": json.dumps(q["assigned_group"].value_counts().sort_index().to_dict()),
                "eta_squared": eta,
                "omega_squared": omega,
                "permutation_p": p_value,
                "permutations": permutations,
                "scope": "complete four-bin support sensitivity; all 929 retained in primary group-level standardisation",
            })
    effects = pd.DataFrame(effect_rows)
    effects["bh_q_value_within_metric_family"] = effects.groupby("metric", group_keys=False)["permutation_p"].apply(bh_adjust)

    observed_complete = all_row_gains[all_row_gains["meter_id"].isin(set(complete_ids))].copy()
    merged = standardised_gains.merge(
        observed_complete[["comparison_id", "meter_id", "metric", "absolute_gain", "relative_gain_pct", "positive_gain"]],
        on=["comparison_id", "meter_id", "metric"],
        validate="one_to_one",
    )
    summary_rows: list[dict[str, object]] = []
    for comparison in COMPARISONS:
        for group in range(1, 5):
            for metric in FORMAL_METRICS:
                q = merged[
                    merged["comparison_id"].eq(comparison["comparison_id"])
                    & merged["assigned_group"].eq(group)
                    & merged["metric"].eq(metric)
                ]
                summary_rows.append({
                    "comparison_id": comparison["comparison_id"],
                    "comparison_label": comparison["comparison_label"],
                    "interpretive_role": comparison["interpretive_role"],
                    "assigned_group": group,
                    "group_description": GROUP_DESCRIPTIONS[group],
                    "metric": metric,
                    "meters": len(q),
                    "observed_gain_mean_complete_support": float(q["absolute_gain"].mean()),
                    "common_composition_standardised_gain_mean": float(q["standardised_absolute_gain"].mean()),
                    "standardisation_change": float(q["standardised_absolute_gain"].mean() - q["absolute_gain"].mean()),
                    "observed_gain_median_complete_support": float(q["absolute_gain"].median()),
                    "standardised_gain_median": float(q["standardised_absolute_gain"].median()),
                    "observed_relative_gain_mean_pct": float(q["relative_gain_pct"].mean()),
                    "standardised_relative_gain_mean_pct": float(q["standardised_relative_gain_pct"].mean()),
                    "observed_positive_gain_share": float(q["positive_gain"].mean()),
                    "standardised_positive_gain_share": float(q["positive_standardised_gain"].mean()),
                })
    return strategy_standardised, standardised_gains, effects, pd.DataFrame(summary_rows)


def bootstrap_intervals(
    pairwise_bin: pd.DataFrame,
    strategy_bin: pd.DataFrame,
    bootstraps: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparison_ids = [x["comparison_id"] for x in COMPARISONS]
    additive_metrics = ["mae", "mse", "smape"]
    gain_columns = {"mae": "gain_mae", "mse": "gain_mse", "smape": "gain_smape"}
    strategy_mse_column = "mse_kwh2"

    grouped_pairwise = {}
    grouped_strategy = {}
    group_sizes = EXPECTED_GROUP_COUNTS
    for group in range(1, 5):
        meters = sorted(pairwise_bin.loc[pairwise_bin["assigned_group"].eq(group), "meter_id"].unique())
        meter_to_position = {meter: index for index, meter in enumerate(meters)}
        shares = np.full((len(meters), len(PRIMARY_BINS)), np.nan)
        gains = np.full((len(COMPARISONS), len(additive_metrics), len(meters), len(PRIMARY_BINS)), np.nan)
        mse_errors = np.full((len(STRATEGIES), len(meters), len(PRIMARY_BINS)), np.nan)
        for bin_index, raw_bin in enumerate(PRIMARY_BINS):
            base = pairwise_bin[
                pairwise_bin["assigned_group"].eq(group)
                & pairwise_bin["comparison_id"].eq(comparison_ids[0])
                & pairwise_bin["raw_bin"].eq(raw_bin)
            ]
            for row in base.itertuples(index=False):
                shares[meter_to_position[row.meter_id], bin_index] = row.row_share
            for comparison_index, comparison_id in enumerate(comparison_ids):
                q = pairwise_bin[
                    pairwise_bin["assigned_group"].eq(group)
                    & pairwise_bin["comparison_id"].eq(comparison_id)
                    & pairwise_bin["raw_bin"].eq(raw_bin)
                ]
                for row in q.itertuples(index=False):
                    position = meter_to_position[row.meter_id]
                    gains[comparison_index, 0, position, bin_index] = row.gain_mae
                    gains[comparison_index, 1, position, bin_index] = row.gain_mse
                    gains[comparison_index, 2, position, bin_index] = row.gain_smape
            for strategy_index, strategy in enumerate(STRATEGIES):
                q = strategy_bin[
                    strategy_bin["assigned_group"].eq(group)
                    & strategy_bin["strategy"].eq(strategy)
                    & strategy_bin["raw_bin"].eq(raw_bin)
                ]
                for row in q.itertuples(index=False):
                    mse_errors[strategy_index, meter_to_position[row.meter_id], bin_index] = getattr(row, strategy_mse_column)
        grouped_pairwise[group] = (shares, gains)
        grouped_strategy[group] = mse_errors
        ensure(len(meters) == group_sizes[group], f"Bootstrap group {group} size mismatch")

    rng = np.random.default_rng(RANDOM_SEED + 7)
    observed_store = np.full((bootstraps, len(COMPARISONS), 4, len(additive_metrics)), np.nan)
    standardised_store = np.full_like(observed_store, np.nan)
    rmse_observed_store = np.full((bootstraps, len(COMPARISONS), 4), np.nan)
    rmse_standardised_store = np.full_like(rmse_observed_store, np.nan)

    for bootstrap in range(bootstraps):
        group_p = {}
        group_d = {}
        group_mse_conditional = {}
        for group in range(1, 5):
            shares, gains = grouped_pairwise[group]
            indexes = rng.choice(np.arange(shares.shape[0]), size=shares.shape[0], replace=True)
            sampled_shares = shares[indexes]
            p = np.nanmean(sampled_shares, axis=0)
            group_p[group] = p
            d = np.full((len(COMPARISONS), len(additive_metrics), len(PRIMARY_BINS)), np.nan)
            for comparison_index in range(len(COMPARISONS)):
                for metric_index in range(len(additive_metrics)):
                    sampled_gains = gains[comparison_index, metric_index, indexes, :]
                    for bin_index in range(len(PRIMARY_BINS)):
                        values = sampled_gains[:, bin_index]
                        weights = sampled_shares[:, bin_index]
                        valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
                        d[comparison_index, metric_index, bin_index] = (
                            np.sum(values[valid] * weights[valid]) / np.sum(weights[valid]) if valid.any() else np.nan
                        )
            group_d[group] = d

            mse_errors = grouped_strategy[group][:, indexes, :]
            conditional_mse = np.full((len(STRATEGIES), len(PRIMARY_BINS)), np.nan)
            for strategy_index in range(len(STRATEGIES)):
                for bin_index in range(len(PRIMARY_BINS)):
                    values = mse_errors[strategy_index, :, bin_index]
                    weights = sampled_shares[:, bin_index]
                    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
                    conditional_mse[strategy_index, bin_index] = (
                        np.sum(values[valid] * weights[valid]) / np.sum(weights[valid]) if valid.any() else np.nan
                    )
            group_mse_conditional[group] = conditional_mse

        reference_p = sum(group_sizes[group] * group_p[group] for group in range(1, 5)) / EXPECTED_METERS
        for group_index, group in enumerate(range(1, 5)):
            for comparison_index in range(len(COMPARISONS)):
                for metric_index in range(len(additive_metrics)):
                    d = group_d[group][comparison_index, metric_index]
                    observed_store[bootstrap, comparison_index, group_index, metric_index] = np.sum(group_p[group] * d)
                    standardised_store[bootstrap, comparison_index, group_index, metric_index] = np.sum(reference_p * d)

                mse_conditional = group_mse_conditional[group]
                observed_strategy_mse = np.sum(mse_conditional * group_p[group][None, :], axis=1)
                standardised_strategy_mse = np.sum(mse_conditional * reference_p[None, :], axis=1)
                comparison = COMPARISONS[comparison_index]
                focal_index = STRATEGIES.index(comparison["focal_strategy"])
                comparator_index = STRATEGIES.index(comparison["comparator_strategy"])
                rmse_observed_store[bootstrap, comparison_index, group_index] = (
                    math.sqrt(max(0.0, observed_strategy_mse[comparator_index]))
                    - math.sqrt(max(0.0, observed_strategy_mse[focal_index]))
                )
                rmse_standardised_store[bootstrap, comparison_index, group_index] = (
                    math.sqrt(max(0.0, standardised_strategy_mse[comparator_index]))
                    - math.sqrt(max(0.0, standardised_strategy_mse[focal_index]))
                )

    interval_rows: list[dict[str, object]] = []
    for comparison_index, comparison in enumerate(COMPARISONS):
        for group_index, group in enumerate(range(1, 5)):
            for metric_index, metric in enumerate(additive_metrics):
                for estimate, store in [
                    ("observed_group_gain", observed_store),
                    ("common_composition_standardised_gain", standardised_store),
                ]:
                    values = store[:, comparison_index, group_index, metric_index]
                    interval_rows.append({
                        "comparison_id": comparison["comparison_id"],
                        "comparison_label": comparison["comparison_label"],
                        "assigned_group": group,
                        "metric": metric,
                        "estimate": estimate,
                        "bootstrap_mean": float(np.mean(values)),
                        "ci_low": float(np.quantile(values, 0.025)),
                        "ci_high": float(np.quantile(values, 0.975)),
                        "bootstraps": bootstraps,
                    })
        for metric_index, metric in enumerate(additive_metrics):
            for group_a in range(1, 5):
                for group_b in range(group_a + 1, 5):
                    values = (
                        standardised_store[:, comparison_index, group_b - 1, metric_index]
                        - standardised_store[:, comparison_index, group_a - 1, metric_index]
                    )
                    interval_rows.append({
                        "comparison_id": comparison["comparison_id"],
                        "comparison_label": comparison["comparison_label"],
                        "assigned_group": 0,
                        "metric": metric,
                        "estimate": f"standardised_G{group_b}_minus_G{group_a}",
                        "bootstrap_mean": float(np.mean(values)),
                        "ci_low": float(np.quantile(values, 0.025)),
                        "ci_high": float(np.quantile(values, 0.975)),
                        "bootstraps": bootstraps,
                    })

    rmse_rows: list[dict[str, object]] = []
    for comparison_index, comparison in enumerate(COMPARISONS):
        for group_index, group in enumerate(range(1, 5)):
            for estimate, store in [
                ("observed_group_gain", rmse_observed_store),
                ("common_composition_standardised_gain", rmse_standardised_store),
            ]:
                values = store[:, comparison_index, group_index]
                rmse_rows.append({
                    "comparison_id": comparison["comparison_id"],
                    "comparison_label": comparison["comparison_label"],
                    "assigned_group": group,
                    "metric": "rmse",
                    "estimate": estimate,
                    "bootstrap_mean": float(np.mean(values)),
                    "ci_low": float(np.quantile(values, 0.025)),
                    "ci_high": float(np.quantile(values, 0.975)),
                    "bootstraps": bootstraps,
                    "scope": "derived from additive group MSE mechanism",
                })
        for group_a in range(1, 5):
            for group_b in range(group_a + 1, 5):
                values = rmse_standardised_store[:, comparison_index, group_b - 1] - rmse_standardised_store[:, comparison_index, group_a - 1]
                rmse_rows.append({
                    "comparison_id": comparison["comparison_id"],
                    "comparison_label": comparison["comparison_label"],
                    "assigned_group": 0,
                    "metric": "rmse",
                    "estimate": f"standardised_G{group_b}_minus_G{group_a}",
                    "bootstrap_mean": float(np.mean(values)),
                    "ci_low": float(np.quantile(values, 0.025)),
                    "ci_high": float(np.quantile(values, 0.975)),
                    "bootstraps": bootstraps,
                    "scope": "derived from additive group MSE mechanism",
                })
    return pd.DataFrame(interval_rows), pd.DataFrame(rmse_rows)


def profile_group_analysis_reproduction_audit(strategy_errors: pd.DataFrame, pairwise_gains: pd.DataFrame, profile_group_analysis_h12: pd.DataFrame) -> pd.DataFrame:
    strategy_mapping = {
        "direct_transfer": "direct_transfer_value_mean",
        "fine_tuning": "fine_tuning_value_mean",
        "cer_scratch_limited": "cer_scratch_limited_value_mean",
        "cer_scratch_full": "cer_scratch_full_value_mean",
    }
    gain_mapping = {
        "source_transfer": "source_transfer_gain_mean",
        "fine_tuning": "fine_tuning_gain_mean",
        "full_data": "full_data_gain_mean",
        "direct_vs_full": "direct_vs_full_gain_mean",
    }
    metric_column = {"mae": "mae_kwh", "rmse": "rmse_kwh", "smape": "smape_percent"}
    rows: list[dict[str, object]] = []
    source = profile_group_analysis_h12.set_index(["entity_id", "criterion"])

    for strategy, profile_group_analysis_column in strategy_mapping.items():
        q = strategy_errors[strategy_errors["strategy"].eq(strategy)].set_index("meter_id")
        for metric, column in metric_column.items():
            computed = q[column].sort_index()
            expected = source.xs(metric, level="criterion")[profile_group_analysis_column].sort_index()
            difference = computed.to_numpy(float) - expected.to_numpy(float)
            rows.append({
                "audit_type": "strategy_error",
                "item": strategy,
                "criterion": metric,
                "meters": len(computed),
                "max_absolute_difference": float(np.max(np.abs(difference))),
                "mean_absolute_difference": float(np.mean(np.abs(difference))),
                "status": "PASS" if float(np.max(np.abs(difference))) <= REPRODUCTION_TOLERANCE[metric] else "FAIL",
            })

    for comparison_id, profile_group_analysis_column in gain_mapping.items():
        for metric in FORMAL_METRICS:
            q = pairwise_gains[
                pairwise_gains["comparison_id"].eq(comparison_id) & pairwise_gains["metric"].eq(metric)
            ].set_index("meter_id")
            expected = source.xs(metric, level="criterion")[profile_group_analysis_column].sort_index()
            difference = q["absolute_gain"].sort_index().to_numpy(float) - expected.to_numpy(float)
            rows.append({
                "audit_type": "primary_gain",
                "item": comparison_id,
                "criterion": metric,
                "meters": len(q),
                "max_absolute_difference": float(np.max(np.abs(difference))),
                "mean_absolute_difference": float(np.mean(np.abs(difference))),
                "status": "PASS" if float(np.max(np.abs(difference))) <= REPRODUCTION_TOLERANCE[metric] else "FAIL",
            })

    # Closure gains are audited against Profile group analysis strategy values rather than against a separate stored gain column.
    closure_pairs = {
        "fine_tuning_vs_limited": ("cer_scratch_limited_value_mean", "fine_tuning_value_mean"),
        "full_vs_fine_tuning": ("fine_tuning_value_mean", "cer_scratch_full_value_mean"),
    }
    for comparison_id, (comparator_column, focal_column) in closure_pairs.items():
        for metric in FORMAL_METRICS:
            q = pairwise_gains[
                pairwise_gains["comparison_id"].eq(comparison_id) & pairwise_gains["metric"].eq(metric)
            ].set_index("meter_id")
            expected_metric = source.xs(metric, level="criterion")
            expected = (expected_metric[comparator_column] - expected_metric[focal_column]).sort_index()
            difference = q["absolute_gain"].sort_index().to_numpy(float) - expected.to_numpy(float)
            rows.append({
                "audit_type": "closure_gain",
                "item": comparison_id,
                "criterion": metric,
                "meters": len(q),
                "max_absolute_difference": float(np.max(np.abs(difference))),
                "mean_absolute_difference": float(np.mean(np.abs(difference))),
                "status": "PASS" if float(np.max(np.abs(difference))) <= REPRODUCTION_TOLERANCE[metric] else "FAIL",
            })
    result = pd.DataFrame(rows)
    failed = result.loc[result["status"].ne("PASS")].copy()
    if not failed.empty:
        print("\n===== PROFILE-GROUP REPRODUCTION AUDIT: FAILED ROWS =====", file=sys.stderr)
        print(failed.to_string(index=False), file=sys.stderr)
        print("\n===== FULL REPRODUCTION AUDIT =====", file=sys.stderr)
        print(result.to_string(index=False), file=sys.stderr)
        raise RuntimeError("Profile group analysis strategy/gain reproduction audit failed")
    return result


def algebraic_identity_audit(pairwise_gains: pd.DataFrame) -> pd.DataFrame:
    pivot = pairwise_gains.pivot_table(index=["meter_id", "metric"], columns="comparison_id", values="absolute_gain", aggfunc="first")
    identities = {
        "fine_tuning_vs_limited_equals_source_plus_fine_tuning": pivot["fine_tuning_vs_limited"] - (pivot["source_transfer"] + pivot["fine_tuning"]),
        "direct_vs_full_equals_source_minus_full_data": pivot["direct_vs_full"] - (pivot["source_transfer"] - pivot["full_data"]),
        "full_vs_fine_tuning_equals_full_data_minus_ft_vs_limited": pivot["full_vs_fine_tuning"] - (pivot["full_data"] - pivot["fine_tuning_vs_limited"]),
    }
    rows: list[dict[str, object]] = []
    for identity, differences in identities.items():
        for metric in FORMAL_METRICS:
            values = differences.xs(metric, level="metric").to_numpy(float)
            maximum = float(np.max(np.abs(values)))
            rows.append({
                "identity": identity,
                "criterion": metric,
                "meters": len(values),
                "max_absolute_difference": maximum,
                "mean_absolute_difference": float(np.mean(np.abs(values))),
                "status": "PASS" if maximum <= TOLERANCE else "FAIL",
            })
    result = pd.DataFrame(rows)
    ensure((result["status"] == "PASS").all(), "Pairwise gain algebraic identity audit failed")
    return result


def gain_error_identity_audit(
    strategy_totals: pd.DataFrame,
    gain_totals: pd.DataFrame,
    strategy_rmse: pd.DataFrame,
    gain_rmse: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    strategy_index = strategy_totals.set_index(["assigned_group", "strategy", "metric"])
    for comparison in COMPARISONS:
        comparison_rows = gain_totals[gain_totals["comparison_id"].eq(comparison["comparison_id"])]
        for group in range(1, 5):
            for metric in ADDITIVE_METRICS:
                gain_row = comparison_rows[
                    comparison_rows["assigned_group"].eq(group) & comparison_rows["metric"].eq(metric)
                ].iloc[0]
                focal = strategy_index.loc[(group, comparison["focal_strategy"], metric)]
                comparator = strategy_index.loc[(group, comparison["comparator_strategy"], metric)]
                for estimate, strategy_column, gain_column in [
                    ("observed", "observed_group_error", "observed_group_gain"),
                    ("common_composition", "common_composition_standardised_error", "common_composition_standardised_gain"),
                    ("composition_only", "composition_only_expected_error", "composition_only_expected_gain"),
                ]:
                    difference = float(gain_row[gain_column] - (comparator[strategy_column] - focal[strategy_column]))
                    rows.append({
                        "comparison_id": comparison["comparison_id"],
                        "assigned_group": group,
                        "metric": metric,
                        "estimate": estimate,
                        "difference": difference,
                        "status": "PASS" if abs(difference) <= TOLERANCE else "FAIL",
                    })
    strategy_rmse_index = strategy_rmse.set_index(["assigned_group", "strategy"])
    for comparison in COMPARISONS:
        comparison_rows = gain_rmse[gain_rmse["comparison_id"].eq(comparison["comparison_id"])]
        for group in range(1, 5):
            gain_row = comparison_rows[comparison_rows["assigned_group"].eq(group)].iloc[0]
            focal = strategy_rmse_index.loc[(group, comparison["focal_strategy"])]
            comparator = strategy_rmse_index.loc[(group, comparison["comparator_strategy"])]
            for estimate, strategy_column, gain_column in [
                ("observed", "observed_rmse_derived_from_group_mse", "observed_group_gain"),
                ("common_composition", "common_composition_standardised_rmse", "common_composition_standardised_gain"),
                ("composition_only", "composition_only_expected_rmse", "composition_only_expected_gain"),
            ]:
                difference = float(gain_row[gain_column] - (comparator[strategy_column] - focal[strategy_column]))
                rows.append({
                    "comparison_id": comparison["comparison_id"],
                    "assigned_group": group,
                    "metric": "rmse",
                    "estimate": estimate,
                    "difference": difference,
                    "status": "PASS" if abs(difference) <= TOLERANCE else "FAIL",
                })
    result = pd.DataFrame(rows)
    ensure((result["status"] == "PASS").all(), "Strategy-error to gain identity audit failed")
    return result


def save_figures(
    figure_dir: Path,
    raw_summary: pd.DataFrame,
    gain_totals: pd.DataFrame,
    rmse_gains: pd.DataFrame,
    decomposition: pd.DataFrame,
) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    populations = ["G1", "G2", "G3", "G4"]
    x = np.arange(len(populations))
    bottom = np.zeros(len(populations))
    fig, ax = plt.subplots(figsize=(10, 6))
    for raw_bin in DESCRIPTIVE_BINS:
        values = [
            float(raw_summary[(raw_summary["population"].eq(population)) & (raw_summary["raw_bin"].eq(raw_bin))]["mean"].iloc[0])
            for population in populations
        ]
        ax.bar(x, values, bottom=bottom, label=DESCRIPTIVE_LABELS[raw_bin])
        bottom += np.asarray(values)
    ax.set_xticks(x, populations)
    ax.set_ylabel("Mean meter-level row share")
    ax.set_title("Load range analysis h=12 raw-load composition by fixed profile group")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    path = figure_dir / "load_range_analysis_h12_raw_load_composition_by_group.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    output_paths.append(path)

    for metric, source_frame, filename, ylabel in [
        ("mae", gain_totals, "load_range_analysis_h12_observed_vs_standardised_mae_all_comparisons.png", "MAE gain (kWh)"),
        ("rmse", rmse_gains, "load_range_analysis_h12_observed_vs_standardised_rmse_all_comparisons.png", "RMSE gain (kWh; derived from group MSE)"),
        ("smape", gain_totals, "load_range_analysis_h12_observed_vs_standardised_smape_all_comparisons.png", "SMAPE gain (percentage points)"),
    ]:
        q = source_frame[source_frame["metric"].eq(metric)].copy()
        q["label"] = q["comparison_id"] + "\nG" + q["assigned_group"].astype(str)
        order = pd.MultiIndex.from_product(
            [[x["comparison_id"] for x in COMPARISONS], [1, 2, 3, 4]],
            names=["comparison_id", "assigned_group"],
        )
        q = q.set_index(["comparison_id", "assigned_group"]).reindex(order).reset_index()
        positions = np.arange(len(q))
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(positions, q["observed_group_gain"], marker="o", label="Observed composition")
        ax.plot(positions, q["common_composition_standardised_gain"], marker="s", label="Common composition")
        labels = [f"{row.comparison_id}\nG{row.assigned_group}" for row in q.itertuples(index=False)]
        ax.set_xticks(positions, labels, rotation=55, ha="right")
        ax.axhline(0.0, linewidth=1)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Load range analysis h=12 observed versus common-composition-standardised {metric.upper()} gains")
        ax.legend()
        path = figure_dir / filename
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(path)

    q = decomposition[decomposition["metric"].eq("mae")].copy()
    q["label"] = q["comparison_id"] + "\n" + q["contrast"]
    positions = np.arange(len(q))
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.bar(positions, q["composition_component"], label="Composition component")
    ax.bar(positions, q["conditional_performance_component"], bottom=q["composition_component"], label="Conditional-performance component")
    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(positions, q["label"], rotation=65, ha="right")
    ax.set_ylabel("MAE gain difference (kWh)")
    ax.set_title("Load range analysis h=12 pairwise group decomposition across all strategy comparisons")
    ax.legend()
    path = figure_dir / "load_range_analysis_h12_pairwise_mae_composition_performance_decomposition_all_comparisons.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    output_paths.append(path)

    return output_paths


def write_decision_note(
    path: Path,
    complete_support_meters: int,
    profile_group_analysis_audit: pd.DataFrame,
) -> None:
    lines = [
        "# Load range analysis complete cross-strategy raw-load composition-standardised gain decomposition",
        "",
        "## Formal scope",
        "",
        "- Report-scoped h=12 cross-strategy load-range analysis using household-first aggregation, common-composition standardisation and complete-bin-support sensitivity.",
        "- Four strategies: Direct Transfer, Fine Tuning, CER Scratch Limited and CER Scratch Full.",
        "- Six unique pairwise strategy comparisons; four lock-controlled comparisons are primary and two algebraically related comparisons provide strategy closure.",
        "- MAE, RMSE and SMAPE remain separate. Additive composition decomposition uses MAE, MSE and SMAPE. RMSE is derived from separately standardised strategy MSE and is not treated as bin-additive.",
        "- Absolute and relative gain summaries are both retained. Relative gains are not used in the additive decomposition.",
        "- All 929 meters remain in the primary group-level analysis. The complete-four-bin individual sensitivity contains exactly 845 meters.",
        "- Read-only post-hoc analysis: no training, prediction modification, prototype refitting, reassignment, exclusion or causal claim.",
        "",
        "## Gain sign convention",
        "",
        "`Gain_E(A,B) = Error_B - Error_A`. Positive values mean focal strategy A has lower error than comparator B.",
        "",
        "## Six strategy comparisons",
        "",
    ]
    for comparison in COMPARISONS:
        lines.append(
            f"- `{comparison['comparison_id']}`: {STRATEGY_LABELS[comparison['focal_strategy']]} versus {STRATEGY_LABELS[comparison['comparator_strategy']]} ({comparison['interpretive_role']})."
        )
    lines.extend([
        "",
        "## Decomposition identity",
        "",
        "Observed group gain equals the sum across raw-load bins of the group's mean exposure share multiplied by its exposure-weighted within-bin conditional strategy gain. Common-composition standardisation replaces group exposure shares with the overall 929-meter CER shares while retaining each group's conditional response.",
        "",
        "## Formal audits",
        "",
        f"- Complete-support meters: {complete_support_meters}.",
        f"- Profile group analysis strategy and gain reproduction: {'PASS' if (profile_group_analysis_audit['status'] == 'PASS').all() else 'FAIL'}; maximum absolute difference {profile_group_analysis_audit['max_absolute_difference'].max():.3e}.",
        "",
        "## Interpretation boundary",
        "",
        "The analysis separates raw-load composition from within-bin conditional strategy response for every pairwise strategy contrast. It does not establish that profile group, load composition or strategy exposure causes an individual meter's outcome. The six pairwise contrasts are algebraically related and must not be treated as six independent replications. MAE, RMSE and SMAPE must be interpreted separately.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def inventory(root: Path, files: list[Path]) -> pd.DataFrame:
    rows = []
    for path in sorted(files):
        rows.append({
            "relative_path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    ensure(root.is_dir(), f"Project root missing: {root}")
    start = time.time()

    raw, profile_group_analysis_h12, all_runs, paths = check_inputs(
        root,
        args.input_all_runs,
        args.input_meter_raw,
        args.input_profile_group_analysis,
    )

    print("=" * 110)
    print("LOAD RANGE ANALYSIS COMPLETE CROSS-STRATEGY RAW-LOAD COMPOSITION-STANDARDISED GAIN DECOMPOSITION")
    print("=" * 110)
    print(f"Project root       : {root}")
    print(f"Canonical lock     : {paths['canonical_lock']}")
    print(f"Meters             : {len(raw)}")
    print(f"Selected all-runs  : {len(all_runs):,}")
    print(f"Strategies         : {len(STRATEGIES)}")
    print(f"Pairwise contrasts : {len(COMPARISONS)}")
    print("Mode               : read-only; meter first; h=12; report-scoped load-range analysis")

    if args.check_only:
        print("\nINPUT CHECK: PASS")
        for name, path in paths.items():
            print(f"  {name:25}: {path}")
        return

    table_dir = root / f"outputs/tables/{ANALYSIS_SLUG}"
    metadata_dir = root / f"outputs/metadata/{ANALYSIS_SLUG}"
    decision_dir = root / "documentation/technical_decisions"
    for directory in [table_dir, metadata_dir, decision_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    primary_all_runs = combine_primary_bins(all_runs, raw)
    strategy_bin = four_seed_strategy_bin(primary_all_runs)
    pairwise_bin = build_meter_bin_pairwise_gains(strategy_bin)
    all_row_strategy, all_row_gains, all_row_group_summary = build_all_row_outputs(all_runs, raw)
    raw_summary, raw_effects, raw_pairwise = raw_distribution_outputs(raw, args.permutations, args.bootstraps)

    strategy_components, strategy_totals, strategy_rmse = strategy_error_components(strategy_bin)
    gain_components_frame, gain_totals, _ = gain_components(pairwise_bin)
    gain_rmse = derive_rmse_gain_from_strategy_errors(None, strategy_rmse)
    pairwise_decomposition = pairwise_group_decomposition(gain_components_frame)

    complete_strategy, complete_gains, complete_effects, complete_summary = complete_support_outputs(
        strategy_bin, all_row_gains, args.permutations
    )
    bootstrap_additive, bootstrap_rmse = bootstrap_intervals(pairwise_bin, strategy_bin, args.bootstraps)

    profile_group_analysis_audit = profile_group_analysis_reproduction_audit(all_row_strategy, all_row_gains, profile_group_analysis_h12)
    algebraic_audit = algebraic_identity_audit(all_row_gains)
    strategy_gain_identity = gain_error_identity_audit(strategy_totals, gain_totals, strategy_rmse, gain_rmse)


    tables: dict[str, pd.DataFrame] = {
        "load_range_analysis_h12_meter_raw_load_bin_shares.csv": raw[[
            "meter_id", "assigned_group", "group_description", "all_observations",
            *[f"{raw_bin}_observations" for raw_bin in DESCRIPTIVE_BINS],
            *[f"{raw_bin}_row_share" for raw_bin in DESCRIPTIVE_BINS],
        ]],
        "load_range_analysis_h12_group_raw_load_bin_distribution_summary.csv": raw_summary,
        "load_range_analysis_h12_raw_load_bin_omnibus_effects.csv": raw_effects,
        "load_range_analysis_h12_raw_load_bin_pairwise_mean_differences.csv": raw_pairwise,
        "load_range_analysis_h12_meter_four_bin_strategy_metrics_all_runs.parquet": primary_all_runs,
        "load_range_analysis_h12_meter_four_bin_strategy_metrics_four_seed.csv": strategy_bin,
        "load_range_analysis_h12_meter_four_bin_pairwise_gains_four_seed.csv": pairwise_bin,
        "load_range_analysis_h12_meter_all_row_strategy_errors_four_seed.csv": all_row_strategy,
        "load_range_analysis_h12_meter_all_row_pairwise_gains_four_seed.csv": all_row_gains,
        "load_range_analysis_h12_group_all_row_pairwise_gain_summary.csv": all_row_group_summary,
        "load_range_analysis_h12_group_strategy_bin_conditional_error_and_contribution.csv": strategy_components,
        "load_range_analysis_h12_group_strategy_observed_vs_common_composition_error.csv": strategy_totals,
        "load_range_analysis_h12_group_strategy_derived_rmse_from_mse.csv": strategy_rmse,
        "load_range_analysis_h12_group_bin_conditional_gain_and_contribution.csv": gain_components_frame,
        "load_range_analysis_h12_group_observed_vs_common_composition_gain.csv": gain_totals,
        "load_range_analysis_h12_group_derived_rmse_gain_from_mse.csv": gain_rmse,
        "load_range_analysis_h12_pairwise_group_composition_performance_decomposition.csv": pairwise_decomposition,
        "load_range_analysis_h12_common_composition_bootstrap_intervals.csv": bootstrap_additive,
        "load_range_analysis_h12_derived_rmse_gain_bootstrap_intervals.csv": bootstrap_rmse,
        "load_range_analysis_h12_complete_bin_support_meter_standardised_strategy_errors.csv": complete_strategy,
        "load_range_analysis_h12_complete_bin_support_meter_standardised_pairwise_gains.csv": complete_gains,
        "load_range_analysis_h12_complete_bin_support_group_effects.csv": complete_effects,
        "load_range_analysis_h12_complete_bin_support_group_standardised_gain_summary.csv": complete_summary,
        "load_range_analysis_h12_profile_group_analysis_strategy_and_gain_reproduction_audit.csv": profile_group_analysis_audit,
        "load_range_analysis_h12_pairwise_gain_algebraic_identity_audit.csv": algebraic_audit,
        "load_range_analysis_h12_strategy_error_gain_identity_audit.csv": strategy_gain_identity,
    }

    written: list[Path] = []
    for filename, frame in tables.items():
        path = table_dir / filename
        if path.suffix == ".parquet":
            frame.to_parquet(path, index=False)
        else:
            frame.to_csv(path, index=False)
        written.append(path)

    decision_path = decision_dir / f"{ANALYSIS_SLUG}_decision.md"
    write_decision_note(decision_path, len(complete_strategy) // len(STRATEGIES), profile_group_analysis_audit)
    written.append(decision_path)

    status = {
        "analysis_id": ANALYSIS_ID,
        "analysis": "Load range analysis complete cross-strategy raw-load composition-standardised gain decomposition",
        "status": "COMPLETE_PASS",
        "canonical_lock_path": str(paths["canonical_lock"]),
        "lead": LEAD,
        "meters": EXPECTED_METERS,
        "group_counts": EXPECTED_GROUP_COUNTS,
        "formal_seeds": FORMAL_SEEDS,
        "strategies": STRATEGIES,
        "pairwise_comparisons": [x["comparison_id"] for x in COMPARISONS],
        "selected_all_run_rows": len(all_runs),
        "primary_strategy_seed_bin_rows": len(primary_all_runs),
        "four_seed_strategy_meter_bin_rows": len(strategy_bin),
        "four_seed_pairwise_meter_bin_rows": len(pairwise_bin),
        "all_row_strategy_meter_rows": len(all_row_strategy),
        "all_row_pairwise_gain_rows": len(all_row_gains),
        "conditional_gain_component_rows": len(gain_components_frame),
        "pairwise_group_decomposition_rows": len(pairwise_decomposition),
        "complete_bin_support_meters": len(complete_strategy) // len(STRATEGIES),
        "permutations": args.permutations,
        "bootstraps": args.bootstraps,
        "all_meters_retained_primary": True,
        "training_performed": False,
        "predictions_modified": False,
        "prototypes_refitted": False,
        "meters_reassigned": False,
        "meters_excluded_primary": False,
        "profile_group_analysis_reproduction": "PASS",
        "interpretation": "descriptive and non-causal",
        "elapsed_seconds": time.time() - start,
    }
    status_path = metadata_dir / f"{ANALYSIS_SLUG}_status.json"
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    written.append(status_path)

    output_inventory = inventory(root, written)
    inventory_path = metadata_dir / "load_range_analysis_output_inventory.csv"
    output_inventory.to_csv(inventory_path, index=False)

    print("\n" + "=" * 110)
    print("LOAD RANGE ANALYSIS COMPLETE — PASS")
    print("=" * 110)
    print(f"Meters retained                       : {EXPECTED_METERS}")
    print(f"Selected seed-level input rows        : {len(all_runs):,}")
    print(f"Four-seed strategy-meter-bin rows     : {len(strategy_bin):,}")
    print(f"Four-seed pairwise meter-bin rows     : {len(pairwise_bin):,}")
    print(f"All-row pairwise gain rows            : {len(all_row_gains):,}")
    print(f"Pairwise group decomposition rows     : {len(pairwise_decomposition):,}")
    print(f"Complete-bin sensitivity meters       : {len(complete_strategy) // len(STRATEGIES)}")
    print(f"Profile group analysis reproduction                 : PASS")
    print(f"Output inventory rows                 : {len(output_inventory)}")
    print("Training performed                    : False")
    print("Predictions modified                  : False")
    print("Final status                          : COMPLETE_PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"LOAD RANGE ANALYSIS FAILED: {error}", file=sys.stderr)
        raise
