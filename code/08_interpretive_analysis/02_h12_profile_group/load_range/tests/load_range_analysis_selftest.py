#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd

RELEASE = Path(__file__).resolve().parents[1]
SOURCE = RELEASE / "src/load_range_analysis_complete_cross_strategy_raw_load_composition_standardised_gain_decomposition.py"
spec = importlib.util.spec_from_file_location("load_range_analysis_main", SOURCE)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def test_gain_sign_and_relative() -> None:
    gain = 0.20 - 0.15
    ensure(abs(gain - 0.05) < 1e-12, "Gain sign convention failed")
    relative = module.safe_relative_gain(gain, 0.20)
    ensure(abs(relative - 25.0) < 1e-12, "Relative gain formula failed")
    ensure(math.isnan(module.safe_relative_gain(1.0, 0.0)), "Zero comparator error must produce undefined relative gain")


def test_effect_sizes() -> None:
    values = np.array([0.0, 0.1, 1.0, 1.1, 2.0, 2.1, 3.0, 3.1])
    groups = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    eta, omega = module.eta_omega(values, groups)
    ensure(0 <= eta <= 1, "Eta squared outside [0,1]")
    ensure(0 <= omega <= 1, "Omega squared outside [0,1]")


def test_pairwise_decomposition_identity() -> None:
    rows = []
    for comparison_index, comparison in enumerate(module.COMPARISONS):
        for group in range(1, 5):
            shares = np.array([0.20, 0.35, 0.25, 0.20]) + (group - 2.5) * np.array([0.005, -0.003, -0.001, -0.001])
            shares = shares / shares.sum()
            for metric_index, metric in enumerate(module.ADDITIVE_METRICS):
                for bin_index, raw_bin in enumerate(module.PRIMARY_BINS):
                    conditional = (
                        0.01 * (comparison_index + 1)
                        + 0.002 * group
                        + 0.001 * bin_index
                        + 0.0001 * metric_index
                    )
                    rows.append({
                        "comparison_id": comparison["comparison_id"],
                        "comparison_label": comparison["comparison_label"],
                        "interpretive_role": comparison["interpretive_role"],
                        "focal_strategy": comparison["focal_strategy"],
                        "comparator_strategy": comparison["comparator_strategy"],
                        "assigned_group": group,
                        "population": f"G{group}",
                        "group_description": module.GROUP_DESCRIPTIONS[group],
                        "raw_bin": raw_bin,
                        "raw_bin_label": module.PRIMARY_LABELS[raw_bin],
                        "metric": metric,
                        "unit": module.ADDITIVE_METRICS[metric]["unit"],
                        "mean_meter_row_share": shares[bin_index],
                        "exposure_weighted_conditional_gain": conditional,
                        "observed_gain_contribution": shares[bin_index] * conditional,
                        "meters_with_bin": 2,
                        "total_meters": 2,
                    })
    frame = pd.DataFrame(rows)
    result = module.pairwise_group_decomposition(frame)
    ensure(len(result) == 108, "Unexpected synthetic pairwise decomposition count")
    ensure(float(result["reconstruction_difference"].abs().max()) <= 1e-12, "Pairwise reconstruction failed")


def test_derived_rmse_gain() -> None:
    rows = []
    for group in range(1, 5):
        for strategy_index, strategy in enumerate(module.STRATEGIES):
            observed = 0.04 + 0.01 * strategy_index + 0.001 * group
            standardised = observed + 0.002
            composition = observed - 0.001
            rows.append({
                "assigned_group": group,
                "group_description": module.GROUP_DESCRIPTIONS[group],
                "strategy": strategy,
                "strategy_label": module.STRATEGY_LABELS[strategy],
                "metric": "rmse",
                "unit": "kWh",
                "observed_rmse_derived_from_group_mse": math.sqrt(observed),
                "common_composition_standardised_rmse": math.sqrt(standardised),
                "composition_only_expected_rmse": math.sqrt(composition),
                "standardisation_change": math.sqrt(standardised) - math.sqrt(observed),
                "scope": "synthetic",
            })
    strategy_rmse = pd.DataFrame(rows)
    gains = module.derive_rmse_gain_from_strategy_errors(None, strategy_rmse)
    ensure(len(gains) == 24, "Unexpected synthetic RMSE gain count")
    source = gains[(gains["comparison_id"] == "source_transfer") & (gains["assigned_group"] == 1)].iloc[0]
    indexed = strategy_rmse.set_index(["assigned_group", "strategy"])
    expected = (
        indexed.loc[(1, "cer_scratch_limited"), "observed_rmse_derived_from_group_mse"]
        - indexed.loc[(1, "direct_transfer"), "observed_rmse_derived_from_group_mse"]
    )
    ensure(abs(source["observed_group_gain"] - expected) <= 1e-12, "Derived RMSE sign identity failed")


def main() -> None:
    test_gain_sign_and_relative()
    test_effect_sizes()
    test_pairwise_decomposition_identity()
    test_derived_rmse_gain()
    print("LOAD RANGE ANALYSIS SELFTEST: PASS")


if __name__ == "__main__":
    main()
