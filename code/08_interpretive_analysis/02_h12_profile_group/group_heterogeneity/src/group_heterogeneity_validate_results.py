#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ANALYSIS_ID = "group_heterogeneity"
ANALYSIS_NAME = "group_heterogeneity_group_heterogeneity_practical_significance"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().absolute()
    tables = root / f"outputs/tables/{ANALYSIS_NAME}"
    metadata = root / f"outputs/metadata/{ANALYSIS_NAME}"
    decision = root / f"documentation/technical_decisions/{ANALYSIS_NAME}_decision.md"
    status_path = metadata / f"{ANALYSIS_NAME}_status.json"
    inventory_path = metadata / "group_heterogeneity_output_inventory.csv"

    status = json.loads(status_path.read_text(encoding="utf-8"))
    expected_status = {
        "analysis_id": ANALYSIS_ID,
        "status": "COMPLETE_PASS",
        "meters": 929,
        "profile_group_analysis_gain_rows": 8361,
        "raw_load_structure_raw_rows": 929,
        "source_transfer_omnibus_rows": 18,
        "h12_pairwise_rows": 36,
        "positive_share_contrast_rows": 18,
        "predictive_utility_rows": 6,
        "other_gain_omnibus_rows": 18,
        "lead_shift_rows": 18,
        "raw_effect_rows": 10,
        "training_performed": False,
        "predictions_modified": False,
        "prototypes_refitted": False,
        "meters_reassigned": False,
        "meters_excluded": False,
        "causal_claims": False,
    }
    for key, expected in expected_status.items():
        actual = status.get(key)
        assert actual == expected, f"Status mismatch for {key}: expected {expected!r}, found {actual!r}"

    expected_rows = {
        "group_heterogeneity_meter_transferability_dataset.csv": 8361,
        "group_heterogeneity_source_transfer_group_descriptive_summary.csv": 72,
        "group_heterogeneity_source_transfer_omnibus_effects.csv": 18,
        "group_heterogeneity_h12_source_transfer_pairwise_contrasts.csv": 36,
        "group_heterogeneity_h12_source_transfer_positive_share_contrasts.csv": 18,
        "group_heterogeneity_h12_group_predictive_utility.csv": 6,
        "group_heterogeneity_h12_other_gain_omnibus_effects.csv": 18,
        "group_heterogeneity_source_transfer_lead_shift_group_effects.csv": 18,
        "group_heterogeneity_raw_group_characterisation_summary.csv": 50,
        "group_heterogeneity_raw_group_effects.csv": 10,
        "group_heterogeneity_h12_primary_evidence_table.csv": 8,
    }
    loaded: dict[str, pd.DataFrame] = {}
    for filename, expected in expected_rows.items():
        path = tables / filename
        assert path.is_file() and path.stat().st_size > 0, f"Missing/empty output: {path}"
        df = pd.read_csv(path)
        loaded[filename] = df
        assert len(df) == expected, f"{filename}: expected {expected} rows, found {len(df)}"
        print(f"PASS row count: {filename} = {len(df):,}")

    transfer = loaded["group_heterogeneity_meter_transferability_dataset.csv"]
    assert transfer["entity_id"].nunique() == 929
    assert sorted(transfer["lead"].unique().tolist()) == [1, 12, 48]
    assert sorted(transfer["criterion"].unique().tolist()) == ["mae", "rmse", "smape"]
    assert transfer["seed_count"].eq(4).all()

    primary = loaded["group_heterogeneity_h12_primary_evidence_table.csv"]
    abs_mae = primary[primary["scale"] == "absolute"].sort_values("assigned_group")
    expected_means = np.array([0.008153, 0.015309, 0.018369, 0.024697])
    assert np.allclose(abs_mae["mean"].to_numpy(), expected_means, atol=5e-7), abs_mae[["assigned_group", "mean"]]
    assert abs_mae["meters"].tolist() == [141, 325, 324, 139]

    omnibus = loaded["group_heterogeneity_source_transfer_omnibus_effects.csv"]
    assert omnibus["eta_squared"].between(0, 1).all()
    assert omnibus["permutation_p_value"].between(0, 1).all()
    assert omnibus["bh_q_value"].between(0, 1).all()

    pairwise = loaded["group_heterogeneity_h12_source_transfer_pairwise_contrasts.csv"]
    g1g4 = pairwise[(pairwise["criterion"] == "mae") & (pairwise["scale"] == "absolute") &
                    (pairwise["group_a"] == 1) & (pairwise["group_b"] == 4)]
    assert len(g1g4) == 1
    assert abs(float(g1g4.iloc[0]["mean_difference_b_minus_a"]) - 0.016544) < 5e-7

    utility = loaded["group_heterogeneity_h12_group_predictive_utility.csv"]
    assert utility["meters"].eq(929).all()
    assert utility["positive_gain_auc_mean"].between(0, 1).all()

    raw = loaded["group_heterogeneity_raw_group_effects.csv"]
    assert raw["eta_squared"].between(0, 1).all()
    assert set(raw["variable"]) == {
        "test_actual_mean_kwh", "test_actual_std_kwh_ddof0", "actual_eq_0_row_share",
        "actual_gt_0_le_0_1_row_share", "actual_gt_0_1_le_0_5_row_share",
        "actual_gt_0_5_le_1_0_row_share", "actual_gt_1_0_row_share",
        "top_10pct_row_share", "top_5pct_row_share", "top_1pct_row_share",
    }

    assert decision.is_file() and decision.stat().st_size > 0, f"Missing decision note: {decision}"

    inventory = pd.read_csv(inventory_path)
    assert len(inventory) == 13, f"Expected 13 inventory rows, found {len(inventory)}"
    for row in inventory.itertuples(index=False):
        path = root / row.relative_path
        assert path.is_file(), f"Missing inventory file: {path}"
        assert path.stat().st_size == int(row.bytes), f"Size mismatch: {path}"
        assert sha256(path) == row.sha256, f"SHA-256 mismatch: {path}"

    print("=" * 100)
    print("GROUP HETEROGENEITY OUTPUT VALIDATION: PASS")
    print("=" * 100)
    print("Meters                         : 929")
    print("Source-transfer omnibus rows : 18")
    print("h=12 pairwise rows           : 36")
    print("Predictive-utility rows      : 6")
    print("Raw-characteristic effects   : 10")
    print("Inventory hashes             : 13/13 PASS")
    print("Final status                 : COMPLETE_PASS")


if __name__ == "__main__":
    main()
