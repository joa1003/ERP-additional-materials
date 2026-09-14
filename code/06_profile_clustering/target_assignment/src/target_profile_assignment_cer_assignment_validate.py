from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from target_profile_assignment_cer_assignment_common import (
    ensure,
    load_config,
    load_fixed_prototypes,
    resolve_path,
    sha256_file,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Target profile assignment CER assignments")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    table_dir = resolve_path(config, config["outputs"]["table_dir"])
    metadata_dir = resolve_path(config, config["outputs"]["metadata_dir"])
    decision_path = resolve_path(config, config["outputs"]["decision_md"])

    prototype_path = resolve_path(config, config["inputs"]["source_prototypes_csv"])
    _, _, groups = load_fixed_prototypes(config)

    profiles = pd.read_parquet(table_dir / "target_profile_assignment_cer_training_profiles.parquet")
    assignments = pd.read_parquet(table_dir / "target_profile_assignment_cer_source_defined_assignments.parquet")
    distances = pd.read_csv(table_dir / "target_profile_assignment_cer_to_source_prototype_dtw_distances.csv")
    group_summary = pd.read_csv(table_dir / "target_profile_assignment_cer_assignment_group_summary.csv")
    audit = pd.read_csv(table_dir / "target_profile_assignment_cer_profile_construction_audit.csv")

    expected_meters = int(config["expected"]["cer_meters"])
    ensure(len(assignments) == expected_meters, f"Expected {expected_meters} assignment rows")
    ensure(assignments["meter_id"].nunique() == expected_meters, "Assignment meter IDs are not unique and complete")
    ensure(len(profiles) == expected_meters * 48, "Expected 48 profile rows per CER meter")
    ensure(not profiles.duplicated(["meter_id", "local_slot"]).any(), "Duplicate meter-slot profile rows")
    ensure(profiles.groupby("meter_id")["local_slot"].nunique().eq(48).all(), "Not every CER meter has 48 slots")
    ensure(profiles["local_slot"].between(1, 48).all(), "Profile slots outside 1 to 48")
    ensure(np.isfinite(profiles["raw_mean_kwh"].to_numpy(dtype=float)).all(), "Non-finite raw profile values")
    ensure(np.isfinite(profiles["profile_zscore"].to_numpy(dtype=float)).all(), "Non-finite profile z-scores")

    z_stats = profiles.groupby("meter_id")["profile_zscore"].agg(["mean", lambda values: values.std(ddof=0)])
    z_stats.columns = ["mean", "std_ddof0"]
    ensure(np.abs(z_stats["mean"]).max() < 1e-10, "Per-profile z-score means are not zero")
    ensure(np.abs(z_stats["std_ddof0"] - 1.0).max() < 1e-10, "Per-profile z-score standard deviations are not one")

    ensure(assignments["assigned_group"].isin(groups).all(), "Assignment contains an invalid group")
    ensure(assignments["second_nearest_group"].isin(groups).all(), "Second-nearest group contains an invalid group")
    ensure((assignments["assigned_group"] != assignments["second_nearest_group"]).all(),
           "Nearest and second-nearest groups must differ")
    ensure((assignments["nearest_dtw_distance"] <= assignments["second_nearest_dtw_distance"] + 1e-12).all(),
           "Nearest distance exceeds second-nearest distance")
    expected_margin = assignments["second_nearest_dtw_distance"] - assignments["nearest_dtw_distance"]
    ensure(np.allclose(assignments["distance_margin"], expected_margin, rtol=0, atol=1e-12),
           "Distance margin calculation mismatch")
    expected_ratio = assignments["nearest_dtw_distance"] / assignments["second_nearest_dtw_distance"]
    ensure(np.allclose(assignments["distance_ratio"], expected_ratio, rtol=0, atol=1e-12),
           "Distance ratio calculation mismatch")
    ensure(assignments["distance_ratio"].between(0, 1).all(), "Distance ratio must lie in [0, 1]")
    ensure((assignments["source_prototypes_refitted"] == False).all(), "Prototype refitting flag must be false")  # noqa: E712
    ensure((assignments["receiving_profiles_used_in_k_selection"] == False).all(),
           "Receiving profiles must not enter K selection")  # noqa: E712
    ensure((assignments["forecasting_metrics_used_for_assignment"] == False).all(),
           "Forecasting metrics must not enter assignment")  # noqa: E712

    ensure(len(distances) == expected_meters and distances["meter_id"].nunique() == expected_meters,
           "Distance matrix must contain one row per meter")
    distance_columns = [f"dtw_distance_group_{group}" for group in groups]
    ensure(set(distance_columns).issubset(distances.columns), "Distance matrix missing group columns")
    ordered_distances = distances.set_index("meter_id").loc[assignments["meter_id"], distance_columns].to_numpy(dtype=float)
    argmin_groups = np.asarray(groups, dtype=int)[np.argmin(ordered_distances, axis=1)]
    ensure(np.array_equal(argmin_groups, assignments["assigned_group"].to_numpy(dtype=int)),
           "Assigned group does not match distance-matrix argmin")

    ensure(int(group_summary["meter_count"].sum()) == expected_meters, "Group counts do not sum to 929")
    ensure(np.isclose(group_summary["meter_share"].sum(), 1.0), "Group shares do not sum to one")
    ensure(set(group_summary["profile_group"].astype(int)) == set(groups), "Group summary must contain Groups 1 to 4")

    ensure(int(audit.loc[0, "cer_meters"]) == expected_meters, "Profile audit meter count mismatch")
    ensure(int(audit.loc[0, "profile_rows"]) == expected_meters * 48, "Profile audit row count mismatch")
    ensure(int(audit.loc[0, "validation_rows_used"]) == 0, "Validation rows were used")
    ensure(int(audit.loc[0, "test_rows_used"]) == 0, "Test rows were used")
    ensure(int(audit.loc[0, "forecasting_metrics_used"]) == 0, "Forecasting metrics were used")
    ensure(int(audit.loc[0, "zero_variation_profile_count"]) == 0, "A receiving profile has zero variation")

    prototype_hash = sha256_file(prototype_path)
    status_path = metadata_dir / "target_profile_assignment_cer_assignment_status.json"
    import json
    status = json.loads(status_path.read_text(encoding="utf-8"))
    ensure(status["source_prototype_sha256_before"] == prototype_hash, "Prototype hash differs from analysis start")
    ensure(status["source_prototype_sha256_after"] == prototype_hash, "Prototype hash differs from analysis end")

    status.update(
        {
            "status": "COMPLETE_PASS",
            "validation_pass": True,
            "validated_assignment_rows": int(len(assignments)),
            "validated_profile_rows": int(len(profiles)),
            "validated_group_count_sum": int(group_summary["meter_count"].sum()),
            "unassigned_meters": int(assignments["assigned_group"].isna().sum()),
        }
    )
    write_json(status_path, status)

    group_lines = "\n".join(
        f"- Group {int(row.profile_group)}: {int(row.meter_count)} meters ({100 * row.meter_share:.2f}%)"
        for row in group_summary.itertuples(index=False)
    )
    decision_text = f"""# Target profile assignment CER Assignment to Fixed Source Profiles\n\n## Final status\n\n```text\nTARGET PROFILE ASSIGNMENT = COMPLETE — PASS\nFixed source K = 4\nSource seed metadata = 71\nSource prototypes refitted = no\nCER meters assigned = 929\nUnassigned meters = 0\nValidation rows used = 0\nTest rows used = 0\nForecasting metrics used = 0\n```\n\n## Governed method\n\nThe fixed Profile clustering K = 4 source prototypes were read without modification. All receiving profiles were constructed from the CER forecasting training split, from `2009-07-15 00:00` to `2010-07-24 11:30`, using local slots 1 to 48. The irregular local dates configured in the experiment lock were excluded from profile averaging when they fell inside the profile period. Each meter profile was standardised independently using a z score with `ddof = 0` and assigned to the nearest fixed source prototype under DTW.\n\nNo receiving profile contributed to K selection or prototype fitting. No forecasting error, Transfer Gain, validation load, test load or survey variable contributed to assignment. All 929 meters retained a final source-defined profile group.\n\nDistance definitions:\n\n```text\ndistance margin = second nearest DTW distance - nearest DTW distance\ndistance ratio = nearest DTW distance / second nearest DTW distance\n```\n\nA larger margin and a smaller ratio indicate clearer separation from the second-nearest source profile. These values are interpretation variables only and are not forecasting inputs.\n\n## Final receiving group counts\n\n{group_lines}\n\n## Formal outputs\n\n- `{table_dir / 'target_profile_assignment_cer_source_defined_assignments.csv'}`\n- `{table_dir / 'target_profile_assignment_cer_source_defined_assignments.parquet'}`\n- `{table_dir / 'target_profile_assignment_cer_training_profiles.parquet'}`\n- `{table_dir / 'target_profile_assignment_cer_to_source_prototype_dtw_distances.csv'}`\n- `{table_dir / 'target_profile_assignment_cer_assignment_group_summary.csv'}`\n- `{metadata_dir / 'target_profile_assignment_cer_assignment_status.json'}`\n\n## Next use\n\nJoin the 929-row assignment table to the completed Household analytical table meter-level analytical table by `meter_id`. That later join supports RQ3 profile-group and DTW-similarity analysis. Target profile assignment itself does not evaluate forecasting performance.\n"""
    decision_path.write_text(decision_text, encoding="utf-8")

    print("=" * 96)
    print("TARGET PROFILE ASSIGNMENT CER ASSIGNMENT OUTPUT VALIDATION PASS")
    print("=" * 96)
    print(f"CER assignments             : {len(assignments)}")
    print("Unassigned meters           : 0")
    print(f"Profile rows                : {len(profiles)}")
    print(f"Group count sum             : {int(group_summary['meter_count'].sum())}")
    print(f"Prototype SHA256 unchanged  : {prototype_hash}")
    print(f"Final status                : COMPLETE_PASS")


if __name__ == "__main__":
    main()
