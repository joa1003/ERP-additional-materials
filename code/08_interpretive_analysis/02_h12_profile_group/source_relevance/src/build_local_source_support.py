#!/usr/bin/env python3
"""Build the local-source-support features used by the reported source-relevance analysis.

For each target household, this producer keeps the locked K=4 source-defined group,
compares the target 48-slot z-standardised training profile with actual source-household
profiles in that group using clock-aligned RMSE, and derives nearest-source support
measures. It then attaches the already generated household strategy errors and gains.

This is a post-hoc interpretive producer. It does not train a forecast model, refit a
prototype, or change any target household assignment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

EXPECTED_SOURCE = 3843
EXPECTED_TARGET = 929
EXPECTED_GROUP_COUNTS = {1: 141, 2: 325, 3: 324, 4: 139}
LEADS = [1, 12, 48]
CRITERIA = ["mae", "rmse", "smape"]
STRATEGY_VALUE_COLUMNS = {
    "direct_transfer": "direct_transfer_value_mean",
    "fine_tuning": "fine_tuning_value_mean",
    "cer_scratch_limited": "cer_scratch_limited_value_mean",
    "cer_scratch_full": "cer_scratch_full_value_mean",
}
COMPARISONS = [
    ("source_transfer", "direct_transfer", "cer_scratch_limited"),
    ("fine_tuning", "fine_tuning", "direct_transfer"),
    ("full_data", "cer_scratch_full", "cer_scratch_limited"),
    ("direct_vs_full", "direct_transfer", "cer_scratch_full"),
    ("fine_tuning_vs_limited", "fine_tuning", "cer_scratch_limited"),
    ("full_vs_fine_tuning", "cer_scratch_full", "fine_tuning"),
]


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def norm_id(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(gain: np.ndarray, comparator_error: np.ndarray) -> np.ndarray:
    out = np.full_like(np.asarray(gain, float), np.nan, dtype=float)
    denominator = np.asarray(comparator_error, float)
    np.divide(100.0 * np.asarray(gain, float), denominator, out=out, where=denominator != 0)
    return out


def read_source_assignments(root: Path) -> pd.DataFrame:
    base = root / "outputs/tables/profile_clustering_clustering_final_decision_dissertation/profile_clustering_final_selected_source_assignments"
    parquet = base.with_suffix(".parquet")
    csv = base.with_suffix(".csv")
    if parquet.is_file():
        frame = pd.read_parquet(parquet)
    elif csv.is_file():
        frame = pd.read_csv(csv)
    else:
        raise FileNotFoundError(f"Missing final source assignments: {parquet} or {csv}")
    required = {"household_id", "profile_group"}
    ensure(required.issubset(frame.columns), f"Source assignments missing columns: {sorted(required-set(frame.columns))}")
    frame = frame[["household_id", "profile_group"]].copy()
    frame["household_id"] = norm_id(frame["household_id"])
    frame["profile_group"] = pd.to_numeric(frame["profile_group"], errors="raise").astype(int)
    ensure(len(frame) == EXPECTED_SOURCE and frame["household_id"].nunique() == EXPECTED_SOURCE,
           "Final source assignments must contain all 3,843 source households")
    return frame


def load_source_profiles(root: Path, assignments: pd.DataFrame) -> pd.DataFrame:
    path = root / "outputs/tables/profile_clustering/profile_construction/source_training_profiles_zscore.csv"
    ensure(path.is_file(), f"Missing source training profiles: {path}")
    frame = pd.read_csv(path)
    ensure("household_id" in frame.columns, "Source z-profile table is missing household_id")
    frame["household_id"] = norm_id(frame["household_id"])
    slot_columns = [f"slot_{slot:02d}" for slot in range(1, 49)]
    if not set(slot_columns).issubset(frame.columns):
        # The canonical profile table uses slot_01..slot_48; retain a defensive fallback for numeric-slot exports.
        alternatives = [str(slot) for slot in range(1, 49)]
        ensure(set(alternatives).issubset(frame.columns), "Source z-profile table does not contain 48 recognised slot columns")
        slot_columns = alternatives
    merged = frame[["household_id", *slot_columns]].merge(assignments, on="household_id", how="inner", validate="one_to_one")
    ensure(len(merged) == EXPECTED_SOURCE, "Source profile/assignment join did not retain all 3,843 households")
    values = merged[slot_columns].to_numpy(float)
    ensure(values.shape == (EXPECTED_SOURCE, 48) and np.isfinite(values).all(), "Invalid source profile matrix")
    return merged.rename(columns={column: f"z_{index+1:02d}" for index, column in enumerate(slot_columns)})


def load_target_profiles(root: Path) -> pd.DataFrame:
    profile_path = root / "outputs/tables/target_profile_assignment_cer_assignment/target_profile_assignment_cer_training_profiles.csv"
    assignment_path = root / "outputs/tables/target_profile_assignment_cer_assignment/target_profile_assignment_cer_source_defined_assignments.csv"
    ensure(profile_path.is_file(), f"Missing target training profiles: {profile_path}")
    ensure(assignment_path.is_file(), f"Missing target profile assignments: {assignment_path}")
    profiles = pd.read_csv(profile_path, usecols=["meter_id", "local_slot", "profile_zscore"])
    profiles["meter_id"] = norm_id(profiles["meter_id"])
    profiles["local_slot"] = pd.to_numeric(profiles["local_slot"], errors="raise").astype(int)
    matrix = profiles.pivot(index="meter_id", columns="local_slot", values="profile_zscore").reindex(columns=range(1,49))
    ensure(matrix.shape == (EXPECTED_TARGET, 48) and not matrix.isna().any().any(), "Target profile matrix must be 929 × 48")
    assignments = pd.read_csv(assignment_path)
    required = {"meter_id", "assigned_group", "nearest_dtw_distance", "distance_margin", "distance_ratio"}
    ensure(required.issubset(assignments.columns), f"Target assignments missing columns: {sorted(required-set(assignments.columns))}")
    assignments["meter_id"] = norm_id(assignments["meter_id"])
    assignments["assigned_group"] = pd.to_numeric(assignments["assigned_group"], errors="raise").astype(int)
    counts = assignments["assigned_group"].value_counts().sort_index().to_dict()
    ensure(counts == EXPECTED_GROUP_COUNTS, f"Unexpected target group counts: {counts}")
    out = assignments[["meter_id", "assigned_group", "nearest_dtw_distance", "distance_margin", "distance_ratio"]].copy()
    for slot in range(1,49):
        out[f"z_{slot:02d}"] = out["meter_id"].map(matrix[slot])
    ensure(np.isfinite(out[[f"z_{slot:02d}" for slot in range(1,49)]].to_numpy(float)).all(), "Non-finite target z profiles")
    return out


def load_prototypes(root: Path) -> dict[int, np.ndarray]:
    path = root / "outputs/tables/profile_clustering_clustering_final_decision_dissertation/profile_clustering_final_selected_source_prototypes.csv"
    ensure(path.is_file(), f"Missing final source prototypes: {path}")
    frame = pd.read_csv(path)
    required = {"profile_group", "local_slot", "prototype_zscore"}
    ensure(required.issubset(frame.columns), f"Prototype table missing columns: {sorted(required-set(frame.columns))}")
    result = {}
    for group, part in frame.groupby("profile_group", sort=True):
        part = part.sort_values("local_slot")
        ensure(part["local_slot"].astype(int).tolist() == list(range(1,49)), f"Prototype G{group} lacks slots 1..48")
        result[int(group)] = part["prototype_zscore"].to_numpy(float)
    ensure(sorted(result) == [1,2,3,4], "Expected four fixed source prototypes")
    return result


def compute_features(source: pd.DataFrame, target: pd.DataFrame, prototypes: dict[int, np.ndarray]) -> pd.DataFrame:
    slot_columns = [f"z_{slot:02d}" for slot in range(1,49)]
    rows: list[dict] = []
    for group in [1,2,3,4]:
        source_group = source[source["profile_group"].eq(group)].copy()
        target_group = target[target["assigned_group"].eq(group)].copy()
        s = source_group[slot_columns].to_numpy(float)
        t = target_group[slot_columns].to_numpy(float)
        ensure(len(s) >= 25, f"Source group G{group} is too small for kNN-25 support")

        # Clock-aligned RMSE_z is Euclidean distance divided by sqrt(48).
        target_source = cdist(t, s, metric="euclidean") / math.sqrt(48.0)
        ordered = np.sort(target_source, axis=1)
        target_knn10 = ordered[:, :10].mean(axis=1)
        target_knn25 = ordered[:, :25].mean(axis=1)

        source_pair = cdist(s, s, metric="euclidean") / math.sqrt(48.0)
        np.fill_diagonal(source_pair, np.inf)
        source_ordered = np.sort(source_pair, axis=1)
        source_knn10 = source_ordered[:, :10].mean(axis=1)

        prototype = prototypes[group]
        target_proto_clock = np.sqrt(np.mean((t - prototype[None, :]) ** 2, axis=1))
        source_proto_clock = np.sqrt(np.mean((s - prototype[None, :]) ** 2, axis=1))

        support_percentile = np.asarray([
            100.0 * np.mean(source_knn10 >= value) for value in target_knn10
        ])
        centrality_percentile = np.asarray([
            100.0 * np.mean(source_proto_clock >= value) for value in target_proto_clock
        ])

        for index, row in enumerate(target_group.itertuples(index=False)):
            rows.append({
                "meter_id": str(row.meter_id),
                "assigned_group": group,
                "nearest_dtw_distance": float(row.nearest_dtw_distance),
                "distance_margin": float(row.distance_margin),
                "distance_ratio": float(row.distance_ratio),
                "source_knn10_mean_rmse_z": float(target_knn10[index]),
                "source_knn25_mean_rmse_z": float(target_knn25[index]),
                "source_support_percentile": float(support_percentile[index]),
                "source_prototype_centrality_percentile": float(centrality_percentile[index]),
                "assigned_clock_rmse_z": float(target_proto_clock[index]),
            })
    out = pd.DataFrame(rows).sort_values("meter_id").reset_index(drop=True)
    ensure(len(out) == EXPECTED_TARGET and out["meter_id"].nunique() == EXPECTED_TARGET, "Local source support feature table must contain 929 households")
    ensure(np.isfinite(out.drop(columns=["meter_id"]).to_numpy(float)).all(), "Non-finite source-support features")
    return out


def build_outcome(root: Path, features: pd.DataFrame) -> pd.DataFrame:
    path = root / "outputs/tables/profile_group_analysis_profile_group_analysis/profile_group_analysis_native_meter_gains_with_group.csv"
    ensure(path.is_file(), f"Missing household/profile-group gain table: {path}")
    gains = pd.read_csv(path)
    required = {"entity_id", "lead", "criterion", "support_scope", "assigned_group", "group_description", "seed_count", *STRATEGY_VALUE_COLUMNS.values()}
    ensure(required.issubset(gains.columns), f"Household gain table missing columns: {sorted(required-set(gains.columns))}")
    gains["meter_id"] = norm_id(gains["entity_id"])
    gains["lead"] = pd.to_numeric(gains["lead"], errors="raise").astype(int)
    gains["criterion"] = gains["criterion"].astype(str).str.lower()
    gains["assigned_group"] = pd.to_numeric(gains["assigned_group"], errors="raise").astype(int)
    ensure(len(gains) == EXPECTED_TARGET * len(LEADS) * len(CRITERIA), f"Expected 8,361 household gain rows; found {len(gains):,}")
    ensure(set(gains["lead"]) == set(LEADS) and set(gains["criterion"]) == set(CRITERIA), "Household gain lead/criterion scope mismatch")
    ensure(gains["support_scope"].eq("native").all() and gains["seed_count"].eq(4).all(), "Household gain table must use native support and four formal seeds")
    base = gains.merge(features, on=["meter_id", "assigned_group"], how="inner", validate="many_to_one")
    ensure(len(base) == len(gains), "Source-support features did not join to all household gain rows")

    for strategy, column in STRATEGY_VALUE_COLUMNS.items():
        base[f"strategy_error_{strategy}"] = pd.to_numeric(base[column], errors="raise")
    for comparison, focal, comparator in COMPARISONS:
        focal_error = base[f"strategy_error_{focal}"].to_numpy(float)
        comparator_error = base[f"strategy_error_{comparator}"].to_numpy(float)
        absolute = comparator_error - focal_error
        base[f"gain_{comparison}_absolute"] = absolute
        base[f"gain_{comparison}_relative_pct"] = safe_relative(absolute, comparator_error)
        base[f"gain_{comparison}_positive"] = absolute > 0
        base[f"gain_{comparison}_focal_error"] = focal_error
        base[f"gain_{comparison}_comparator_error"] = comparator_error
    base["source_transfer_gain_relative_pct"] = base["gain_source_transfer_relative_pct"]
    ensure(np.isfinite(base[[f"strategy_error_{s}" for s in STRATEGY_VALUE_COLUMNS]].to_numpy(float)).all(), "Non-finite strategy errors in source-support outcome")
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()

    source_assignments = read_source_assignments(root)
    source_profiles = load_source_profiles(root, source_assignments)
    target_profiles = load_target_profiles(root)
    prototypes = load_prototypes(root)
    features = compute_features(source_profiles, target_profiles, prototypes)
    outcome = build_outcome(root, features)

    table_dir = root / "outputs/tables/local_source_support_complete_cross_strategy_forecasting_aligned_source_support"
    metadata_dir = root / "outputs/metadata/local_source_support_complete_cross_strategy_forecasting_aligned_source_support"
    table_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    feature_path = table_dir / "local_source_support_meter_features.csv"
    outcome_path = table_dir / "local_source_support_cross_strategy_profile_support_outcome_dataset.csv"
    features.to_csv(feature_path, index=False)
    outcome.to_csv(outcome_path, index=False)

    status = {
        "analysis_id": "local_source_support",
        "status": "COMPLETE_PASS",
        "source_households": EXPECTED_SOURCE,
        "target_meters": EXPECTED_TARGET,
        "feature_rows": int(len(features)),
        "outcome_rows": int(len(outcome)),
        "leads": LEADS,
        "criteria": CRITERIA,
        "group_counts": {str(k): v for k, v in EXPECTED_GROUP_COUNTS.items()},
        "training_performed": False,
        "predictions_modified": False,
        "prototypes_refitted": False,
        "meters_reassigned": False,
        "meters_excluded": False,
        "interpretation": "post-hoc descriptive source-profile support",
    }
    status_path = metadata_dir / "local_source_support_status.json"
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    inventory = pd.DataFrame([
        {"path": str(path.relative_to(root)), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in [feature_path, outcome_path, status_path]
    ])
    inventory.to_csv(metadata_dir / "local_source_support_output_inventory.csv", index=False)
    print("LOCAL SOURCE SUPPORT: PASS")
    print(f"Source households: {EXPECTED_SOURCE}")
    print(f"Target households: {EXPECTED_TARGET}")
    print(f"Outcome rows: {len(outcome):,}")


if __name__ == "__main__":
    main()
