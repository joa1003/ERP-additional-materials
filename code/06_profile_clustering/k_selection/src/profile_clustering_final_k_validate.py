from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from profile_clustering_final_k_common import ensure, load_yaml, resolve_path


def validate(config_path: Path) -> None:
    config = load_yaml(config_path)
    root = Path(config["project_root"]).resolve()
    paths = config["paths"]
    tables = resolve_path(root, paths["output_tables"])
    metadata = resolve_path(root, paths["output_metadata"])
    decision = resolve_path(root, paths["decision_document"])

    required_tables = {
        "profile_clustering_final_k_all_run_diagnostics.csv": 76,
        "profile_clustering_final_k_cluster_composition_all_runs.csv": 380,
        "profile_clustering_final_k_candidate_prototypes_all_runs.csv": 18240,
        "profile_clustering_final_k_all_source_assignments_all_runs.parquet": 292068,
        "profile_clustering_final_k_pairwise_ari_all_runs.csv": 594,
        "profile_clustering_candidate_k_selection_table.csv": 7,
        "profile_clustering_initialisation_sensitivity_k4_k5_k6.csv": 3,
    }
    loaded: dict[str, pd.DataFrame] = {}
    for name, expected_rows in required_tables.items():
        path = tables / name
        ensure(path.is_file(), f"Missing output table: {path}")
        frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        ensure(len(frame) == expected_rows, f"Unexpected row count for {name}: {len(frame)} != {expected_rows}")
        loaded[name] = frame

    run_df = loaded["profile_clustering_final_k_all_run_diagnostics.csv"]
    ensure(sorted(run_df["k"].unique().tolist()) == [2, 3, 4, 5, 6, 7, 8], "Run diagnostics K coverage mismatch")
    ensure(len(run_df[run_df["k"].isin([4, 5, 6])].groupby("k")) == 3, "Focused K groups missing")
    focused_counts = run_df[run_df["k"].isin([4, 5, 6])].groupby("k")["seed"].nunique().to_dict()
    ensure(focused_counts == {4: 20, 5: 20, 6: 20}, f"Focused seed counts mismatch: {focused_counts}")
    broad_counts = run_df.groupby("k").apply(lambda frame: frame[frame["seed"].isin([42, 123, 2026, 31415])]["seed"].nunique(), include_groups=False).to_dict()
    ensure(broad_counts == {2: 4, 3: 4, 4: 4, 5: 4, 6: 4, 7: 4, 8: 4}, f"Broad seed counts mismatch: {broad_counts}")
    numeric_columns = [
        "inertia_dtw",
        "full_sample_dtw_silhouette",
        "minimum_cluster_share_fit",
        "maximum_within_group_low_signal_share",
        "fit_seconds",
    ]
    ensure(np.isfinite(run_df[numeric_columns].to_numpy(dtype=float)).all(), "Run diagnostics contain non-finite values")
    ensure((run_df["iterations"] > 0).all(), "Invalid iteration counts")

    k_summary = loaded["profile_clustering_candidate_k_selection_table.csv"]
    ensure(k_summary["k"].astype(int).tolist() == [2, 3, 4, 5, 6, 7, 8], "K summary order mismatch")
    if "automated_final_k_decision" in k_summary.columns:
        ensure(
            set(k_summary["automated_final_k_decision"].astype(str)) == {"NOT_PERFORMED"},
            "K was selected automatically",
        )
    ensure(set(k_summary["selection_status"].astype(str)) == {"REPORTED_K_SELECTION_EVIDENCE"}, "K-selection status mismatch")

    focused = loaded["profile_clustering_initialisation_sensitivity_k4_k5_k6.csv"]
    ensure(focused["k"].astype(int).tolist() == [4, 5, 6], "Focused summary K order mismatch")
    ensure(set(focused["independent_initialisations"].astype(int)) == {20}, "Focused summary does not have 20 initialisations")

    assignments = loaded["profile_clustering_final_k_all_source_assignments_all_runs.parquet"]
    ensure(assignments["household_id"].nunique() == 3843, "Assignment household count mismatch")
    ensure(int(assignments["low_signal_prototype_fit_exclusion"].sum()) == 54 * 76, "Low-signal assignment count mismatch")

    status_path = metadata / "profile_clustering_final_k_selection_status.json"
    summary_path = metadata / "profile_clustering_final_k_selection_summary.md"
    ensure(status_path.is_file(), f"Missing status JSON: {status_path}")
    ensure(summary_path.is_file(), f"Missing summary: {summary_path}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    ensure(status.get("status") == "COMPLETE", "Completion status mismatch")
    ensure(int(status.get("final_k_decision")) == 4, "Reported K mismatch")
    ensure(status.get("cer_rows_used") == 0, "CER rows were used")
    ensure(status.get("forecasting_metrics_used") == 0, "Forecasting metrics were used")
    ensure(status.get("survey_variables_used") == 0, "Survey variables were used")
    ensure(status.get("time_basis", {}).get("raw_gmt_hour_used_as_local_time") is False, "Raw GMT hour was used as local time")
    ensure(decision.is_file(), f"Missing decision review document: {decision}")

    print("=" * 100)
    print("PROFILE CLUSTERING FINAL K-SELECTION OUTPUT VALIDATION PASS")
    print("=" * 100)
    print("Status              : COMPLETE")
    print("Source households   : 3843")
    print("Independent fits    : 76")
    print("Broad K             : [2, 3, 4, 5, 6, 7, 8]")
    print("Focused K           : [4, 5, 6] × 20 initialisations")
    print("Next action         : manual K review; CER assignment remains blocked")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    validate(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
