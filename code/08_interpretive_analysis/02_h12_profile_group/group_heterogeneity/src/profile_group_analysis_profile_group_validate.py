from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from profile_group_analysis_profile_group_common import ensure, load_config, read_json, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(Path(args.config).resolve())
    project_root = Path(config["project_root"]).resolve()
    table_dir = resolve_path(project_root, config["outputs"]["table_dir"])
    metadata_dir = resolve_path(project_root, config["outputs"]["metadata_dir"])

    status_path = metadata_dir / "profile_group_analysis_profile_group_analysis_status.json"
    ensure(status_path.is_file(), f"Missing Profile group analysis status: {status_path}")
    status = read_json(status_path)
    ensure(status.get("status") == "COMPLETE_PASS", "Profile group analysis status is not COMPLETE_PASS")
    ensure(status.get("meters") == 929, "Profile group analysis meter count is not 929")
    ensure(status.get("unmatched_meters") == 0, "Profile group analysis has unmatched meters")
    ensure(status.get("group_counts") == {"1":141,"2":325,"3":324,"4":139}, "Profile group analysis group counts differ")
    ensure(status.get("group_strategy_summary_rows") == 144, "Unexpected group-strategy summary row count")
    ensure(status.get("group_gain_summary_rows") == 144, "Unexpected group-gain summary row count")
    ensure(status.get("group_lead_shift_summary_rows") == 144, "Unexpected group lead-shift summary row count")
    ensure(status.get("model_training_performed") == 0, "Model training was performed unexpectedly")
    ensure(status.get("prototypes_refitted") == 0, "Source prototypes were refitted unexpectedly")
    ensure(status.get("meters_reassigned") == 0, "Meters were reassigned unexpectedly")
    ensure(status.get("continuous_dtw_relationship_analysis_performed") == 0, "Source similarity analysis leaked into Profile group analysis")

    expected_tables = {
        "profile_group_analysis_meter_group_membership.csv": 929,
        "profile_group_analysis_native_meter_metrics_with_group.csv": 11148,
        "profile_group_analysis_native_meter_gains_with_group.csv": 8361,
        "profile_group_analysis_group_strategy_metric_summary.csv": 144,
        "profile_group_analysis_group_gain_summary.csv": 144,
        "profile_group_analysis_group_lead_shift_summary.csv": 144,
        "profile_group_analysis_main_text_group_gain_table.csv": 108,
        "profile_group_analysis_h12_profile_group_strategy_table.csv": 12,
    }
    for name, rows in expected_tables.items():
        path = table_dir / name
        ensure(path.is_file(), f"Missing output table: {path}")
        actual = len(pd.read_csv(path))
        ensure(actual == rows, f"Unexpected rows in {name}: expected={rows}, actual={actual}")

    print("=" * 96)
    print("PROFILE GROUP ANALYSIS OUTPUT VALIDATION PASS")
    print("=" * 96)
    print("Meters               : 929")
    print("Group count sum       : 929")
    print("Unmatched meters      : 0")
    print("Primary support       : native lead-specific")
    print("Cross-lead support    : strict common-row")
    print("Aggregation           : equal meter")
    print("Continuous DTW test   : deferred to Source similarity")
    print("Final status          : COMPLETE_PASS")

if __name__ == "__main__":
    main()
