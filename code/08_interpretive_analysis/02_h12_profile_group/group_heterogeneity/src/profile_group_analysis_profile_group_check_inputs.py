from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from profile_group_analysis_profile_group_common import ensure, load_config, normalise_id, read_json, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    project_root = Path(config["project_root"]).resolve()

    input_paths = {key: resolve_path(project_root, value) for key, value in config["inputs"].items()}
    for key, path in input_paths.items():
        ensure(path.is_file(), f"Missing required input {key}: {path}")

    status8 = read_json(input_paths["household_analytical_status"])
    status9 = read_json(input_paths["target_profile_assignment_status"])
    ensure(status8.get("status") == "COMPLETE_PASS", "Household analytical table is not COMPLETE_PASS")
    ensure(status9.get("status") == "COMPLETE_PASS", "Target profile assignment is not COMPLETE_PASS")

    assignments = pd.read_csv(input_paths["target_profile_assignment_assignments"], usecols=["meter_id", "assigned_group"])
    metrics = pd.read_csv(input_paths["household_analytical_native_metrics"], usecols=["entity_id", "strategy", "lead", "support_scope", "seed_count"])
    gains = pd.read_csv(input_paths["household_analytical_native_gains"], usecols=["entity_id", "lead", "criterion", "support_scope", "seed_count"])
    shifts = pd.read_csv(input_paths["household_analytical_common_lead_shifts"], usecols=["entity_id", "strategy", "criterion"])

    assignments["meter_id"] = normalise_id(assignments["meter_id"])
    metrics["entity_id"] = normalise_id(metrics["entity_id"])
    gains["entity_id"] = normalise_id(gains["entity_id"])
    shifts["entity_id"] = normalise_id(shifts["entity_id"])

    expected = config["expected"]
    ensure(len(assignments) == expected["meters"] and assignments["meter_id"].nunique() == expected["meters"], "Assignment population gate failed")
    ensure(len(metrics) == expected["native_metric_rows"], "Household analytical table metric row-count gate failed")
    ensure(len(gains) == expected["native_gain_rows"], "Household analytical table gain row-count gate failed")
    ensure(len(shifts) == expected["common_lead_shift_rows"], "Household analytical table lead-shift row-count gate failed")
    ids = set(assignments["meter_id"])
    ensure(set(metrics["entity_id"]) == ids and set(gains["entity_id"]) == ids and set(shifts["entity_id"]) == ids, "Exact meter-ID set equality gate failed")

    group_counts = assignments["assigned_group"].astype(int).value_counts().sort_index().to_dict()
    expected_counts = {int(k): int(v) for k, v in expected["group_counts"].items()}
    ensure(group_counts == expected_counts, f"Group-count gate failed: {group_counts}")

    out_dirs = [resolve_path(project_root, value) for key, value in config["outputs"].items() if key.endswith("_dir")]
    for path in out_dirs:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".profile_group_analysis_write_test"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()

    print("=" * 96)
    print("PROFILE GROUP ANALYSIS MINIMAL INPUT CHECK: PASS")
    print("=" * 96)
    print(f"Python executable     : {sys.executable}")
    print(f"Meters                : {len(ids)}")
    print(f"Group counts          : {group_counts}")
    print(f"Metric rows           : {len(metrics)}")
    print(f"Gain rows             : {len(gains)}")
    print(f"Lead-shift rows       : {len(shifts)}")
    print("Lock prose parsing    : NOT USED")
    print("Synthetic selftest    : NOT RERUN DURING SUBMISSION")

if __name__ == "__main__":
    main()
