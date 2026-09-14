from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

from target_profile_assignment_cer_assignment_common import (
    ensure,
    load_config,
    load_fixed_prototypes,
    resolve_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Target profile assignment preflight")
    parser.add_argument("--config", required=True)
    parser.add_argument("--release-root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.release_root).expanduser().resolve()
    config = load_config(args.config)
    config["project_root"] = str(root)

    for module_name in ["numpy", "pandas", "pyarrow", "matplotlib", "sklearn", "tslearn"]:
        importlib.import_module(module_name)

    prototype_path = resolve_path(config, config["inputs"]["source_prototypes_csv"])
    group_summary_path = resolve_path(config, config["inputs"]["source_group_summary_csv"])
    cer_path = resolve_path(config, config["inputs"]["cer_canonical_parquet"])
    lock_path = resolve_path(config, config["inputs"]["experiment_design_lock"])
    for path in [prototype_path, group_summary_path, cer_path, lock_path]:
        ensure(path.exists(), f"Required generated input missing: {path}")

    _, prototype_matrix, groups = load_fixed_prototypes(config)
    ensure(prototype_matrix.shape == (4, 48), f"Unexpected prototype matrix shape: {prototype_matrix.shape}")
    ensure(groups == [1, 2, 3, 4], f"Unexpected profile groups: {groups}")

    group_summary = pd.read_csv(group_summary_path)
    required_group_columns = {"k", "seed", "profile_group", "all_assigned_n", "fit_eligible_n", "low_signal_n"}
    ensure(required_group_columns.issubset(group_summary.columns),
           f"Source group summary missing columns: {sorted(required_group_columns.difference(group_summary.columns))}")
    ensure(group_summary["profile_group"].astype(int).tolist() == [1, 2, 3, 4], "Source group order must be 1 to 4")
    ensure(int(group_summary["all_assigned_n"].sum()) == int(config["expected"]["source_formal_cohort"]),
           "Source all-assigned count mismatch")
    ensure(int(group_summary["fit_eligible_n"].sum()) == int(config["expected"]["source_prototype_estimation_profiles"]),
           "Source prototype-fit count mismatch")
    ensure(int(group_summary["low_signal_n"].sum()) == int(config["expected"]["source_profiles_assigned_after_fixing_prototypes"]),
           "Source low-signal count mismatch")

    dataset = ds.dataset(str(cer_path), format="parquet")
    required_cer_columns = {"meter_id", "local_date", "local_slot", "kwh"}
    ensure(required_cer_columns.issubset(dataset.schema.names),
           f"CER canonical parquet missing columns: {sorted(required_cer_columns.difference(dataset.schema.names))}")
    row_count = int(dataset.count_rows())
    ensure(row_count == int(config["expected"]["cer_canonical_rows"]),
           f"Expected {config['expected']['cer_canonical_rows']} CER canonical rows, found {row_count}")
    ensure(lock_path.read_text(encoding="utf-8").strip(), f"Design summary is empty: {lock_path}")

    for output_key in ["table_dir", "figure_dir", "metadata_dir"]:
        output_dir = resolve_path(config, config["outputs"][output_key])
        output_dir.mkdir(parents=True, exist_ok=True)
        ensure(os.access(output_dir, os.W_OK), f"Output directory not writable: {output_dir}")

    print("TARGET PROFILE ASSIGNMENT PREFLIGHT: PASS")
    print(f"Project root                   : {root}")
    print(f"Fixed source prototypes       : {prototype_matrix.shape}")
    print(f"Source assignment identity    : {int(group_summary['fit_eligible_n'].sum())} + {int(group_summary['low_signal_n'].sum())} = {int(group_summary['all_assigned_n'].sum())}")
    print(f"CER canonical rows            : {row_count}")
    print(f"Expected target households    : {config['expected']['cer_meters']}")
    print("Prototype refitting permitted : NO")
    print("Forecasting metrics used      : NO")


if __name__ == "__main__":
    main()
