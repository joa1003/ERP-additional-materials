from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tslearn.metrics import cdist_dtw

HERE = Path(__file__).resolve()
PACKAGE_ROOT = HERE.parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
from profile_clustering_final_k_analysis import run_analysis


def main() -> int:
    rng = np.random.default_rng(20260726)
    with tempfile.TemporaryDirectory(prefix="profile_clustering_final_k_selftest_") as temporary:
        root = Path(temporary)
        input_dir = root / "inputs"
        input_dir.mkdir(parents=True)

        slots = np.arange(48, dtype=np.float64)
        templates = [
            np.exp(-0.5 * ((slots - 16) / 2.5) ** 2),
            np.exp(-0.5 * ((slots - 36) / 3.0) ** 2),
            np.exp(-0.5 * ((slots - 40) / 3.0) ** 2),
            np.exp(-0.5 * ((slots - 44) / 2.5) ** 2),
            0.6 * np.exp(-0.5 * ((slots - 18) / 3.0) ** 2) + np.exp(-0.5 * ((slots - 38) / 3.5) ** 2),
            np.exp(-0.5 * ((slots - 30) / 5.0) ** 2),
        ]
        profiles: list[np.ndarray] = []
        for template in templates:
            for _ in range(6):
                profiles.append(template + rng.normal(0.0, 0.03, size=48))
        for _ in range(6):
            profiles.append(rng.normal(0.0, 0.001, size=48))
        raw = np.asarray(profiles, dtype=np.float64)
        means = raw.mean(axis=1, keepdims=True)
        stds = raw.std(axis=1, ddof=0, keepdims=True)
        zscore = (raw - means) / np.maximum(stds, 1e-12)
        household_ids = [f"S{index:04d}" for index in range(len(raw))]
        slot_names = [f"slot_{index:02d}" for index in range(1, 49)]

        raw_frame = pd.DataFrame(raw, columns=slot_names)
        raw_frame.insert(0, "household_id", household_ids)
        raw_frame.to_csv(input_dir / "raw.csv", index=False)
        z_frame = pd.DataFrame(zscore, columns=slot_names)
        z_frame.insert(0, "household_id", household_ids)
        z_frame.to_csv(input_dir / "zscore.csv", index=False)
        low_mask = np.zeros(len(raw), dtype=bool)
        low_mask[-6:] = True
        low_frame = pd.DataFrame({
            "household_id": household_ids,
            "low_signal_prototype_fit_exclusion": low_mask,
        })
        low_frame.to_csv(input_dir / "low.csv", index=False)
        fit_values = zscore[~low_mask, :, None]
        distance = cdist_dtw(fit_values, n_jobs=1).astype(np.float32)
        np.save(input_dir / "distance.npy", distance)

        config = {
            "project_root": str(root),
            "paths": {
                "profiles_zscore": "inputs/zscore.csv",
                "profiles_raw_kwh": "inputs/raw.csv",
                "low_signal_audit": "inputs/low.csv",
                "distance_matrix": "inputs/distance.npy",
                "profile_construction_audit": "inputs/not_used.csv",
                "phase1_status": "inputs/not_used.json",
                "time_map": "inputs/not_used.parquet",
                "output_tables": "outputs/tables",
                "output_figures": "outputs/figures",
                "output_metadata": "outputs/metadata",
                "decision_document": "documentation/review.md",
            },
            "expected": {
                "source_households": 42,
                "fit_eligible_households": 36,
                "low_signal_households": 6,
            },
            "clustering": {
                "broad_k": [2, 3, 4, 5, 6],
                "broad_seeds": [42, 123],
                "focused_k": [4, 5, 6],
                "focused_seeds": [42, 123],
                "reference_seed": 42,
                "max_iter": 3,
                "max_iter_barycenter": 3,
                "tol": 1e-4,
                "max_parallel_workers": 1,
            },
            "time_rules": {
                "excluded_local_dates": ["2013-03-31"],
            },
        }
        config_path = root / "selftest.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        metadata = run_analysis(config_path)
        if metadata["status"] != "COMPLETE":
            raise AssertionError(metadata)
        required = [
            root / "outputs/tables/profile_clustering_final_k_all_run_diagnostics.csv",
            root / "outputs/tables/profile_clustering_candidate_k_selection_table.csv",
            root / "outputs/tables/profile_clustering_initialisation_sensitivity_k4_k5_k6.csv",
            root / "outputs/metadata/profile_clustering_final_k_selection_status.json",
            root / "documentation/review.md",
        ]
        for path in required:
            if not path.is_file() or path.stat().st_size == 0:
                raise AssertionError(f"Missing selftest output: {path}")
        status = json.loads((root / "outputs/metadata/profile_clustering_final_k_selection_status.json").read_text())
        if status["cer_rows_used"] != 0 or int(status["final_k_decision"]) != 4:
            raise AssertionError(status)

    print("PROFILE CLUSTERING FINAL K-SELECTION SYNTHETIC END-TO-END SELFTEST: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
