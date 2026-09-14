from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    package_root = Path(__file__).resolve().parents[1]
    analysis = package_root / "src" / "target_profile_assignment_cer_assignment_analysis.py"
    validate = package_root / "src" / "target_profile_assignment_cer_assignment_validate.py"

    with tempfile.TemporaryDirectory(prefix="target_profile_assignment_selftest_") as temp_dir_name:
        root = Path(temp_dir_name)
        input_dir = root / "inputs"
        input_dir.mkdir(parents=True)

        slots = np.arange(1, 49)
        prototype_values = {
            1: np.exp(-0.5 * ((slots - 15) / 3.0) ** 2),
            2: np.exp(-0.5 * ((slots - 38) / 3.5) ** 2),
            3: np.exp(-0.5 * ((slots - 43) / 3.0) ** 2),
            4: 0.6 * np.exp(-0.5 * ((slots - 45) / 2.5) ** 2) + 0.4 * np.exp(-0.5 * ((slots - 2) / 2.0) ** 2),
        }
        prototype_rows = []
        group_rows = []
        for group, raw in prototype_values.items():
            z = (raw - raw.mean()) / raw.std(ddof=0)
            for slot, value in zip(slots, z):
                minutes = (slot - 1) * 30
                prototype_rows.append(
                    {
                        "k": 4,
                        "seed": 71,
                        "profile_group": group,
                        "local_slot": slot,
                        "prototype_zscore": value,
                        "local_time": f"{minutes // 60:02d}:{minutes % 60:02d}",
                    }
                )
            group_rows.append(
                {
                    "k": 4,
                    "seed": 71,
                    "profile_group": group,
                    "all_assigned_n": [540, 976, 1181, 1146][group - 1],
                    "fit_eligible_n": [523, 974, 1167, 1125][group - 1],
                    "low_signal_n": [17, 2, 14, 21][group - 1],
                }
            )
        prototype_csv = input_dir / "prototypes.csv"
        group_csv = input_dir / "groups.csv"
        pd.DataFrame(prototype_rows).to_csv(prototype_csv, index=False)
        pd.DataFrame(group_rows).to_csv(group_csv, index=False)

        dates = pd.date_range("2009-07-15", "2009-07-18", freq="D")
        cer_rows = []
        rng = np.random.default_rng(123)
        for group in [1, 2, 3, 4]:
            base = prototype_values[group]
            base = 0.35 + 0.12 * (base - base.min()) / (base.max() - base.min())
            for meter_offset in [1, 2]:
                meter_id = group * 1000 + meter_offset
                for local_date in dates:
                    day_scale = 1.0 + rng.normal(0, 0.01)
                    for slot, value in zip(slots, base):
                        cer_rows.append(
                            {
                                "meter_id": meter_id,
                                "local_date": local_date.date(),
                                "local_slot": slot,
                                "kwh": max(0.001, value * day_scale + rng.normal(0, 0.001)),
                            }
                        )
        cer_parquet = input_dir / "cer.parquet"
        pd.DataFrame(cer_rows).to_parquet(cer_parquet, index=False)
        lock_path = input_dir / "lock.md"
        lock_path.write_text("Final K = 4\n2009-07-15 00:00 to 2010-07-24 11:30\nCER does not fit independent prototypes\nnearest_dtw_distance\nsecond_nearest_dtw_distance\ndistance_margin\ndistance_ratio\n", encoding="utf-8")

        config = {
            "project_root": str(root),
            "analysis_id": "target_profile_assignment_selftest",
            "inputs": {
                "source_prototypes_csv": str(prototype_csv),
                "source_group_summary_csv": str(group_csv),
                "cer_canonical_parquet": str(cer_parquet),
                "experiment_design_lock": str(lock_path),
            },
            "profile": {
                "start_date": "2009-07-15",
                "start_slot": 1,
                "end_date": "2009-07-18",
                "end_slot": 48,
                "period_start_label": "2009-07-15 00:00",
                "period_end_label": "2009-07-18 23:30",
                "excluded_local_dates": ["2009-10-25", "2010-03-28", "2010-10-31"],
                "daily_profile_variation_flag_threshold_kwh": 0.01,
                "arrow_batch_size": 256,
            },
            "expected": {
                "source_k": 4,
                "source_seed": 71,
                "cer_meters": 8,
                "cer_canonical_rows": len(cer_rows),
            },
            "outputs": {
                "table_dir": "outputs/tables/target_profile_assignment_cer_assignment",
                "figure_dir": "outputs/figures/target_profile_assignment_cer_assignment",
                "metadata_dir": "outputs/metadata/target_profile_assignment_cer_assignment",
                "decision_md": "documentation/technical_decisions/target_profile_assignment_cer_assignment_decision.md",
            },
        }
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

        subprocess.run([sys.executable, str(analysis), "--config", str(config_path)], check=True)
        subprocess.run([sys.executable, str(validate), "--config", str(config_path)], check=True)

        assignments = pd.read_csv(root / "outputs/tables/target_profile_assignment_cer_assignment/target_profile_assignment_cer_source_defined_assignments.csv")
        assert len(assignments) == 8
        assert assignments["meter_id"].nunique() == 8
        assert set(assignments["assigned_group"]) == {1, 2, 3, 4}
        status = json.loads((root / "outputs/metadata/target_profile_assignment_cer_assignment/target_profile_assignment_cer_assignment_status.json").read_text())
        assert status["status"] == "COMPLETE_PASS"

    print("TARGET PROFILE ASSIGNMENT CER ASSIGNMENT SYNTHETIC END TO END SELFTEST: PASS")


if __name__ == "__main__":
    main()
