from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    release = Path(__file__).resolve().parents[1]
    root = Path(tempfile.mkdtemp(prefix="profile_group_analysis_selftest_"))
    try:
        (root / "outputs/tables/metric_generation/household_analytical_dataset").mkdir(parents=True)
        (root / "outputs/metadata/metric_generation/household_analytical_dataset").mkdir(parents=True)
        (root / "outputs/tables/target_profile_assignment_cer_assignment").mkdir(parents=True)
        (root / "outputs/metadata/target_profile_assignment_cer_assignment").mkdir(parents=True)
        (root / "documentation").mkdir(parents=True)
        (root / "documentation/reproducibility_design_contract.md").write_text("synthetic public contract\n")

        ids = [str(1000+i) for i in range(8)]
        groups = [1,1,2,2,3,3,4,4]
        assign = pd.DataFrame({
            "meter_id": ids,
            "assigned_group": groups,
            "nearest_dtw_distance": np.linspace(1,2,8),
            "distance_margin": np.linspace(.1,.8,8),
            "distance_ratio": np.linspace(.5,.9,8),
        })
        assign.to_csv(root / "outputs/tables/target_profile_assignment_cer_assignment/target_profile_assignment_cer_source_defined_assignments.csv", index=False)
        (root / "outputs/metadata/target_profile_assignment_cer_assignment/target_profile_assignment_cer_assignment_status.json").write_text(json.dumps({"status":"COMPLETE_PASS"}))
        (root / "outputs/metadata/metric_generation/household_analytical_dataset/household_analytical_dataset_status.json").write_text(json.dumps({"status":"COMPLETE_PASS"}))

        metric_rows=[]
        strategies=["direct_transfer","fine_tuning","cer_scratch_limited","cer_scratch_full"]
        for entity in ids:
            for lead in [1,12,48]:
                for sidx,strategy in enumerate(strategies):
                    base=.1+.01*lead/12+.005*sidx+.001*int(entity[-1])
                    metric_rows.append({"entity_id":entity,"strategy":strategy,"lead":lead,"support_scope":"native","seed_count":4,"mae_kwh_mean":base,"rmse_kwh_mean":base*1.7,"smape_percent_mean":base*100})
        pd.DataFrame(metric_rows).to_csv(root / "outputs/tables/metric_generation/household_analytical_dataset/household_metrics_native_four_seed.csv", index=False)

        gain_rows=[]
        for entity in ids:
            for lead in [1,12,48]:
                for criterion in ["mae","rmse","smape"]:
                    scale=1 if criterion!="smape" else 100
                    gain_rows.append({
                        "entity_id":entity,"lead":lead,"criterion":criterion,"support_scope":"native","seed_count":4,
                        "source_transfer_gain_mean":.01*scale,"fine_tuning_gain_mean":-.005*scale,"full_data_gain_mean":.02*scale,"direct_vs_full_gain_mean":.01*scale,
                    })
        pd.DataFrame(gain_rows).to_csv(root / "outputs/tables/metric_generation/household_analytical_dataset/household_strategy_gains_native_four_seed.csv", index=False)

        shift_rows=[]
        for entity in ids:
            for strategy in strategies:
                for criterion in ["mae","rmse","smape"]:
                    shift_rows.append({"entity_id":entity,"strategy":strategy,"criterion":criterion,"h12_minus_h1":.02,"h48_minus_h12":-.01,"h48_minus_h1":.01})
        pd.DataFrame(shift_rows).to_csv(root / "outputs/tables/metric_generation/household_analytical_dataset/household_error_lead_shifts_common.csv", index=False)

        config = json.loads((release / "configs/profile_group_analysis_profile_group_analysis.json").read_text())
        config["project_root"] = str(root)
        config["python_executable"] = sys.executable
        config["expected"] = {
            "meters":8,"groups":[1,2,3,4],"group_counts":{"1":2,"2":2,"3":2,"4":2},"leads":[1,12,48],
            "strategies":strategies,"criteria":["mae","rmse","smape"],"seed_count":4,
            "native_metric_rows":96,"native_gain_rows":72,"common_lead_shift_rows":96,
        }
        config_path=root/"config.json"
        config_path.write_text(json.dumps(config,indent=2))
        subprocess.run([sys.executable, str(release/"src/profile_group_analysis_profile_group_analysis.py"), "--config", str(config_path)], check=True, env={**dict(__import__('os').environ), "PYTHONPATH":str(release/"src")})
        status=json.loads((root/"outputs/metadata/profile_group_analysis_profile_group_analysis/profile_group_analysis_profile_group_analysis_status.json").read_text())
        assert status["status"]=="COMPLETE_PASS"
        assert status["meters"]==8
        assert status["unmatched_meters"]==0
        print("PROFILE GROUP ANALYSIS SYNTHETIC END TO END SELFTEST: PASS")
    finally:
        shutil.rmtree(root, ignore_errors=True)

if __name__ == "__main__":
    main()
