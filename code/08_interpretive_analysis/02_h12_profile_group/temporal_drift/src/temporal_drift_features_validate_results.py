#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import os as _erp_os
_ERP_PROJECT_ROOT = Path(_erp_os.environ.get("ERP_PROJECT_ROOT", ".")).expanduser().resolve()
import pandas as pd
ROOT=Path(sys.argv[1] if len(sys.argv)>1 else str(_ERP_PROJECT_ROOT))
ANALYSIS="temporal_drift_features_target_span_representativeness_temporal_drift"
T=ROOT/f"outputs/tables/{ANALYSIS}"; M=ROOT/f"outputs/metadata/{ANALYSIS}"
def ensure(c,m):
    if not c: raise RuntimeError(m)
status=json.loads((M/f"{ANALYSIS}_status.json").read_text())
ensure(status["analysis_id"]=="temporal_drift_features","Wrong release")
ensure(status["status"]=="COMPLETE_PASS","Status not COMPLETE_PASS")
ensure(status["meters"]==929,"Meter count mismatch")
checks={
"temporal_drift_features_meter_period_characteristics.csv":3716,
"temporal_drift_features_meter_temporal_drift_features.csv":929,
"temporal_drift_features_h12_drift_outcome_dataset.csv":2787,
"temporal_drift_features_overall_spearman_summary.csv":108,
"temporal_drift_features_group_adjusted_rank_summary.csv":108,
"temporal_drift_features_within_group_spearman_summary.csv":432,
"temporal_drift_features_h12_primary_drift_evidence.csv":27,
}
for name,n in checks.items():
    p=T/name; ensure(p.is_file(),f"Missing {p}"); df=pd.read_csv(p); ensure(len(df)==n,f"{name}: expected {n}, found {len(df)}")
profiles=pd.read_parquet(T/"temporal_drift_features_meter_period_profiles.parquet"); ensure(len(profiles)==178368,"Profile row count mismatch")
ensure(profiles.meter_id.nunique()==929,"Profile meter count mismatch")
drift=pd.read_csv(T/"temporal_drift_features_meter_temporal_drift_features.csv"); ensure(drift.meter_id.nunique()==929,"Drift meter count mismatch")
out=pd.read_csv(T/"temporal_drift_features_h12_drift_outcome_dataset.csv"); ensure(sorted(out.criterion.unique())==["mae","rmse","smape"],"Criteria mismatch")
inv=pd.read_csv(M/"temporal_drift_features_output_inventory.csv")
for r in inv.itertuples(index=False):
    p=ROOT/r.relative_path; ensure(p.is_file(),f"Missing inventory file {p}"); ensure(p.stat().st_size==int(r.bytes),f"Size mismatch {p}"); ensure(hashlib.sha256(p.read_bytes()).hexdigest()==r.sha256,f"Hash mismatch {p}")
print("="*100); print("TEMPORAL DRIFT FEATURES OUTPUT VALIDATION: PASS"); print("="*100)
print("Meters                         : 929")
print("Meter-period rows               : 3,716")
print("Meter drift rows                : 929")
print("h=12 outcome rows               : 2,787")
print("Overall relationships           : 108")
print("Group-adjusted relationships    : 108")
print("Within-group relationships      : 432")
print("Primary bootstrap relationships : 27")
print(f"Inventory hashes                : {len(inv)}/{len(inv)} PASS")
print("Final status                    : COMPLETE_PASS")
