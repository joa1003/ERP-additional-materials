from __future__ import annotations
import argparse, importlib, json, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from profile_clustering_final_k_common import ensure, load_yaml, resolve_path, slot_columns

def preflight(config_path: Path, release_root: Path, run_selftest: bool=True)->None:
    cfg=load_yaml(config_path); root=release_root.resolve(); paths=cfg['paths']; exp=cfg['expected']
    for name in ['numpy','pandas','pyarrow','yaml','matplotlib','sklearn','scipy','tslearn']:
        importlib.import_module(name)
    profiles=pd.read_csv(resolve_path(root,paths['profiles_zscore']))
    raw=pd.read_csv(resolve_path(root,paths['profiles_raw_kwh']))
    low=pd.read_csv(resolve_path(root,paths['low_signal_audit']))
    slots=slot_columns(profiles.columns.tolist())
    ensure(len(slots)==48,'Expected 48 profile slots')
    ensure(len(profiles)==int(exp['source_households']),'Source household count mismatch')
    ensure(raw['household_id'].astype(str).tolist()==profiles['household_id'].astype(str).tolist(),'Raw/z profile ID mismatch')
    ensure(low['household_id'].astype(str).tolist()==profiles['household_id'].astype(str).tolist(),'Variation audit ID mismatch')
    eligible=low['prototype_fit_eligible'].astype(bool).to_numpy()
    ensure(int(eligible.sum())==int(exp['fit_eligible_households']),'Fit-eligible count mismatch')
    ensure(int((~eligible).sum())==int(exp['low_signal_households']),'Low-signal count mismatch')
    z=profiles[slots].to_numpy(float)[eligible]
    ensure(np.max(np.abs(z.mean(axis=1)))<1e-8,'Profiles are not row-centred')
    ensure(np.max(np.abs(z.std(axis=1,ddof=0)-1.0))<1e-8,'Profiles are not ddof=0 z-scored')
    d=np.load(resolve_path(root,paths['distance_matrix']),mmap_mode='r')
    ensure(list(d.shape)==[int(x) for x in exp['distance_matrix_shape']],'DTW matrix shape mismatch')
    ensure(np.allclose(np.diag(d),0,atol=1e-6),'DTW diagonal is not zero')
    status=json.loads(resolve_path(root,paths['phase1_status']).read_text())
    ensure(status.get('status')==exp['phase1_status'],'Profile-construction status mismatch')
    ensure(int(status.get('source_households'))==int(exp['source_households']),'Profile source count mismatch')
    ensure(int(status.get('prototype_fit_households'))==int(exp['fit_eligible_households']),'Profile fit count mismatch')
    ensure(int(status.get('low_signal_profiles'))==int(exp['low_signal_households']),'Profile low-signal count mismatch')
    audit=pd.read_csv(resolve_path(root,paths['profile_construction_audit']))
    ensure(audit['local_slot'].astype(int).tolist()==list(range(1,49)),'Profile slot audit mismatch')
    ensure(set(audit['excluded_local_dates'].astype(str))=={'2013-03-31'},'Profile date exclusion mismatch')
    tm=resolve_path(root,paths['time_map']); ensure(tm.is_file(),f'Missing time map: {tm}')
    required=set(cfg['time_rules']['required_time_map_columns']); have=set(pq.ParquetFile(tm).schema_arrow.names)
    ensure(required.issubset(have),f'Missing time-map columns: {sorted(required-have)}')
    ensure(cfg['scope_guards'].get('automatic_final_k_selection_allowed') is False,'K selection must not auto-overwrite reported K')
    if run_selftest: subprocess.run([sys.executable,str(release_root/'code/06_profile_clustering/k_selection/tests/profile_clustering_final_k_selftest.py')],check=True)
    print('PROFILE CLUSTERING K-SELECTION PREFLIGHT: PASS')

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--config',type=Path,required=True); ap.add_argument('--release-root',type=Path,required=True); ap.add_argument('--skip-selftest',action='store_true'); a=ap.parse_args()
    preflight(a.config,a.release_root,not a.skip_selftest); return 0
if __name__=='__main__': raise SystemExit(main())
