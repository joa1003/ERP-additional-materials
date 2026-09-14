from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd
from profile_clustering_final_decision_common import ensure, load_json, resolve

def preflight(config_path:Path, release_root:Path, run_selftest:bool=True)->None:
    cfg=load_json(config_path); root=release_root.resolve(); p=cfg['paths']
    for name in ['numpy','pandas','pyarrow','matplotlib']: __import__(name)
    for key in ['run_diagnostics','all_assignments','phase1_status','profile_construction_audit','time_map']:
        ensure(resolve(root,p[key]).is_file(),f'Missing generated input {key}: {resolve(root,p[key])}')
    run_df=pd.read_csv(resolve(root,p['run_diagnostics']))
    assn=pd.read_parquet(resolve(root,p['all_assignments']),columns=['household_id','k','seed','cluster_id_aligned_to_reference','prototype_fit_eligible','low_signal_prototype_fit_exclusion'])
    seeds=sorted(int(x) for x in cfg['expected']['focused_seeds'])
    ensure(assn['household_id'].nunique()==cfg['expected']['source_households'],'Source household count mismatch')
    for k in [4,5]:
        frame=run_df[(run_df.k==k)&(run_df.seed.isin(seeds))]
        ensure(len(frame)==20,f'Expected 20 K={k} runs')
        ensure(sorted(frame.seed.astype(int).tolist())==seeds,f'Seed set mismatch K={k}')
        ensure((~frame.reached_iteration_cap.astype(bool)).all(),f'Iteration cap hit K={k}')
        af=assn[(assn.k==k)&(assn.seed.isin(seeds))]
        ensure(len(af)==20*cfg['expected']['source_households'],f'Assignment rows mismatch K={k}')
    phase=json.loads(resolve(root,p['phase1_status']).read_text())
    ensure(phase.get('status')=='COMPLETE_PASS','Profile construction status mismatch')
    ensure(int(phase.get('source_households'))==3843,'Source cohort mismatch')
    ensure(int(phase.get('prototype_fit_households'))==3789,'Fit cohort mismatch')
    ensure(int(phase.get('low_signal_profiles'))==54,'Low-signal count mismatch')
    ensure(phase.get('profile_period_start')==cfg['time_rules']['profile_period_start'],'Profile start mismatch')
    ensure(phase.get('profile_period_end')==cfg['time_rules']['profile_period_end'],'Profile end mismatch')
    ensure(phase.get('excluded_local_dates')==cfg['time_rules']['excluded_local_dates'],'Excluded-date mismatch')
    ensure(int(phase.get('validation_rows_used'))==0 and int(phase.get('test_rows_used'))==0 and int(phase.get('target_rows_used'))==0,'Non-training rows used in source profile construction')
    audit=pd.read_csv(resolve(root,p['profile_construction_audit']))
    ensure('2013-03-31' in ' '.join(audit.astype(str).to_numpy().ravel()),'DST exclusion absent from audit')
    if run_selftest: subprocess.run([sys.executable,str(release_root/'code/06_profile_clustering/final_k4_clustering/tests/profile_clustering_final_decision_selftest.py')],check=True)
    print('PROFILE CLUSTERING FINAL K=4 PREFLIGHT: PASS')

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--config',type=Path,required=True); ap.add_argument('--release-root',type=Path,required=True); ap.add_argument('--skip-selftest',action='store_true'); a=ap.parse_args(); preflight(a.config,a.release_root,not a.skip_selftest); return 0
if __name__=='__main__': raise SystemExit(main())
