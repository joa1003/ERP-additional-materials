from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from profile_clustering_final_decision_common import ensure, load_json, resolve

def validate(config_path: Path) -> None:
    cfg=load_json(config_path); root=Path(cfg['project_root']).resolve(); p=cfg['paths']
    out_t=resolve(root,p['output_tables']); out_m=resolve(root,p['output_metadata'])
    required=[
        out_t/'profile_clustering_k45_iteration_initialisation_audit.csv',
        out_t/'profile_clustering_k45_transition_by_seed.csv',
        out_t/'profile_clustering_k45_structural_decision_evidence.csv',
        out_t/'profile_clustering_k4_vs_k5_decision_table_dissertation.csv',
        out_t/'profile_clustering_final_selected_profile_groups_dissertation.csv',
        out_t/'profile_clustering_final_selected_source_prototypes.csv',
        out_m/'profile_clustering_final_decision_dissertation_status.json',
        resolve(root,p['decision_document']),
    ]
    for path in required: ensure(path.is_file() and path.stat().st_size>0,f'Missing output: {path}')
    audit=pd.read_csv(out_t/'profile_clustering_k45_iteration_initialisation_audit.csv')
    ensure(len(audit)==40,'Iteration audit must contain 40 paired K4/K5 runs')
    ensure(audit.groupby('k').seed.nunique().to_dict()=={4:20,5:20},'Seed count mismatch')
    ensure((~audit.iteration_cap_hit.astype(bool)).all(),'Iteration cap hit in audit')
    transitions=pd.read_csv(out_t/'profile_clustering_k45_transition_by_seed.csv')
    ensure(len(transitions)==20,'Transition audit must contain 20 paired seeds')
    groups=pd.read_csv(out_t/'profile_clustering_final_selected_profile_groups_dissertation.csv')
    ensure(int(groups.all_assigned_n.sum())==3843,'Final source group counts do not sum to 3,843')
    proto=pd.read_csv(out_t/'profile_clustering_final_selected_source_prototypes.csv')
    k=groups.profile_group.nunique(); ensure(len(proto)==k*48,'Final prototype row count mismatch')
    print('PROFILE CLUSTERING FINAL DECISION OUTPUT VALIDATION PASS')

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True,type=Path); a=ap.parse_args(); validate(a.config); return 0
if __name__=='__main__': raise SystemExit(main())
