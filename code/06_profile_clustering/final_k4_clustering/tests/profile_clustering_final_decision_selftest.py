from __future__ import annotations
import json, tempfile, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve(); RELEASE=HERE.parents[1]; sys.path.insert(0,str(RELEASE/'src'))
from profile_clustering_final_decision_analysis import run

SEEDS=[42,123,2026,31415,7,11,19,31,53,71,97,131,173,211,257,307,353,401,457,509]

def prototype(peak: int, width: float=3.0) -> np.ndarray:
    x=np.arange(1,49); base=-0.6+0.05*np.sin(x/3)
    return base+2.5*np.exp(-0.5*((x-peak)/width)**2)

def main()->int:
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); tables=root/'inputs'; tables.mkdir(parents=True)
        households=[f'H{i:03d}' for i in range(100)]
        run_rows=[]; comp=[]; prot=[]; assn=[]
        for seed in SEEDS:
            for k in [4,5]:
                if k==4:
                    labels=np.repeat([1,2,3,4],25)
                    peaks=[16,37,41,45]
                    sizes=[25]*4
                    sil=0.16+((seed%5)-2)*0.001
                    inertia=4.1+(seed%7)*0.002
                else:
                    labels=np.array([1]*15+[5]*10+[2]*25+[3]*25+[4]*25)
                    peaks=[16,37,41,45,39]
                    sizes=[15,25,25,25,10]
                    sil=0.15+((seed%5)-2)*0.001
                    inertia=3.85+(seed%7)*0.002
                run_rows.append({'k':k,'seed':seed,'run_scope':'focused','reference_seed':42,'inertia_dtw':inertia,
                    'full_sample_dtw_silhouette':sil,'minimum_cluster_size_fit':min(sizes),'maximum_cluster_size_fit':max(sizes),
                    'minimum_cluster_share_fit':min(sizes)/100,'maximum_cluster_share_fit':max(sizes)/100,'cluster_size_cv_fit':0.1,
                    'minimum_cluster_size_all_assigned':min(sizes),'maximum_cluster_size_all_assigned':max(sizes),
                    'minimum_cluster_share_all_assigned':min(sizes)/100,'maximum_cluster_share_all_assigned':max(sizes)/100,
                    'cluster_size_cv_all_assigned':0.1,'maximum_low_signal_count_in_one_group':1,'maximum_within_group_low_signal_share':0.02,
                    'fit_seconds':1.0,'iterations':30+(seed%5),'max_iter_configured':100,'max_iter_barycenter_configured':100,
                    'tolerance':1e-6,'reached_iteration_cap':False,'converged_proxy':True})
                for cid,(size,peak) in enumerate(zip(sizes,peaks),start=1):
                    comp.append({'k':k,'seed':seed,'run_scope':'focused','cluster_id_aligned_to_reference':cid,
                        'fit_eligible_n':size,'fit_eligible_share':size/100,'all_assigned_n':size,'all_assigned_share':size/100,
                        'low_signal_n':0,'within_group_low_signal_share':0.0,'matched_prototype_dtw_to_reference':0.0})
                    vec=prototype(peak)
                    for slot,val in enumerate(vec,start=1):
                        prot.append({'k':k,'seed':seed,'run_scope':'focused','reference_seed':42,'cluster_id_aligned_to_reference':cid,
                            'local_slot':slot,'prototype_zscore':val,'fit_eligible_n':size,'all_assigned_n':size})
                for hid,label in zip(households,labels):
                    assn.append({'household_id':hid,'k':k,'seed':seed,'run_scope':'focused','cluster_id_aligned_to_reference':int(label),
                        'low_signal_prototype_fit_exclusion':False,'prototype_fit_eligible':True})
        pd.DataFrame(run_rows).to_csv(tables/'runs.csv',index=False)
        pd.DataFrame(comp).to_csv(tables/'composition.csv',index=False)
        pd.DataFrame(prot).to_csv(tables/'prototypes.csv',index=False)
        pd.DataFrame(assn).to_csv(tables/'assignments.csv',index=False)
        ktab=pd.DataFrame({'k':range(2,9),'inertia_mean':[5,4.5,4.1,3.85,3.7,3.6,3.5],
            'silhouette_mean':[.29,.19,.16,.15,.13,.11,.10],
            'minimum_cluster_share_fit_minimum':[.4,.2,.15,.1,.08,.06,.04],
            'pairwise_ari_mean':[.9,.45,.5,.55,.4,.35,.3]})
        ktab.to_csv(tables/'ktable.csv',index=False)
        focused=pd.DataFrame({'k':[4,5,6],'pairwise_ari_mean':[.45,.49,.46]}); focused.to_csv(tables/'focused.csv',index=False)
        pd.DataFrame({'k':[]}).to_csv(tables/'ari.csv',index=False)
        cfg={
            'project_root':str(root),'write_parquet_outputs':False,
            'paths':{'run_diagnostics':'inputs/runs.csv','candidate_k_table':'inputs/ktable.csv','focused_summary':'inputs/focused.csv',
                'cluster_composition':'inputs/composition.csv','candidate_prototypes':'inputs/prototypes.csv','all_assignments':'inputs/assignments.csv',
                'pairwise_ari':'inputs/ari.csv','output_tables':'outputs/tables','output_figures':'outputs/figures','output_metadata':'outputs/metadata',
                'decision_document':'documentation/decision.md'},
            'expected':{'focused_seeds':SEEDS},
            'candidate_run_rule':{'minimum_all_assigned_share':.05,'maximum_within_group_low_signal_share':.06,
                'require_silhouette_at_or_above_candidate_median':True},
            'k5_structural_rule':{'parent_mode_share_minimum':.8,'median_origin_purity_minimum':.75,'median_extra_group_share_minimum':.08,
                'peak_band_mode_share_minimum':.75,'median_two_child_parent_coverage_minimum':.8,'median_distinctness_ratio_minimum':.6,
                'mean_silhouette_loss_maximum':.02,'mean_minimum_share_loss_maximum':.03,'minimum_k5_group_share_floor':.05}
        }
        cpath=root/'config.json'; cpath.write_text(json.dumps(cfg))
        status=run(cpath)
        assert status['source_households']==100
        assert (root/'outputs/tables/profile_clustering_k45_transition_by_seed.csv').is_file()
        assert len(pd.read_csv(root/'outputs/tables/profile_clustering_k45_transition_by_seed.csv'))==20
        print('PROFILE CLUSTERING FINAL DECISION SYNTHETIC END-TO-END SELFTEST: PASS')
    return 0
if __name__=='__main__': raise SystemExit(main())
