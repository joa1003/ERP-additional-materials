#!/usr/bin/env python3
"""Collect regenerated dissertation figures into one stable report-facing directory."""
from __future__ import annotations
import argparse, shutil
from pathlib import Path


def mapping(root: Path):
    load=root/'outputs/figures/load_range_analysis_complete_cross_strategy_raw_load_composition_standardised_gain_decomposition'
    drift=root/'outputs/figures/temporal_drift_analysis_complete_cross_strategy_target_span_representativeness_temporal_drift'
    cluster=root/'outputs/figures/profile_clustering_clustering_final_decision_dissertation'
    group=root/'outputs/figures/group_heterogeneity_group_heterogeneity_practical_significance'
    report=root/'outputs/figures/report'
    return {
      'Figure_3_1.png': report/'figure_3_1_strategy_design.png',
      'Figure_3_2.png': cluster/'profile_clustering_final_selected_source_prototypes_dissertation.png',
      'Figure_4_1.png': group/'figure_4_1_relative_mae_ld_gain_by_group_h12.png',
      'Figure_4_2a_source_prototype_distance.png': report/'figure_4_2_source_similarity.png',
      'Figure_4_2b_local_source_support.png': report/'figure_4_2_local_source_support.png',
      'Figure_4_3.png': load/'figure_4_3_ld_group_load_composition.png',
      'Figure_4_4.png': drift/'figure_4_4_ld_relative_gain_by_temporal_drift_quartile.png',
      'Figure_4_5.png': load/'figure_4_5_ld_cumulative_load_range_contribution.png',
      'Figure_4_6.png': load/'figure_4_6_dt_gain_by_load_range.png',
      'Figure_4_7.png': load/'figure_4_7_dt_cumulative_load_range_contribution.png',
      'Figure_4_8.png': load/'figure_4_8_lf_gain_by_load_range.png',
      'Figure_4_9.png': load/'figure_4_9_lf_cumulative_load_range_contribution.png',
      'Figure_4_10.png': load/'figure_4_10_lf_gain_by_drift_quartile.png',
      'Figure_4_11.png': load/'figure_4_11_fd_gain_by_load_range.png',
      'Figure_4_12.png': load/'figure_4_12_fd_cumulative_load_range_contribution.png',
      'Figure_4_13.png': load/'figure_4_13_fd_gain_by_drift_quartile.png',
      'Figure_B_1.png': root/'outputs/figures/eligibility/lcl_rule3_training_window_counts.png',
      'Figure_C_1.png': root/'outputs/figures/eda/appendix_c1_average_daily_load_profiles.png',
      'Figure_C_2.png': root/'outputs/figures/eda/appendix_c2_household_mean_load_distribution.png',
      'Figure_D_1.png': root/'outputs/figures/horizon_impact/forecast_lead_analysis.png',
      'Figure_E_1.png': cluster/'profile_clustering_candidate_k_selection_evidence_dissertation.png',
      'Figure_E_2.png': cluster/'profile_clustering_k4_vs_k5_paired_metric_comparison_dissertation.png',
      'Figure_E_3.png': report/'appendix_e_target_profile_assignment.png',
      'Figure_K_1.png': drift/'appendix_k_figure_k1_ld_relative_mae_gain_by_group.png',
      'Figure_K_2.png': drift/'appendix_k_figure_k2_ld_relative_rmse_gain_by_group.png',
      'Figure_K_3.png': drift/'appendix_k_figure_k3_ld_relative_smape_gain_by_group.png',
    }


def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--project-root',type=Path,default=Path.cwd()); ap.add_argument('--allow-partial',action='store_true'); a=ap.parse_args(); root=a.project_root.resolve(); out=root/'outputs/report_figures'; out.mkdir(parents=True,exist_ok=True)
 missing=[]; copied=0
 for name,src in mapping(root).items():
  if src.is_file(): shutil.copy2(src,out/name); copied+=1; print(f'  collected {name}')
  else: missing.append((name,src))
 print(f'REPORT FIGURES COLLECTED: {copied}/26 PNG files')
 if missing and not a.allow_partial:
  for name,src in missing: print(f'MISSING {name}: {src.relative_to(root)}')
  return 1
 if missing: print('Deferred:', ', '.join(n for n,_ in missing))
 return 0
if __name__=='__main__': raise SystemExit(main())
