#!/usr/bin/env python3
"""Validate that every submitted dissertation figure has been regenerated."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=Path.cwd())
    a = ap.parse_args()
    root = a.project_root.resolve()

    load_dir = root / "outputs/figures/load_range_analysis_complete_cross_strategy_raw_load_composition_standardised_gain_decomposition"
    drift_dir = root / "outputs/figures/temporal_drift_analysis_complete_cross_strategy_target_span_representativeness_temporal_drift"
    cluster_dir = root / "outputs/figures/profile_clustering_clustering_final_decision_dissertation"
    group_dir = root / "outputs/figures/group_heterogeneity_group_heterogeneity_practical_significance"
    report_dir = root / "outputs/figures/report"

    expected = {
        "Figure 3.1": [report_dir / "figure_3_1_strategy_design.png"],
        "Figure 3.2": [cluster_dir / "profile_clustering_final_selected_source_prototypes_dissertation.png"],
        "Figure 4.1": [group_dir / "figure_4_1_relative_mae_ld_gain_by_group_h12.png"],
        "Figure 4.2": [report_dir / "figure_4_2_source_similarity.png", report_dir / "figure_4_2_local_source_support.png"],
        "Figure 4.3": [load_dir / "figure_4_3_ld_group_load_composition.png"],
        "Figure 4.4": [drift_dir / "figure_4_4_ld_relative_gain_by_temporal_drift_quartile.png"],
        "Figure 4.5": [load_dir / "figure_4_5_ld_cumulative_load_range_contribution.png"],
        "Figure 4.6": [load_dir / "figure_4_6_dt_gain_by_load_range.png"],
        "Figure 4.7": [load_dir / "figure_4_7_dt_cumulative_load_range_contribution.png"],
        "Figure 4.8": [load_dir / "figure_4_8_lf_gain_by_load_range.png"],
        "Figure 4.9": [load_dir / "figure_4_9_lf_cumulative_load_range_contribution.png"],
        "Figure 4.10": [load_dir / "figure_4_10_lf_gain_by_drift_quartile.png"],
        "Figure 4.11": [load_dir / "figure_4_11_fd_gain_by_load_range.png"],
        "Figure 4.12": [load_dir / "figure_4_12_fd_cumulative_load_range_contribution.png"],
        "Figure 4.13": [load_dir / "figure_4_13_fd_gain_by_drift_quartile.png"],
        "Figure B.1": [root / "outputs/figures/eligibility/lcl_rule3_training_window_counts.png"],
        "Figure C.1": [root / "outputs/figures/eda/appendix_c1_average_daily_load_profiles.png"],
        "Figure C.2": [root / "outputs/figures/eda/appendix_c2_household_mean_load_distribution.png"],
        "Figure D.1": [root / "outputs/figures/horizon_impact/forecast_lead_analysis.png"],
        "Figure E.1": [cluster_dir / "profile_clustering_candidate_k_selection_evidence_dissertation.png"],
        "Figure E.2": [cluster_dir / "profile_clustering_k4_vs_k5_paired_metric_comparison_dissertation.png"],
        "Figure E.3": [report_dir / "appendix_e_target_profile_assignment.png"],
        "Figure K.1": [drift_dir / "appendix_k_figure_k1_ld_relative_mae_gain_by_group.png"],
        "Figure K.2": [drift_dir / "appendix_k_figure_k2_ld_relative_rmse_gain_by_group.png"],
        "Figure K.3": [drift_dir / "appendix_k_figure_k3_ld_relative_smape_gain_by_group.png"],
    }

    failures = []
    print("=" * 96)
    print("SUBMITTED DISSERTATION FIGURE INVENTORY")
    print("=" * 96)
    for label, paths in expected.items():
        ok = True
        for path in paths:
            if not path.is_file() or path.stat().st_size < 10_000:
                failures.append(f"{label}: missing or unexpectedly small: {path.relative_to(root)}")
                ok = False
        print(f"{label:<12} : {'PASS' if ok else 'FAIL'}")

    if failures:
        print("\nREPORT FIGURE INVENTORY: FAIL")
        for failure in failures:
            print("-", failure)
        return 1

    collected = root / "outputs/report_figures"
    collected_pngs = sorted(collected.glob("Figure_*.png")) if collected.is_dir() else []
    if len(collected_pngs) != 26:
        print(f"\nREPORT FIGURE INVENTORY: FAIL - dissertation-numbered collection has {len(collected_pngs)}/26 PNG files")
        return 1
    print("\nREPORT FIGURE INVENTORY: PASS")
    print("Figure captions covered: 25")
    print("Rendered PNG files       : 26 (Figure 4.2 has two submitted components)")
    print("Stable collection        : outputs/report_figures/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
