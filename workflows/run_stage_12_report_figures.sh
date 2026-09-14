#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
resolve_cluster_python
cd "$ERP_PROJECT_ROOT"
run_step "STAGE 12.0 - Strategy-design schematic" "$ERP_CLUSTER_PYTHON" code/09_report_output_generation/figure_3_1_design_schematic.py --project-root "$ERP_PROJECT_ROOT"
run_step "STAGE 12.1 - Clustering dissertation figures" "$ERP_CLUSTER_PYTHON" code/09_report_output_generation/clustering/profile_clustering_dissertation_figures.py --project-root "$ERP_PROJECT_ROOT"
run_step "STAGE 12.2 - Target profile assignment figure" "$ERP_CLUSTER_PYTHON" code/09_report_output_generation/clustering/target_profile_assignment_fixed_source_and_target_profiles_with_composition.py
run_step "STAGE 12.3 - Group heterogeneity figures" "$ERP_CLUSTER_PYTHON" code/09_report_output_generation/group_heterogeneity_group_heterogeneity_figures.py
run_step "STAGE 12.4 - Source relevance figure" "$ERP_CLUSTER_PYTHON" code/09_report_output_generation/figure_4_2_source_relevance.py --project-root "$ERP_PROJECT_ROOT"
run_step "STAGE 12.5 - Load-composition figure" "$ERP_CLUSTER_PYTHON" code/09_report_output_generation/load_range_analysis_group_load_composition_figure.py
run_step "STAGE 12.6 - Temporal-drift L-D figures" "$ERP_CLUSTER_PYTHON" code/09_report_output_generation/plot_temporal_drift_analysis_temporal_drift_ld_gain_with_appendix_metrics.py
run_step "STAGE 12.7 - Load-range appendix figures" "$ERP_CLUSTER_PYTHON" code/09_report_output_generation/load_range_analysis_appendix_j2_bar_and_cumulative_figures.py
run_step "STAGE 12.8 - D-T main-text figures" "$ERP_CLUSTER_PYTHON" code/09_report_output_generation/load_range_analysis_dt_maintext_figures.py
run_step "STAGE 12.9 - L-F figures" "$ERP_CLUSTER_PYTHON" code/09_report_output_generation/load_range_analysis_temporal_drift_analysis_lf_figures.py
run_step "STAGE 12.10 - F-D figures" "$ERP_CLUSTER_PYTHON" code/09_report_output_generation/load_range_analysis_temporal_drift_analysis_fd_figures.py
run_step "STAGE 12.11 - Submitted dissertation report tables" "$ERP_CLUSTER_PYTHON" code/09_report_output_generation/generate_submitted_report_tables.py --project-root "$ERP_PROJECT_ROOT"
run_step "STAGE 12.12 - Collect dissertation-numbered figures" "$ERP_CLUSTER_PYTHON" code/09_report_output_generation/collect_submitted_report_figures.py --project-root "$ERP_PROJECT_ROOT"
run_step "STAGE 12.13 - Submitted dissertation report artifact inventory" "$ERP_CLUSTER_PYTHON" validation/validate_report_artifacts.py --project-root "$ERP_PROJECT_ROOT"
echo; echo "STAGE 12 REPORT FIGURES AND TABLES: PASS"
