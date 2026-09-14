#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
resolve_cluster_python
check_python_312 "$ERP_CLUSTER_PYTHON" "CLUSTER"
cd "$ERP_PROJECT_ROOT"

run_step "STAGE 6.1 - Source training profiles" \
  "$ERP_CLUSTER_PYTHON" code/06_profile_clustering/profile_construction/src/build_source_training_profiles.py --project-root "$ERP_PROJECT_ROOT"

run_step "STAGE 6.2 - K-selection preflight" \
  "$ERP_CLUSTER_PYTHON" code/06_profile_clustering/k_selection/src/profile_clustering_final_k_preflight.py --config code/06_profile_clustering/k_selection/configs/profile_clustering_final_k_selection.yaml --release-root "$ERP_PROJECT_ROOT"
run_step "STAGE 6.3 - K-selection analysis" \
  "$ERP_CLUSTER_PYTHON" code/06_profile_clustering/k_selection/src/profile_clustering_final_k_analysis.py --config code/06_profile_clustering/k_selection/configs/profile_clustering_final_k_selection.yaml
run_step "STAGE 6.4 - K-selection validation" \
  "$ERP_CLUSTER_PYTHON" code/06_profile_clustering/k_selection/src/profile_clustering_final_k_validate.py --config code/06_profile_clustering/k_selection/configs/profile_clustering_final_k_selection.yaml

run_step "STAGE 6.5 - Final K=4 preflight" \
  "$ERP_CLUSTER_PYTHON" code/06_profile_clustering/final_k4_clustering/src/profile_clustering_final_decision_preflight.py --config code/06_profile_clustering/final_k4_clustering/configs/profile_clustering_final_decision_dissertation.json --release-root "$ERP_PROJECT_ROOT"
run_step "STAGE 6.6 - Final K=4 analysis" \
  "$ERP_CLUSTER_PYTHON" code/06_profile_clustering/final_k4_clustering/src/profile_clustering_final_decision_analysis.py --config code/06_profile_clustering/final_k4_clustering/configs/profile_clustering_final_decision_dissertation.json
run_step "STAGE 6.7 - Final K=4 validation" \
  "$ERP_CLUSTER_PYTHON" code/06_profile_clustering/final_k4_clustering/src/profile_clustering_final_decision_validate.py --config code/06_profile_clustering/final_k4_clustering/configs/profile_clustering_final_decision_dissertation.json

run_step "STAGE 6.8 - Target assignment preflight" \
  "$ERP_CLUSTER_PYTHON" code/06_profile_clustering/target_assignment/src/target_profile_assignment_cer_assignment_preflight.py --config code/06_profile_clustering/target_assignment/configs/target_profile_assignment_cer_assignment.json --release-root "$ERP_PROJECT_ROOT"
run_step "STAGE 6.9 - Target assignment analysis" \
  "$ERP_CLUSTER_PYTHON" code/06_profile_clustering/target_assignment/src/target_profile_assignment_cer_assignment_analysis.py --config code/06_profile_clustering/target_assignment/configs/target_profile_assignment_cer_assignment.json
run_step "STAGE 6.10 - Target assignment validation" \
  "$ERP_CLUSTER_PYTHON" code/06_profile_clustering/target_assignment/src/target_profile_assignment_cer_assignment_validate.py --config code/06_profile_clustering/target_assignment/configs/target_profile_assignment_cer_assignment.json

echo
echo "STAGE 6 PROFILE CLUSTERING: PASS"
