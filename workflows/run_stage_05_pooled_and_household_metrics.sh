#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
resolve_cluster_python
check_python_312 "$ERP_CLUSTER_PYTHON" "CLUSTER"
cd "$ERP_PROJECT_ROOT"

run_step "STAGE 5.1 - Pooled cross-lead comparison" \
  "$ERP_CLUSTER_PYTHON" code/08_interpretive_analysis/01_pooled_cross_lead/three_lead_common_row_comparison.py --project-root "$ERP_PROJECT_ROOT"
run_step "STAGE 5.2 - Household analytical dataset" \
  "$ERP_CLUSTER_PYTHON" code/05_metric_generation/household_analytical_dataset/src/build_household_analytical_dataset.py --project-root "$ERP_PROJECT_ROOT"

echo
echo "STAGE 5 POOLED AND HOUSEHOLD METRICS: PASS"
