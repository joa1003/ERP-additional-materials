#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
resolve_cluster_python
cd "$ERP_PROJECT_ROOT"
run_step "STAGE 9.1 - Temporal-drift features" "$ERP_CLUSTER_PYTHON" code/08_interpretive_analysis/02_h12_profile_group/temporal_drift/src/temporal_drift_features_target_span_representativeness_temporal_drift.py --root "$ERP_PROJECT_ROOT"
run_step "STAGE 9.2 - Temporal-drift feature validation" "$ERP_CLUSTER_PYTHON" code/08_interpretive_analysis/02_h12_profile_group/temporal_drift/src/temporal_drift_features_validate_results.py "$ERP_PROJECT_ROOT"
run_step "STAGE 9.3 - Cross-strategy temporal-drift analysis" "$ERP_CLUSTER_PYTHON" code/08_interpretive_analysis/02_h12_profile_group/temporal_drift/src/temporal_drift_analysis_complete_cross_strategy_target_span_representativeness_temporal_drift.py --root "$ERP_PROJECT_ROOT"
echo; echo "STAGE 9 TEMPORAL DRIFT: PASS"
