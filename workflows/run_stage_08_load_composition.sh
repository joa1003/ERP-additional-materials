#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
resolve_cluster_python
cd "$ERP_PROJECT_ROOT"
run_step "STAGE 8.1 - Raw load structure" "$ERP_CLUSTER_PYTHON" code/08_interpretive_analysis/02_h12_profile_group/load_range/src/build_raw_load_structure.py --project-root "$ERP_PROJECT_ROOT"
run_step "STAGE 8.2 - Group practical significance" "$ERP_CLUSTER_PYTHON" code/08_interpretive_analysis/02_h12_profile_group/group_heterogeneity/src/group_heterogeneity_group_heterogeneity_practical_significance.py --root "$ERP_PROJECT_ROOT"
run_step "STAGE 8.3 - Group result validation" "$ERP_CLUSTER_PYTHON" code/08_interpretive_analysis/02_h12_profile_group/group_heterogeneity/src/group_heterogeneity_validate_results.py --root "$ERP_PROJECT_ROOT"
run_step "STAGE 8.4 - Load-range decomposition" "$ERP_CLUSTER_PYTHON" code/08_interpretive_analysis/02_h12_profile_group/load_range/src/load_range_analysis_complete_cross_strategy_raw_load_composition_standardised_gain_decomposition.py --root "$ERP_PROJECT_ROOT"
echo; echo "STAGE 8 LOAD COMPOSITION: PASS"
