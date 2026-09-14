#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
resolve_cluster_python
cd "$ERP_PROJECT_ROOT"
CFG="code/08_interpretive_analysis/02_h12_profile_group/group_heterogeneity/configs/profile_group_analysis_profile_group_analysis.json"
run_step "STAGE 7.1 - Profile-group input check" "$ERP_CLUSTER_PYTHON" code/08_interpretive_analysis/02_h12_profile_group/group_heterogeneity/src/profile_group_analysis_profile_group_check_inputs.py --config "$CFG"
run_step "STAGE 7.2 - Profile-group analysis" "$ERP_CLUSTER_PYTHON" code/08_interpretive_analysis/02_h12_profile_group/group_heterogeneity/src/profile_group_analysis_profile_group_analysis.py --config "$CFG"
run_step "STAGE 7.3 - Profile-group validation" "$ERP_CLUSTER_PYTHON" code/08_interpretive_analysis/02_h12_profile_group/group_heterogeneity/src/profile_group_analysis_profile_group_validate.py --config "$CFG"
echo; echo "STAGE 7 PROFILE-GROUP HETEROGENEITY: PASS"
