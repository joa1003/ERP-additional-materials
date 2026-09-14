#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
resolve_cluster_python
cd "$ERP_PROJECT_ROOT"
run_step "STAGE 10.1 - Local source support" "$ERP_CLUSTER_PYTHON" code/08_interpretive_analysis/02_h12_profile_group/source_relevance/src/build_local_source_support.py --project-root "$ERP_PROJECT_ROOT"
run_step "STAGE 10.2 - Source-side cross-strategy diagnostics" "$ERP_CLUSTER_PYTHON" code/08_interpretive_analysis/02_h12_profile_group/source_relevance/src/source_side_cross_strategy_appendix_source_side_cross_strategy_diagnostics.py --project-root "$ERP_PROJECT_ROOT" --config code/08_interpretive_analysis/02_h12_profile_group/source_relevance/configs/source_side_cross_strategy_appendix_source_side_cross_strategy_diagnostics_config.json
run_step "STAGE 10.3 - Cross-strategy profile-group appendix closure" "$ERP_CLUSTER_PYTHON" code/08_interpretive_analysis/02_h12_profile_group/group_heterogeneity/src/cross_strategy_profile_group_cross_strategy_profile_group_appendix_closure.py --project-root "$ERP_PROJECT_ROOT" --config code/08_interpretive_analysis/02_h12_profile_group/group_heterogeneity/configs/cross_strategy_profile_group_cross_strategy_profile_group_appendix_closure_config.json
echo; echo "STAGE 10 SOURCE SIMILARITY: PASS"
