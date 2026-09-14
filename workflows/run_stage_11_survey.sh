#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
resolve_all_pythons
cd "$ERP_PROJECT_ROOT"
run_step "STAGE 11.1 - Survey linkage" "$ERP_TORCH_PYTHON" code/07_survey_data_governance/linkage/src/survey_linkage_survey_inventory_linkage.py --root "$ERP_PROJECT_ROOT" --config code/07_survey_data_governance/linkage/configs/survey_linkage_survey_inventory_linkage.json --mode run
run_step "STAGE 11.2 - Survey recoding" "$ERP_TORCH_PYTHON" code/07_survey_data_governance/recoding/src/survey_recoding_survey_recoding_harmonisation.py --root "$ERP_PROJECT_ROOT" --config code/07_survey_data_governance/recoding/configs/survey_recoding_survey_recoding_harmonisation.json --run
run_step "STAGE 11.3 - Survey group composition" "$ERP_CLUSTER_PYTHON" code/08_interpretive_analysis/03_h12_survey_posthoc/group_composition/src/survey_group_composition_survey_group_composition.py --root "$ERP_PROJECT_ROOT" --config code/08_interpretive_analysis/03_h12_survey_posthoc/group_composition/configs/survey_group_composition_survey_group_composition.json --run
run_step "STAGE 11.4 - Group-adjusted survey gain associations" "$ERP_CLUSTER_PYTHON" code/08_interpretive_analysis/03_h12_survey_posthoc/group_adjusted_gain/src/survey_gain_associations_group_adjusted_survey_gain_associations.py --root "$ERP_PROJECT_ROOT" --config code/08_interpretive_analysis/03_h12_survey_posthoc/group_adjusted_gain/configs/survey_gain_associations_group_adjusted_survey_gain_associations_config.json --run
echo; echo "STAGE 11 SURVEY ANALYSES: PASS"
