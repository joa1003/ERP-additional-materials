#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
resolve_all_pythons
cd "$ERP_PROJECT_ROOT"
for stage in \
  run_stage_05_pooled_and_household_metrics.sh \
  run_stage_06_profile_clustering.sh \
  run_stage_07_profile_group.sh \
  run_stage_08_load_composition.sh \
  run_stage_09_temporal_drift.sh \
  run_stage_10_source_similarity.sh \
  run_stage_11_survey.sh \
  run_stage_12_report_figures.sh \
  run_stage_13_validate_results.sh; do
  bash "workflows/$stage"
done

echo; echo "POST-FORECASTING ANALYSIS WORKFLOW: PASS"
