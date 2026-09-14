#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
resolve_torch_python
check_python_312 "$ERP_TORCH_PYTHON" "TORCH"
cd "$ERP_PROJECT_ROOT"

run_step "STAGE 2.1 - Reported cohort diagnostics" \
  "$ERP_TORCH_PYTHON" code/02_eda_and_data_diagnostics/generate_reported_eda.py --project-root "$ERP_PROJECT_ROOT"
require_file "$ERP_PROJECT_ROOT/outputs/tables/eda/eda_48slot_average_daily_profile.csv"
require_file "$ERP_PROJECT_ROOT/outputs/tables/eda/eda_load_distribution_summary.csv"

run_step "STAGE 2.2 - Forecast-lead diagnostic" \
  "$ERP_TORCH_PYTHON" code/02_eda_and_data_diagnostics/generate_forecast_lead_diagnostic.py --project-root "$ERP_PROJECT_ROOT"
require_file "$ERP_PROJECT_ROOT/outputs/tables/horizon_impact/lcl_final3843_horizon_difficulty_summary_h1_h48.csv"
require_file "$ERP_PROJECT_ROOT/outputs/figures/horizon_impact/forecast_lead_analysis.png"

echo
echo "STAGE 2 DIAGNOSTICS: PASS"
