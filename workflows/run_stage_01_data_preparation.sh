#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
resolve_torch_python
check_python_312 "$ERP_TORCH_PYTHON" "TORCH"
cd "$ERP_PROJECT_ROOT"

bash workflows/check_raw_inputs.sh

run_step "STAGE 1.1 - CER control cohort" \
  "$ERP_TORCH_PYTHON" code/01_data_preparation/cohort_selection/build_cer_control_cohort.py --project-root "$ERP_PROJECT_ROOT"
require_file "$ERP_PROJECT_ROOT/outputs/tables/cer_residential_control_candidate_ids.csv"

run_step "STAGE 1.2 - LCL local-clock map" \
  "$ERP_TORCH_PYTHON" code/01_data_preparation/time_alignment/build_lcl_local_clock_map.py --project-root "$ERP_PROJECT_ROOT"
require_file "$ERP_PROJECT_ROOT/data/processed/time_aligned/lcl_full_local_clock_map.parquet"

run_step "STAGE 1.3 - CER canonical grid" \
  "$ERP_TORCH_PYTHON" code/01_data_preparation/time_alignment/build_cer_canonical_grid.py --project-root "$ERP_PROJECT_ROOT"
require_file "$ERP_PROJECT_ROOT/data/processed/time_aligned/cer_canonical_local_grid.parquet"
require_file "$ERP_PROJECT_ROOT/data/processed/time_aligned/cer_excluded_nonstandard_slots.parquet"

run_step "STAGE 1.4 - LCL source-period diagnostics" \
  "$ERP_TORCH_PYTHON" code/01_data_preparation/period_selection/lcl_source_period_diagnostics.py
require_file "$ERP_PROJECT_ROOT/outputs/tables/period_selection/lcl_candidate_period_comparison.csv"

run_step "STAGE 1.5 - CER target-period diagnostics" \
  "$ERP_TORCH_PYTHON" code/01_data_preparation/period_selection/cer_target_period_diagnostics.py --project-root "$ERP_PROJECT_ROOT"
require_file "$ERP_PROJECT_ROOT/outputs/tables/period_selection/cer_candidate_period_comparison.csv"

run_step "STAGE 1.6 - LCL eligibility" \
  "$ERP_TORCH_PYTHON" code/01_data_preparation/eligibility/build_lcl_eligibility.py
require_file "$ERP_PROJECT_ROOT/outputs/tables/horizon_impact/lcl_p0_final_eligible_households_new.csv"
require_file "$ERP_PROJECT_ROOT/outputs/tables/horizon_impact/lcl_rule3_lowest_30_training_window_counts.csv"

run_step "STAGE 1.7 - CER eligibility" \
  "$ERP_TORCH_PYTHON" code/01_data_preparation/eligibility/build_cer_eligibility.py --project-root "$ERP_PROJECT_ROOT"
require_file "$ERP_PROJECT_ROOT/outputs/tables/eligibility/cer_r1_r4_eligibility.csv"
require_file "$ERP_PROJECT_ROOT/outputs/tables/eligibility/cer_r1_r4_eligibility_flow.csv"

echo
echo "STAGE 1 DATA PREPARATION: PASS"
