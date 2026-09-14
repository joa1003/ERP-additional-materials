#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
resolve_torch_python
check_python_312 "$ERP_TORCH_PYTHON" "TORCH"
cd "$ERP_PROJECT_ROOT"

run_step "STAGE 3 - Forecast sample construction and scalers" \
  "$ERP_TORCH_PYTHON" code/03_sample_construction/sample_construction_production_indexes_scalers.py

for lead in 1 12 48; do
  require_file "$ERP_PROJECT_ROOT/outputs/tables/sample_construction/lcl_production_sample_index_h${lead}.parquet"
  require_file "$ERP_PROJECT_ROOT/outputs/tables/sample_construction/cer_production_sample_index_h${lead}.parquet"
done
require_file "$ERP_PROJECT_ROOT/outputs/tables/sample_construction/lcl_production_scaler_table.csv"
require_file "$ERP_PROJECT_ROOT/outputs/tables/sample_construction/cer_production_scaler_table.csv"
require_file "$ERP_PROJECT_ROOT/outputs/metadata/sample_construction/sample_construction_production_indexes_scalers_confirmation.md"

echo
echo "STAGE 3 SAMPLE CONSTRUCTION: PASS"
