#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"

usage() {
  cat <<'TXT'
Usage:
  bash workflows/run_stage_04_one_seed.sh <lead> <seed>

Runs the five formal forecasting invocations for one lead/seed pair in the
required dependency order: source model first, then the four target strategies.
This helper is intended for GPU scheduler jobs and does not change the formal
forecasting protocol.

Allowed leads: 1, 12, 48
Allowed seeds: 42, 123, 2026, 31415
TXT
}

[[ $# -eq 2 ]] || { usage >&2; exit 2; }
lead="$1"
seed="$2"

case "$lead" in
  1|12|48) ;;
  *) fail "Unsupported forecast lead: $lead (allowed: 1, 12, 48)" ;;
esac
case "$seed" in
  42|123|2026|31415) ;;
  *) fail "Unsupported formal seed: $seed (allowed: 42, 123, 2026, 31415)" ;;
esac

resolve_torch_python
check_python_312 "$ERP_TORCH_PYTHON" "TORCH"
cd "$ERP_PROJECT_ROOT"
check_cuda

# Stage 3 outputs consumed by every forecasting runner.
for h in 1 12 48; do
  require_file "$ERP_PROJECT_ROOT/outputs/tables/sample_construction/lcl_production_sample_index_h${h}.parquet"
  require_file "$ERP_PROJECT_ROOT/outputs/tables/sample_construction/cer_production_sample_index_h${h}.parquet"
done
require_file "$ERP_PROJECT_ROOT/outputs/tables/sample_construction/lcl_production_scaler_table.csv"
require_file "$ERP_PROJECT_ROOT/outputs/tables/sample_construction/cer_production_scaler_table.csv"

runner="code/04_forecasting/run_h${lead}_formal_forecasting.py"
config="configs/forecasting_h${lead}.yaml"
require_file "$ERP_PROJECT_ROOT/$runner"
require_file "$ERP_PROJECT_ROOT/$config"

run_step "STAGE 4 - h=${lead}, seed=${seed}, source model" \
  "$ERP_TORCH_PYTHON" "$runner" \
  --project-root "$ERP_PROJECT_ROOT" \
  --config "$config" \
  --strategy lcl_source \
  --seed "$seed"

for strategy in direct_transfer fine_tuning cer_scratch_limited cer_scratch_full; do
  run_step "STAGE 4 - h=${lead}, seed=${seed}, strategy=${strategy}" \
    "$ERP_TORCH_PYTHON" "$runner" \
    --project-root "$ERP_PROJECT_ROOT" \
    --config "$config" \
    --strategy "$strategy" \
    --seed "$seed"
done

"$ERP_TORCH_PYTHON" validation/validate_stage_04_forecasting.py --project-root "$ERP_PROJECT_ROOT" --lead "$lead" --seed "$seed"

echo
echo "STAGE 4 LEAD/SEED BUNDLE: PASS (h=${lead}, seed=${seed})"
