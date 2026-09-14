#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
resolve_torch_python
check_python_312 "$ERP_TORCH_PYTHON" "TORCH"
cd "$ERP_PROJECT_ROOT"
check_cuda

for lead in 1 12 48; do
  runner="code/04_forecasting/run_h${lead}_formal_forecasting.py"
  config="configs/forecasting_h${lead}.yaml"
  for seed in 42 123 2026 31415; do
    run_step "STAGE 4 - h=${lead}, seed=${seed}, source model" \
      "$ERP_TORCH_PYTHON" "$runner" --project-root "$ERP_PROJECT_ROOT" --config "$config" --strategy lcl_source --seed "$seed"
    for strategy in direct_transfer fine_tuning cer_scratch_limited cer_scratch_full; do
      run_step "STAGE 4 - h=${lead}, seed=${seed}, strategy=${strategy}" \
        "$ERP_TORCH_PYTHON" "$runner" --project-root "$ERP_PROJECT_ROOT" --config "$config" --strategy "$strategy" --seed "$seed"
    done
  done
done

"$ERP_TORCH_PYTHON" validation/validate_stage_04_forecasting.py --project-root "$ERP_PROJECT_ROOT"

echo
echo "STAGE 4 FORMAL FORECASTING: PASS"
