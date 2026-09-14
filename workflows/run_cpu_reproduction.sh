#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
resolve_all_pythons
check_python_312 "$ERP_TORCH_PYTHON" "TORCH"
check_python_312 "$ERP_CLUSTER_PYTHON" "CLUSTER"
cd "$ERP_PROJECT_ROOT"

echo "Repository root : $ERP_PROJECT_ROOT"
echo "TORCH Python    : $ERP_TORCH_PYTHON"
echo "CLUSTER Python  : $ERP_CLUSTER_PYTHON"

bash workflows/check_raw_inputs.sh
check_disk_space_25gib "$ERP_PROJECT_ROOT"
bash workflows/run_preflight.sh
bash workflows/run_stage_01_data_preparation.sh
bash workflows/run_stage_02_diagnostics.sh
bash workflows/run_stage_03_sample_construction.sh
"$ERP_TORCH_PYTHON" code/09_report_output_generation/generate_submitted_report_tables.py --project-root "$ERP_PROJECT_ROOT" --allow-partial
"$ERP_TORCH_PYTHON" code/09_report_output_generation/collect_submitted_report_figures.py --project-root "$ERP_PROJECT_ROOT" --allow-partial

echo
echo "CPU PREPARATION WORKFLOW: PASS"
echo "Stages 0-3 are complete. Formal forecasting (Stage 4) requires an NVIDIA CUDA-capable environment."
