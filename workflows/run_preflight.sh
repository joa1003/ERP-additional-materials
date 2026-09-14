#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
resolve_all_pythons
check_python_312 "$ERP_TORCH_PYTHON" "TORCH"
check_python_312 "$ERP_CLUSTER_PYTHON" "CLUSTER"
cd "$ERP_PROJECT_ROOT"

"$ERP_CLUSTER_PYTHON" "$ERP_PROJECT_ROOT/validation/validate_static.py"
"$ERP_CLUSTER_PYTHON" "$ERP_PROJECT_ROOT/validation/validate_public_privacy.py"
"$ERP_CLUSTER_PYTHON" "$ERP_PROJECT_ROOT/validation/validate_portability_placeholders.py" --repo-root "$ERP_PROJECT_ROOT"
"$ERP_CLUSTER_PYTHON" "$ERP_PROJECT_ROOT/validation/validate_environment_routing.py" \
  --repo-root "$ERP_PROJECT_ROOT" \
  --torch-python "$ERP_TORCH_PYTHON" \
  --cluster-python "$ERP_CLUSTER_PYTHON"
"$ERP_CLUSTER_PYTHON" "$ERP_PROJECT_ROOT/validation/run_packaged_selftests.py" \
  --repo-root "$ERP_PROJECT_ROOT" \
  --torch-python "$ERP_TORCH_PYTHON" \
  --cluster-python "$ERP_CLUSTER_PYTHON"

echo "PUBLIC REPRODUCIBILITY PACKAGE PREFLIGHT: PASS"
echo "Training: NO"
