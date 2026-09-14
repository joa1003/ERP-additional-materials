#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
resolve_cluster_python
cd "$ERP_PROJECT_ROOT"
run_step "STAGE 13 - Post-reproduction validation" "$ERP_CLUSTER_PYTHON" validation/validate_reproduced_results.py
echo; echo "STAGE 13 RESULT VALIDATION: PASS"
