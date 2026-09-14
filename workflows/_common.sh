#!/usr/bin/env bash
# Shared workflow helpers. This file is sourced by the public workflow runners.

workflow_root() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "$script_dir/.." && pwd
}

ERP_PROJECT_ROOT="${ERP_PROJECT_ROOT:-$(workflow_root)}"
export ERP_PROJECT_ROOT

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

require_file() {
  local path="$1"
  [[ -f "$path" ]] || fail "Required file is missing: $path"
  [[ -s "$path" ]] || fail "Required file is empty: $path"
}

require_dir() {
  local path="$1"
  [[ -d "$path" ]] || fail "Required directory is missing: $path"
}

resolve_conda_python() {
  local env_name="$1"
  command -v conda >/dev/null 2>&1 || return 1
  local candidate
  candidate="$(conda run -n "$env_name" python -c 'import sys; print(sys.executable)' 2>/dev/null | tail -n 1)" || return 1
  [[ -n "$candidate" && -x "$candidate" ]] || return 1
  printf '%s\n' "$candidate"
}

resolve_torch_python() {
  if [[ -n "${ERP_TORCH_PYTHON:-}" ]]; then
    [[ -x "$ERP_TORCH_PYTHON" ]] || fail "ERP_TORCH_PYTHON is not executable: $ERP_TORCH_PYTHON"
    return 0
  fi
  if [[ -x "$ERP_PROJECT_ROOT/.venv_torch/bin/python" ]]; then
    ERP_TORCH_PYTHON="$ERP_PROJECT_ROOT/.venv_torch/bin/python"
  else
    ERP_TORCH_PYTHON="$(resolve_conda_python erp_torch || true)"
  fi
  [[ -n "${ERP_TORCH_PYTHON:-}" && -x "$ERP_TORCH_PYTHON" ]] || fail \
    "Could not locate the TORCH Python environment. Create it as documented in environment/README_environment.md or set ERP_TORCH_PYTHON."
  export ERP_TORCH_PYTHON
}

resolve_cluster_python() {
  if [[ -n "${ERP_CLUSTER_PYTHON:-}" ]]; then
    [[ -x "$ERP_CLUSTER_PYTHON" ]] || fail "ERP_CLUSTER_PYTHON is not executable: $ERP_CLUSTER_PYTHON"
    return 0
  fi
  if [[ -x "$ERP_PROJECT_ROOT/.venv_cluster/bin/python" ]]; then
    ERP_CLUSTER_PYTHON="$ERP_PROJECT_ROOT/.venv_cluster/bin/python"
  else
    ERP_CLUSTER_PYTHON="$(resolve_conda_python erp_cluster || true)"
  fi
  [[ -n "${ERP_CLUSTER_PYTHON:-}" && -x "$ERP_CLUSTER_PYTHON" ]] || fail \
    "Could not locate the CLUSTER Python environment. Create it as documented in environment/README_environment.md or set ERP_CLUSTER_PYTHON."
  export ERP_CLUSTER_PYTHON
}

resolve_all_pythons() {
  resolve_torch_python
  resolve_cluster_python
}

check_python_312() {
  local python="$1"
  local label="$2"
  local version
  version="$($python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  [[ "$version" == "3.12" ]] || fail "$label environment must use Python 3.12; found Python $version at $python"
}

run_step() {
  local label="$1"
  shift
  echo
  echo "=============================================================================="
  echo "$label"
  echo "=============================================================================="
  "$@"
}

check_disk_space_25gib() {
  local path="${1:-$ERP_PROJECT_ROOT}"
  local avail_kib
  avail_kib="$(df -Pk "$path" | awk 'NR==2 {print $4}')"
  [[ "$avail_kib" =~ ^[0-9]+$ ]] || fail "Could not determine free disk space for: $path"
  local required_kib=$((25 * 1024 * 1024))
  local avail_gib
  avail_gib="$(awk -v k="$avail_kib" 'BEGIN {printf "%.2f", k/1024/1024}')"
  echo "Disk space check : ${avail_gib} GiB free"
  echo "Minimum required : 25.00 GiB"
  echo "Recommended      : 30 GiB or more"
  if (( avail_kib < required_kib )); then
    fail "At least 25 GiB free disk space is required before Stages 1-3 begin. Free additional space or use a larger filesystem."
  fi
  echo "DISK SPACE CHECK: PASS"
}

check_cuda() {
  resolve_torch_python
  "$ERP_TORCH_PYTHON" -c 'import torch; print("torch:", torch.__version__); print("cuda available:", torch.cuda.is_available()); raise SystemExit(0 if torch.cuda.is_available() else 1)' \
    || fail "Formal forecasting requires an NVIDIA CUDA-capable PyTorch environment."
}
