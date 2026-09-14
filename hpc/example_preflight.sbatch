#!/usr/bin/env bash
#SBATCH --job-name=erp_repro_preflight
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --output=preflight_%j.out
#SBATCH --error=preflight_%j.err

set -euo pipefail

PACKAGE_ROOT="${1:-$PWD}"

: "${ERP_TORCH_PYTHON:?Set ERP_TORCH_PYTHON to the forecasting environment interpreter.}"
: "${ERP_CLUSTER_PYTHON:?Set ERP_CLUSTER_PYTHON to the clustering/statistical environment interpreter.}"

cd "${PACKAGE_ROOT}"
bash workflows/run_preflight.sh
