# Software environments

The workflow uses two independent Python 3.12 environments. The forecasting environment contains PyTorch and the preprocessing dependencies. The clustering/analysis environment contains SciPy, scikit-learn and tslearn. Conda is optional: either the supplied Conda environment files or standard Python `venv` plus `pip` can be used.

The tested software versions are recorded in `tested_software_versions.txt`. The installation manifests pin versions that were retained in that validation record; dependencies without a recorded exact version remain unpinned and are checked by the package preflight.


## Storage requirement

Stages 1–3 require at least **25 GiB of free disk space** on the output filesystem. Sample construction temporarily writes six large Parquet indexes before finalisation. **30 GiB or more is recommended.** `workflows/run_cpu_reproduction.sh` and `workflows/run_full_reproduction.sh` check this before Stage 1 begins.

## Option A: Conda

Run from the repository root:

```bash
conda env create -f environment/environment_torch.yml
conda env create -f environment/environment_cluster.yml
```

The public workflow runners automatically locate the environments by their supplied names, `erp_torch` and `erp_cluster`. Activating either environment is not required when using the workflow runners.

## Option B: Python venv and pip

This route does not require Conda. It requires a Python 3.12 interpreter to be installed already. The commands below use a POSIX shell.

```bash
python3.12 -m venv .venv_torch
./.venv_torch/bin/python -m pip install --upgrade pip
./.venv_torch/bin/python -m pip install -r environment/requirements_torch.txt

python3.12 -m venv .venv_cluster
./.venv_cluster/bin/python -m pip install --upgrade pip
./.venv_cluster/bin/python -m pip install -r environment/requirements_cluster.txt
```

The public workflow runners automatically locate these repository-local virtual environments.

## Optional explicit interpreter overrides

If the environments use different names or locations, set the interpreter paths explicitly:

```bash
export ERP_PROJECT_ROOT="$(pwd)"
export ERP_TORCH_PYTHON="/path/to/python-with-pytorch"
export ERP_CLUSTER_PYTHON="/path/to/python-with-scipy-sklearn-tslearn"
```

## Verify the installation

After creating either pair of environments, run:

```bash
bash workflows/run_preflight.sh
```

The preflight resolves the two interpreters, checks required imports and runs the packaged self-tests. It is deliberately non-training and does not require a GPU. Passing it on a CPU-only machine therefore does not imply that formal forecasting can run there.

## PyTorch and CUDA

`requirements_torch.txt` and `environment_torch.yml` install PyTorch 2.12.1. The validated forecasting environment used PyTorch 2.12.1+cu130. Formal forecasting requires an NVIDIA CUDA-capable environment and `torch.cuda.is_available()` must be true. If the generic installation produces a CPU-only build on the intended forecasting machine, install the PyTorch 2.12.1 build appropriate for that machine's CUDA platform using the official PyTorch installation instructions, then rerun the package preflight.

Stage 4 checks CUDA automatically and stops with a clear error if CUDA is unavailable.
