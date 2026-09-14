# Cross-region residential load forecasting — Additional Materials

This repository is the code-first reproducibility package for the submitted dissertation. Starting from lawfully obtained LCL, CER and survey source files, the workflow regenerates the processed data, forecasting outputs, source-defined profile groups, post-hoc analyses and the reported tables and figures retained in the dissertation.

## Before you start

The CPU preparation workflow (Stages 1–3) requires **at least 25 GiB of free disk space on the output filesystem** because sample construction creates six large temporary Parquet indexes before finalisation. **30 GiB or more is recommended.** The workflow checks this requirement before Stage 1 begins.

Formal forecasting in Stage 4 requires an **NVIDIA CUDA-capable PyTorch environment**. CPU-only systems and Apple Silicon systems can reproduce Stages 1–3 but cannot execute the formal Stage 4 forecasting protocol.

Two Python 3.12 environments are used:

- `erp_torch` for data preparation, sample construction and forecasting;
- `erp_cluster` for clustering, statistical analysis and report-output generation.

Both Conda and standard `venv`/`pip` installation routes are supplied in `environment/README_environment.md`.

## Start here

1. Read `data_access/README_data_access.md`.
2. Run `bash workflows/prepare_raw_inputs.sh` to create the required raw-data directories. The package never searches the user's computer for source files. If an existing raw-data tree already matches the documented structure, it can be copied explicitly with `bash workflows/prepare_raw_inputs.sh --source-root /path/to/raw`.
3. Place the 13 required provider files exactly as shown in `data_access/expected_input_structure.txt`, then run `bash workflows/check_raw_inputs.sh`.
4. Create the two Python 3.12 environments described in `environment/README_environment.md`.
5. On a CPU machine, run `bash workflows/run_cpu_reproduction.sh` to perform the raw-data input check, disk-space check, package preflight and Stages 1–3 in the required order.
6. On a CUDA-capable machine, run `bash workflows/run_full_reproduction.sh` for the complete clean reproduction through Stage 13.
7. Use `reference_outputs/` only as optional human-facing verification after reproduction. No executable code depends on it.

The workflow runners automatically locate the supplied Conda environments (`erp_torch`, `erp_cluster`) or repository-local virtual environments (`.venv_torch`, `.venv_cluster`). `ERP_TORCH_PYTHON` and `ERP_CLUSTER_PYTHON` remain available as explicit overrides.

## Report-figure fidelity

The submitted dissertation is the authority for reported figure content. Report-output producers regenerate the reported figure structure, labels, legends, reference lines, numerical annotations and panel organisation from reproduced results. This includes the eligibility diagnostic in Figure B.1, the cohort diagnostics in Figures C.1–C.2, the lead diagnostic in Figure D.1, the design schematic in Figure 3.1, and the main-text Figures 3.2 and 4.1–4.13. See `documentation/current_report_figure_contract.md`.

## Scope

The public package covers the submitted main text and Appendices A–M only. It intentionally excludes development history, unreported analyses, raw restricted datasets and private validation records.

The fixed analytical governance values used by package checks are summarised in `documentation/reproducibility_design_contract.md`; executable configuration files remain the authority for exact runtime values.

## Validation

`workflows/run_preflight.sh` checks public-package structure, portability, environment routing and packaged self-tests without training forecasting models or reading provider raw datasets. The static source audit explicitly excludes user-supplied `data/raw/`, generated `data/processed/` and `outputs/`, and repository-local virtual environments. It is an installation check, not a substitute for the full raw-data workflow.

After reproduction, `validation/validate_reproduced_results.py` checks regenerated numerical results against embedded report anchors. Stage 12 additionally checks that all reported figure files have been generated.

## Dissertation-facing outputs

A complete run regenerates every retained table and figure reported in the submitted dissertation. Helper analytical outputs remain under `outputs/tables/`; stable dissertation-numbered tables are collected under `outputs/report_tables/`. Stage 12 validates the full 66-item report inventory (Methodology, Results and Appendices A-M). See `documentation/submitted_report_artifact_contract.md`.

Privacy and publication checks are defined in `documentation/privacy_and_publication_contract.md`. `workflows/run_preflight.sh` runs the public privacy validator as part of the release gate.
