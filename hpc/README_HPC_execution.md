# GPU and HPC execution

The package is not tied to a specific institutional cluster or scheduler. Formal forecasting can be run on any suitable execution environment that provides an NVIDIA CUDA-capable GPU, sufficient memory and storage, and the documented Python environments.

The validated forecasting environment used PyTorch 2.12.1+cu130. `workflows/run_stage_04_forecasting.sh` and `workflows/run_full_reproduction.sh` check `torch.cuda.is_available()` before training and stop with a clear error if CUDA is unavailable.

## Storage

Stages 1–3 require at least 25 GiB free disk space; 30 GiB or more is recommended. On shared HPC systems, use the site's high-capacity job/scratch filesystem rather than a small home directory.

## Recommended scheduler split

For schedulers such as Slurm, the workflow can be separated cleanly without changing any research code:

1. CPU job: `bash workflows/run_cpu_reproduction.sh` — raw-data check, preflight and Stages 1–3.
2. GPU execution: either `bash workflows/run_stage_04_forecasting.sh` for the complete sequential schedule, or submit the 12 independent lead/seed bundles with `bash workflows/run_stage_04_one_seed.sh <lead> <seed>`. Each bundle runs the source model first and then the four target strategies for that lead/seed. This is the recommended scheduler route because one failed or timed-out bundle does not force the other eleven bundles to be repeated.
3. CPU job: `bash workflows/run_analysis_after_forecasting.sh` — Stages 5–13, including clustering, post-hoc analyses, submitted dissertation figures and final result validation.

All three jobs must use the same repository/project root so that generated outputs remain available to the next stage.

For a clean CUDA-capable machine where running all stages in one process is appropriate, use:

```bash
bash workflows/run_full_reproduction.sh
```

For CPU-only preparation before moving the same project root to a CUDA-capable machine, use:

```bash
bash workflows/run_cpu_reproduction.sh
```

A generic no-training scheduler example is provided in `example_preflight.sbatch`. Site-specific partition, wall-clock, CPU, memory and GPU settings must follow the local scheduler policy. The package itself does not assume Manchester CSF or any other specific cluster.


## Scheduler-friendly Stage 4 split

The formal schedule contains 12 independent lead/seed bundles:

- leads: `1`, `12`, `48`;
- seeds: `42`, `123`, `2026`, `31415`.

Each bundle contains five runner invocations in dependency order: `lcl_source`, `direct_transfer`, `fine_tuning`, `cer_scratch_limited`, and `cer_scratch_full`. A scheduler may therefore submit the 12 bundles as separate GPU jobs or as a 12-element job array. The public helper is:

```bash
bash workflows/run_stage_04_one_seed.sh 12 42
```

Do not run two jobs for the same lead/seed pair against the same project root. Formal forecasting intentionally refuses to overwrite existing run outputs. If a job is interrupted mid-strategy, inspect that run before using the forecasting runner's documented `--resume` facility; do not delete or overwrite formal outputs blindly.
