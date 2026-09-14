# Exact reproduction workflow

Run the workflow from a clean checkout of the repository. The supplied workflow runners execute the dependency chain in the required order and stop immediately if a command fails. Stage 1 additionally verifies that each principal generated dependency exists before the next consumer runs. `reference_outputs/` is never an input.

## Preferred automated route

First prepare the raw-data directories with `bash workflows/prepare_raw_inputs.sh` (or copy an existing documented raw tree with `--source-root`). After the raw inputs pass and the two Python environments are created, the recommended CPU-side command is:

```bash
bash workflows/run_cpu_reproduction.sh
```

This performs the raw-input check, an **up-front disk-space check (minimum 25 GiB free; 30 GiB or more recommended)**, package preflight and Stages 1–3. It is suitable for a CPU-only machine and deliberately stops before formal forecasting.

On a clean NVIDIA CUDA-capable machine, the complete workflow can be launched with:

```bash
bash workflows/run_full_reproduction.sh
```

If Stage 4 forecasting has already been completed and its generated outputs are present, the remaining analysis and report-output stages can be launched with:

```bash
bash workflows/run_analysis_after_forecasting.sh
```

Individual stage runners are also supplied as `workflows/run_stage_01_*.sh` through `workflows/run_stage_13_*.sh`. The detailed commands below are retained for transparency and manual inspection.

## 0. Environment and raw data

Create the raw-data directories with `bash workflows/prepare_raw_inputs.sh`, then place the provider files exactly as described in `data_access/expected_input_structure.txt`. The package does not search the user's computer for downloads. If a documented raw tree already exists, use `bash workflows/prepare_raw_inputs.sh --source-root /path/to/raw`. Create the two Python environments using either the Conda or `venv`/`pip` route in `environment/README_environment.md`. The workflow runners automatically resolve the standard supplied environment names and locations. Explicit interpreter overrides remain available through `ERP_TORCH_PYTHON` and `ERP_CLUSTER_PYTHON`.

Check the raw inputs and run the no-training package checks:

```bash
bash workflows/check_raw_inputs.sh
bash workflows/run_preflight.sh
```

Before Stages 1–3, ensure at least **25 GiB free disk space** on the output filesystem; **30 GiB or more is recommended**. The automated CPU/full runners enforce this before Stage 1 begins.

The package preflight audits distributable source files only. It ignores provider files under `data/raw/`, generated runtime artifacts under `data/processed/` and `outputs/`, and repository-local virtual environments. Binary source documents are checked for presence by `check_raw_inputs.sh`, not parsed by the static source audit.

The forecasting stage is the computationally expensive stage and requires a CUDA-capable execution environment. The other stages are CPU analyses unless the local runtime chooses otherwise.

## 1. Raw-data preprocessing, period diagnostics and formal cohorts

```bash
$ERP_TORCH_PYTHON code/01_data_preparation/cohort_selection/build_cer_control_cohort.py --project-root "$ERP_PROJECT_ROOT"
$ERP_TORCH_PYTHON code/01_data_preparation/time_alignment/build_lcl_local_clock_map.py --project-root "$ERP_PROJECT_ROOT"
$ERP_TORCH_PYTHON code/01_data_preparation/time_alignment/build_cer_canonical_grid.py --project-root "$ERP_PROJECT_ROOT"

$ERP_TORCH_PYTHON code/01_data_preparation/period_selection/lcl_source_period_diagnostics.py
$ERP_TORCH_PYTHON code/01_data_preparation/period_selection/cer_target_period_diagnostics.py --project-root "$ERP_PROJECT_ROOT"

$ERP_TORCH_PYTHON code/01_data_preparation/eligibility/build_lcl_eligibility.py
$ERP_TORCH_PYTHON code/01_data_preparation/eligibility/build_cer_eligibility.py --project-root "$ERP_PROJECT_ROOT"
```

Principal generated dependencies include:

- `data/processed/time_aligned/lcl_full_local_clock_map.parquet`
- `data/processed/time_aligned/cer_canonical_local_grid.parquet`
- `outputs/tables/horizon_impact/lcl_p0_final_eligible_households_new.csv`
- the 929-household CER control/eligibility outputs
- Appendix A and B diagnostic tables and Figure B.1 inputs/outputs

## 2. Reported cohort diagnostics and forecast-lead diagnostic

```bash
$ERP_TORCH_PYTHON code/02_eda_and_data_diagnostics/generate_reported_eda.py --project-root "$ERP_PROJECT_ROOT"
$ERP_TORCH_PYTHON code/02_eda_and_data_diagnostics/generate_forecast_lead_diagnostic.py --project-root "$ERP_PROJECT_ROOT"
```

This stage regenerates the retained Appendix C diagnostics and Appendix D forecast-lead diagnostic using the submitted dissertation figure contract, including the reported panel structure, legends, markers, reference lines and numerical annotations.

## 3. Forecast sample construction and scalers

```bash
$ERP_TORCH_PYTHON code/03_sample_construction/sample_construction_production_indexes_scalers.py
```

This generates the lead-specific LCL/CER sample indexes, train-only household scalers and sample-retention summaries consumed by forecasting. The stage requires 25 GiB free space because six large temporary Parquet indexes are created before transaction-like finalisation.

## 4. Formal forecasting at h = 1, 12 and 48

Formal forecasting requires a CUDA-capable PyTorch environment. Confirm this before launching the schedule:

```bash
"$ERP_TORCH_PYTHON" -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); raise SystemExit(0 if torch.cuda.is_available() else 1)"
```

For each lead and seed, run the source model first, then the four target-dataset strategies. The loop below is the complete forecasting schedule:

```bash
for lead in 1 12 48; do
  runner="code/04_forecasting/run_h${lead}_formal_forecasting.py"
  config="configs/forecasting_h${lead}.yaml"

  for seed in 42 123 2026 31415; do
    $ERP_TORCH_PYTHON "$runner" \
      --project-root "$ERP_PROJECT_ROOT" \
      --config "$config" \
      --strategy lcl_source \
      --seed "$seed"

    for strategy in direct_transfer fine_tuning cer_scratch_limited cer_scratch_full; do
      $ERP_TORCH_PYTHON "$runner" \
        --project-root "$ERP_PROJECT_ROOT" \
        --config "$config" \
        --strategy "$strategy" \
        --seed "$seed"
    done
  done
done
```

The formal run outputs generated in this stage provide the native-support strategy metrics used for Table 4.1 and the prediction rows used by later controlled comparisons. Test data are evaluation-only.

## 5. Pooled cross-lead comparison and household analytical bridge

```bash
$ERP_CLUSTER_PYTHON code/08_interpretive_analysis/01_pooled_cross_lead/three_lead_common_row_comparison.py --project-root "$ERP_PROJECT_ROOT"
$ERP_CLUSTER_PYTHON code/05_metric_generation/household_analytical_dataset/src/build_household_analytical_dataset.py --project-root "$ERP_PROJECT_ROOT"
```

This stage produces the strict three-lead common-support comparison used for Table 4.2 and Appendix D.5, plus the household-level strategy metrics and gains used by later profile analyses. Table 4.1 remains the native-support pooled performance summary from the formal forecasting outputs.

## 6. Source profiles, K diagnostics, final K = 4 clustering and target assignment

```bash
$ERP_CLUSTER_PYTHON code/06_profile_clustering/profile_construction/src/build_source_training_profiles.py --project-root "$ERP_PROJECT_ROOT"

$ERP_CLUSTER_PYTHON code/06_profile_clustering/k_selection/src/profile_clustering_final_k_preflight.py \
  --config code/06_profile_clustering/k_selection/configs/profile_clustering_final_k_selection.yaml \
  --release-root "$ERP_PROJECT_ROOT"
$ERP_CLUSTER_PYTHON code/06_profile_clustering/k_selection/src/profile_clustering_final_k_analysis.py \
  --config code/06_profile_clustering/k_selection/configs/profile_clustering_final_k_selection.yaml
$ERP_CLUSTER_PYTHON code/06_profile_clustering/k_selection/src/profile_clustering_final_k_validate.py \
  --config code/06_profile_clustering/k_selection/configs/profile_clustering_final_k_selection.yaml

$ERP_CLUSTER_PYTHON code/06_profile_clustering/final_k4_clustering/src/profile_clustering_final_decision_preflight.py \
  --config code/06_profile_clustering/final_k4_clustering/configs/profile_clustering_final_decision_dissertation.json \
  --release-root "$ERP_PROJECT_ROOT"
$ERP_CLUSTER_PYTHON code/06_profile_clustering/final_k4_clustering/src/profile_clustering_final_decision_analysis.py \
  --config code/06_profile_clustering/final_k4_clustering/configs/profile_clustering_final_decision_dissertation.json
$ERP_CLUSTER_PYTHON code/06_profile_clustering/final_k4_clustering/src/profile_clustering_final_decision_validate.py \
  --config code/06_profile_clustering/final_k4_clustering/configs/profile_clustering_final_decision_dissertation.json

$ERP_CLUSTER_PYTHON code/06_profile_clustering/target_assignment/src/target_profile_assignment_cer_assignment_preflight.py \
  --config code/06_profile_clustering/target_assignment/configs/target_profile_assignment_cer_assignment.json \
  --release-root "$ERP_PROJECT_ROOT"
$ERP_CLUSTER_PYTHON code/06_profile_clustering/target_assignment/src/target_profile_assignment_cer_assignment_analysis.py \
  --config code/06_profile_clustering/target_assignment/configs/target_profile_assignment_cer_assignment.json
$ERP_CLUSTER_PYTHON code/06_profile_clustering/target_assignment/src/target_profile_assignment_cer_assignment_validate.py \
  --config code/06_profile_clustering/target_assignment/configs/target_profile_assignment_cer_assignment.json
```

The final source prototypes and the 929 target assignments generated here are fixed inputs to all profile-group interpretation. Clustering does not enter the forecasting model.

## 7. Profile-group heterogeneity

```bash
$ERP_CLUSTER_PYTHON code/08_interpretive_analysis/02_h12_profile_group/group_heterogeneity/src/profile_group_analysis_profile_group_check_inputs.py \
  --config code/08_interpretive_analysis/02_h12_profile_group/group_heterogeneity/configs/profile_group_analysis_profile_group_analysis.json
$ERP_CLUSTER_PYTHON code/08_interpretive_analysis/02_h12_profile_group/group_heterogeneity/src/profile_group_analysis_profile_group_analysis.py \
  --config code/08_interpretive_analysis/02_h12_profile_group/group_heterogeneity/configs/profile_group_analysis_profile_group_analysis.json
$ERP_CLUSTER_PYTHON code/08_interpretive_analysis/02_h12_profile_group/group_heterogeneity/src/profile_group_analysis_profile_group_validate.py \
  --config code/08_interpretive_analysis/02_h12_profile_group/group_heterogeneity/configs/profile_group_analysis_profile_group_analysis.json
```

This stage generates the base h = 12 group results and cross-lead profile-group diagnostics. The practical-significance audit is completed in Stage 8 after the raw-load structure has been generated.

## 8. Observed load composition, load-range decomposition and group practical significance

```bash
$ERP_CLUSTER_PYTHON code/08_interpretive_analysis/02_h12_profile_group/load_range/src/build_raw_load_structure.py \
  --project-root "$ERP_PROJECT_ROOT"

$ERP_CLUSTER_PYTHON code/08_interpretive_analysis/02_h12_profile_group/group_heterogeneity/src/group_heterogeneity_group_heterogeneity_practical_significance.py \
  --root "$ERP_PROJECT_ROOT"
$ERP_CLUSTER_PYTHON code/08_interpretive_analysis/02_h12_profile_group/group_heterogeneity/src/group_heterogeneity_validate_results.py \
  --root "$ERP_PROJECT_ROOT"

$ERP_CLUSTER_PYTHON code/08_interpretive_analysis/02_h12_profile_group/load_range/src/load_range_analysis_complete_cross_strategy_raw_load_composition_standardised_gain_decomposition.py \
  --root "$ERP_PROJECT_ROOT"
```

The raw-load producer first establishes the common h = 12 household load structure. It then supports the group practical-significance audit used by Table 4.3, Figure 4.1 and Appendix F, followed by the L−D and cross-strategy load-range analyses used by Figure 4.3, Figures 4.5–4.9, Figures 4.11–4.12 and Appendices I–J. RMSE standardisation is derived through additive MSE before taking the square root.

## 9. Temporal drift

```bash
$ERP_CLUSTER_PYTHON code/08_interpretive_analysis/02_h12_profile_group/temporal_drift/src/temporal_drift_features_target_span_representativeness_temporal_drift.py \
  --root "$ERP_PROJECT_ROOT"
$ERP_CLUSTER_PYTHON code/08_interpretive_analysis/02_h12_profile_group/temporal_drift/src/temporal_drift_features_validate_results.py \
  "$ERP_PROJECT_ROOT"
$ERP_CLUSTER_PYTHON code/08_interpretive_analysis/02_h12_profile_group/temporal_drift/src/temporal_drift_analysis_complete_cross_strategy_target_span_representativeness_temporal_drift.py \
  --root "$ERP_PROJECT_ROOT"
```

This stage generates the drift features and cross-strategy associations used by Figures 4.4, 4.10 and 4.13 and Appendices K–L. The analysis is descriptive and post-hoc; it does not train or select forecasting strategies.

## 10. Source similarity and local source support

```bash
$ERP_CLUSTER_PYTHON code/08_interpretive_analysis/02_h12_profile_group/source_relevance/src/build_local_source_support.py \
  --project-root "$ERP_PROJECT_ROOT"
$ERP_CLUSTER_PYTHON code/08_interpretive_analysis/02_h12_profile_group/source_relevance/src/source_side_cross_strategy_appendix_source_side_cross_strategy_diagnostics.py \
  --project-root "$ERP_PROJECT_ROOT" \
  --config code/08_interpretive_analysis/02_h12_profile_group/source_relevance/configs/source_side_cross_strategy_appendix_source_side_cross_strategy_diagnostics_config.json

$ERP_CLUSTER_PYTHON code/08_interpretive_analysis/02_h12_profile_group/group_heterogeneity/src/cross_strategy_profile_group_cross_strategy_profile_group_appendix_closure.py \
  --project-root "$ERP_PROJECT_ROOT" \
  --config code/08_interpretive_analysis/02_h12_profile_group/group_heterogeneity/configs/cross_strategy_profile_group_cross_strategy_profile_group_appendix_closure_config.json
```

The local-support producer uses generated 48-slot source/target profiles and fixed source-defined assignments. It is a post-hoc similarity measure, not a nearest-neighbour forecasting model. This stage produces Figure 4.2 and Appendices G–H inputs and completes the retained cross-strategy profile-group appendix tables.

## 11. Survey linkage, recoding and post-hoc analyses

```bash
$ERP_TORCH_PYTHON code/07_survey_data_governance/linkage/src/survey_linkage_survey_inventory_linkage.py \
  --root "$ERP_PROJECT_ROOT" \
  --config code/07_survey_data_governance/linkage/configs/survey_linkage_survey_inventory_linkage.json \
  --mode run

$ERP_TORCH_PYTHON code/07_survey_data_governance/recoding/src/survey_recoding_survey_recoding_harmonisation.py \
  --root "$ERP_PROJECT_ROOT" \
  --config code/07_survey_data_governance/recoding/configs/survey_recoding_survey_recoding_harmonisation.json \
  --run

$ERP_CLUSTER_PYTHON code/08_interpretive_analysis/03_h12_survey_posthoc/group_composition/src/survey_group_composition_survey_group_composition.py \
  --root "$ERP_PROJECT_ROOT" \
  --config code/08_interpretive_analysis/03_h12_survey_posthoc/group_composition/configs/survey_group_composition_survey_group_composition.json \
  --run

$ERP_CLUSTER_PYTHON code/08_interpretive_analysis/03_h12_survey_posthoc/group_adjusted_gain/src/survey_gain_associations_group_adjusted_survey_gain_associations.py \
  --root "$ERP_PROJECT_ROOT" \
  --config code/08_interpretive_analysis/03_h12_survey_posthoc/group_adjusted_gain/configs/survey_gain_associations_group_adjusted_survey_gain_associations_config.json \
  --run
```

These analyses generate the Appendix M tables. Survey information remains post-hoc contextual information and is not a forecasting input.

## 12. Reported figures

After the corresponding analysis tables exist, run:

```bash
# Figure 3.2 and Appendix E.1–E.2
$ERP_CLUSTER_PYTHON code/09_report_output_generation/clustering/profile_clustering_dissertation_figures.py \
  --project-root "$ERP_PROJECT_ROOT"

# Appendix E.3
$ERP_CLUSTER_PYTHON code/09_report_output_generation/clustering/target_profile_assignment_fixed_source_and_target_profiles_with_composition.py

# Figures 4.1–4.3
$ERP_CLUSTER_PYTHON code/09_report_output_generation/group_heterogeneity_group_heterogeneity_figures.py
$ERP_CLUSTER_PYTHON code/09_report_output_generation/figure_4_2_source_relevance.py --project-root "$ERP_PROJECT_ROOT"
$ERP_CLUSTER_PYTHON code/09_report_output_generation/load_range_analysis_group_load_composition_figure.py

# Figure 4.4 and Appendix K.1–K.3
$ERP_CLUSTER_PYTHON code/09_report_output_generation/plot_temporal_drift_analysis_temporal_drift_ld_gain_with_appendix_metrics.py

# Figures 4.5–4.7
$ERP_CLUSTER_PYTHON code/09_report_output_generation/load_range_analysis_appendix_j2_bar_and_cumulative_figures.py
$ERP_CLUSTER_PYTHON code/09_report_output_generation/load_range_analysis_dt_maintext_figures.py

# Figures 4.8–4.10
$ERP_CLUSTER_PYTHON code/09_report_output_generation/load_range_analysis_temporal_drift_analysis_lf_figures.py

# Figures 4.11–4.13
$ERP_CLUSTER_PYTHON code/09_report_output_generation/load_range_analysis_temporal_drift_analysis_fd_figures.py
```

All figure scripts consume generated outputs from the earlier stages. No report figure reads `reference_outputs/`.

## 13. Post-reproduction verification

```bash
$ERP_CLUSTER_PYTHON validation/validate_reproduced_results.py
```

This checks generated results against a compact set of report anchors. It is a verification step only; reference CSVs are not computational dependencies.
