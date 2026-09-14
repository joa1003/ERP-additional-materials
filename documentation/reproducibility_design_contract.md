# Reproducibility design contract

This public contract records the fixed analytical design required by the executable reproduction workflow. It replaces private development notes as the file-level governance anchor used by package checks.

## Forecasting design

- Source dataset: Low Carbon London residential half-hourly electricity data for 2013.
- Target dataset: Commission for Energy Regulation residential control-household data for 2009–2010.
- Forecasting unit: household level.
- Model: one global CNN-LSTM architecture shared across households.
- Input window: previous 48 half-hourly load observations plus calendar features.
- Formal forecast leads: h = 1, 12 and 48.
- Formal random seeds: 42, 123, 2026 and 31415.
- Strategies: Direct Transfer, Fine Tuning, CER Scratch Limited and CER Scratch Full.
- Scaling: per household, training data only, population standard deviation with a 0.01 kWh denominator floor.
- Weather is not used as a forecasting input.

## Eligibility and sample construction

- R1: coverage at least 90 percent.
- R2: non-constant full-period load after the documented measurement-precision treatment.
- R3: at least 100 informative training windows.
- R4: missing-window influence controlled by the documented threshold.
- Final source cohort: 3,843 households.
- Final target cohort: 929 households.
- Forecast timestamp splits and target-training budgets are defined in the public forecasting configuration files.

## Profile interpretation

- Source profiles are fitted with time-series K-means using DTW.
- Final governed source profile count: K = 4.
- Final source prototype seed: 71.
- Target households are assigned to the nearest fixed source prototype.
- Target assignment does not refit source prototypes and does not use forecasting outcomes.

## Post-hoc interpretation

- Household profile groups, load composition, temporal drift, source similarity and survey information are interpretive analyses.
- Survey information is not a forecasting input.
- Survey analyses are descriptive and non-causal.
- The primary detailed interpretation lead is h = 12, with formal cross-lead checks at h = 1 and h = 48.

## Report scope

The reproducibility package covers the submitted dissertation main text and Appendices A–M. Appendix N and Table L.2 are outside the retained report scope.

The executable configuration files remain the operational authority for exact paths, date boundaries, thresholds and output names.
