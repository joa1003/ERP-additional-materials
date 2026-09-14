from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import os as _erp_os
_ERP_PROJECT_ROOT = Path(_erp_os.environ.get("ERP_PROJECT_ROOT", ".")).expanduser().resolve()
import gc
import shutil

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import yaml


# ============================================================
# Sample construction — Production sample indexes and scaler tables
# Status: formal, full population
#
# Full scope:
#   - LCL: 3,843 final R1-R4 households
#   - CER: 929 formal control meters
#   - Horizons: h = {1, 12, 48}
#
# Architecture:
#   - LCL CSV read in row chunks into a float64 matrix.
#   - CER parquet scanned in record batches and scattered into a
#     preallocated float64 matrix; no 23.9M-row pandas table is built.
#   - Six horizon-specific sample indexes are written block-by-block
#     to temporary Parquet files.
#   - No complete dense X/y tensors are stored.
#   - Entity-level and aggregate count summaries prove:
#       candidate - missing - continuity = valid
#   - All temporary outputs are checked before transaction-like
#     finalization; metadata is renamed last as the completion marker.
# ============================================================


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------
PROJECT_ROOT = Path(str(_ERP_PROJECT_ROOT))

EXPECTED_RELATIVE_PATHS = {
    "lcl_raw": (
        "data/raw/LCL/csv/data_collection/"
        "data_tables/consumption_n.csv"
    ),
    "lcl_time_map": (
        "data/processed/time_aligned/"
        "lcl_full_local_clock_map.parquet"
    ),
    "lcl_final_cohort": (
        "outputs/tables/horizon_impact/"
        "lcl_p0_final_eligible_households_new.csv"
    ),
    "cer_grid": (
        "data/processed/time_aligned/"
        "cer_canonical_local_grid.parquet"
    ),
}

PATHS = {
    "runtime_yaml": PROJECT_ROOT / "configs" / "forecasting_runtime.yaml",
    **{
        name: PROJECT_ROOT / relative_path
        for name, relative_path in EXPECTED_RELATIVE_PATHS.items()
    },
}

for name, path in PATHS.items():
    if not path.exists():
        raise FileNotFoundError(
            f"Required input was not found:\n{name}: {path}"
        )


# ------------------------------------------------------------
# Output paths
# ------------------------------------------------------------
OUT_TABLES = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "sample_construction"
)
OUT_META = (
    PROJECT_ROOT
    / "outputs"
    / "metadata"
    / "sample_construction"
)

for directory in (OUT_TABLES, OUT_META):
    directory.mkdir(parents=True, exist_ok=True)

INDEX_FILES = {
    ("LCL", 1): OUT_TABLES / "lcl_production_sample_index_h1.parquet",
    ("LCL", 12): OUT_TABLES / "lcl_production_sample_index_h12.parquet",
    ("LCL", 48): OUT_TABLES / "lcl_production_sample_index_h48.parquet",
    ("CER", 1): OUT_TABLES / "cer_production_sample_index_h1.parquet",
    ("CER", 12): OUT_TABLES / "cer_production_sample_index_h12.parquet",
    ("CER", 48): OUT_TABLES / "cer_production_sample_index_h48.parquet",
}

OUT_LCL_SCALER = OUT_TABLES / "lcl_production_scaler_table.csv"
OUT_CER_SCALER = OUT_TABLES / "cer_production_scaler_table.csv"
OUT_ENTITY_SUMMARY = (
    OUT_TABLES / "sample_construction_production_sample_count_by_entity.csv"
)
OUT_AGGREGATE_SUMMARY = (
    OUT_TABLES / "sample_construction_production_sample_count_summary.csv"
)
OUT_META_FILE = (
    OUT_META / "sample_construction_production_indexes_scalers_confirmation.md"
)

ALL_FINAL_OUTPUTS = (
    list(INDEX_FILES.values())
    + [
        OUT_LCL_SCALER,
        OUT_CER_SCALER,
        OUT_ENTITY_SUMMARY,
        OUT_AGGREGATE_SUMMARY,
        OUT_META_FILE,
    ]
)

for path in ALL_FINAL_OUTPUTS:
    if path.exists():
        raise FileExistsError(
            "Sample construction stopped because a required  output already "
            f"exists and must not be overwritten:\n{path}"
        )

TMP_INDEX_FILES = {
    key: final_path.with_suffix(".tmp.parquet")
    for key, final_path in INDEX_FILES.items()
}
TMP_LCL_SCALER = OUT_LCL_SCALER.with_suffix(".tmp.csv")
TMP_CER_SCALER = OUT_CER_SCALER.with_suffix(".tmp.csv")
TMP_ENTITY_SUMMARY = OUT_ENTITY_SUMMARY.with_suffix(".tmp.csv")
TMP_AGGREGATE_SUMMARY = OUT_AGGREGATE_SUMMARY.with_suffix(".tmp.csv")
TMP_META_FILE = OUT_META_FILE.with_suffix(".tmp.md")

ALL_TEMP_OUTPUTS = (
    list(TMP_INDEX_FILES.values())
    + [
        TMP_LCL_SCALER,
        TMP_CER_SCALER,
        TMP_ENTITY_SUMMARY,
        TMP_AGGREGATE_SUMMARY,
        TMP_META_FILE,
    ]
)

# Remove stale temporary files only. Formal outputs are never overwritten.
for path in ALL_TEMP_OUTPUTS:
    if path.exists():
        path.unlink()


# ------------------------------------------------------------
# Locked constants
# ------------------------------------------------------------
EXPECTED_RUNTIME_ROLE = "reproduction_runtime_parameters"
LOOKBACK = 48
HORIZONS = [1, 12, 48]
STD_DDOF = 0
STD_FLOOR = 0.01
SLOTS_PER_DAY = 48
RESOLUTION = pd.Timedelta("30min")

FEATURE_ORDER = [
    "scaled_load",
    "time_of_day_sin",
    "time_of_day_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "weekend_indicator",
    "month_sin",
    "month_cos",
]

LCL_P0_START = pd.Timestamp("2013-01-01 00:00:00")
LCL_P0_END = pd.Timestamp("2013-12-31 23:30:00")
LCL_SPLITS = {
    "train": (
        pd.Timestamp("2013-01-02 00:00:00"),
        pd.Timestamp("2013-09-13 18:30:00"),
    ),
    "validation": (
        pd.Timestamp("2013-09-13 19:00:00"),
        pd.Timestamp("2013-10-20 04:00:00"),
    ),
    "test": (
        pd.Timestamp("2013-10-20 04:30:00"),
        pd.Timestamp("2013-12-31 23:30:00"),
    ),
}
LCL_NATIVE_CANDIDATES = {
    1: {"train": 12_230, "validation": 1_747, "test": 3_495},
    12: {"train": 12_219, "validation": 1_747, "test": 3_495},
    48: {"train": 12_183, "validation": 1_747, "test": 3_495},
}
EXPECTED_LCL_RAW = 4_173
EXPECTED_LCL_FINAL = 3_843
EXPECTED_LCL_POSITIONS = 17_520

CER_P0_START = pd.Timestamp("2009-07-14 00:00:00")
CER_P0_END = pd.Timestamp("2010-12-31 23:30:00")
CER_SPLITS = {
    "train": (
        pd.Timestamp("2009-07-15 00:00:00"),
        pd.Timestamp("2010-07-24 11:30:00"),
    ),
    "validation": (
        pd.Timestamp("2010-07-24 12:00:00"),
        pd.Timestamp("2010-09-15 23:30:00"),
    ),
    "test": (
        pd.Timestamp("2010-09-16 00:00:00"),
        pd.Timestamp("2010-12-31 23:30:00"),
    ),
}
CER_NATIVE_CANDIDATES = {
    1: {"train": 17_976, "validation": 2_568, "test": 5_136},
    12: {"train": 17_965, "validation": 2_568, "test": 5_136},
    48: {"train": 17_929, "validation": 2_568, "test": 5_136},
}
EXPECTED_CER_METERS = 929
EXPECTED_CER_DAYS = 536
EXPECTED_CER_POSITIONS_PER_METER = EXPECTED_CER_DAYS * SLOTS_PER_DAY
EXPECTED_CER_TOTAL_ROWS = (
    EXPECTED_CER_METERS * EXPECTED_CER_POSITIONS_PER_METER
)

CER_30DAY_SCALER_START = pd.Timestamp("2009-07-15 00:00:00")
CER_30DAY_SCALER_END = pd.Timestamp("2009-08-13 23:30:00")
CER_30DAY_EXPECTED_CANDIDATES = 1_440
CER_FULL_TRAIN_EXPECTED_CANDIDATES = 17_976

CER_COMPRESSION_BOUNDARY_DATES = [
    pd.Timestamp("2009-10-25").date(),
    pd.Timestamp("2010-10-31").date(),
]

NEGATIVE_TEST_IDS = {
    "N2035": "R3_fail",
    "N2041": "R4_fail",
}

# Execution controls. These do not change the formal sample definition.
LCL_CSV_CHUNK_ROWS = 750
LCL_ENTITY_BLOCK_SIZE = 64
CER_ENTITY_BLOCK_SIZE = 32
CER_LOAD_BATCH_SIZE = 500_000
PARQUET_WRITE_CHUNK_ROWS = 250_000
PARQUET_COMPRESSION = "zstd"
PARQUET_COMPRESSION_LEVEL = 3
REPRESENTATIVE_EXAMPLES_PER_STRATUM = 5
MIN_FREE_DISK_GIB = 25.0

INDEX_SCHEMA = pa.schema(
    [
        ("dataset", pa.string()),
        ("entity_id", pa.string()),
        ("horizon", pa.int16()),
        ("target_timestamp", pa.timestamp("ns")),
        ("split", pa.string()),
    ]
)


print("=" * 78)
print("SAMPLE CONSTRUCTION — PRODUCTION SAMPLE INDEXES AND SCALER TABLES")
print("Status: formal, full population, streaming architecture")
print("=" * 78)


# ============================================================
# STEP 0a — Runtime YAML consistency
# ============================================================
print("\n" + "=" * 78)
print("STEP 0a — RUNTIME YAML CONSISTENCY CHECK")
print("=" * 78)

with PATHS["runtime_yaml"].open("r", encoding="utf-8") as file:
    runtime = yaml.safe_load(file)

if not isinstance(runtime, dict):
    raise TypeError(
        "forecasting_runtime.yaml did not load as a mapping."
    )

runtime_issues: list[str] = []


def nested_get(mapping, *keys):
    current = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def check_runtime(label, actual, expected):
    if actual != expected:
        runtime_issues.append(
            f"{label}: YAML has {actual!r}; expected {expected!r}"
        )
    else:
        print(f"  PASS  {label}: {actual!r}")


def check_runtime_timestamp(label, actual, expected):
    try:
        actual_timestamp = pd.Timestamp(actual)
    except Exception:
        runtime_issues.append(
            f"{label}: YAML value {actual!r} cannot be parsed as a timestamp"
        )
        return

    if actual_timestamp != expected:
        runtime_issues.append(
            f"{label}: YAML has {actual_timestamp}; expected {expected}"
        )
    else:
        print(f"  PASS  {label}: {actual_timestamp}")


def check_runtime_date(label, actual, expected_date):
    try:
        actual_date = pd.Timestamp(actual).date()
    except Exception:
        runtime_issues.append(
            f"{label}: YAML value {actual!r} cannot be parsed as a date"
        )
        return

    if actual_date != expected_date:
        runtime_issues.append(
            f"{label}: YAML has {actual_date}; expected {expected_date}"
        )
    else:
        print(f"  PASS  {label}: {actual_date}")


# Governance role and canonical paths.
check_runtime(
    "runtime_projection.role",
    nested_get(runtime, "runtime_projection", "role"),
    EXPECTED_RUNTIME_ROLE,
)
check_runtime(
    "canonical_paths.lcl_load_values",
    nested_get(runtime, "canonical_paths", "lcl_load_values"),
    EXPECTED_RELATIVE_PATHS["lcl_raw"],
)
check_runtime(
    "canonical_paths.lcl_time_and_calendar",
    nested_get(runtime, "canonical_paths", "lcl_time_and_calendar"),
    EXPECTED_RELATIVE_PATHS["lcl_time_map"],
)
check_runtime(
    "canonical_paths.lcl_final_cohort",
    nested_get(runtime, "canonical_paths", "lcl_final_cohort"),
    EXPECTED_RELATIVE_PATHS["lcl_final_cohort"],
)
check_runtime(
    "canonical_paths.cer_standard_grid",
    nested_get(runtime, "canonical_paths", "cer_standard_grid"),
    EXPECTED_RELATIVE_PATHS["cer_grid"],
)

# Dataset counts and periods.
check_runtime(
    "datasets.lcl.raw_household_count",
    nested_get(runtime, "datasets", "lcl", "raw_household_count"),
    EXPECTED_LCL_RAW,
)
check_runtime(
    "datasets.lcl.final_household_count",
    nested_get(runtime, "datasets", "lcl", "final_household_count"),
    EXPECTED_LCL_FINAL,
)
check_runtime(
    "datasets.cer.meter_count",
    nested_get(runtime, "datasets", "cer", "meter_count"),
    EXPECTED_CER_METERS,
)

check_runtime_timestamp(
    "datasets.lcl.full_data_period.start",
    nested_get(runtime, "datasets", "lcl", "full_data_period", "start"),
    LCL_P0_START,
)
check_runtime_timestamp(
    "datasets.lcl.full_data_period.end",
    nested_get(runtime, "datasets", "lcl", "full_data_period", "end"),
    LCL_P0_END,
)
check_runtime(
    "datasets.lcl.full_data_period.canonical_positions",
    nested_get(
        runtime,
        "datasets",
        "lcl",
        "full_data_period",
        "canonical_positions",
    ),
    EXPECTED_LCL_POSITIONS,
)
check_runtime_date(
    "datasets.lcl.support_only_date",
    nested_get(runtime, "datasets", "lcl", "support_only_date"),
    LCL_P0_START.date(),
)
check_runtime_timestamp(
    "datasets.lcl.forecasting_target_period.start",
    nested_get(
        runtime,
        "datasets",
        "lcl",
        "forecasting_target_period",
        "start",
    ),
    LCL_SPLITS["train"][0],
)
check_runtime_timestamp(
    "datasets.lcl.forecasting_target_period.end",
    nested_get(
        runtime,
        "datasets",
        "lcl",
        "forecasting_target_period",
        "end",
    ),
    LCL_P0_END,
)

check_runtime_timestamp(
    "datasets.cer.full_data_period.start",
    nested_get(runtime, "datasets", "cer", "full_data_period", "start"),
    CER_P0_START,
)
check_runtime_timestamp(
    "datasets.cer.full_data_period.end",
    nested_get(runtime, "datasets", "cer", "full_data_period", "end"),
    CER_P0_END,
)
check_runtime(
    "datasets.cer.full_data_period.canonical_positions",
    nested_get(
        runtime,
        "datasets",
        "cer",
        "full_data_period",
        "canonical_positions",
    ),
    EXPECTED_CER_TOTAL_ROWS,
)
check_runtime_date(
    "datasets.cer.support_only_date",
    nested_get(runtime, "datasets", "cer", "support_only_date"),
    CER_P0_START.date(),
)
check_runtime_timestamp(
    "datasets.cer.forecasting_target_period.start",
    nested_get(
        runtime,
        "datasets",
        "cer",
        "forecasting_target_period",
        "start",
    ),
    CER_SPLITS["train"][0],
)
check_runtime_timestamp(
    "datasets.cer.forecasting_target_period.end",
    nested_get(
        runtime,
        "datasets",
        "cer",
        "forecasting_target_period",
        "end",
    ),
    CER_P0_END,
)

# Forecasting task.
check_runtime(
    "forecasting_task.task_type",
    nested_get(runtime, "forecasting_task", "task_type"),
    "independent_single_target",
)
check_runtime(
    "forecasting_task.formal_horizons",
    nested_get(runtime, "forecasting_task", "formal_horizons"),
    HORIZONS,
)
check_runtime(
    "forecasting_task.input_length",
    nested_get(runtime, "forecasting_task", "input_length"),
    LOOKBACK,
)
check_runtime(
    "forecasting_task.output_length",
    nested_get(runtime, "forecasting_task", "output_length"),
    1,
)
check_runtime(
    "forecasting_task.input_feature_count",
    nested_get(runtime, "forecasting_task", "input_feature_count"),
    len(FEATURE_ORDER),
)
check_runtime(
    "forecasting_task.input_shape",
    nested_get(runtime, "forecasting_task", "input_shape"),
    [LOOKBACK, len(FEATURE_ORDER)],
)
check_runtime(
    "forecasting_task.output_shape",
    nested_get(runtime, "forecasting_task", "output_shape"),
    [1],
)
check_runtime(
    "forecasting_task.split_assignment",
    nested_get(runtime, "forecasting_task", "split_assignment"),
    "final_target_timestamp",
)
check_runtime(
    "features.included",
    nested_get(runtime, "features", "included"),
    FEATURE_ORDER,
)

# Sample construction rules.
check_runtime(
    "sample_construction.native_support_by_horizon",
    nested_get(
        runtime,
        "sample_construction",
        "native_support_by_horizon",
    ),
    True,
)
check_runtime(
    "sample_construction.common_target_support_formal",
    nested_get(
        runtime,
        "sample_construction",
        "common_target_support_formal",
    ),
    False,
)
check_runtime(
    "sample_construction.physical_continuity_required",
    nested_get(
        runtime,
        "sample_construction",
        "physical_continuity_required",
    ),
    True,
)
check_runtime(
    "sample_construction.required_observed_positions.inputs",
    nested_get(
        runtime,
        "sample_construction",
        "required_observed_positions",
        "inputs",
    ),
    LOOKBACK,
)
check_runtime(
    "sample_construction.required_observed_positions.target",
    nested_get(
        runtime,
        "sample_construction",
        "required_observed_positions",
        "target",
    ),
    1,
)
check_runtime(
    "sample_construction.intermediate_positions_required_for_higher_horizons",
    nested_get(
        runtime,
        "sample_construction",
        "intermediate_positions_required_for_higher_horizons",
    ),
    False,
)
check_runtime(
    "sample_construction.input_imputation",
    nested_get(runtime, "sample_construction", "input_imputation"),
    False,
)
check_runtime(
    "sample_construction.target_imputation",
    nested_get(runtime, "sample_construction", "target_imputation"),
    False,
)
check_runtime(
    "sample_construction.observed_zero_is_valid",
    nested_get(runtime, "sample_construction", "observed_zero_is_valid"),
    True,
)
check_runtime(
    "sample_construction.missing_window_action",
    nested_get(runtime, "sample_construction", "missing_window_action"),
    "skip_affected_sample_only",
)
check_runtime(
    "sample_construction.full_dense_X_y_saved_to_disk",
    nested_get(
        runtime,
        "sample_construction",
        "full_dense_X_y_saved_to_disk",
    ),
    False,
)

# Scaling rules.
check_runtime(
    "scaling.method",
    nested_get(runtime, "scaling", "method"),
    "per_entity_zscore",
)
check_runtime(
    "scaling.standard_deviation_ddof",
    nested_get(runtime, "scaling", "standard_deviation_ddof"),
    STD_DDOF,
)
check_runtime(
    "scaling.standard_deviation_floor_kwh",
    nested_get(runtime, "scaling", "standard_deviation_floor_kwh"),
    STD_FLOOR,
)
check_runtime(
    "scaling.exclude_missing_from_fit",
    nested_get(runtime, "scaling", "exclude_missing_from_fit"),
    True,
)
check_runtime(
    "scaling.validation_used_for_fit",
    nested_get(runtime, "scaling", "validation_used_for_fit"),
    False,
)
check_runtime(
    "scaling.test_used_for_fit",
    nested_get(runtime, "scaling", "test_used_for_fit"),
    False,
)
check_runtime(
    "scaling.one_scaler_shared_across_horizons_per_entity",
    nested_get(
        runtime,
        "scaling",
        "one_scaler_shared_across_horizons_per_entity",
    ),
    True,
)

check_runtime_timestamp(
    "scaling.lcl.fit_start",
    nested_get(runtime, "scaling", "lcl", "fit_start"),
    LCL_SPLITS["train"][0],
)
check_runtime_timestamp(
    "scaling.lcl.fit_end",
    nested_get(runtime, "scaling", "lcl", "fit_end"),
    LCL_SPLITS["train"][1],
)

for strategy in (
    "direct_transfer",
    "fine_tuning",
    "target_scratch_limited",
):
    check_runtime_timestamp(
        f"scaling.cer.{strategy}.fit_start",
        nested_get(runtime, "scaling", "cer", strategy, "fit_start"),
        CER_30DAY_SCALER_START,
    )
    check_runtime_timestamp(
        f"scaling.cer.{strategy}.fit_end",
        nested_get(runtime, "scaling", "cer", strategy, "fit_end"),
        CER_30DAY_SCALER_END,
    )
    check_runtime(
        f"scaling.cer.{strategy}.candidate_timestamps",
        nested_get(
            runtime,
            "scaling",
            "cer",
            strategy,
            "candidate_timestamps",
        ),
        CER_30DAY_EXPECTED_CANDIDATES,
    )

check_runtime_timestamp(
    "scaling.cer.target_scratch_full.fit_start",
    nested_get(
        runtime,
        "scaling",
        "cer",
        "target_scratch_full",
        "fit_start",
    ),
    CER_SPLITS["train"][0],
)
check_runtime_timestamp(
    "scaling.cer.target_scratch_full.fit_end",
    nested_get(
        runtime,
        "scaling",
        "cer",
        "target_scratch_full",
        "fit_end",
    ),
    CER_SPLITS["train"][1],
)
check_runtime(
    "scaling.cer.target_scratch_full.candidate_timestamps",
    nested_get(
        runtime,
        "scaling",
        "cer",
        "target_scratch_full",
        "candidate_timestamps",
    ),
    CER_FULL_TRAIN_EXPECTED_CANDIDATES,
)

# CER non-standard-slot dates.
runtime_boundary_dates = nested_get(
    runtime,
    "datasets",
    "cer",
    "cer_irregular_slots",
    "nonstandard_slots",
    "dates",
)
if runtime_boundary_dates is not None:
    try:
        runtime_boundary_dates = [
            pd.Timestamp(value).date()
            for value in runtime_boundary_dates
        ]
    except Exception:
        runtime_boundary_dates = None
check_runtime(
    "CER compression-boundary dates",
    runtime_boundary_dates,
    CER_COMPRESSION_BOUNDARY_DATES,
)

# Split boundaries and native candidate counts.
for dataset_name, expected_splits in (
    ("lcl", LCL_SPLITS),
    ("cer", CER_SPLITS),
):
    for split_name, (expected_start, expected_end) in expected_splits.items():
        check_runtime_timestamp(
            f"splits.{dataset_name}.{split_name}.start",
            nested_get(
                runtime,
                "splits",
                dataset_name,
                split_name,
                "start",
            ),
            expected_start,
        )
        check_runtime_timestamp(
            f"splits.{dataset_name}.{split_name}.end",
            nested_get(
                runtime,
                "splits",
                dataset_name,
                split_name,
                "end",
            ),
            expected_end,
        )

runtime_lcl_candidates = nested_get(
    runtime,
    "splits",
    "lcl",
    "native_candidate_targets_per_household",
)
runtime_cer_candidates = nested_get(
    runtime,
    "splits",
    "cer",
    "native_candidate_targets_per_meter",
)

for horizon in HORIZONS:
    check_runtime(
        f"LCL YAML candidate counts h={horizon}",
        (
            runtime_lcl_candidates.get(horizon)
            if isinstance(runtime_lcl_candidates, dict)
            else None
        ),
        LCL_NATIVE_CANDIDATES[horizon],
    )
    check_runtime(
        f"CER YAML candidate counts h={horizon}",
        (
            runtime_cer_candidates.get(horizon)
            if isinstance(runtime_cer_candidates, dict)
            else None
        ),
        CER_NATIVE_CANDIDATES[horizon],
    )

if runtime_issues:
    print("\nRuntime YAML disagreements found:")
    for issue in runtime_issues:
        print(f"  - {issue}")
    raise AssertionError(
        "Sample construction stopped because forecasting_runtime.yaml does not "
        "match the formal Sample construction design. Correct the YAML rather "
        "than bypassing this gate."
    )

print("\nRuntime YAML is consistent with Sample construction.  PASS")


# ============================================================
# STEP 0b — Disk-space preflight
# ============================================================
print("\n" + "=" * 78)
print("STEP 0b — DISK-SPACE PREFLIGHT")
print("=" * 78)

free_bytes = shutil.disk_usage(OUT_TABLES).free
free_gib = free_bytes / (1024 ** 3)
print(f"Free space on output filesystem: {free_gib:.2f} GiB")

if free_gib < MIN_FREE_DISK_GIB:
    raise OSError(
        "Sample construction stopped before computation because the output "
        f"filesystem has {free_gib:.2f} GiB free. At least "
        f"{MIN_FREE_DISK_GIB:.1f} GiB is required for the six "
        "large temporary Parquet files plus safe finalization."
    )

print(
    f"Disk-space preflight: PASS (minimum {MIN_FREE_DISK_GIB:.1f} GiB)."
)


# ============================================================
# STEP 0c — LCL cohort integrity
# ============================================================
print("\n" + "=" * 78)
print("STEP 0c — LCL COHORT INTEGRITY")
print("=" * 78)

final_cohort = pd.read_csv(PATHS["lcl_final_cohort"])

if "household_id" not in final_cohort.columns:
    raise KeyError(
        "The final LCL cohort file does not contain household_id."
    )

cohort_ids = final_cohort["household_id"].astype(str)
row_count = len(cohort_ids)
unique_count = cohort_ids.nunique()
duplicate_count = int(cohort_ids.duplicated().sum())

if row_count != EXPECTED_LCL_FINAL:
    raise AssertionError(
        f"Cohort file has {row_count:,} rows; expected "
        f"{EXPECTED_LCL_FINAL:,}."
    )
if unique_count != EXPECTED_LCL_FINAL:
    raise AssertionError(
        f"Cohort file has {unique_count:,} unique IDs; expected "
        f"{EXPECTED_LCL_FINAL:,}."
    )
if duplicate_count != 0:
    raise AssertionError(
        f"Cohort file contains {duplicate_count:,} duplicate IDs."
    )

final_ids = sorted(cohort_ids.tolist())

for household_id, reason in NEGATIVE_TEST_IDS.items():
    if household_id in final_ids:
        raise AssertionError(
            f"Negative-test household {household_id} ({reason}) "
            "must not be in the final cohort."
        )

print(
    f"LCL cohort: {row_count:,} rows, {unique_count:,} unique IDs, "
    "0 duplicates; negative-test households absent.  PASS"
)


# ============================================================
# STEP 1 — Load LCL matrix in row chunks
# ============================================================
print("\n" + "=" * 78)
print("STEP 1 — LOAD LCL FULL PRODUCTION MATRIX (float64, chunked CSV)")
print("=" * 78)

time_map = pd.read_parquet(
    PATHS["lcl_time_map"],
    columns=[
        "raw_row_index",
        "timestamp_gmt",
        "local_slot",
        "local_day_of_week",
        "local_weekend",
        "local_month",
    ],
)

gmt_all = pd.to_datetime(time_map["timestamp_gmt"], errors="raise")
p0_mask = (gmt_all >= LCL_P0_START) & (gmt_all <= LCL_P0_END)
p0_time_map = time_map.loc[p0_mask].reset_index(drop=True)
p0_raw_indices = (
    p0_time_map["raw_row_index"].astype(np.int64).to_numpy()
)

if len(p0_raw_indices) != EXPECTED_LCL_POSITIONS:
    raise AssertionError(
        f"LCL P0 contains {len(p0_raw_indices):,} positions; "
        f"expected {EXPECTED_LCL_POSITIONS:,}."
    )

expected_raw_indices = np.arange(
    p0_raw_indices[0],
    p0_raw_indices[-1] + 1,
)
if not np.array_equal(p0_raw_indices, expected_raw_indices):
    raise AssertionError(
        "LCL P0 raw_row_index block is not contiguous."
    )

first_raw_row = int(p0_raw_indices[0])
LCL_GRID = pd.DatetimeIndex(gmt_all.to_numpy()[p0_mask.to_numpy()])

if LCL_GRID[0] != LCL_P0_START or LCL_GRID[-1] != LCL_P0_END:
    raise AssertionError(
        "LCL grid boundaries do not match the locked P0 period."
    )
lcl_grid_int64_ns = LCL_GRID.to_numpy(dtype="datetime64[ns]").view("int64")
if not np.all(np.diff(lcl_grid_int64_ns) == int(RESOLUTION.value)):
    raise AssertionError(
        "LCL GMT physical sequence is not uniformly 30 minutes."
    )

lcl_matrix = np.full(
    (EXPECTED_LCL_POSITIONS, EXPECTED_LCL_FINAL),
    np.nan,
    dtype=np.float64,
)

row_cursor = 0
lcl_reader = pd.read_csv(
    PATHS["lcl_raw"],
    usecols=final_ids,
    skiprows=range(1, first_raw_row + 1),
    nrows=EXPECTED_LCL_POSITIONS,
    chunksize=LCL_CSV_CHUNK_ROWS,
    low_memory=False,
)

for chunk_number, chunk in enumerate(lcl_reader, start=1):
    chunk = chunk[final_ids]
    numeric_chunk = (
        chunk.apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float64)
    )
    next_cursor = row_cursor + len(numeric_chunk)
    lcl_matrix[row_cursor:next_cursor, :] = numeric_chunk
    row_cursor = next_cursor
    print(
        f"  LCL chunk {chunk_number}: loaded through row "
        f"{row_cursor:,}/{EXPECTED_LCL_POSITIONS:,}"
    )

if row_cursor != EXPECTED_LCL_POSITIONS:
    raise AssertionError(
        f"LCL chunked read filled {row_cursor:,} rows; expected "
        f"{EXPECTED_LCL_POSITIONS:,}."
    )

lcl_present = np.isfinite(lcl_matrix)
lcl_local_slot = p0_time_map["local_slot"].to_numpy()
lcl_local_dow = p0_time_map["local_day_of_week"].to_numpy()
lcl_local_weekend = (
    p0_time_map["local_weekend"].to_numpy().astype(bool)
)
lcl_local_month = p0_time_map["local_month"].to_numpy()
lcl_compression_break_after = np.zeros(
    len(LCL_GRID),
    dtype=bool,
)

print(
    f"LCL matrix shape: {lcl_matrix.shape}, dtype={lcl_matrix.dtype}, "
    f"memory={lcl_matrix.nbytes / (1024 ** 2):.1f} MiB"
)
print(f"LCL missing cells: {int((~lcl_present).sum()):,}")
print("LCL physical GMT continuity: PASS")


# ============================================================
# STEP 2 — Load CER through batch scatter
# ============================================================
print("\n" + "=" * 78)
print("STEP 2 — LOAD CER VIA BATCH SCATTER (float64)")
print("=" * 78)

cer_dataset = ds.dataset(PATHS["cer_grid"], format="parquet")


def record_batch_column(batch: pa.RecordBatch, name: str):
    field_index = batch.schema.get_field_index(name)
    if field_index < 0:
        raise KeyError(f"Record batch does not contain column {name!r}.")
    return batch.column(field_index)


# Streaming single-column prepass for the governed meter universe.
meter_id_set: set[int] = set()
meter_scanner = cer_dataset.scanner(
    columns=["meter_id"],
    batch_size=CER_LOAD_BATCH_SIZE,
)
for batch in meter_scanner.to_batches():
    meter_values = record_batch_column(batch, "meter_id").to_numpy(
        zero_copy_only=False
    )
    meter_id_set.update(np.unique(meter_values).tolist())

unique_meter_ids_numeric = np.asarray(
    sorted(meter_id_set),
    dtype=np.int64,
)
if len(unique_meter_ids_numeric) != EXPECTED_CER_METERS:
    raise AssertionError(
        f"Expected {EXPECTED_CER_METERS:,} CER meters; found "
        f"{len(unique_meter_ids_numeric):,}."
    )

# Artifact-facing CER IDs are strings everywhere: indexes, summaries,
# scaler tables and downstream joins.
cer_entity_ids = [str(int(value)) for value in unique_meter_ids_numeric]
n_cer_meters = len(cer_entity_ids)

CER_GRID = pd.date_range(
    CER_P0_START,
    CER_P0_END,
    freq="30min",
)
n_cer_positions = len(CER_GRID)
if n_cer_positions != EXPECTED_CER_POSITIONS_PER_METER:
    raise AssertionError(
        f"CER deterministic grid has {n_cer_positions:,} positions; "
        f"expected {EXPECTED_CER_POSITIONS_PER_METER:,}."
    )

grid_int64 = CER_GRID.to_numpy(dtype="datetime64[ns]").view("int64")
cer_matrix = np.full(
    (n_cer_positions, n_cer_meters),
    np.nan,
    dtype=np.float64,
)
cer_seen = np.zeros(
    (n_cer_positions, n_cer_meters),
    dtype=bool,
)
cer_calendar = np.full(
    (n_cer_positions, 4),
    -1,
    dtype=np.int32,
)
cer_calendar_seen = np.zeros(n_cer_positions, dtype=bool)

scanner = cer_dataset.scanner(
    columns=[
        "meter_id",
        "timestamp_local",
        "local_slot",
        "local_day_of_week",
        "local_weekend",
        "local_month",
        "kwh",
        "is_missing",
    ],
    batch_size=CER_LOAD_BATCH_SIZE,
)

batch_count = 0
total_rows = 0

for batch in scanner.to_batches():
    batch_count += 1
    total_rows += batch.num_rows

    meter_arr = record_batch_column(batch, "meter_id").to_numpy(
        zero_copy_only=False
    )
    timestamp_arr = record_batch_column(
        batch, "timestamp_local"
    ).to_numpy(zero_copy_only=False)
    kwh_arr = np.asarray(
        record_batch_column(batch, "kwh").to_numpy(
            zero_copy_only=False
        ),
        dtype=np.float64,
    )
    is_missing_arr = np.asarray(
        record_batch_column(batch, "is_missing").to_numpy(
            zero_copy_only=False
        ),
        dtype=bool,
    )
    slot_arr = np.asarray(
        record_batch_column(batch, "local_slot").to_numpy(
            zero_copy_only=False
        ),
        dtype=np.int32,
    )
    dow_arr = np.asarray(
        record_batch_column(batch, "local_day_of_week").to_numpy(
            zero_copy_only=False
        ),
        dtype=np.int32,
    )
    weekend_arr = np.asarray(
        record_batch_column(batch, "local_weekend").to_numpy(
            zero_copy_only=False
        ),
        dtype=bool,
    )
    month_arr = np.asarray(
        record_batch_column(batch, "local_month").to_numpy(
            zero_copy_only=False
        ),
        dtype=np.int32,
    )

    if np.any(is_missing_arr != np.isnan(kwh_arr)):
        raise AssertionError(
            f"CER is_missing/kWh-NaN disagreement in batch {batch_count}."
        )

    meter_idx = np.searchsorted(unique_meter_ids_numeric, meter_arr)
    meter_in_bounds = (
        (meter_idx >= 0)
        & (meter_idx < n_cer_meters)
    )
    if not np.all(meter_in_bounds):
        raise AssertionError(
            f"Batch {batch_count}: a CER meter ID falls outside the "
            "governed meter universe."
        )
    if not np.array_equal(
        unique_meter_ids_numeric[meter_idx],
        meter_arr,
    ):
        raise AssertionError(
            f"Batch {batch_count}: a CER meter ID did not exactly match "
            "the governed meter universe."
        )

    timestamp_int64 = (
        timestamp_arr.astype("datetime64[ns]").view("int64")
    )
    time_idx = np.searchsorted(grid_int64, timestamp_int64)
    time_in_bounds = (
        (time_idx >= 0)
        & (time_idx < n_cer_positions)
    )
    if not np.all(time_in_bounds):
        raise AssertionError(
            f"Batch {batch_count}: a CER timestamp falls outside the "
            "locked canonical grid."
        )
    if not np.array_equal(grid_int64[time_idx], timestamp_int64):
        raise AssertionError(
            f"Batch {batch_count}: a CER timestamp did not exactly match "
            "the locked half-hour grid."
        )

    # Detect duplicate coordinates within the current batch before any
    # global seen-mask assignment occurs.
    linear_key = (
        time_idx.astype(np.int64) * n_cer_meters
        + meter_idx.astype(np.int64)
    )
    if np.unique(linear_key).size != linear_key.size:
        raise AssertionError(
            f"Batch {batch_count}: duplicate meter-timestamp rows exist "
            "within the same batch."
        )

    # Detect duplicates against all earlier batches.
    already_seen = cer_seen[time_idx, meter_idx]
    if np.any(already_seen):
        raise AssertionError(
            f"Batch {batch_count}: {int(already_seen.sum()):,} "
            "meter-timestamp rows duplicate earlier batches."
        )

    # Verify calendar consistency within this batch for every timestamp.
    calendar_values = np.column_stack(
        [
            slot_arr,
            dow_arr,
            weekend_arr.astype(np.int32),
            month_arr,
        ]
    ).astype(np.int32, copy=False)

    order = np.argsort(time_idx, kind="mergesort")
    sorted_time = time_idx[order]
    sorted_calendar = calendar_values[order]
    group_start = np.r_[
        True,
        sorted_time[1:] != sorted_time[:-1],
    ]
    group_number = np.cumsum(group_start) - 1
    batch_calendar_reference = sorted_calendar[group_start]

    if not np.all(
        sorted_calendar
        == batch_calendar_reference[group_number]
    ):
        raise AssertionError(
            f"Batch {batch_count}: CER calendar fields disagree across "
            "meters for the same timestamp within the batch."
        )

    unique_time_idx = sorted_time[group_start]
    unique_calendar_values = batch_calendar_reference
    globally_seen = cer_calendar_seen[unique_time_idx]

    if np.any(globally_seen):
        existing_idx = unique_time_idx[globally_seen]
        existing_values = unique_calendar_values[globally_seen]
        if not np.array_equal(
            cer_calendar[existing_idx],
            existing_values,
        ):
            raise AssertionError(
                f"Batch {batch_count}: CER calendar fields disagree with "
                "calendar data recorded by an earlier batch."
            )

    new_calendar_mask = ~globally_seen
    if np.any(new_calendar_mask):
        new_idx = unique_time_idx[new_calendar_mask]
        cer_calendar[new_idx] = unique_calendar_values[new_calendar_mask]
        cer_calendar_seen[new_idx] = True

    cer_seen[time_idx, meter_idx] = True
    cer_matrix[time_idx, meter_idx] = kwh_arr

    print(
        f"  CER batch {batch_count}: {batch.num_rows:,} rows scattered "
        f"(cumulative {total_rows:,}/{EXPECTED_CER_TOTAL_ROWS:,})"
    )

    del (
        meter_arr,
        timestamp_arr,
        kwh_arr,
        is_missing_arr,
        slot_arr,
        dow_arr,
        weekend_arr,
        month_arr,
        meter_idx,
        timestamp_int64,
        time_idx,
        linear_key,
        calendar_values,
        order,
        sorted_time,
        sorted_calendar,
        group_start,
        group_number,
        batch_calendar_reference,
        unique_time_idx,
        unique_calendar_values,
    )

if total_rows != EXPECTED_CER_TOTAL_ROWS:
    raise AssertionError(
        f"CER batch scatter read {total_rows:,} rows; expected "
        f"{EXPECTED_CER_TOTAL_ROWS:,}."
    )
if not cer_seen.all():
    raise AssertionError(
        f"{int((~cer_seen).sum()):,} CER meter-timestamp cells were "
        "never represented by a canonical-grid row."
    )
if not cer_calendar_seen.all():
    raise AssertionError(
        "Some CER grid positions never received calendar data."
    )

del cer_seen
gc.collect()

cer_present = np.isfinite(cer_matrix)
cer_local_slot = cer_calendar[:, 0]
cer_local_dow = cer_calendar[:, 1]
cer_local_weekend = cer_calendar[:, 2].astype(bool)
cer_local_month = cer_calendar[:, 3]

if not np.all((cer_local_slot >= 1) & (cer_local_slot <= 48)):
    raise AssertionError("CER canonical local_slot is not confined to 1-48.")
if not np.all((cer_local_dow >= 0) & (cer_local_dow <= 6)):
    raise AssertionError("CER local_day_of_week is outside 0-6.")
if not np.all((cer_local_month >= 1) & (cer_local_month <= 12)):
    raise AssertionError("CER local_month is outside 1-12.")

cer_dates = CER_GRID.date
cer_compression_break_after = np.zeros(
    n_cer_positions,
    dtype=bool,
)
for boundary_date in CER_COMPRESSION_BOUNDARY_DATES:
    positions_on_date = np.flatnonzero(cer_dates == boundary_date)
    if len(positions_on_date) != SLOTS_PER_DAY:
        raise AssertionError(
            f"CER boundary date {boundary_date} has "
            f"{len(positions_on_date)} canonical positions; expected 48."
        )
    boundary_position = int(positions_on_date[-1])
    if cer_local_slot[boundary_position] != 48:
        raise AssertionError(
            f"CER boundary date {boundary_date} does not end at slot 48."
        )
    cer_compression_break_after[boundary_position] = True
    print(
        "  Conservative continuity boundary marked after "
        f"{CER_GRID[boundary_position]}."
    )

if int(cer_compression_break_after.sum()) != 2:
    raise AssertionError(
        "Expected exactly two CER conservative continuity boundaries."
    )

print(
    f"CER matrix shape: {cer_matrix.shape}, dtype={cer_matrix.dtype}, "
    f"memory={cer_matrix.nbytes / (1024 ** 2):.1f} MiB"
)
print(f"CER missing cells: {int((~cer_present).sum()):,}")
print("CER row count, duplicate, missing-flag and calendar gates: PASS")


# ============================================================
# STEP 3 — Stream-build six production indexes
# ============================================================
print("\n" + "=" * 78)
print("STEP 3 — STREAM-BUILD SIX PRODUCTION SAMPLE INDEXES")
print("=" * 78)

writers: dict[tuple[str, int], pq.ParquetWriter] = {}
entity_summary_rows: list[dict] = []
representative_candidates: dict[tuple[str, int, str], list[dict]] = (
    defaultdict(list)
)
representative_rng = np.random.default_rng(20260712)


def write_valid_rows(
    writer: pq.ParquetWriter,
    dataset_name: str,
    horizon: int,
    entity_ids: np.ndarray,
    target_timestamps: np.ndarray,
    split_values: np.ndarray,
):
    total = len(entity_ids)
    for chunk_start in range(0, total, PARQUET_WRITE_CHUNK_ROWS):
        chunk_end = min(chunk_start + PARQUET_WRITE_CHUNK_ROWS, total)
        chunk_length = chunk_end - chunk_start

        table = pa.Table.from_arrays(
            [
                pa.array(
                    [dataset_name] * chunk_length,
                    type=pa.string(),
                ),
                pa.array(
                    entity_ids[chunk_start:chunk_end],
                    type=pa.string(),
                ),
                pa.array(
                    np.full(chunk_length, horizon, dtype=np.int16),
                    type=pa.int16(),
                ),
                pa.array(
                    target_timestamps[chunk_start:chunk_end],
                    type=pa.timestamp("ns"),
                ),
                pa.array(
                    split_values[chunk_start:chunk_end],
                    type=pa.string(),
                ),
            ],
            schema=INDEX_SCHEMA,
        )
        writer.write_table(
            table,
            row_group_size=PARQUET_WRITE_CHUNK_ROWS,
        )


def stream_build_index(
    dataset_name: str,
    grid: pd.DatetimeIndex,
    matrix: np.ndarray,
    present: np.ndarray,
    entity_ids: list[str],
    splits: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
    compression_break_after: np.ndarray,
    block_size: int,
):
    n_positions = len(grid)
    if matrix.shape != present.shape:
        raise AssertionError(
            f"{dataset_name}: matrix/present shape mismatch."
        )
    if matrix.shape[0] != n_positions:
        raise AssertionError(
            f"{dataset_name}: matrix/grid length mismatch."
        )
    if matrix.shape[1] != len(entity_ids):
        raise AssertionError(
            f"{dataset_name}: matrix/entity count mismatch."
        )
    if len(compression_break_after) != n_positions:
        raise AssertionError(
            f"{dataset_name}: continuity mask length mismatch."
        )

    boundary_positions = np.flatnonzero(compression_break_after)
    precomputed: dict[int, dict] = {}

    for horizon in HORIZONS:
        first_target_position = LOOKBACK - 1 + horizon
        target_positions = np.arange(
            first_target_position,
            n_positions,
            dtype=np.int64,
        )
        input_starts = (
            target_positions
            - horizon
            - (LOOKBACK - 1)
        )
        input_ends = target_positions - horizon

        if not np.all(input_ends - input_starts == LOOKBACK - 1):
            raise AssertionError(
                f"{dataset_name} h={horizon}: input span is not 48 steps."
            )
        if not np.all(target_positions - input_ends == horizon):
            raise AssertionError(
                f"{dataset_name} h={horizon}: target offset is incorrect."
            )

        target_timestamps_index = grid[target_positions]
        target_timestamps = target_timestamps_index.to_numpy(
            dtype="datetime64[ns]"
        )
        split_of_target = np.full(
            len(target_positions),
            "",
            dtype=object,
        )
        split_masks: dict[str, np.ndarray] = {}

        for split_name, (split_start, split_end) in splits.items():
            split_mask = (
                (target_timestamps_index >= split_start)
                & (target_timestamps_index <= split_end)
            )
            split_masks[split_name] = np.asarray(split_mask, dtype=bool)
            split_of_target[split_mask] = split_name

        in_any_split = split_of_target != ""
        continuity_ok = np.ones(len(target_positions), dtype=bool)
        for boundary_position in boundary_positions:
            continuity_ok &= ~(
                (input_starts <= boundary_position)
                & (boundary_position < target_positions)
            )

        precomputed[horizon] = {
            "target_positions": target_positions,
            "input_starts": input_starts,
            "target_timestamps": target_timestamps,
            "split_of_target": split_of_target,
            "split_masks": split_masks,
            "in_any_split": in_any_split,
            "continuity_ok": continuity_ok,
        }

    entity_array = np.asarray(entity_ids, dtype=object)
    n_entities = len(entity_ids)

    for block_start in range(0, n_entities, block_size):
        block_end = min(block_start + block_size, n_entities)
        block_ids = entity_array[block_start:block_end]
        block_present = present[:, block_start:block_end]

        observed_cumulative = np.vstack(
            [
                np.zeros(
                    (1, block_present.shape[1]),
                    dtype=np.int32,
                ),
                np.cumsum(
                    block_present,
                    axis=0,
                    dtype=np.int32,
                ),
            ]
        )

        for horizon in HORIZONS:
            pc = precomputed[horizon]
            target_positions = pc["target_positions"]
            input_starts = pc["input_starts"]

            input_observed = (
                observed_cumulative[input_starts + LOOKBACK, :]
                - observed_cumulative[input_starts, :]
            )
            input_ok = input_observed == LOOKBACK
            target_ok = block_present[target_positions, :]
            complete_without_continuity = (
                input_ok
                & target_ok
                & pc["in_any_split"][:, None]
            )
            window_valid = (
                complete_without_continuity
                & pc["continuity_ok"][:, None]
            )

            valid_row_idx, valid_col_idx = np.nonzero(window_valid)

            if len(valid_row_idx) > 0:
                valid_entity_ids = block_ids[valid_col_idx]
                valid_timestamps = pc["target_timestamps"][valid_row_idx]
                valid_splits = pc["split_of_target"][valid_row_idx]

                write_valid_rows(
                    writer=writers[(dataset_name, horizon)],
                    dataset_name=dataset_name,
                    horizon=horizon,
                    entity_ids=valid_entity_ids,
                    target_timestamps=valid_timestamps,
                    split_values=valid_splits,
                )

                # Collect one deterministic random candidate from every
                # processed entity block for each split. Final spot-check
                # examples are selected across the full list of blocks,
                # avoiding a first-block-only bias.
                for split_name in splits:
                    split_positions = np.flatnonzero(
                        valid_splits == split_name
                    )
                    if len(split_positions) == 0:
                        continue
                    chosen_position = int(
                        split_positions[
                            representative_rng.integers(
                                0,
                                len(split_positions),
                            )
                        ]
                    )
                    representative_candidates[
                        (dataset_name, horizon, split_name)
                    ].append(
                        {
                            "entity_id": str(
                                valid_entity_ids[chosen_position]
                            ),
                            "target_timestamp": pd.Timestamp(
                                valid_timestamps[chosen_position]
                            ),
                        }
                    )

            for local_entity_position, entity_id in enumerate(block_ids):
                for split_name, split_mask in pc["split_masks"].items():
                    candidate_count = int(split_mask.sum())
                    complete_count = int(
                        (
                            complete_without_continuity[
                                :, local_entity_position
                            ]
                            & split_mask
                        ).sum()
                    )
                    continuity_excluded_count = int(
                        (
                            complete_without_continuity[
                                :, local_entity_position
                            ]
                            & split_mask
                            & (~pc["continuity_ok"])
                        ).sum()
                    )
                    valid_count = int(
                        (
                            window_valid[:, local_entity_position]
                            & split_mask
                        ).sum()
                    )
                    missing_excluded_count = (
                        candidate_count - complete_count
                    )

                    entity_summary_rows.append(
                        {
                            "dataset": dataset_name,
                            "entity_id": str(entity_id),
                            "horizon": horizon,
                            "split": split_name,
                            "candidate_windows": candidate_count,
                            "missing_input_or_target_excluded": (
                                missing_excluded_count
                            ),
                            "continuity_excluded": (
                                continuity_excluded_count
                            ),
                            "valid_windows": valid_count,
                        }
                    )

            del (
                input_observed,
                input_ok,
                target_ok,
                complete_without_continuity,
                window_valid,
                valid_row_idx,
                valid_col_idx,
            )

        del observed_cumulative, block_present
        gc.collect()
        print(
            f"  {dataset_name}: entities {block_start + 1:,}-"
            f"{block_end:,}/{n_entities:,} streamed"
        )


try:
    for key, temp_path in TMP_INDEX_FILES.items():
        writers[key] = pq.ParquetWriter(
            where=str(temp_path),
            schema=INDEX_SCHEMA,
            compression=PARQUET_COMPRESSION,
            compression_level=PARQUET_COMPRESSION_LEVEL,
            use_dictionary=True,
            write_statistics=True,
        )

    stream_build_index(
        dataset_name="LCL",
        grid=LCL_GRID,
        matrix=lcl_matrix,
        present=lcl_present,
        entity_ids=final_ids,
        splits=LCL_SPLITS,
        compression_break_after=lcl_compression_break_after,
        block_size=LCL_ENTITY_BLOCK_SIZE,
    )
    stream_build_index(
        dataset_name="CER",
        grid=CER_GRID,
        matrix=cer_matrix,
        present=cer_present,
        entity_ids=cer_entity_ids,
        splits=CER_SPLITS,
        compression_break_after=cer_compression_break_after,
        block_size=CER_ENTITY_BLOCK_SIZE,
    )
finally:
    for writer in writers.values():
        writer.close()

print("\nAll six temporary Parquet indexes were closed successfully.")


# ============================================================
# STEP 4 — Count identities and aggregate totals
# ============================================================
print("\n" + "=" * 78)
print("STEP 4 — COUNT IDENTITIES AND AGGREGATE TOTALS")
print("=" * 78)

entity_summary = pd.DataFrame(entity_summary_rows)
del entity_summary_rows

expected_entity_summary_rows = (
    (EXPECTED_LCL_FINAL + EXPECTED_CER_METERS)
    * len(HORIZONS)
    * 3
)
if len(entity_summary) != expected_entity_summary_rows:
    raise AssertionError(
        f"Entity summary has {len(entity_summary):,} rows; expected "
        f"{expected_entity_summary_rows:,}."
    )

if entity_summary.duplicated(
    subset=["dataset", "entity_id", "horizon", "split"]
).any():
    raise AssertionError(
        "Entity summary contains duplicate dataset/entity/horizon/split rows."
    )

identity_left = (
    entity_summary["candidate_windows"]
    - entity_summary["missing_input_or_target_excluded"]
    - entity_summary["continuity_excluded"]
)
identity_ok = identity_left == entity_summary["valid_windows"]
if not identity_ok.all():
    bad_count = int((~identity_ok).sum())
    raise AssertionError(
        "candidate - missing - continuity != valid for "
        f"{bad_count:,} entity-level rows."
    )

for count_column in (
    "candidate_windows",
    "missing_input_or_target_excluded",
    "continuity_excluded",
    "valid_windows",
):
    if (entity_summary[count_column] < 0).any():
        raise AssertionError(
            f"Entity summary contains negative values in {count_column}."
        )

aggregate_summary = (
    entity_summary.groupby(
        ["dataset", "horizon", "split"],
        as_index=False,
    )[
        [
            "candidate_windows",
            "missing_input_or_target_excluded",
            "continuity_excluded",
            "valid_windows",
        ]
    ]
    .sum()
    .sort_values(["dataset", "horizon", "split"])
    .reset_index(drop=True)
)

if len(aggregate_summary) != 18:
    raise AssertionError(
        f"Aggregate summary has {len(aggregate_summary)} rows; expected 18."
    )

for row in aggregate_summary.itertuples(index=False):
    expected_candidates = (
        LCL_NATIVE_CANDIDATES
        if row.dataset == "LCL"
        else CER_NATIVE_CANDIDATES
    )
    entity_count = (
        EXPECTED_LCL_FINAL
        if row.dataset == "LCL"
        else EXPECTED_CER_METERS
    )
    expected_total = (
        expected_candidates[int(row.horizon)][row.split]
        * entity_count
    )
    if int(row.candidate_windows) != expected_total:
        raise AssertionError(
            f"{row.dataset} h={row.horizon} {row.split}: candidate "
            f"total {int(row.candidate_windows):,} does not match "
            f"the deterministic expectation {expected_total:,}."
        )

print(
    "Entity-level identity candidate - missing - continuity = valid: "
    f"PASS for all {len(entity_summary):,} rows."
)
print("Deterministic candidate totals: PASS for all 18 aggregate cells.")
print("\n" + aggregate_summary.to_string(index=False))


# ============================================================
# STEP 5 — Validate actual temporary Parquet artifacts
# ============================================================
print("\n" + "=" * 78)
print("STEP 5 — TEMPORARY PARQUET ARTIFACT VALIDATION")
print("=" * 78)


def normalize_stat_value(value):
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    return value


for (dataset_name, horizon), temp_path in TMP_INDEX_FILES.items():
    if not temp_path.exists() or temp_path.stat().st_size == 0:
        raise AssertionError(
            f"Temporary Parquet output is missing or empty: {temp_path}"
        )

    parquet_file = pq.ParquetFile(temp_path)
    actual_rows = parquet_file.metadata.num_rows
    expected_rows = int(
        aggregate_summary.loc[
            (aggregate_summary["dataset"] == dataset_name)
            & (aggregate_summary["horizon"] == horizon),
            "valid_windows",
        ].sum()
    )

    if actual_rows != expected_rows:
        raise AssertionError(
            f"{dataset_name} h={horizon}: Parquet has "
            f"{actual_rows:,} rows, but summary expects "
            f"{expected_rows:,}."
        )
    if actual_rows <= 0:
        raise AssertionError(
            f"{dataset_name} h={horizon}: Parquet contains no rows."
        )
    if not parquet_file.schema_arrow.equals(
        INDEX_SCHEMA,
        check_metadata=False,
    ):
        raise AssertionError(
            f"{dataset_name} h={horizon}: Parquet schema does not "
            "match the locked index schema."
        )

    dataset_column_index = INDEX_SCHEMA.get_field_index("dataset")
    horizon_column_index = INDEX_SCHEMA.get_field_index("horizon")

    if parquet_file.metadata.num_row_groups <= 0:
        raise AssertionError(
            f"{dataset_name} h={horizon}: no Parquet row groups exist."
        )

    for row_group_index in range(parquet_file.metadata.num_row_groups):
        row_group = parquet_file.metadata.row_group(row_group_index)
        dataset_stats = row_group.column(dataset_column_index).statistics
        horizon_stats = row_group.column(horizon_column_index).statistics

        if dataset_stats is None or horizon_stats is None:
            raise AssertionError(
                f"{dataset_name} h={horizon}: missing Parquet statistics "
                f"in row group {row_group_index}."
            )

        dataset_min = normalize_stat_value(dataset_stats.min)
        dataset_max = normalize_stat_value(dataset_stats.max)
        horizon_min = int(horizon_stats.min)
        horizon_max = int(horizon_stats.max)

        if dataset_min != dataset_name or dataset_max != dataset_name:
            raise AssertionError(
                f"{dataset_name} h={horizon}: row group "
                f"{row_group_index} contains an incorrect dataset value."
            )
        if horizon_min != horizon or horizon_max != horizon:
            raise AssertionError(
                f"{dataset_name} h={horizon}: row group "
                f"{row_group_index} contains an incorrect horizon value."
            )

    print(
        f"  {dataset_name} h={horizon}: rows={actual_rows:,}, "
        f"row_groups={parquet_file.metadata.num_row_groups:,}, "
        f"size={temp_path.stat().st_size / (1024 ** 3):.2f} GiB  PASS"
    )


# ============================================================
# STEP 6 — Formal scaler tables and invariants
# ============================================================
print("\n" + "=" * 78)
print("STEP 6 — FORMAL SCALER TABLES (float64)")
print("=" * 78)


def fit_scaler(values_window: np.ndarray):
    observed = values_window[np.isfinite(values_window)]
    if observed.size == 0:
        raise AssertionError(
            "Scaler fitting window contains no observed values."
        )

    mean = float(np.mean(observed))
    raw_std = float(np.std(observed, ddof=STD_DDOF))
    effective_std = max(raw_std, STD_FLOOR)

    if not np.isfinite(mean):
        raise AssertionError("Scaler mean is not finite.")
    if not np.isfinite(raw_std) or raw_std < 0:
        raise AssertionError("Scaler raw standard deviation is invalid.")
    if not np.isfinite(effective_std) or effective_std < STD_FLOOR:
        raise AssertionError("Scaler effective standard deviation is invalid.")

    return mean, raw_std, effective_std, int(observed.size)


lcl_train_mask = (
    (LCL_GRID >= LCL_SPLITS["train"][0])
    & (LCL_GRID <= LCL_SPLITS["train"][1])
)
if int(lcl_train_mask.sum()) != 12_230:
    raise AssertionError(
        "LCL scaler-fit timestamp count is not 12,230."
    )

lcl_scaler_rows: list[dict] = []
for entity_position, household_id in enumerate(final_ids):
    mean, raw_std, effective_std, n_observed = fit_scaler(
        lcl_matrix[lcl_train_mask, entity_position]
    )
    lcl_scaler_rows.append(
        {
            "dataset": "LCL",
            "entity_id": household_id,
            "span_setting": "lcl_full_training",
            "scaler_fit_start": LCL_SPLITS["train"][0],
            "scaler_fit_end": LCL_SPLITS["train"][1],
            "n_observed": n_observed,
            "training_mean_kwh": mean,
            "training_std_kwh": raw_std,
            "effective_training_std_kwh": effective_std,
            "ddof": STD_DDOF,
        }
    )

lcl_scaler_table = pd.DataFrame(lcl_scaler_rows)

cer_full_train_mask = (
    (CER_GRID >= CER_SPLITS["train"][0])
    & (CER_GRID <= CER_SPLITS["train"][1])
)
cer_30day_mask = (
    (CER_GRID >= CER_30DAY_SCALER_START)
    & (CER_GRID <= CER_30DAY_SCALER_END)
)

if int(cer_full_train_mask.sum()) != CER_FULL_TRAIN_EXPECTED_CANDIDATES:
    raise AssertionError(
        "CER full-training scaler timestamp count mismatch."
    )
if int(cer_30day_mask.sum()) != CER_30DAY_EXPECTED_CANDIDATES:
    raise AssertionError(
        "CER 30-day scaler timestamp count mismatch."
    )
if np.any((CER_GRID < CER_SPLITS["train"][0]) & cer_30day_mask):
    raise AssertionError(
        "CER support day leaked into the 30-day scaler window."
    )
if CER_30DAY_SCALER_END >= CER_SPLITS["validation"][0]:
    raise AssertionError(
        "CER 30-day scaler period reaches validation."
    )

cer_scaler_rows: list[dict] = []
for entity_position, meter_id_numeric in enumerate(unique_meter_ids_numeric):
    entity_id = str(int(meter_id_numeric))
    for span_name, span_mask, fit_start, fit_end in (
        (
            "cer_30day",
            cer_30day_mask,
            CER_30DAY_SCALER_START,
            CER_30DAY_SCALER_END,
        ),
        (
            "cer_full_training",
            cer_full_train_mask,
            CER_SPLITS["train"][0],
            CER_SPLITS["train"][1],
        ),
    ):
        mean, raw_std, effective_std, n_observed = fit_scaler(
            cer_matrix[span_mask, entity_position]
        )
        cer_scaler_rows.append(
            {
                "dataset": "CER",
                "entity_id": entity_id,
                "span_setting": span_name,
                "scaler_fit_start": fit_start,
                "scaler_fit_end": fit_end,
                "n_observed": n_observed,
                "training_mean_kwh": mean,
                "training_std_kwh": raw_std,
                "effective_training_std_kwh": effective_std,
                "ddof": STD_DDOF,
            }
        )

cer_scaler_table = pd.DataFrame(cer_scaler_rows)


def validate_scaler_table(
    table: pd.DataFrame,
    label: str,
    expected_rows: int,
    expected_entities: int,
    expected_spans_per_entity: int,
):
    if len(table) != expected_rows:
        raise AssertionError(
            f"{label} scaler table has {len(table):,} rows; expected "
            f"{expected_rows:,}."
        )
    if table["entity_id"].nunique() != expected_entities:
        raise AssertionError(
            f"{label} scaler table has "
            f"{table['entity_id'].nunique():,} unique entities; expected "
            f"{expected_entities:,}."
        )
    if table.duplicated(subset=["entity_id", "span_setting"]).any():
        raise AssertionError(
            f"{label} scaler table contains duplicate entity/span rows."
        )

    span_counts = table.groupby("entity_id")["span_setting"].nunique()
    if not (span_counts == expected_spans_per_entity).all():
        raise AssertionError(
            f"{label} scaler table does not contain the expected number "
            "of scaler spans for every entity."
        )

    if not (table["n_observed"] > 0).all():
        raise AssertionError(
            f"{label} scaler table contains n_observed <= 0."
        )
    if not (table["ddof"] == STD_DDOF).all():
        raise AssertionError(
            f"{label} scaler table contains an incorrect ddof."
        )

    numeric_columns = [
        "training_mean_kwh",
        "training_std_kwh",
        "effective_training_std_kwh",
    ]
    numeric_values = table[numeric_columns].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(numeric_values)):
        raise AssertionError(
            f"{label} scaler table contains non-finite values."
        )
    if not (table["training_std_kwh"] >= 0).all():
        raise AssertionError(
            f"{label} scaler table contains a negative raw standard deviation."
        )
    if not (
        table["effective_training_std_kwh"] >= STD_FLOOR
    ).all():
        raise AssertionError(
            f"{label} scaler table violates the 0.01 kWh floor."
        )

    expected_effective = np.maximum(
        table["training_std_kwh"].to_numpy(dtype=np.float64),
        STD_FLOOR,
    )
    if not np.allclose(
        table["effective_training_std_kwh"].to_numpy(dtype=np.float64),
        expected_effective,
        rtol=0.0,
        atol=1e-15,
    ):
        raise AssertionError(
            f"{label} effective standard deviations do not equal "
            "max(raw_std, 0.01)."
        )


validate_scaler_table(
    table=lcl_scaler_table,
    label="LCL",
    expected_rows=EXPECTED_LCL_FINAL,
    expected_entities=EXPECTED_LCL_FINAL,
    expected_spans_per_entity=1,
)
validate_scaler_table(
    table=cer_scaler_table,
    label="CER",
    expected_rows=EXPECTED_CER_METERS * 2,
    expected_entities=EXPECTED_CER_METERS,
    expected_spans_per_entity=2,
)

n_floored_lcl = int(
    (lcl_scaler_table["training_std_kwh"] < STD_FLOOR).sum()
)
n_floored_cer = int(
    (cer_scaler_table["training_std_kwh"] < STD_FLOOR).sum()
)

# Retain the real branch check already demonstrated in Data eligibility.
n3487_row = lcl_scaler_table.loc[
    lcl_scaler_table["entity_id"] == "N3487"
]
if len(n3487_row) != 1:
    raise AssertionError(
        "N3487 is missing or duplicated in the formal LCL scaler table."
    )
if not (
    float(n3487_row.iloc[0]["training_std_kwh"]) < STD_FLOOR
    and np.isclose(
        float(n3487_row.iloc[0]["effective_training_std_kwh"]),
        STD_FLOOR,
    )
):
    raise AssertionError(
        "N3487 did not trigger the expected real standard-deviation floor."
    )

print(
    f"LCL scaler table: {len(lcl_scaler_table):,} rows, "
    f"{n_floored_lcl:,} floor-triggered.  PASS"
)
print(
    f"CER scaler table: {len(cer_scaler_table):,} rows, "
    f"{n_floored_cer:,} floor-triggered.  PASS"
)
print("Scaler finite-value, uniqueness, ddof and floor invariants: PASS")


# ============================================================
# STEP 7 — Representative lazy-reconstruction checks
# ============================================================
print("\n" + "=" * 78)
print("STEP 7 — REPRESENTATIVE LAZY-RECONSTRUCTION CHECKS")
print("=" * 78)


def make_features(
    local_slot_window,
    local_dow_window,
    local_weekend_window,
    local_month_window,
    load_window,
    mean,
    effective_std,
):
    scaled_load = (load_window - mean) / effective_std
    time_of_day_angle = (
        2 * np.pi * (local_slot_window - 1) / SLOTS_PER_DAY
    )
    day_of_week_angle = 2 * np.pi * local_dow_window / 7
    month_angle = 2 * np.pi * (local_month_window - 1) / 12

    return np.stack(
        [
            scaled_load,
            np.sin(time_of_day_angle),
            np.cos(time_of_day_angle),
            np.sin(day_of_week_angle),
            np.cos(day_of_week_angle),
            local_weekend_window.astype(np.float64),
            np.sin(month_angle),
            np.cos(month_angle),
        ],
        axis=-1,
    )


# Select examples across early, middle and late processed blocks.
representative_examples: dict[tuple[str, int, str], list[dict]] = {}
missing_strata: list[tuple[str, int, str]] = []

for dataset_name in ("LCL", "CER"):
    split_names = (
        LCL_SPLITS.keys()
        if dataset_name == "LCL"
        else CER_SPLITS.keys()
    )
    for horizon in HORIZONS:
        for split_name in split_names:
            key = (dataset_name, horizon, split_name)
            candidates = representative_candidates.get(key, [])
            if not candidates:
                missing_strata.append(key)
                continue

            n_examples = min(
                REPRESENTATIVE_EXAMPLES_PER_STRATUM,
                len(candidates),
            )
            selected_positions = np.unique(
                np.round(
                    np.linspace(
                        0,
                        len(candidates) - 1,
                        num=n_examples,
                    )
                ).astype(int)
            )
            representative_examples[key] = [
                candidates[position]
                for position in selected_positions
            ]

if missing_strata:
    raise AssertionError(
        f"No representative production sample was collected for: "
        f"{missing_strata}"
    )

lcl_scaler_lookup = (
    lcl_scaler_table.set_index("entity_id")
    [["training_mean_kwh", "effective_training_std_kwh"]]
)
cer_scaler_lookup = (
    cer_scaler_table.loc[
        cer_scaler_table["span_setting"] == "cer_full_training"
    ]
    .set_index("entity_id")
    [["training_mean_kwh", "effective_training_std_kwh"]]
)

lcl_id_to_column = {
    entity_id: position
    for position, entity_id in enumerate(final_ids)
}
cer_id_to_column = {
    entity_id: position
    for position, entity_id in enumerate(cer_entity_ids)
}

dataset_arrays = {
    "LCL": {
        "grid": LCL_GRID,
        "matrix": lcl_matrix,
        "id_to_column": lcl_id_to_column,
        "slot": lcl_local_slot,
        "dow": lcl_local_dow,
        "weekend": lcl_local_weekend,
        "month": lcl_local_month,
        "break_after": lcl_compression_break_after,
        "scalers": lcl_scaler_lookup,
        "splits": LCL_SPLITS,
    },
    "CER": {
        "grid": CER_GRID,
        "matrix": cer_matrix,
        "id_to_column": cer_id_to_column,
        "slot": cer_local_slot,
        "dow": cer_local_dow,
        "weekend": cer_local_weekend,
        "month": cer_local_month,
        "break_after": cer_compression_break_after,
        "scalers": cer_scaler_lookup,
        "splits": CER_SPLITS,
    },
}

checked_examples = 0
for (dataset_name, horizon, split_name), examples in (
    representative_examples.items()
):
    arrays = dataset_arrays[dataset_name]
    boundary_positions = np.flatnonzero(arrays["break_after"])

    for example in examples:
        entity_id = example["entity_id"]
        target_timestamp = pd.Timestamp(example["target_timestamp"])

        if entity_id not in arrays["id_to_column"]:
            raise AssertionError(
                f"Spot-check entity {entity_id} is not in the {dataset_name} matrix."
            )
        entity_position = arrays["id_to_column"][entity_id]

        target_position = int(
            arrays["grid"].get_indexer([target_timestamp])[0]
        )
        if target_position < 0:
            raise AssertionError(
                "Spot-check target timestamp is not in the governed grid."
            )

        input_start = (
            target_position
            - horizon
            - (LOOKBACK - 1)
        )
        input_end = target_position - horizon
        input_end_exclusive = input_start + LOOKBACK

        if input_start < 0:
            raise AssertionError("Spot-check produced a negative input start.")
        if input_end - input_start != LOOKBACK - 1:
            raise AssertionError("Spot-check input span is not 48 steps.")
        if target_position - input_end != horizon:
            raise AssertionError("Spot-check target offset is incorrect.")

        for boundary_position in boundary_positions:
            if input_start <= boundary_position < target_position:
                raise AssertionError(
                    "Spot-check sample crosses a conservative continuity boundary."
                )

        computed_split = None
        for candidate_split, (split_start, split_end) in arrays[
            "splits"
        ].items():
            if split_start <= target_timestamp <= split_end:
                computed_split = candidate_split
                break
        if computed_split != split_name:
            raise AssertionError(
                f"Spot-check split mismatch: recorded {split_name}, "
                f"computed {computed_split}."
            )

        load_window = arrays["matrix"][
            input_start:input_end_exclusive,
            entity_position,
        ]
        raw_target = arrays["matrix"][
            target_position,
            entity_position,
        ]

        if load_window.shape != (LOOKBACK,):
            raise AssertionError("Spot-check load window is not length 48.")
        if not np.all(np.isfinite(load_window)):
            raise AssertionError("Spot-check input window contains missing values.")
        if not np.isfinite(raw_target):
            raise AssertionError("Spot-check target is not finite.")

        scaler_row = arrays["scalers"].loc[entity_id]
        mean = float(scaler_row["training_mean_kwh"])
        effective_std = float(
            scaler_row["effective_training_std_kwh"]
        )

        feature_tensor = make_features(
            local_slot_window=arrays["slot"][
                input_start:input_end_exclusive
            ],
            local_dow_window=arrays["dow"][
                input_start:input_end_exclusive
            ],
            local_weekend_window=arrays["weekend"][
                input_start:input_end_exclusive
            ],
            local_month_window=arrays["month"][
                input_start:input_end_exclusive
            ],
            load_window=load_window,
            mean=mean,
            effective_std=effective_std,
        )

        if feature_tensor.shape != (LOOKBACK, len(FEATURE_ORDER)):
            raise AssertionError(
                "Spot-check feature tensor has the wrong shape."
            )
        if not np.all(np.isfinite(feature_tensor)):
            raise AssertionError(
                "Spot-check feature tensor contains NaN or infinity."
            )

        scaled_target = (raw_target - mean) / effective_std
        recovered_target = scaled_target * effective_std + mean
        if abs(recovered_target - raw_target) > 1e-9:
            raise AssertionError(
                "Spot-check inverse-scaling round-trip failed."
            )

        checked_examples += 1

if len(representative_examples) != 18:
    raise AssertionError(
        f"Representative checks covered {len(representative_examples)} "
        "strata; expected 18."
    )

print(
    f"Checked {checked_examples:,} representative rows across all 18 "
    "dataset × horizon × split strata."
)
print(
    "Target offset, input length, split, continuity, finite input/target, "
    "feature shape and inverse scaling: ALL PASS"
)


# ============================================================
# STEP 8 — Write and verify all temporary small outputs
# ============================================================
print("\n" + "=" * 78)
print("STEP 8 — WRITE AND VERIFY TEMPORARY SMALL OUTPUTS")
print("=" * 78)

entity_summary = (
    entity_summary.sort_values(
        ["dataset", "entity_id", "horizon", "split"]
    )
    .reset_index(drop=True)
)
aggregate_summary = (
    aggregate_summary.sort_values(
        ["dataset", "horizon", "split"]
    )
    .reset_index(drop=True)
)

lcl_scaler_table.to_csv(TMP_LCL_SCALER, index=False)
cer_scaler_table.to_csv(TMP_CER_SCALER, index=False)
entity_summary.to_csv(TMP_ENTITY_SUMMARY, index=False)
aggregate_summary.to_csv(TMP_AGGREGATE_SUMMARY, index=False)

# Read-back row-count checks for every temporary CSV artifact.
temp_csv_expectations = {
    TMP_LCL_SCALER: EXPECTED_LCL_FINAL,
    TMP_CER_SCALER: EXPECTED_CER_METERS * 2,
    TMP_ENTITY_SUMMARY: expected_entity_summary_rows,
    TMP_AGGREGATE_SUMMARY: 18,
}
for temp_path, expected_rows in temp_csv_expectations.items():
    if not temp_path.exists() or temp_path.stat().st_size == 0:
        raise AssertionError(
            f"Temporary CSV output is missing or empty: {temp_path}"
        )
    observed_rows = len(pd.read_csv(temp_path))
    if observed_rows != expected_rows:
        raise AssertionError(
            f"{temp_path.name}: read-back row count {observed_rows:,} "
            f"does not match expected {expected_rows:,}."
        )
    print(
        f"  {temp_path.name}: {observed_rows:,} rows read back.  PASS"
    )

metadata_lines = [
    "---",
    "task: sample_construction_production_sample_indexes_and_scalers",
    "status: formal",
    "scope: full_population",
    "analysis_status: PASS",
    f"lcl_households: {EXPECTED_LCL_FINAL}",
    f"cer_meters: {EXPECTED_CER_METERS}",
    f"formal_horizons: {HORIZONS}",
    "index_format: parquet_one_file_per_dataset_horizon",
    "scaler_format: csv",
    "entity_id_artifact_type: string_for_both_datasets",
    "lcl_scaler_span: lcl_full_training",
    "cer_scaler_spans: [cer_30day, cer_full_training]",
    f"cer_conservative_continuity_dates: {[str(value) for value in CER_COMPRESSION_BOUNDARY_DATES]}",
    "full_dense_X_y_saved_to_disk: false",
    "streaming_architecture: true",
    "metadata_is_completion_marker: true",
    "---",
    "",
    "# Sample construction — Production Sample Indexes and Scaler Tables",
    "",
    "## Checks",
    "",
    "- Runtime YAML consistency: `PASS`",
    "- Disk-space preflight: `PASS`",
    "- LCL cohort row count, uniqueness and duplicate checks: `PASS`",
    "- LCL chunked float64 matrix load and GMT continuity: `PASS`",
    "- CER batch-scatter total-row and full-grid coverage: `PASS`",
    "- CER within-batch and cross-batch duplicate checks: `PASS`",
    "- CER is_missing / kWh-NaN consistency: `PASS`",
    "- CER within-batch and cross-batch calendar consistency: `PASS`",
    "- Candidate - missing - continuity = valid for all entity cells: `PASS`",
    "- Deterministic candidate totals for all aggregate cells: `PASS`",
    "- Temporary Parquet row counts, schema and row-group constants: `PASS`",
    "- Scaler uniqueness, finite values, ddof and std-floor invariants: `PASS`",
    "- Representative lazy reconstruction across all 18 strata: `PASS`",
    "- No dense X/y tensors were saved: `PASS`",
    "",
    "## Aggregate counts",
    "",
]

for row in aggregate_summary.itertuples(index=False):
    metadata_lines.append(
        f"- {row.dataset} h={int(row.horizon)} {row.split}: "
        f"candidate={int(row.candidate_windows):,}; "
        f"missing_excluded={int(row.missing_input_or_target_excluded):,}; "
        f"continuity_excluded={int(row.continuity_excluded):,}; "
        f"valid={int(row.valid_windows):,}"
    )

metadata_lines.extend(
    [
        "",
        "## Scalers",
        "",
        f"- LCL scaler rows: `{len(lcl_scaler_table)}`",
        f"- LCL std-floor-triggered rows: `{n_floored_lcl}`",
        f"- CER scaler rows: `{len(cer_scaler_table)}`",
        f"- CER std-floor-triggered rows: `{n_floored_cer}`",
        "",
        "## Output files",
        "",
    ]
)

for final_path in ALL_FINAL_OUTPUTS:
    metadata_lines.append(f"- `{final_path.name}`")

metadata_lines.extend(
    [
        "",
        "## Result",
        "",
        "- Sample construction gate: `PASS`",
        "- Full production indexes and scaler tables are complete.",
        "- Model configuration lock remains Model configuration.",
        "- Forecasting model smoke tests remain Model pipeline.",
        "- Formal forecasting runs remain Formal forecasting.",
        "",
    ]
)

TMP_META_FILE.write_text(
    "\n".join(metadata_lines),
    encoding="utf-8",
)
if not TMP_META_FILE.exists() or TMP_META_FILE.stat().st_size == 0:
    raise AssertionError("Temporary Sample construction metadata file was not written.")
print(f"  {TMP_META_FILE.name}: written.  PASS")


# ============================================================
# STEP 9 — Transaction-like finalization
# ============================================================
print("\n" + "=" * 78)
print("STEP 9 — TRANSACTION-LIKE FINALIZATION")
print("=" * 78)

# Metadata is deliberately renamed last and acts as the completion marker.
rename_pairs = [
    *[
        (TMP_INDEX_FILES[key], INDEX_FILES[key])
        for key in sorted(TMP_INDEX_FILES)
    ],
    (TMP_LCL_SCALER, OUT_LCL_SCALER),
    (TMP_CER_SCALER, OUT_CER_SCALER),
    (TMP_ENTITY_SUMMARY, OUT_ENTITY_SUMMARY),
    (TMP_AGGREGATE_SUMMARY, OUT_AGGREGATE_SUMMARY),
    (TMP_META_FILE, OUT_META_FILE),
]

renamed_pairs: list[tuple[Path, Path]] = []
try:
    for temp_path, final_path in rename_pairs:
        if not temp_path.exists():
            raise FileNotFoundError(
                f"Temporary output disappeared before finalization: {temp_path}"
            )
        temp_path.replace(final_path)
        renamed_pairs.append((temp_path, final_path))
        print(f"  finalized: {final_path.name}")
except Exception:
    # Best-effort rollback: restore already-renamed files to temporary
    # names so a partial formal Sample construction state is not left behind.
    for temp_path, final_path in reversed(renamed_pairs):
        try:
            if final_path.exists() and not temp_path.exists():
                final_path.replace(temp_path)
        except Exception:
            pass
    raise

for final_path in ALL_FINAL_OUTPUTS:
    if not final_path.exists() or final_path.stat().st_size == 0:
        raise AssertionError(
            f"Finalized Sample construction output is missing or empty: {final_path}"
        )

print("\nSAMPLE CONSTRUCTION result: PASS")
print("Final outputs:")
for final_path in ALL_FINAL_OUTPUTS:
    print(f"  {final_path}")
