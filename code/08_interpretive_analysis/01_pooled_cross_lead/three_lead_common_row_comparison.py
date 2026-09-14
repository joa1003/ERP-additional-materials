#!/usr/bin/env python3
"""Strict h={1,12,48} common-row comparison.

The analysis aligns target household and target timestamp across all three leads,
audits the fixed forecasting settings, reproduces native pooled metrics, and
reports common-support lead and strategy comparisons. Input timestamps differ by
design because forecast lead is the experimental variable. No training is run.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import os as _erp_os
_ERP_PROJECT_ROOT = Path(_erp_os.environ.get("ERP_PROJECT_ROOT", ".")).expanduser().resolve()
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

LEADS = (1, 12, 48)
STRATEGIES = (
    "direct_transfer",
    "fine_tuning",
    "cer_scratch_limited",
    "cer_scratch_full",
)
SEEDS = (42, 123, 2026, 31415)
PAIRWISE_LEADS = ((1, 12), (12, 48), (1, 48))

EXPECTED_ROWS = {
    1: 4_719_456,
    12: 4_708_511,
    48: 4_672_691,
}
EXPECTED_METERS = 929

CONFIG_PATHS = {
    1: Path("configs/forecasting_h1.yaml"),
    12: Path("configs/forecasting_h12.yaml"),
    48: Path("configs/forecasting_h48.yaml"),
}
EXPECTED_CONFIG_SHA256 = {
    1: "c329772b1f2208c7af0efa45d9e8bacecc65c0485d518647f097e88a1b3823b3",
    12: "c8a805b2e92cdb21c4edf7b8267b8e99a3a095b15a9a6ef939c9a0420e1158e7",
    48: "da0e788fd9b728fd4f8f0026f428dfd29ed4d2b20879a3dbfb06f04ebbeccb95",
}
EXPECTED_INDEX_PATHS = {
    1: {
        "LCL": "outputs/tables/sample_construction/lcl_production_sample_index_h1.parquet",
        "CER": "outputs/tables/sample_construction/cer_production_sample_index_h1.parquet",
    },
    12: {
        "LCL": "outputs/tables/sample_construction/lcl_production_sample_index_h12.parquet",
        "CER": "outputs/tables/sample_construction/cer_production_sample_index_h12.parquet",
    },
    48: {
        "LCL": "outputs/tables/sample_construction/lcl_production_sample_index_h48.parquet",
        "CER": "outputs/tables/sample_construction/cer_production_sample_index_h48.parquet",
    },
}
EXPECTED_CER_INDEX_SPLITS = {
    1: {"train": 16_600_790, "validation": 2_385_672, "test": 4_719_456},
    12: {"train": 16_578_521, "validation": 2_385_672, "test": 4_708_511},
    48: {"train": 16_508_681, "validation": 2_385_672, "test": 4_672_691},
}

KEY_COLUMNS = ("entity_id", "target_timestamp_ns")
PREDICTION_COLUMNS = (
    "entity_id",
    "target_timestamp",
    "actual_kwh",
    "predicted_kwh",
    "error_kwh",
    "absolute_error_kwh",
    "squared_error_kwh2",
    "smape_percent",
    "persistence_kwh",
    "daily_seasonal_naive_kwh",
)
SUPPORT_SIGNATURE_COLUMNS = (
    "entity_id",
    "target_timestamp_ns",
    "actual_kwh",
    "persistence_kwh",
    "daily_seasonal_naive_kwh",
)
FORMAL_METRIC_COLUMNS = {
    "mae_kwh": "mae_kwh",
    "rmse_kwh": "rmse_kwh",
    "smape_percent": "smape_percent",
}

LOAD_BIN_LABELS = {
    0: "actual_eq_0",
    1: "0_lt_actual_le_0_1",
    2: "0_1_lt_actual_le_0_5",
    3: "0_5_lt_actual_le_1_0",
    4: "actual_gt_1_0",
}

SUCCESS_LABELS = {
    "complete",
    "completed",
    "complete_pass",
    "pass",
    "passed",
    "success",
}

FLOAT_TOLERANCE = 2e-6
ROW_DERIVATION_TOLERANCE = 1e-4
METRIC_TOLERANCE = 5e-6
TIE_TOLERANCE = 1e-12

# These fields contain the actual modelling, data-treatment, strategy and
# optimisation settings. They must be exactly equal across all three configs.
CRITICAL_CONFIG_PATHS = (
    "status",
    "valid_for_formal_results",
    "forecast_task.formal_leads",
    "forecast_task.lead_task_type",
    "forecast_task.lookback_steps",
    "forecast_task.input_features",
    "forecast_task.output_steps",
    "forecast_task.input_target_rule",
    "architecture",
    "locked_training_components",
    "data_rules.scaler_tables",
    "data_rules.support_days",
    "data_rules.missing_rule",
    "data_rules.continuity_rule",
    "data_rules.full_dense_tensors_saved_to_disk",
    "data_rules.lazy_reconstruction",
    "execution_profile.formal_seed_policy",
    "execution_profile.final_common_batch_size",
    "execution_profile.num_workers",
    "execution_profile.pin_memory",
    "execution_profile.formal_source_and_scratch_learning_rate",
    "execution_profile.formal_fine_tuning_learning_rate",
    "execution_profile.formal_fine_tuning_scope",
    "execution_profile.fine_tuning_layers",
    "execution_profile.formal_validation_frequency",
    "execution_profile.formal_validation_interval_epochs",
    "execution_profile.formal_validation_scope",
    "execution_profile.formal_max_epochs",
    "execution_profile.conditional_absolute_max_epochs",
    "execution_profile.early_stopping",
    "execution_profile.scheduler",
)

STRATEGY_CONTRASTS = (
    (
        "direct_transfer_minus_cer_scratch_limited",
        "direct_transfer",
        "cer_scratch_limited",
    ),
    ("fine_tuning_minus_direct_transfer", "fine_tuning", "direct_transfer"),
    ("direct_transfer_minus_cer_scratch_full", "direct_transfer", "cer_scratch_full"),
    ("cer_scratch_full_minus_cer_scratch_limited", "cer_scratch_full", "cer_scratch_limited"),
)


@dataclass(frozen=True)
class RunFiles:
    lead: int
    strategy: str
    seed: int
    prediction: Path
    pooled_metrics: Path
    status: Path


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(str(_ERP_PROJECT_ROOT)),
    )
    parser.add_argument(
        "--metric-tolerance",
        type=float,
        default=METRIC_TOLERANCE,
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run config/input/status gates without loading all prediction rows.",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def nested_get(payload: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for key in dotted_path.split("."):
        if not isinstance(current, Mapping) or key not in current:
            raise KeyError(f"Missing config path: {dotted_path}")
        current = current[key]
    return current


def normalise_for_csv(value: Any) -> str:
    if isinstance(value, (dict, list, tuple, bool)) or value is None:
        return canonical_json(value)
    return str(value)


def build_run_files(root: Path) -> dict[tuple[int, str, int], RunFiles]:
    runs: dict[tuple[int, str, int], RunFiles] = {}
    for lead in LEADS:
        for strategy in STRATEGIES:
            for seed in SEEDS:
                relative = Path(f"h{lead}") / strategy / f"seed_{seed}"
                table_dir = root / "outputs/tables/pooled_strategy_evaluation_formal" / relative
                metadata_dir = root / "outputs/metadata/pooled_strategy_evaluation_formal" / relative
                status = metadata_dir / "run_status.json"
                key = (lead, strategy, seed)
                runs[key] = RunFiles(
                    lead=lead,
                    strategy=strategy,
                    seed=seed,
                    prediction=table_dir / "test_predictions.parquet",
                    pooled_metrics=table_dir / "test_pooled_metrics.csv",
                    status=status,
                )
    return runs


def read_status(path: Path) -> dict[str, Any]:
    require_file(path, "formal run status JSON")
    payload = json.loads(path.read_text(encoding="utf-8"))
    label = str(payload.get("status", "")).strip().lower()
    leakage = payload.get("test_rows_used_for_training_or_selection", np.nan)
    if leakage is None:
        leakage = np.nan
    return {
        "status_path": str(path),
        "status_label": str(payload.get("status", "")),
        "status_recognised_as_success": label in SUCCESS_LABELS,
        "config_sha256": str(payload.get("config_sha256", "")),
        "horizon_in_status": payload.get("horizon", np.nan),
        "strategy_in_status": str(payload.get("strategy", "")),
        "seed_in_status": payload.get("seed", np.nan),
        "test_rows_used_for_training_or_selection": leakage,
    }


def smape_percent(actual: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    actual = np.asarray(actual, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    denominator = np.abs(actual) + np.abs(prediction)
    output = np.zeros_like(denominator, dtype=np.float64)
    mask = denominator > 0.0
    output[mask] = (
        200.0
        * np.abs(prediction[mask] - actual[mask])
        / denominator[mask]
    )
    return output


def metric_values(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    error = prediction - actual
    return {
        "mae_kwh": float(np.mean(np.abs(error))),
        "rmse_kwh": float(np.sqrt(np.mean(np.square(error)))),
        "smape_percent": float(np.mean(smape_percent(actual, prediction))),
        "bias_kwh": float(np.mean(error)),
    }


def audit_configs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, dict[str, Any]]]:
    configs: dict[int, dict[str, Any]] = {}
    config_hashes: dict[int, str] = {}

    for lead in LEADS:
        path = root / CONFIG_PATHS[lead]
        require_file(path, f"h={lead} formal config")
        observed_hash = sha256_file(path)
        expected_hash = EXPECTED_CONFIG_SHA256[lead]
        if observed_hash != expected_hash:
            raise AssertionError(
                f"h={lead} config SHA mismatch: observed={observed_hash}; "
                f"expected={expected_hash}; path={path}"
            )
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise AssertionError(f"h={lead} config is not a mapping: {path}")
        configs[lead] = payload
        config_hashes[lead] = observed_hash

    fairness_rows: list[dict[str, Any]] = []
    critical_payload_by_lead: dict[int, dict[str, Any]] = {lead: {} for lead in LEADS}
    for dotted_path in CRITICAL_CONFIG_PATHS:
        values = {lead: nested_get(configs[lead], dotted_path) for lead in LEADS}
        exact_match = values[1] == values[12] == values[48]
        row = {
            "config_path": dotted_path,
            "exact_match_across_leads": exact_match,
            "h1_value": normalise_for_csv(values[1]),
            "h12_value": normalise_for_csv(values[12]),
            "h48_value": normalise_for_csv(values[48]),
        }
        fairness_rows.append(row)
        for lead in LEADS:
            critical_payload_by_lead[lead][dotted_path] = values[lead]
        if not exact_match:
            raise AssertionError(
                f"Critical config mismatch at {dotted_path}: "
                f"h1={values[1]!r}; h12={values[12]!r}; h48={values[48]!r}"
            )

    critical_fingerprints = {
        lead: hashlib.sha256(
            canonical_json(critical_payload_by_lead[lead]).encode("utf-8")
        ).hexdigest()
        for lead in LEADS
    }
    if len(set(critical_fingerprints.values())) != 1:
        raise AssertionError(
            f"Critical config fingerprints differ: {critical_fingerprints}"
        )

    lead_specific_rows: list[dict[str, Any]] = []
    for lead in LEADS:
        config = configs[lead]
        priority = int(nested_get(config, "forecast_task.current_execution_priority"))
        if priority != lead:
            raise AssertionError(
                f"h={lead} current_execution_priority={priority}; expected={lead}"
            )

        sample_indexes = nested_get(config, "data_rules.sample_indexes")
        for dataset in ("LCL", "CER"):
            key = f"{dataset}_h{lead}"
            observed = str(sample_indexes.get(key, ""))
            expected = EXPECTED_INDEX_PATHS[lead][dataset]
            if observed != expected:
                raise AssertionError(
                    f"h={lead} {key} path mismatch: observed={observed}; "
                    f"expected={expected}"
                )
            require_file(root / observed, f"h={lead} {dataset} production index")
            lead_specific_rows.append(
                {
                    "lead": lead,
                    "item": key,
                    "observed": observed,
                    "expected": expected,
                    "status": "PASS",
                }
            )

        lookback = int(nested_get(config, "forecast_task.lookback_steps"))
        input_start = -lead - lookback + 1
        input_end = -lead
        if input_end - input_start + 1 != lookback:
            raise AssertionError(f"h={lead} derived input length is not {lookback}")
        lead_specific_rows.extend(
            [
                {
                    "lead": lead,
                    "item": "config_sha256",
                    "observed": config_hashes[lead],
                    "expected": EXPECTED_CONFIG_SHA256[lead],
                    "status": "PASS",
                },
                {
                    "lead": lead,
                    "item": "current_execution_priority",
                    "observed": priority,
                    "expected": lead,
                    "status": "PASS",
                },
                {
                    "lead": lead,
                    "item": "input_window_offsets",
                    "observed": f"[{input_start},{input_end}]",
                    "expected": f"[{input_start},{input_end}]",
                    "status": "PASS",
                },
                {
                    "lead": lead,
                    "item": "input_window_steps",
                    "observed": lookback,
                    "expected": 48,
                    "status": "PASS" if lookback == 48 else "FAIL",
                },
            ]
        )
        if lookback != 48:
            raise AssertionError(f"h={lead} lookback={lookback}; expected=48")

    fairness = pd.DataFrame(fairness_rows)
    lead_specific = pd.DataFrame(lead_specific_rows)
    return fairness, lead_specific, configs


def audit_cer_index(path: Path, lead: int, deep: bool = True) -> dict[str, Any]:
    require_file(path, f"h={lead} CER production index")
    parquet = pq.ParquetFile(path)
    required = {"dataset", "entity_id", "horizon", "target_timestamp", "split"}
    missing = sorted(required - set(parquet.schema_arrow.names))
    if missing:
        raise AssertionError(f"{path}: missing index columns {missing}")

    expected_total = sum(EXPECTED_CER_INDEX_SPLITS[lead].values())
    observed_total = int(parquet.metadata.num_rows)
    if observed_total != expected_total:
        raise AssertionError(
            f"h={lead} CER index rows={observed_total:,}; expected={expected_total:,}"
        )

    if not deep:
        return {
            "lead": lead,
            "path": str(path),
            "rows": observed_total,
            "meters": np.nan,
            "horizon_values": "metadata_only",
            "train_rows": EXPECTED_CER_INDEX_SPLITS[lead]["train"],
            "validation_rows": EXPECTED_CER_INDEX_SPLITS[lead]["validation"],
            "test_rows": EXPECTED_CER_INDEX_SPLITS[lead]["test"],
            "status": "METADATA_PASS; deep content audit runs inside batch job",
        }

    split_counts: Counter[str] = Counter()
    entities: set[str] = set()
    horizons: set[int] = set()
    datasets: set[str] = set()
    for batch in parquet.iter_batches(
        batch_size=500_000,
        columns=["dataset", "entity_id", "horizon", "split"],
    ):
        frame = batch.to_pandas()
        datasets.update(frame["dataset"].astype(str).str.upper().unique())
        entities.update(frame["entity_id"].astype(str).unique())
        horizons.update(
            pd.to_numeric(frame["horizon"], errors="raise").astype(int).unique()
        )
        split_counts.update(
            frame["split"].astype(str).str.lower().value_counts().to_dict()
        )
    if datasets != {"CER"}:
        raise AssertionError(f"h={lead} CER index dataset values={datasets}")
    if horizons != {lead}:
        raise AssertionError(f"h={lead} CER index horizon values={horizons}")
    if len(entities) != EXPECTED_METERS:
        raise AssertionError(
            f"h={lead} CER index meters={len(entities)}; expected={EXPECTED_METERS}"
        )
    if dict(split_counts) != EXPECTED_CER_INDEX_SPLITS[lead]:
        raise AssertionError(
            f"h={lead} CER index split counts={dict(split_counts)}; "
            f"expected={EXPECTED_CER_INDEX_SPLITS[lead]}"
        )
    return {
        "lead": lead,
        "path": str(path),
        "rows": observed_total,
        "meters": len(entities),
        "horizon_values": ",".join(map(str, sorted(horizons))),
        "train_rows": split_counts["train"],
        "validation_rows": split_counts["validation"],
        "test_rows": split_counts["test"],
        "status": "PASS",
    }


def audit_run_inventory_preflight(
    root: Path,
    runs: Mapping[tuple[int, str, int], RunFiles],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key in sorted(runs):
        run = runs[key]
        require_file(run.prediction, "formal prediction parquet")
        require_file(run.pooled_metrics, "formal pooled metric CSV")
        require_file(run.status, "formal status JSON")

        parquet = pq.ParquetFile(run.prediction)
        missing = [
            column
            for column in PREDICTION_COLUMNS
            if column not in parquet.schema_arrow.names
        ]
        if missing:
            raise AssertionError(f"{run.prediction}: missing columns {missing}")
        observed_rows = int(parquet.metadata.num_rows)
        if observed_rows != EXPECTED_ROWS[run.lead]:
            raise AssertionError(
                f"{run.prediction}: rows={observed_rows:,}; "
                f"expected={EXPECTED_ROWS[run.lead]:,}"
            )

        status = read_status(run.status)
        if not status["status_recognised_as_success"]:
            raise AssertionError(f"{run.status}: status is not successful")
        if int(status["horizon_in_status"]) != run.lead:
            raise AssertionError(f"{run.status}: horizon mismatch")
        if status["strategy_in_status"] != run.strategy:
            raise AssertionError(f"{run.status}: strategy mismatch")
        if int(status["seed_in_status"]) != run.seed:
            raise AssertionError(f"{run.status}: seed mismatch")
        if status["config_sha256"] != EXPECTED_CONFIG_SHA256[run.lead]:
            raise AssertionError(
                f"{run.status}: config SHA mismatch: "
                f"{status['config_sha256']}"
            )
        leakage = status["test_rows_used_for_training_or_selection"]
        if not pd.isna(leakage) and int(leakage) != 0:
            raise AssertionError(f"{run.status}: non-zero test leakage={leakage}")

        pooled = pd.read_csv(run.pooled_metrics)
        if len(pooled) != 1:
            raise AssertionError(f"{run.pooled_metrics}: expected one row")
        pooled_row = pooled.iloc[0]
        if int(pooled_row["horizon"]) != run.lead:
            raise AssertionError(f"{run.pooled_metrics}: horizon mismatch")
        if str(pooled_row["strategy"]) != run.strategy:
            raise AssertionError(f"{run.pooled_metrics}: strategy mismatch")
        if int(pooled_row["seed"]) != run.seed:
            raise AssertionError(f"{run.pooled_metrics}: seed mismatch")
        if int(pooled_row["test_observations"]) != EXPECTED_ROWS[run.lead]:
            raise AssertionError(f"{run.pooled_metrics}: observation count mismatch")

        rows.append(
            {
                "lead": run.lead,
                "strategy": run.strategy,
                "seed": run.seed,
                "prediction_path": str(run.prediction),
                "prediction_rows": observed_rows,
                "pooled_metric_path": str(run.pooled_metrics),
                **status,
                "preflight_status": "PASS",
            }
        )
    if len(rows) != 48:
        raise AssertionError(f"Expected 48 formal CER runs; found {len(rows)}")
    return pd.DataFrame(rows)


def print_preflight_summary(
    fairness: pd.DataFrame,
    lead_specific: pd.DataFrame,
    index_audit: pd.DataFrame,
    inventory: pd.DataFrame,
) -> None:
    critical_hash = hashlib.sha256(
        "\n".join(
            fairness.loc[:, ["config_path", "h1_value"]].astype(str).agg("=".join, axis=1)
        ).encode("utf-8")
    ).hexdigest()
    print("=" * 104)
    print("POOLED CROSS LEAD COMPARISON THREE-LEAD FAIRNESS PREFLIGHT")
    print("=" * 104)
    print(f"Critical config fields audited : {len(fairness)}")
    print("Critical config exact match    : PASS")
    print(f"Critical config fingerprint    : {critical_hash}")
    print("Lead priorities                : h1=1, h12=12, h48=48")
    print("Input steps per lead           : 48, 48, 48")
    print("Formal CER runs                : 48")
    print("Formal status/config/leakage   : PASS")
    print("CER production indexes         : PASS")
    print("Target-support alignment rule  : entity_id + target_timestamp intersection")
    print("Model training                 : none")
    print("Formal predictions modified    : 0")
    print("Preflight result               : COMPLETE_PASS")
    print("=" * 104)


def normalise_prediction_frame(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    result = frame.copy()
    result["entity_id"] = pd.to_numeric(
        result["entity_id"], errors="raise"
    ).astype("int64")
    timestamp = pd.to_datetime(result["target_timestamp"], errors="raise")
    result["target_timestamp_ns"] = timestamp.astype("int64")
    result = result.drop(columns=["target_timestamp"])

    for column in PREDICTION_COLUMNS:
        if column in {"entity_id", "target_timestamp"}:
            continue
        result[column] = pd.to_numeric(
            result[column], errors="raise"
        ).astype("float64")

    if result.isna().any().any():
        missing_columns = result.columns[result.isna().any()].tolist()
        raise AssertionError(
            f"{path}: prediction frame contains missing values in {missing_columns}"
        )

    duplicate_count = int(result.duplicated(list(KEY_COLUMNS)).sum())
    if duplicate_count:
        raise AssertionError(
            f"{path}: {duplicate_count:,} duplicate entity/timestamp keys"
        )

    actual = result["actual_kwh"].to_numpy(dtype=np.float64, copy=False)
    predicted = result["predicted_kwh"].to_numpy(dtype=np.float64, copy=False)
    error = predicted - actual
    checks = {
        "error_kwh": error,
        "absolute_error_kwh": np.abs(error),
        "squared_error_kwh2": np.square(error),
        "smape_percent": smape_percent(actual, predicted),
    }
    for column, recalculated in checks.items():
        stored = result[column].to_numpy(dtype=np.float64, copy=False)
        maximum = float(np.max(np.abs(stored - recalculated), initial=0.0))
        if maximum > ROW_DERIVATION_TOLERANCE:
            raise AssertionError(
                f"{path}: stored {column} differs from recalculation; "
                f"max abs difference={maximum}; tolerance={ROW_DERIVATION_TOLERANCE}"
            )
    return result


def load_prediction(path: Path, expected_rows: int) -> pd.DataFrame:
    require_file(path, "formal prediction parquet")
    parquet = pq.ParquetFile(path)
    missing = [
        column
        for column in PREDICTION_COLUMNS
        if column not in parquet.schema_arrow.names
    ]
    if missing:
        raise AssertionError(f"{path}: missing required columns {missing}")
    row_count = int(parquet.metadata.num_rows)
    if row_count != expected_rows:
        raise AssertionError(
            f"{path}: row count={row_count:,}; expected={expected_rows:,}"
        )
    table = pq.read_table(path, columns=list(PREDICTION_COLUMNS), use_threads=True)
    frame = table.to_pandas(split_blocks=True, self_destruct=True)
    del table
    frame = normalise_prediction_frame(frame, path)
    if len(frame) != expected_rows:
        raise AssertionError(
            f"{path}: loaded rows={len(frame):,}; expected={expected_rows:,}"
        )
    meter_count = int(frame["entity_id"].nunique())
    if meter_count != EXPECTED_METERS:
        raise AssertionError(
            f"{path}: meters={meter_count}; expected={EXPECTED_METERS}"
        )
    return frame


def support_signature(frame: pd.DataFrame) -> dict[str, int]:
    support = frame.loc[:, list(SUPPORT_SIGNATURE_COLUMNS)]
    hashes = pd.util.hash_pandas_object(
        support, index=False, categorize=False
    ).to_numpy(dtype=np.uint64, copy=False)
    salt = np.uint64(0x9E3779B185EBCA87)
    return {
        "support_hash_sum_1": int(hashes.sum(dtype=np.uint64)),
        "support_hash_sum_2": int((hashes * salt).sum(dtype=np.uint64)),
        "support_hash_xor": int(
            np.bitwise_xor.reduce(hashes, initial=np.uint64(0))
        ),
    }


def read_formal_pooled_metric(path: Path, run: RunFiles) -> pd.Series:
    require_file(path, "formal pooled metric CSV")
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise AssertionError(f"{path}: expected one row, found {len(frame)}")
    row = frame.iloc[0]
    if str(row["strategy"]) != run.strategy:
        raise AssertionError(f"{path}: strategy mismatch")
    if int(row["seed"]) != run.seed:
        raise AssertionError(f"{path}: seed mismatch")
    if int(row["horizon"]) != run.lead:
        raise AssertionError(f"{path}: horizon mismatch")
    return row


def verify_native_metrics(
    frame: pd.DataFrame,
    run: RunFiles,
    tolerance: float,
) -> dict[str, Any]:
    formal = read_formal_pooled_metric(run.pooled_metrics, run)
    recalculated = metric_values(
        frame["actual_kwh"].to_numpy(dtype=np.float64, copy=False),
        frame["predicted_kwh"].to_numpy(dtype=np.float64, copy=False),
    )
    record: dict[str, Any] = {
        "lead": run.lead,
        "strategy": run.strategy,
        "seed": run.seed,
        "rows": len(frame),
    }
    maximum_difference = 0.0
    for output_name, formal_column in FORMAL_METRIC_COLUMNS.items():
        observed = recalculated[output_name]
        expected = float(formal[formal_column])
        difference = observed - expected
        maximum_difference = max(maximum_difference, abs(difference))
        record[f"recalculated_{output_name}"] = observed
        record[f"formal_{output_name}"] = expected
        record[f"difference_{output_name}"] = difference
    record["maximum_abs_difference"] = maximum_difference
    if maximum_difference > tolerance:
        raise AssertionError(
            f"h={run.lead} {run.strategy} seed={run.seed}: native pooled "
            f"metric reproduction max difference={maximum_difference}; "
            f"tolerance={tolerance}"
        )
    return record


def load_bin_codes(actual: np.ndarray) -> np.ndarray:
    actual = np.asarray(actual, dtype=np.float64)
    codes = np.full(len(actual), 4, dtype=np.int8)
    codes[actual <= 1.0] = 3
    codes[actual <= 0.5] = 2
    codes[actual <= 0.1] = 1
    codes[actual == 0.0] = 0
    return codes


def timestamp_dimensions(timestamp_ns: np.ndarray) -> pd.DataFrame:
    timestamp = pd.to_datetime(timestamp_ns)
    slot = (timestamp.hour * 2 + timestamp.minute // 30 + 1).astype(np.int16)
    month = timestamp.month.astype(np.int8)
    day_of_week = timestamp.dayofweek.astype(np.int8)
    weekpart = np.where(day_of_week >= 5, "weekend", "weekday")
    date = timestamp.strftime("%Y-%m-%d")
    return pd.DataFrame(
        {
            "slot": slot,
            "month": month,
            "day_of_week": day_of_week,
            "weekpart": weekpart,
            "date": date,
        }
    )


def window_alignment_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    intervals: dict[int, tuple[int, int]] = {}
    for lead in LEADS:
        start = -lead - 47
        end = -lead
        intervals[lead] = (start, end)
        rows.append(
            {
                "lead": lead,
                "hours_ahead": lead / 2.0,
                "input_start_offset_steps": start,
                "input_end_offset_steps": end,
                "input_steps": end - start + 1,
                "target_offset_steps": 0,
                "alignment_key": "entity_id + target_timestamp",
                "formal_window_required": True,
            }
        )
    overlap_rows: list[dict[str, Any]] = []
    for first, second in PAIRWISE_LEADS:
        a_start, a_end = intervals[first]
        b_start, b_end = intervals[second]
        overlap_start = max(a_start, b_start)
        overlap_end = min(a_end, b_end)
        overlap_steps = max(0, overlap_end - overlap_start + 1)
        overlap_rows.append(
            {
                "lead_a": first,
                "lead_b": second,
                "lead_a_window": f"[{a_start},{a_end}]",
                "lead_b_window": f"[{b_start},{b_end}]",
                "overlap_start_offset_steps": overlap_start if overlap_steps else np.nan,
                "overlap_end_offset_steps": overlap_end if overlap_steps else np.nan,
                "overlap_steps": overlap_steps,
                "same_input_timestamps_required": False,
                "reason": "Different input timestamps define the forecast lead; fairness is enforced by equal 48-step size and common target support.",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(overlap_rows)


def support_membership_tables(
    reference_frames: Mapping[int, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    membership: pd.DataFrame | None = None
    for lead in LEADS:
        keys = reference_frames[lead].loc[:, list(KEY_COLUMNS)].copy()
        keys[f"in_h{lead}"] = True
        membership = keys if membership is None else membership.merge(
            keys,
            on=list(KEY_COLUMNS),
            how="outer",
            validate="one_to_one",
            sort=False,
        )
    assert membership is not None
    for lead in LEADS:
        membership[f"in_h{lead}"] = membership[f"in_h{lead}"].fillna(False).astype(bool)
    membership["support_pattern"] = (
        membership["in_h1"].astype(int).astype(str)
        + membership["in_h12"].astype(int).astype(str)
        + membership["in_h48"].astype(int).astype(str)
    )
    pattern = (
        membership.groupby(["support_pattern", "in_h1", "in_h12", "in_h48"], as_index=False)
        .size()
        .rename(columns={"size": "rows"})
        .sort_values("support_pattern")
    )

    common_mask = membership[["in_h1", "in_h12", "in_h48"]].all(axis=1)
    common_keys = membership.loc[common_mask, list(KEY_COLUMNS)].copy()
    common_rows = len(common_keys)
    native = {lead: len(reference_frames[lead]) for lead in LEADS}

    summary = pd.DataFrame(
        [
            {
                "h1_native_rows": native[1],
                "h12_native_rows": native[12],
                "h48_native_rows": native[48],
                "three_way_common_rows": common_rows,
                "common_share_h1": common_rows / native[1],
                "common_share_h12": common_rows / native[12],
                "common_share_h48": common_rows / native[48],
                "h12_not_h1_rows": int((membership["in_h12"] & ~membership["in_h1"]).sum()),
                "h48_not_h1_rows": int((membership["in_h48"] & ~membership["in_h1"]).sum()),
                "h48_not_h12_rows": int((membership["in_h48"] & ~membership["in_h12"]).sum()),
                "h1_not_h12_rows": int((membership["in_h1"] & ~membership["in_h12"]).sum()),
                "h1_not_h48_rows": int((membership["in_h1"] & ~membership["in_h48"]).sum()),
                "h12_not_h48_rows": int((membership["in_h12"] & ~membership["in_h48"]).sum()),
                "alignment_rule": "strict intersection on entity_id + target_timestamp",
            }
        ]
    )
    return summary, pattern, common_keys


def merge_three_frames(frames: Mapping[int, pd.DataFrame]) -> pd.DataFrame:
    columns = [
        "entity_id",
        "target_timestamp_ns",
        "actual_kwh",
        "predicted_kwh",
        "persistence_kwh",
        "daily_seasonal_naive_kwh",
    ]
    renamed: dict[int, pd.DataFrame] = {}
    for lead in LEADS:
        mapping = {
            column: f"{column}_h{lead}"
            for column in columns
            if column not in KEY_COLUMNS
        }
        renamed[lead] = frames[lead].loc[:, columns].rename(columns=mapping)

    merged = renamed[1].merge(
        renamed[12],
        on=list(KEY_COLUMNS),
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    merged = merged.merge(
        renamed[48],
        on=list(KEY_COLUMNS),
        how="inner",
        validate="one_to_one",
        sort=False,
    )

    actual_ref = merged["actual_kwh_h1"].to_numpy(dtype=np.float64, copy=False)
    daily_ref = merged["daily_seasonal_naive_kwh_h1"].to_numpy(
        dtype=np.float64, copy=False
    )
    for lead in (12, 48):
        actual = merged[f"actual_kwh_h{lead}"].to_numpy(dtype=np.float64, copy=False)
        maximum = float(np.max(np.abs(actual - actual_ref), initial=0.0))
        if maximum > FLOAT_TOLERANCE:
            raise AssertionError(
                f"Common-row actual mismatch h1 versus h{lead}; max={maximum}"
            )
        daily = merged[f"daily_seasonal_naive_kwh_h{lead}"].to_numpy(
            dtype=np.float64, copy=False
        )
        maximum_daily = float(np.max(np.abs(daily - daily_ref), initial=0.0))
        if maximum_daily != 0.0:
            raise AssertionError(
                f"Daily seasonal-naive mismatch h1 versus h{lead}; "
                f"max={maximum_daily}"
            )

    h48_persistence = merged["persistence_kwh_h48"].to_numpy(
        dtype=np.float64, copy=False
    )
    h48_daily = merged["daily_seasonal_naive_kwh_h48"].to_numpy(
        dtype=np.float64, copy=False
    )
    h48_difference = float(
        np.max(np.abs(h48_persistence - h48_daily), initial=0.0)
    )
    if h48_difference != 0.0:
        raise AssertionError(
            f"h=48 Persistence != Daily seasonal-naive; max={h48_difference}"
        )
    return merged


def build_work_frame(merged: pd.DataFrame) -> pd.DataFrame:
    actual = merged["actual_kwh_h1"].to_numpy(dtype=np.float64, copy=False)
    frame = pd.DataFrame(
        {
            "entity_id": merged["entity_id"].to_numpy(dtype=np.int64, copy=False),
            "target_timestamp_ns": merged["target_timestamp_ns"].to_numpy(
                dtype=np.int64, copy=False
            ),
            "actual_kwh": actual,
        }
    )
    for lead in LEADS:
        prediction = merged[f"predicted_kwh_h{lead}"].to_numpy(
            dtype=np.float64, copy=False
        )
        error = prediction - actual
        frame[f"prediction_h{lead}_kwh"] = prediction
        frame[f"bias_h{lead}_kwh"] = error
        frame[f"ae_h{lead}_kwh"] = np.abs(error)
        frame[f"se_h{lead}_kwh2"] = np.square(error)
        frame[f"smape_h{lead}_percent"] = smape_percent(actual, prediction)
    dims = timestamp_dimensions(frame["target_timestamp_ns"].to_numpy())
    frame = pd.concat([frame, dims], axis=1)
    frame["load_bin_code"] = load_bin_codes(actual)
    frame["load_bin"] = frame["load_bin_code"].map(LOAD_BIN_LABELS)
    return frame


def common_metric_records(
    work: pd.DataFrame,
    strategy: str,
    seed: int,
) -> list[dict[str, Any]]:
    actual = work["actual_kwh"].to_numpy(dtype=np.float64, copy=False)
    rows: list[dict[str, Any]] = []
    for lead in LEADS:
        prediction = work[f"prediction_h{lead}_kwh"].to_numpy(
            dtype=np.float64, copy=False
        )
        rows.append(
            {
                "strategy": strategy,
                "seed": seed,
                "lead": lead,
                "common_rows": len(work),
                **metric_values(actual, prediction),
            }
        )
    return rows


def pairwise_metric_deltas(common_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (strategy, seed), subset in common_metrics.groupby(
        ["strategy", "seed"], sort=True
    ):
        indexed = subset.set_index("lead")
        for from_lead, to_lead in PAIRWISE_LEADS:
            record = {
                "strategy": strategy,
                "seed": seed,
                "from_lead": from_lead,
                "to_lead": to_lead,
                "comparison": f"h{to_lead}_minus_h{from_lead}",
                "common_rows": int(indexed.loc[from_lead, "common_rows"]),
            }
            for metric in ("mae_kwh", "rmse_kwh", "smape_percent", "bias_kwh"):
                record[f"from_{metric}"] = float(indexed.loc[from_lead, metric])
                record[f"to_{metric}"] = float(indexed.loc[to_lead, metric])
                record[f"delta_{metric}"] = (
                    float(indexed.loc[to_lead, metric])
                    - float(indexed.loc[from_lead, metric])
                )
            rows.append(record)
    return pd.DataFrame(rows)


def row_outcome_counts(difference: np.ndarray) -> dict[str, Any]:
    difference = np.asarray(difference, dtype=np.float64)
    shorter_lower = difference > TIE_TOLERANCE
    longer_lower = difference < -TIE_TOLERANCE
    tie = ~(shorter_lower | longer_lower)
    return {
        "rows": len(difference),
        "shorter_lead_lower_rows": int(shorter_lower.sum()),
        "longer_lead_lower_rows": int(longer_lower.sum()),
        "tie_rows": int(tie.sum()),
        "shorter_lead_lower_share": float(shorter_lower.mean()),
        "longer_lead_lower_share": float(longer_lower.mean()),
        "tie_share": float(tie.mean()),
        "mean_longer_minus_shorter": float(difference.mean()),
        "median_longer_minus_shorter": float(np.median(difference)),
        "q25_longer_minus_shorter": float(np.quantile(difference, 0.25)),
        "q75_longer_minus_shorter": float(np.quantile(difference, 0.75)),
    }


def pairwise_row_outcomes(
    work: pd.DataFrame,
    strategy: str,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shorter, longer in PAIRWISE_LEADS:
        for criterion, prefix in (
            ("absolute_error", "ae"),
            ("squared_error", "se"),
            ("smape", "smape"),
        ):
            unit = "kwh" if prefix == "ae" else "kwh2" if prefix == "se" else "percent"
            shorter_values = work[f"{prefix}_h{shorter}_{unit}"].to_numpy(
                dtype=np.float64, copy=False
            )
            longer_values = work[f"{prefix}_h{longer}_{unit}"].to_numpy(
                dtype=np.float64, copy=False
            )
            rows.append(
                {
                    "strategy": strategy,
                    "seed": seed,
                    "shorter_lead": shorter,
                    "longer_lead": longer,
                    "comparison": f"h{longer}_minus_h{shorter}",
                    "criterion": criterion,
                    **row_outcome_counts(longer_values - shorter_values),
                }
            )
    return rows


def aggregate_group_long(
    work: pd.DataFrame,
    group_columns: Sequence[str],
    group_type: str,
    strategy: str,
    seed: int,
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for lead in LEADS:
        columns = list(group_columns) + [
            "actual_kwh",
            f"bias_h{lead}_kwh",
            f"ae_h{lead}_kwh",
            f"se_h{lead}_kwh2",
            f"smape_h{lead}_percent",
        ]
        frame = work.loc[:, columns]
        grouped = frame.groupby(list(group_columns), observed=True, sort=True)
        result = grouped.agg(
            rows=("actual_kwh", "size"),
            actual_mean_kwh=("actual_kwh", "mean"),
            bias_kwh=(f"bias_h{lead}_kwh", "mean"),
            mae_kwh=(f"ae_h{lead}_kwh", "mean"),
            mean_squared_error_kwh2=(f"se_h{lead}_kwh2", "mean"),
            smape_percent=(f"smape_h{lead}_percent", "mean"),
        ).reset_index()
        result["rmse_kwh"] = np.sqrt(result.pop("mean_squared_error_kwh2"))
        result.insert(0, "lead", lead)
        result.insert(0, "seed", seed)
        result.insert(0, "strategy", strategy)
        result.insert(0, "group_type", group_type)
        outputs.append(result)
    return pd.concat(outputs, ignore_index=True)


def peak_aggregate_long(
    work: pd.DataFrame,
    thresholds: Mapping[str, float],
    strategy: str,
    seed: int,
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for label, threshold in thresholds.items():
        subset = work.loc[work["actual_kwh"] >= threshold].copy()
        if subset.empty:
            raise AssertionError(f"No rows found for peak group {label}")
        subset["peak_group"] = label
        result = aggregate_group_long(
            subset,
            ["peak_group"],
            "peak",
            strategy,
            seed,
        )
        result.insert(
            result.columns.get_loc("peak_group") + 1,
            "threshold_kwh",
            threshold,
        )
        outputs.append(result)
    return pd.concat(outputs, ignore_index=True)


def four_seed_summary(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
    value_columns: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, subset in frame.groupby(list(group_columns), sort=True, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_columns, keys))
        seeds = tuple(sorted(subset["seed"].astype(int).unique().tolist()))
        if seeds != SEEDS:
            raise AssertionError(
                f"Group {record}: seeds={seeds}; expected={SEEDS}"
            )
        for column in value_columns:
            values = pd.to_numeric(subset[column], errors="raise").astype(float)
            record[f"{column}_mean"] = float(values.mean())
            record[f"{column}_sample_std"] = float(values.std(ddof=1))
            record[f"{column}_minimum"] = float(values.min())
            record[f"{column}_maximum"] = float(values.max())
        rows.append(record)
    return pd.DataFrame(rows)


def group_pairwise_deltas(
    group_frame: pd.DataFrame,
    identity_columns: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouping = ["strategy", "seed", *identity_columns]
    for keys, subset in group_frame.groupby(grouping, sort=True, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(grouping, keys))
        indexed = subset.set_index("lead")
        if tuple(sorted(indexed.index.astype(int).tolist())) != LEADS:
            raise AssertionError(f"Group {base}: missing lead")
        for shorter, longer in PAIRWISE_LEADS:
            record = {
                **base,
                "shorter_lead": shorter,
                "longer_lead": longer,
                "comparison": f"h{longer}_minus_h{shorter}",
            }
            for metric in ("mae_kwh", "rmse_kwh", "smape_percent", "bias_kwh"):
                record[f"shorter_{metric}"] = float(indexed.loc[shorter, metric])
                record[f"longer_{metric}"] = float(indexed.loc[longer, metric])
                record[f"delta_{metric}"] = (
                    float(indexed.loc[longer, metric])
                    - float(indexed.loc[shorter, metric])
                )
            rows.append(record)
    return pd.DataFrame(rows)


def baseline_work_from_reference(merged: pd.DataFrame) -> pd.DataFrame:
    actual = merged["actual_kwh_h1"].to_numpy(dtype=np.float64, copy=False)
    frame = pd.DataFrame(
        {
            "entity_id": merged["entity_id"].to_numpy(dtype=np.int64, copy=False),
            "target_timestamp_ns": merged["target_timestamp_ns"].to_numpy(
                dtype=np.int64, copy=False
            ),
            "actual_kwh": actual,
        }
    )
    for lead in LEADS:
        frame[f"persistence_h{lead}_kwh"] = merged[
            f"persistence_kwh_h{lead}"
        ].to_numpy(dtype=np.float64, copy=False)
        frame[f"daily_h{lead}_kwh"] = merged[
            f"daily_seasonal_naive_kwh_h{lead}"
        ].to_numpy(dtype=np.float64, copy=False)
    dims = timestamp_dimensions(frame["target_timestamp_ns"].to_numpy())
    frame = pd.concat([frame, dims], axis=1)
    frame["load_bin_code"] = load_bin_codes(actual)
    frame["load_bin"] = frame["load_bin_code"].map(LOAD_BIN_LABELS)
    return frame


def baseline_metric_records(frame: pd.DataFrame) -> pd.DataFrame:
    actual = frame["actual_kwh"].to_numpy(dtype=np.float64, copy=False)
    rows: list[dict[str, Any]] = []
    for baseline in ("persistence", "daily_seasonal_naive"):
        for lead in LEADS:
            column = f"persistence_h{lead}_kwh" if baseline == "persistence" else f"daily_h{lead}_kwh"
            prediction = frame[column].to_numpy(dtype=np.float64, copy=False)
            rows.append(
                {
                    "baseline": baseline,
                    "lead": lead,
                    "common_rows": len(frame),
                    **metric_values(actual, prediction),
                }
            )
    return pd.DataFrame(rows)


def baseline_pairwise_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for baseline, subset in metrics.groupby("baseline", sort=True):
        indexed = subset.set_index("lead")
        for shorter, longer in PAIRWISE_LEADS:
            record = {
                "baseline": baseline,
                "shorter_lead": shorter,
                "longer_lead": longer,
                "comparison": f"h{longer}_minus_h{shorter}",
            }
            for metric in ("mae_kwh", "rmse_kwh", "smape_percent", "bias_kwh"):
                record[f"shorter_{metric}"] = float(indexed.loc[shorter, metric])
                record[f"longer_{metric}"] = float(indexed.loc[longer, metric])
                record[f"delta_{metric}"] = (
                    float(indexed.loc[longer, metric])
                    - float(indexed.loc[shorter, metric])
                )
            rows.append(record)
    return pd.DataFrame(rows)


def baseline_row_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    actual = frame["actual_kwh"].to_numpy(dtype=np.float64, copy=False)
    rows: list[dict[str, Any]] = []
    for baseline in ("persistence", "daily_seasonal_naive"):
        predictions = {
            lead: frame[
                f"persistence_h{lead}_kwh"
                if baseline == "persistence"
                else f"daily_h{lead}_kwh"
            ].to_numpy(dtype=np.float64, copy=False)
            for lead in LEADS
        }
        for shorter, longer in PAIRWISE_LEADS:
            for criterion, values_by_lead in (
                (
                    "absolute_error",
                    {lead: np.abs(predictions[lead] - actual) for lead in LEADS},
                ),
                (
                    "squared_error",
                    {lead: np.square(predictions[lead] - actual) for lead in LEADS},
                ),
                (
                    "smape",
                    {lead: smape_percent(actual, predictions[lead]) for lead in LEADS},
                ),
            ):
                rows.append(
                    {
                        "baseline": baseline,
                        "shorter_lead": shorter,
                        "longer_lead": longer,
                        "comparison": f"h{longer}_minus_h{shorter}",
                        "criterion": criterion,
                        **row_outcome_counts(
                            values_by_lead[longer] - values_by_lead[shorter]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def baseline_group_metrics(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
    group_type: str,
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    actual = frame["actual_kwh"].to_numpy(dtype=np.float64, copy=False)
    for baseline in ("persistence", "daily_seasonal_naive"):
        for lead in LEADS:
            pred_column = (
                f"persistence_h{lead}_kwh"
                if baseline == "persistence"
                else f"daily_h{lead}_kwh"
            )
            prediction = frame[pred_column].to_numpy(dtype=np.float64, copy=False)
            error = prediction - actual
            work = frame.loc[:, list(group_columns) + ["actual_kwh"]].copy()
            work["bias_kwh"] = error
            work["absolute_error_kwh"] = np.abs(error)
            work["squared_error_kwh2"] = np.square(error)
            work["smape_percent"] = smape_percent(actual, prediction)
            result = work.groupby(
                list(group_columns), observed=True, sort=True
            ).agg(
                rows=("actual_kwh", "size"),
                actual_mean_kwh=("actual_kwh", "mean"),
                bias_kwh=("bias_kwh", "mean"),
                mae_kwh=("absolute_error_kwh", "mean"),
                mean_squared_error_kwh2=("squared_error_kwh2", "mean"),
                smape_percent=("smape_percent", "mean"),
            ).reset_index()
            result["rmse_kwh"] = np.sqrt(result.pop("mean_squared_error_kwh2"))
            result.insert(0, "lead", lead)
            result.insert(0, "baseline", baseline)
            result.insert(0, "group_type", group_type)
            outputs.append(result)
    return pd.concat(outputs, ignore_index=True)


def build_strategy_contrasts(common_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    indexed = common_metrics.set_index(["lead", "seed", "strategy"])
    for lead in LEADS:
        for seed in SEEDS:
            for label, left, right in STRATEGY_CONTRASTS:
                record = {
                    "lead": lead,
                    "seed": seed,
                    "contrast": label,
                    "left_strategy": left,
                    "right_strategy": right,
                    "difference_definition": "left minus right; negative means left has lower error for MAE/RMSE/SMAPE",
                }
                for metric in ("mae_kwh", "rmse_kwh", "smape_percent", "bias_kwh"):
                    left_value = float(indexed.loc[(lead, seed, left), metric])
                    right_value = float(indexed.loc[(lead, seed, right), metric])
                    record[f"left_{metric}"] = left_value
                    record[f"right_{metric}"] = right_value
                    record[f"difference_{metric}"] = left_value - right_value
                rows.append(record)
    return pd.DataFrame(rows)


def meter_direction_counts(meter_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for strategy, subset in meter_summary.groupby("strategy", sort=True):
        for shorter, longer in PAIRWISE_LEADS:
            for criterion, metric in (
                ("MAE", "mae_kwh_mean"),
                ("RMSE", "rmse_kwh_mean"),
                ("SMAPE", "smape_percent_mean"),
            ):
                pivot = subset.pivot(index="entity_id", columns="lead", values=metric)
                difference = pivot[longer] - pivot[shorter]
                rows.append(
                    {
                        "strategy": strategy,
                        "shorter_lead": shorter,
                        "longer_lead": longer,
                        "comparison": f"h{longer}_minus_h{shorter}",
                        "criterion": criterion,
                        "meters": len(difference),
                        "shorter_lead_lower_meters": int((difference > TIE_TOLERANCE).sum()),
                        "longer_lead_lower_meters": int((difference < -TIE_TOLERANCE).sum()),
                        "tie_meters": int((np.abs(difference) <= TIE_TOLERANCE).sum()),
                        "shorter_lead_lower_share": float((difference > TIE_TOLERANCE).mean()),
                        "longer_lead_lower_share": float((difference < -TIE_TOLERANCE).mean()),
                        "median_longer_minus_shorter": float(np.median(difference)),
                        "mean_longer_minus_shorter": float(np.mean(difference)),
                    }
                )
    return pd.DataFrame(rows)


def meter_strategy_contrast_counts(meter_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for lead in LEADS:
        subset = meter_summary.loc[meter_summary["lead"] == lead]
        for label, left, right in STRATEGY_CONTRASTS:
            for criterion, metric in (
                ("MAE", "mae_kwh_mean"),
                ("RMSE", "rmse_kwh_mean"),
                ("SMAPE", "smape_percent_mean"),
            ):
                pivot = subset.pivot(index="entity_id", columns="strategy", values=metric)
                difference = pivot[left] - pivot[right]
                rows.append(
                    {
                        "lead": lead,
                        "contrast": label,
                        "left_strategy": left,
                        "right_strategy": right,
                        "criterion": criterion,
                        "meters": len(difference),
                        "left_lower_meters": int((difference < -TIE_TOLERANCE).sum()),
                        "right_lower_meters": int((difference > TIE_TOLERANCE).sum()),
                        "tie_meters": int((np.abs(difference) <= TIE_TOLERANCE).sum()),
                        "left_lower_share": float((difference < -TIE_TOLERANCE).mean()),
                        "right_lower_share": float((difference > TIE_TOLERANCE).mean()),
                        "median_left_minus_right": float(np.median(difference)),
                        "mean_left_minus_right": float(np.mean(difference)),
                    }
                )
    return pd.DataFrame(rows)


def write_group_outputs(
    frame: pd.DataFrame,
    table_dir: Path,
    base_name: str,
    identity_columns: Sequence[str],
    save_all_as_parquet: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_path = table_dir / f"{base_name}_all_runs.{'parquet' if save_all_as_parquet else 'csv'}"
    if save_all_as_parquet:
        frame.to_parquet(all_path, index=False)
    else:
        frame.to_csv(all_path, index=False)

    value_columns = [
        "rows",
        "actual_mean_kwh",
        "bias_kwh",
        "mae_kwh",
        "rmse_kwh",
        "smape_percent",
    ]
    summary = four_seed_summary(
        frame,
        ["strategy", "lead", *identity_columns],
        value_columns,
    )
    summary.to_csv(
        table_dir / f"{base_name}_four_seed_mean_std.csv",
        index=False,
    )
    deltas = group_pairwise_deltas(frame, identity_columns)
    deltas.to_csv(
        table_dir / f"{base_name}_pairwise_lead_deltas_all_runs.csv",
        index=False,
    )
    delta_value_columns = [
        "delta_mae_kwh",
        "delta_rmse_kwh",
        "delta_smape_percent",
        "delta_bias_kwh",
    ]
    delta_summary = four_seed_summary(
        deltas,
        ["strategy", *identity_columns, "shorter_lead", "longer_lead", "comparison"],
        delta_value_columns,
    )
    delta_summary.to_csv(
        table_dir / f"{base_name}_pairwise_lead_deltas_four_seed_mean_std.csv",
        index=False,
    )
    return summary, delta_summary


def plot_metric_by_lead(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for strategy in STRATEGIES:
        subset = summary.loc[summary["strategy"] == strategy].sort_values("lead")
        ax.errorbar(
            subset["lead"],
            subset[f"{metric}_mean"],
            yerr=subset[f"{metric}_sample_std"],
            marker="o",
            linewidth=1.5,
            capsize=3,
            label=strategy.replace("_", " ").title(),
        )
    ax.set_xticks(list(LEADS))
    ax.set_xticklabels(["h=1", "h=12", "h=48"])
    ax.set_xlabel("Forecast lead (half-hour steps)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Strict common-row {ylabel} by forecast lead")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_transfer_contrast(
    contrast_summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    path: Path,
) -> None:
    subset = contrast_summary.loc[
        contrast_summary["contrast"] == "direct_transfer_minus_cer_scratch_limited"
    ].sort_values("lead")
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.errorbar(
        subset["lead"],
        subset[f"difference_{metric}_mean"],
        yerr=subset[f"difference_{metric}_sample_std"],
        marker="o",
        linewidth=1.5,
        capsize=3,
    )
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xticks(list(LEADS))
    ax.set_xticklabels(["h=1", "h=12", "h=48"])
    ax.set_xlabel("Forecast lead (half-hour steps)")
    ax.set_ylabel(f"Direct Transfer minus Limited: {ylabel}")
    ax.set_title("Strict common-row transfer contrast by forecast lead")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_summary_markdown(
    path: Path,
    support_summary: pd.DataFrame,
    common_summary: pd.DataFrame,
    delta_summary: pd.DataFrame,
    contrast_summary: pd.DataFrame,
    config_fingerprint: str,
) -> None:
    support = support_summary.iloc[0]
    display_metrics = common_summary.loc[
        :,
        [
            "strategy",
            "lead",
            "mae_kwh_mean",
            "mae_kwh_sample_std",
            "rmse_kwh_mean",
            "rmse_kwh_sample_std",
            "smape_percent_mean",
            "smape_percent_sample_std",
        ],
    ].sort_values(["strategy", "lead"])
    display_deltas = delta_summary.loc[
        :,
        [
            "strategy",
            "comparison",
            "delta_mae_kwh_mean",
            "delta_rmse_kwh_mean",
            "delta_smape_percent_mean",
        ],
    ].sort_values(["strategy", "comparison"])
    display_contrast = contrast_summary.loc[
        :,
        [
            "lead",
            "contrast",
            "difference_mae_kwh_mean",
            "difference_rmse_kwh_mean",
            "difference_smape_percent_mean",
        ],
    ].sort_values(["contrast", "lead"])

    lines = [
        "# Pooled cross lead comparison h={1,12,48} strict common-row comparison",
        "",
        "## Status",
        "",
        "```text",
        "COMPLETE — PASS",
        "```",
        "",
        "## Fairness design",
        "",
        "All critical modelling, data-treatment, strategy and optimisation settings were required to match exactly across the three formal configs. The forecast lead, governed lead-specific index path, expected count assertions and output labels were allowed to differ.",
        "",
        f"Critical-setting fingerprint: `{config_fingerprint}`",
        "",
        "Every model uses a 48-step input and the same eight features. Input timestamps are intentionally different because they define h=1, h=12 and h=48. The comparison aligns `entity_id + target_timestamp`; it does not force identical input timestamps, which would change the forecast tasks.",
        "",
        "## Strict common support",
        "",
        f"- h=1 native rows: {int(support['h1_native_rows']):,}",
        f"- h=12 native rows: {int(support['h12_native_rows']):,}",
        f"- h=48 native rows: {int(support['h48_native_rows']):,}",
        f"- three-way common rows: {int(support['three_way_common_rows']):,}",
        f"- common share of h=1: {float(support['common_share_h1']):.6%}",
        f"- common share of h=12: {float(support['common_share_h12']):.6%}",
        f"- common share of h=48: {float(support['common_share_h48']):.6%}",
        "",
        "## Four-seed common-row metrics",
        "",
        "```text",
        display_metrics.to_string(index=False),
        "```",
        "",
        "## Pairwise lead changes",
        "",
        "Positive MAE/RMSE/SMAPE deltas mean the longer lead had higher error.",
        "",
        "```text",
        display_deltas.to_string(index=False),
        "```",
        "",
        "## Strategy contrasts on the same three-way support",
        "",
        "Differences are left strategy minus right strategy. Negative MAE/RMSE/SMAPE values mean the left strategy had lower error.",
        "",
        "```text",
        display_contrast.to_string(index=False),
        "```",
        "",
        "## Protection gates",
        "",
        "- Formal prediction files modified: 0",
        "- Model training performed: 0",
        "- Hyperparameters changed: 0",
        "- Exact critical-config equality: PASS",
        "- Expected formal config SHA for each lead: PASS",
        "- Same 48-step input length: PASS",
        "- Same target rows across all reported lead comparisons: PASS",
        "- Common-row actual identity: PASS",
        "- Common-row Daily seasonal-naive identity: PASS",
        "- h=48 Persistence/Daily-naive exact identity: PASS",
        "- Native formal pooled metrics reproduced: PASS",
        "- Test leakage markers: zero",
        "",
        "## Evidence boundary",
        "",
        "Native-support results remain the primary formal result for each lead. This strict common-row analysis isolates forecast-lead differences from changes in meter/timestamp support. It does not make the three inputs identical, because doing so would remove or redefine the lead difference.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_preflight(
    root: Path,
    deep_index_audit: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fairness, lead_specific, configs = audit_configs(root)
    index_rows = []
    for lead in LEADS:
        cer_path = root / EXPECTED_INDEX_PATHS[lead]["CER"]
        index_rows.append(audit_cer_index(cer_path, lead, deep=deep_index_audit))
    index_audit = pd.DataFrame(index_rows)
    runs = build_run_files(root)
    inventory = audit_run_inventory_preflight(root, runs)
    print_preflight_summary(fairness, lead_specific, index_audit, inventory)
    return fairness, lead_specific, index_audit, inventory


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()

    fairness, lead_specific, index_audit, preflight_inventory = run_preflight(
        root, deep_index_audit=not args.preflight_only
    )
    if args.preflight_only:
        return

    table_dir = root / "outputs/tables/pooled_strategy_evaluation_3c_three_lead_common_row_comparison"
    metadata_dir = root / "outputs/metadata/pooled_strategy_evaluation_3c_three_lead_common_row_comparison"
    status_path = metadata_dir / "pooled_strategy_evaluation_3c_three_lead_common_row_comparison_status.json"

    existing = [path for path in (table_dir, metadata_dir) if path.exists()]
    if existing:
        raise FileExistsError(
            "Pooled cross lead comparison output path already exists; refusing overwrite: "
            + ", ".join(map(str, existing))
        )
    table_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)

    fairness.to_csv(table_dir / "pooled_strategy_evaluation_3c_config_fairness_audit.csv", index=False)
    lead_specific.to_csv(
        table_dir / "pooled_strategy_evaluation_3c_lead_specific_config_audit.csv", index=False
    )
    index_audit.to_csv(
        table_dir / "pooled_strategy_evaluation_3c_cer_production_index_audit.csv", index=False
    )
    preflight_inventory.to_csv(
        table_dir / "pooled_strategy_evaluation_3c_preflight_formal_run_inventory.csv", index=False
    )

    config_fingerprint = hashlib.sha256(
        canonical_json(
            {
                row["config_path"]: row["h1_value"]
                for row in fairness.to_dict(orient="records")
            }
        ).encode("utf-8")
    ).hexdigest()

    window_table, overlap_table = window_alignment_tables()
    window_table.to_csv(
        table_dir / "pooled_strategy_evaluation_3c_window_alignment_definition.csv", index=False
    )
    overlap_table.to_csv(
        table_dir / "pooled_strategy_evaluation_3c_window_overlap_audit.csv", index=False
    )

    status: dict[str, Any] = {
        "analysis": "Pooled cross lead comparison h={1,12,48} strict common-row comparison",
        "status": "running",
        "started_utc": now_utc(),
        "project_root": str(root),
        "python": sys.version,
        "platform": platform.platform(),
        "critical_config_fingerprint": config_fingerprint,
        "formal_predictions_modified": 0,
        "model_training_performed": 0,
        "hyperparameters_changed": 0,
    }
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    try:
        runs = build_run_files(root)
        if len(runs) != 48:
            raise AssertionError(f"Expected 48 CER runs; found {len(runs)}")

        # Establish native support from the same reference strategy/seed at all leads.
        reference_frames: dict[int, pd.DataFrame] = {}
        reference_signatures: dict[int, dict[str, int]] = {}
        for lead in LEADS:
            print(f"Loading reference h={lead} Direct Transfer seed 42...", flush=True)
            run = runs[(lead, "direct_transfer", 42)]
            frame = load_prediction(run.prediction, EXPECTED_ROWS[lead])
            reference_frames[lead] = frame
            reference_signatures[lead] = support_signature(frame)

        support_summary, support_patterns, common_keys = support_membership_tables(
            reference_frames
        )
        support_summary.to_csv(
            table_dir / "pooled_strategy_evaluation_3c_common_support_summary.csv", index=False
        )
        support_patterns.to_csv(
            table_dir / "pooled_strategy_evaluation_3c_support_membership_patterns.csv", index=False
        )
        common_keys_output = common_keys.copy()
        common_keys_output["target_timestamp"] = pd.to_datetime(
            common_keys_output.pop("target_timestamp_ns")
        )
        common_keys_output.sort_values(
            ["entity_id", "target_timestamp"]
        ).to_parquet(
            table_dir / "pooled_strategy_evaluation_3c_three_way_common_keys.parquet", index=False
        )

        reference_merged = merge_three_frames(reference_frames)
        expected_common_rows = int(support_summary.iloc[0]["three_way_common_rows"])
        if len(reference_merged) != expected_common_rows:
            raise AssertionError(
                f"Reference triple merge rows={len(reference_merged):,}; "
                f"support intersection={expected_common_rows:,}"
            )
        actual_common = reference_merged["actual_kwh_h1"].to_numpy(
            dtype=np.float64, copy=False
        )
        peak_thresholds = {
            "top_10_percent": float(np.quantile(actual_common, 0.90)),
            "top_5_percent": float(np.quantile(actual_common, 0.95)),
            "top_1_percent": float(np.quantile(actual_common, 0.99)),
        }
        pd.DataFrame(
            [
                {"peak_group": label, "threshold_kwh": threshold}
                for label, threshold in peak_thresholds.items()
            ]
        ).to_csv(
            table_dir / "pooled_strategy_evaluation_3c_common_support_peak_thresholds.csv",
            index=False,
        )

        baseline_frame = baseline_work_from_reference(reference_merged)
        baseline_metrics = baseline_metric_records(baseline_frame)
        baseline_metrics.to_csv(
            table_dir / "pooled_strategy_evaluation_3c_baseline_common_row_metrics.csv", index=False
        )
        baseline_pairwise_deltas(baseline_metrics).to_csv(
            table_dir / "pooled_strategy_evaluation_3c_baseline_pairwise_lead_deltas.csv", index=False
        )
        baseline_row_outcomes(baseline_frame).to_csv(
            table_dir / "pooled_strategy_evaluation_3c_baseline_paired_row_outcomes.csv", index=False
        )
        for group_type, columns in (
            ("load_bin", ["load_bin_code", "load_bin"]),
            ("slot", ["slot"]),
            ("month", ["month"]),
            ("weekpart", ["weekpart"]),
            ("meter", ["entity_id"]),
        ):
            base_group = baseline_group_metrics(baseline_frame, columns, group_type)
            suffix = "parquet" if group_type == "meter" else "csv"
            path = table_dir / f"pooled_strategy_evaluation_3c_baseline_{group_type}_metrics.{suffix}"
            if suffix == "parquet":
                base_group.to_parquet(path, index=False)
            else:
                base_group.to_csv(path, index=False)

        baseline_peak_outputs: list[pd.DataFrame] = []
        for peak_group, threshold in peak_thresholds.items():
            peak_subset = baseline_frame.loc[
                baseline_frame["actual_kwh"] >= threshold
            ].copy()
            peak_subset["peak_group"] = peak_group
            peak_metrics = baseline_group_metrics(
                peak_subset, ["peak_group"], "peak"
            )
            peak_metrics.insert(
                peak_metrics.columns.get_loc("peak_group") + 1,
                "threshold_kwh",
                threshold,
            )
            baseline_peak_outputs.append(peak_metrics)
        pd.concat(baseline_peak_outputs, ignore_index=True).to_csv(
            table_dir / "pooled_strategy_evaluation_3c_baseline_peak_metrics.csv",
            index=False,
        )

        # h=48 baseline identity is checked on both native and common support.
        native_h48_persistence = reference_frames[48]["persistence_kwh"].to_numpy(
            dtype=np.float64, copy=False
        )
        native_h48_daily = reference_frames[48][
            "daily_seasonal_naive_kwh"
        ].to_numpy(dtype=np.float64, copy=False)
        native_h48_max_difference = float(
            np.max(np.abs(native_h48_persistence - native_h48_daily), initial=0.0)
        )
        if native_h48_max_difference != 0.0:
            raise AssertionError(
                f"Native h=48 baseline identity failed: {native_h48_max_difference}"
            )
        pd.DataFrame(
            [
                {
                    "scope": "native_h48",
                    "rows": len(reference_frames[48]),
                    "differing_rows": int(
                        np.count_nonzero(native_h48_persistence != native_h48_daily)
                    ),
                    "maximum_abs_difference_kwh": native_h48_max_difference,
                    "status": "PASS",
                },
                {
                    "scope": "three_way_common_h48",
                    "rows": len(baseline_frame),
                    "differing_rows": int(
                        np.count_nonzero(
                            baseline_frame["persistence_h48_kwh"].to_numpy()
                            != baseline_frame["daily_h48_kwh"].to_numpy()
                        )
                    ),
                    "maximum_abs_difference_kwh": float(
                        np.max(
                            np.abs(
                                baseline_frame["persistence_h48_kwh"].to_numpy()
                                - baseline_frame["daily_h48_kwh"].to_numpy()
                            ),
                            initial=0.0,
                        )
                    ),
                    "status": "PASS",
                },
            ]
        ).to_csv(
            table_dir / "pooled_strategy_evaluation_3c_h48_baseline_identity_audit.csv",
            index=False,
        )

        del baseline_frame, actual_common
        gc.collect()

        cached_frames: dict[tuple[int, str, int], pd.DataFrame] = {
            (lead, "direct_transfer", 42): reference_frames[lead]
            for lead in LEADS
        }

        inventory_rows: list[dict[str, Any]] = []
        native_metric_rows: list[dict[str, Any]] = []
        common_metric_rows: list[dict[str, Any]] = []
        outcome_rows: list[dict[str, Any]] = []
        load_bin_frames: list[pd.DataFrame] = []
        peak_frames: list[pd.DataFrame] = []
        slot_frames: list[pd.DataFrame] = []
        month_frames: list[pd.DataFrame] = []
        weekpart_frames: list[pd.DataFrame] = []
        meter_frames: list[pd.DataFrame] = []

        pair_number = 0
        total_pairs = len(STRATEGIES) * len(SEEDS)
        for strategy in STRATEGIES:
            for seed in SEEDS:
                pair_number += 1
                print(
                    f"[{pair_number:02d}/{total_pairs}] {strategy} seed={seed}",
                    flush=True,
                )
                frames: dict[int, pd.DataFrame] = {}
                for lead in LEADS:
                    key = (lead, strategy, seed)
                    run = runs[key]
                    frame = cached_frames.pop(key, None)
                    if frame is None:
                        frame = load_prediction(run.prediction, EXPECTED_ROWS[lead])
                    signature = support_signature(frame)
                    if signature != reference_signatures[lead]:
                        raise AssertionError(
                            f"h={lead} {strategy} seed={seed}: support differs "
                            "from lead reference"
                        )
                    native_metric_rows.append(
                        verify_native_metrics(
                            frame,
                            run,
                            tolerance=args.metric_tolerance,
                        )
                    )
                    status_record = read_status(run.status)
                    if not status_record["status_recognised_as_success"]:
                        raise AssertionError(f"{run.status}: unsuccessful status")
                    leakage = status_record[
                        "test_rows_used_for_training_or_selection"
                    ]
                    if not pd.isna(leakage) and int(leakage) != 0:
                        raise AssertionError(
                            f"{run.status}: non-zero test leakage={leakage}"
                        )
                    if status_record["config_sha256"] != EXPECTED_CONFIG_SHA256[lead]:
                        raise AssertionError(f"{run.status}: config SHA mismatch")
                    inventory_rows.append(
                        {
                            "lead": lead,
                            "strategy": strategy,
                            "seed": seed,
                            "prediction_path": str(run.prediction),
                            "pooled_metric_path": str(run.pooled_metrics),
                            "prediction_rows": len(frame),
                            "meters": int(frame["entity_id"].nunique()),
                            **signature,
                            **status_record,
                        }
                    )
                    frames[lead] = frame

                merged = merge_three_frames(frames)
                if len(merged) != expected_common_rows:
                    raise AssertionError(
                        f"{strategy} seed={seed}: common rows={len(merged):,}; "
                        f"reference={expected_common_rows:,}"
                    )
                work = build_work_frame(merged)
                common_metric_rows.extend(common_metric_records(work, strategy, seed))
                outcome_rows.extend(pairwise_row_outcomes(work, strategy, seed))
                load_bin_frames.append(
                    aggregate_group_long(
                        work,
                        ["load_bin_code", "load_bin"],
                        "load_bin",
                        strategy,
                        seed,
                    )
                )
                peak_frames.append(
                    peak_aggregate_long(work, peak_thresholds, strategy, seed)
                )
                slot_frames.append(
                    aggregate_group_long(work, ["slot"], "slot", strategy, seed)
                )
                month_frames.append(
                    aggregate_group_long(work, ["month"], "month", strategy, seed)
                )
                weekpart_frames.append(
                    aggregate_group_long(
                        work, ["weekpart"], "weekpart", strategy, seed
                    )
                )
                meter_frames.append(
                    aggregate_group_long(
                        work, ["entity_id"], "meter", strategy, seed
                    )
                )

                del work, merged
                for lead in LEADS:
                    del frames[lead]
                gc.collect()

        inventory = pd.DataFrame(inventory_rows)
        inventory.to_csv(
            table_dir / "pooled_strategy_evaluation_3c_formal_run_inventory.csv", index=False
        )
        native_metrics = pd.DataFrame(native_metric_rows)
        native_metrics.to_csv(
            table_dir / "pooled_strategy_evaluation_3c_native_metric_reproduction.csv", index=False
        )

        common_metrics = pd.DataFrame(common_metric_rows)
        common_metrics.to_csv(
            table_dir / "pooled_strategy_evaluation_3c_common_row_metrics_all_runs.csv", index=False
        )
        common_summary = four_seed_summary(
            common_metrics,
            ["strategy", "lead"],
            ["common_rows", "mae_kwh", "rmse_kwh", "smape_percent", "bias_kwh"],
        )
        common_summary.to_csv(
            table_dir / "pooled_strategy_evaluation_3c_common_row_metrics_four_seed_mean_std.csv",
            index=False,
        )

        lead_deltas = pairwise_metric_deltas(common_metrics)
        lead_deltas.to_csv(
            table_dir / "pooled_strategy_evaluation_3c_pairwise_lead_deltas_all_runs.csv", index=False
        )
        lead_delta_summary = four_seed_summary(
            lead_deltas,
            ["strategy", "from_lead", "to_lead", "comparison"],
            [
                "common_rows",
                "delta_mae_kwh",
                "delta_rmse_kwh",
                "delta_smape_percent",
                "delta_bias_kwh",
            ],
        )
        lead_delta_summary.to_csv(
            table_dir / "pooled_strategy_evaluation_3c_pairwise_lead_deltas_four_seed_mean_std.csv",
            index=False,
        )

        outcomes = pd.DataFrame(outcome_rows)
        outcomes.to_csv(
            table_dir / "pooled_strategy_evaluation_3c_paired_row_outcomes_all_runs.csv", index=False
        )
        outcome_summary = four_seed_summary(
            outcomes,
            [
                "strategy",
                "shorter_lead",
                "longer_lead",
                "comparison",
                "criterion",
            ],
            [
                "rows",
                "shorter_lead_lower_share",
                "longer_lead_lower_share",
                "tie_share",
                "mean_longer_minus_shorter",
                "median_longer_minus_shorter",
                "q25_longer_minus_shorter",
                "q75_longer_minus_shorter",
            ],
        )
        outcome_summary.to_csv(
            table_dir / "pooled_strategy_evaluation_3c_paired_row_outcomes_four_seed_mean_std.csv",
            index=False,
        )

        strategy_contrasts = build_strategy_contrasts(common_metrics)
        strategy_contrasts.to_csv(
            table_dir / "pooled_strategy_evaluation_3c_strategy_contrasts_all_runs.csv", index=False
        )
        contrast_summary = four_seed_summary(
            strategy_contrasts,
            [
                "lead",
                "contrast",
                "left_strategy",
                "right_strategy",
                "difference_definition",
            ],
            [
                "difference_mae_kwh",
                "difference_rmse_kwh",
                "difference_smape_percent",
                "difference_bias_kwh",
            ],
        )
        contrast_summary.to_csv(
            table_dir / "pooled_strategy_evaluation_3c_strategy_contrasts_four_seed_mean_std.csv",
            index=False,
        )

        load_bin_all = pd.concat(load_bin_frames, ignore_index=True)
        peak_all = pd.concat(peak_frames, ignore_index=True)
        slot_all = pd.concat(slot_frames, ignore_index=True)
        month_all = pd.concat(month_frames, ignore_index=True)
        weekpart_all = pd.concat(weekpart_frames, ignore_index=True)
        meter_all = pd.concat(meter_frames, ignore_index=True)

        write_group_outputs(
            load_bin_all,
            table_dir,
            "pooled_strategy_evaluation_3c_load_bin_metrics",
            ["load_bin_code", "load_bin"],
        )
        write_group_outputs(
            peak_all,
            table_dir,
            "pooled_strategy_evaluation_3c_peak_metrics",
            ["peak_group", "threshold_kwh"],
        )
        write_group_outputs(
            slot_all,
            table_dir,
            "pooled_strategy_evaluation_3c_slot_metrics",
            ["slot"],
        )
        write_group_outputs(
            month_all,
            table_dir,
            "pooled_strategy_evaluation_3c_month_metrics",
            ["month"],
        )
        write_group_outputs(
            weekpart_all,
            table_dir,
            "pooled_strategy_evaluation_3c_weekpart_metrics",
            ["weekpart"],
        )
        meter_summary, meter_delta_summary = write_group_outputs(
            meter_all,
            table_dir,
            "pooled_strategy_evaluation_3c_meter_metrics",
            ["entity_id"],
            save_all_as_parquet=True,
        )
        meter_direction_counts(meter_summary).to_csv(
            table_dir / "pooled_strategy_evaluation_3c_meter_lead_direction_counts.csv",
            index=False,
        )
        meter_strategy_contrast_counts(meter_summary).to_csv(
            table_dir / "pooled_strategy_evaluation_3c_meter_strategy_contrast_counts.csv",
            index=False,
        )

        write_summary_markdown(
            metadata_dir / "pooled_strategy_evaluation_3c_three_lead_common_row_summary.md",
            support_summary,
            common_summary,
            lead_delta_summary,
            contrast_summary,
            config_fingerprint,
        )

        status.update(
            {
                "status": "complete_pass",
                "completed_utc": now_utc(),
                "formal_runs_audited": len(inventory),
                "strategy_seed_groups": total_pairs,
                "h1_native_rows": int(support_summary.iloc[0]["h1_native_rows"]),
                "h12_native_rows": int(support_summary.iloc[0]["h12_native_rows"]),
                "h48_native_rows": int(support_summary.iloc[0]["h48_native_rows"]),
                "three_way_common_rows": expected_common_rows,
                "native_metric_max_abs_difference": float(
                    native_metrics["maximum_abs_difference"].max()
                ),
                "critical_config_exact_match": True,
                "config_sha_match": True,
                "same_input_length_48": True,
                "common_target_support_enforced": True,
                "actual_common_row_identity": True,
                "daily_naive_common_row_identity": True,
                "h48_persistence_daily_naive_identity": True,
                "formal_predictions_modified": 0,
                "model_training_performed": 0,
                "hyperparameters_changed": 0,
            }
        )
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

        print()
        print("=" * 104)
        print("POOLED CROSS LEAD COMPARISON h={1,12,48} STRICT COMMON-ROW COMPARISON: COMPLETE_PASS")
        print("=" * 104)
        print(support_summary.to_string(index=False))
        print()
        print("Four-seed common-row metrics:")
        print(
            common_summary.loc[
                :,
                [
                    "strategy",
                    "lead",
                    "mae_kwh_mean",
                    "mae_kwh_sample_std",
                    "rmse_kwh_mean",
                    "rmse_kwh_sample_std",
                    "smape_percent_mean",
                    "smape_percent_sample_std",
                ],
            ].to_string(index=False)
        )
        print()
        print("Critical config equality: PASS")
        print("Same 48-step input length: PASS")
        print("Strict common target support: PASS")
        print("Actual and Daily-naive identity: PASS")
        print("h=48 Persistence/Daily-naive identity: PASS")
        print("Formal predictions modified: 0")
        print("Model training performed: 0")
        print(f"Tables   : {table_dir}")
        print(f"Metadata : {metadata_dir}")

    except Exception as exc:
        status.update(
            {
                "status": "failed",
                "failed_utc": now_utc(),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
