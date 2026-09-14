from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from torch import nn


# ============================================================================
# FORMAL FORECASTING — FORMAL FORECASTING RUNNER
#
# Formal h=48 production runner for:
#   - lcl_source
#   - direct_transfer
#   - fine_tuning
#   - cer_scratch_limited
#   - cer_scratch_full
#
# The runner reads the lead-specific formal config configs/forecasting_h48.yaml
# authority. It never uses test rows for training, scheduling, checkpoint
# selection, early stopping, or the 30 -> 45 extension decision.
# ============================================================================

FORMAL_STRATEGIES = {
    "lcl_source",
    "direct_transfer",
    "fine_tuning",
    "cer_scratch_limited",
    "cer_scratch_full",
}

MODEL_TOP_LEVELS = {"conv1", "conv2", "lstm", "fc1", "output"}
EXPECTED_PARAMETER_COUNT = 34_977
LOOKBACK = 48
INPUT_FEATURES = 8
OUTPUT_STEPS = 1

LCL_START = pd.Timestamp("2013-01-01 00:00:00")
LCL_END = pd.Timestamp("2013-12-31 23:30:00")
CER_START = pd.Timestamp("2009-07-14 00:00:00")
CER_END = pd.Timestamp("2010-12-31 23:30:00")

# Exact governed h=48 counts from Sample construction production construction.
EXPECTED_H48_COUNTS: dict[str, dict[str, int]] = {
    "lcl_source": {
        "train": 46_551_002,
        "validation": 6_668_951,
        "test": 13_310_868,
    },
    "direct_transfer": {
        "train": 0,
        "validation": 2_385_672,
        "test": 4_672_691,
    },
    "fine_tuning": {
        "train": 1_294_097,
        "validation": 2_385_672,
        "test": 4_672_691,
    },
    "cer_scratch_limited": {
        "train": 1_294_097,
        "validation": 2_385_672,
        "test": 4_672_691,
    },
    "cer_scratch_full": {
        "train": 16_508_681,
        "validation": 2_385_672,
        "test": 4_672_691,
    },
}

# Explicit optimizer/scheduler defaults matching Model pipeline PyTorch behaviour.
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
ADAM_WEIGHT_DECAY = 0.0
ADAM_AMSGRAD = False
SCHEDULER_THRESHOLD = 1e-4
SCHEDULER_THRESHOLD_MODE = "rel"
SCHEDULER_COOLDOWN = 0
SCHEDULER_EPS = 1e-8
MIXED_PRECISION_ENABLED = False


class UniqueKeyLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise RuntimeError(
                f"Duplicate YAML key {key!r} at line "
                f"{key_node.start_mark.line + 1}."
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pooled strategy summary formal h=48 forecasting runner."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Defaults to <project-root>/configs/forecasting_h48.yaml.",
    )
    parser.add_argument(
        "--strategy",
        choices=sorted(FORMAL_STRATEGIES),
        default=None,
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previously interrupted trainable run from last_state.pt.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate canonical config and required input paths without training.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def heading(text: str) -> None:
    print("\n" + "=" * 104)
    print(text)
    print("=" * 104, flush=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_epoch_seed(seed: int, epoch: int) -> int:
    # Valid deterministic 32-bit seed; same formal seed/epoch gives the same
    # training row-group and within-row-group order across paired strategies.
    return int((int(seed) + 1_000_003 * int(epoch)) % (2**32 - 1))


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Fixed formal runtime policy. This improves repeatability without claiming
    # that separate CUDA executions are guaranteed bitwise identical.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def synchronise(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def clear_device(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def to_ns(values: pd.Series | pd.DatetimeIndex | np.ndarray) -> np.ndarray:
    converted = pd.to_datetime(values, errors="raise")
    return np.asarray(converted, dtype="datetime64[ns]").astype(np.int64)


def atomic_json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv_dump(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


class GlobalCNNLSTM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(8, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.transpose(1, 2)
        sequence, _ = self.lstm(x)
        x = sequence[:, -1, :]
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        return self.output(x)


@dataclass
class CanonicalSettings:
    project_root: Path
    config_path: Path
    config_text: str
    config_sha256: str
    config: dict[str, Any]
    seed: int
    horizon: int
    batch_size: int
    num_workers: int
    pin_memory: bool
    source_scratch_lr: float
    fine_tuning_lr: float
    fine_tuning_scope: str
    fine_tuning_frozen: list[str]
    fine_tuning_trainable: list[str]
    gradient_clip: float
    scheduler_factor: float
    scheduler_patience: int
    scheduler_min_lr: float
    early_stopping_patience: int
    minimum_delta: float
    max_epochs: int
    absolute_max_epochs: int
    lcl_index_path: Path
    cer_index_path: Path
    lcl_scaler_path: Path
    cer_scaler_path: Path


@dataclass
class BaseArrays:
    dataset: str
    grid: pd.DatetimeIndex
    grid_ns: np.ndarray
    entity_ids: list[str]
    entity_to_column: dict[str, int]
    load_matrix: np.ndarray
    calendar_features: np.ndarray


@dataclass
class ScalerArrays:
    name: str
    means: np.ndarray
    stds: np.ndarray
    fit_start: pd.Timestamp | None
    fit_end: pd.Timestamp | None


@dataclass
class StrategySpec:
    name: str
    dataset: str
    index_path: Path
    arrays: BaseArrays
    scalers: ScalerArrays
    train_target_end: pd.Timestamp | None
    learning_rate: float | None
    trainable_layers: list[str]
    initial_state: dict[str, torch.Tensor] | None
    source_checkpoint_path: Path | None


@dataclass
class RunPaths:
    model_dir: Path
    table_dir: Path
    figure_dir: Path
    metadata_dir: Path
    best_checkpoint: Path
    last_state: Path
    epoch_log: Path
    learning_curve: Path
    predictions: Path
    pooled_metrics: Path
    household_metrics: Path
    metadata: Path
    status: Path
    source_reference: Path


def resolve_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def public_repo_path(project_root: Path, value: str | Path | None) -> str | None:
    """Return a repository-relative path for public metadata.

    Runtime computation may use absolute paths internally, but public metadata
    must not persist user-, machine- or account-specific path prefixes.
    """
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        # External runtime executables or files must never expose their full
        # machine-specific path in public metadata.
        return path.name


def public_runtime_text(project_root: Path, value: object) -> str:
    """Redact project-root and home-directory prefixes from diagnostic text."""
    text = str(value)
    replacements = (
        (str(project_root), "${ERP_PROJECT_ROOT}"),
        (str(Path.home()), "$HOME"),
    )
    for prefix, replacement in replacements:
        if prefix:
            text = text.replace(prefix, replacement)
    return text


def load_canonical_settings(
    project_root: Path,
    config_path: Path,
    seed: int,
    horizon: int,
) -> CanonicalSettings:
    if not config_path.exists():
        raise FileNotFoundError(f"Canonical config not found: {config_path}")

    text = config_path.read_text(encoding="utf-8")
    config = yaml.load(text, Loader=UniqueKeyLoader)
    if config.get("status") != "formal_production_ready":
        raise RuntimeError(f"Unexpected config status: {config.get('status')!r}")
    if config.get("valid_for_formal_results") is not True:
        raise RuntimeError("valid_for_formal_results is not true.")

    profile = config["execution_profile"]
    forecast = config["forecast_task"]
    architecture = config["architecture"]
    locked = config["locked_training_components"]

    if profile.get("device") != "cuda":
        raise RuntimeError("Formal forecasting requires a CUDA execution profile.")
    if profile.get("formal_results_allowed") is not True:
        raise RuntimeError("Canonical profile does not allow formal results.")
    if horizon != int(forecast["current_execution_priority"]):
        raise RuntimeError(
            f"This runner is for current priority h={forecast['current_execution_priority']}; "
            f"received h={horizon}."
        )
    if horizon != 48:
        raise RuntimeError("This Pooled strategy summary runner authorises h=48 only.")

    seeds = [int(value) for value in profile["formal_seed_policy"]["seeds"]]
    if seed not in seeds:
        raise RuntimeError(f"Seed {seed} is not in canonical formal seeds {seeds}.")
    if seeds != [42, 123, 2026, 31415]:
        raise RuntimeError(f"Unexpected canonical formal seed set: {seeds}")

    if int(architecture["parameter_count"]) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("Canonical architecture parameter count mismatch.")
    if list(architecture["input_shape"]) != [LOOKBACK, INPUT_FEATURES]:
        raise RuntimeError("Canonical model input shape mismatch.")
    if int(forecast["output_steps"]) != OUTPUT_STEPS:
        raise RuntimeError("Canonical output step count mismatch.")

    if locked["loss"] != "mse_scaled_space":
        raise RuntimeError("Unexpected formal loss.")
    if str(locked["optimizer_family"]).lower() != "adam":
        raise RuntimeError("Unexpected optimizer family.")
    if locked["checkpoint_selection"] != "lowest_full_validation_mse":
        raise RuntimeError("Unexpected checkpoint-selection rule.")
    if locked["checkpoint_reload_required"] is not True:
        raise RuntimeError("Best-checkpoint reload is not required in config.")

    if profile["formal_validation_frequency"] != "every_complete_epoch":
        raise RuntimeError("Formal validation must occur after every complete epoch.")
    if int(profile["formal_validation_interval_epochs"]) != 1:
        raise RuntimeError("Formal validation interval must equal one epoch.")
    if profile["formal_validation_scope"] != "full_validation_split":
        raise RuntimeError("Formal validation scope is not the full split.")
    if profile["formal_fine_tuning_scope"] != "dense_head_only":
        raise RuntimeError("Fine-tuning scope is not dense_head_only.")

    data = config["data_rules"]
    lcl_index = resolve_project_path(
        project_root,
        data["sample_indexes"]["LCL_h48"],
    )
    cer_index = resolve_project_path(
        project_root,
        data["sample_indexes"]["CER_h48"],
    )
    lcl_scaler = resolve_project_path(project_root, data["scaler_tables"]["LCL"])
    cer_scaler = resolve_project_path(project_root, data["scaler_tables"]["CER"])

    settings = CanonicalSettings(
        project_root=project_root,
        config_path=config_path,
        config_text=text,
        config_sha256=sha256_text(text),
        config=config,
        seed=seed,
        horizon=horizon,
        batch_size=int(profile["final_common_batch_size"]),
        num_workers=int(profile["num_workers"]),
        pin_memory=bool(profile["pin_memory"]),
        source_scratch_lr=float(profile["formal_source_and_scratch_learning_rate"]),
        fine_tuning_lr=float(profile["formal_fine_tuning_learning_rate"]),
        fine_tuning_scope=str(profile["formal_fine_tuning_scope"]),
        fine_tuning_frozen=list(profile["fine_tuning_layers"]["frozen"]),
        fine_tuning_trainable=list(profile["fine_tuning_layers"]["trainable"]),
        gradient_clip=float(locked["gradient_clipping_max_norm"]),
        scheduler_factor=float(profile["scheduler"]["factor"]),
        scheduler_patience=int(profile["scheduler"]["patience_validation_checks"]),
        scheduler_min_lr=float(profile["scheduler"]["minimum_learning_rate"]),
        early_stopping_patience=int(
            profile["early_stopping"]["patience_validation_checks"]
        ),
        minimum_delta=float(profile["early_stopping"]["minimum_delta"]),
        max_epochs=int(profile["formal_max_epochs"]),
        absolute_max_epochs=int(profile["conditional_absolute_max_epochs"]),
        lcl_index_path=lcl_index,
        cer_index_path=cer_index,
        lcl_scaler_path=lcl_scaler,
        cer_scaler_path=cer_scaler,
    )

    if settings.batch_size != 512:
        raise RuntimeError("Canonical batch size is not 512.")
    if settings.num_workers != 0:
        raise RuntimeError("This formal runner requires canonical num_workers=0.")
    if not math.isclose(settings.source_scratch_lr, 1e-3, rel_tol=0, abs_tol=1e-15):
        raise RuntimeError("Unexpected source/scratch learning rate.")
    if not math.isclose(settings.fine_tuning_lr, 3e-5, rel_tol=0, abs_tol=1e-15):
        raise RuntimeError("Unexpected fine-tuning learning rate.")
    if settings.early_stopping_patience != 5:
        raise RuntimeError("Unexpected early-stopping patience.")
    if settings.max_epochs != 30 or settings.absolute_max_epochs != 45:
        raise RuntimeError("Unexpected epoch limits.")
    if set(settings.fine_tuning_frozen) != {"conv1", "conv2", "lstm"}:
        raise RuntimeError("Unexpected fine-tuning frozen layers.")
    if set(settings.fine_tuning_trainable) != {"fc1", "output"}:
        raise RuntimeError("Unexpected fine-tuning trainable layers.")

    return settings




def move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def build_calendar_features(
    local_slot: np.ndarray,
    local_dow: np.ndarray,
    local_weekend: np.ndarray,
    local_month: np.ndarray,
) -> np.ndarray:
    slot_angle = 2.0 * np.pi * (local_slot.astype(np.float64) - 1.0) / 48.0
    dow_angle = 2.0 * np.pi * local_dow.astype(np.float64) / 7.0
    month_angle = 2.0 * np.pi * (local_month.astype(np.float64) - 1.0) / 12.0

    result = np.column_stack(
        [
            np.sin(slot_angle),
            np.cos(slot_angle),
            np.sin(dow_angle),
            np.cos(dow_angle),
            local_weekend.astype(np.float64),
            np.sin(month_angle),
            np.cos(month_angle),
        ]
    ).astype(np.float32)

    if result.shape[1] != 7 or not np.isfinite(result).all():
        raise AssertionError("Calendar feature construction failed.")
    return np.ascontiguousarray(result)


def read_all_lcl_arrays(
    project_root: Path,
    entity_ids: list[str],
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    time_map_path = (
        project_root
        / "data/processed/time_aligned/lcl_full_local_clock_map.parquet"
    )
    raw_path = (
        project_root
        / "data/raw/LCL/csv/data_collection/data_tables/consumption_n.csv"
    )

    time_map = pd.read_parquet(
        time_map_path,
        columns=[
            "raw_row_index",
            "timestamp_gmt",
            "local_slot",
            "local_day_of_week",
            "local_weekend",
            "local_month",
        ],
    )
    timestamp = pd.to_datetime(time_map["timestamp_gmt"], errors="raise")
    mask = (timestamp >= LCL_START) & (timestamp <= LCL_END)
    period_map = time_map.loc[mask].reset_index(drop=True)

    grid = pd.DatetimeIndex(
        pd.to_datetime(period_map["timestamp_gmt"], errors="raise")
    )
    if len(grid) != 17_520:
        raise AssertionError(f"Unexpected LCL governed-grid length: {len(grid)}")

    raw_indices = period_map["raw_row_index"].astype(np.int64).to_numpy()
    if not np.array_equal(
        raw_indices,
        np.arange(raw_indices[0], raw_indices[-1] + 1),
    ):
        raise AssertionError("LCL governed rows are not contiguous.")

    first_raw_row = int(raw_indices[0])
    frame = pd.read_csv(
        raw_path,
        usecols=entity_ids,
        skiprows=range(1, first_raw_row + 1),
        nrows=len(grid),
        low_memory=False,
    )
    frame = frame[entity_ids]
    load_matrix = (
        frame.apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float32)
    )
    del frame

    calendar = build_calendar_features(
        period_map["local_slot"].to_numpy(),
        period_map["local_day_of_week"].to_numpy(),
        period_map["local_weekend"].to_numpy(dtype=bool),
        period_map["local_month"].to_numpy(),
    )
    return grid, np.ascontiguousarray(load_matrix), calendar


def read_all_cer_arrays(
    project_root: Path,
    entity_ids: list[str],
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    grid_path = (
        project_root
        / "data/processed/time_aligned/cer_canonical_local_grid.parquet"
    )
    numeric_ids = np.asarray([int(entity) for entity in entity_ids], dtype=np.int64)
    grid = pd.date_range(CER_START, CER_END, freq="30min")

    frame = pd.read_parquet(
        grid_path,
        columns=[
            "meter_id",
            "timestamp_local",
            "kwh",
            "local_slot",
            "local_day_of_week",
            "local_weekend",
            "local_month",
        ],
    )
    frame["meter_id"] = frame["meter_id"].astype(np.int64)
    frame = frame.loc[frame["meter_id"].isin(numeric_ids)].copy()
    frame["timestamp_local"] = pd.to_datetime(
        frame["timestamp_local"], errors="raise"
    )

    expected_rows = len(grid) * len(numeric_ids)
    if len(frame) != expected_rows:
        raise AssertionError(
            f"CER rows={len(frame):,}; expected={expected_rows:,}."
        )

    id_to_col = {
        meter_id: index
        for index, meter_id in enumerate(numeric_ids.tolist())
    }
    columns = frame["meter_id"].map(id_to_col).to_numpy(dtype=np.int64)
    positions = np.searchsorted(
        grid.to_numpy(dtype="datetime64[ns]").astype(np.int64),
        to_ns(frame["timestamp_local"]),
    )

    load_matrix = np.full(
        (len(grid), len(entity_ids)),
        np.nan,
        dtype=np.float32,
    )
    load_matrix[positions, columns] = (
        pd.to_numeric(frame["kwh"], errors="coerce")
        .to_numpy(dtype=np.float32)
    )

    reference_id = int(numeric_ids[0])
    calendar_frame = (
        frame.loc[
            frame["meter_id"].eq(reference_id),
            [
                "timestamp_local",
                "local_slot",
                "local_day_of_week",
                "local_weekend",
                "local_month",
            ],
        ]
        .sort_values("timestamp_local")
        .reset_index(drop=True)
    )
    if len(calendar_frame) != len(grid):
        raise AssertionError("CER calendar length mismatch.")

    calendar = build_calendar_features(
        calendar_frame["local_slot"].to_numpy(),
        calendar_frame["local_day_of_week"].to_numpy(),
        calendar_frame["local_weekend"].to_numpy(dtype=bool),
        calendar_frame["local_month"].to_numpy(),
    )
    del frame, calendar_frame
    return grid, np.ascontiguousarray(load_matrix), calendar


def make_base_arrays(
    project_root: Path,
    dataset: str,
    entity_ids: list[str],
) -> BaseArrays:
    if dataset == "LCL":
        grid, load_matrix, calendar = read_all_lcl_arrays(project_root, entity_ids)
    elif dataset == "CER":
        grid, load_matrix, calendar = read_all_cer_arrays(project_root, entity_ids)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return BaseArrays(
        dataset=dataset,
        grid=grid,
        grid_ns=grid.to_numpy(dtype="datetime64[ns]").astype(np.int64),
        entity_ids=entity_ids,
        entity_to_column={entity: index for index, entity in enumerate(entity_ids)},
        load_matrix=load_matrix,
        calendar_features=calendar,
    )


def make_scalers(
    path: Path,
    entity_ids: list[str],
    span_setting: str | None,
) -> ScalerArrays:
    frame = pd.read_csv(path, dtype={"entity_id": str})
    if span_setting is not None:
        frame = frame.loc[frame["span_setting"].astype(str).eq(span_setting)].copy()
    if frame["entity_id"].duplicated().any():
        raise AssertionError(
            f"Scaler selection {span_setting!r} contains duplicate entity IDs."
        )
    frame = frame.set_index("entity_id").loc[entity_ids].reset_index()

    means = frame["training_mean_kwh"].to_numpy(dtype=np.float32)
    stds = frame["effective_training_std_kwh"].to_numpy(dtype=np.float32)
    if not np.isfinite(means).all() or not np.isfinite(stds).all():
        raise AssertionError("Non-finite scaler values.")
    if np.any(stds <= 0):
        raise AssertionError("Non-positive effective scaler standard deviation.")

    fit_start = None
    fit_end = None
    if "scaler_fit_start" in frame.columns:
        values = pd.to_datetime(frame["scaler_fit_start"], errors="raise").unique()
        if len(values) != 1:
            raise AssertionError("Scaler fit start is not common across entities.")
        fit_start = pd.Timestamp(values[0])
    if "scaler_fit_end" in frame.columns:
        values = pd.to_datetime(frame["scaler_fit_end"], errors="raise").unique()
        if len(values) != 1:
            raise AssertionError("Scaler fit end is not common across entities.")
        fit_end = pd.Timestamp(values[0])

    return ScalerArrays(
        name=span_setting or "lcl_training",
        means=np.ascontiguousarray(means),
        stds=np.ascontiguousarray(stds),
        fit_start=fit_start,
        fit_end=fit_end,
    )


def index_row_batches(
    index_path: Path,
    split: str,
    batch_size: int,
    epoch_seed: int | None,
    target_end: pd.Timestamp | None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    parquet_file = pq.ParquetFile(index_path)
    groups = np.arange(parquet_file.metadata.num_row_groups, dtype=np.int64)
    rng: np.random.Generator | None = None
    if epoch_seed is not None:
        rng = np.random.default_rng(epoch_seed)
        rng.shuffle(groups)

    carry_entities = np.empty(0, dtype=object)
    carry_timestamps = np.empty(0, dtype=np.int64)

    for row_group in groups.tolist():
        table = parquet_file.read_row_group(
            row_group,
            columns=["entity_id", "target_timestamp", "split"],
        )
        frame = table.to_pandas()
        mask = frame["split"].astype(str).eq(split)
        if target_end is not None:
            timestamps = pd.to_datetime(frame["target_timestamp"], errors="raise")
            mask &= timestamps <= target_end
        frame = frame.loc[mask, ["entity_id", "target_timestamp"]].reset_index(drop=True)
        if frame.empty:
            continue

        entities = frame["entity_id"].astype(str).to_numpy()
        timestamps_ns = to_ns(frame["target_timestamp"])
        if rng is not None:
            order = rng.permutation(len(frame))
            entities = entities[order]
            timestamps_ns = timestamps_ns[order]

        if len(carry_entities):
            entities = np.concatenate([carry_entities, entities])
            timestamps_ns = np.concatenate([carry_timestamps, timestamps_ns])

        full_length = (len(entities) // batch_size) * batch_size
        for start in range(0, full_length, batch_size):
            yield (
                entities[start : start + batch_size],
                timestamps_ns[start : start + batch_size],
            )
        carry_entities = entities[full_length:]
        carry_timestamps = timestamps_ns[full_length:]

    # Formal rule: never drop the final partial batch.
    if len(carry_entities):
        yield carry_entities, carry_timestamps


def reconstruct_batch(
    entity_ids: np.ndarray,
    target_ns: np.ndarray,
    arrays: BaseArrays,
    scalers: ScalerArrays,
    horizon: int,
    include_raw: bool,
) -> dict[str, Any]:
    columns = np.fromiter(
        (arrays.entity_to_column[str(value)] for value in entity_ids),
        dtype=np.int64,
        count=len(entity_ids),
    )
    target_positions = np.searchsorted(arrays.grid_ns, target_ns)
    if np.any(target_positions >= len(arrays.grid_ns)):
        raise AssertionError(f"{arrays.dataset}: target outside governed grid.")
    if not np.array_equal(arrays.grid_ns[target_positions], target_ns):
        raise AssertionError(f"{arrays.dataset}: target-grid mismatch.")

    starts = target_positions - horizon - (LOOKBACK - 1)
    if np.any(starts < 0):
        raise AssertionError("Negative input start.")
    positions = starts[:, None] + np.arange(LOOKBACK, dtype=np.int64)[None, :]

    raw_x = arrays.load_matrix[positions, columns[:, None]]
    raw_y = arrays.load_matrix[target_positions, columns]
    if not np.isfinite(raw_x).all() or not np.isfinite(raw_y).all():
        raise AssertionError(f"{arrays.dataset}: missing data reconstructed.")

    means = scalers.means[columns]
    stds = scalers.stds[columns]
    x = np.empty((len(entity_ids), LOOKBACK, INPUT_FEATURES), dtype=np.float32)
    x[:, :, 0] = (raw_x - means[:, None]) / stds[:, None]
    x[:, :, 1:] = arrays.calendar_features[positions]
    y = ((raw_y - means) / stds).astype(np.float32)[:, None]

    result: dict[str, Any] = {
        "x": torch.from_numpy(x),
        "y": torch.from_numpy(y),
        "columns": columns,
    }
    if include_raw:
        persistence_positions = target_positions - horizon
        daily_positions = target_positions - 48
        if np.any(persistence_positions < 0) or np.any(daily_positions < 0):
            raise AssertionError("Baseline position is outside governed support.")
        persistence = arrays.load_matrix[persistence_positions, columns]
        daily = arrays.load_matrix[daily_positions, columns]
        if not np.isfinite(persistence).all() or not np.isfinite(daily).all():
            raise AssertionError("Legal test sample produced a missing baseline value.")
        result.update(
            {
                "raw_target": raw_y.astype(np.float32),
                "mean": means.astype(np.float32),
                "std": stds.astype(np.float32),
                "persistence": persistence.astype(np.float32),
                "daily": daily.astype(np.float32),
            }
        )
    return result


def move_tensor(
    tensor: torch.Tensor,
    device: torch.device,
    pin_memory: bool,
) -> torch.Tensor:
    if pin_memory and device.type == "cuda" and not tensor.is_pinned():
        tensor = tensor.pin_memory()
    return tensor.to(device, non_blocking=(pin_memory and device.type == "cuda"))


def build_run_paths(
    project_root: Path,
    horizon: int,
    strategy: str,
    seed: int,
) -> RunPaths:
    relative = Path(f"h{horizon}") / strategy / f"seed_{seed}"
    model_dir = project_root / "outputs/models/pooled_strategy_evaluation_formal" / relative
    table_dir = project_root / "outputs/tables/pooled_strategy_evaluation_formal" / relative
    figure_dir = project_root / "outputs/figures/pooled_strategy_evaluation_formal" / relative
    metadata_dir = project_root / "outputs/metadata/pooled_strategy_evaluation_formal" / relative
    return RunPaths(
        model_dir=model_dir,
        table_dir=table_dir,
        figure_dir=figure_dir,
        metadata_dir=metadata_dir,
        best_checkpoint=model_dir / "best_checkpoint.pt",
        last_state=model_dir / "last_state.pt",
        epoch_log=table_dir / "epoch_log.csv",
        learning_curve=figure_dir / "learning_curve.png",
        predictions=table_dir / "test_predictions.parquet",
        pooled_metrics=table_dir / "test_pooled_metrics.csv",
        household_metrics=table_dir / "test_household_metrics.csv",
        metadata=metadata_dir / "run_metadata.md",
        status=metadata_dir / "run_status.json",
        source_reference=metadata_dir / "source_checkpoint_reference.json",
    )


def source_checkpoint_path(project_root: Path, horizon: int, seed: int) -> Path:
    return (
        project_root
        / "outputs/models/pooled_strategy_evaluation_formal"
        / f"h{horizon}"
        / "lcl_source"
        / f"seed_{seed}"
        / "best_checkpoint.pt"
    )


def ensure_fresh_or_resumable(paths: RunPaths, resume: bool, trainable: bool) -> None:
    all_outputs = [
        paths.best_checkpoint,
        paths.last_state,
        paths.epoch_log,
        paths.learning_curve,
        paths.predictions,
        paths.pooled_metrics,
        paths.household_metrics,
        paths.metadata,
        paths.status,
        paths.source_reference,
    ]
    existing = [path for path in all_outputs if path.exists()]

    if resume:
        if not trainable:
            raise RuntimeError("--resume is valid only for trainable strategies.")
        if not paths.last_state.exists():
            raise FileNotFoundError(
                f"Resume requested but last state is absent: {paths.last_state}"
            )
        if paths.predictions.exists() or paths.pooled_metrics.exists():
            raise RuntimeError(
                "Resume is blocked because test outputs already exist. Review the run manually."
            )
        return

    if existing:
        raise FileExistsError(
            "Formal forecasting will not overwrite existing run outputs:\n"
            + "\n".join(str(path) for path in existing)
        )


def load_formal_checkpoint(
    path: Path,
    expected_strategy: str,
    expected_seed: int,
    expected_horizon: int,
    expected_config_sha256: str,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required formal checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("valid_for_formal_results") is not True:
        raise RuntimeError(f"Checkpoint is not formal: {path}")
    if checkpoint.get("strategy") != expected_strategy:
        raise RuntimeError("Checkpoint strategy mismatch.")
    if int(checkpoint.get("seed")) != expected_seed:
        raise RuntimeError("Checkpoint seed mismatch.")
    if int(checkpoint.get("horizon")) != expected_horizon:
        raise RuntimeError("Checkpoint horizon mismatch.")
    if checkpoint.get("config_sha256") != expected_config_sha256:
        raise RuntimeError("Checkpoint canonical-config hash mismatch.")
    return checkpoint


def configure_trainable_layers(
    model: nn.Module,
    trainable_layers: list[str],
) -> tuple[list[str], list[str]]:
    trainable_set = set(trainable_layers)
    if not trainable_set.issubset(MODEL_TOP_LEVELS):
        raise RuntimeError("Unknown trainable top-level layer.")
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.split(".")[0] in trainable_set

    actual_trainable = sorted(
        {
            name.split(".")[0]
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
    )
    actual_frozen = sorted(MODEL_TOP_LEVELS - set(actual_trainable))
    if set(actual_trainable) != trainable_set:
        raise RuntimeError("Trainable-layer configuration mismatch.")
    return actual_trainable, actual_frozen


def parameter_snapshot(
    model: nn.Module,
    top_levels: list[str],
) -> dict[str, torch.Tensor]:
    selected = set(top_levels)
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if name.split(".")[0] in selected
    }


def maximum_parameter_difference(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> float:
    maximum = 0.0
    for name, old_value in before.items():
        difference = torch.max(torch.abs(after[name].detach().cpu() - old_value)).item()
        maximum = max(maximum, float(difference))
    return maximum


def train_complete_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    spec: StrategySpec,
    settings: CanonicalSettings,
    epoch: int,
    device: torch.device,
) -> dict[str, Any]:
    model.train()
    squared_error_sum = 0.0
    observation_count = 0
    optimizer_steps = 0
    maximum_gradient_norm = 0.0
    start = time.perf_counter()

    for entities, timestamps_ns in index_row_batches(
        index_path=spec.index_path,
        split="train",
        batch_size=settings.batch_size,
        epoch_seed=stable_epoch_seed(settings.seed, epoch),
        target_end=spec.train_target_end,
    ):
        batch = reconstruct_batch(
            entity_ids=entities,
            target_ns=timestamps_ns,
            arrays=spec.arrays,
            scalers=spec.scalers,
            horizon=settings.horizon,
            include_raw=False,
        )
        x = move_tensor(batch["x"], device, settings.pin_memory)
        y = move_tensor(batch["y"], device, settings.pin_memory)

        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        squared_error = torch.square(prediction - y)
        loss = squared_error.mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{spec.name}: non-finite training loss.")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            max_norm=settings.gradient_clip,
        )
        gradient_value = float(torch.as_tensor(gradient_norm).detach().cpu())
        if not np.isfinite(gradient_value):
            raise FloatingPointError(f"{spec.name}: non-finite gradient norm.")
        optimizer.step()

        squared_error_sum += float(squared_error.sum().detach().cpu())
        observation_count += int(y.numel())
        optimizer_steps += 1
        maximum_gradient_norm = max(maximum_gradient_norm, gradient_value)

    synchronise(device)
    elapsed = time.perf_counter() - start
    expected = EXPECTED_H48_COUNTS[spec.name]["train"]
    if observation_count != expected:
        raise AssertionError(
            f"{spec.name}: train observations={observation_count:,}; expected={expected:,}."
        )
    if optimizer_steps <= 0:
        raise AssertionError("Training epoch produced no optimizer steps.")

    return {
        "train_mse": squared_error_sum / observation_count,
        "train_elapsed_seconds": elapsed,
        "train_samples_per_second": observation_count / elapsed,
        "optimizer_steps_this_epoch": optimizer_steps,
        "samples_processed_this_epoch": observation_count,
        "maximum_preclip_gradient_norm": maximum_gradient_norm,
    }


def validate_complete_split(
    model: nn.Module,
    spec: StrategySpec,
    settings: CanonicalSettings,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    squared_error_sum = 0.0
    observation_count = 0
    start = time.perf_counter()

    with torch.no_grad():
        for entities, timestamps_ns in index_row_batches(
            index_path=spec.index_path,
            split="validation",
            batch_size=settings.batch_size,
            epoch_seed=None,
            target_end=None,
        ):
            batch = reconstruct_batch(
                entity_ids=entities,
                target_ns=timestamps_ns,
                arrays=spec.arrays,
                scalers=spec.scalers,
                horizon=settings.horizon,
                include_raw=False,
            )
            x = move_tensor(batch["x"], device, settings.pin_memory)
            y = move_tensor(batch["y"], device, settings.pin_memory)
            prediction = model(x)
            if not torch.isfinite(prediction).all():
                raise FloatingPointError(
                    f"{spec.name}: non-finite validation prediction."
                )
            squared_error_sum += float(
                torch.square(prediction - y).sum().detach().cpu()
            )
            observation_count += int(y.numel())

    synchronise(device)
    elapsed = time.perf_counter() - start
    expected = EXPECTED_H48_COUNTS[spec.name]["validation"]
    if observation_count != expected:
        raise AssertionError(
            f"{spec.name}: validation observations={observation_count:,}; "
            f"expected={expected:,}."
        )
    return {
        "full_validation_mse": squared_error_sum / observation_count,
        "validation_elapsed_seconds": elapsed,
        "validation_samples_per_second": observation_count / elapsed,
        "validation_observations": observation_count,
    }


def extension_gate(validation_values: list[float], epoch: int) -> dict[str, Any]:
    if epoch != 30:
        raise ValueError("The formal extension gate is evaluated only at epoch 30.")
    if len(validation_values) < 6:
        raise RuntimeError("Insufficient validation history for extension gate.")
    best_epoch = int(np.argmin(validation_values) + 1)
    previous_mean = float(np.mean(validation_values[-6:-3]))
    recent_mean = float(np.mean(validation_values[-3:]))
    relative = (
        (previous_mean - recent_mean) / previous_mean
        if previous_mean > 0
        else 0.0
    )
    best_recent = best_epoch >= epoch - 1
    material = recent_mean < previous_mean and relative >= 0.001
    approved = bool(best_recent and material)
    return {
        "approved": approved,
        "best_epoch": best_epoch,
        "best_epoch_within_last_2": best_recent,
        "previous_3_mean": previous_mean,
        "recent_3_mean": recent_mean,
        "relative_improvement": relative,
        "relative_improvement_threshold": 0.001,
    }


def save_best_checkpoint(
    path: Path,
    model: nn.Module,
    spec: StrategySpec,
    settings: CanonicalSettings,
    epoch: int,
    validation_mse: float,
    trainable_layers: list[str],
    frozen_layers: list[str],
) -> None:
    payload = {
        "analysis": "Formal forecasting",
        "formal": True,
        "diagnostic_only": False,
        "valid_for_formal_results": True,
        "strategy": spec.name,
        "seed": settings.seed,
        "horizon": settings.horizon,
        "architecture": "global_cnn_lstm",
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "batch_size": settings.batch_size,
        "learning_rate": spec.learning_rate,
        "trainable_layers": trainable_layers,
        "frozen_layers": frozen_layers,
        "best_epoch": epoch,
        "best_validation_mse_scaled": validation_mse,
        "config_path": public_repo_path(settings.project_root, settings.config_path),
        "config_sha256": settings.config_sha256,
        "created_utc": utc_now(),
        "model_state_dict": cpu_state_dict(model),
    }
    atomic_torch_save(path, payload)


def save_last_state(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    spec: StrategySpec,
    settings: CanonicalSettings,
    epoch: int,
    best_epoch: int,
    best_validation_mse: float,
    no_improvement_checks: int,
    cumulative_optimizer_steps: int,
    cumulative_samples: int,
    maximum_epoch_allowed: int,
    extension_decision: dict[str, Any] | None,
    epoch_rows: list[dict[str, Any]],
) -> None:
    payload = {
        "analysis": "Formal forecasting",
        "formal": True,
        "valid_for_formal_results": True,
        "strategy": spec.name,
        "seed": settings.seed,
        "horizon": settings.horizon,
        "config_sha256": settings.config_sha256,
        "last_completed_epoch": epoch,
        "best_epoch": best_epoch,
        "best_validation_mse": best_validation_mse,
        "no_improvement_checks": no_improvement_checks,
        "cumulative_optimizer_steps": cumulative_optimizer_steps,
        "cumulative_samples": cumulative_samples,
        "maximum_epoch_allowed": maximum_epoch_allowed,
        "extension_decision": extension_decision,
        "epoch_rows": epoch_rows,
        "model_state_dict": cpu_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "rng_state": capture_rng_state(),
        "saved_utc": utc_now(),
    }
    atomic_torch_save(path, payload)


def plot_learning_curve(
    epoch_frame: pd.DataFrame,
    output_path: Path,
    strategy: str,
    seed: int,
) -> None:
    training_rows = epoch_frame.loc[epoch_frame["epoch"] > 0].copy()
    if training_rows.empty:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(training_rows["epoch"], training_rows["train_mse"], label="Training MSE")
    axis.plot(
        training_rows["epoch"],
        training_rows["full_validation_mse"],
        label="Full validation MSE",
    )
    best_index = training_rows["full_validation_mse"].astype(float).idxmin()
    best_row = training_rows.loc[best_index]
    axis.scatter(
        [best_row["epoch"]],
        [best_row["full_validation_mse"]],
        s=70,
        label=f"Best epoch {int(best_row['epoch'])}",
    )
    for _, row in training_rows.loc[
        training_rows["learning_rate_reduced_this_epoch"].astype(bool)
    ].iterrows():
        axis.axvline(int(row["epoch"]), linestyle="--", linewidth=1)
    extension_rows = training_rows.loc[
        training_rows["cap_extension_triggered"].astype(bool)
    ]
    for _, row in extension_rows.iterrows():
        axis.axvline(int(row["epoch"]), linestyle=":", linewidth=1)
    axis.set_title(f"Formal forecasting h=48 {strategy} — seed {seed}")
    axis.set_xlabel("Complete training epoch")
    axis.set_ylabel("MSE in strategy-specific scaled space")
    axis.grid(True)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def build_strategy_spec(
    strategy: str,
    settings: CanonicalSettings,
) -> StrategySpec:
    lcl_scaler_frame = pd.read_csv(settings.lcl_scaler_path, dtype={"entity_id": str})
    cer_scaler_frame = pd.read_csv(settings.cer_scaler_path, dtype={"entity_id": str})
    lcl_ids = lcl_scaler_frame["entity_id"].astype(str).tolist()
    cer_ids = sorted(
        cer_scaler_frame["entity_id"].astype(str).unique().tolist(),
        key=int,
    )
    if len(lcl_ids) != 3_843:
        raise AssertionError(f"Unexpected LCL population: {len(lcl_ids)}")
    if len(cer_ids) != 929:
        raise AssertionError(f"Unexpected CER population: {len(cer_ids)}")

    source_path: Path | None = None
    initial_state: dict[str, torch.Tensor] | None = None

    if strategy == "lcl_source":
        arrays = make_base_arrays(settings.project_root, "LCL", lcl_ids)
        scalers = make_scalers(settings.lcl_scaler_path, lcl_ids, None)
        return StrategySpec(
            name=strategy,
            dataset="LCL",
            index_path=settings.lcl_index_path,
            arrays=arrays,
            scalers=scalers,
            train_target_end=None,
            learning_rate=settings.source_scratch_lr,
            trainable_layers=sorted(MODEL_TOP_LEVELS),
            initial_state=None,
            source_checkpoint_path=None,
        )

    arrays = make_base_arrays(settings.project_root, "CER", cer_ids)
    if strategy in {"direct_transfer", "fine_tuning", "cer_scratch_limited"}:
        scalers = make_scalers(settings.cer_scaler_path, cer_ids, "cer_30day")
    else:
        scalers = make_scalers(settings.cer_scaler_path, cer_ids, "cer_full_training")

    if strategy in {"direct_transfer", "fine_tuning"}:
        source_path = source_checkpoint_path(
            settings.project_root,
            settings.horizon,
            settings.seed,
        )
        source_checkpoint = load_formal_checkpoint(
            source_path,
            expected_strategy="lcl_source",
            expected_seed=settings.seed,
            expected_horizon=settings.horizon,
            expected_config_sha256=settings.config_sha256,
        )
        initial_state = source_checkpoint["model_state_dict"]

    if strategy == "direct_transfer":
        return StrategySpec(
            name=strategy,
            dataset="CER",
            index_path=settings.cer_index_path,
            arrays=arrays,
            scalers=scalers,
            train_target_end=None,
            learning_rate=None,
            trainable_layers=[],
            initial_state=initial_state,
            source_checkpoint_path=source_path,
        )
    if strategy == "fine_tuning":
        if scalers.fit_end is None:
            raise RuntimeError("CER 30-day scaler fit end is unavailable.")
        return StrategySpec(
            name=strategy,
            dataset="CER",
            index_path=settings.cer_index_path,
            arrays=arrays,
            scalers=scalers,
            train_target_end=scalers.fit_end,
            learning_rate=settings.fine_tuning_lr,
            trainable_layers=settings.fine_tuning_trainable,
            initial_state=initial_state,
            source_checkpoint_path=source_path,
        )
    if strategy == "cer_scratch_limited":
        if scalers.fit_end is None:
            raise RuntimeError("CER 30-day scaler fit end is unavailable.")
        return StrategySpec(
            name=strategy,
            dataset="CER",
            index_path=settings.cer_index_path,
            arrays=arrays,
            scalers=scalers,
            train_target_end=scalers.fit_end,
            learning_rate=settings.source_scratch_lr,
            trainable_layers=sorted(MODEL_TOP_LEVELS),
            initial_state=None,
            source_checkpoint_path=None,
        )
    if strategy == "cer_scratch_full":
        return StrategySpec(
            name=strategy,
            dataset="CER",
            index_path=settings.cer_index_path,
            arrays=arrays,
            scalers=scalers,
            train_target_end=None,
            learning_rate=settings.source_scratch_lr,
            trainable_layers=sorted(MODEL_TOP_LEVELS),
            initial_state=None,
            source_checkpoint_path=None,
        )
    raise ValueError(strategy)


def train_formal_strategy(
    spec: StrategySpec,
    settings: CanonicalSettings,
    paths: RunPaths,
    device: torch.device,
    resume: bool,
) -> dict[str, Any]:
    heading(f"FORMAL TRAINING — {spec.name} — seed {settings.seed}")
    set_global_seed(settings.seed)

    model = GlobalCNNLSTM().to(device)
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETER_COUNT:
        raise AssertionError("Model parameter count mismatch.")
    if spec.initial_state is not None:
        model.load_state_dict(spec.initial_state)

    trainable_layers, frozen_layers = configure_trainable_layers(
        model,
        spec.trainable_layers,
    )
    frozen_before = parameter_snapshot(model, frozen_layers)
    trainable_before = parameter_snapshot(model, trainable_layers)

    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(spec.learning_rate),
        betas=ADAM_BETAS,
        eps=ADAM_EPS,
        weight_decay=ADAM_WEIGHT_DECAY,
        amsgrad=ADAM_AMSGRAD,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=settings.scheduler_factor,
        patience=settings.scheduler_patience,
        threshold=SCHEDULER_THRESHOLD,
        threshold_mode=SCHEDULER_THRESHOLD_MODE,
        cooldown=SCHEDULER_COOLDOWN,
        min_lr=settings.scheduler_min_lr,
        eps=SCHEDULER_EPS,
    )

    epoch_rows: list[dict[str, Any]] = []
    start_epoch = 1
    best_epoch = 0
    best_validation_mse = float("inf")
    no_improvement_checks = 0
    cumulative_optimizer_steps = 0
    cumulative_samples = 0
    maximum_epoch_allowed = settings.max_epochs
    extension_decision: dict[str, Any] | None = None

    if resume:
        state = torch.load(paths.last_state, map_location="cpu", weights_only=False)
        if state.get("config_sha256") != settings.config_sha256:
            raise RuntimeError("Resume state config hash mismatch.")
        if state.get("strategy") != spec.name or int(state.get("seed")) != settings.seed:
            raise RuntimeError("Resume state strategy/seed mismatch.")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        move_optimizer_state_to_device(optimizer, device)
        scheduler.load_state_dict(state["scheduler_state_dict"])
        restore_rng_state(state["rng_state"])
        start_epoch = int(state["last_completed_epoch"]) + 1
        best_epoch = int(state["best_epoch"])
        best_validation_mse = float(state["best_validation_mse"])
        no_improvement_checks = int(state["no_improvement_checks"])
        cumulative_optimizer_steps = int(state["cumulative_optimizer_steps"])
        cumulative_samples = int(state["cumulative_samples"])
        maximum_epoch_allowed = int(state["maximum_epoch_allowed"])
        extension_decision = state.get("extension_decision")
        epoch_rows = list(state["epoch_rows"])
        print(f"Resuming after epoch {start_epoch - 1}.")
    elif spec.name == "fine_tuning":
        # Required diagnostic: source checkpoint on CER validation before any
        # CER update. It is not eligible for best-checkpoint selection.
        epoch_zero = validate_complete_split(model, spec, settings, device)
        epoch_rows.append(
            {
                "strategy": spec.name,
                "seed": settings.seed,
                "horizon": settings.horizon,
                "epoch": 0,
                "epoch_role": "pre_update_validation_diagnostic",
                "eligible_for_best_checkpoint_selection": False,
                "train_mse": np.nan,
                "full_validation_mse": epoch_zero["full_validation_mse"],
                "learning_rate_before_scheduler": spec.learning_rate,
                "learning_rate_after_scheduler": spec.learning_rate,
                "learning_rate_reduced_this_epoch": False,
                "best_validation_mse_so_far": np.nan,
                "best_epoch_so_far": np.nan,
                "checkpoint_saved_this_epoch": False,
                "train_elapsed_seconds": 0.0,
                "validation_elapsed_seconds": epoch_zero[
                    "validation_elapsed_seconds"
                ],
                "optimizer_steps_this_epoch": 0,
                "cumulative_optimizer_steps": 0,
                "samples_processed_this_epoch": 0,
                "cumulative_samples_processed": 0,
                "maximum_preclip_gradient_norm": np.nan,
                "early_stopping_triggered": False,
                "cap_extension_triggered": False,
                "stopping_reason": "",
            }
        )
        print(
            "Fine-tuning epoch 0 CER validation diagnostic: "
            f"{epoch_zero['full_validation_mse']:.8f}"
        )

    validation_values = [
        float(row["full_validation_mse"])
        for row in epoch_rows
        if int(row["epoch"]) > 0
    ]
    stopping_reason = ""

    for epoch in range(start_epoch, settings.absolute_max_epochs + 1):
        if epoch > maximum_epoch_allowed:
            break

        lr_before = float(optimizer.param_groups[0]["lr"])
        train_result = train_complete_epoch(
            model=model,
            optimizer=optimizer,
            spec=spec,
            settings=settings,
            epoch=epoch,
            device=device,
        )
        validation_result = validate_complete_split(
            model=model,
            spec=spec,
            settings=settings,
            device=device,
        )
        validation_mse = float(validation_result["full_validation_mse"])
        validation_values.append(validation_mse)

        improved = validation_mse < best_validation_mse - settings.minimum_delta
        if improved:
            best_validation_mse = validation_mse
            best_epoch = epoch
            no_improvement_checks = 0
            save_best_checkpoint(
                path=paths.best_checkpoint,
                model=model,
                spec=spec,
                settings=settings,
                epoch=epoch,
                validation_mse=validation_mse,
                trainable_layers=trainable_layers,
                frozen_layers=frozen_layers,
            )
        else:
            no_improvement_checks += 1

        scheduler.step(validation_mse)
        lr_after = float(optimizer.param_groups[0]["lr"])
        reduced = lr_after < lr_before

        cumulative_optimizer_steps += int(
            train_result["optimizer_steps_this_epoch"]
        )
        cumulative_samples += int(train_result["samples_processed_this_epoch"])

        early_stop = no_improvement_checks >= settings.early_stopping_patience
        extension_triggered = False
        epoch_stopping_reason = ""

        if early_stop:
            epoch_stopping_reason = "early_stopping_patience_reached"
            stopping_reason = epoch_stopping_reason
        elif epoch == settings.max_epochs:
            extension_decision = extension_gate(validation_values, epoch)
            if extension_decision["approved"]:
                maximum_epoch_allowed = settings.absolute_max_epochs
                extension_triggered = True
            else:
                epoch_stopping_reason = "normal_max_epochs_reached_no_extension"
                stopping_reason = epoch_stopping_reason
        elif epoch == settings.absolute_max_epochs:
            epoch_stopping_reason = "conditional_absolute_maximum_reached"
            stopping_reason = epoch_stopping_reason

        row = {
            "strategy": spec.name,
            "seed": settings.seed,
            "horizon": settings.horizon,
            "epoch": epoch,
            "epoch_role": "complete_training_epoch",
            "eligible_for_best_checkpoint_selection": True,
            **train_result,
            **validation_result,
            "learning_rate_before_scheduler": lr_before,
            "learning_rate_after_scheduler": lr_after,
            "learning_rate_reduced_this_epoch": reduced,
            "best_validation_mse_so_far": best_validation_mse,
            "best_epoch_so_far": best_epoch,
            "checkpoint_saved_this_epoch": improved,
            "cumulative_optimizer_steps": cumulative_optimizer_steps,
            "cumulative_samples_processed": cumulative_samples,
            "early_stopping_triggered": early_stop,
            "cap_extension_triggered": extension_triggered,
            "stopping_reason": epoch_stopping_reason,
        }
        epoch_rows.append(row)
        atomic_csv_dump(paths.epoch_log, pd.DataFrame(epoch_rows))
        save_last_state(
            path=paths.last_state,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            spec=spec,
            settings=settings,
            epoch=epoch,
            best_epoch=best_epoch,
            best_validation_mse=best_validation_mse,
            no_improvement_checks=no_improvement_checks,
            cumulative_optimizer_steps=cumulative_optimizer_steps,
            cumulative_samples=cumulative_samples,
            maximum_epoch_allowed=maximum_epoch_allowed,
            extension_decision=extension_decision,
            epoch_rows=epoch_rows,
        )

        print(
            f"{spec.name} | seed={settings.seed} | epoch={epoch:02d} | "
            f"train={train_result['train_mse']:.8f} | "
            f"val={validation_mse:.8f} | lr={lr_after:.8g} | "
            f"best_epoch={best_epoch} | no_improve={no_improvement_checks}",
            flush=True,
        )
        if extension_triggered:
            print(
                "  Epoch-30 extension gate: APPROVED — continue to maximum 45.",
                flush=True,
            )
        if stopping_reason:
            print(f"  Stopping reason: {stopping_reason}", flush=True)
            break

    if not paths.best_checkpoint.exists():
        raise RuntimeError("Formal training finished without a best checkpoint.")
    if not stopping_reason:
        previous_stopping_reasons = [
            str(row.get("stopping_reason", ""))
            for row in epoch_rows
            if str(row.get("stopping_reason", ""))
        ]
        stopping_reason = (
            previous_stopping_reasons[-1]
            if previous_stopping_reasons
            else "loop_completed_at_maximum_epoch_allowed"
        )

    best_checkpoint = load_formal_checkpoint(
        paths.best_checkpoint,
        expected_strategy=spec.name,
        expected_seed=settings.seed,
        expected_horizon=settings.horizon,
        expected_config_sha256=settings.config_sha256,
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])

    final_state = cpu_state_dict(model)
    frozen_best_difference = maximum_parameter_difference(
        frozen_before,
        final_state,
    ) if frozen_before else 0.0
    trainable_best_difference = maximum_parameter_difference(
        trainable_before,
        final_state,
    ) if trainable_before else 0.0

    if spec.name == "fine_tuning":
        if frozen_best_difference != 0.0:
            raise AssertionError(
                f"Fine-tuning frozen parameters changed by {frozen_best_difference}."
            )
        if trainable_best_difference <= 0.0:
            raise AssertionError("Fine-tuning dense head did not change.")

    epoch_frame = pd.DataFrame(epoch_rows)
    atomic_csv_dump(paths.epoch_log, epoch_frame)
    plot_learning_curve(
        epoch_frame=epoch_frame,
        output_path=paths.learning_curve,
        strategy=spec.name,
        seed=settings.seed,
    )

    return {
        "model": model,
        "best_epoch": int(best_checkpoint["best_epoch"]),
        "best_validation_mse": float(
            best_checkpoint["best_validation_mse_scaled"]
        ),
        "stopping_epoch": int(
            epoch_frame.loc[epoch_frame["epoch"] > 0, "epoch"].max()
        ),
        "stopping_reason": stopping_reason,
        "extension_decision": extension_decision,
        "trainable_layers": trainable_layers,
        "frozen_layers": frozen_layers,
        "frozen_best_max_abs_difference": frozen_best_difference,
        "trainable_best_max_abs_difference": trainable_best_difference,
        "source_checkpoint": public_repo_path(
            settings.project_root,
            spec.source_checkpoint_path,
        ),
    }


def smape_percent(actual: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    denominator = np.abs(actual) + np.abs(prediction)
    contribution = np.zeros_like(actual, dtype=np.float64)
    nonzero = denominator > 0
    contribution[nonzero] = (
        200.0
        * np.abs(prediction[nonzero] - actual[nonzero])
        / denominator[nonzero]
    )
    return contribution


def evaluate_test_and_write_predictions(
    model: nn.Module,
    spec: StrategySpec,
    settings: CanonicalSettings,
    paths: RunPaths,
    device: torch.device,
) -> dict[str, Any]:
    heading(f"FORMAL TEST EVALUATION — {spec.name} — seed {settings.seed}")
    if paths.predictions.exists():
        raise FileExistsError(f"Prediction output already exists: {paths.predictions}")

    paths.table_dir.mkdir(parents=True, exist_ok=True)
    temporary_predictions = paths.predictions.with_suffix(".parquet.tmp")
    if temporary_predictions.exists():
        temporary_predictions.unlink()

    schema = pa.schema(
        [
            ("entity_id", pa.string()),
            ("target_timestamp", pa.timestamp("ns")),
            ("actual_kwh", pa.float32()),
            ("predicted_kwh", pa.float32()),
            ("error_kwh", pa.float32()),
            ("absolute_error_kwh", pa.float32()),
            ("squared_error_kwh2", pa.float32()),
            ("smape_percent", pa.float32()),
            ("persistence_kwh", pa.float32()),
            ("daily_seasonal_naive_kwh", pa.float32()),
        ]
    )
    writer = pq.ParquetWriter(
        temporary_predictions,
        schema=schema,
        compression="zstd",
    )

    # Buffer multiple model batches into larger Parquet row groups. Writing one
    # 512-row Parquet row group per model batch would create excessive metadata.
    parquet_flush_rows = 131_072
    prediction_buffer: dict[str, list[np.ndarray]] = {
        "entity_id": [],
        "target_timestamp": [],
        "actual_kwh": [],
        "predicted_kwh": [],
        "error_kwh": [],
        "absolute_error_kwh": [],
        "squared_error_kwh2": [],
        "smape_percent": [],
        "persistence_kwh": [],
        "daily_seasonal_naive_kwh": [],
    }
    buffered_rows = 0

    def flush_prediction_buffer() -> None:
        nonlocal buffered_rows
        if buffered_rows == 0:
            return
        arrays_to_write = {
            key: np.concatenate(parts)
            for key, parts in prediction_buffer.items()
        }
        output_table = pa.Table.from_arrays(
            [
                pa.array(arrays_to_write["entity_id"].astype(str), type=pa.string()),
                pa.array(
                    arrays_to_write["target_timestamp"].astype("datetime64[ns]"),
                    type=pa.timestamp("ns"),
                ),
                pa.array(arrays_to_write["actual_kwh"].astype(np.float32), type=pa.float32()),
                pa.array(arrays_to_write["predicted_kwh"].astype(np.float32), type=pa.float32()),
                pa.array(arrays_to_write["error_kwh"].astype(np.float32), type=pa.float32()),
                pa.array(arrays_to_write["absolute_error_kwh"].astype(np.float32), type=pa.float32()),
                pa.array(arrays_to_write["squared_error_kwh2"].astype(np.float32), type=pa.float32()),
                pa.array(arrays_to_write["smape_percent"].astype(np.float32), type=pa.float32()),
                pa.array(arrays_to_write["persistence_kwh"].astype(np.float32), type=pa.float32()),
                pa.array(
                    arrays_to_write["daily_seasonal_naive_kwh"].astype(np.float32),
                    type=pa.float32(),
                ),
            ],
            schema=schema,
        )
        writer.write_table(output_table)
        for parts in prediction_buffer.values():
            parts.clear()
        buffered_rows = 0

    entity_count = len(spec.arrays.entity_ids)
    counts = np.zeros(entity_count, dtype=np.int64)
    absolute_sum = np.zeros(entity_count, dtype=np.float64)
    squared_sum = np.zeros(entity_count, dtype=np.float64)
    smape_sum = np.zeros(entity_count, dtype=np.float64)
    persistence_abs_sum = np.zeros(entity_count, dtype=np.float64)
    persistence_sq_sum = np.zeros(entity_count, dtype=np.float64)
    persistence_smape_sum = np.zeros(entity_count, dtype=np.float64)
    daily_abs_sum = np.zeros(entity_count, dtype=np.float64)
    daily_sq_sum = np.zeros(entity_count, dtype=np.float64)
    daily_smape_sum = np.zeros(entity_count, dtype=np.float64)

    total_count = 0
    total_absolute = 0.0
    total_squared = 0.0
    total_smape = 0.0
    total_persistence_absolute = 0.0
    total_persistence_squared = 0.0
    total_persistence_smape = 0.0
    total_daily_absolute = 0.0
    total_daily_squared = 0.0
    total_daily_smape = 0.0
    negative_prediction_count = 0
    inverse_check_max_abs = 0.0
    start = time.perf_counter()

    model.eval()
    try:
        with torch.no_grad():
            for entities, timestamps_ns in index_row_batches(
                index_path=spec.index_path,
                split="test",
                batch_size=settings.batch_size,
                epoch_seed=None,
                target_end=None,
            ):
                batch = reconstruct_batch(
                    entity_ids=entities,
                    target_ns=timestamps_ns,
                    arrays=spec.arrays,
                    scalers=spec.scalers,
                    horizon=settings.horizon,
                    include_raw=True,
                )
                x = move_tensor(batch["x"], device, settings.pin_memory)
                prediction_scaled = model(x)
                if not torch.isfinite(prediction_scaled).all():
                    raise FloatingPointError("Non-finite formal test prediction.")

                prediction_scaled_np = (
                    prediction_scaled.detach().cpu().numpy().reshape(-1).astype(np.float64)
                )
                actual = batch["raw_target"].astype(np.float64)
                means = batch["mean"].astype(np.float64)
                stds = batch["std"].astype(np.float64)
                predicted = prediction_scaled_np * stds + means
                restored_actual = (
                    batch["y"].numpy().reshape(-1).astype(np.float64) * stds + means
                )
                inverse_check_max_abs = max(
                    inverse_check_max_abs,
                    float(np.max(np.abs(restored_actual - actual))),
                )

                persistence = batch["persistence"].astype(np.float64)
                daily = batch["daily"].astype(np.float64)
                error = predicted - actual
                absolute = np.abs(error)
                squared = np.square(error)
                smape = smape_percent(actual, predicted)

                persistence_error = persistence - actual
                persistence_absolute = np.abs(persistence_error)
                persistence_squared = np.square(persistence_error)
                persistence_smape = smape_percent(actual, persistence)

                daily_error = daily - actual
                daily_absolute = np.abs(daily_error)
                daily_squared = np.square(daily_error)
                daily_smape = smape_percent(actual, daily)

                columns = batch["columns"]
                np.add.at(counts, columns, 1)
                np.add.at(absolute_sum, columns, absolute)
                np.add.at(squared_sum, columns, squared)
                np.add.at(smape_sum, columns, smape)
                np.add.at(persistence_abs_sum, columns, persistence_absolute)
                np.add.at(persistence_sq_sum, columns, persistence_squared)
                np.add.at(persistence_smape_sum, columns, persistence_smape)
                np.add.at(daily_abs_sum, columns, daily_absolute)
                np.add.at(daily_sq_sum, columns, daily_squared)
                np.add.at(daily_smape_sum, columns, daily_smape)

                total_count += len(actual)
                total_absolute += float(absolute.sum())
                total_squared += float(squared.sum())
                total_smape += float(smape.sum())
                total_persistence_absolute += float(persistence_absolute.sum())
                total_persistence_squared += float(persistence_squared.sum())
                total_persistence_smape += float(persistence_smape.sum())
                total_daily_absolute += float(daily_absolute.sum())
                total_daily_squared += float(daily_squared.sum())
                total_daily_smape += float(daily_smape.sum())
                negative_prediction_count += int(np.sum(predicted < 0))

                prediction_buffer["entity_id"].append(entities.astype(str))
                prediction_buffer["target_timestamp"].append(timestamps_ns.copy())
                prediction_buffer["actual_kwh"].append(actual.astype(np.float32))
                prediction_buffer["predicted_kwh"].append(predicted.astype(np.float32))
                prediction_buffer["error_kwh"].append(error.astype(np.float32))
                prediction_buffer["absolute_error_kwh"].append(absolute.astype(np.float32))
                prediction_buffer["squared_error_kwh2"].append(squared.astype(np.float32))
                prediction_buffer["smape_percent"].append(smape.astype(np.float32))
                prediction_buffer["persistence_kwh"].append(persistence.astype(np.float32))
                prediction_buffer["daily_seasonal_naive_kwh"].append(daily.astype(np.float32))
                buffered_rows += len(actual)
                if buffered_rows >= parquet_flush_rows:
                    flush_prediction_buffer()
        flush_prediction_buffer()
    finally:
        writer.close()

    synchronise(device)
    elapsed = time.perf_counter() - start
    expected = EXPECTED_H48_COUNTS[spec.name]["test"]
    if total_count != expected:
        temporary_predictions.unlink(missing_ok=True)
        raise AssertionError(
            f"{spec.name}: test observations={total_count:,}; expected={expected:,}."
        )
    if inverse_check_max_abs > 1e-5:
        temporary_predictions.unlink(missing_ok=True)
        raise AssertionError(
            f"Inverse-scaling target check failed: {inverse_check_max_abs}."
        )
    os.replace(temporary_predictions, paths.predictions)

    pooled = pd.DataFrame(
        [
            {
                "strategy": spec.name,
                "seed": settings.seed,
                "horizon": settings.horizon,
                "test_observations": total_count,
                "mae_kwh": total_absolute / total_count,
                "rmse_kwh": math.sqrt(total_squared / total_count),
                "smape_percent": total_smape / total_count,
                "persistence_mae_kwh": total_persistence_absolute / total_count,
                "persistence_rmse_kwh": math.sqrt(
                    total_persistence_squared / total_count
                ),
                "persistence_smape_percent": total_persistence_smape / total_count,
                "daily_seasonal_naive_mae_kwh": total_daily_absolute / total_count,
                "daily_seasonal_naive_rmse_kwh": math.sqrt(
                    total_daily_squared / total_count
                ),
                "daily_seasonal_naive_smape_percent": total_daily_smape / total_count,
                "negative_prediction_count": negative_prediction_count,
                "negative_prediction_rate": negative_prediction_count / total_count,
                "inverse_scaling_target_max_abs_difference": inverse_check_max_abs,
                "test_elapsed_seconds": elapsed,
                "test_samples_per_second": total_count / elapsed,
                "smape_definition": (
                    "200*abs(predicted-actual)/(abs(actual)+abs(predicted)); "
                    "zero when actual=prediction=0"
                ),
            }
        ]
    )
    atomic_csv_dump(paths.pooled_metrics, pooled)

    valid = counts > 0
    if not valid.all():
        missing_entities = np.asarray(spec.arrays.entity_ids, dtype=object)[~valid]
        raise AssertionError(
            f"Entities without formal test rows: {missing_entities[:10].tolist()}"
        )
    household = pd.DataFrame(
        {
            "entity_id": spec.arrays.entity_ids,
            "strategy": spec.name,
            "seed": settings.seed,
            "horizon": settings.horizon,
            "test_observations": counts,
            "mae_kwh": absolute_sum / counts,
            "rmse_kwh": np.sqrt(squared_sum / counts),
            "smape_percent": smape_sum / counts,
            "persistence_mae_kwh": persistence_abs_sum / counts,
            "persistence_rmse_kwh": np.sqrt(persistence_sq_sum / counts),
            "persistence_smape_percent": persistence_smape_sum / counts,
            "daily_seasonal_naive_mae_kwh": daily_abs_sum / counts,
            "daily_seasonal_naive_rmse_kwh": np.sqrt(daily_sq_sum / counts),
            "daily_seasonal_naive_smape_percent": daily_smape_sum / counts,
        }
    )
    atomic_csv_dump(paths.household_metrics, household)

    print(pooled.to_string(index=False), flush=True)
    return pooled.iloc[0].to_dict()


def write_run_metadata(
    paths: RunPaths,
    spec: StrategySpec,
    settings: CanonicalSettings,
    device: torch.device,
    training_summary: dict[str, Any] | None,
    test_summary: dict[str, Any],
) -> None:
    paths.metadata_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Formal forecasting Formal Forecasting Run",
        "",
        f"- status: COMPLETE — PASS",
        f"- completed_utc: {utc_now()}",
        f"- strategy: {spec.name}",
        f"- dataset: {spec.dataset}",
        f"- seed: {settings.seed}",
        f"- horizon: {settings.horizon}",
        f"- valid_for_formal_results: true",
        f"- canonical_config: `{public_repo_path(settings.project_root, settings.config_path)}`",
        f"- canonical_config_sha256: `{settings.config_sha256}`",
        f"- python_executable: `{Path(sys.executable).name}`",
        f"- python_version: {sys.version.split()[0]}",
        f"- pytorch_version: {torch.__version__}",
        f"- cuda_build: {torch.version.cuda}",
        f"- cuda_device: {torch.cuda.get_device_name(device)}",
        f"- batch_size: {settings.batch_size}",
        f"- num_workers: {settings.num_workers}",
        f"- pin_memory: {settings.pin_memory}",
        f"- mixed_precision: {MIXED_PRECISION_ENABLED}",
        f"- test_rows_used_for_training_or_selection: 0",
        "",
        "## Fixed optimizer and scheduler implementation",
        "",
        f"- optimizer: Adam",
        f"- betas: {ADAM_BETAS}",
        f"- epsilon: {ADAM_EPS}",
        f"- weight_decay: {ADAM_WEIGHT_DECAY}",
        f"- amsgrad: {ADAM_AMSGRAD}",
        f"- gradient_clip_max_norm: {settings.gradient_clip}",
        f"- scheduler: ReduceLROnPlateau",
        f"- scheduler_factor: {settings.scheduler_factor}",
        f"- scheduler_patience_checks: {settings.scheduler_patience}",
        f"- scheduler_threshold: {SCHEDULER_THRESHOLD}",
        f"- scheduler_threshold_mode: {SCHEDULER_THRESHOLD_MODE}",
        f"- scheduler_cooldown: {SCHEDULER_COOLDOWN}",
        f"- scheduler_min_lr: {settings.scheduler_min_lr}",
        f"- scheduler_epsilon: {SCHEDULER_EPS}",
        "",
        "## Outputs",
        "",
        f"- predictions: `{public_repo_path(settings.project_root, paths.predictions)}`",
        f"- pooled_metrics: `{public_repo_path(settings.project_root, paths.pooled_metrics)}`",
        f"- household_metrics: `{public_repo_path(settings.project_root, paths.household_metrics)}`",
    ]
    if training_summary is not None:
        lines.extend(
            [
                f"- best_checkpoint: `{public_repo_path(settings.project_root, paths.best_checkpoint)}`",
                f"- epoch_log: `{public_repo_path(settings.project_root, paths.epoch_log)}`",
                f"- learning_curve: `{public_repo_path(settings.project_root, paths.learning_curve)}`",
                "",
                "## Training outcome",
                "",
                f"- best_epoch: {training_summary['best_epoch']}",
                f"- best_validation_mse_scaled: {training_summary['best_validation_mse']}",
                f"- stopping_epoch: {training_summary['stopping_epoch']}",
                f"- stopping_reason: {training_summary['stopping_reason']}",
                f"- extension_decision: `{json.dumps(training_summary['extension_decision'], default=str)}`",
                f"- trainable_layers: {training_summary['trainable_layers']}",
                f"- frozen_layers: {training_summary['frozen_layers']}",
                f"- frozen_best_max_abs_difference: {training_summary['frozen_best_max_abs_difference']}",
                f"- trainable_best_max_abs_difference: {training_summary['trainable_best_max_abs_difference']}",
                f"- matching_source_checkpoint: {training_summary['source_checkpoint']}",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Direct-transfer rule",
                "",
                "- CER parameter updates: 0",
                f"- matching_source_checkpoint: `{public_repo_path(settings.project_root, spec.source_checkpoint_path)}`",
                "- CER training curve: not applicable",
            ]
        )

    lines.extend(
        [
            "",
            "## Formal pooled test metrics",
            "",
            f"- observations: {test_summary['test_observations']}",
            f"- MAE_kWh: {test_summary['mae_kwh']}",
            f"- RMSE_kWh: {test_summary['rmse_kwh']}",
            f"- SMAPE_percent: {test_summary['smape_percent']}",
            "",
            "Scaled validation losses are checkpoint diagnostics only. Formal strategy comparison uses the inverse-transformed original-kWh test metrics.",
        ]
    )
    temporary = paths.metadata.with_suffix(paths.metadata.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, paths.metadata)


def run_direct_transfer(
    spec: StrategySpec,
    settings: CanonicalSettings,
    paths: RunPaths,
    device: torch.device,
) -> tuple[None, dict[str, Any]]:
    heading(f"DIRECT TRANSFER — seed {settings.seed}")
    if spec.initial_state is None or spec.source_checkpoint_path is None:
        raise RuntimeError("Direct transfer lacks a matching source checkpoint.")
    set_global_seed(settings.seed)
    model = GlobalCNNLSTM().to(device)
    model.load_state_dict(spec.initial_state)
    for parameter in model.parameters():
        parameter.requires_grad = False

    atomic_json_dump(
        paths.source_reference,
        {
            "strategy": spec.name,
            "seed": settings.seed,
            "horizon": settings.horizon,
            "matching_source_checkpoint": public_repo_path(settings.project_root, spec.source_checkpoint_path),
            "CER_parameter_updates": 0,
            "config_sha256": settings.config_sha256,
        },
    )
    test_summary = evaluate_test_and_write_predictions(
        model=model,
        spec=spec,
        settings=settings,
        paths=paths,
        device=device,
    )
    return None, test_summary


def validate_required_paths(settings: CanonicalSettings) -> None:
    required = [
        settings.config_path,
        settings.lcl_index_path,
        settings.cer_index_path,
        settings.lcl_scaler_path,
        settings.cer_scaler_path,
        settings.project_root
        / "data/raw/LCL/csv/data_collection/data_tables/consumption_n.csv",
        settings.project_root
        / "data/processed/time_aligned/lcl_full_local_clock_map.parquet",
        settings.project_root
        / "data/processed/time_aligned/cer_canonical_local_grid.parquet",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Formal forecasting required inputs:\n"
            + "\n".join(str(path) for path in missing)
        )


def print_preflight(settings: CanonicalSettings) -> None:
    heading("FORMAL FORECASTING FORMAL RUNNER PREFLIGHT")
    profile = settings.config["execution_profile"]
    print("Canonical config       :", settings.config_path)
    print("Config SHA256          :", settings.config_sha256)
    print("Status                 :", settings.config["status"])
    print("Formal results allowed :", settings.config["valid_for_formal_results"])
    print("Device policy          :", profile["device"])
    print("Horizon                :", settings.horizon)
    print("Batch size             :", settings.batch_size)
    print("Seeds                  :", profile["formal_seed_policy"]["seeds"])
    print("Fine-tuning scope      :", settings.fine_tuning_scope)
    print("Validation             :", profile["formal_validation_frequency"])
    print("Patience               :", settings.early_stopping_patience)
    print("Maximum epochs         :", settings.max_epochs)
    print("Conditional maximum    :", settings.absolute_max_epochs)
    print("Required input paths   : PASS")
    print("Preflight result       : PASS")


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    config_path = (
        args.config.resolve()
        if args.config is not None
        else project_root / "configs/forecasting_h48.yaml"
    )

    if args.preflight_only:
        # Use first formal seed only to validate the canonical schema.
        temporary_text = config_path.read_text(encoding="utf-8")
        temporary_config = yaml.load(temporary_text, Loader=UniqueKeyLoader)
        first_seed = int(
            temporary_config["execution_profile"]["formal_seed_policy"]["seeds"][0]
        )
        settings = load_canonical_settings(
            project_root=project_root,
            config_path=config_path,
            seed=first_seed,
            horizon=args.horizon,
        )
        validate_required_paths(settings)
        print_preflight(settings)
        return

    if args.strategy is None or args.seed is None:
        raise SystemExit("--strategy and --seed are required unless --preflight-only is used.")

    settings = load_canonical_settings(
        project_root=project_root,
        config_path=config_path,
        seed=args.seed,
        horizon=args.horizon,
    )
    validate_required_paths(settings)

    if not torch.cuda.is_available():
        raise RuntimeError("Formal forecasting formal execution requires an allocated CUDA GPU.")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    paths = build_run_paths(
        project_root=project_root,
        horizon=settings.horizon,
        strategy=args.strategy,
        seed=settings.seed,
    )
    trainable = args.strategy != "direct_transfer"
    ensure_fresh_or_resumable(paths, args.resume, trainable=trainable)
    for directory in [paths.model_dir, paths.table_dir, paths.figure_dir, paths.metadata_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    atomic_json_dump(
        paths.status,
        {
            "status": "RUNNING",
            "started_utc": utc_now(),
            "strategy": args.strategy,
            "seed": settings.seed,
            "horizon": settings.horizon,
            "config_sha256": settings.config_sha256,
            "resume": args.resume,
        },
    )

    heading("FORMAL FORECASTING FORMAL FORECASTING")
    print("Timestamp UTC      :", utc_now())
    print("Python executable  :", Path(sys.executable).name)
    print("PyTorch version    :", torch.__version__)
    print("CUDA build         :", torch.version.cuda)
    print("GPU                :", torch.cuda.get_device_name(device))
    print("Strategy           :", args.strategy)
    print("Seed               :", settings.seed)
    print("Horizon            :", settings.horizon)
    print("Canonical config   :", public_repo_path(settings.project_root, settings.config_path))
    print("Config SHA256      :", settings.config_sha256)
    print("Test rows used for training/selection: 0")

    try:
        spec = build_strategy_spec(args.strategy, settings)

        if args.strategy == "direct_transfer":
            training_summary, test_summary = run_direct_transfer(
                spec=spec,
                settings=settings,
                paths=paths,
                device=device,
            )
        else:
            training_summary = train_formal_strategy(
                spec=spec,
                settings=settings,
                paths=paths,
                device=device,
                resume=args.resume,
            )
            model = training_summary.pop("model")
            test_summary = evaluate_test_and_write_predictions(
                model=model,
                spec=spec,
                settings=settings,
                paths=paths,
                device=device,
            )

        write_run_metadata(
            paths=paths,
            spec=spec,
            settings=settings,
            device=device,
            training_summary=training_summary,
            test_summary=test_summary,
        )
        atomic_json_dump(
            paths.status,
            {
                "status": "COMPLETE_PASS",
                "completed_utc": utc_now(),
                "strategy": args.strategy,
                "seed": settings.seed,
                "horizon": settings.horizon,
                "config_sha256": settings.config_sha256,
                "test_rows_used_for_training_or_selection": 0,
                "pooled_metrics": test_summary,
            },
        )
        heading("FORMAL FORECASTING RUN RESULT")
        print("Formal forecasting formal run: PASS")
        print("Strategy           :", args.strategy)
        print("Seed               :", settings.seed)
        print("Predictions        :", public_repo_path(settings.project_root, paths.predictions))
        print("Pooled metrics     :", public_repo_path(settings.project_root, paths.pooled_metrics))
        print("Household metrics  :", public_repo_path(settings.project_root, paths.household_metrics))
        print("Metadata           :", public_repo_path(settings.project_root, paths.metadata))

    except Exception as error:
        atomic_json_dump(
            paths.status,
            {
                "status": "FAILED",
                "failed_utc": utc_now(),
                "strategy": args.strategy,
                "seed": settings.seed,
                "horizon": settings.horizon,
                "config_sha256": settings.config_sha256,
                "error_type": type(error).__name__,
                "error": public_runtime_text(settings.project_root, error),
                "traceback": public_runtime_text(settings.project_root, traceback.format_exc()),
            },
        )
        raise
    finally:
        clear_device(device)


if __name__ == "__main__":
    main()
