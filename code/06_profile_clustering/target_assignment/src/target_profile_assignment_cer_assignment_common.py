from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    ensure(config_path.is_file(), f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["_config_path"] = str(config_path)
    return config


def resolve_path(config: dict[str, Any], value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    project_root = Path(config["project_root"]).expanduser().resolve()
    return (project_root / path).resolve()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def local_time_label(local_slot: int) -> str:
    ensure(1 <= int(local_slot) <= 48, f"Invalid local slot: {local_slot}")
    minutes = (int(local_slot) - 1) * 30
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def load_fixed_prototypes(config: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray, list[int]]:
    path = resolve_path(config, config["inputs"]["source_prototypes_csv"])
    ensure(path.is_file(), f"Fixed source prototype file not found: {path}")
    frame = pd.read_csv(path)
    required = {
        "k",
        "seed",
        "profile_group",
        "local_slot",
        "prototype_zscore",
        "local_time",
    }
    missing = sorted(required.difference(frame.columns))
    ensure(not missing, f"Prototype file missing columns: {missing}")

    expected_k = int(config["expected"]["source_k"])
    expected_seed = int(config["expected"]["source_seed"])
    ensure(frame["k"].nunique() == 1 and int(frame["k"].iloc[0]) == expected_k,
           f"Expected fixed K={expected_k}")
    ensure(frame["seed"].nunique() == 1 and int(frame["seed"].iloc[0]) == expected_seed,
           f"Expected fixed source seed={expected_seed}")

    groups = sorted(int(value) for value in frame["profile_group"].unique())
    ensure(groups == list(range(1, expected_k + 1)), f"Unexpected profile groups: {groups}")
    ensure(len(frame) == expected_k * 48, f"Expected {expected_k * 48} prototype rows, found {len(frame)}")

    prototype_matrix: list[np.ndarray] = []
    for group in groups:
        part = frame.loc[frame["profile_group"] == group].sort_values("local_slot")
        ensure(part["local_slot"].tolist() == list(range(1, 49)),
               f"Group {group} prototype does not contain slots 1 to 48 exactly once")
        values = part["prototype_zscore"].to_numpy(dtype=np.float64)
        ensure(np.isfinite(values).all(), f"Group {group} prototype contains non-finite values")
        prototype_matrix.append(values)

    return frame, np.vstack(prototype_matrix), groups


def profile_zscore(raw_profiles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ensure(raw_profiles.ndim == 2 and raw_profiles.shape[1] == 48,
           f"Expected profile matrix n x 48, found {raw_profiles.shape}")
    means = raw_profiles.mean(axis=1)
    stds = raw_profiles.std(axis=1, ddof=0)
    ensure(np.isfinite(means).all(), "Profile means contain non-finite values")
    ensure(np.isfinite(stds).all(), "Profile standard deviations contain non-finite values")
    ensure((stds > 0).all(), "At least one receiving profile has zero 48-slot standard deviation")
    zscores = (raw_profiles - means[:, None]) / stds[:, None]
    ensure(np.isfinite(zscores).all(), "Profile z-score matrix contains non-finite values")
    return zscores, means, stds


def compute_dtw_assignments(
    receiving_profiles_z: np.ndarray,
    prototype_matrix: np.ndarray,
    groups: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        from tslearn.metrics import dtw
    except ImportError as exc:  # pragma: no cover - depends on optional runtime package
        raise RuntimeError("tslearn is required because Profile clustering used tslearn DTW") from exc

    ensure(receiving_profiles_z.ndim == 2 and receiving_profiles_z.shape[1] == 48,
           f"Unexpected receiving profile shape: {receiving_profiles_z.shape}")
    ensure(prototype_matrix.shape == (len(groups), 48),
           f"Unexpected prototype matrix shape: {prototype_matrix.shape}")

    distances = np.empty((receiving_profiles_z.shape[0], len(groups)), dtype=np.float64)
    for row_index, profile in enumerate(receiving_profiles_z):
        for group_index, prototype in enumerate(prototype_matrix):
            distances[row_index, group_index] = float(dtw(profile, prototype))

    ensure(np.isfinite(distances).all(), "DTW distance matrix contains non-finite values")
    order = np.argsort(distances, axis=1, kind="stable")
    nearest_index = order[:, 0]
    second_index = order[:, 1]
    nearest_group = np.asarray([groups[index] for index in nearest_index], dtype=np.int64)
    second_group = np.asarray([groups[index] for index in second_index], dtype=np.int64)
    nearest_distance = distances[np.arange(len(distances)), nearest_index]
    second_distance = distances[np.arange(len(distances)), second_index]
    margin = second_distance - nearest_distance
    ratio = np.divide(
        nearest_distance,
        second_distance,
        out=np.full_like(nearest_distance, np.nan),
        where=second_distance > 0,
    )

    ensure((margin >= -1e-12).all(), "At least one distance margin is negative")
    ensure(np.isfinite(ratio).all(), "At least one distance ratio is non-finite")
    ensure(((ratio >= -1e-12) & (ratio <= 1.0 + 1e-12)).all(),
           "Distance ratio must equal nearest / second nearest and lie in [0, 1]")
    return distances, nearest_group, second_group, nearest_distance, second_distance, margin, ratio
