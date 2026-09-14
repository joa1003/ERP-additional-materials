from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


class ProfileClusteringFinalKError(RuntimeError):
    """Raised when a Profile clustering final-K gate fails."""


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ProfileClusteringFinalKError(message)


def load_yaml(path: Path) -> dict[str, Any]:
    ensure(path.is_file(), f"Missing YAML config: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    ensure(isinstance(data, dict), f"YAML root must be a mapping: {path}")
    return data


def resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def slot_columns(columns: list[str]) -> list[str]:
    result = [f"slot_{index:02d}" for index in range(1, 49)]
    ensure(all(column in columns for column in result), "Expected slot_01 ... slot_48 columns")
    return result


def sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return variance ** 0.5
