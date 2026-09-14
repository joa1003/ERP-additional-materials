from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

GROUP_DESCRIPTIONS = {
    1: "Morning–evening double peak",
    2: "Early evening peak",
    3: "Late evening peak",
    4: "Late night peak",
}

STRATEGY_LABELS = {
    "direct_transfer": "Direct Transfer",
    "fine_tuning": "Fine Tuning",
    "cer_scratch_limited": "CER Scratch Limited",
    "cer_scratch_full": "CER Scratch Full",
}

CRITERION_VALUE_COLUMNS = {
    "mae": "mae_kwh_mean",
    "rmse": "rmse_kwh_mean",
    "smape": "smape_percent_mean",
}

GAIN_COLUMNS = {
    "source_transfer_gain": "source_transfer_gain_mean",
    "fine_tuning_gain": "fine_tuning_gain_mean",
    "full_data_gain": "full_data_gain_mean",
    "direct_vs_full_gain": "direct_vs_full_gain_mean",
}

GAIN_LABELS = {
    "source_transfer_gain": "Source Transfer Gain",
    "fine_tuning_gain": "Fine-Tuning Gain",
    "full_data_gain": "Full-Data Gain",
    "direct_vs_full_gain": "Direct-versus-Full Gain",
}


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalise_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def quantile_summary(series: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    ensure(len(clean) > 0, "Cannot summarise an empty numeric series")
    return {
        "mean": float(clean.mean()),
        "sample_std": float(clean.std(ddof=1)) if len(clean) > 1 else 0.0,
        "median": float(clean.median()),
        "q25": float(clean.quantile(0.25)),
        "q75": float(clean.quantile(0.75)),
        "minimum": float(clean.min()),
        "maximum": float(clean.max()),
    }
