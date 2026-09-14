#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import os as _erp_os
_ERP_PROJECT_ROOT = Path(_erp_os.environ.get("ERP_PROJECT_ROOT", ".")).expanduser().resolve()
from typing import Any

import numpy as np
import pandas as pd
import scipy

GROUPS = ["G1", "G2", "G3", "G4"]
MAIN_OUTCOMES = [
    "direct_over_limited_relative_mae_gain_pct",
    "direct_over_full_relative_rmse_gain_pct",
]
FINE_AUDIT_OUTCOME = "fine_tuning_over_direct_relative_mae_gain_pct"


class SurveyGainAssociationsError(RuntimeError):
    pass


@dataclass(frozen=True)
class VariableSpec:
    domain: str
    variable: str
    display_name: str
    analysis_role: str
    family: str
    category_order: tuple[str, ...]
    category_labels: tuple[str, ...]
    denominator_note: str


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def normalize_category(value: Any) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return None
        if float(value).is_integer():
            return str(int(value))
        return str(float(value))
    text = str(value).strip()
    if not text or text.lower() in {"nan", "<na>"}:
        return None
    try:
        num = float(text)
        if np.isfinite(num) and num.is_integer():
            return str(int(num))
    except ValueError:
        pass
    return text


def markdown_escape(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(markdown_escape(c) for c in cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in df.itertuples(index=False, name=None):
        cells: list[str] = []
        for value in row:
            if pd.isna(value):
                cells.append("")
            elif isinstance(value, (float, np.floating)):
                cells.append(format(float(value), floatfmt))
            elif isinstance(value, (int, np.integer)):
                cells.append(str(int(value)))
            else:
                cells.append(markdown_escape(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def parse_variable_specs(config: dict[str, Any]) -> list[VariableSpec]:
    specs: list[VariableSpec] = []
    for item in config["analysis_spec"]:
        order = tuple(str(x) for x in item["category_order"])
        labels = tuple(str(x) for x in item.get("category_labels", item["category_order"]))
        if len(order) != len(labels):
            raise SurveyGainAssociationsError(f"Category-label length mismatch: {item['variable']}")
        specs.append(
            VariableSpec(
                domain=str(item["domain"]),
                variable=str(item["variable"]),
                display_name=str(item["display_name"]),
                analysis_role=str(item["analysis_role"]),
                family=str(item["family"]),
                category_order=order,
                category_labels=labels,
                denominator_note=str(item.get("denominator_note", "")),
            )
        )
    return specs


def resolve_recorded_path(recorded: str, root: Path) -> Path:
    raw = Path(str(recorded))
    if raw.is_file():
        return raw.resolve()
    text = str(recorded).replace("\\", "/")
    for anchor in ("outputs/", "documentation/", "logs/"):
        pos = text.find(anchor)
        if pos >= 0:
            candidate = (root / text[pos:]).resolve()
            if candidate.is_file():
                return candidate
    return raw


def normalize_inventory_table(inv: pd.DataFrame, label: str) -> pd.DataFrame:
    """Normalize governed upstream inventory column aliases.

    Survey recoding uses path,size_bytes,sha256, while the validated local-source-support
    inventory uses relative_path,bytes,sha256. Both schemas carry the same
    governed information and are accepted without weakening size/hash gates.
    """
    lookup = {str(col).strip().lower(): str(col) for col in inv.columns}
    aliases = {
        "path": ("path", "relative_path"),
        "size_bytes": ("size_bytes", "bytes"),
        "sha256": ("sha256",),
    }
    rename: dict[str, str] = {}
    for canonical, candidates in aliases.items():
        source = next((lookup[name] for name in candidates if name in lookup), None)
        if source is None:
            raise SurveyGainAssociationsError(
                f"{label} inventory schema mismatch: expected aliases {aliases}; "
                f"actual columns={list(inv.columns)}"
            )
        rename[source] = canonical
    out = inv.rename(columns=rename)[["path", "size_bytes", "sha256"]].copy()
    out["path"] = out["path"].astype(str).str.strip()
    out["size_bytes"] = pd.to_numeric(out["size_bytes"], errors="raise").astype("int64")
    out["sha256"] = out["sha256"].astype(str).str.strip().str.lower()
    if out.empty:
        raise SurveyGainAssociationsError(f"{label} inventory is empty")
    if out["path"].eq("").any():
        raise SurveyGainAssociationsError(f"{label} inventory contains an empty path")
    if (out["size_bytes"] < 0).any():
        raise SurveyGainAssociationsError(f"{label} inventory contains a negative byte count")
    if not out["sha256"].str.fullmatch(r"[0-9a-f]{64}").all():
        raise SurveyGainAssociationsError(f"{label} inventory contains an invalid SHA-256 value")
    return out


# -----------------------------------------------------------------------------
# Formal upstream gates
# -----------------------------------------------------------------------------

def require_exact_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SurveyGainAssociationsError(f"Missing {label}: {path}")


def validate_survey_recoding_and_11c(config: dict[str, Any], root: Path) -> dict[str, Path]:
    inputs = config["formal_inputs"]
    paths = {
        "survey_recoding_status": root / inputs["survey_recoding_status"],
        "survey_recoding_inventory": root / inputs["survey_recoding_inventory"],
        "survey_recoding_cer_recoded": root / inputs["survey_recoding_cer_recoded"],
        "survey_recoding_analysis_lock": root / inputs["survey_recoding_analysis_lock"],
        "survey_group_composition_status": root / inputs["survey_group_composition_status"],
        "survey_group_composition_inventory": root / inputs["survey_group_composition_inventory"],
        "survey_group_composition_decision_note": root / inputs["survey_group_composition_decision_note"],
    }
    for key, path in paths.items():
        require_exact_file(path, key)

    analysis_lock = pd.read_csv(paths["survey_recoding_analysis_lock"], low_memory=False)
    required_lock_columns = {"dataset", "analysis_variable", "analysis_role"}
    if not required_lock_columns.issubset(analysis_lock.columns):
        raise SurveyGainAssociationsError("Survey recoding analysis-variable lock schema mismatch")
    cer_lock = analysis_lock[analysis_lock["dataset"].astype(str).str.upper().eq("CER")].copy()
    locked_main: set[str] = set()
    for value in cer_lock.loc[cer_lock["analysis_role"].astype(str).eq("main"), "analysis_variable"]:
        locked_main.update(x for x in str(value).split("|") if x)
    expected_main = {
        "household_composition_common", "daytime_presence_category",
        "electric_thermal_load", "dwelling_scale_main",
        "double_glazing_category", "attic_insulation_category",
        "wall_insulation_category", "energy_vulnerability",
    }
    if locked_main != expected_main:
        raise SurveyGainAssociationsError(f"Survey recoding locked CER main-variable set mismatch: {locked_main ^ expected_main}")
    locked_secondary = set(cer_lock.loc[
        cer_lock["analysis_role"].astype(str).eq("secondary_sensitivity"), "analysis_variable"
    ].astype(str))
    if locked_secondary != {"economic_position_sensitivity"}:
        raise SurveyGainAssociationsError("Survey recoding secondary-sensitivity lock mismatch")

    b_status = read_json(paths["survey_recoding_status"])
    if b_status.get("computational_status") != "COMPLETE_PASS":
        raise SurveyGainAssociationsError("Survey recoding status is not COMPLETE_PASS")
    if b_status.get("analysis_gate_status") != "PASS_TO_SURVEY_GROUP_COMPOSITION_SPECIFICATION":
        raise SurveyGainAssociationsError("Survey recoding gate is not PASS_TO_SURVEY_GROUP_COMPOSITION_SPECIFICATION")
    if b_status.get("cer_formal_n") != int(config["expected"]["cer_formal_meters"]):
        raise SurveyGainAssociationsError("Survey recoding CER population mismatch")
    if b_status.get("cer_core_main_variable_n") != 8:
        raise SurveyGainAssociationsError("Survey recoding must lock eight CER main variables")
    if b_status.get("thermal_efficiency_main_status") != "DEPRECATED_NOT_FOR_FORMAL_ANALYSIS":
        raise SurveyGainAssociationsError("Survey recoding thermal semantic gate mismatch")
    for key in (
        "training_performed", "predictions_modified", "forecasting_inputs_modified",
        "clustering_modified", "regression_performed", "ProfileInterpretation_outcomes_read",
        "outcome_dependent_variable_selection",
    ):
        if b_status.get(key) is not False:
            raise SurveyGainAssociationsError(f"Survey recoding governance flag must be False: {key}")

    c_status = read_json(paths["survey_group_composition_status"])
    if c_status.get("computational_status") != "COMPLETE_PASS":
        raise SurveyGainAssociationsError("Survey group composition status is not COMPLETE_PASS")
    if c_status.get("analysis_gate_status") != "PASS_TO_SURVEY_GAIN_ASSOCIATIONS_SPECIFICATION":
        raise SurveyGainAssociationsError("Survey group composition gate is not PASS_TO_SURVEY_GAIN_ASSOCIATIONS_SPECIFICATION")
    if c_status.get("cer_formal_n") != int(config["expected"]["cer_formal_meters"]):
        raise SurveyGainAssociationsError("Survey group composition CER population mismatch")
    if c_status.get("cer_main_test_n") != 8:
        raise SurveyGainAssociationsError("Survey group composition must contain eight CER main tests")
    if c_status.get("reported_thermal_feature_count_role") != "SUPPLEMENTARY_DESCRIPTIVE_ONLY":
        raise SurveyGainAssociationsError("Survey group composition supplementary thermal-count gate mismatch")
    if c_status.get("ProfileInterpretation_outcomes_read") is not False:
        raise SurveyGainAssociationsError("Survey group composition unexpectedly read Profile interpretation outcomes")

    note = paths["survey_group_composition_decision_note"].read_text(encoding="utf-8")
    if "PASS_TO_SURVEY_GAIN_ASSOCIATIONS_SPECIFICATION" not in note:
        raise SurveyGainAssociationsError("Survey group composition decision note missing PASS_TO_SURVEY_GAIN_ASSOCIATIONS_SPECIFICATION")
    if c_status.get("causal_inference_performed") is not False:
        raise SurveyGainAssociationsError("Survey group composition causal-inference governance gate mismatch")
    if c_status.get("thermal_efficiency_main_status") != "DEPRECATED_NOT_FOR_FORMAL_ANALYSIS":
        raise SurveyGainAssociationsError("Survey group composition thermal-efficiency semantic gate mismatch")

    # Verify the exact upstream files used by this analysis.
    for inventory_key, used_keys in (
        ("survey_recoding_inventory", ("survey_recoding_status", "survey_recoding_cer_recoded", "survey_recoding_analysis_lock")),
        ("survey_group_composition_inventory", ("survey_group_composition_status", "survey_group_composition_decision_note")),
    ):
        inv = normalize_inventory_table(
            pd.read_csv(paths[inventory_key], low_memory=False),
            f"Upstream {inventory_key}",
        )
        indexed: dict[Path, pd.Series] = {}
        for _, row in inv.iterrows():
            indexed[resolve_recorded_path(str(row["path"]), root)] = row
        for used_key in used_keys:
            target = paths[used_key].resolve()
            row = indexed.get(target)
            if row is None:
                raise SurveyGainAssociationsError(f"Upstream inventory missing required file: {target}")
            if int(row["size_bytes"]) != target.stat().st_size:
                raise SurveyGainAssociationsError(f"Upstream size mismatch: {target}")
            if str(row["sha256"]) != sha256_file(target):
                raise SurveyGainAssociationsError(f"Upstream hash mismatch: {target}")

    return paths


def find_single_file(directory: Path, pattern: str, label: str) -> Path:
    if not directory.is_dir():
        raise SurveyGainAssociationsError(f"Missing {label} directory: {directory}")
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise SurveyGainAssociationsError(f"Expected exactly one {label} matching {pattern} in {directory}; found {matches}")
    return matches[0]


def status_is_complete(status: dict[str, Any]) -> bool:
    candidates = [
        status.get("computational_status"), status.get("status"), status.get("final_status"),
        status.get("analysis_status"),
    ]
    return any(str(x).upper() in {"COMPLETE_PASS", "COMPLETE", "PASS"} for x in candidates if x is not None)


def infer_column(columns: list[str], aliases: tuple[str, ...], label: str) -> str:
    lower = {str(c).lower(): str(c) for c in columns}
    for alias in aliases:
        if alias.lower() in lower:
            return lower[alias.lower()]
    for col in columns:
        lc = str(col).lower()
        if any(alias.lower() in lc for alias in aliases):
            return str(col)
    raise SurveyGainAssociationsError(f"Could not infer {label} column from {columns}")


def infer_strategy_error_column(columns: list[str], required_tokens: tuple[str, ...]) -> str:
    candidates: list[str] = []
    for col in columns:
        lc = str(col).lower()
        if all(token in lc for token in required_tokens):
            if any(block in lc for block in ("gain", "relative", "difference", "delta", "positive")):
                continue
            candidates.append(str(col))
    # Prefer explicit error columns, then metric-value columns.
    candidates.sort(key=lambda c: ("error" not in c.lower(), "metric" not in c.lower(), len(c)))
    if not candidates:
        raise SurveyGainAssociationsError(f"Could not infer strategy error column for tokens={required_tokens}")
    return candidates[0]


def normalize_local_source_support_outcome_table(path: Path, expected_meters: int) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        raw = pd.read_parquet(path)
    else:
        raw = pd.read_csv(path, low_memory=False)
    cols = [str(c) for c in raw.columns]
    meter_col = infer_column(cols, ("meter_id", "entity_id"), "meter ID")
    horizon_col = infer_column(cols, ("horizon", "lead", "h"), "horizon")
    criterion_col = infer_column(cols, ("criterion", "metric"), "criterion")
    direct_col = infer_strategy_error_column(cols, ("direct",))
    fine_col = infer_strategy_error_column(cols, ("fine",))
    limited_col = infer_strategy_error_column(cols, ("limited",))
    full_col = infer_strategy_error_column(cols, ("full",))

    out = raw[[meter_col, horizon_col, criterion_col, direct_col, fine_col, limited_col, full_col]].copy()
    out.columns = [
        "meter_id", "horizon", "criterion", "direct_transfer_error",
        "fine_tuning_error", "cer_scratch_limited_error", "cer_scratch_full_error",
    ]
    out["meter_id"] = out["meter_id"].astype(str)
    horizon_text = out["horizon"].astype(str).str.extract(r"(\d+)", expand=False)
    out["horizon"] = pd.to_numeric(horizon_text, errors="coerce").astype("Int64")
    out["criterion"] = out["criterion"].astype(str).str.upper().str.strip()
    for col in (
        "direct_transfer_error", "fine_tuning_error", "cer_scratch_limited_error",
        "cer_scratch_full_error",
    ):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out[out["criterion"].isin({"MAE", "RMSE", "SMAPE"})].copy()
    if out[["meter_id", "horizon", "criterion"]].duplicated().any():
        raise SurveyGainAssociationsError(f"Local source support outcome table has duplicate meter/horizon/criterion keys: {path}")
    if out["meter_id"].nunique() != expected_meters:
        raise SurveyGainAssociationsError(f"Local source support outcome table meter count mismatch: {path}")
    if set(out["horizon"].dropna().astype(int)) != {1, 12, 48}:
        raise SurveyGainAssociationsError(f"Local source support outcome table horizon mismatch: {path}")
    expected_rows = expected_meters * 3 * 3
    if len(out) != expected_rows:
        raise SurveyGainAssociationsError(f"Local source support normalized outcome rows={len(out)}, expected={expected_rows}: {path}")
    if out[[
        "direct_transfer_error", "fine_tuning_error", "cer_scratch_limited_error",
        "cer_scratch_full_error",
    ]].isna().any().any():
        raise SurveyGainAssociationsError(f"Local source support outcome errors contain missing values: {path}")
    if (out[[
        "direct_transfer_error", "fine_tuning_error", "cer_scratch_limited_error",
        "cer_scratch_full_error",
    ]] < 0).any().any():
        raise SurveyGainAssociationsError(f"Local source support outcome errors contain negative values: {path}")
    return out


def discover_local_source_support_inputs(config: dict[str, Any], root: Path) -> tuple[Path, Path, Path, pd.DataFrame]:
    table_root = root / config["formal_inputs"]["local_source_support_table_root"]
    meta_root = root / config["formal_inputs"]["local_source_support_metadata_root"]
    status_path = find_single_file(meta_root, "*status.json", "Local source support status")
    inventory_path = find_single_file(meta_root, "*output_inventory.csv", "Local source support output inventory")
    status = read_json(status_path)
    if not status_is_complete(status):
        raise SurveyGainAssociationsError(f"Local source support status is not complete: {status_path}")
    for key in ("training_performed", "predictions_modified"):
        if key in status and status.get(key) is not False:
            raise SurveyGainAssociationsError(f"Local source support governance flag must be False: {key}")

    candidates = sorted(list(table_root.glob("*.csv")) + list(table_root.glob("*.parquet")))
    successful: list[tuple[Path, pd.DataFrame]] = []
    errors: list[str] = []
    for path in candidates:
        try:
            normalized = normalize_local_source_support_outcome_table(path, int(config["expected"]["cer_formal_meters"]))
            successful.append((path, normalized))
        except Exception as exc:  # candidate-screening only
            errors.append(f"{path.name}: {exc}")
    if len(successful) != 1:
        raise SurveyGainAssociationsError(
            "Local source support outcome discovery expected exactly one valid 8,361-row cross-strategy table; "
            f"found {[p.name for p, _ in successful]}. Candidate diagnostics: {errors}"
        )
    outcome_path, outcomes = successful[0]

    inv = normalize_inventory_table(
        pd.read_csv(inventory_path, low_memory=False),
        "Local source support output",
    )
    indexed: dict[Path, pd.Series] = {}
    for _, row in inv.iterrows():
        indexed[resolve_recorded_path(str(row["path"]), root)] = row
    for target in (status_path.resolve(), outcome_path.resolve()):
        row = indexed.get(target)
        if row is None:
            raise SurveyGainAssociationsError(f"Local source support inventory missing required file: {target}")
        if int(row["size_bytes"]) != target.stat().st_size:
            raise SurveyGainAssociationsError(f"Local source support size mismatch: {target}")
        if str(row["sha256"]) != sha256_file(target):
            raise SurveyGainAssociationsError(f"Local source support hash mismatch: {target}")
    return status_path, inventory_path, outcome_path, outcomes


# -----------------------------------------------------------------------------
# Outcome construction and survey join
# -----------------------------------------------------------------------------

def build_meter_outcomes(outcomes: pd.DataFrame) -> pd.DataFrame:
    h12 = outcomes[outcomes["horizon"].astype(int) == 12].copy()
    pivot = h12.set_index(["meter_id", "criterion"])[[
        "direct_transfer_error", "fine_tuning_error", "cer_scratch_limited_error",
        "cer_scratch_full_error",
    ]].unstack("criterion")
    pivot.columns = [f"{error}_{criterion.lower()}" for error, criterion in pivot.columns]
    pivot = pivot.reset_index()
    required = [
        "direct_transfer_error_mae", "fine_tuning_error_mae", "cer_scratch_limited_error_mae",
        "cer_scratch_full_error_rmse", "direct_transfer_error_rmse",
    ]
    missing = [c for c in required if c not in pivot.columns]
    if missing:
        raise SurveyGainAssociationsError(f"Missing h=12 error columns: {missing}")
    if (pivot["cer_scratch_limited_error_mae"] <= 0).any():
        raise SurveyGainAssociationsError("Limited MAE denominator must be positive")
    if (pivot["cer_scratch_full_error_rmse"] <= 0).any():
        raise SurveyGainAssociationsError("Full RMSE denominator must be positive")
    if (pivot["direct_transfer_error_mae"] <= 0).any():
        raise SurveyGainAssociationsError("Direct MAE denominator must be positive")

    pivot[MAIN_OUTCOMES[0]] = 100.0 * (
        pivot["cer_scratch_limited_error_mae"] - pivot["direct_transfer_error_mae"]
    ) / pivot["cer_scratch_limited_error_mae"]
    pivot[MAIN_OUTCOMES[1]] = 100.0 * (
        pivot["cer_scratch_full_error_rmse"] - pivot["direct_transfer_error_rmse"]
    ) / pivot["cer_scratch_full_error_rmse"]
    pivot[FINE_AUDIT_OUTCOME] = 100.0 * (
        pivot["direct_transfer_error_mae"] - pivot["fine_tuning_error_mae"]
    ) / pivot["direct_transfer_error_mae"]
    return pivot.sort_values("meter_id").reset_index(drop=True)


def load_and_validate_survey(path: Path, config: dict[str, Any], specs: list[VariableSpec]) -> pd.DataFrame:
    survey = pd.read_csv(path, low_memory=False, dtype={"meter_id": str, "entity_id": str})
    meter_col = "meter_id" if "meter_id" in survey.columns else "entity_id"
    survey = survey.rename(columns={meter_col: "meter_id"})
    survey["meter_id"] = survey["meter_id"].astype(str)
    if len(survey) != int(config["expected"]["cer_formal_meters"]):
        raise SurveyGainAssociationsError("Survey recoding CER recoded row count mismatch")
    if survey["meter_id"].duplicated().any():
        raise SurveyGainAssociationsError("Survey recoding CER recoded has duplicate meter IDs")
    if "profile_group" not in survey.columns:
        raise SurveyGainAssociationsError("Survey recoding CER recoded missing profile_group")
    counts = survey["profile_group"].astype(str).value_counts().to_dict()
    for group, expected in config["expected"]["cer_group_counts"].items():
        if int(counts.get(group, -1)) != int(expected):
            raise SurveyGainAssociationsError(f"CER group count mismatch: {group}")
    for spec in specs:
        if spec.variable not in survey.columns:
            raise SurveyGainAssociationsError(f"Survey recoding CER recoded missing locked variable: {spec.variable}")
        normalized = survey[spec.variable].map(normalize_category)
        observed = set(normalized.dropna())
        allowed = set(spec.category_order)
        if not observed.issubset(allowed):
            raise SurveyGainAssociationsError(
                f"Unexpected categories for {spec.variable}: {sorted(observed - allowed)}"
            )
        survey[spec.variable] = normalized
    return survey


def join_survey_outcomes(survey: pd.DataFrame, meter_outcomes: pd.DataFrame) -> pd.DataFrame:
    if set(survey["meter_id"]) != set(meter_outcomes["meter_id"]):
        missing_survey = sorted(set(meter_outcomes["meter_id"]) - set(survey["meter_id"]))[:10]
        missing_outcomes = sorted(set(survey["meter_id"]) - set(meter_outcomes["meter_id"]))[:10]
        raise SurveyGainAssociationsError(
            f"Survey/outcome meter-set mismatch; missing survey={missing_survey}, missing outcome={missing_outcomes}"
        )
    merged = survey.merge(meter_outcomes, on="meter_id", how="inner", validate="one_to_one")
    if len(merged) != len(survey):
        raise SurveyGainAssociationsError("Survey/outcome join row-count mismatch")
    return merged.sort_values("meter_id").reset_index(drop=True)


# -----------------------------------------------------------------------------
# Statistical helpers
# -----------------------------------------------------------------------------

def design_matrix(groups: pd.Series, category: pd.Series | None, levels: tuple[str, ...] | None) -> np.ndarray:
    n = len(groups)
    cols = [np.ones(n, dtype=float)]
    group_text = groups.astype(str).to_numpy()
    for group in GROUPS[1:]:
        cols.append((group_text == group).astype(float))
    if category is not None and levels is not None:
        cat = category.astype(str).to_numpy()
        for level in levels[1:]:
            cols.append((cat == level).astype(float))
    return np.column_stack(cols)


def ols_predict(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    beta = np.linalg.pinv(X_train) @ Y_train
    return X_test @ beta


def sse_from_predictions(Y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.sum((Y - pred) ** 2, axis=0)


def r2_from_predictions(Y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    sse = sse_from_predictions(Y, pred)
    centered = Y - np.mean(Y, axis=0, keepdims=True)
    sst = np.sum(centered ** 2, axis=0)
    return np.where(sst > 0, 1.0 - sse / sst, np.nan)


def fit_in_sample(groups: pd.Series, category: pd.Series, levels: tuple[str, ...], Y: np.ndarray) -> dict[str, np.ndarray]:
    X_base = design_matrix(groups, None, None)
    X_aug = design_matrix(groups, category, levels)
    pred_base = ols_predict(X_base, Y, X_base)
    pred_aug = ols_predict(X_aug, Y, X_aug)
    sse_base = sse_from_predictions(Y, pred_base)
    sse_aug = sse_from_predictions(Y, pred_aug)
    partial_r2 = np.where(sse_base > 0, np.maximum(0.0, (sse_base - sse_aug) / sse_base), np.nan)
    return {
        "sse_base": sse_base,
        "sse_aug": sse_aug,
        "partial_r2": partial_r2,
        "r2_base": r2_from_predictions(Y, pred_base),
        "r2_aug": r2_from_predictions(Y, pred_aug),
        "pred_base": pred_base,
        "pred_aug": pred_aug,
    }


def permute_within_groups(values: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = values.copy()
    for group in GROUPS:
        idx = np.flatnonzero(groups == group)
        if len(idx):
            out[idx] = values[rng.permutation(idx)]
    return out


def permutation_partial_r2(
    groups: pd.Series,
    category: pd.Series,
    levels: tuple[str, ...],
    Y: np.ndarray,
    observed_partial_r2: np.ndarray,
    n_perm: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Permutation test for the survey block after fixed profile group.

    Survey labels are permuted within fixed G1-G4. The implementation uses
    Frisch-Waugh-Lovell residualisation: both outcomes and category dummies
    are residualised against group first, so each permutation only solves a
    small category-block normal equation. This is algebraically equivalent
    to refitting the full group-plus-survey OLS model and is substantially
    faster for 20,000 governed permutations.
    """
    group_arr = groups.astype(str).to_numpy()
    cat_arr = category.astype(str).to_numpy()
    group_indices = [np.flatnonzero(group_arr == group) for group in GROUPS]

    # Base-model residuals are outcome deviations from their fixed-group means.
    residual_y = Y.astype(float).copy()
    for idx in group_indices:
        residual_y[idx] -= residual_y[idx].mean(axis=0, keepdims=True)
    base_sse = np.sum(residual_y ** 2, axis=0)
    if np.any(base_sse <= 0):
        raise SurveyGainAssociationsError("Permutation base SSE must be positive")

    level_to_code = {level: i for i, level in enumerate(levels)}
    try:
        original_codes = np.array([level_to_code[x] for x in cat_arr], dtype=np.int16)
    except KeyError as exc:
        raise SurveyGainAssociationsError(f"Unexpected permutation category: {exc}") from exc
    non_reference_codes = np.arange(1, len(levels), dtype=np.int16)

    def block_partial_r2(codes: np.ndarray) -> np.ndarray:
        dummies = (codes[:, None] == non_reference_codes[None, :]).astype(float)
        for idx in group_indices:
            dummies[idx] -= dummies[idx].mean(axis=0, keepdims=True)
        gram = dummies.T @ dummies
        cross = dummies.T @ residual_y
        beta = np.linalg.pinv(gram, rcond=1e-12) @ cross
        explained = np.sum(cross * beta, axis=0)
        return np.clip(explained / base_sse, 0.0, 1.0)

    computed_observed = block_partial_r2(original_codes)
    if not np.allclose(computed_observed, observed_partial_r2, atol=1e-10, rtol=1e-8):
        raise SurveyGainAssociationsError(
            f"Permutation FWL statistic does not reproduce fitted partial R2: "
            f"{computed_observed} versus {observed_partial_r2}"
        )

    exceed = np.zeros(Y.shape[1], dtype=int)
    rng = np.random.default_rng(seed)
    permuted = original_codes.copy()
    for _ in range(n_perm):
        for idx in group_indices:
            permuted[idx] = rng.permutation(original_codes[idx])
        perm_partial = block_partial_r2(permuted)
        exceed += perm_partial >= (observed_partial_r2 - 1e-15)
    p = (exceed + 1.0) / (n_perm + 1.0)
    return p, exceed


def make_repeated_folds(groups: np.ndarray, n_splits: int, rng: np.random.Generator) -> np.ndarray:
    folds = np.empty(len(groups), dtype=int)
    for group in GROUPS:
        idx = np.flatnonzero(groups == group)
        shuffled = rng.permutation(idx)
        folds[shuffled] = np.arange(len(shuffled)) % n_splits
    return folds


def repeated_cv(
    groups: pd.Series,
    category: pd.Series,
    levels: tuple[str, ...],
    Y: np.ndarray,
    repeats: int,
    folds: int,
    seed: int,
) -> pd.DataFrame:
    group_arr = groups.astype(str).to_numpy()
    cat_arr = category.astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    for repeat in range(repeats):
        rng = np.random.default_rng(seed + repeat * 1009)
        fold_id = make_repeated_folds(group_arr, folds, rng)
        pred_base = np.full_like(Y, np.nan, dtype=float)
        pred_aug = np.full_like(Y, np.nan, dtype=float)
        for fold in range(folds):
            test = fold_id == fold
            train = ~test
            Xb_train = design_matrix(pd.Series(group_arr[train]), None, None)
            Xb_test = design_matrix(pd.Series(group_arr[test]), None, None)
            Xa_train = design_matrix(pd.Series(group_arr[train]), pd.Series(cat_arr[train]), levels)
            Xa_test = design_matrix(pd.Series(group_arr[test]), pd.Series(cat_arr[test]), levels)
            pred_base[test] = ols_predict(Xb_train, Y[train], Xb_test)
            pred_aug[test] = ols_predict(Xa_train, Y[train], Xa_test)
        r2_base = r2_from_predictions(Y, pred_base)
        r2_aug = r2_from_predictions(Y, pred_aug)
        for j, outcome in enumerate(MAIN_OUTCOMES):
            rows.append({
                "repeat": repeat + 1,
                "outcome": outcome,
                "cv_r2_group_only": float(r2_base[j]),
                "cv_r2_group_plus_survey": float(r2_aug[j]),
                "incremental_cv_r2": float(r2_aug[j] - r2_base[j]),
            })
    return pd.DataFrame(rows)


def bh_adjust(p_values: pd.Series) -> pd.Series:
    p = p_values.astype(float).to_numpy()
    order = np.argsort(p)
    ranked = p[order]
    m = len(p)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return pd.Series(out, index=p_values.index)


def category_summary(
    df: pd.DataFrame,
    spec: VariableSpec,
    outcome: str,
    group_adjusted_residual: np.ndarray,
) -> pd.DataFrame:
    working = df[["profile_group", spec.variable, outcome]].copy()
    working["group_adjusted_residual_pct"] = group_adjusted_residual
    rows: list[dict[str, Any]] = []
    label_map = dict(zip(spec.category_order, spec.category_labels))
    for level in spec.category_order:
        sub = working[working[spec.variable] == level]
        values = sub[outcome].astype(float)
        residual = sub["group_adjusted_residual_pct"].astype(float)
        group_counts = sub["profile_group"].astype(str).value_counts().to_dict()
        rows.append({
            "domain": spec.domain,
            "variable": spec.variable,
            "display_name": spec.display_name,
            "analysis_role": spec.analysis_role,
            "outcome": outcome,
            "category": level,
            "category_label": label_map[level],
            "n": len(sub),
            "G1_n": int(group_counts.get("G1", 0)),
            "G2_n": int(group_counts.get("G2", 0)),
            "G3_n": int(group_counts.get("G3", 0)),
            "G4_n": int(group_counts.get("G4", 0)),
            "raw_mean_gain_pct": float(values.mean()) if len(sub) else np.nan,
            "raw_median_gain_pct": float(values.median()) if len(sub) else np.nan,
            "raw_q1_gain_pct": float(values.quantile(0.25)) if len(sub) else np.nan,
            "raw_q3_gain_pct": float(values.quantile(0.75)) if len(sub) else np.nan,
            "positive_gain_share": float((values > 0).mean()) if len(sub) else np.nan,
            "group_adjusted_mean_residual_pct": float(residual.mean()) if len(sub) else np.nan,
            "group_adjusted_median_residual_pct": float(residual.median()) if len(sub) else np.nan,
            "group_adjusted_q1_residual_pct": float(residual.quantile(0.25)) if len(sub) else np.nan,
            "group_adjusted_q3_residual_pct": float(residual.quantile(0.75)) if len(sub) else np.nan,
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Formal analysis
# -----------------------------------------------------------------------------

def run_analysis(config: dict[str, Any], root: Path, write_outputs: bool) -> dict[str, Any]:
    specs = parse_variable_specs(config)
    if len([s for s in specs if s.analysis_role == "main"]) != 8:
        raise SurveyGainAssociationsError("Exactly eight main survey variables must be analysed")
    if len([s for s in specs if s.analysis_role == "secondary_sensitivity"]) != 1:
        raise SurveyGainAssociationsError("Exactly one secondary survey variable must be analysed")
    if any(s.variable in {"reported_thermal_feature_count", "thermal_efficiency_main", "thermal_efficiency_proxy_count"} for s in specs):
        raise SurveyGainAssociationsError("Deprecated or supplementary thermal count must not enter Survey gain associations models")

    upstream_paths = validate_survey_recoding_and_11c(config, root)
    t10_status, t10_inventory, t10_outcome_path, t10_outcomes = discover_local_source_support_inputs(config, root)
    meter_outcomes = build_meter_outcomes(t10_outcomes)
    survey = load_and_validate_survey(upstream_paths["survey_recoding_cer_recoded"], config, specs)
    joined = join_survey_outcomes(survey, meter_outcomes)

    main_model_rows: list[dict[str, Any]] = []
    secondary_model_rows: list[dict[str, Any]] = []
    main_category_frames: list[pd.DataFrame] = []
    secondary_category_frames: list[pd.DataFrame] = []
    fine_frames: list[pd.DataFrame] = []
    cv_frames: list[pd.DataFrame] = []
    sample_rows: list[dict[str, Any]] = []

    n_perm = int(config["statistics"]["within_group_permutations"])
    perm_seed = int(config["statistics"]["permutation_seed"])
    cv_repeats = int(config["statistics"]["cv_repeats"])
    cv_folds = int(config["statistics"]["cv_folds"])
    cv_seed = int(config["statistics"]["cv_seed"])

    for spec_index, spec in enumerate(specs):
        sub = joined[joined[spec.variable].notna()].copy()
        sub[spec.variable] = sub[spec.variable].astype(str)
        if len(sub) == 0:
            raise SurveyGainAssociationsError(f"No available rows for {spec.variable}")
        observed = set(sub[spec.variable])
        if not observed.issubset(set(spec.category_order)):
            raise SurveyGainAssociationsError(f"Unexpected categories after filtering for {spec.variable}")
        missing_levels = set(spec.category_order) - observed
        if missing_levels:
            raise SurveyGainAssociationsError(f"Locked category absent from formal sample for {spec.variable}: {missing_levels}")
        group_counts = sub["profile_group"].astype(str).value_counts().to_dict()
        category_counts = sub[spec.variable].value_counts().to_dict()
        cross = pd.crosstab(sub["profile_group"], sub[spec.variable])
        min_cross = int(cross.to_numpy().min()) if cross.size else 0
        sample_rows.append({
            "domain": spec.domain,
            "variable": spec.variable,
            "display_name": spec.display_name,
            "analysis_role": spec.analysis_role,
            "family": spec.family,
            "available_n": len(sub),
            "missing_n_from_formal_929": int(config["expected"]["cer_formal_meters"]) - len(sub),
            "category_n": len(spec.category_order),
            "minimum_category_n": int(min(category_counts.values())),
            "minimum_group_by_category_n": min_cross,
            "rare_cell_flag": bool(min_cross < int(config["statistics"]["minimum_group_category_n_flag"])),
            "G1_n": int(group_counts.get("G1", 0)),
            "G2_n": int(group_counts.get("G2", 0)),
            "G3_n": int(group_counts.get("G3", 0)),
            "G4_n": int(group_counts.get("G4", 0)),
            "denominator_note": spec.denominator_note,
        })

        Y = sub[MAIN_OUTCOMES].to_numpy(dtype=float)
        fit = fit_in_sample(sub["profile_group"], sub[spec.variable], spec.category_order, Y)
        p_values, exceed = permutation_partial_r2(
            sub["profile_group"], sub[spec.variable], spec.category_order, Y,
            fit["partial_r2"], n_perm, perm_seed + spec_index * 100003,
        )
        cv = repeated_cv(
            sub["profile_group"], sub[spec.variable], spec.category_order, Y,
            cv_repeats, cv_folds, cv_seed + spec_index * 100003,
        )
        cv.insert(0, "variable", spec.variable)
        cv.insert(0, "analysis_role", spec.analysis_role)
        cv_frames.append(cv)

        # Group-only residuals are the group-adjusted descriptive scale.
        residuals = Y - fit["pred_base"]
        model_target = main_model_rows if spec.analysis_role == "main" else secondary_model_rows
        category_target = main_category_frames if spec.analysis_role == "main" else secondary_category_frames
        for j, outcome in enumerate(MAIN_OUTCOMES):
            cv_sub = cv[cv["outcome"] == outcome]
            delta = cv_sub["incremental_cv_r2"].astype(float)
            row = {
                "domain": spec.domain,
                "variable": spec.variable,
                "display_name": spec.display_name,
                "analysis_role": spec.analysis_role,
                "family": spec.family,
                "outcome": outcome,
                "n": len(sub),
                "category_n": len(spec.category_order),
                "reference_category": spec.category_order[0],
                "group_only_r2_in_sample": float(fit["r2_base"][j]),
                "group_plus_survey_r2_in_sample": float(fit["r2_aug"][j]),
                "group_adjusted_partial_r2": float(fit["partial_r2"][j]),
                "permutation_p": float(p_values[j]),
                "permutation_exceedances": int(exceed[j]),
                "permutations": n_perm,
                "mean_cv_r2_group_only": float(cv_sub["cv_r2_group_only"].mean()),
                "mean_cv_r2_group_plus_survey": float(cv_sub["cv_r2_group_plus_survey"].mean()),
                "mean_incremental_cv_r2": float(delta.mean()),
                "median_incremental_cv_r2": float(delta.median()),
                "incremental_cv_r2_p025": float(delta.quantile(0.025)),
                "incremental_cv_r2_p975": float(delta.quantile(0.975)),
                "positive_incremental_cv_repeat_share": float((delta > 0).mean()),
                "rare_cell_flag": bool(sample_rows[-1]["rare_cell_flag"]),
                "causal_interpretation_permitted": False,
            }
            model_target.append(row)
            category_target.append(category_summary(sub, spec, outcome, residuals[:, j]))

        # Fine-tuning audit is descriptive only: no p-value, q-value or CV selection.
        fine_values = sub[FINE_AUDIT_OUTCOME].to_numpy(dtype=float)
        group_means = sub.groupby("profile_group")[FINE_AUDIT_OUTCOME].transform("mean").to_numpy(dtype=float)
        fine_residual = fine_values - group_means
        fine_frames.append(category_summary(sub, spec, FINE_AUDIT_OUTCOME, fine_residual))

    main_model = pd.DataFrame(main_model_rows)
    secondary_model = pd.DataFrame(secondary_model_rows)
    for outcome in MAIN_OUTCOMES:
        mask = main_model["outcome"] == outcome
        main_model.loc[mask, "q_bh_within_outcome_main8"] = bh_adjust(main_model.loc[mask, "permutation_p"])
    main_model["multiplicity_evidence_q_lt_0_05"] = main_model["q_bh_within_outcome_main8"] < 0.05
    secondary_model["q_bh_within_outcome_main8"] = np.nan
    secondary_model["multiplicity_evidence_q_lt_0_05"] = np.nan

    main_category = pd.concat(main_category_frames, ignore_index=True)
    secondary_category = pd.concat(secondary_category_frames, ignore_index=True)
    fine_audit = pd.concat(fine_frames, ignore_index=True)
    cv_results = pd.concat(cv_frames, ignore_index=True)
    sample_audit = pd.DataFrame(sample_rows)

    variable_spec = pd.DataFrame([
        {
            "domain": s.domain,
            "variable": s.variable,
            "display_name": s.display_name,
            "analysis_role": s.analysis_role,
            "family": s.family,
            "category_order": "|".join(s.category_order),
            "category_labels": "|".join(s.category_labels),
            "denominator_note": s.denominator_note,
        }
        for s in specs
    ])
    input_join_audit = pd.DataFrame([
        {"check": "CER formal survey rows", "expected": 929, "actual": len(survey), "pass": len(survey) == 929},
        {"check": "Local source support outcome meters", "expected": 929, "actual": meter_outcomes["meter_id"].nunique(), "pass": meter_outcomes["meter_id"].nunique() == 929},
        {"check": "Joined meters", "expected": 929, "actual": len(joined), "pass": len(joined) == 929},
        {"check": "Unmatched survey meters", "expected": 0, "actual": len(set(survey["meter_id"]) - set(meter_outcomes["meter_id"])), "pass": set(survey["meter_id"]) == set(meter_outcomes["meter_id"])},
        {"check": "Main variables analysed regardless of Survey group composition significance", "expected": 8, "actual": int((variable_spec["analysis_role"] == "main").sum()), "pass": int((variable_spec["analysis_role"] == "main").sum()) == 8},
        {"check": "Supplementary thermal feature count excluded", "expected": False, "actual": "reported_thermal_feature_count" in set(variable_spec["variable"]), "pass": "reported_thermal_feature_count" not in set(variable_spec["variable"])},
        {"check": "Primary outcomes", "expected": "|".join(MAIN_OUTCOMES), "actual": "|".join(sorted(main_model["outcome"].unique())), "pass": set(main_model["outcome"]) == set(MAIN_OUTCOMES)},
    ])
    if not input_join_audit["pass"].all():
        raise SurveyGainAssociationsError("Input/join audit failed")

    results = {
        "meter_outcomes": meter_outcomes,
        "joined": joined,
        "variable_spec": variable_spec,
        "sample_audit": sample_audit,
        "main_model": main_model,
        "main_category": main_category,
        "secondary_model": secondary_model,
        "secondary_category": secondary_category,
        "fine_audit": fine_audit,
        "cv_results": cv_results,
        "input_join_audit": input_join_audit,
        "selected_local_source_support_status": t10_status,
        "selected_local_source_support_inventory": t10_inventory,
        "selected_local_source_support_outcome": t10_outcome_path,
    }

    if write_outputs:
        write_formal_outputs(config, root, results)
    return results


# -----------------------------------------------------------------------------
# Outputs and decision note
# -----------------------------------------------------------------------------

def write_decision_note(config: dict[str, Any], root: Path, results: dict[str, Any]) -> Path:
    output = root / config["output_roots"]["decision_note"]
    output.parent.mkdir(parents=True, exist_ok=True)
    main = results["main_model"].copy()
    show = main[[
        "display_name", "outcome", "n", "group_adjusted_partial_r2", "permutation_p",
        "q_bh_within_outcome_main8", "mean_incremental_cv_r2",
        "positive_incremental_cv_repeat_share", "rare_cell_flag",
    ]].copy()
    supported = main[main["multiplicity_evidence_q_lt_0_05"] == True]
    lines = [
        "# Survey gain associations — Group-adjusted survey–gain associations",
        "",
        f"**Release:** `{config['analysis_id']}`  ",
        "**Computational status:** `COMPLETE_PASS`  ",
        "**Analysis gate:** `PASS_TO_SURVEY_ADDITIONAL_ANALYSIS_SPECIFICATION`",
        "",
        "## Purpose",
        "",
        "Survey gain associations asks whether each pre-specified CER survey variable adds limited descriptive information about two pre-specified h=12 meter-level strategy-gain outcomes after fixed G1–G4 profile group is included.",
        "",
        "Primary outcomes:",
        "",
        "- Direct-over-Limited relative MAE Gain `(L−D)`.",
        "- Direct-over-Full relative RMSE Gain `(F−D)`.",
        "",
        "The eight main variables were analysed regardless of their Survey group composition p-value or q-value. No all-variable model, interaction search or outcome-driven variable selection was performed.",
        "",
        "## Method",
        "",
        "For each survey variable and outcome, the base model was `gain ~ fixed profile group` and the augmented model was `gain ~ fixed profile group + one survey variable`. The primary group-adjusted effect is partial R². Its p-value comes from deterministic survey-label permutations within fixed profile group. Benjamini–Hochberg adjustment is applied separately to the eight main tests for each pre-specified outcome. Repeated 20×10-fold cross-validation reports whether any in-sample association adds out-of-sample explanatory utility.",
        "",
        "Fine-Tuning-over-Direct relative MAE Gain `(D−T)` is included only as a supplementary category audit, without significance testing or subgroup selection.",
        "",
        "## Main model results",
        "",
        markdown_table(show, ".5f"),
        "",
        "## Multiplicity-adjusted evidence",
        "",
    ]
    if supported.empty:
        lines.append("No main survey variable had `q < .05` for either primary outcome after the pre-specified within-outcome BH adjustment.")
    else:
        lines.append("The following variable–outcome rows had `q < .05`; effect magnitude and repeated-CV utility must still be interpreted separately:")
        lines.append("")
        lines.append(markdown_table(supported[[
            "display_name", "outcome", "n", "group_adjusted_partial_r2",
            "q_bh_within_outcome_main8", "mean_incremental_cv_r2",
        ]], ".5f"))
    lines += [
        "",
        "## Claim boundary",
        "",
        "These results are post-hoc, descriptive and associational. They do not show that a household characteristic caused profile membership or strategy performance. A positive partial R² does not establish a deployment rule. A low permutation p-value can coexist with negligible or unstable repeated-CV utility. Daytime-presence results apply only to eligible non-live-alone respondents. Sparse-category and survey-linkage limits remain visible in the sample-audit table.",
        "",
        "Survey gain associations did not retrain a model, change predictions, modify forecasting inputs, refit clustering, reassign meters, exclude meters, use LCL survey variables to explain CER gains, or select variables from Survey group composition significance.",
        "",
        "## Next gate",
        "",
        "`PASS_TO_SURVEY_ADDITIONAL_ANALYSIS_SPECIFICATION`: Survey supplementary analysis may align these survey associations with already completed load-composition, temporal-representativeness and local-source-support mechanisms, while retaining the same non-causal boundary.",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def write_formal_outputs(config: dict[str, Any], root: Path, results: dict[str, Any]) -> None:
    table_root = root / config["output_roots"]["tables"]
    meta_root = root / config["output_roots"]["metadata"]
    table_root.mkdir(parents=True, exist_ok=True)
    meta_root.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []
    table_map = {
        "survey_gain_associations_meter_outcomes.csv": results["meter_outcomes"],
        "survey_gain_associations_joined_survey_gain_table.csv": results["joined"],
        "survey_gain_associations_variable_specification.csv": results["variable_spec"],
        "survey_gain_associations_analysis_sample_audit.csv": results["sample_audit"],
        "survey_gain_associations_main_model_summary.csv": results["main_model"],
        "survey_gain_associations_main_category_summary.csv": results["main_category"],
        "survey_gain_associations_secondary_model_summary.csv": results["secondary_model"],
        "survey_gain_associations_secondary_category_summary.csv": results["secondary_category"],
        "survey_gain_associations_fine_tuning_category_audit.csv": results["fine_audit"],
        "survey_gain_associations_cv_repeat_results.csv": results["cv_results"],
        "survey_gain_associations_input_join_audit.csv": results["input_join_audit"],
    }
    for name, df in table_map.items():
        path = table_root / name
        df.to_csv(path, index=False)
        files.append(path)

    selected_input = {
        "local_source_support_status": str(results["selected_local_source_support_status"]),
        "local_source_support_inventory": str(results["selected_local_source_support_inventory"]),
        "local_source_support_cross_strategy_outcome": str(results["selected_local_source_support_outcome"]),
        "local_source_support_cross_strategy_outcome_sha256": sha256_file(results["selected_local_source_support_outcome"]),
        "survey_recoding_cer_recoded": str(root / config["formal_inputs"]["survey_recoding_cer_recoded"]),
        "survey_group_composition_status": str(root / config["formal_inputs"]["survey_group_composition_status"]),
    }
    selected_path = meta_root / "survey_gain_associations_selected_input_record.json"
    write_json(selected_path, selected_input)
    files.append(selected_path)

    note_path = write_decision_note(config, root, results)
    files.append(note_path)

    status = {
        "analysis": config["analysis"],
        "analysis_id": config["analysis_id"],
        "created_at_utc": utc_now(),
        "computational_status": "COMPLETE_PASS",
        "analysis_gate_status": "PASS_TO_SURVEY_ADDITIONAL_ANALYSIS_SPECIFICATION",
        "cer_formal_n": int(config["expected"]["cer_formal_meters"]),
        "cer_group_counts": config["expected"]["cer_group_counts"],
        "main_variable_n": 8,
        "secondary_variable_n": 1,
        "primary_outcome_n": 2,
        "main_model_rows": len(results["main_model"]),
        "secondary_model_rows": len(results["secondary_model"]),
        "main_category_rows": len(results["main_category"]),
        "secondary_category_rows": len(results["secondary_category"]),
        "fine_tuning_audit_rows": len(results["fine_audit"]),
        "cv_repeat_rows": len(results["cv_results"]),
        "within_group_permutations": int(config["statistics"]["within_group_permutations"]),
        "cv_repeats": int(config["statistics"]["cv_repeats"]),
        "cv_folds": int(config["statistics"]["cv_folds"]),
        "bh_family_rule": "SEPARATE_EIGHT_MAIN_TESTS_WITHIN_EACH_PRIMARY_OUTCOME",
        "fine_tuning_audit_role": "SUPPLEMENTARY_DESCRIPTIVE_ONLY",
        "figures_generated": False,
        "training_performed": False,
        "predictions_modified": False,
        "forecasting_inputs_modified": False,
        "clustering_modified": False,
        "meters_reassigned": False,
        "meters_excluded": False,
        "lcl_survey_used_to_explain_cer_gain": False,
        "outcome_dependent_variable_selection": False,
        "all_variable_model_fitted": False,
        "interaction_search_performed": False,
        "causal_inference_performed": False,
        "ProfileInterpretation_outcomes_read": True,
    }
    status_path = meta_root / "survey_gain_associations_group_adjusted_survey_gain_associations_status.json"
    write_json(status_path, status)
    files.append(status_path)

    inventory_path = meta_root / "survey_gain_associations_output_inventory.csv"
    inventory_rows = []
    for path in sorted(files):
        inventory_rows.append({
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    pd.DataFrame(inventory_rows).to_csv(inventory_path, index=False)

    print(f"FORMAL CER METERS: {len(results['joined'])}")
    print("MAIN SURVEY VARIABLES: 8")
    print("SECONDARY SURVEY VARIABLES: 1")
    print("PRIMARY OUTCOMES: 2")
    print(f"MAIN MODEL ROWS: {len(results['main_model'])}")
    print(f"SECONDARY MODEL ROWS: {len(results['secondary_model'])}")
    print(f"WITHIN-GROUP PERMUTATIONS: {config['statistics']['within_group_permutations']}")
    print(f"REPEATED CV: {config['statistics']['cv_repeats']}x{config['statistics']['cv_folds']}-fold")
    print("FIGURES GENERATED: False")
    print("OUTCOME-DEPENDENT VARIABLE SELECTION: False")
    print("ALL-VARIABLE MODEL FITTED: False")
    print("CAUSAL INFERENCE PERFORMED: False")
    print(f"OUTPUT FILES WRITTEN: {len(files) + 1}")
    print("ANALYSIS GATE: PASS_TO_SURVEY_ADDITIONAL_ANALYSIS_SPECIFICATION")
    print("SURVEY GAIN ASSOCIATIONS COMPUTATION COMPLETE — PASS")


# -----------------------------------------------------------------------------
# Check-input and self-test modes
# -----------------------------------------------------------------------------

def check_inputs(config: dict[str, Any], root: Path) -> None:
    specs = parse_variable_specs(config)
    paths = validate_survey_recoding_and_11c(config, root)
    t10_status, t10_inventory, t10_outcome_path, outcomes = discover_local_source_support_inputs(config, root)
    meter_outcomes = build_meter_outcomes(outcomes)
    survey = load_and_validate_survey(paths["survey_recoding_cer_recoded"], config, specs)
    joined = join_survey_outcomes(survey, meter_outcomes)
    print("SURVEY GROUP COMPOSITION UPSTREAM GATE: PASS_TO_SURVEY_GAIN_ASSOCIATIONS_SPECIFICATION")
    print(f"CER FORMAL ROWS: {len(joined)}")
    print("LOCKED MAIN SURVEY VARIABLES: 8")
    print("LOCKED SECONDARY SURVEY VARIABLES: 1")
    print("PRIMARY OUTCOMES:")
    print("- h=12 Direct-over-Limited relative MAE Gain (L-D)")
    print("- h=12 Direct-over-Full relative RMSE Gain (F-D)")
    print(f"SELECTED LOCAL SOURCE SUPPORT STATUS: {t10_status}")
    print(f"SELECTED LOCAL SOURCE SUPPORT INVENTORY: {t10_inventory}")
    print(f"SELECTED LOCAL SOURCE SUPPORT OUTCOME TABLE: {t10_outcome_path}")
    print("BH FAMILY RULE: 8 MAIN VARIABLES SEPARATELY WITHIN EACH PRIMARY OUTCOME")
    print("DAYTIME PRESENCE SAMPLE: ELIGIBLE NON-LIVE-ALONE RESPONDENTS ONLY")
    print("REPORTED THERMAL FEATURE COUNT USED: False")
    print("SOCIAL CLASS ROLE: SECONDARY SENSITIVITY")
    print("FINE-TUNING GAIN ROLE: SUPPLEMENTARY CATEGORY AUDIT ONLY")
    print("LCL SURVEY USED TO EXPLAIN CER GAIN: False")
    print("OUTCOME-DEPENDENT VARIABLE SELECTION: False")
    print("ALL-VARIABLE MODEL FITTED: False")
    print("INTERACTION SEARCH PERFORMED: False")
    print("TRAINING PERFORMED: False")
    print("PREDICTIONS MODIFIED: False")
    print("SURVEY GAIN ASSOCIATIONS INPUT CHECK: PASS")


def selftest() -> None:
    # Inventory-schema compatibility audit: upstream conventions are
    # normalized to the same strict path/size/hash representation.
    digest = "0" * 64
    for fixture in (
        pd.DataFrame({"path": ["outputs/example.csv"], "size_bytes": [7], "sha256": [digest]}),
        pd.DataFrame({"relative_path": ["outputs/example.csv"], "bytes": [7], "sha256": [digest]}),
    ):
        normalized = normalize_inventory_table(fixture, "selftest")
        if list(normalized.columns) != ["path", "size_bytes", "sha256"]:
            raise SurveyGainAssociationsError("Selftest inventory normalization failed")
        if normalized.iloc[0].to_dict() != {
            "path": "outputs/example.csv", "size_bytes": 7, "sha256": digest
        }:
            raise SurveyGainAssociationsError("Selftest inventory values changed during normalization")

    rng = np.random.default_rng(112233)
    n = 160
    groups = pd.Series(np.repeat(GROUPS, n // 4))
    category = pd.Series(np.tile(np.array(["A", "B"]), n // 2))
    group_effect = groups.map({"G1": -1.0, "G2": -0.3, "G3": 0.4, "G4": 1.1}).to_numpy()
    survey_effect = np.where(category.to_numpy() == "B", 0.8, 0.0)
    y1 = group_effect + survey_effect + rng.normal(0, 0.4, n)
    y2 = -0.5 * group_effect + 0.4 * survey_effect + rng.normal(0, 0.5, n)
    Y = np.column_stack([y1, y2])
    fit = fit_in_sample(groups, category, ("A", "B"), Y)
    if not np.all(fit["partial_r2"] > 0.05):
        raise SurveyGainAssociationsError("Selftest partial R2 failed")
    p, _ = permutation_partial_r2(groups, category, ("A", "B"), Y, fit["partial_r2"], 300, 445566)
    if not np.all(p < 0.05):
        raise SurveyGainAssociationsError("Selftest permutation failed")
    cv = repeated_cv(groups, category, ("A", "B"), Y, 3, 5, 778899)
    if len(cv) != 6:
        raise SurveyGainAssociationsError("Selftest CV row count failed")
    # Gain sign audit.
    limited, direct, full, fine = 0.40, 0.36, 0.34, 0.38
    d_l = 100 * (limited - direct) / limited
    d_f = 100 * (full - direct) / full
    t_d = 100 * (direct - fine) / direct
    if not (d_l > 0 and d_f < 0 and t_d < 0):
        raise SurveyGainAssociationsError("Selftest gain sign failed")
    print("SURVEY GAIN ASSOCIATIONS SELFTEST: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--root", type=Path, default=Path(str(_ERP_PROJECT_ROOT)))
    parser.add_argument("--check-inputs", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return
    if args.config is None:
        raise SurveyGainAssociationsError("--config is required")
    config = read_json(args.config)
    root = args.root.expanduser().resolve()
    print("Python executable:", sys.executable)
    print("Python version:", platform.python_version())
    print("pandas version:", pd.__version__)
    print("numpy version:", np.__version__)
    print("scipy version:", scipy.__version__)
    if args.check_inputs:
        check_inputs(config, root)
    elif args.run:
        run_analysis(config, root, write_outputs=True)
    else:
        raise SurveyGainAssociationsError("Choose --check-inputs, --run or --selftest")


if __name__ == "__main__":
    main()
