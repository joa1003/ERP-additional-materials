#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats

GROUPS = ["G1", "G2", "G3", "G4"]
FORBIDDEN_OUTCOME_TOKENS = (
    "mae", "rmse", "smape", "gain", "strategy", "prediction", "forecast_error",
    "direct_transfer", "fine_tuning", "scratch_limited", "scratch_full"
)


class SurveyGroupCompositionError(RuntimeError):
    pass


@dataclass(frozen=True)
class AnalysisItem:
    dataset: str
    domain: str
    variable: str
    display_name: str
    variable_type: str
    analysis_role: str
    family: str
    category_order: tuple[str, ...] = ()
    category_labels: tuple[str, ...] = ()
    denominator_note: str = ""

    @property
    def tested(self) -> bool:
        return self.analysis_role != "supplementary_descriptive"


# -----------------------------
# General utilities
# -----------------------------

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


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


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
    text = str(value).replace("|", "\\|")
    return text.replace("\n", "<br>")


def markdown_table(df: pd.DataFrame, floatfmt: str = ".2f") -> str:
    if df.empty:
        return "_No rows._"
    headers = [markdown_escape(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        cells = []
        for value in row.tolist():
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


def inventory_record_path(recorded: str, root: Path) -> Path:
    raw = Path(str(recorded))
    if raw.exists():
        return raw.resolve()
    text = str(recorded).replace("\\", "/")
    for anchor in ("outputs/", "documentation/", "logs/"):
        pos = text.find(anchor)
        if pos >= 0:
            candidate = root / text[pos:]
            if candidate.exists():
                return candidate.resolve()
    return raw


def expected_group_counts(df: pd.DataFrame, expected: dict[str, int], dataset: str) -> None:
    if "profile_group" not in df.columns:
        raise SurveyGroupCompositionError(f"{dataset}: missing profile_group")
    counts = df["profile_group"].astype(str).value_counts().to_dict()
    if set(counts) != set(GROUPS):
        raise SurveyGroupCompositionError(f"{dataset}: group labels={sorted(counts)} expected={GROUPS}")
    for group in GROUPS:
        if int(counts.get(group, -1)) != int(expected[group]):
            raise SurveyGroupCompositionError(
                f"{dataset}: {group} count={counts.get(group)} expected={expected[group]}"
            )


# -----------------------------
# Upstream and input gates
# -----------------------------

def validate_upstream(config: dict[str, Any], root: Path) -> dict[str, Path]:
    paths = {key: root / rel for key, rel in config["formal_inputs"].items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise SurveyGroupCompositionError(f"Missing Survey recoding upstream inputs: {missing}")

    status = read_json(paths["survey_recoding_status"])
    required_status = {
        "computational_status": "COMPLETE_PASS",
        "analysis_gate_status": "PASS_TO_SURVEY_GROUP_COMPOSITION_SPECIFICATION",
        "lcl_formal_n": int(config["expected"]["lcl_formal_households"]),
        "cer_formal_n": int(config["expected"]["cer_formal_meters"]),
        "lcl_core_main_domain_n": 4,
        "lcl_core_main_variable_n": 4,
        "cer_core_main_domain_n": 6,
        "cer_core_main_variable_n": 8,
        "cer_secondary_sensitivity_variable_n": 1,
        "dwelling_scale_main": "bedroom_category",
        "reported_thermal_features_main": "double_glazing_category|attic_insulation_category|wall_insulation_category",
        "thermal_efficiency_main_status": "DEPRECATED_NOT_FOR_FORMAL_ANALYSIS",
        "reported_thermal_feature_count_role": "SUPPLEMENTARY_DESCRIPTIVE_ONLY",
        "economic_position_sensitivity": "social_class",
        "income_recode_status": "NOT_USED_UNRESOLVED_RELEASE_SEMANTICS",
        "survey_group_composition_relationship": "SURVEY_VARIABLE_DISTRIBUTION_BY_FIXED_PROFILE_GROUP",
        "strategy_performance_explained_in_survey_recoding_or_survey_group_composition": False,
    }
    for key, value in required_status.items():
        if status.get(key) != value:
            raise SurveyGroupCompositionError(
                f"Survey recoding status mismatch: {key}={status.get(key)!r}, expected={value!r}"
            )
    for key in (
        "training_performed", "predictions_modified", "forecasting_inputs_modified",
        "clustering_modified", "regression_performed", "significance_testing_performed",
        "outcome_dependent_variable_selection", "ProfileInterpretation_outcomes_read",
    ):
        if status.get(key) is not False:
            raise SurveyGroupCompositionError(f"Survey recoding governance flag must be False: {key}")

    note = paths["survey_recoding_decision_note"].read_text(encoding="utf-8")
    for phrase in (
        "PASS_TO_SURVEY_GROUP_COMPOSITION_SPECIFICATION",
        "Survey recoding and Survey group composition do not explain strategy performance",
        "The previous `thermal_efficiency_main` interpretation is deprecated",
        "No regression is performed",
        "No Profile interpretation outcome is read",
        "income semantics remain unresolved",
    ):
        if phrase not in note:
            raise SurveyGroupCompositionError(f"Survey recoding decision note missing phrase: {phrase}")

    inventory = read_csv(paths["survey_recoding_output_inventory"])
    required_cols = {"path", "size_bytes", "sha256"}
    if not required_cols.issubset(inventory.columns):
        raise SurveyGroupCompositionError("Survey recoding output inventory schema mismatch")
    indexed: dict[Path, pd.Series] = {}
    for _, row in inventory.iterrows():
        indexed[inventory_record_path(str(row["path"]), root)] = row
    for key in paths:
        if key == "survey_recoding_output_inventory":
            continue
        target = paths[key].resolve()
        if target not in indexed:
            raise SurveyGroupCompositionError(f"Survey recoding inventory missing required file: {target}")
        row = indexed[target]
        if target.stat().st_size != int(row["size_bytes"]):
            raise SurveyGroupCompositionError(f"Survey recoding upstream size mismatch: {target}")
        if sha256_file(target) != str(row["sha256"]):
            raise SurveyGroupCompositionError(f"Survey recoding upstream hash mismatch: {target}")
    return paths


def parse_items(config: dict[str, Any]) -> list[AnalysisItem]:
    items: list[AnalysisItem] = []
    for dataset, specs in config["analysis_spec"].items():
        for spec in specs:
            category_order = tuple(str(value) for value in spec.get("category_order", []))
            labels = tuple(str(value) for value in spec.get("category_labels", category_order))
            if category_order and len(labels) != len(category_order):
                raise SurveyGroupCompositionError(f"Category-label length mismatch: {dataset}/{spec['variable']}")
            items.append(
                AnalysisItem(
                    dataset=dataset,
                    domain=str(spec["domain"]),
                    variable=str(spec["variable"]),
                    display_name=str(spec["display_name"]),
                    variable_type=str(spec["variable_type"]),
                    analysis_role=str(spec["analysis_role"]),
                    family=str(spec["family"]),
                    category_order=category_order,
                    category_labels=labels,
                    denominator_note=str(spec.get("denominator_note", "")),
                )
            )
    return items


def validate_analysis_lock(lock_df: pd.DataFrame, items: list[AnalysisItem]) -> None:
    required_cols = {"dataset", "domain", "analysis_variable", "analysis_role"}
    if not required_cols.issubset(lock_df.columns):
        raise SurveyGroupCompositionError("Survey recoding analysis-variable lock schema mismatch")
    locked: set[tuple[str, str, str, str]] = set()
    for _, row in lock_df.iterrows():
        for variable in str(row["analysis_variable"]).split("|"):
            locked.add((str(row["dataset"]), str(row["domain"]), variable, str(row["analysis_role"])))
    for item in items:
        acceptable_roles = {item.analysis_role}
        if item.analysis_role == "main_domain_component":
            acceptable_roles.add("main")
        if not any(
            (item.dataset, item.domain, item.variable, role) in locked
            for role in acceptable_roles
        ):
            raise SurveyGroupCompositionError(
                f"Analysis item not supported by Survey recoding lock: "
                f"{item.dataset}/{item.domain}/{item.variable}/{item.analysis_role}"
            )


def validate_input_tables(
    config: dict[str, Any], paths: dict[str, Path], items: list[AnalysisItem]
) -> dict[str, pd.DataFrame]:
    lcl = read_csv(paths["lcl_recoded"])
    cer = read_csv(paths["cer_recoded"])
    lock_df = read_csv(paths["analysis_variable_lock"])
    fallback = read_csv(paths["fallback_audit"])

    expected = config["expected"]
    if len(lcl) != int(expected["lcl_formal_households"]):
        raise SurveyGroupCompositionError(f"LCL rows={len(lcl)} expected={expected['lcl_formal_households']}")
    if len(cer) != int(expected["cer_formal_meters"]):
        raise SurveyGroupCompositionError(f"CER rows={len(cer)} expected={expected['cer_formal_meters']}")

    for dataset, df in (("LCL", lcl), ("CER", cer)):
        if df["entity_id"].isna().any() or df["entity_id"].astype(str).duplicated().any():
            raise SurveyGroupCompositionError(f"{dataset}: entity_id is missing or duplicated")
        forbidden = [
            column for column in df.columns
            if any(token in column.lower() for token in FORBIDDEN_OUTCOME_TOKENS)
        ]
        if forbidden:
            raise SurveyGroupCompositionError(f"{dataset}: forbidden outcome-like columns present: {forbidden}")

    expected_group_counts(lcl, expected["lcl_group_counts"], "LCL")
    expected_group_counts(cer, expected["cer_group_counts"], "CER")

    response_expectations = {
        "LCL": {
            "usable_response": expected["lcl_usable_response"],
            "id_only_no_response": expected["lcl_id_only_no_response"],
            "no_survey_row": expected["lcl_no_survey_row"],
        },
        "CER": {
            "usable_response": expected["cer_usable_response"],
            "id_only_no_response": expected["cer_id_only_no_response"],
            "no_survey_row": expected["cer_no_survey_row"],
        },
    }
    for dataset, df in (("LCL", lcl), ("CER", cer)):
        counts = df["survey_response_status"].astype(str).value_counts().to_dict()
        for status_name, expected_n in response_expectations[dataset].items():
            if int(counts.get(status_name, 0)) != int(expected_n):
                raise SurveyGroupCompositionError(
                    f"{dataset}: {status_name}={counts.get(status_name, 0)} expected={expected_n}"
                )

    validate_analysis_lock(lock_df, items)

    selected = fallback.set_index("domain")["selected_variable"].astype(str).to_dict()
    required_fallbacks = {
        "dwelling_scale": "bedroom_category",
        "reported_thermal_features": "double_glazing_category|attic_insulation_category|wall_insulation_category",
        "economic_position": "social_class",
    }
    for domain, expected_value in required_fallbacks.items():
        if selected.get(domain) != expected_value:
            raise SurveyGroupCompositionError(
                f"Fallback mismatch: {domain}={selected.get(domain)} expected={expected_value}"
            )

    item_by_dataset: dict[str, list[AnalysisItem]] = {"LCL": [], "CER": []}
    for item in items:
        item_by_dataset[item.dataset].append(item)
    for dataset, df in (("LCL", lcl), ("CER", cer)):
        for item in item_by_dataset[dataset]:
            if item.variable not in df.columns:
                raise SurveyGroupCompositionError(f"{dataset}: missing locked variable {item.variable}")
            reason_col = f"{item.variable}__missing_reason"
            if reason_col not in df.columns:
                raise SurveyGroupCompositionError(f"{dataset}: missing missing-reason column {reason_col}")
            bad_available = df[item.variable].notna() & ~df[reason_col].astype(str).str.startswith("available")
            if bad_available.any():
                examples = df.loc[bad_available, ["entity_id", item.variable, reason_col]].head(5)
                raise SurveyGroupCompositionError(
                    f"{dataset}/{item.variable}: nonmissing values with nonavailable reasons:\n{examples}"
                )
            if item.variable_type == "categorical":
                observed = {normalize_category(value) for value in df[item.variable].dropna()}
                observed.discard(None)
                if observed != set(item.category_order):
                    raise SurveyGroupCompositionError(
                        f"{dataset}/{item.variable}: categories={sorted(observed)} "
                        f"expected={sorted(item.category_order)}"
                    )
            elif item.variable_type == "ordinal_numeric":
                numeric = pd.to_numeric(df[item.variable], errors="coerce")
                if int(numeric.notna().sum()) != int(df[item.variable].notna().sum()):
                    raise SurveyGroupCompositionError(f"{dataset}/{item.variable}: nonnumeric ordinal values")
            else:
                raise SurveyGroupCompositionError(f"Unsupported variable type: {item.variable_type}")

    # CER semantic gates.
    live_alone = cer["household_composition_common"].astype(str).eq("live_alone")
    if cer.loc[live_alone, "daytime_presence_category"].notna().any():
        raise SurveyGroupCompositionError("CER live-alone respondents must remain structurally missing for daytime presence")
    reasons = set(cer.loc[live_alone, "daytime_presence_category__missing_reason"].astype(str))
    if reasons != {"structurally_not_asked_live_alone"}:
        raise SurveyGroupCompositionError(f"CER live-alone daytime reasons unexpected: {sorted(reasons)}")

    refused_bedroom = cer["dwelling_scale_main__missing_reason"].astype(str).eq("refused")
    if int(refused_bedroom.sum()) != 2:
        raise SurveyGroupCompositionError(f"CER Q460 refused count={int(refused_bedroom.sum())} expected=2")
    refused_groups = cer.loc[refused_bedroom, "profile_group"].astype(str).value_counts().to_dict()
    if refused_groups != {"G3": 1, "G4": 1}:
        raise SurveyGroupCompositionError(f"CER Q460 refused groups={refused_groups} expected G3=1,G4=1")

    if "thermal_efficiency_main" in cer.columns:
        raise SurveyGroupCompositionError("Deprecated thermal_efficiency_main must not exist in the formal CER table")
    if any(item.variable == "thermal_efficiency_proxy_count" for item in items):
        raise SurveyGroupCompositionError("Deprecated thermal_efficiency_proxy_count cannot enter Survey group composition analysis")

    return {"LCL": lcl, "CER": cer, "lock": lock_df, "fallback": fallback}


# -----------------------------
# Statistical analysis
# -----------------------------

def effect_label(effect_name: str, value: float) -> str:
    if not np.isfinite(value):
        return "not_estimable"
    if effect_name == "cramers_v":
        if value < 0.10:
            return "negligible"
        if value < 0.30:
            return "small"
        if value < 0.50:
            return "moderate"
        return "large"
    if value < 0.01:
        return "negligible"
    if value < 0.06:
        return "small"
    if value < 0.14:
        return "moderate"
    return "large"


def chi_square_from_table(table: np.ndarray) -> tuple[float, int, float, np.ndarray]:
    if table.ndim != 2 or table.shape[0] < 2 or table.shape[1] < 2:
        raise SurveyGroupCompositionError(f"Invalid contingency-table shape: {table.shape}")
    chi2, p_value, dof, expected = stats.chi2_contingency(table, correction=False)
    return float(chi2), int(dof), float(p_value), np.asarray(expected, dtype=float)


def monte_carlo_chi_square_p(
    group_codes: np.ndarray,
    category_codes: np.ndarray,
    n_groups: int,
    n_categories: int,
    observed_chi2: float,
    permutations: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    group_codes = np.asarray(group_codes, dtype=np.int64)
    category_codes = np.asarray(category_codes, dtype=np.int64)
    observed = np.bincount(
        group_codes * n_categories + category_codes,
        minlength=n_groups * n_categories,
    ).reshape(n_groups, n_categories)
    row_totals = observed.sum(axis=1, keepdims=True)
    col_totals = observed.sum(axis=0, keepdims=True)
    expected = row_totals @ col_totals / observed.sum()
    if np.any(expected <= 0):
        raise SurveyGroupCompositionError("Permutation expected counts contain zero")
    exceed = 0
    for _ in range(permutations):
        permuted_groups = rng.permutation(group_codes)
        permuted = np.bincount(
            permuted_groups * n_categories + category_codes,
            minlength=n_groups * n_categories,
        ).reshape(n_groups, n_categories)
        statistic = float(np.sum((permuted - expected) ** 2 / expected))
        if statistic >= observed_chi2 - 1e-12:
            exceed += 1
    return float((exceed + 1) / (permutations + 1))


def categorical_analysis(
    df: pd.DataFrame,
    item: AnalysisItem,
    formal_counts: dict[str, int],
    permutations: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    work = df[["profile_group", "survey_response_status", item.variable]].copy()
    work[item.variable] = work[item.variable].map(normalize_category)
    available = work.dropna(subset=[item.variable]).copy()
    category_to_code = {category: index for index, category in enumerate(item.category_order)}
    available["_category_code"] = available[item.variable].map(category_to_code)
    if available["_category_code"].isna().any():
        raise SurveyGroupCompositionError(f"Unexpected category: {item.dataset}/{item.variable}")
    group_to_code = {group: index for index, group in enumerate(GROUPS)}
    available["_group_code"] = available["profile_group"].map(group_to_code)

    table = np.zeros((len(GROUPS), len(item.category_order)), dtype=int)
    for group_index, category_index in zip(available["_group_code"], available["_category_code"]):
        table[int(group_index), int(category_index)] += 1

    composition_rows: list[dict[str, Any]] = []
    availability_rows: list[dict[str, Any]] = []
    for group_index, group in enumerate(GROUPS):
        group_df = df[df["profile_group"].astype(str).eq(group)]
        usable_n = int(group_df["survey_response_status"].astype(str).eq("usable_response").sum())
        available_n = int(table[group_index].sum())
        availability_rows.append({
            "dataset": item.dataset,
            "domain": item.domain,
            "variable": item.variable,
            "display_name": item.display_name,
            "analysis_role": item.analysis_role,
            "profile_group": group,
            "formal_group_n": int(formal_counts[group]),
            "usable_response_n": usable_n,
            "available_n": available_n,
            "coverage_among_formal_pct": 100.0 * available_n / int(formal_counts[group]),
            "coverage_among_usable_pct": 100.0 * available_n / usable_n if usable_n else np.nan,
            "denominator_note": item.denominator_note,
        })
        for category_index, category in enumerate(item.category_order):
            count = int(table[group_index, category_index])
            composition_rows.append({
                "dataset": item.dataset,
                "domain": item.domain,
                "variable": item.variable,
                "display_name": item.display_name,
                "analysis_role": item.analysis_role,
                "profile_group": group,
                "category": category,
                "category_label": item.category_labels[category_index],
                "available_n": available_n,
                "category_n": count,
                "category_pct": 100.0 * count / available_n if available_n else np.nan,
                "denominator_note": item.denominator_note,
            })

    if not item.tested:
        return composition_rows, availability_rows, {}

    chi2, dof, p_asym, expected = chi_square_from_table(table)
    p_mc = monte_carlo_chi_square_p(
        available["_group_code"].to_numpy(dtype=int),
        available["_category_code"].to_numpy(dtype=int),
        len(GROUPS),
        len(item.category_order),
        chi2,
        permutations,
        seed,
    )
    n_complete = int(table.sum())
    denominator = n_complete * min(table.shape[0] - 1, table.shape[1] - 1)
    cramers_v = float(math.sqrt(chi2 / denominator)) if denominator > 0 else 0.0
    test_row = {
        "dataset": item.dataset,
        "domain": item.domain,
        "variable": item.variable,
        "display_name": item.display_name,
        "variable_type": item.variable_type,
        "analysis_role": item.analysis_role,
        "bh_family": item.family,
        "n_complete": n_complete,
        "group_min_n": int(table.sum(axis=1).min()),
        "category_n": int(table.shape[1]),
        "test_name": "pearson_chi_square_with_monte_carlo_permutation",
        "statistic": chi2,
        "degrees_of_freedom": dof,
        "p_asymptotic": p_asym,
        "p_monte_carlo": p_mc,
        "p_primary": p_mc,
        "effect_name": "cramers_v",
        "effect_value": cramers_v,
        "effect_magnitude": effect_label("cramers_v", cramers_v),
        "minimum_expected_count": float(expected.min()),
        "expected_cells_below_5": int((expected < 5).sum()),
        "observed_cells_below_10": int((table < 10).sum()),
        "rare_cell_flag": bool((table < 10).any() or (expected < 5).any()),
        "denominator_note": item.denominator_note,
        "claim_boundary": "descriptive_association_not_causation_or_profile_renaming",
    }
    return composition_rows, availability_rows, test_row


def ordinal_analysis(
    df: pd.DataFrame,
    item: AnalysisItem,
    formal_counts: dict[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    numeric_rows: list[dict[str, Any]] = []
    availability_rows: list[dict[str, Any]] = []
    groups_data: list[np.ndarray] = []
    for group in GROUPS:
        group_df = df[df["profile_group"].astype(str).eq(group)]
        values = pd.to_numeric(group_df[item.variable], errors="coerce").dropna().to_numpy(dtype=float)
        if len(values) == 0:
            raise SurveyGroupCompositionError(f"No values for {item.dataset}/{item.variable}/{group}")
        groups_data.append(values)
        usable_n = int(group_df["survey_response_status"].astype(str).eq("usable_response").sum())
        q1, median, q3 = np.quantile(values, [0.25, 0.50, 0.75])
        availability_rows.append({
            "dataset": item.dataset,
            "domain": item.domain,
            "variable": item.variable,
            "display_name": item.display_name,
            "analysis_role": item.analysis_role,
            "profile_group": group,
            "formal_group_n": int(formal_counts[group]),
            "usable_response_n": usable_n,
            "available_n": len(values),
            "coverage_among_formal_pct": 100.0 * len(values) / int(formal_counts[group]),
            "coverage_among_usable_pct": 100.0 * len(values) / usable_n if usable_n else np.nan,
            "denominator_note": item.denominator_note,
        })
        numeric_rows.append({
            "dataset": item.dataset,
            "domain": item.domain,
            "variable": item.variable,
            "display_name": item.display_name,
            "analysis_role": item.analysis_role,
            "profile_group": group,
            "available_n": len(values),
            "mean": float(np.mean(values)),
            "std_ddof1": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
            "q1": float(q1),
            "median": float(median),
            "q3": float(q3),
            "iqr": float(q3 - q1),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "denominator_note": item.denominator_note,
        })

    result = stats.kruskal(*groups_data, nan_policy="omit")
    h_stat, p_value = float(result.statistic), float(result.pvalue)
    n_total = int(sum(len(values) for values in groups_data))
    k = len(groups_data)
    epsilon_sq = max(0.0, float((h_stat - k + 1) / (n_total - k))) if n_total > k else 0.0
    test_row = {
        "dataset": item.dataset,
        "domain": item.domain,
        "variable": item.variable,
        "display_name": item.display_name,
        "variable_type": item.variable_type,
        "analysis_role": item.analysis_role,
        "bh_family": item.family,
        "n_complete": n_total,
        "group_min_n": int(min(len(values) for values in groups_data)),
        "category_n": np.nan,
        "test_name": "kruskal_wallis",
        "statistic": h_stat,
        "degrees_of_freedom": k - 1,
        "p_asymptotic": p_value,
        "p_monte_carlo": np.nan,
        "p_primary": p_value,
        "effect_name": "epsilon_squared",
        "effect_value": epsilon_sq,
        "effect_magnitude": effect_label("epsilon_squared", epsilon_sq),
        "minimum_expected_count": np.nan,
        "expected_cells_below_5": np.nan,
        "observed_cells_below_10": np.nan,
        "rare_cell_flag": False,
        "denominator_note": item.denominator_note,
        "claim_boundary": "descriptive_association_not_causation_or_profile_renaming",
    }
    return numeric_rows, availability_rows, test_row


def bh_adjust(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    if len(p) == 0:
        return p
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise SurveyGroupCompositionError("BH adjustment received invalid p-values")
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def availability_by_group(df: pd.DataFrame, dataset: str, formal_counts: dict[str, int]) -> pd.DataFrame:
    rows = []
    for group in GROUPS:
        part = df[df["profile_group"].astype(str).eq(group)]
        counts = part["survey_response_status"].astype(str).value_counts().to_dict()
        formal_n = int(formal_counts[group])
        usable = int(counts.get("usable_response", 0))
        id_only = int(counts.get("id_only_no_response", 0))
        no_row = int(counts.get("no_survey_row", 0))
        rows.append({
            "dataset": dataset,
            "profile_group": group,
            "formal_group_n": formal_n,
            "usable_response_n": usable,
            "id_only_no_response_n": id_only,
            "no_survey_row_n": no_row,
            "usable_response_rate_pct": 100.0 * usable / formal_n,
            "survey_row_rate_pct": 100.0 * (usable + id_only) / formal_n,
        })
    return pd.DataFrame(rows)


def missing_reason_by_group(data: dict[str, pd.DataFrame], items: list[AnalysisItem]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in items:
        df = data[item.dataset]
        reason_col = f"{item.variable}__missing_reason"
        for group in GROUPS:
            part = df[df["profile_group"].astype(str).eq(group)]
            counts = part[reason_col].fillna("missing_reason_not_recorded").astype(str).value_counts()
            for reason, count in counts.items():
                rows.append({
                    "dataset": item.dataset,
                    "domain": item.domain,
                    "variable": item.variable,
                    "display_name": item.display_name,
                    "analysis_role": item.analysis_role,
                    "profile_group": group,
                    "missing_reason": reason,
                    "n": int(count),
                })
    return pd.DataFrame(rows)


def derive_rooms_bands(
    lcl: pd.DataFrame, bands: list[dict[str, Any]]
) -> pd.DataFrame:
    values = pd.to_numeric(lcl["rooms_count"], errors="coerce")
    labels = pd.Series(pd.NA, index=lcl.index, dtype="object")
    for band in bands:
        minimum = float(band["minimum"])
        maximum = band["maximum"]
        mask = values.ge(minimum)
        if maximum is not None:
            mask &= values.le(float(maximum))
        labels.loc[mask] = str(band["label"])
    if labels[values.notna()].isna().any():
        raise SurveyGroupCompositionError("Rooms display bands failed to classify available values")
    rows = []
    order = [str(band["label"]) for band in bands]
    for group in GROUPS:
        part = labels[lcl["profile_group"].astype(str).eq(group)].dropna()
        available_n = len(part)
        counts = part.value_counts().to_dict()
        for label in order:
            count = int(counts.get(label, 0))
            rows.append({
                "dataset": "LCL",
                "variable": "rooms_count",
                "display_variable": "rooms_count_display_band",
                "profile_group": group,
                "band": label,
                "available_n": available_n,
                "band_n": count,
                "band_pct": 100.0 * count / available_n if available_n else np.nan,
                "inferential_use": "display_only_raw_rooms_used_for_kruskal_wallis",
            })
    return pd.DataFrame(rows)


# -----------------------------
# Figures
# -----------------------------

def slugify(text: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in text.lower()).strip("_")


def plot_stacked_distribution(
    composition: pd.DataFrame,
    item: AnalysisItem,
    output_png: Path,
    output_pdf: Path,
) -> None:
    part = composition[
        composition["dataset"].eq(item.dataset)
        & composition["variable"].eq(item.variable)
    ].copy()
    pivot = part.pivot(index="profile_group", columns="category", values="category_pct").reindex(GROUPS)
    counts = part.groupby("profile_group")["available_n"].first().reindex(GROUPS)
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    bottom = np.zeros(len(GROUPS))
    label_map = dict(zip(item.category_order, item.category_labels))
    for category in item.category_order:
        values = pivot[category].to_numpy(dtype=float)
        ax.bar(GROUPS, values, bottom=bottom, label=label_map[category])
        bottom += values
    ax.set_ylim(0, 108)
    ax.set_ylabel("Percentage of available responses")
    ax.set_xlabel("Fixed source-defined profile group")
    ax.set_title(f"{item.dataset}: {item.display_name}")
    for index, group in enumerate(GROUPS):
        ax.text(index, 102.0, f"n={int(counts.loc[group])}", ha="center", va="bottom", fontsize=9)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=min(3, len(item.category_order)))
    ax.text(
        0.0, -0.29,
        item.denominator_note + ". Percentages describe linked respondents and do not establish causality.",
        transform=ax.transAxes,
        fontsize=8.5,
        wrap=True,
    )
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_rooms_box(lcl: pd.DataFrame, output_png: Path, output_pdf: Path) -> None:
    data = [
        pd.to_numeric(
            lcl.loc[lcl["profile_group"].astype(str).eq(group), "rooms_count"],
            errors="coerce",
        ).dropna().to_numpy(dtype=float)
        for group in GROUPS
    ]
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.boxplot(data, tick_labels=GROUPS, showfliers=True)
    ax.set_xlabel("Fixed source-defined profile group")
    ax.set_ylabel("Reported number of rooms")
    ax.set_title("LCL: Number of rooms (raw available responses)")
    for index, values in enumerate(data, start=1):
        ax.text(index, ax.get_ylim()[1], f"n={len(values)}", ha="center", va="top", fontsize=9)
    ax.text(
        0.0, -0.18,
        "Raw values are retained, including released extremes. The inferential test is rank-based; this figure is descriptive.",
        transform=ax.transAxes,
        fontsize=8.5,
        wrap=True,
    )
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_rooms_bands(rooms_bands: pd.DataFrame, output_png: Path, output_pdf: Path) -> None:
    order = rooms_bands["band"].drop_duplicates().tolist()
    pivot = rooms_bands.pivot(index="profile_group", columns="band", values="band_pct").reindex(GROUPS)
    counts = rooms_bands.groupby("profile_group")["available_n"].first().reindex(GROUPS)
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    bottom = np.zeros(len(GROUPS))
    for band in order:
        values = pivot[band].to_numpy(dtype=float)
        ax.bar(GROUPS, values, bottom=bottom, label=band)
        bottom += values
    ax.set_ylim(0, 108)
    ax.set_ylabel("Percentage of available responses")
    ax.set_xlabel("Fixed source-defined profile group")
    ax.set_title("LCL: Number-of-rooms display bands")
    for index, group in enumerate(GROUPS):
        ax.text(index, 102.0, f"n={int(counts.loc[group])}", ha="center", va="bottom", fontsize=9)
    ax.legend(title="Rooms", loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    ax.text(
        0.0, -0.38,
        "Bands are for transparent visualisation only. Kruskal-Wallis uses the raw room counts.",
        transform=ax.transAxes,
        fontsize=8.5,
    )
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def make_effect_plot(test_df: pd.DataFrame, dataset: str, output_png: Path, output_pdf: Path) -> None:
    part = test_df[
        test_df["dataset"].eq(dataset)
        & test_df["bh_family"].isin(["LCL_MAIN", "CER_MAIN"])
    ].copy().sort_values("effect_value", ascending=True)
    fig_height = max(4.5, 0.65 * len(part) + 1.8)
    fig, ax = plt.subplots(figsize=(9.5, fig_height))
    positions = np.arange(len(part))
    ax.barh(positions, part["effect_value"].to_numpy(dtype=float))
    ax.set_yticks(positions)
    ax.set_yticklabels(part["display_name"].tolist())
    ax.set_xlabel("Effect size (Cramér's V or ε²)")
    ax.set_title(f"{dataset}: association magnitude across fixed profile groups")
    ax.set_xlim(left=0)
    for position, (_, row) in zip(positions, part.iterrows()):
        ax.text(
            float(row["effect_value"]) + 0.004,
            position,
            f"{row['effect_value']:.3f}; q={row['q_bh']:.3g}",
            va="center",
            fontsize=9,
        )
    ax.text(
        0.0, -0.16,
        "Effect sizes describe group association only. Survey variables do not rename profiles or establish causality.",
        transform=ax.transAxes,
        fontsize=9,
    )
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Main computation
# -----------------------------

def compute_analysis(config: dict[str, Any], root: Path) -> dict[str, Any]:
    paths = validate_upstream(config, root)
    items = parse_items(config)
    data = validate_input_tables(config, paths, items)
    expected = config["expected"]
    permutations = int(config["statistics"]["monte_carlo_permutations"])
    base_seed = int(config["statistics"]["random_seed"])

    survey_availability = pd.concat([
        availability_by_group(data["LCL"], "LCL", expected["lcl_group_counts"]),
        availability_by_group(data["CER"], "CER", expected["cer_group_counts"]),
    ], ignore_index=True)

    categorical_rows: list[dict[str, Any]] = []
    ordinal_rows: list[dict[str, Any]] = []
    availability_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []

    for index, item in enumerate(items):
        df = data[item.dataset]
        formal_counts = expected["lcl_group_counts"] if item.dataset == "LCL" else expected["cer_group_counts"]
        if item.variable_type == "categorical":
            composition, availability, test = categorical_analysis(
                df, item, formal_counts, permutations, base_seed + index * 1009
            )
            categorical_rows.extend(composition)
            availability_rows.extend(availability)
            if test:
                test_rows.append(test)
        else:
            summary, availability, test = ordinal_analysis(df, item, formal_counts)
            ordinal_rows.extend(summary)
            availability_rows.extend(availability)
            test_rows.append(test)

    categorical = pd.DataFrame(categorical_rows)
    ordinal = pd.DataFrame(ordinal_rows)
    variable_availability = pd.DataFrame(availability_rows)
    tests = pd.DataFrame(test_rows)

    tests["q_bh"] = np.nan
    for family in ("LCL_MAIN", "CER_MAIN"):
        mask = tests["bh_family"].eq(family)
        tests.loc[mask, "q_bh"] = bh_adjust(tests.loc[mask, "p_primary"].to_numpy(dtype=float))
    secondary_mask = tests["bh_family"].eq("CER_SECONDARY")
    tests.loc[secondary_mask, "q_bh"] = tests.loc[secondary_mask, "p_primary"]
    tests["bh_adjustment_scope"] = np.where(
        tests["bh_family"].eq("CER_SECONDARY"),
        "secondary_sensitivity_not_in_main_family",
        tests["bh_family"],
    )
    tests["q_below_0_05_flag"] = tests["q_bh"].astype(float) < 0.05

    lcl_main = tests[tests["bh_family"].eq("LCL_MAIN")].copy()
    cer_main = tests[tests["bh_family"].eq("CER_MAIN")].copy()
    cer_secondary = tests[tests["bh_family"].eq("CER_SECONDARY")].copy()
    supplementary_items = [item for item in items if item.analysis_role == "supplementary_descriptive"]

    if len(lcl_main) != int(expected["lcl_main_test_n"]):
        raise SurveyGroupCompositionError(f"LCL main tests={len(lcl_main)} expected={expected['lcl_main_test_n']}")
    if len(cer_main) != int(expected["cer_main_test_n"]):
        raise SurveyGroupCompositionError(f"CER main tests={len(cer_main)} expected={expected['cer_main_test_n']}")
    if len(cer_secondary) != int(expected["cer_secondary_test_n"]):
        raise SurveyGroupCompositionError(f"CER secondary tests={len(cer_secondary)} expected={expected['cer_secondary_test_n']}")
    if len(supplementary_items) != int(expected["cer_supplementary_descriptive_n"]):
        raise SurveyGroupCompositionError("CER supplementary descriptive item count mismatch")

    rooms_bands = derive_rooms_bands(data["LCL"], config["rooms_display_bands"])
    missing_reasons = missing_reason_by_group(data, items)
    registry = pd.DataFrame([
        {
            "dataset": item.dataset,
            "domain": item.domain,
            "variable": item.variable,
            "display_name": item.display_name,
            "variable_type": item.variable_type,
            "analysis_role": item.analysis_role,
            "bh_family": item.family,
            "inferential_test_performed": item.tested,
            "denominator_note": item.denominator_note,
        }
        for item in items
    ])

    audit = pd.DataFrame([
        {"check": "survey_recoding_upstream_gate", "observed": "PASS_TO_SURVEY_GROUP_COMPOSITION_SPECIFICATION", "expected": "PASS_TO_SURVEY_GROUP_COMPOSITION_SPECIFICATION", "status": "PASS"},
        {"check": "lcl_formal_rows", "observed": len(data["LCL"]), "expected": expected["lcl_formal_households"], "status": "PASS"},
        {"check": "cer_formal_rows", "observed": len(data["CER"]), "expected": expected["cer_formal_meters"], "status": "PASS"},
        {"check": "lcl_main_tests", "observed": len(lcl_main), "expected": expected["lcl_main_test_n"], "status": "PASS"},
        {"check": "cer_main_tests", "observed": len(cer_main), "expected": expected["cer_main_test_n"], "status": "PASS"},
        {"check": "cer_supplementary_descriptive_items", "observed": len(supplementary_items), "expected": expected["cer_supplementary_descriptive_n"], "status": "PASS"},
        {"check": "cer_secondary_tests", "observed": len(cer_secondary), "expected": expected["cer_secondary_test_n"], "status": "PASS"},
        {"check": "cer_q460_refused", "observed": int(data["CER"]["dwelling_scale_main__missing_reason"].astype(str).eq("refused").sum()), "expected": 2, "status": "PASS"},
        {"check": "thermal_efficiency_main_formal_analysis", "observed": False, "expected": False, "status": "PASS"},
        {"check": "reported_thermal_feature_count_in_main_bh", "observed": False, "expected": False, "status": "PASS"},
        {"check": "monte_carlo_permutations", "observed": permutations, "expected": permutations, "status": "PASS"},
        {"check": "ProfileInterpretation_outcomes_read", "observed": False, "expected": False, "status": "PASS"},
        {"check": "regression_performed", "observed": False, "expected": False, "status": "PASS"},
        {"check": "profile_groups_renamed_from_survey", "observed": False, "expected": False, "status": "PASS"},
    ])

    return {
        "survey_availability": survey_availability,
        "variable_availability": variable_availability,
        "categorical_composition": categorical,
        "ordinal_summary": ordinal,
        "rooms_bands": rooms_bands,
        "missing_reasons": missing_reasons,
        "registry": registry,
        "tests": tests,
        "lcl_main": lcl_main,
        "cer_main": cer_main,
        "cer_secondary": cer_secondary,
        "audit": audit,
        "items": items,
        "data": data,
    }


# -----------------------------
# Decision note and outputs
# -----------------------------

def distribution_summary_table(
    composition: pd.DataFrame, dataset: str, variable: str
) -> pd.DataFrame:
    part = composition[
        composition["dataset"].eq(dataset) & composition["variable"].eq(variable)
    ].copy()
    pivot = part.pivot(index="category_label", columns="profile_group", values="category_pct")
    pivot = pivot.reindex(columns=GROUPS)
    return pivot.reset_index().rename(columns={"category_label": "Category"})


def write_decision_note(path: Path, config: dict[str, Any], results: dict[str, Any]) -> None:
    tests = results["tests"]
    availability = results["survey_availability"]
    categorical = results["categorical_composition"]
    ordinal = results["ordinal_summary"]
    refused = results["missing_reasons"]
    refused = refused[
        refused["dataset"].eq("CER")
        & refused["variable"].eq("dwelling_scale_main")
        & refused["missing_reason"].eq("refused")
    ][["profile_group", "n"]]

    lines = [
        "# Survey group composition — Survey composition across fixed source-defined profile groups",
        "",
        f"Created at UTC: `{utc_now()}`",
        "",
        "## Formal status",
        "",
        "```text",
        "Computational status: COMPLETE_PASS",
        "Analysis gate: PASS_TO_SURVEY_GAIN_ASSOCIATIONS_SPECIFICATION",
        "Profile interpretation outcomes read: False",
        "Regression performed: False",
        "Significance testing performed: True",
        "Outcome-dependent variable selection: False",
        "Causal inference performed: False",
        "Profile groups renamed from survey: False",
        "```",
        "",
        "## Purpose and two-part structure",
        "",
        "Survey group composition has two separate layers. Part 1 reports the observed variable distributions across fixed G1–G4. Part 2 evaluates association magnitude, multiplicity-adjusted evidence and data-quality limits. The survey variables are not forecasting inputs and Survey group composition does not explain strategy performance.",
        "",
        "## Population and survey linkage",
        "",
        markdown_table(availability, floatfmt=".2f"),
        "",
        "LCL results apply to survey-linked source households and are limited by lower and uneven linkage, particularly in G4. CER results apply to linked receiving households. Variable-specific denominators are reported separately.",
        "",
        "# Part 1 — Descriptive distributions",
        "",
        "## LCL source-side composition",
        "",
        "The six LCL main variables describe household size, room count, three separate reported thermal features and electric thermal end use. Binary thermal indicators remain separate; no complete-case thermal composite is used in the main block.",
        "",
        "## CER receiving-side composition",
        "",
        "The six CER conceptual domains are operationalised through eight main variables because Q4906, Q4908 and Q4909 retain their original response meanings as separate categorical variables.",
        "",
        "### Household composition (%)",
        "",
        markdown_table(distribution_summary_table(categorical, "CER", "household_composition_common"), floatfmt=".1f"),
        "",
        "### Daytime household presence among eligible non-live-alone respondents (%)",
        "",
        markdown_table(distribution_summary_table(categorical, "CER", "daytime_presence_category"), floatfmt=".1f"),
        "",
        "### Bedroom category (%)",
        "",
        markdown_table(distribution_summary_table(categorical, "CER", "dwelling_scale_main"), floatfmt=".1f"),
        "",
        "Q460 code 6 (Refused) is not treated as a bedroom category. It is retained as a missing reason and excluded from percentages and association testing.",
        "",
        markdown_table(refused, floatfmt=".0f"),
        "",
        "### Reported thermal-feature semantics",
        "",
        "Q4906 double glazing, Q4908 attic insulation and Q4909 wall insulation are analysed separately using their original categories. Their numeric codes are answer labels, not a common energy-efficiency scale. `reported_thermal_feature_count` is supplementary descriptive information only and is not included in the CER main BH family.",
        "",
        "# Part 2 — Association, credibility and claim boundaries",
        "",
        f"- Categorical variables: Pearson chi-square statistic with `{config['statistics']['monte_carlo_permutations']}` deterministic label permutations as the primary p-value; Cramér's V as effect size.",
        "- LCL rooms: Kruskal–Wallis test on raw room counts; epsilon-squared as effect size.",
        "- Benjamini–Hochberg adjustment is performed separately within the six LCL main tests and eight CER main tests.",
        "- Social class is a secondary sensitivity and is not included in either main BH family.",
        "- Rare-cell flags, formal and usable denominators, variable-specific coverage, routing restrictions and missing reasons are reported.",
        "",
        "## LCL main inferential block",
        "",
        markdown_table(results["lcl_main"][[
            "variable", "n_complete", "effect_name", "effect_value", "effect_magnitude", "p_primary", "q_bh", "rare_cell_flag"
        ]].sort_values("effect_value", ascending=False), floatfmt=".4f"),
        "",
        "## CER main inferential block",
        "",
        markdown_table(results["cer_main"][[
            "variable", "n_complete", "effect_name", "effect_value", "effect_magnitude", "p_primary", "q_bh", "rare_cell_flag"
        ]].sort_values("effect_value", ascending=False), floatfmt=".4f"),
        "",
        "## Claim boundary",
        "",
        "Part 1 describes the current composition of linked respondents. Part 2 indicates whether the observed group differences are larger than expected under label exchangeability, how large the association is, and what data limitations constrain interpretation. Neither layer establishes that a household characteristic caused profile membership, caused forecasting error or caused transfer-learning benefit. G1–G4 must not be renamed as demographic or dwelling types.",
        "",
        "## Gate",
        "",
        "`PASS_TO_SURVEY_GAIN_ASSOCIATIONS_SPECIFICATION`",
        "",
        "Survey gain associations may later examine pre-specified group-adjusted associations between locked CER variables and selected pre-existing strategy gains. Variables must not be selected for Survey gain associations according to Survey group composition significance.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(config: dict[str, Any], root: Path, results: dict[str, Any]) -> list[Path]:
    """Write the retained Appendix M survey-composition tables and metadata only."""
    out = config["output_roots"]
    tables = root / out["tables"]
    metadata = root / out["metadata"]
    decision = root / out["decision_note"]
    for directory in (tables, metadata, decision.parent):
        directory.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    table_map = {
        "survey_group_composition_survey_availability_by_group.csv": results["survey_availability"],
        "survey_group_composition_variable_availability_by_group.csv": results["variable_availability"],
        "survey_group_composition_categorical_composition_by_group.csv": results["categorical_composition"],
        "survey_group_composition_ordinal_summary_by_group.csv": results["ordinal_summary"],
        "survey_group_composition_lcl_rooms_band_composition_by_group.csv": results["rooms_bands"],
        "survey_group_composition_variable_missing_reason_by_group.csv": results["missing_reasons"],
        "survey_group_composition_descriptive_distribution_registry.csv": results["registry"],
        "survey_group_composition_group_comparison_tests.csv": results["tests"],
        "survey_group_composition_lcl_main_effect_summary.csv": results["lcl_main"],
        "survey_group_composition_cer_main_effect_summary.csv": results["cer_main"],
        "survey_group_composition_cer_secondary_sensitivity.csv": results["cer_secondary"],
        "survey_group_composition_analysis_audit.csv": results["audit"],
    }
    for filename, dataframe in table_map.items():
        output = tables / filename
        dataframe.to_csv(output, index=False)
        paths.append(output)

    write_decision_note(decision, config, results)
    paths.append(decision)

    status = {
        "analysis": config["analysis"],
        "analysis_id": config["analysis_id"],
        "created_at_utc": utc_now(),
        "computational_status": "COMPLETE_PASS",
        "analysis_gate_status": "PASS_TO_SURVEY_GAIN_ASSOCIATIONS_SPECIFICATION",
        "analysis_structure": "DESCRIPTIVE_DISTRIBUTIONS_PLUS_ASSOCIATION_CREDIBILITY",
        "lcl_formal_n": int(config["expected"]["lcl_formal_households"]),
        "cer_formal_n": int(config["expected"]["cer_formal_meters"]),
        "lcl_main_test_n": int(len(results["lcl_main"])),
        "cer_main_test_n": int(len(results["cer_main"])),
        "cer_secondary_test_n": int(len(results["cer_secondary"])),
        "monte_carlo_permutations": int(config["statistics"]["monte_carlo_permutations"]),
        "training_performed": False,
        "predictions_modified": False,
        "forecasting_inputs_modified": False,
        "clustering_modified": False,
        "regression_performed": False,
        "significance_testing_performed": True,
        "outcome_dependent_variable_selection": False,
        "reported_thermal_feature_count_role": "SUPPLEMENTARY_DESCRIPTIVE_ONLY",
        "thermal_efficiency_main_status": "DEPRECATED_NOT_FOR_FORMAL_ANALYSIS",
        "ProfileInterpretation_outcomes_read": False,
        "causal_inference_performed": False,
        "profile_groups_renamed_from_survey": False,
    }
    status_path = metadata / "survey_group_composition_survey_group_composition_status.json"
    write_json(status_path, status)
    paths.append(status_path)

    inventory_rows = [
        {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in paths
    ]
    inventory_path = metadata / "survey_group_composition_output_inventory.csv"
    pd.DataFrame(inventory_rows).to_csv(inventory_path, index=False)
    paths.append(inventory_path)
    return paths

def print_environment(config: dict[str, Any]) -> None:
    print(f"PYTHON: {sys.executable}")
    print(f"PYTHON VERSION: {platform.python_version()}")
    print(f"PANDAS VERSION: {pd.__version__}")
    print(f"NUMPY VERSION: {np.__version__}")
    print(f"SCIPY VERSION: {scipy.__version__}")
    print(f"MATPLOTLIB VERSION: {matplotlib.__version__}")
    print(f"ANALYSIS ID: {config['analysis_id']}")


def run_selftest() -> None:
    toy = pd.DataFrame({
        "entity_id": [f"x{index}" for index in range(16)],
        "profile_group": np.repeat(GROUPS, 4),
        "survey_response_status": ["usable_response"] * 16,
        "cat": ["0", "0", "1", "1", "0", "1", "1", "1", "0", "0", "0", "1", "1", "1", "1", "1"],
        "ord": [1, 1, 2, 2, 1, 2, 2, 3, 2, 2, 3, 3, 3, 3, 4, 4],
    })
    cat_item = AnalysisItem(
        "CER", "toy", "cat", "Toy categorical", "categorical", "main", "CER_MAIN",
        ("0", "1"), ("No", "Yes"), "Toy denominator"
    )
    ord_item = AnalysisItem(
        "LCL", "toy", "ord", "Toy ordinal", "ordinal_numeric", "main", "LCL_MAIN",
        (), (), "Toy denominator"
    )
    composition, availability, test = categorical_analysis(
        toy, cat_item, {group: 4 for group in GROUPS}, 500, 123
    )
    if len(composition) != 8 or len(availability) != 4 or not (0 <= test["p_primary"] <= 1):
        raise SurveyGroupCompositionError("Categorical self-test failed")
    summary, availability2, test2 = ordinal_analysis(
        toy, ord_item, {group: 4 for group in GROUPS}
    )
    if len(summary) != 4 or len(availability2) != 4 or not (0 <= test2["p_primary"] <= 1):
        raise SurveyGroupCompositionError("Ordinal self-test failed")
    supplementary_item = AnalysisItem(
        "CER", "toy", "cat", "Toy supplementary", "categorical",
        "supplementary_descriptive", "CER_SUPPLEMENTARY",
        ("0", "1"), ("No", "Yes"), "Toy denominator"
    )
    _, _, supplementary_test = categorical_analysis(
        toy, supplementary_item, {group: 4 for group in GROUPS}, 500, 123
    )
    if supplementary_test != {}:
        raise SurveyGroupCompositionError("Supplementary item must not be tested")
    q_values = bh_adjust([0.01, 0.04, 0.20])
    if not np.allclose(q_values, [0.03, 0.06, 0.20]):
        raise SurveyGroupCompositionError(f"BH self-test failed: {q_values}")
    escaped = markdown_table(pd.DataFrame({"a": [1], "b": ["x|y"]}))
    if "x\\|y" not in escaped:
        raise SurveyGroupCompositionError("Markdown renderer self-test failed")
    print("SURVEY GROUP COMPOSITION SELFTEST: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--check-inputs", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        run_selftest()
        return
    if not args.config:
        parser.error("--config is required unless --selftest is used")
    config = read_json(args.config)
    root = args.root or Path(config["root"])
    print_environment(config)
    results = compute_analysis(config, root)

    print("SURVEY RECODING UPSTREAM GATE: PASS_TO_SURVEY_GROUP_COMPOSITION_SPECIFICATION")
    print(f"LCL FORMAL ROWS: {config['expected']['lcl_formal_households']}")
    print(f"CER FORMAL ROWS: {config['expected']['cer_formal_meters']}")
    print("ANALYSIS STRUCTURE: PART1_DESCRIPTIVE_DISTRIBUTIONS_PLUS_PART2_ASSOCIATION_CREDIBILITY")
    print(f"LCL MAIN DISTRIBUTIONS: 6")
    print(f"CER MAIN DISTRIBUTIONS: 8")
    print(f"CER SUPPLEMENTARY DISTRIBUTIONS: 1")
    print(f"CER SECONDARY DISTRIBUTIONS: 1")
    print(f"LCL MAIN COMPOSITION TESTS: {len(results['lcl_main'])}")
    print(f"CER MAIN COMPOSITION TESTS: {len(results['cer_main'])}")
    print(f"CER SECONDARY SENSITIVITY TESTS: {len(results['cer_secondary'])}")
    print("Q460 REFUSED RESPONSES: 2 (G3=1, G4=1)")
    print("THERMAL EFFICIENCY MAIN STATUS: DEPRECATED_NOT_FOR_FORMAL_ANALYSIS")
    print("REPORTED THERMAL FEATURE COUNT ROLE: SUPPLEMENTARY_DESCRIPTIVE_ONLY")
    print(f"MONTE CARLO PERMUTATIONS PER CATEGORICAL TEST: {config['statistics']['monte_carlo_permutations']}")
    print("PROFILE INTERPRETATION OUTCOMES READ: False")
    print("REGRESSION PERFORMED: False")
    print("OUTCOME-DEPENDENT VARIABLE SELECTION: False")
    print("PROFILE GROUPS RENAMED FROM SURVEY: False")

    if args.check_inputs:
        print("PROVISIONAL ANALYSIS GATE: PASS_TO_SURVEY_GAIN_ASSOCIATIONS_SPECIFICATION")
        print("SURVEY GROUP COMPOSITION INPUT CHECK: PASS")
        return
    if args.run:
        written = write_outputs(config, root, results)
        print(f"OUTPUT FILES WRITTEN: {len(written)}")
        print("ANALYSIS GATE: PASS_TO_SURVEY_GAIN_ASSOCIATIONS_SPECIFICATION")
        print("TRAINING PERFORMED: False")
        print("PREDICTIONS MODIFIED: False")
        print("FORECASTING INPUTS MODIFIED: False")
        print("CLUSTERING MODIFIED: False")
        print("REGRESSION PERFORMED: False")
        print("SIGNIFICANCE TESTING PERFORMED: True")
        print("PROFILE INTERPRETATION OUTCOMES READ: False")
        print("OUTCOME-DEPENDENT VARIABLE SELECTION: False")
        print("CAUSAL INFERENCE PERFORMED: False")
        print("PROFILE GROUPS RENAMED FROM SURVEY: False")
        print("SURVEY GROUP COMPOSITION COMPUTATION COMPLETE — PASS")
        return
    parser.error("Specify --check-inputs or --run")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"SURVEY GROUP COMPOSITION ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
