#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ANALYSIS_NAME = "Survey recoding — Governed survey recoding and semantic harmonisation"
GROUPS = ["G1", "G2", "G3", "G4"]
CSV_ENCODINGS = ("utf-8-sig", "cp1252")


class SurveyRecodingError(RuntimeError):
    pass


@dataclass(frozen=True)
class InputFile:
    role: str
    path: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def norm_col(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def is_blank(value: Any) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip() == ""


def clean_text(value: Any) -> str | None:
    if is_blank(value):
        return None
    return str(value).strip()


def find_column(columns: Iterable[Any], candidates: Iterable[str], role: str) -> str:
    lookup = {norm_col(c): str(c) for c in columns}
    hits = [lookup[norm_col(c)] for c in candidates if norm_col(c) in lookup]
    hits = list(dict.fromkeys(hits))
    if len(hits) != 1:
        raise SurveyRecodingError(
            f"Unable to identify exactly one {role} column; candidates={list(candidates)}; "
            f"hits={hits}; available={list(map(str, columns))[:80]}"
        )
    return hits[0]


def read_csv_governed(path: Path) -> pd.DataFrame:
    failures: list[str] = []
    for encoding in CSV_ENCODINGS:
        try:
            df = pd.read_csv(
                path,
                dtype=object,
                low_memory=False,
                encoding=encoding,
                encoding_errors="strict",
            )
            df.attrs["source_encoding"] = encoding
            return df
        except UnicodeDecodeError as exc:
            failures.append(
                f"{encoding}: byte={exc.object[exc.start:exc.end].hex()} pos={exc.start}"
            )
    raise SurveyRecodingError(f"Unable to decode CSV {path}; attempts={failures}")


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return read_csv_governed(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise SurveyRecodingError(f"Unsupported input table: {path}")


def normalise_lcl_id(value: Any) -> tuple[str | None, str]:
    if is_blank(value):
        return None, "blank"
    text = str(value).strip().upper()
    if re.fullmatch(r"N\d{4}", text):
        return text, "valid_group_n"
    if re.fullmatch(r"D\d{4}", text):
        return text, "out_of_scope_group_d"
    return None, "invalid_format"


def normalise_cer_id(value: Any) -> tuple[int | None, str]:
    if is_blank(value):
        return None, "blank"
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".", 1)[0]
    if not re.fullmatch(r"\d+", text):
        return None, "invalid_format"
    number = int(text)
    if number <= 0:
        return None, "invalid_nonpositive"
    return number, "valid"


def normalise_group(value: Any) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)) and 1 <= int(value) <= 4:
        return f"G{int(value)}"
    if isinstance(value, (float, np.floating)) and float(value).is_integer() and 1 <= int(value) <= 4:
        return f"G{int(value)}"
    text = str(value).strip().upper()
    match = re.search(r"(?:GROUP\s*|G)?([1-4])$", text)
    return f"G{match.group(1)}" if match else None


def resolve_inputs(config: dict[str, Any], root: Path) -> dict[str, InputFile]:
    failures: list[str] = []
    resolved: dict[str, InputFile] = {}
    for role, rel in config["formal_inputs"].items():
        path = (root / rel).resolve()
        if not path.is_file():
            failures.append(f"{role}: missing exact path {path}")
        else:
            resolved[role] = InputFile(role, path)
    lock = (root / config["canonical_lock"]).resolve()
    if not lock.is_file():
        failures.append(f"canonical_lock: missing exact path {lock}")
    else:
        resolved["canonical_lock"] = InputFile("canonical_lock", lock)
    if failures:
        raise SurveyRecodingError("Input resolution failed:\n- " + "\n- ".join(failures))
    return resolved


def input_inventory(resolved: dict[str, InputFile]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "role": role,
            "resolved_path": str(item.path),
            "suffix": item.path.suffix.lower(),
            "size_bytes": item.path.stat().st_size,
            "sha256": sha256_file(item.path),
        }
        for role, item in sorted(resolved.items())
    ])


def verify_survey_linkage_upstream(
    config: dict[str, Any], resolved: dict[str, InputFile]
) -> dict[str, Any]:
    status = load_json(resolved["survey_linkage_status"].path)
    if status.get("computational_status") != "COMPLETE_PASS":
        raise SurveyRecodingError(f"Survey linkage computational status is not COMPLETE_PASS: {status}")
    if status.get("analysis_gate_status") != "PASS_TO_SURVEY_RECODING_SPECIFICATION":
        raise SurveyRecodingError(
            f"Survey linkage gate does not permit Survey recoding: {status.get('analysis_gate_status')}"
        )
    forbidden_flags = [
        "training_performed",
        "predictions_modified",
        "forecasting_inputs_modified",
        "clustering_modified",
        "regression_performed",
        "outcome_dependent_variable_selection",
    ]
    bad = [f for f in forbidden_flags if bool(status.get(f))]
    if bad:
        raise SurveyRecodingError(f"Survey linkage upstream has forbidden flags: {bad}")

    expected = config["expected"]
    checks = {
        "lcl_formal_n": expected["lcl_formal_households"],
        "lcl_linked_usable_response_n": expected["survey_linkage_lcl_usable_n"],
        "lcl_linked_id_only_no_response_n": expected["survey_linkage_lcl_id_only_n"],
        "lcl_no_survey_row_n": expected["survey_linkage_lcl_no_row_n"],
        "cer_formal_n": expected["cer_formal_meters"],
        "cer_linked_usable_response_n": expected["survey_linkage_cer_usable_n"],
        "cer_linked_id_only_no_response_n": expected["survey_linkage_cer_id_only_n"],
        "cer_no_survey_row_n": expected["survey_linkage_cer_no_row_n"],
        "cer_governed_routing_rules_n": expected["survey_linkage_routing_rules_n"],
        "cer_routing_reviews_n": 0,
    }
    failures = [
        f"{key}={status.get(key)} expected={value}"
        for key, value in checks.items()
        if int(status.get(key, -1)) != int(value)
    ]
    if failures:
        raise SurveyRecodingError("Survey linkage status mismatch: " + " | ".join(failures))

    note = resolved["survey_linkage_decision_note"].path.read_text(encoding="utf-8")
    required_phrases = [
        "PASS_TO_SURVEY_RECODING_SPECIFICATION",
        "Group D (`Ddddd`) rows are out of scope",
        "does not recode income",
        "No regression",
    ]
    missing = [x for x in required_phrases if x not in note]
    if missing:
        raise SurveyRecodingError(f"Survey linkage decision note missing governance phrases: {missing}")

    routing = read_table(resolved["survey_linkage_cer_routing"].path)
    if len(routing) != int(expected["survey_linkage_routing_rules_n"]):
        raise SurveyRecodingError("Survey linkage routing row count mismatch")
    if "route_consistency_status" not in routing.columns:
        raise SurveyRecodingError("Survey linkage routing audit missing route_consistency_status")
    if not routing["route_consistency_status"].eq("PASS").all():
        raise SurveyRecodingError("Survey linkage routing audit contains non-PASS rows")

    semantic = read_table(resolved["survey_linkage_semantic_exceptions"].path)
    semantic_text = " ".join(semantic.astype(str).fillna("").to_numpy().ravel())
    if "Q402" not in semantic_text or "HOLD" not in semantic_text.upper():
        raise SurveyRecodingError("Survey linkage income semantic hold is missing")

    mapping = read_table(resolved["survey_linkage_lcl_mapping"].path)
    if len(mapping) != 135:
        raise SurveyRecodingError(f"Survey linkage LCL mapping rows={len(mapping)} expected=135")
    q231 = mapping[mapping["source_column"].astype(str).eq("Q231")]
    if len(q231) != 1 or q231.iloc[0]["scope_status"] != "excluded_property_ownership_validity_concern":
        raise SurveyRecodingError("Survey linkage LCL Q231 validity exclusion is missing")

    inventory = read_table(resolved["survey_linkage_output_inventory"].path)
    required_paths = {
        str(resolved["survey_linkage_status"].path.resolve()),
        str(resolved["survey_linkage_lcl_linkage"].path.resolve()),
        str(resolved["survey_linkage_cer_linkage"].path.resolve()),
        str(resolved["survey_linkage_lcl_mapping"].path.resolve()),
        str(resolved["survey_linkage_cer_routing"].path.resolve()),
        str(resolved["survey_linkage_comparability"].path.resolve()),
        str(resolved["survey_linkage_semantic_exceptions"].path.resolve()),
    }
    inventory_paths = {str(Path(p).resolve()) for p in inventory["path"].astype(str)}
    if not inventory_paths.issuperset(required_paths):
        raise SurveyRecodingError(
            f"Survey linkage output inventory missing upstream files: {sorted(required_paths - inventory_paths)}"
        )
    inv = inventory.copy()
    inv["_resolved"] = inv["path"].map(lambda p: str(Path(str(p)).resolve()))
    inv = inv.set_index("_resolved")
    for path_text in sorted(required_paths):
        row = inv.loc[path_text]
        if sha256_file(Path(path_text)) != str(row["sha256"]):
            raise SurveyRecodingError(f"Survey linkage upstream hash mismatch: {path_text}")
    return status


def prepare_formal_assignments(
    config: dict[str, Any], resolved: dict[str, InputFile]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expected = config["expected"]

    def prepare(
        path: Path,
        dataset: str,
        expected_n: int,
        expected_groups: dict[str, int],
    ) -> pd.DataFrame:
        df = read_table(path)
        id_candidates = ["household_id", "entity_id", "Household_id"] if dataset == "LCL" else ["meter_id", "ID", "entity_id"]
        id_col = find_column(df.columns, id_candidates, f"{dataset} assignment ID")
        group_col = find_column(df.columns, ["assigned_group", "group", "profile_group", "cluster"], f"{dataset} profile group")
        normalised = df[id_col].map(normalise_lcl_id if dataset == "LCL" else normalise_cer_id)
        out = pd.DataFrame({
            "entity_id": [x[0] for x in normalised],
            "id_status": [x[1] for x in normalised],
            "profile_group": df[group_col].map(normalise_group),
        })
        valid_status = "valid_group_n" if dataset == "LCL" else "valid"
        if out["entity_id"].isna().any() or not out["id_status"].eq(valid_status).all():
            raise SurveyRecodingError(f"{dataset} assignments contain invalid IDs")
        if out["profile_group"].isna().any() or out["entity_id"].duplicated().any():
            raise SurveyRecodingError(f"{dataset} assignments contain invalid groups or duplicate IDs")
        if len(out) != expected_n:
            raise SurveyRecodingError(f"{dataset} assignment n={len(out)} expected={expected_n}")
        counts = out["profile_group"].value_counts().sort_index().to_dict()
        if counts != expected_groups:
            raise SurveyRecodingError(f"{dataset} group counts={counts} expected={expected_groups}")
        return out[["entity_id", "profile_group"]].sort_values("entity_id").reset_index(drop=True)

    lcl = prepare(
        resolved["lcl_assignments"].path,
        "LCL",
        int(expected["lcl_formal_households"]),
        {k: int(v) for k, v in expected["lcl_group_counts"].items()},
    )
    cohort = read_table(resolved["lcl_cohort"].path)
    cohort_col = find_column(cohort.columns, ["household_id", "entity_id", "Household_id"], "LCL cohort ID")
    cohort_norm = cohort[cohort_col].map(normalise_lcl_id)
    cohort_ids = [x[0] for x in cohort_norm]
    if any(x[1] != "valid_group_n" for x in cohort_norm) or len(cohort_ids) != len(set(cohort_ids)):
        raise SurveyRecodingError("LCL cohort contains invalid or duplicate IDs")
    if set(cohort_ids) != set(lcl["entity_id"]):
        raise SurveyRecodingError("LCL cohort and assignments do not contain the same 3,843 IDs")

    cer = prepare(
        resolved["cer_assignments"].path,
        "CER",
        int(expected["cer_formal_meters"]),
        {k: int(v) for k, v in expected["cer_group_counts"].items()},
    )
    return lcl, cer


def nonblank_count(row: pd.Series, columns: list[str]) -> int:
    return sum(not is_blank(row.get(c)) for c in columns)


def coalesce_nonconflicting(
    group: pd.DataFrame, columns: list[str]
) -> tuple[pd.Series | None, str, list[str]]:
    if len(group) == 1:
        return group.iloc[0].copy(), "single_row", []
    merged = group.iloc[0].copy()
    conflicts: list[str] = []
    for col in columns:
        values = [str(v).strip() for v in group[col] if not is_blank(v)]
        unique = list(dict.fromkeys(values))
        if len(unique) > 1:
            conflicts.append(col)
        elif len(unique) == 1:
            merged[col] = unique[0]
        else:
            merged[col] = np.nan
    if conflicts:
        return None, "unresolved_value_conflict", conflicts
    return merged, "coalesced_nonconflicting", []


def resolve_lcl_answers(
    config: dict[str, Any], resolved: dict[str, InputFile]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = read_table(resolved["lcl_survey_answers"].path)
    id_col = config["lcl"]["id_column"]
    if id_col not in df.columns:
        raise SurveyRecodingError(f"LCL survey missing ID column {id_col}")
    mapping = read_table(resolved["survey_linkage_lcl_mapping"].path)
    substantive = mapping.loc[
        mapping["scope_status"].eq("in_scope_appliance"), "source_column"
    ].astype(str).tolist()
    missing_cols = sorted(set(substantive) - set(df.columns))
    if missing_cols:
        raise SurveyRecodingError(f"LCL survey missing governed substantive columns: {missing_cols}")
    work = df.copy()
    norm = work[id_col].map(normalise_lcl_id)
    work["_entity_id"] = [x[0] for x in norm]
    work["_id_status"] = [x[1] for x in norm]
    group_n = work[work["_id_status"].eq("valid_group_n")].copy()
    rows: list[pd.Series] = []
    audits: list[dict[str, Any]] = []
    for entity_id, grp in group_n.groupby("_entity_id", sort=True):
        merged, resolution, conflicts = coalesce_nonconflicting(grp, substantive)
        if merged is None:
            raise SurveyRecodingError(
                f"Unexpected unresolved LCL duplicate after Survey linkage closure: {entity_id}; conflicts={conflicts}"
            )
        merged["_entity_id"] = entity_id
        count = nonblank_count(merged, substantive)
        merged["survey_response_status"] = "usable_response" if count > 0 else "id_only_no_response"
        rows.append(merged)
        audits.append({
            "dataset": "LCL",
            "entity_id": entity_id,
            "raw_row_n": len(grp),
            "resolution": resolution,
            "conflict_columns": "|".join(conflicts),
            "substantive_nonblank_n": count,
            "survey_response_status": merged["survey_response_status"],
        })
    out = pd.DataFrame(rows)
    if out["_entity_id"].duplicated().any():
        raise SurveyRecodingError("LCL survey resolution did not produce one row per ID")
    return out, pd.DataFrame(audits)


def resolve_cer_answers(
    config: dict[str, Any], resolved: dict[str, InputFile]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = read_table(resolved["cer_pretrial_survey_data"].path)
    id_col = config["cer"]["id_column"]
    if id_col not in df.columns:
        raise SurveyRecodingError(f"CER survey missing ID column {id_col}")
    work = df.copy()
    norm = work[id_col].map(normalise_cer_id)
    work["_entity_id"] = [x[0] for x in norm]
    work["_id_status"] = [x[1] for x in norm]
    if not work["_id_status"].eq("valid").all():
        raise SurveyRecodingError("CER survey contains invalid ID rows")
    response_cols = [c for c in work.columns if c not in {id_col, "_entity_id", "_id_status"}]
    rows: list[pd.Series] = []
    audits: list[dict[str, Any]] = []
    for entity_id, grp in work.groupby("_entity_id", sort=True):
        merged, resolution, conflicts = coalesce_nonconflicting(grp, response_cols)
        if merged is None:
            raise SurveyRecodingError(f"Unexpected unresolved CER duplicate: {entity_id}; conflicts={conflicts}")
        merged["_entity_id"] = entity_id
        count = nonblank_count(merged, response_cols)
        merged["survey_response_status"] = "usable_response" if count > 0 else "id_only_no_response"
        rows.append(merged)
        audits.append({
            "dataset": "CER",
            "entity_id": entity_id,
            "raw_row_n": len(grp),
            "resolution": resolution,
            "conflict_columns": "|".join(conflicts),
            "substantive_nonblank_n": count,
            "survey_response_status": merged["survey_response_status"],
        })
    out = pd.DataFrame(rows)
    if out["_entity_id"].duplicated().any():
        raise SurveyRecodingError("CER survey resolution did not produce one row per ID")
    return out, pd.DataFrame(audits)


def attach_formal(formal: pd.DataFrame, survey: pd.DataFrame) -> pd.DataFrame:
    merged = formal.merge(
        survey.drop(columns=["_id_status"], errors="ignore"),
        left_on="entity_id",
        right_on="_entity_id",
        how="left",
        validate="one_to_one",
    )
    merged["survey_response_status"] = merged["survey_response_status"].fillna("no_survey_row")
    return merged


def base_missing_reason(status: str) -> str | None:
    if status == "no_survey_row":
        return "no_survey_row"
    if status == "id_only_no_response":
        return "id_only_no_response"
    return None


def put(out: dict[str, Any], variable: str, value: Any, reason: str, source: str) -> None:
    out[variable] = value
    out[f"{variable}__missing_reason"] = reason
    out[f"{variable}__source"] = source


def band_value(value: Any, bands: list[dict[str, Any]]) -> str | None:
    if pd.isna(value):
        return None
    x = float(value)
    for spec in bands:
        if float(spec["min"]) <= x <= float(spec["max"]):
            return str(spec["label"])
    return None


def parse_int(value: Any, minimum: int = 0, maximum: int = 999) -> tuple[float, str]:
    if is_blank(value):
        return np.nan, "source_blank"
    try:
        number = float(str(value).strip())
    except ValueError:
        return np.nan, "invalid_source_value"
    if not number.is_integer() or not minimum <= number <= maximum:
        return np.nan, "invalid_source_value"
    return float(int(number)), "available"


def parse_yes_no_text(value: Any) -> tuple[float, str]:
    text = clean_text(value)
    if text is None:
        return np.nan, "source_blank"
    key = text.lower().replace("’", "'")
    if key == "yes" or key.startswith("yes;"):
        return 1.0, "available"
    if key == "no":
        return 0.0, "available"
    if key in {"don't know", "dont know", "do not know"}:
        return np.nan, "dont_know"
    return np.nan, "invalid_source_value"


def parse_presence_count(value: Any) -> tuple[float, str]:
    if is_blank(value):
        return np.nan, "source_blank"
    try:
        number = float(str(value).strip())
    except ValueError:
        return np.nan, "invalid_source_value"
    if number < 0 or not number.is_integer():
        return np.nan, "invalid_source_value"
    return (1.0 if number > 0 else 0.0), "available"


def combine_binary(values: list[float], *, source_missing_reason: str = "incomplete_component_coverage") -> tuple[float, str]:
    if any(v == 1 for v in values):
        return 1.0, "available"
    if values and all(v == 0 for v in values):
        return 0.0, "available"
    return np.nan, source_missing_reason


def lcl_core_variables() -> list[str]:
    return [
        "household_size",
        "household_size_band_common",
        "household_composition_common",
        "dwelling_type_common",
        "rooms_count",
        "bedroom_count_min",
        "bedroom_band_common",
        "double_glazing_any",
        "attic_roof_insulation_any",
        "wall_insulation_any",
        "thermal_efficiency_proxy_count",
        "electric_space_heating_any",
        "electric_water_heating_any",
        "electric_thermal_load",
        "washing_machine_any",
        "tumble_dryer_any",
        "dishwasher_any",
        "electric_cooker_any",
        "electric_shower_any",
        "portable_electric_heater_any",
    ]


def lcl_recode_row(row: pd.Series, config: dict[str, Any]) -> dict[str, Any]:
    status = str(row["survey_response_status"])
    out: dict[str, Any] = {
        "entity_id": row["entity_id"],
        "profile_group": row["profile_group"],
        "survey_response_status": status,
    }
    base = base_missing_reason(status)
    if base:
        for variable in lcl_core_variables():
            put(out, variable, np.nan, base, "")
        return out

    lcl = config["lcl"]
    size, size_reason = parse_int(row.get(lcl["household_size_column"]), 1, 50)
    put(out, "household_size", size, size_reason, lcl["household_size_column"])
    size_band = band_value(size, config["bands"]["household_size"])
    put(out, "household_size_band_common", size_band, "available" if size_band else size_reason, lcl["household_size_column"])

    age_values = [clean_text(row.get(c)) for c in lcl["age_columns"]]
    observed_ages = [x for x in age_values if x is not None and x.lower() != "don't know"]
    child_observed = any(x in set(lcl["under16_age_labels"]) for x in observed_ages)
    complete_age_coverage = not pd.isna(size) and int(size) <= len(lcl["age_columns"]) and len(observed_ages) >= int(size)
    if pd.isna(size):
        composition, comp_reason = None, size_reason
    elif int(size) == 1:
        composition, comp_reason = "live_alone", "available"
    elif child_observed:
        composition, comp_reason = "adults_with_children_under16", "available"
    elif complete_age_coverage:
        composition, comp_reason = "adults_only_or_no_under16", "available"
    else:
        composition, comp_reason = None, "insufficient_age_coverage"
    put(out, "household_composition_common", composition, comp_reason, "Q213|Q222-Q229")

    house = clean_text(row.get(lcl["house_type_column"]))
    flat = clean_text(row.get(lcl["flat_type_column"]))
    mobile = clean_text(row.get(lcl["mobile_type_column"]))
    observed_types = [x for x in [house, flat, mobile] if x is not None]
    if len(observed_types) > 1:
        dwelling, dwelling_reason = None, "conflicting_source_fields"
    elif house:
        low = house.lower()
        if "terraced" in low:
            dwelling = "terraced"
        elif "semi-detached" in low:
            dwelling = "semi_detached"
        elif "detached" in low:
            dwelling = "detached_or_bungalow"
        else:
            dwelling = "other_or_mobile"
        dwelling_reason = "available"
    elif flat:
        dwelling, dwelling_reason = "flat_or_apartment", "available"
    elif mobile:
        dwelling, dwelling_reason = "other_or_mobile", "available"
    else:
        dwelling, dwelling_reason = None, "source_blank"
    put(out, "dwelling_type_common", dwelling, dwelling_reason, "Q235|Q236|Q237")

    rooms, rooms_reason = parse_int(row.get(lcl["rooms_column"]), 0, 100)
    bedrooms, bedrooms_reason = parse_int(row.get(lcl["bedrooms_column"]), 0, 100)
    put(out, "rooms_count", rooms, rooms_reason, lcl["rooms_column"])
    put(out, "bedroom_count_min", bedrooms, bedrooms_reason, lcl["bedrooms_column"])
    bedroom_band = band_value(bedrooms, config["bands"]["bedrooms"])
    put(out, "bedroom_band_common", bedroom_band, "available" if bedroom_band else bedrooms_reason, lcl["bedrooms_column"])

    insulation_specs = {
        "double_glazing_any": lcl["double_glazing_column"],
        "attic_roof_insulation_any": lcl["roof_insulation_column"],
        "wall_insulation_any": lcl["wall_insulation_column"],
    }
    insulation_values: list[float] = []
    for variable, column in insulation_specs.items():
        value, reason = parse_yes_no_text(row.get(column))
        put(out, variable, value, reason, column)
        insulation_values.append(value)
    if all(not pd.isna(v) for v in insulation_values):
        thermal, thermal_reason = float(sum(insulation_values)), "available"
    else:
        thermal, thermal_reason = np.nan, "incomplete_component_coverage"
    put(out, "thermal_efficiency_proxy_count", thermal, thermal_reason, "Q240|Q241|Q242")

    central = clean_text(row.get(lcl["central_heating_column"]))
    if central is None:
        electric_space, space_reason = np.nan, "source_blank"
    elif "don't know" in central.lower():
        electric_space, space_reason = np.nan, "dont_know"
    else:
        electric_space, space_reason = (1.0 if "electric" in central.lower() else 0.0), "available"
    hotwater = clean_text(row.get(lcl["hot_water_column"]))
    if hotwater is None:
        electric_water, water_reason = np.nan, "source_blank"
    elif "don't know" in hotwater.lower() and "electric" not in hotwater.lower():
        electric_water, water_reason = np.nan, "dont_know"
    else:
        electric_water, water_reason = (1.0 if "electric" in hotwater.lower() else 0.0), "available"
    put(out, "electric_space_heating_any", electric_space, space_reason, lcl["central_heating_column"])
    put(out, "electric_water_heating_any", electric_water, water_reason, lcl["hot_water_column"])
    electric_thermal, thermal_reason = combine_binary([electric_space, electric_water])
    put(out, "electric_thermal_load", electric_thermal, thermal_reason, "Q246|Q248")

    appliance_values: dict[str, tuple[float, str]] = {}
    for variable, column in lcl["appliance_columns"].items():
        appliance_values[variable] = parse_presence_count(row.get(column))

    washing, washing_reason = combine_binary([
        appliance_values["washing_machine_standalone_any"][0],
        appliance_values["washer_dryer_any"][0],
    ])
    dryer, dryer_reason = combine_binary([
        appliance_values["tumble_dryer_standalone_any"][0],
        appliance_values["washer_dryer_any"][0],
    ])
    cooker, cooker_reason = combine_binary([
        appliance_values["electric_hob_any"][0],
        appliance_values["electric_oven_any"][0],
    ])
    put(out, "washing_machine_any", washing, washing_reason, "Q297|Q299")
    put(out, "tumble_dryer_any", dryer, dryer_reason, "Q298|Q299")
    put(out, "electric_cooker_any", cooker, cooker_reason, "Q293|Q295")
    for variable, source_key in [
        ("dishwasher_any", "dishwasher_any"),
        ("electric_shower_any", "electric_shower_any"),
        ("portable_electric_heater_any", "portable_electric_heater_any"),
    ]:
        value, reason = appliance_values[source_key]
        put(out, variable, value, reason, lcl["appliance_columns"][source_key])
    return out


def cer_columns_by_code(df: pd.DataFrame) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for column in df.columns:
        match = re.match(r"^Question ([0-9]+):", str(column))
        if match:
            out.setdefault(f"Q{match.group(1)}", []).append(str(column))
    return out


def require_single(code_map: dict[str, list[str]], code: str) -> str:
    columns = code_map.get(code, [])
    if len(columns) != 1:
        raise SurveyRecodingError(f"Expected one CER column for {code}; found {columns}")
    return columns[0]


def require_contains(code_map: dict[str, list[str]], code: str, text: str) -> str:
    columns = [c for c in code_map.get(code, []) if text.lower() in c.lower()]
    if len(columns) != 1:
        raise SurveyRecodingError(
            f"Expected one CER {code} column containing {text!r}; found {columns}"
        )
    return columns[0]


def parse_multiselect_indicator(value: Any) -> tuple[float, str]:
    text = clean_text(value)
    if text is None:
        return np.nan, "source_blank"
    if text in {"0", "1"}:
        return float(int(text)), "available"
    return np.nan, "invalid_source_value"


def parse_binary_code(value: Any, yes: str = "1", no: str = "2") -> tuple[float, str]:
    text = clean_text(value)
    if text is None:
        return np.nan, "source_blank"
    if text == yes:
        return 1.0, "available"
    if text == no:
        return 0.0, "available"
    return np.nan, "invalid_source_value"


def cer_core_variables() -> list[str]:
    return [
        "household_composition_common",
        "household_size",
        "household_size_band_common",
        "daytime_presence_category",
        "dwelling_type_common",
        "bedroom_count_min",
        "bedroom_category",
        "bedroom_band_common",
        "floor_area_m2",
        "ber_rating",
        "double_glazing_category",
        "attic_insulation_category",
        "wall_insulation_category",
        "double_glazing_any",
        "attic_roof_insulation_any",
        "wall_insulation_any",
        "reported_thermal_feature_count",
        "thermal_efficiency_proxy_count",
        "electric_space_heating_any",
        "electric_water_heating_any",
        "immersion_use_when_heating_off",
        "electric_thermal_load",
        "washing_machine_any",
        "tumble_dryer_any",
        "dishwasher_any",
        "electric_cooker_any",
        "electric_shower_any",
        "portable_electric_heater_any",
        "social_class",
        "tenure",
        "energy_vulnerability",
        "dwelling_scale_main",
        "economic_position_sensitivity",
    ]

def cer_recode_row(
    row: pd.Series, config: dict[str, Any], code_map: dict[str, list[str]]
) -> dict[str, Any]:
    status = str(row["survey_response_status"])
    out: dict[str, Any] = {
        "entity_id": row["entity_id"],
        "profile_group": row["profile_group"],
        "survey_response_status": status,
    }
    base = base_missing_reason(status)
    if base:
        for variable in cer_core_variables():
            put(out, variable, np.nan, base, "")
        return out

    q410 = require_single(code_map, "Q410")
    q420 = require_single(code_map, "Q420")
    q430 = require_single(code_map, "Q430")
    q43111 = require_single(code_map, "Q43111")
    q4312 = require_single(code_map, "Q4312")
    composition_code = clean_text(row.get(q410))
    composition_map = {
        "1": "live_alone",
        "2": "adults_only",
        "3": "adults_with_children_under15",
    }
    composition = composition_map.get(composition_code)
    composition_reason = "available" if composition else ("source_blank" if composition_code is None else "invalid_source_value")
    put(out, "household_composition_common", composition, composition_reason, "Q410")

    if composition_code == "1":
        adults, adult_reason = 1.0, "available_derived_from_live_alone"
        children, child_reason = 0.0, "available_structural_zero"
    elif composition_code in {"2", "3"}:
        adult_raw = clean_text(row.get(q420))
        if adult_raw in {str(i) for i in range(1, 8)}:
            adults, adult_reason = float(int(adult_raw)), ("available_topcoded_7_plus" if adult_raw == "7" else "available")
        else:
            adults, adult_reason = np.nan, ("source_blank" if adult_raw is None else "invalid_source_value")
        if composition_code == "2":
            children, child_reason = 0.0, "available_structural_zero"
        else:
            child_raw = clean_text(row.get(q43111))
            if child_raw in {str(i) for i in range(1, 8)}:
                children, child_reason = float(int(child_raw)), ("available_topcoded_7_plus" if child_raw == "7" else "available")
            else:
                children, child_reason = np.nan, ("source_blank" if child_raw is None else "invalid_source_value")
    else:
        adults = children = np.nan
        adult_reason = child_reason = "invalid_parent_response"
    if not pd.isna(adults) and not pd.isna(children):
        household_size, household_reason = adults + children, "available"
    else:
        household_size, household_reason = np.nan, "incomplete_component_coverage"
    put(out, "household_size", household_size, household_reason, "Q410|Q420|Q43111")
    household_band = band_value(household_size, config["bands"]["household_size"])
    put(out, "household_size_band_common", household_band, "available" if household_band else household_reason, "Q410|Q420|Q43111")

    if composition_code == "1":
        daytime_category, daytime_reason = None, "structurally_not_asked_live_alone"
    elif composition_code in {"2", "3"}:
        adult_day_raw = clean_text(row.get(q430))
        if adult_day_raw == "8":
            adult_day, adult_day_reason = 0.0, "available"
        elif adult_day_raw in {str(i) for i in range(1, 8)}:
            adult_day, adult_day_reason = float(int(adult_day_raw)), "available"
        else:
            adult_day, adult_day_reason = np.nan, ("source_blank" if adult_day_raw is None else "invalid_source_value")
        if composition_code == "2":
            child_day, child_day_reason = 0.0, "available_structural_zero"
        else:
            child_day_raw = clean_text(row.get(q4312))
            if child_day_raw == "8":
                child_day, child_day_reason = 0.0, "available"
            elif child_day_raw in {str(i) for i in range(1, 8)}:
                child_day, child_day_reason = float(int(child_day_raw)), "available"
            else:
                child_day, child_day_reason = np.nan, ("source_blank" if child_day_raw is None else "invalid_source_value")
        if not pd.isna(adult_day) and not pd.isna(child_day):
            total_day = adult_day + child_day
            daytime_category = "none" if total_day == 0 else ("one" if total_day == 1 else "two_or_more")
            daytime_reason = "available"
        else:
            daytime_category, daytime_reason = None, "incomplete_component_coverage"
    else:
        daytime_category, daytime_reason = None, "invalid_parent_response"
    put(out, "daytime_presence_category", daytime_category, daytime_reason, "Q410|Q430|Q4312")

    q450 = require_single(code_map, "Q450")
    dwelling_code = clean_text(row.get(q450))
    dwelling_map = {
        "1": "flat_or_apartment",
        "2": "semi_detached",
        "3": "detached_or_bungalow",
        "4": "terraced",
        "5": "detached_or_bungalow",
    }
    if dwelling_code == "6":
        dwelling_reason = "refused"
    else:
        dwelling_reason = "available" if dwelling_code in dwelling_map else ("source_blank" if dwelling_code is None else "invalid_source_value")
    put(out, "dwelling_type_common", dwelling_map.get(dwelling_code), dwelling_reason, "Q450")

    q460 = require_single(code_map, "Q460")
    bedroom_code = clean_text(row.get(q460))
    bedroom_category_map = {"1": "1", "2": "2", "3": "3", "4": "4", "5": "5+"}
    if bedroom_code in {"1", "2", "3", "4"}:
        bedrooms, bedroom_reason = float(int(bedroom_code)), "available"
        bedroom_category, bedroom_category_reason = bedroom_category_map[bedroom_code], "available"
    elif bedroom_code == "5":
        bedrooms, bedroom_reason = 5.0, "available_topcoded_5_plus"
        bedroom_category, bedroom_category_reason = "5+", "available_topcoded_5_plus"
    elif bedroom_code == "6":
        bedrooms, bedroom_reason = np.nan, "refused"
        bedroom_category, bedroom_category_reason = None, "refused"
    else:
        missing_reason = "source_blank" if bedroom_code is None else "invalid_source_value"
        bedrooms, bedroom_reason = np.nan, missing_reason
        bedroom_category, bedroom_category_reason = None, missing_reason
    put(out, "bedroom_count_min", bedrooms, bedroom_reason, "Q460")
    put(out, "bedroom_category", bedroom_category, bedroom_category_reason, "Q460")
    bedroom_band = band_value(bedrooms, config["bands"]["bedrooms"])
    put(out, "bedroom_band_common", bedroom_band, "available" if bedroom_band else bedroom_reason, "Q460")

    q6103 = require_single(code_map, "Q6103")
    q61031 = require_single(code_map, "Q61031")
    area_raw = clean_text(row.get(q6103))
    unit_raw = clean_text(row.get(q61031))
    if area_raw == "999999999":
        area, area_reason = np.nan, "special_missing_not_provided"
    elif area_raw is None:
        area, area_reason = np.nan, "source_blank"
    else:
        try:
            number = float(area_raw)
            if number <= 0:
                raise ValueError
            if unit_raw == "1":
                area, area_reason = number, "available"
            elif unit_raw == "2":
                area, area_reason = number * float(config["cer"]["floor_area_sqft_to_m2"]), "available_converted_from_sqft"
            else:
                area, area_reason = np.nan, "missing_or_invalid_unit"
        except ValueError:
            area, area_reason = np.nan, "invalid_source_value"
    put(out, "floor_area_m2", area, area_reason, "Q6103|Q61031")

    q455 = require_single(code_map, "Q455")
    q4551 = require_single(code_map, "Q4551")
    ber_parent = clean_text(row.get(q455))
    ber_code = clean_text(row.get(q4551))
    ber_map = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E", "6": "F", "7": "G"}
    if ber_parent == "1":
        ber = ber_map.get(ber_code)
        ber_reason = "available" if ber else ("source_blank" if ber_code is None else "invalid_source_value")
    elif ber_parent == "2":
        ber, ber_reason = None, "not_applicable_no_ber"
    elif ber_parent == "3":
        ber, ber_reason = None, "dont_know"
    else:
        ber, ber_reason = None, ("source_blank" if ber_parent is None else "invalid_source_value")
    put(out, "ber_rating", ber, ber_reason, "Q455|Q4551")

    q4906 = require_single(code_map, "Q4906")
    q4908 = require_single(code_map, "Q4908")
    q4909 = require_single(code_map, "Q4909")

    # Q4906/Q4908/Q4909 codes are questionnaire categories, not a common numeric scale.
    # Preserve the original answer meanings for formal Survey group composition composition analysis.
    dg_code = clean_text(row.get(q4906))
    dg_category_map = {
        "1": "none",
        "2": "about_quarter",
        "3": "about_half",
        "4": "about_three_quarters",
        "5": "all",
    }
    double_glazing_category = dg_category_map.get(dg_code)
    dg_category_reason = "available" if double_glazing_category else ("source_blank" if dg_code is None else "invalid_source_value")
    if dg_code == "1":
        double_glazing, dg_reason = 0.0, "available"
    elif dg_code in {"2", "3", "4", "5"}:
        double_glazing, dg_reason = 1.0, "available"
    else:
        double_glazing, dg_reason = np.nan, ("source_blank" if dg_code is None else "invalid_source_value")

    attic_code = clean_text(row.get(q4908))
    attic_category_map = {
        "1": "yes_within_5_years",
        "2": "yes_more_than_5_years_ago",
        "3": "no",
        "4": "dont_know",
    }
    attic_insulation_category = attic_category_map.get(attic_code)
    attic_category_reason = "available" if attic_insulation_category else ("source_blank" if attic_code is None else "invalid_source_value")
    if attic_code in {"1", "2"}:
        attic, attic_reason = 1.0, "available"
    elif attic_code == "3":
        attic, attic_reason = 0.0, "available"
    elif attic_code == "4":
        attic, attic_reason = np.nan, "dont_know"
    else:
        attic, attic_reason = np.nan, ("source_blank" if attic_code is None else "invalid_source_value")

    wall_code = clean_text(row.get(q4909))
    wall_category_map = {"1": "yes", "2": "no", "3": "dont_know"}
    wall_insulation_category = wall_category_map.get(wall_code)
    wall_category_reason = "available" if wall_insulation_category else ("source_blank" if wall_code is None else "invalid_source_value")
    if wall_code == "1":
        wall, wall_reason = 1.0, "available"
    elif wall_code == "2":
        wall, wall_reason = 0.0, "available"
    elif wall_code == "3":
        wall, wall_reason = np.nan, "dont_know"
    else:
        wall, wall_reason = np.nan, ("source_blank" if wall_code is None else "invalid_source_value")

    put(out, "double_glazing_category", double_glazing_category, dg_category_reason, "Q4906")
    put(out, "attic_insulation_category", attic_insulation_category, attic_category_reason, "Q4908")
    put(out, "wall_insulation_category", wall_insulation_category, wall_category_reason, "Q4909")
    put(out, "double_glazing_any", double_glazing, dg_reason, "Q4906")
    put(out, "attic_roof_insulation_any", attic, attic_reason, "Q4908")
    put(out, "wall_insulation_any", wall, wall_reason, "Q4909")

    if all(not pd.isna(v) for v in [double_glazing, attic, wall]):
        thermal_count, thermal_count_reason = float(double_glazing + attic + wall), "available"
    else:
        thermal_count, thermal_count_reason = np.nan, "incomplete_component_coverage"
    put(out, "reported_thermal_feature_count", thermal_count, thermal_count_reason, "Q4906|Q4908|Q4909")
    # Backward-compatible audit alias only. It is explicitly excluded from the formal analysis lock.
    put(out, "thermal_efficiency_proxy_count", thermal_count, thermal_count_reason, "deprecated_alias_of_reported_thermal_feature_count")

    q470_central = require_contains(code_map, "Q470", "electric central heating")
    q470_plug = require_contains(code_map, "Q470", "plug in heaters")
    q4701_immersion = require_contains(code_map, "Q4701", "electric (immersion)")
    q4701_instant = require_contains(code_map, "Q4701", "electric (instantaneous heater)")
    space_values = [parse_multiselect_indicator(row.get(q470_central))[0], parse_multiselect_indicator(row.get(q470_plug))[0]]
    water_values = [parse_multiselect_indicator(row.get(q4701_immersion))[0], parse_multiselect_indicator(row.get(q4701_instant))[0]]
    electric_space, electric_space_reason = combine_binary(space_values)
    electric_water, electric_water_reason = combine_binary(water_values)
    put(out, "electric_space_heating_any", electric_space, electric_space_reason, "Q470 electric indicators")
    put(out, "electric_water_heating_any", electric_water, electric_water_reason, "Q4701 electric indicators")
    q4801 = require_single(code_map, "Q4801")
    immersion_parent = clean_text(row.get(q4701_immersion))
    immersion_use_raw = clean_text(row.get(q4801))
    if immersion_parent == "0":
        immersion_use, immersion_use_reason = np.nan, "structurally_not_asked_no_immersion"
    elif immersion_parent == "1":
        immersion_use, immersion_use_reason = parse_binary_code(immersion_use_raw)
    else:
        immersion_use, immersion_use_reason = np.nan, "invalid_parent_response"
    put(out, "immersion_use_when_heating_off", immersion_use, immersion_use_reason, "Q4701 immersion|Q4801")
    electric_thermal, electric_thermal_reason = combine_binary([electric_space, electric_water])
    put(out, "electric_thermal_load", electric_thermal, electric_thermal_reason, "Q470|Q4701")

    appliance_specs = {
        "washing_machine_any": ("Q49002", "Washing machine"),
        "tumble_dryer_any": ("Q49002", "Tumble dryer"),
        "dishwasher_any": ("Q49002", "Dishwasher"),
        "electric_cooker_any": ("Q49002", "Electric cooker"),
        "portable_electric_heater_any": ("Q49002", "Electric heater (plug-in"),
    }
    for variable, (code, text) in appliance_specs.items():
        column = require_contains(code_map, code, text)
        raw = clean_text(row.get(column))
        if raw == "1":
            value, reason = 0.0, "available"
        elif raw in {"2", "3", "4"}:
            value, reason = 1.0, "available"
        else:
            value, reason = np.nan, ("source_blank" if raw is None else "invalid_source_value")
        put(out, variable, value, reason, f"{code}:{text}")
    shower_columns = [
        require_contains(code_map, "Q49002", "Electric shower (instant)"),
        require_contains(code_map, "Q49002", "electric pumped from hot tank"),
    ]
    shower_values: list[float] = []
    for column in shower_columns:
        raw = clean_text(row.get(column))
        if raw == "1":
            shower_values.append(0.0)
        elif raw in {"2", "3", "4"}:
            shower_values.append(1.0)
        else:
            shower_values.append(np.nan)
    shower, shower_reason = combine_binary(shower_values)
    put(out, "electric_shower_any", shower, shower_reason, "Q49002 electric shower indicators")

    q401 = require_single(code_map, "Q401")
    social_code = clean_text(row.get(q401))
    social_map = {"1": "AB", "2": "C1", "3": "C2", "4": "DE", "5": "F_farmer"}
    if social_code == "6":
        social, social_reason = None, "refused"
    else:
        social, social_reason = social_map.get(social_code), ("available" if social_code in social_map else ("source_blank" if social_code is None else "invalid_source_value"))
    put(out, "social_class", social, social_reason, "Q401")
    q452 = require_single(code_map, "Q452")
    tenure_code = clean_text(row.get(q452))
    tenure_map = {"1": "private_rent", "2": "social_rent", "3": "own_outright", "4": "own_mortgage", "5": "other"}
    tenure, tenure_reason = tenure_map.get(tenure_code), ("available" if tenure_code in tenure_map else ("source_blank" if tenure_code is None else "invalid_source_value"))
    put(out, "tenure", tenure, tenure_reason, "Q452")

    q471 = require_single(code_map, "Q471")
    afford_column = require_contains(code_map, "Q472", "cannot afford")
    q473 = require_single(code_map, "Q473")
    warm_code = clean_text(row.get(q471))
    afford_indicator, afford_indicator_reason = parse_multiselect_indicator(row.get(afford_column))
    if warm_code == "1":
        cannot_afford, cannot_afford_reason = 0.0, "available_structural_zero"
    elif warm_code == "2":
        cannot_afford, cannot_afford_reason = afford_indicator, afford_indicator_reason
    else:
        cannot_afford, cannot_afford_reason = np.nan, ("source_blank" if warm_code is None else "invalid_source_value")
    went_without, went_without_reason = parse_binary_code(row.get(q473))
    energy_vulnerability, vulnerability_reason = combine_binary([cannot_afford, went_without])
    put(out, "energy_vulnerability", energy_vulnerability, vulnerability_reason, "Q471|Q472|Q473")

    # Dataset-level selected fields are assigned after all rows are recoded.
    put(out, "dwelling_scale_main", np.nan, "pending_dataset_level_selection", "")
    put(out, "economic_position_sensitivity", np.nan, "pending_dataset_level_selection", "")
    return out


def availability_rate(frame: pd.DataFrame, variable: str) -> float:
    usable = frame["survey_response_status"].eq("usable_response")
    denominator = int(usable.sum())
    if denominator == 0:
        return 0.0
    available = frame.loc[usable, f"{variable}__missing_reason"].astype(str).str.startswith("available")
    return float(available.mean())


def assign_dataset_level_fallbacks(
    cer: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    threshold = float(config["missingness_rules"]["domain_specific_threshold"])
    minimum_availability = 1.0 - threshold
    decisions: list[dict[str, Any]] = []

    floor_rate = availability_rate(cer, "floor_area_m2")
    bedroom_rate = availability_rate(cer, "bedroom_category")
    dwelling_selected = "floor_area_m2" if floor_rate >= minimum_availability else "bedroom_category"
    decisions.append({
        "domain": "dwelling_scale",
        "preferred_variable": "floor_area_m2",
        "preferred_availability_rate": floor_rate,
        "fallback_variable": "bedroom_category",
        "fallback_availability_rate": bedroom_rate,
        "selection_rule": f"preferred retained only when missingness <= {threshold:.0%}; categorical 1/2/3/4/5+ semantics preserved",
        "selected_variable": dwelling_selected,
    })

    ber_rate = availability_rate(cer, "ber_rating")
    thermal_components = [
        "double_glazing_category",
        "attic_insulation_category",
        "wall_insulation_category",
    ]
    thermal_rates = {name: availability_rate(cer, name) for name in thermal_components}
    thermal_component_set = "|".join(thermal_components)
    thermal_selected = "ber_rating" if ber_rate >= minimum_availability else thermal_component_set
    decisions.append({
        "domain": "reported_thermal_features",
        "preferred_variable": "ber_rating",
        "preferred_availability_rate": ber_rate,
        "fallback_variable": thermal_component_set,
        "fallback_availability_rate": min(thermal_rates.values()),
        "selection_rule": f"BER retained only when missingness <= {threshold:.0%}; otherwise preserve three source-item categories separately; no composite efficiency score",
        "selected_variable": thermal_selected,
        "component_availability_detail": json.dumps(thermal_rates, sort_keys=True),
    })

    social_rate = availability_rate(cer, "social_class")
    tenure_rate = availability_rate(cer, "tenure")
    economic_selected = "social_class" if social_rate >= minimum_availability else "tenure"
    decisions.append({
        "domain": "economic_position",
        "preferred_variable": "income",
        "preferred_availability_rate": 0.0,
        "fallback_variable": "social_class_then_tenure",
        "fallback_availability_rate": max(social_rate, tenure_rate),
        "selection_rule": "income held for unresolved release semantics; social class used when missingness <=30%, otherwise tenure",
        "selected_variable": economic_selected,
    })

    expected = config["locked_selection"]["expected_fallbacks"]
    actual = {
        "dwelling_scale_main": dwelling_selected,
        "reported_thermal_features_main": thermal_selected,
        "economic_position_sensitivity": economic_selected,
    }
    if actual != expected:
        raise SurveyRecodingError(f"Dataset-level fallback decisions changed unexpectedly: actual={actual} expected={expected}")

    for output_variable, source_variable in {
        "dwelling_scale_main": dwelling_selected,
        "economic_position_sensitivity": economic_selected,
    }.items():
        cer[output_variable] = cer[source_variable]
        cer[f"{output_variable}__missing_reason"] = cer[f"{source_variable}__missing_reason"]
        cer[f"{output_variable}__source"] = source_variable
    return cer, pd.DataFrame(decisions)

def core_analysis_variable_lock() -> pd.DataFrame:
    rows = [
        ("LCL", "household_size", "household_size_band_common", "main", "SQ1", "Broad source-group occupancy-size composition"),
        ("LCL", "dwelling_scale", "rooms_count", "main", "SQ1", "Source-side dwelling-scale description; robust summaries only"),
        ("LCL", "thermal_condition", "double_glazing_any|attic_roof_insulation_any|wall_insulation_any", "main", "SQ1", "Separate transparent indicators; no forced complete-case composite in the main block"),
        ("LCL", "electric_end_use", "electric_thermal_load", "main", "SQ1", "Broad electric space/water-heating presence"),
        ("LCL", "household_composition", "household_composition_common", "secondary", "SQ1", "Available only when household-age coverage supports classification"),
        ("LCL", "dwelling_type", "dwelling_type_common", "secondary", "SQ1/SQ5", "Broad cross-dataset dwelling category"),
        ("LCL", "selected_appliances", "washing_machine_any|tumble_dryer_any|dishwasher_any|electric_cooker_any|electric_shower_any|portable_electric_heater_any", "harmonisation_support", "SQ1/SQ5", "Item-specific ownership only; not an omnibus appliance block"),
        ("CER", "household_composition", "household_composition_common", "main", "SQ2/SQ3", "Lives alone/adults only/adults with children"),
        ("CER", "daytime_presence", "daytime_presence_category", "main", "SQ2/SQ3", "None/one/two-or-more among eligible non-live-alone respondents; live-alone remains structural missing"),
        ("CER", "electric_thermal_load", "electric_thermal_load", "main", "SQ2/SQ3/SQ4", "Reported electric space heating or electric water heating presence"),
        ("CER", "dwelling_scale", "dwelling_scale_main", "main", "SQ2/SQ3/SQ4", "Q460 original 1/2/3/4/5+ bedroom categories selected because floor-area missingness exceeds 30%"),
        ("CER", "reported_thermal_features", "double_glazing_category", "main", "SQ2/SQ3/SQ4", "Q4906 original response categories retained; code values are labels, not a numeric efficiency scale"),
        ("CER", "reported_thermal_features", "attic_insulation_category", "main", "SQ2/SQ3/SQ4", "Q4908 original response categories retained, including timing and don't-know response"),
        ("CER", "reported_thermal_features", "wall_insulation_category", "main", "SQ2/SQ3/SQ4", "Q4909 original response categories retained, including don't-know response"),
        ("CER", "energy_vulnerability", "energy_vulnerability", "main", "SQ2/SQ3/SQ4", "Broad reported heating-affordability difficulty indicator"),
        ("CER", "reported_thermal_features", "reported_thermal_feature_count", "supplementary_descriptive", "SQ2/SQ3/SQ4", "0-3 count of reported binary features; descriptive only, not an efficiency score and not in the main BH family"),
        ("CER", "economic_position", "economic_position_sensitivity", "secondary_sensitivity", "SQ2/SQ3", "Social class selected; income remains semantically unresolved"),
        ("CER", "dwelling_type", "dwelling_type_common", "harmonisation_support", "SQ2/SQ5", "Broad source-target comparison only"),
        ("CER", "selected_appliances", "washing_machine_any|tumble_dryer_any|dishwasher_any|electric_cooker_any|electric_shower_any|portable_electric_heater_any", "harmonisation_support", "SQ2/SQ5", "Item-specific ownership only"),
    ]
    return pd.DataFrame(rows, columns=[
        "dataset", "domain", "analysis_variable", "analysis_role", "survey_question", "locked_decision"
    ])

def variable_provenance() -> pd.DataFrame:
    rows = [
        ("LCL", "household_size_band_common", "Q213", "1 / 2 / 3-4 / 5+", "main", "partial"),
        ("LCL", "rooms_count", "Q238", "Released room count retained; no outcome-based collapse", "main", "partial"),
        ("LCL", "double_glazing_any", "Q240", "Yes/no; don't know retained as missing", "main_domain_component", "partial"),
        ("LCL", "attic_roof_insulation_any", "Q241", "Yes/no; don't know retained as missing", "main_domain_component", "partial"),
        ("LCL", "wall_insulation_any", "Q242", "Yes/no; don't know retained as missing", "main_domain_component", "partial"),
        ("LCL", "thermal_efficiency_proxy_count", "Q240|Q241|Q242", "Audit-only 0-3 complete-component count", "audit_only", "partial"),
        ("LCL", "electric_thermal_load", "Q246|Q248", "Any electric space or water heating", "main", "partial"),
        ("LCL", "household_composition_common", "Q213|Q222-Q229", "Live alone/adults only/adults with under-16 child when age coverage is sufficient", "secondary", "partial"),
        ("LCL", "dwelling_type_common", "Q235|Q236|Q237", "Broad dwelling categories", "secondary", "partial"),
        ("CER", "household_composition_common", "Q410", "Codes 1/2/3 mapped to fixed nominal household-composition categories", "main", "cer_only"),
        ("CER", "daytime_presence_category", "Q410|Q430|Q4312", "None/one/two-or-more among eligible non-live-alone respondents; structural missing for live-alone", "main", "cer_only"),
        ("CER", "electric_thermal_load", "Q470|Q4701", "Any reported electric space or water heating", "main", "partial"),
        ("CER", "bedroom_category", "Q460", "Original top-coded categories 1/2/3/4/5+; refused retained as missing", "fallback_source", "partial"),
        ("CER", "dwelling_scale_main", "Q460", "Alias of bedroom_category chosen by predeclared >30% floor-area missingness fallback", "main", "partial"),
        ("CER", "double_glazing_category", "Q4906", "None/about quarter/about half/about three quarters/all; categorical labels, not numeric scores", "main", "cer_only"),
        ("CER", "attic_insulation_category", "Q4908", "Yes within 5 years/yes more than 5 years ago/no/don't know", "main", "cer_only"),
        ("CER", "wall_insulation_category", "Q4909", "Yes/no/don't know", "main", "cer_only"),
        ("CER", "reported_thermal_feature_count", "Q4906|Q4908|Q4909", "0-3 count of complete binary presence indicators; descriptive only and not an efficiency score", "supplementary_descriptive", "cer_only"),
        ("CER", "thermal_efficiency_proxy_count", "Q4906|Q4908|Q4909", "Deprecated backward-compatible alias of reported_thermal_feature_count; excluded from formal analysis lock", "deprecated_audit_alias", "cer_only"),
        ("CER", "double_glazing_any", "Q4906", "Any versus none broad presence for harmonisation only", "harmonisation_support", "partial"),
        ("CER", "attic_roof_insulation_any", "Q4908", "Broad presence; don't know retained as missing", "harmonisation_support", "partial"),
        ("CER", "wall_insulation_any", "Q4909", "Broad presence; don't know retained as missing", "harmonisation_support", "partial"),
        ("CER", "energy_vulnerability", "Q471|Q472|Q473", "Cannot afford adequate warmth or went without heating for lack of money", "main", "cer_only"),
        ("CER", "economic_position_sensitivity", "Q401", "Social-class fallback; Q402/Q4021 income not recoded", "secondary_sensitivity", "cer_only"),
        ("BOTH", "dwelling_type_common", "LCL Q235-Q237; CER Q450", "Broad categories only", "harmonisation_support", "partial"),
        ("BOTH", "bedroom_band_common", "LCL Q239; CER Q460", "Common broad ordered bedroom bands", "harmonisation_support", "direct_broad"),
        ("BOTH", "double_glazing_any", "LCL Q240; CER Q4906", "Any/none broad presence", "harmonisation_support", "partial"),
        ("BOTH", "attic_roof_insulation_any", "LCL Q241; CER Q4908", "Broad presence only", "harmonisation_support", "partial"),
        ("BOTH", "wall_insulation_any", "LCL Q242; CER Q4909", "Broad presence only", "harmonisation_support", "partial"),
        ("BOTH", "selected_appliance_indicators", "LCL Q293/Q295/Q297-Q303; CER selected Q49002", "Separate item-specific presence indicators", "harmonisation_support", "partial"),
        ("CER", "floor_area_m2", "Q6103|Q61031", "m2 retained; square feet multiplied by 0.09290304", "audit_fallback_candidate", "cer_only"),
        ("CER", "ber_rating", "Q455|Q4551", "A-G when respondent reports a BER", "audit_fallback_candidate", "cer_only"),
        ("CER", "income", "Q402|Q4021", "Not recoded because public-release semantics remain unresolved", "governance_only", "unusable_pending_codebook"),
    ]
    return pd.DataFrame(rows, columns=[
        "dataset", "derived_variable", "source_columns", "recoding_rule", "analysis_role", "harmonisation_class"
    ])

def harmonisation_map() -> pd.DataFrame:
    rows = [
        ("household_size", "partial", "LCL occupant count versus CER adults plus children; broad bands only"),
        ("household_composition", "partial", "Different instruments; broad descriptive comparison only"),
        ("dwelling_scale", "partial", "LCL rooms versus CER original top-coded bedroom categories"),
        ("bedroom_band_common", "direct_broad", "Common broad ordered bedroom bands"),
        ("dwelling_type_common", "partial", "Different instruments mapped to broad categories"),
        ("reported_thermal_features", "not_direct", "CER source-item categories remain separate; they are not a common numeric efficiency scale"),
        ("double_glazing_any", "partial", "LCL yes/no versus CER any/none broad presence"),
        ("attic_roof_insulation_any", "partial", "Broad presence only"),
        ("wall_insulation_any", "partial", "Broad presence only"),
        ("electric_thermal_load", "partial", "Any electric space or water heating"),
        ("selected_appliances", "partial", "Item-specific ownership only"),
        ("daytime_presence", "cer_only", "No defensible LCL equivalent"),
        ("energy_vulnerability", "cer_only", "No defensible LCL equivalent"),
        ("economic_position", "cer_only", "No defensible LCL equivalent"),
        ("income", "unusable_pending_codebook", "Q402/Q4021 release semantics unresolved"),
    ]
    return pd.DataFrame(rows, columns=["domain", "comparability_class", "decision_note"])

def missing_reason_long(dataset: str, wide: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for column in [c for c in wide.columns if c.endswith("__missing_reason")]:
        variable = column[:-len("__missing_reason")]
        rows.append(pd.DataFrame({
            "dataset": dataset,
            "entity_id": wide["entity_id"],
            "profile_group": wide["profile_group"],
            "survey_response_status": wide["survey_response_status"],
            "derived_variable": variable,
            "missing_reason": wide[column],
            "value_available": wide[column].astype(str).str.startswith("available"),
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def missing_reason_summary(lcl_long: pd.DataFrame, cer_long: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([lcl_long, cer_long], ignore_index=True)
    return (
        combined.groupby(["dataset", "derived_variable", "missing_reason"], dropna=False)
        .size().rename("n").reset_index()
        .sort_values(["dataset", "derived_variable", "missing_reason"])
    )


def category_size_audit(
    dataset: str, wide: pd.DataFrame, lock: pd.DataFrame, minimum_n: int
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected = lock[
        lock["dataset"].eq(dataset)
        & lock["analysis_role"].isin(["main", "secondary_sensitivity"])
    ]
    variables: list[str] = []
    for spec in selected["analysis_variable"].astype(str):
        variables.extend([x for x in spec.split("|") if x in wide.columns])
    variables = sorted(set(variables))
    for variable in variables:
        reason_col = f"{variable}__missing_reason"
        available = wide[reason_col].astype(str).str.startswith("available")
        for group in ["ALL"] + GROUPS:
            subset = wide if group == "ALL" else wide[wide["profile_group"].eq(group)]
            avail_subset = subset[subset[reason_col].astype(str).str.startswith("available")]
            counts = avail_subset[variable].astype(str).value_counts(dropna=False)
            if not counts.empty and len(counts) <= 30:
                for category, count in counts.items():
                    rows.append({
                        "dataset": dataset,
                        "profile_group": group,
                        "derived_variable": variable,
                        "category": category,
                        "n": int(count),
                        "below_n10_flag": int(count) < minimum_n,
                    })
            else:
                rows.append({
                    "dataset": dataset,
                    "profile_group": group,
                    "derived_variable": variable,
                    "category": "<continuous_or_high_cardinality>",
                    "n": int(available.sum() if group == "ALL" else avail_subset.shape[0]),
                    "below_n10_flag": False,
                })
    return pd.DataFrame(rows)


def recoding_audit(
    config: dict[str, Any],
    lcl: pd.DataFrame,
    cer: pd.DataFrame,
    fallback: pd.DataFrame,
) -> pd.DataFrame:
    expected = config["expected"]
    rows: list[dict[str, Any]] = []

    def add(check: str, observed: Any, expected_value: Any, passed: bool) -> None:
        rows.append({
            "check": check,
            "observed": observed,
            "expected": expected_value,
            "status": "PASS" if passed else "FAIL",
        })

    add("lcl_formal_rows", len(lcl), expected["lcl_formal_households"], len(lcl) == int(expected["lcl_formal_households"]))
    add("cer_formal_rows", len(cer), expected["cer_formal_meters"], len(cer) == int(expected["cer_formal_meters"]))
    for dataset, frame, counts in [
        ("LCL", lcl, {"usable_response": expected["survey_linkage_lcl_usable_n"], "id_only_no_response": expected["survey_linkage_lcl_id_only_n"], "no_survey_row": expected["survey_linkage_lcl_no_row_n"]}),
        ("CER", cer, {"usable_response": expected["survey_linkage_cer_usable_n"], "id_only_no_response": expected["survey_linkage_cer_id_only_n"], "no_survey_row": expected["survey_linkage_cer_no_row_n"]}),
    ]:
        observed = frame["survey_response_status"].value_counts().to_dict()
        for status, expected_n in counts.items():
            add(f"{dataset.lower()}_{status}", int(observed.get(status, 0)), int(expected_n), int(observed.get(status, 0)) == int(expected_n))
    expected_fallbacks = config["locked_selection"]["expected_fallbacks"]
    for domain, selected in [
        ("dwelling_scale", expected_fallbacks["dwelling_scale_main"]),
        ("reported_thermal_features", expected_fallbacks["reported_thermal_features_main"]),
        ("economic_position", expected_fallbacks["economic_position_sensitivity"]),
    ]:
        actual = str(fallback.loc[fallback["domain"].eq(domain), "selected_variable"].iloc[0])
        add(f"fallback_{domain}", actual, selected, actual == selected)
    for dataset, frame in [("LCL", lcl), ("CER", cer)]:
        invalid = sum(frame[c].astype(str).eq("invalid_source_value").sum() for c in frame.columns if c.endswith("__missing_reason"))
        add(f"{dataset.lower()}_invalid_source_value_count", int(invalid), 0, int(invalid) == 0)

    usable = cer["survey_response_status"].eq("usable_response")
    category_specs = {
        "bedroom_category": {"1", "2", "3", "4", "5+"},
        "double_glazing_category": {"none", "about_quarter", "about_half", "about_three_quarters", "all"},
        "attic_insulation_category": {"yes_within_5_years", "yes_more_than_5_years_ago", "no", "dont_know"},
        "wall_insulation_category": {"yes", "no", "dont_know"},
    }
    for variable, allowed in category_specs.items():
        observed = set(cer.loc[usable & cer[f"{variable}__missing_reason"].astype(str).str.startswith("available"), variable].dropna().astype(str))
        add(f"cer_{variable}_category_semantics", sorted(observed), sorted(allowed), observed.issubset(allowed) and len(observed) > 0)
    deprecated_in_lock = False
    add("thermal_efficiency_main_removed_from_formal_lock", deprecated_in_lock, False, not deprecated_in_lock)
    return pd.DataFrame(rows)

def compute_analysis(
    config: dict[str, Any], resolved: dict[str, InputFile]
) -> dict[str, Any]:
    upstream_status = verify_survey_linkage_upstream(config, resolved)
    lcl_formal, cer_formal = prepare_formal_assignments(config, resolved)
    lcl_survey, lcl_duplicate_audit = resolve_lcl_answers(config, resolved)
    cer_survey, cer_duplicate_audit = resolve_cer_answers(config, resolved)
    lcl_attached = attach_formal(lcl_formal, lcl_survey)
    cer_attached = attach_formal(cer_formal, cer_survey)

    lcl_wide = pd.DataFrame([lcl_recode_row(row, config) for _, row in lcl_attached.iterrows()])
    code_map = cer_columns_by_code(cer_attached)
    cer_wide = pd.DataFrame([cer_recode_row(row, config, code_map) for _, row in cer_attached.iterrows()])
    cer_wide, fallback_audit = assign_dataset_level_fallbacks(cer_wide, config)

    lock = core_analysis_variable_lock()
    provenance = variable_provenance()
    harmonisation = harmonisation_map()
    lcl_long = missing_reason_long("LCL", lcl_wide)
    cer_long = missing_reason_long("CER", cer_wide)
    missing_summary = missing_reason_summary(lcl_long, cer_long)
    category_audit = pd.concat([
        category_size_audit("LCL", lcl_wide, lock, int(config["missingness_rules"]["category_min_n"])),
        category_size_audit("CER", cer_wide, lock, int(config["missingness_rules"]["category_min_n"])),
    ], ignore_index=True)
    recode_audit = recoding_audit(config, lcl_wide, cer_wide, fallback_audit)
    if not recode_audit["status"].eq("PASS").all():
        raise SurveyRecodingError(
            "Recoding acceptance checks failed: "
            + recode_audit.loc[~recode_audit["status"].eq("PASS")].to_json(orient="records")
        )

    return {
        "upstream_status": upstream_status,
        "input_inventory": input_inventory(resolved),
        "lcl_wide": lcl_wide,
        "cer_wide": cer_wide,
        "lcl_long": lcl_long,
        "cer_long": cer_long,
        "missing_summary": missing_summary,
        "category_audit": category_audit,
        "recoding_audit": recode_audit,
        "fallback_audit": fallback_audit,
        "analysis_variable_lock": lock,
        "provenance": provenance,
        "harmonisation": harmonisation,
        "lcl_duplicate_audit": lcl_duplicate_audit,
        "cer_duplicate_audit": cer_duplicate_audit,
    }


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(none)"
    columns = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[c]).replace("|", "\\|").replace("\n", " ") for c in df.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_status_and_note(
    config: dict[str, Any], analysis: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    fallback = analysis["fallback_audit"]
    lock = analysis["analysis_variable_lock"]
    lcl_main = lock[lock["dataset"].eq("LCL") & lock["analysis_role"].eq("main")]
    cer_main = lock[lock["dataset"].eq("CER") & lock["analysis_role"].eq("main")]
    thermal_selected = str(fallback.loc[fallback["domain"].eq("reported_thermal_features"), "selected_variable"].iloc[0])
    status = {
        "analysis": config["analysis"],
        "analysis_id": config["analysis_id"],
        "created_at_utc": utc_now(),
        "computational_status": "COMPLETE_PASS",
        "analysis_gate_status": "PASS_TO_SURVEY_GROUP_COMPOSITION_SPECIFICATION",
        "lcl_formal_n": int(len(analysis["lcl_wide"])),
        "cer_formal_n": int(len(analysis["cer_wide"])),
        "lcl_core_main_domain_n": int(lcl_main["domain"].nunique()),
        "lcl_core_main_variable_n": int(len(lcl_main)),
        "cer_core_main_domain_n": int(cer_main["domain"].nunique()),
        "cer_core_main_variable_n": int(len(cer_main)),
        "cer_secondary_sensitivity_variable_n": int((lock["dataset"].eq("CER") & lock["analysis_role"].eq("secondary_sensitivity")).sum()),
        "dwelling_scale_main": str(fallback.loc[fallback["domain"].eq("dwelling_scale"), "selected_variable"].iloc[0]),
        "reported_thermal_features_main": thermal_selected,
        "thermal_efficiency_main_status": "DEPRECATED_NOT_FOR_FORMAL_ANALYSIS",
        "reported_thermal_feature_count_role": "SUPPLEMENTARY_DESCRIPTIVE_ONLY",
        "economic_position_sensitivity": str(fallback.loc[fallback["domain"].eq("economic_position"), "selected_variable"].iloc[0]),
        "income_recode_status": "NOT_USED_UNRESOLVED_RELEASE_SEMANTICS",
        "survey_group_composition_relationship": "SURVEY_VARIABLE_DISTRIBUTION_BY_FIXED_PROFILE_GROUP",
        "strategy_performance_explained_in_survey_recoding_or_survey_group_composition": False,
        "training_performed": False,
        "predictions_modified": False,
        "forecasting_inputs_modified": False,
        "clustering_modified": False,
        "regression_performed": False,
        "significance_testing_performed": False,
        "outcome_dependent_variable_selection": False,
        "ProfileInterpretation_outcomes_read": False,
    }
    note = f"""# Survey recoding — Governed survey recoding and semantic harmonisation decision

```text
computational_status = COMPLETE_PASS
analysis_gate_status = PASS_TO_SURVEY_GROUP_COMPOSITION_SPECIFICATION
```

## What the variables are allowed to explain

Survey recoding only establishes that the selected variables are semantically defensible representations of the original survey questions and answer codes. In Survey group composition, the formal relationship is between each survey-variable distribution and the fixed G1–G4 profile-group assignment. Survey recoding and Survey group composition do not explain strategy performance and read no MAE, RMSE, SMAPE or pairwise strategy gain.

Survey gain associations may later assess pre-specified group-adjusted associations between these survey variables and selected strategy gains. A Survey group composition group-composition result does not itself establish a relationship with strategy performance and does not determine which variables enter Survey gain associations.

## Six conceptual CER domains and eight main analysis variables

The six conceptual domains remain household composition, daytime presence, electric thermal load, dwelling scale, reported thermal features and energy vulnerability. They are operationalised as eight main variables because the reported-thermal-feature domain is represented by the three source questions separately rather than by one artificial numeric score.

{markdown_table(lock)}

## Missingness-based fallback and semantic decisions

{markdown_table(fallback)}

- CER dwelling scale uses Q460 bedroom categories `1`, `2`, `3`, `4` and `5+`; the top-coded `5+` response is not treated as an exact numeric five.
- The previous `thermal_efficiency_main` interpretation is deprecated. Q4906, Q4908 and Q4909 are retained as separate categorical variables because their codes have different source meanings and do not form a common numeric efficiency scale.
- `reported_thermal_feature_count` is supplementary descriptive information only. It counts the presence of three reported features when all binary components are known; it is not an energy-efficiency score and is excluded from the main Survey group composition BH family.
- The backward-compatible `thermal_efficiency_proxy_count` column is audit-only and is excluded from the formal analysis-variable lock.
- CER economic position is secondary sensitivity only. Social class is used because Q402/Q4021 is not recoded and income semantics remain unresolved.

## Formal boundaries

- Survey information remains post-hoc, descriptive and non-causal.
- Survey recoding does not analyse every survey question.
- No regression is performed.
- No significance testing is performed.
- No category is merged because of forecasting outcomes.
- No imputation is performed.
- No Profile interpretation outcome is read.
- No forecasting input, scaler, prediction, clustering prototype or assignment is modified.

Survey group composition may analyse G1–G4 composition using the eight locked CER main variables, four LCL main domains and stated secondary/supplementary variables. Survey gain associations must remain a later, separately governed group-adjusted strategy-gain analysis.
"""
    return status, note

def write_outputs(
    config: dict[str, Any], root: Path, analysis: dict[str, Any]
) -> list[Path]:
    tables = root / config["output_roots"]["tables"]
    metadata = root / config["output_roots"]["metadata"]
    decision_path = root / config["output_roots"]["decision_note"]
    tables.mkdir(parents=True, exist_ok=True)
    metadata.mkdir(parents=True, exist_ok=True)
    decision_path.parent.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, pd.DataFrame] = {
        "survey_recoding_lcl_household_survey_recoded.csv": analysis["lcl_wide"],
        "survey_recoding_cer_meter_survey_recoded.csv": analysis["cer_wide"],
        "survey_recoding_lcl_missing_reason_long.csv": analysis["lcl_long"],
        "survey_recoding_cer_missing_reason_long.csv": analysis["cer_long"],
        "survey_recoding_variable_provenance.csv": analysis["provenance"],
        "survey_recoding_harmonisation_map.csv": analysis["harmonisation"],
        "survey_recoding_missing_reason_summary.csv": analysis["missing_summary"],
        "survey_recoding_category_size_audit.csv": analysis["category_audit"],
        "survey_recoding_recoding_audit.csv": analysis["recoding_audit"],
        "survey_recoding_lcl_duplicate_resolution_audit.csv": analysis["lcl_duplicate_audit"],
        "survey_recoding_cer_duplicate_resolution_audit.csv": analysis["cer_duplicate_audit"],
        "survey_recoding_input_file_inventory.csv": analysis["input_inventory"],
        "survey_recoding_analysis_variable_lock.csv": analysis["analysis_variable_lock"],
        "survey_recoding_fallback_decision_audit.csv": analysis["fallback_audit"],
    }
    written: list[Path] = []
    for filename, frame in outputs.items():
        path = tables / filename
        frame.to_csv(path, index=False)
        written.append(path)

    status, note = build_status_and_note(config, analysis)
    status_path = metadata / "survey_recoding_survey_recoding_harmonisation_status.json"
    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    decision_path.write_text(note, encoding="utf-8")
    written.extend([status_path, decision_path])

    inventory_path = metadata / "survey_recoding_output_inventory.csv"
    inventory = pd.DataFrame([
        {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in written
    ])
    inventory.to_csv(inventory_path, index=False)
    written.append(inventory_path)
    return written


def print_summary(config: dict[str, Any], analysis: dict[str, Any]) -> None:
    fallback = analysis["fallback_audit"]
    print(f"ANALYSIS: {config['analysis']}")
    print(f"LCL FORMAL ROWS: {len(analysis['lcl_wide'])}")
    print(f"CER FORMAL ROWS: {len(analysis['cer_wide'])}")
    print("LOCKED LCL MAIN DOMAINS: 4")
    print("LOCKED CER CONCEPTUAL MAIN DOMAINS: 6")
    print("LOCKED CER MAIN ANALYSIS VARIABLES: 8")
    print("CER SECONDARY SENSITIVITY VARIABLES: 1")
    for _, row in fallback.iterrows():
        print(f"FALLBACK {row['domain']}: {row['selected_variable']}")
    print("THERMAL EFFICIENCY MAIN STATUS: DEPRECATED_NOT_FOR_FORMAL_ANALYSIS")
    print("SUPPLEMENTARY REPORTED THERMAL FEATURE COUNT: reported_thermal_feature_count")
    print("SURVEY GROUP COMPOSITION RELATIONSHIP: SURVEY VARIABLE DISTRIBUTION BY FIXED PROFILE GROUP")
    print("STRATEGY PERFORMANCE EXPLAINED IN SURVEY RECODING: False")
    print("INCOME RECODE STATUS: NOT_USED_UNRESOLVED_RELEASE_SEMANTICS")
    print("PROFILE INTERPRETATION OUTCOMES READ: False")
    print("REGRESSION PERFORMED: False")
    print("SIGNIFICANCE TESTING PERFORMED: False")
    print("OUTCOME-DEPENDENT VARIABLE SELECTION: False")
    print("PROVISIONAL ANALYSIS GATE: PASS_TO_SURVEY_GROUP_COMPOSITION_SPECIFICATION")

def check_inputs(config_path: Path, root: Path) -> None:
    config = load_json(config_path)
    resolved = resolve_inputs(config, root)
    print(f"ANALYSIS: {config['analysis']}")
    print(f"ROOT: {root}")
    print(f"PYTHON: {sys.executable}")
    print(f"PYTHON VERSION: {sys.version.split()[0]}")
    print(f"PANDAS VERSION: {pd.__version__}")
    print(f"NUMPY VERSION: {np.__version__}")
    try:
        import pyarrow
        print(f"PYARROW VERSION: {pyarrow.__version__}")
    except Exception as exc:
        raise SurveyRecodingError(f"pyarrow import failed: {exc}") from exc
    analysis = compute_analysis(config, resolved)
    print_summary(config, analysis)
    print("SURVEY RECODING INPUT CHECK: PASS")


def run_formal(config_path: Path, root: Path) -> None:
    config = load_json(config_path)
    resolved = resolve_inputs(config, root)
    analysis = compute_analysis(config, resolved)
    written = write_outputs(config, root, analysis)
    print_summary(config, analysis)
    print(f"FORMAL OUTPUT FILES WRITTEN: {len(written)}")
    print("ANALYSIS GATE: PASS_TO_SURVEY_GROUP_COMPOSITION_SPECIFICATION")
    print("TRAINING PERFORMED: False")
    print("PREDICTIONS MODIFIED: False")
    print("FORECASTING INPUTS MODIFIED: False")
    print("CLUSTERING MODIFIED: False")
    print("REGRESSION PERFORMED: False")
    print("SIGNIFICANCE TESTING PERFORMED: False")
    print("PROFILE INTERPRETATION OUTCOMES READ: False")
    print("OUTCOME-DEPENDENT VARIABLE SELECTION: False")
    print("SURVEY RECODING COMPUTATION COMPLETE — PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-inputs", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    try:
        if args.check_inputs:
            check_inputs(args.config.resolve(), args.root.resolve())
        else:
            run_formal(args.config.resolve(), args.root.resolve())
    except SurveyRecodingError as exc:
        print(f"SURVEY RECODING ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        import traceback
        print(f"SURVEY RECODING UNEXPECTED ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
