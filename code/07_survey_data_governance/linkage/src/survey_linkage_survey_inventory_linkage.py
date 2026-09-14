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

ANALYSIS_NAME = "Survey linkage — Dual LCL/CER survey inventory and linkage audit"
GROUPS = ["G1", "G2", "G3", "G4"]
CSV_TEXT_ENCODINGS = ("utf-8-sig", "cp1252")


class SurveyLinkageError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResolvedInput:
    role: str
    path: Path
    required: bool
    resolution: str
    match_count: int
    warning: str = ""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def norm_col(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def is_blank(value: Any) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip() == ""


def nonblank_mask(values: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    if isinstance(values, pd.DataFrame):
        stripped_blank = values.astype("string").apply(lambda column: column.str.strip().eq(""))
        return ~(values.isna() | stripped_blank)
    return ~(values.isna() | values.astype("string").str.strip().eq(""))


def find_column(columns: Iterable[Any], candidates: Iterable[str], *, role: str) -> str:
    lookup = {norm_col(col): str(col) for col in columns}
    hits: list[str] = []
    for candidate in candidates:
        key = norm_col(candidate)
        if key in lookup:
            hits.append(lookup[key])
    hits = list(dict.fromkeys(hits))
    if len(hits) != 1:
        raise SurveyLinkageError(
            f"Unable to identify exactly one {role} column. Candidates={list(candidates)}; "
            f"hits={hits}; available={list(map(str, columns))[:80]}"
        )
    return hits[0]


def normalise_group(value: Any) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)) and 1 <= int(value) <= 4:
        return f"G{int(value)}"
    if isinstance(value, (float, np.floating)) and float(value).is_integer() and 1 <= int(value) <= 4:
        return f"G{int(value)}"
    text = str(value).strip().upper()
    match = re.search(r"(?:GROUP\s*|G)?([1-4])$", text)
    if match:
        return f"G{match.group(1)}"
    if text in GROUPS:
        return text
    return None


def normalise_lcl_any_id(value: Any) -> tuple[str | None, str]:
    if pd.isna(value):
        return None, "blank"
    text = str(value).strip().upper()
    if not text:
        return None, "blank"
    if re.fullmatch(r"N\d{4}", text):
        return text, "group_n"
    if re.fullmatch(r"D\d{4}", text):
        return text, "group_d_out_of_scope"
    return None, "invalid_format"


def normalise_lcl_formal_id(value: Any) -> tuple[str | None, str]:
    entity_id, status = normalise_lcl_any_id(value)
    if status == "group_n":
        return entity_id, "valid"
    return None, status


def normalise_cer_id(value: Any) -> tuple[int | None, str]:
    if pd.isna(value):
        return None, "blank"
    text = str(value).strip()
    if not text:
        return None, "blank"
    if re.fullmatch(r"[+-]?\d+\.0+", text):
        text = text.split(".", 1)[0]
    if not re.fullmatch(r"[+-]?\d+", text):
        return None, "invalid_format"
    number = int(text)
    if number <= 0:
        return None, "invalid_nonpositive"
    return number, "valid"


def read_csv_with_governed_encoding(path: Path) -> pd.DataFrame:
    failures: list[str] = []
    for encoding in CSV_TEXT_ENCODINGS:
        try:
            frame = pd.read_csv(
                path,
                dtype=object,
                low_memory=False,
                encoding=encoding,
                encoding_errors="strict",
            )
        except UnicodeDecodeError as exc:
            failures.append(
                f"{encoding}: byte={exc.object[exc.start:exc.end].hex()} position={exc.start}"
            )
            continue
        frame.attrs["source_format"] = "csv"
        frame.attrs["source_encoding"] = encoding
        frame.attrs["source_path"] = str(path)
        return frame
    raise SurveyLinkageError(
        f"CSV encoding could not be resolved for {path}. "
        f"Governed attempts={list(CSV_TEXT_ENCODINGS)}; failures={' | '.join(failures)}"
    )


def read_tabular(path: Path, *, sheet_name: str | int | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return read_csv_with_governed_encoding(path)
    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
        frame.attrs.update(source_format="parquet", source_encoding="binary", source_path=str(path))
        return frame
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        frame = pd.read_excel(path, sheet_name=sheet_name, dtype=object)
        frame.attrs.update(source_format="excel", source_encoding="binary", source_path=str(path))
        return frame
    raise SurveyLinkageError(f"Unsupported tabular input: {path}")


def search_matches(root: Path, search_roots: list[str], patterns: list[str]) -> list[Path]:
    matches: set[Path] = set()
    for rel_root in search_roots:
        base = root / rel_root
        if not base.exists():
            continue
        for pattern in patterns:
            for candidate in base.rglob(pattern):
                if candidate.is_file():
                    matches.add(candidate.resolve())
    return sorted(matches)


def resolve_inputs(config: dict[str, Any], root: Path) -> dict[str, ResolvedInput]:
    resolved: dict[str, ResolvedInput] = {}
    failures: list[str] = []
    for role, spec in config["formal_inputs"].items():
        required = bool(spec.get("required", False))
        if spec.get("exact_relative_path"):
            path = (root / spec["exact_relative_path"]).resolve()
            if path.exists():
                resolved[role] = ResolvedInput(role, path, required, "exact", 1)
            else:
                message = f"{role}: missing exact path {path}"
                if required:
                    failures.append(message)
                else:
                    resolved[role] = ResolvedInput(role, path, required, "missing_optional", 0, message)
            continue

        matches = search_matches(root, spec.get("search_roots", []), spec.get("basename_patterns", []))
        if len(matches) == 1:
            warning = ""
            if re.search(r"\(\d+\)(?=\.[^.]+$)", matches[0].name):
                warning = "filename_contains_interface_duplicate_suffix"
            resolved[role] = ResolvedInput(role, matches[0], required, "unique_search_match", 1, warning)
        elif len(matches) == 0:
            message = (
                f"{role}: no match under roots={spec.get('search_roots', [])} "
                f"for patterns={spec.get('basename_patterns', [])}"
            )
            if required:
                failures.append(message)
            else:
                resolved[role] = ResolvedInput(role, Path(""), required, "missing_optional", 0, message)
        else:
            message = f"{role}: multiple matches ({len(matches)}): " + " | ".join(map(str, matches))
            if required:
                failures.append(message)
            else:
                resolved[role] = ResolvedInput(role, matches[0], required, "ambiguous_optional", len(matches), message)

    lock_path = Path(config["canonical_lock"])
    if not lock_path.exists():
        failures.append(f"canonical_lock: missing {lock_path}")
    else:
        resolved["canonical_lock"] = ResolvedInput("canonical_lock", lock_path.resolve(), True, "exact", 1)

    if failures:
        raise SurveyLinkageError("Input resolution failed:\n- " + "\n- ".join(failures))
    return resolved


def input_inventory(resolved: dict[str, ResolvedInput]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for role, item in resolved.items():
        exists = item.resolution != "missing_optional" and bool(item.path) and item.path.is_file()
        rows.append(
            {
                "role": role,
                "required": item.required,
                "resolved_path": str(item.path) if item.path else "",
                "resolution": item.resolution,
                "match_count": item.match_count,
                "exists": exists,
                "size_bytes": item.path.stat().st_size if exists else np.nan,
                "suffix": item.path.suffix.lower() if exists else "",
                "sha256": sha256_file(item.path) if exists else "",
                "warning": item.warning,
            }
        )
    return pd.DataFrame(rows).sort_values("role").reset_index(drop=True)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for column in display.columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else str(value).replace("|", "\\|").replace("\n", " ")
        )
    headers = [str(column).replace("|", "\\|") for column in display.columns]
    rows = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for values in display.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(map(str, values)) + " |")
    return "\n".join(rows)


def prepare_formal_assignments(
    cohort_path: Path,
    assignment_path: Path,
    *,
    dataset: str,
    expected_n: int,
    expected_groups: dict[str, int],
) -> pd.DataFrame:
    assignments = read_tabular(assignment_path)
    id_candidates = ["household_id", "entity_id", "Household_id"] if dataset == "LCL" else ["meter_id", "ID", "entity_id"]
    id_col = find_column(assignments.columns, id_candidates, role=f"{dataset} assignment ID")
    group_col = find_column(assignments.columns, ["assigned_group", "group", "profile_group", "cluster"], role=f"{dataset} group")

    normalised = assignments[id_col].map(normalise_lcl_formal_id if dataset == "LCL" else normalise_cer_id)
    assignments = assignments.copy()
    assignments["entity_id"] = [item[0] for item in normalised]
    assignments["id_status"] = [item[1] for item in normalised]
    assignments["profile_group"] = assignments[group_col].map(normalise_group)

    if assignments["entity_id"].isna().any():
        raise SurveyLinkageError(f"{dataset} assignments contain invalid IDs")
    if assignments["profile_group"].isna().any():
        raise SurveyLinkageError(f"{dataset} assignments contain invalid group labels")
    if assignments["entity_id"].duplicated().any():
        raise SurveyLinkageError(f"{dataset} assignments contain duplicate entity IDs")
    if len(assignments) != expected_n:
        raise SurveyLinkageError(f"{dataset} assignment count {len(assignments)} != expected {expected_n}")
    actual_counts = assignments["profile_group"].value_counts().sort_index().to_dict()
    if actual_counts != expected_groups:
        raise SurveyLinkageError(f"{dataset} group counts {actual_counts} != expected {expected_groups}")

    if dataset == "LCL":
        cohort = read_tabular(cohort_path)
        cohort_id_col = find_column(cohort.columns, ["household_id", "entity_id", "Household_id"], role="LCL cohort ID")
        cohort_norm = cohort[cohort_id_col].map(normalise_lcl_formal_id)
        cohort_ids = pd.Series([item[0] for item in cohort_norm], dtype=object)
        statuses = pd.Series([item[1] for item in cohort_norm], dtype=object)
        if (statuses != "valid").any() or cohort_ids.isna().any():
            raise SurveyLinkageError("LCL cohort contains invalid household IDs")
        if cohort_ids.duplicated().any():
            raise SurveyLinkageError("LCL cohort contains duplicate household IDs")
        if len(cohort_ids) != expected_n:
            raise SurveyLinkageError(f"LCL cohort count {len(cohort_ids)} != expected {expected_n}")
        if set(cohort_ids) != set(assignments["entity_id"]):
            raise SurveyLinkageError("LCL cohort and source assignments do not contain the same 3,843 IDs")

    return assignments[["entity_id", "profile_group"]].sort_values("entity_id").reset_index(drop=True)


def excel_sheet_inventory(path: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    book = pd.ExcelFile(path)
    frames: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for sheet in book.sheet_names:
        frame = pd.read_excel(path, sheet_name=sheet, dtype=object)
        frames[sheet] = frame
        id_hits = [str(col) for col in frame.columns if norm_col(col) in {"id", "meterid", "meter_id", "code"}]
        rows.append(
            {
                "file": str(path),
                "sheet": sheet,
                "rows": len(frame),
                "columns": len(frame.columns),
                "id_like_columns": "|".join(id_hits),
                "first_columns": "|".join(map(str, list(frame.columns)[:20])),
                "text_encoding": "binary",
            }
        )
    return pd.DataFrame(rows), frames


def select_cer_data_sheet(path: Path) -> tuple[str, pd.DataFrame, str, pd.DataFrame]:
    if path.suffix.lower() == ".csv":
        frame = read_tabular(path)
        id_columns = [str(col) for col in frame.columns if norm_col(col) in {"id", "meterid", "meter_id"}]
        if not id_columns:
            raise SurveyLinkageError(f"No ID/meter_id column found in CER pre-trial CSV {path}")
        candidates: list[tuple[int, str]] = []
        for id_col in id_columns:
            valid_count = sum(normalise_cer_id(value)[1] == "valid" for value in frame[id_col])
            candidates.append((valid_count, id_col))
        candidates.sort(reverse=True)
        if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
            raise SurveyLinkageError(f"CER pre-trial CSV has ambiguous ID columns: {candidates}")
        inventory = pd.DataFrame(
            [{
                "file": str(path),
                "sheet": "CSV_SINGLE_TABLE",
                "rows": len(frame),
                "columns": len(frame.columns),
                "id_like_columns": "|".join(id_columns),
                "first_columns": "|".join(map(str, list(frame.columns)[:20])),
                "text_encoding": frame.attrs.get("source_encoding", ""),
            }]
        )
        return "CSV_SINGLE_TABLE", frame.copy(), candidates[0][1], inventory

    inventory, frames = excel_sheet_inventory(path)
    candidates: list[tuple[int, int, str, str, pd.DataFrame]] = []
    for sheet, frame in frames.items():
        for id_col in [str(col) for col in frame.columns if norm_col(col) in {"id", "meterid", "meter_id"}]:
            valid_count = sum(normalise_cer_id(value)[1] == "valid" for value in frame[id_col])
            candidates.append((valid_count, len(frame), sheet, id_col, frame))
    if not candidates:
        raise SurveyLinkageError(f"No ID/meter_id column found in CER pre-trial workbook {path}")
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if len(candidates) > 1 and candidates[0][:2] == candidates[1][:2]:
        raise SurveyLinkageError("CER pre-trial workbook has ambiguous data sheets")
    best = candidates[0]
    return best[2], best[4].copy(), best[3], inventory


def allocation_inventory(path: Path, formal_cer_ids: set[int]) -> tuple[pd.DataFrame, dict[str, Any]]:
    sheet_inventory, frames = excel_sheet_inventory(path)
    best_linked = 0
    best_sheet = ""
    best_col = ""
    for sheet, frame in frames.items():
        for col in frame.columns:
            if norm_col(col) not in {"id", "meterid", "meter_id"}:
                continue
            ids = {item[0] for item in frame[col].map(normalise_cer_id) if item[1] == "valid"}
            linked = len(formal_cer_ids & ids)
            if linked > best_linked:
                best_linked = linked
                best_sheet = sheet
                best_col = str(col)
    return sheet_inventory, {
        "allocation_best_sheet": best_sheet,
        "allocation_best_id_column": best_col,
        "allocation_best_linked_formal_n": best_linked,
        "allocation_all_929_found": best_linked == len(formal_cer_ids),
    }


def lcl_question_metadata(question_meta: pd.DataFrame) -> tuple[dict[str, dict[str, str]], list[str]]:
    qid_col = find_column(question_meta.columns, ["Question_id", "question_id", "QuestionID"], role="LCL question ID")
    wording_col = find_column(question_meta.columns, ["Question", "question", "wording", "Question_text"], role="LCL question wording")
    survey_col = find_column(question_meta.columns, ["Survey", "survey"], role="LCL survey section")
    meta: dict[str, dict[str, str]] = {}
    attitude_qids: list[str] = []
    appliance_n = 0
    for _, row in question_meta.iterrows():
        qid = str(row[qid_col]).strip()
        section = "" if pd.isna(row[survey_col]) else str(row[survey_col]).strip().lower()
        wording = "" if pd.isna(row[wording_col]) else str(row[wording_col]).strip()
        meta[qid] = {"survey_section": section, "question_wording": wording}
        if section == "attitudes":
            attitude_qids.append(qid)
        elif section == "appliance":
            appliance_n += 1
    if appliance_n != 134 or len(attitude_qids) != 210:
        raise SurveyLinkageError(
            f"Unexpected LCL question sections: appliance={appliance_n} attitudes={len(attitude_qids)}"
        )
    return meta, attitude_qids


def lcl_answer_metadata_map(meta: dict[str, dict[str, str]]) -> tuple[list[dict[str, str]], list[str]]:
    """Map public answer columns to their actual appliance-question meanings.

    Direct inspection of the supplied files shows that survey_answers column positions
    are not a direct Question_id join to survey_questions:
      * metadata Q211 (Format) is stored at answer Q344;
      * answer Q211-Q230 correspond to metadata Q212-Q231;
      * answer Q231 is the property-ownership field removed from the public question map
        because of the documented validity concern;
      * answer Q232 corresponds to metadata Q232;
      * metadata Q233 (online newspapers) is absent from survey_answers;
      * answer Q233-Q343 correspond to metadata Q234-Q344.

    This governed positional map prevents silent semantic mislabelling.
    """
    rows: list[dict[str, str]] = []
    content_cols: list[str] = []
    for data_n in range(211, 345):
        data_qid = f"Q{data_n}"
        if 211 <= data_n <= 230:
            meta_qid = f"Q{data_n + 1}"
            scope = "in_scope_appliance"
        elif data_n == 231:
            meta_qid = ""
            scope = "excluded_property_ownership_validity_concern"
        elif data_n == 232:
            meta_qid = "Q232"
            scope = "in_scope_appliance"
        elif 233 <= data_n <= 343:
            meta_qid = f"Q{data_n + 1}"
            scope = "in_scope_appliance"
        else:  # Q344
            meta_qid = "Q211"
            scope = "survey_administration_format"

        if meta_qid:
            if meta_qid not in meta:
                raise SurveyLinkageError(f"LCL positional map references missing metadata {meta_qid}")
            wording = meta[meta_qid]["question_wording"]
            section = meta[meta_qid]["survey_section"]
        else:
            wording = "Property ownership field excluded because of documented validity concern"
            section = "appliance"
        rows.append(
            {
                "source_column": data_qid,
                "metadata_question_id": meta_qid,
                "question_wording": wording,
                "survey_section": section,
                "scope_status": scope,
            }
        )
        if scope == "in_scope_appliance":
            content_cols.append(data_qid)

    rows.append(
        {
            "source_column": "",
            "metadata_question_id": "Q233",
            "question_wording": meta["Q233"]["question_wording"],
            "survey_section": meta["Q233"]["survey_section"],
            "scope_status": "missing_from_public_answers",
        }
    )
    if len(content_cols) != 132:
        raise SurveyLinkageError(f"Unexpected governed LCL content-column count: {len(content_cols)}")
    return rows, content_cols


def coalesce_rows(group: pd.DataFrame, response_cols: list[str]) -> tuple[pd.Series | None, dict[str, Any]]:
    conflict_columns: list[str] = []
    for col in response_cols:
        values = [str(value).strip() for value in group[col] if not is_blank(value)]
        if len(set(values)) > 1:
            conflict_columns.append(col)
    row_nonblank = [int(nonblank_mask(group.iloc[i][response_cols]).sum()) for i in range(len(group))]
    if conflict_columns:
        return None, {
            "resolution_type": "unresolved_value_conflict",
            "conflict_columns": "|".join(conflict_columns),
            "row_nonblank_counts_json": json.dumps(row_nonblank),
        }

    merged = group.iloc[0].copy()
    for col in response_cols:
        values = [value for value in group[col] if not is_blank(value)]
        merged[col] = values[0] if values else np.nan

    if len(group) == 1:
        resolution_type = "unique"
    else:
        signatures = group[response_cols].astype("string").fillna("").apply(lambda s: s.str.strip()).agg("\x1f".join, axis=1)
        if signatures.nunique() == 1:
            resolution_type = "exact_duplicate_collapsed"
        elif any(count == 0 for count in row_nonblank) and max(row_nonblank) > 0:
            resolution_type = "empty_duplicate_collapsed"
        else:
            resolution_type = "complementary_duplicate_merged"
    return merged, {
        "resolution_type": resolution_type,
        "conflict_columns": "",
        "row_nonblank_counts_json": json.dumps(row_nonblank),
    }


def resolve_lcl_survey(
    frame: pd.DataFrame,
    raw_id_col: str,
    appliance_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = frame.copy()
    norm = work[raw_id_col].map(normalise_lcl_any_id)
    work["_normalised_id"] = [item[0] for item in norm]
    work["_id_status"] = [item[1] for item in norm]

    actual_appliance_cols = [col for col in appliance_cols if col in work.columns]
    if len(actual_appliance_cols) != len(appliance_cols):
        missing = sorted(set(appliance_cols) - set(actual_appliance_cols))
        raise SurveyLinkageError(f"LCL survey_answers is missing appliance columns: {missing[:30]}")

    invalid = work[work["_id_status"].isin(["blank", "invalid_format"])][[raw_id_col, "_id_status"]].copy()
    invalid.columns = ["raw_id", "id_status"]
    invalid.insert(0, "dataset", "LCL")

    group_d = work[work["_id_status"] == "group_d_out_of_scope"].copy()
    group_d_summary = pd.DataFrame([
        {
            "dataset": "LCL",
            "scope_status": "out_of_scope_group_d",
            "row_n": len(group_d),
            "unique_id_n": int(group_d["_normalised_id"].nunique()),
            "duplicate_id_n": int((group_d["_normalised_id"].value_counts() > 1).sum()),
            "rows_with_any_appliance_response_n": int(
                (nonblank_mask(group_d[actual_appliance_cols]).sum(axis=1) > 0).sum()
            ),
        }
    ])

    group_n = work[work["_id_status"] == "group_n"].copy()
    resolved_rows: list[pd.Series] = []
    audit_rows: list[dict[str, Any]] = []
    for entity_id, group in group_n.groupby("_normalised_id", sort=True):
        merged, info = coalesce_rows(group, actual_appliance_cols)
        resolved_for_analysis = merged is not None
        if merged is not None:
            merged["_normalised_id"] = entity_id
            nonblank_n = int(nonblank_mask(merged[actual_appliance_cols]).sum())
            merged["_in_scope_nonblank_n"] = nonblank_n
            merged["_response_status"] = "usable_response" if nonblank_n > 0 else "id_only_no_response"
            merged["_resolution_type"] = info["resolution_type"]
            resolved_rows.append(merged)
        audit_rows.append(
            {
                "dataset": "LCL",
                "entity_id": entity_id,
                "row_count": len(group),
                "resolution_type": info["resolution_type"],
                "conflict_columns": info["conflict_columns"],
                "row_nonblank_counts_json": info["row_nonblank_counts_json"],
                "resolved_for_analysis": resolved_for_analysis,
            }
        )

    resolved = pd.DataFrame(resolved_rows)
    if not resolved.empty and resolved["_normalised_id"].duplicated().any():
        raise SurveyLinkageError("LCL duplicate resolution did not produce one row per resolved Group N ID")
    audit = pd.DataFrame(audit_rows)
    return resolved, audit, invalid, group_d_summary


def resolve_cer_survey(
    frame: pd.DataFrame,
    raw_id_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = frame.copy()
    norm = work[raw_id_col].map(normalise_cer_id)
    work["_normalised_id"] = [item[0] for item in norm]
    work["_id_status"] = [item[1] for item in norm]
    invalid = work[work["_id_status"] != "valid"][[raw_id_col, "_id_status"]].copy()
    invalid.columns = ["raw_id", "id_status"]
    invalid.insert(0, "dataset", "CER")
    valid = work[work["_id_status"] == "valid"].copy()
    response_cols = [col for col in valid.columns if col not in {raw_id_col, "_normalised_id", "_id_status"}]
    resolved_rows: list[pd.Series] = []
    audit_rows: list[dict[str, Any]] = []
    for entity_id, group in valid.groupby("_normalised_id", sort=True):
        merged, info = coalesce_rows(group, response_cols)
        if merged is not None:
            merged["_normalised_id"] = entity_id
            nonblank_n = int(nonblank_mask(merged[response_cols]).sum())
            merged["_in_scope_nonblank_n"] = nonblank_n
            merged["_response_status"] = "usable_response" if nonblank_n > 0 else "id_only_no_response"
            merged["_resolution_type"] = info["resolution_type"]
            resolved_rows.append(merged)
        audit_rows.append(
            {
                "dataset": "CER",
                "entity_id": entity_id,
                "row_count": len(group),
                "resolution_type": info["resolution_type"],
                "conflict_columns": info["conflict_columns"],
                "row_nonblank_counts_json": info["row_nonblank_counts_json"],
                "resolved_for_analysis": merged is not None,
            }
        )
    resolved = pd.DataFrame(resolved_rows)
    if not resolved.empty and resolved["_normalised_id"].duplicated().any():
        raise SurveyLinkageError("CER duplicate resolution did not produce one row per resolved ID")
    return resolved, pd.DataFrame(audit_rows), invalid


def linkage_audit(
    formal: pd.DataFrame,
    resolved: pd.DataFrame,
    duplicate_audit: pd.DataFrame,
    invalid_rows: pd.DataFrame,
    *,
    dataset: str,
    out_of_scope_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    usable_ids = set(resolved.loc[resolved["_response_status"] == "usable_response", "_normalised_id"])
    id_only_ids = set(resolved.loc[resolved["_response_status"] == "id_only_no_response", "_normalised_id"])
    conflict_ids = set(
        duplicate_audit.loc[duplicate_audit["resolution_type"] == "unresolved_value_conflict", "entity_id"]
    )
    survey_row_ids = usable_ids | id_only_ids | conflict_ids
    formal_ids = set(formal["entity_id"])
    survey_only_usable = usable_ids - formal_ids
    survey_only_id_only = id_only_ids - formal_ids
    out_scope_rows = 0
    out_scope_unique = 0
    if out_of_scope_summary is not None and not out_of_scope_summary.empty:
        out_scope_rows = int(out_of_scope_summary.iloc[0]["row_n"])
        out_scope_unique = int(out_of_scope_summary.iloc[0]["unique_id_n"])

    rows: list[dict[str, Any]] = []
    for group in ["ALL"] + GROUPS:
        subgroup = formal if group == "ALL" else formal[formal["profile_group"] == group]
        group_ids = set(subgroup["entity_id"])
        linked_usable = group_ids & usable_ids
        linked_id_only = group_ids & id_only_ids
        linked_conflict = group_ids & conflict_ids
        linked_row_present = group_ids & survey_row_ids
        no_row = group_ids - survey_row_ids
        rows.append(
            {
                "dataset": dataset,
                "profile_group": group,
                "formal_population_n": len(group_ids),
                "linked_row_present_n": len(linked_row_present),
                "linked_usable_response_n": len(linked_usable),
                "linked_id_only_no_response_n": len(linked_id_only),
                "linked_unresolved_conflict_n": len(linked_conflict),
                "no_survey_row_n": len(no_row),
                "row_presence_rate": len(linked_row_present) / len(group_ids) if group_ids else np.nan,
                "usable_response_rate": len(linked_usable) / len(group_ids) if group_ids else np.nan,
                "linked_resolved_n": len(linked_usable),
                "unlinked_n": len(no_row),
                "resolved_linkage_rate": len(linked_usable) / len(group_ids) if group_ids else np.nan,
                "survey_only_usable_ids_n": len(survey_only_usable) if group == "ALL" else np.nan,
                "survey_only_id_only_ids_n": len(survey_only_id_only) if group == "ALL" else np.nan,
                "invalid_or_blank_survey_rows_n": len(invalid_rows) if group == "ALL" else np.nan,
                "out_of_scope_group_d_rows_n": out_scope_rows if group == "ALL" and dataset == "LCL" else np.nan,
                "out_of_scope_group_d_unique_ids_n": out_scope_unique if group == "ALL" and dataset == "LCL" else np.nan,
            }
        )
    return pd.DataFrame(rows)


def attach_groups(usable_survey: pd.DataFrame, formal: pd.DataFrame) -> pd.DataFrame:
    usable = usable_survey[usable_survey["_response_status"] == "usable_response"].copy()
    return usable.merge(
        formal.rename(columns={"entity_id": "_normalised_id"}),
        on="_normalised_id",
        how="inner",
        validate="one_to_one",
    )


def lcl_domain(metadata_question_id: str) -> tuple[str, str]:
    match = re.fullmatch(r"Q(\d+)", metadata_question_id.upper())
    if not match:
        return "unclassified", "not_planned"
    number = int(match.group(1))
    if 211 <= number <= 213:
        return "survey_administration", "not_planned"
    if 214 <= number <= 231:
        return "household_composition", "main_candidate"
    if 232 <= number <= 235:
        return "occupancy_context", "supporting"
    if 236 <= number <= 240:
        return "dwelling_scale", "main_candidate"
    if 241 <= number <= 249:
        return "thermal_efficiency_heating", "main_candidate"
    if 250 <= number <= 279:
        return "lighting", "supporting"
    if 280 <= number <= 327:
        return "appliances", "main_candidate"
    if 328 <= number <= 332:
        return "smart_meter_use", "not_planned"
    if 333 <= number <= 344:
        return "attitudes_billing_context", "not_planned"
    return "unclassified", "not_planned"


def lcl_eligibility_mask(
    frame: pd.DataFrame,
    metadata_question_id: str,
) -> tuple[pd.Series | None, str]:
    match = re.fullmatch(r"Q(\d+)", metadata_question_id.upper())
    if not match:
        return None, "none"
    number = int(match.group(1))
    # Metadata Q214 (household size) is answer column Q213.
    if 215 <= number <= 222 or 223 <= number <= 230:
        index = number - 214 if number <= 222 else number - 222
        size = pd.to_numeric(frame.get("Q213"), errors="coerce")
        return size.ge(index), "household_size"
    # Metadata Q305 (number of televisions) is answer column Q304.
    if 316 <= number <= 321 or 322 <= number <= 327:
        index = number - 315 if number <= 321 else number - 321
        tv_count = pd.to_numeric(frame.get("Q304"), errors="coerce")
        return tv_count.ge(index), "television_count"
    return None, "none"


def basic_text_class(value: Any) -> str:
    if is_blank(value):
        return "blank"
    low = re.sub(r"\s+", " ", str(value).strip().lower())
    if "refus" in low or low in {"r", "ref"}:
        return "refused"
    if "don't know" in low or "dont know" in low or low in {"dk", "d/k", "unknown"}:
        return "dont_know"
    if low in {"not applicable", "n/a", "not app", "na - not applicable"}:
        return "not_applicable"
    return "valid_observed"


def top_values_json(series: pd.Series, max_items: int = 12) -> str:
    values = series[nonblank_mask(series)].astype(str).str.strip()
    counts = values.value_counts().head(max_items)
    return json.dumps({str(k): int(v) for k, v in counts.items()}, ensure_ascii=False, sort_keys=True)


def lcl_question_inventory(
    linked_usable: pd.DataFrame,
    mapping_rows: list[dict[str, str]],
    attitude_qids: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    mapping_audit = pd.DataFrame(mapping_rows)

    for item in mapping_rows:
        source_column = item["source_column"]
        meta_qid = item["metadata_question_id"]
        scope = item["scope_status"]
        domain, role = lcl_domain(meta_qid) if meta_qid else ("excluded_invalid_field", "not_planned")
        if source_column and source_column in linked_usable.columns:
            series = linked_usable[source_column]
        else:
            series = pd.Series(np.nan, index=linked_usable.index, dtype=object)
        eligibility, rule = lcl_eligibility_mask(linked_usable, meta_qid) if meta_qid else (None, "none")
        observed = nonblank_mask(series)
        if eligibility is None:
            structural = pd.Series(False, index=series.index)
            unexplained_blank = ~observed
            observed_ineligible = pd.Series(False, index=series.index)
        else:
            structural = (~eligibility) & (~observed)
            unexplained_blank = eligibility & (~observed)
            observed_ineligible = (~eligibility) & observed
        classes = series.map(basic_text_class)
        valid = classes == "valid_observed"
        counts = series[valid].astype(str).str.strip().value_counts()
        is_in_scope = scope == "in_scope_appliance"
        rows.append(
            {
                "dataset": "LCL",
                "question_id": source_column or f"MISSING_{meta_qid}",
                "source_column": source_column,
                "metadata_question_id": meta_qid,
                "question_wording": item["question_wording"],
                "survey_section": item["survey_section"],
                "domain": domain,
                "analysis_role": role,
                "scope_status": scope,
                "rows_in_usable_linked": len(series) if is_in_scope else 0,
                "valid_observed_n": int(valid.sum()) if is_in_scope else 0,
                "raw_blank_n": int((classes == "blank").sum()) if is_in_scope else 0,
                "structurally_not_asked_n": int(structural.sum()) if is_in_scope else 0,
                "unexplained_blank_n": int(unexplained_blank.sum()) if is_in_scope else 0,
                "refused_n": int((classes == "refused").sum()) if is_in_scope else 0,
                "dont_know_n": int((classes == "dont_know").sum()) if is_in_scope else 0,
                "not_applicable_n": int((classes == "not_applicable").sum()) if is_in_scope else 0,
                "observed_when_ineligible_n": int(observed_ineligible.sum()) if is_in_scope else 0,
                "routing_rule": rule,
                "unique_valid_values": int(series[valid].astype(str).str.strip().nunique()) if is_in_scope else 0,
                "smallest_valid_category_n": int(counts.min()) if not counts.empty and is_in_scope else np.nan,
                "categories_under_n10": int((counts < 10).sum()) if not counts.empty and is_in_scope else 0,
                "top_raw_values_json": top_values_json(series) if is_in_scope else "{}",
            }
        )
        if is_in_scope:
            for group in ["ALL"] + GROUPS:
                subset = linked_usable if group == "ALL" else linked_usable[linked_usable["profile_group"] == group]
                subseries = subset[source_column]
                subelig, _ = lcl_eligibility_mask(subset, meta_qid)
                subobs = nonblank_mask(subseries)
                if subelig is None:
                    substruct = pd.Series(False, index=subseries.index)
                    subunexplained = ~subobs
                else:
                    substruct = (~subelig) & (~subobs)
                    subunexplained = subelig & (~subobs)
                missing_rows.append(
                    {
                        "dataset": "LCL",
                        "profile_group": group,
                        "question_id": source_column,
                        "source_column": source_column,
                        "metadata_question_id": meta_qid,
                        "group_n_usable_linked": len(subset),
                        "valid_or_raw_observed_n": int(subobs.sum()),
                        "structurally_not_asked_n": int(substruct.sum()),
                        "unexplained_blank_n": int(subunexplained.sum()),
                    }
                )

    for qid in attitude_qids:
        rows.append(
            {
                "dataset": "LCL",
                "question_id": qid,
                "source_column": qid,
                "metadata_question_id": qid,
                "question_wording": "Group D attitudes survey field",
                "survey_section": "attitudes",
                "domain": "group_d_attitudes",
                "analysis_role": "not_planned",
                "scope_status": "excluded_group_d_attitudes",
                "rows_in_usable_linked": 0,
                "valid_observed_n": 0,
                "raw_blank_n": 0,
                "structurally_not_asked_n": 0,
                "unexplained_blank_n": 0,
                "refused_n": 0,
                "dont_know_n": 0,
                "not_applicable_n": 0,
                "observed_when_ineligible_n": 0,
                "routing_rule": "out_of_scope",
                "unique_valid_values": 0,
                "smallest_valid_category_n": np.nan,
                "categories_under_n10": 0,
                "top_raw_values_json": "{}",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(missing_rows), mapping_audit


def cer_question_code(column: str) -> str:
    match = re.match(r"\s*Question\s+(\d+)\s*:", str(column), flags=re.IGNORECASE)
    return f"Q{match.group(1)}" if match else "UNPARSED"


def cer_domain_and_role(code: str) -> tuple[str, str]:
    main_map = {
        "Q410": "household_composition", "Q420": "household_composition", "Q43111": "household_composition",
        "Q430": "daytime_presence", "Q4312": "daytime_presence",
        "Q470": "electric_thermal_load", "Q4701": "electric_thermal_load", "Q4801": "electric_thermal_load", "Q4704": "electric_thermal_load",
        "Q450": "dwelling_scale", "Q453": "dwelling_scale", "Q4531": "dwelling_scale", "Q6103": "dwelling_scale", "Q61031": "dwelling_scale", "Q460": "dwelling_scale",
        "Q455": "thermal_efficiency", "Q4551": "thermal_efficiency", "Q4906": "thermal_efficiency", "Q4907": "thermal_efficiency", "Q4908": "thermal_efficiency", "Q4909": "thermal_efficiency",
        "Q310": "economic_position", "Q401": "economic_position", "Q402": "economic_position", "Q4021": "economic_position", "Q403": "economic_position", "Q404": "economic_position", "Q5418": "economic_position", "Q452": "economic_position",
        "Q471": "energy_vulnerability", "Q472": "energy_vulnerability", "Q473": "energy_vulnerability", "Q474": "energy_vulnerability",
    }
    supporting_map = {
        "Q200": "demographic_context", "Q300": "demographic_context", "Q405": "internet_context", "Q406": "internet_context",
    }
    if code in main_map:
        return main_map[code], "main_candidate"
    if code in supporting_map:
        return supporting_map[code], "supporting"
    if code.startswith("Q490"):
        return "appliances", "supporting"
    return "trial_attitudes_or_other", "not_planned"


def cer_columns_by_code(frame: pd.DataFrame, id_col: str) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    helper = {id_col, "_normalised_id", "_id_status", "_in_scope_nonblank_n", "_response_status", "_resolution_type", "profile_group"}
    for col in frame.columns:
        if col in helper:
            continue
        mapping.setdefault(cer_question_code(str(col)), []).append(str(col))
    return mapping


def select_cer_parent_column(mapping: dict[str, list[str]], code: str, contains: str | None = None) -> str:
    cols = mapping.get(code, [])
    if contains is not None:
        cols = [col for col in cols if contains.lower() in col.lower()]
    if len(cols) != 1:
        raise SurveyLinkageError(f"Unable to identify one CER parent column for {code} contains={contains!r}: {cols}")
    return cols[0]


def build_cer_route_masks(
    frame: pd.DataFrame,
    id_col: str,
    rules: list[dict[str, Any]],
) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    mapping = cer_columns_by_code(frame, id_col)
    masks: dict[str, pd.Series] = {}
    audit_rows: list[dict[str, Any]] = []
    for rule in rules:
        targets: list[str] = []
        for target_code in rule["target_codes"]:
            targets.extend(mapping.get(target_code, []))
        if not targets:
            raise SurveyLinkageError(f"CER routing rule {rule['rule_id']} found no target columns")
        parent = select_cer_parent_column(mapping, rule["parent_code"], rule.get("parent_column_contains"))
        parent_text = frame[parent].astype("string").str.strip()
        operator = rule["operator"]
        if operator == "equals":
            eligible = parent_text.eq(str(rule["value"]))
        elif operator == "in":
            eligible = parent_text.isin([str(value) for value in rule["values"]])
        elif operator == "numeric_lt":
            eligible = pd.to_numeric(parent_text, errors="coerce").lt(float(rule["value"]))
        else:
            raise SurveyLinkageError(f"Unsupported routing operator: {operator}")
        eligible = eligible.fillna(False)
        for target in targets:
            if target in masks:
                raise SurveyLinkageError(f"Multiple CER routing rules target the same source column: {target}")
            masks[target] = eligible
            observed = nonblank_mask(frame[target])
            audit_rows.append(
                {
                    "rule_id": rule["rule_id"],
                    "question_code": cer_question_code(target),
                    "source_column": target,
                    "parent_question_code": rule["parent_code"],
                    "parent_source_column": parent,
                    "eligible_n": int(eligible.sum()),
                    "ineligible_n": int((~eligible).sum()),
                    "observed_when_eligible_n": int((eligible & observed).sum()),
                    "blank_when_eligible_n": int((eligible & ~observed).sum()),
                    "observed_when_ineligible_n": int((~eligible & observed).sum()),
                    "blank_when_ineligible_n": int((~eligible & ~observed).sum()),
                    "route_consistency_status": "PASS" if int((eligible & ~observed).sum()) == 0 and int((~eligible & observed).sum()) == 0 else "REVIEW",
                }
            )
    return masks, pd.DataFrame(audit_rows)


def cer_special_class(code: str, value: Any, special_rules: dict[str, dict[str, str]]) -> str:
    text = str(value).strip()
    semantic = special_rules.get(code, {}).get(text)
    return semantic if semantic is not None else "valid_observed"


def cer_question_inventory(
    linked_usable: pd.DataFrame,
    id_col: str,
    route_rules: list[dict[str, Any]],
    special_rules: dict[str, dict[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    route_masks, route_audit = build_cer_route_masks(linked_usable, id_col, route_rules)
    mapping = cer_columns_by_code(linked_usable, id_col)
    rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for code, columns in mapping.items():
        for source_column in columns:
            series = linked_usable[source_column]
            observed = nonblank_mask(series)
            eligibility = route_masks.get(source_column)
            if eligibility is None:
                structural = pd.Series(False, index=series.index)
                unexplained_blank = ~observed
                observed_ineligible = pd.Series(False, index=series.index)
                routing_rule = "none_or_not_required"
            else:
                structural = (~eligibility) & (~observed)
                unexplained_blank = eligibility & (~observed)
                observed_ineligible = (~eligibility) & observed
                routing_rule = route_audit.loc[route_audit["source_column"] == source_column, "rule_id"].iloc[0]
            semantic_classes = pd.Series("blank", index=series.index, dtype=object)
            semantic_classes.loc[observed] = [
                cer_special_class(code, value, special_rules) for value in series.loc[observed]
            ]
            valid = semantic_classes == "valid_observed"
            counts = series[valid].astype(str).str.strip().value_counts()
            domain, role = cer_domain_and_role(code)
            rows.append(
                {
                    "dataset": "CER",
                    "question_id": source_column,
                    "question_code": code,
                    "source_column": source_column,
                    "question_wording": source_column.split(":", 1)[1].strip() if ":" in source_column else source_column,
                    "domain": domain,
                    "analysis_role": role,
                    "scope_status": "in_scope_pretrial",
                    "rows_in_usable_linked": len(series),
                    "valid_observed_n": int(valid.sum()),
                    "raw_blank_n": int((~observed).sum()),
                    "structurally_not_asked_n": int(structural.sum()),
                    "unexplained_blank_n": int(unexplained_blank.sum()),
                    "refused_n": int((semantic_classes == "refused").sum()),
                    "dont_know_n": int((semantic_classes == "dont_know").sum()),
                    "special_missing_n": int(semantic_classes.astype(str).str.startswith("special_missing").sum()),
                    "observed_when_ineligible_n": int(observed_ineligible.sum()),
                    "routing_rule": routing_rule,
                    "unique_valid_values": int(series[valid].astype(str).str.strip().nunique()),
                    "smallest_valid_category_n": int(counts.min()) if not counts.empty else np.nan,
                    "categories_under_n10": int((counts < 10).sum()) if not counts.empty else 0,
                    "top_raw_values_json": top_values_json(series),
                }
            )
            for group in ["ALL"] + GROUPS:
                subset = linked_usable if group == "ALL" else linked_usable[linked_usable["profile_group"] == group]
                subseries = subset[source_column]
                subobserved = nonblank_mask(subseries)
                if source_column in route_masks:
                    subeligibility = route_masks[source_column].loc[subset.index]
                    substructural = (~subeligibility) & (~subobserved)
                    subunexplained = subeligibility & (~subobserved)
                else:
                    substructural = pd.Series(False, index=subseries.index)
                    subunexplained = ~subobserved
                missing_rows.append(
                    {
                        "dataset": "CER",
                        "profile_group": group,
                        "question_id": source_column,
                        "source_column": source_column,
                        "group_n_usable_linked": len(subset),
                        "valid_or_raw_observed_n": int(subobserved.sum()),
                        "structurally_not_asked_n": int(substructural.sum()),
                        "unexplained_blank_n": int(subunexplained.sum()),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(missing_rows), route_audit


def semantic_exception_audit(cer_linked: pd.DataFrame, id_col: str) -> pd.DataFrame:
    mapping = cer_columns_by_code(cer_linked, id_col)
    rows: list[dict[str, Any]] = [
        {
            "issue_id": "LCL_ANSWER_METADATA_POSITIONAL_MAP",
            "question_codes": "Q211-Q344",
            "observed_evidence": (
                "survey_answers columns are position-coded rather than a direct Question_id join: "
                "answer Q211-Q230 map to metadata Q212-Q231; answer Q231 is the excluded "
                "property-ownership field; answer Q232 maps to metadata Q232; metadata Q233 is absent; "
                "answer Q233-Q343 map to metadata Q234-Q344; answer Q344 is metadata Q211 Format"
            ),
            "status": "RESOLVED_BY_GOVERNED_POSITIONAL_MAP",
            "survey_linkage_effect": "Prevents semantic mislabelling; property ownership and Format are excluded from substantive inventory",
            "survey_recoding_rule": "Use source_column plus metadata_question_id from the mapping audit; never join LCL answers to metadata by identical Question_id alone",
        }
    ]
    q402 = select_cer_parent_column(mapping, "Q402")
    q4021 = select_cer_parent_column(mapping, "Q4021")
    q403 = select_cer_parent_column(mapping, "Q403")
    q404 = select_cer_parent_column(mapping, "Q404")
    q402_values = sorted(cer_linked.loc[nonblank_mask(cer_linked[q402]), q402].astype(str).str.strip().unique())
    q4021_values = sorted(cer_linked.loc[nonblank_mask(cer_linked[q4021]), q4021].astype(str).str.strip().unique())
    mutually_exclusive = bool((nonblank_mask(cer_linked[q402]) & nonblank_mask(cer_linked[q4021])).sum() == 0)
    rows.append(
        {
            "issue_id": "CER_INCOME_PUBLIC_RELEASE_SEMANTICS",
            "question_codes": "Q402|Q4021|Q403|Q404",
            "observed_evidence": (
                f"Q402 values={q402_values}; Q4021 values={q4021_values}; "
                f"Q402/Q4021 mutually exclusive={mutually_exclusive}; "
                f"Q403 nonblank={int(nonblank_mask(cer_linked[q403]).sum())}; "
                f"Q404 nonblank={int(nonblank_mask(cer_linked[q404]).sum())}"
            ),
            "status": "HOLD_INCOME_RECODING_UNTIL_SURVEY_RECODING_CODEBOOK_DECISION",
            "survey_linkage_effect": "Does not invalidate linkage or inventory; income is not recoded in Survey linkage",
            "survey_recoding_rule": "Use social class or tenure fallback unless released-data income code semantics are independently confirmed",
        }
    )
    return pd.DataFrame(rows)


def comparability_map() -> pd.DataFrame:
    rows = [
        ("household_size", "Q213", "Q214", "Q410|Q420|Q43111", "partial", "Different instruments and age thresholds; compare broad size bands only"),
        ("bedroom_count", "Q239", "Q240", "Q460", "direct_broad", "Both record bedroom count; top categories require harmonised broad band"),
        ("dwelling_type", "Q235|Q236|Q237", "Q236|Q237|Q238", "Q450", "partial", "LCL broad house/flat/mobile indicators versus more detailed CER dwelling type"),
        ("dwelling_scale", "Q238|Q239", "Q239|Q240", "Q6103|Q61031|Q460", "partial", "Rooms/bedrooms versus floor area/bedrooms; no exact instrument equivalence"),
        ("double_glazing", "Q240", "Q241", "Q4906", "partial", "LCL presence indicator versus CER proportion of windows"),
        ("roof_or_attic_insulation", "Q241", "Q242", "Q4908", "partial", "Broad presence versus timing categories"),
        ("wall_insulation", "Q242", "Q243", "Q4909", "partial", "Broad binary/unknown comparison"),
        ("heating_and_hot_water", "Q246|Q247|Q248", "Q247|Q248|Q249", "Q470|Q4701|Q4801", "partial", "Different categorical structures; only broad electric-thermal indicators are defensible"),
        ("selected_appliances", "Q293|Q295|Q297|Q298|Q300|Q301|Q303", "Q294|Q296|Q298|Q299|Q301|Q302|Q304", "Q4704|Q49002|Q490004|Q4900004|Q4900005|Q4900006", "partial", "Counts and usage questions differ; compare ownership/end-use presence only"),
        ("daytime_presence", "Q234", "Q235", "Q430|Q4312", "dataset_specific", "Work-from-home is not equivalent to numbers present during the day"),
        ("economic_position", "", "", "Q401|Q402|Q4021|Q452|Q5418", "cer_only", "No defensible LCL equivalent in the supplied Group N appliance survey"),
        ("energy_vulnerability", "", "", "Q471|Q472|Q473|Q474", "cer_only", "No defensible LCL equivalent"),
        ("ber_rating", "", "", "Q455|Q4551", "cer_only", "No direct LCL BER measure"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "domain", "lcl_source_columns", "lcl_metadata_questions", "cer_questions",
            "comparability_class", "decision_note"
        ],
    )


def build_decision_note(
    config: dict[str, Any],
    lcl_linkage: pd.DataFrame,
    cer_linkage: pd.DataFrame,
    route_audit: pd.DataFrame,
    duplicates: pd.DataFrame,
    semantic_exceptions: pd.DataFrame,
    allocation_summary: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    lcl_all = lcl_linkage[lcl_linkage["profile_group"] == "ALL"].iloc[0]
    cer_all = cer_linkage[cer_linkage["profile_group"] == "ALL"].iloc[0]
    unresolved_conflicts = int((duplicates["resolution_type"] == "unresolved_value_conflict").sum())
    route_reviews = int((route_audit["route_consistency_status"] != "PASS").sum())
    income_hold = int((semantic_exceptions["status"].str.startswith("HOLD")).sum())
    gate_reasons: list[str] = []
    if unresolved_conflicts:
        gate_reasons.append(f"{unresolved_conflicts} unresolved value-conflict survey IDs remain")
    if route_reviews:
        gate_reasons.append(f"{route_reviews} governed CER routing checks require review")
    if not allocation_summary.get("allocation_all_929_found", False):
        gate_reasons.append("CER allocation workbook did not independently locate all 929 formal meters")
    analysis_gate_status = "PASS_TO_SURVEY_RECODING_SPECIFICATION" if not gate_reasons else "HOLD_FOR_MANUAL_SEMANTIC_REVIEW"

    status = {
        "analysis": ANALYSIS_NAME,
        "analysis_id": config["analysis_id"],
        "created_at_utc": utc_now(),
        "computational_status": "COMPLETE_PASS",
        "analysis_gate_status": analysis_gate_status,
        "gate_reasons": gate_reasons,
        "lcl_formal_n": int(lcl_all["formal_population_n"]),
        "lcl_linked_row_present_n": int(lcl_all["linked_row_present_n"]),
        "lcl_linked_usable_response_n": int(lcl_all["linked_usable_response_n"]),
        "lcl_linked_id_only_no_response_n": int(lcl_all["linked_id_only_no_response_n"]),
        "lcl_no_survey_row_n": int(lcl_all["no_survey_row_n"]),
        "lcl_usable_response_rate": float(lcl_all["usable_response_rate"]),
        "cer_formal_n": int(cer_all["formal_population_n"]),
        "cer_linked_row_present_n": int(cer_all["linked_row_present_n"]),
        "cer_linked_usable_response_n": int(cer_all["linked_usable_response_n"]),
        "cer_linked_id_only_no_response_n": int(cer_all["linked_id_only_no_response_n"]),
        "cer_no_survey_row_n": int(cer_all["no_survey_row_n"]),
        "cer_usable_response_rate": float(cer_all["usable_response_rate"]),
        "cer_governed_routing_rules_n": int(len(route_audit)),
        "cer_routing_reviews_n": route_reviews,
        "income_semantics_hold_n": income_hold,
        "training_performed": False,
        "predictions_modified": False,
        "forecasting_inputs_modified": False,
        "clustering_modified": False,
        "regression_performed": False,
        "outcome_dependent_variable_selection": False,
    }
    gate_lines = "\n".join(f"- {reason}" for reason in gate_reasons) if gate_reasons else "- None"
    note = f"""# Survey linkage Survey Inventory and Linkage Decision

## Status

```text
computational_status = COMPLETE_PASS
analysis_gate_status = {analysis_gate_status}
```

Survey linkage distinguishes three different situations that must not be conflated:

1. a formal household/meter has a usable survey response;
2. a formal household has an ID row but no in-scope answers;
3. a formal household/meter has no row in the survey file.

Within-survey blanks are separately classified as structural routing, explicit special codes, or unexplained blanks. Forecasting outcomes were generated without survey inputs; all survey work remains post-hoc, descriptive and non-causal.

## Formal linkage and response availability

### LCL Group N

{markdown_table(lcl_linkage)}

### CER residential control meters

{markdown_table(cer_linkage)}

## Duplicate and scope decisions

- Group D (`Ddddd`) rows are out of scope for the Group N source cohort and are not invalid IDs.
- N0217 and any analogous duplicate with one complete row plus an empty row are collapsed by non-conflicting coalescence.
- A duplicate is held only when two nonblank values for the same field disagree.
- ID-only Group N rows are counted as row-present but not as usable survey responses.
- LCL answer columns are not joined to `survey_questions.csv` by identical Question_id. The governed positional map is recorded in `survey_linkage_lcl_answer_metadata_mapping_audit.csv`; Q231 property ownership is excluded, Q233 online-newspaper metadata is absent from the answer table, and Q344 is the Format field.

## CER routing audit

- Governed routed source columns checked: {len(route_audit):,}
- Routing checks requiring review: {route_reviews:,}
- Allocation workbook formal IDs located: {allocation_summary.get('allocation_best_linked_formal_n', 0):,} / {config['expected']['cer_formal_meters']:,}

## Semantic exception retained for Survey recoding

The released CER income columns do not reproduce the questionnaire's raw numeric-income routing literally: Q402 and Q4021 are mutually exclusive released fields containing codes rather than a raw annual amount. Survey linkage records this as a semantic exception and does not recode income. Survey recoding must either confirm the released-data coding independently or use the predeclared social-class/tenure fallback. This exception does not invalidate Survey linkage linkage or routing closure.

## Gate reasons

{gate_lines}

## Output boundary

A PASS permits specification of Survey recoding governed recoding and harmonisation only. No regression, pairwise-gain testing, significance screening, category collapsing, imputation or outcome-dependent selection is performed here.
"""
    return note, status


def write_output_inventory(output_paths: list[Path], path: Path) -> None:
    pd.DataFrame([
        {"path": str(output), "size_bytes": output.stat().st_size, "sha256": sha256_file(output)}
        for output in sorted(output_paths)
    ]).to_csv(path, index=False)


def compute_analysis(config: dict[str, Any], resolved: dict[str, ResolvedInput]) -> dict[str, Any]:
    expected = config["expected"]
    inputs = input_inventory(resolved)
    lcl_formal = prepare_formal_assignments(
        resolved["lcl_cohort"].path,
        resolved["lcl_assignments"].path,
        dataset="LCL",
        expected_n=int(expected["lcl_formal_households"]),
        expected_groups={k: int(v) for k, v in expected["lcl_group_counts"].items()},
    )
    cer_formal = prepare_formal_assignments(
        resolved["cer_assignments"].path,
        resolved["cer_assignments"].path,
        dataset="CER",
        expected_n=int(expected["cer_formal_meters"]),
        expected_groups={k: int(v) for k, v in expected["cer_group_counts"].items()},
    )

    lcl_meta = read_tabular(resolved["lcl_survey_questions"].path)
    lcl_meta_lookup, attitude_qids = lcl_question_metadata(lcl_meta)
    lcl_mapping_rows, lcl_content_cols = lcl_answer_metadata_map(lcl_meta_lookup)
    lcl_answers = read_tabular(resolved["lcl_survey_answers"].path)
    lcl_id_col = find_column(lcl_answers.columns, ["Household_id", "household_id", "entity_id"], role="LCL survey household ID")
    lcl_resolved, lcl_duplicates, lcl_invalid, group_d_summary = resolve_lcl_survey(
        lcl_answers, lcl_id_col, lcl_content_cols
    )
    lcl_linked_usable = attach_groups(lcl_resolved, lcl_formal)

    cer_sheet, cer_answers, cer_id_col, cer_sheet_inventory = select_cer_data_sheet(
        resolved["cer_pretrial_survey_data"].path
    )
    cer_resolved, cer_duplicates, cer_invalid = resolve_cer_survey(cer_answers, cer_id_col)
    cer_linked_usable = attach_groups(cer_resolved, cer_formal)

    lcl_linkage = linkage_audit(
        lcl_formal, lcl_resolved, lcl_duplicates, lcl_invalid,
        dataset="LCL", out_of_scope_summary=group_d_summary,
    )
    cer_linkage = linkage_audit(
        cer_formal, cer_resolved, cer_duplicates, cer_invalid,
        dataset="CER",
    )

    allocation_sheet_inventory, allocation_summary = allocation_inventory(
        resolved["cer_allocation_data"].path, set(cer_formal["entity_id"])
    )
    allocation_sheet_inventory.insert(0, "workbook_role", "cer_allocation_data")
    cer_sheet_inventory.insert(0, "workbook_role", "cer_pretrial_survey_data")
    excel_inventory = pd.concat([cer_sheet_inventory, allocation_sheet_inventory], ignore_index=True)

    lcl_questions, lcl_missing, lcl_mapping_audit = lcl_question_inventory(
        lcl_linked_usable, lcl_mapping_rows, attitude_qids
    )
    cer_questions, cer_missing, route_audit = cer_question_inventory(
        cer_linked_usable,
        cer_id_col,
        config["cer_main_routing_rules"],
        config["cer_special_code_rules"],
    )
    missingness = pd.concat([lcl_missing, cer_missing], ignore_index=True)
    semantic_exceptions = semantic_exception_audit(cer_linked_usable, cer_id_col)
    comparison = comparability_map()
    duplicates = pd.concat([lcl_duplicates, cer_duplicates], ignore_index=True)
    invalids = pd.concat([lcl_invalid, cer_invalid], ignore_index=True)

    note, status = build_decision_note(
        config, lcl_linkage, cer_linkage, route_audit, duplicates,
        semantic_exceptions, allocation_summary,
    )
    status.update(allocation_summary)
    status["cer_pretrial_sheet"] = cer_sheet
    status["cer_pretrial_id_column"] = cer_id_col
    status["lcl_survey_questions_encoding"] = lcl_meta.attrs.get("source_encoding", "unknown")
    status["lcl_survey_answers_encoding"] = lcl_answers.attrs.get("source_encoding", "unknown")
    status["cer_pretrial_survey_encoding"] = cer_answers.attrs.get("source_encoding", "unknown")
    status["lcl_raw_group_n_unique_ids_n"] = int(lcl_resolved["_normalised_id"].nunique())
    status["lcl_raw_group_n_usable_response_ids_n"] = int((lcl_resolved["_response_status"] == "usable_response").sum())
    status["lcl_raw_group_n_id_only_ids_n"] = int((lcl_resolved["_response_status"] == "id_only_no_response").sum())
    status["lcl_answer_metadata_mapping_status"] = "PASS_GOVERNED_POSITIONAL_MAP"
    status["lcl_valid_content_columns_n"] = 132
    status["lcl_excluded_property_source_column"] = "Q231"
    status["lcl_missing_public_answer_metadata_question"] = "Q233"
    status["lcl_format_source_column"] = "Q344"
    status["cer_raw_survey_unique_ids_n"] = int(cer_resolved["_normalised_id"].nunique())
    status["cer_raw_id_only_ids_n"] = int((cer_resolved["_response_status"] == "id_only_no_response").sum())

    return {
        "input_inventory": inputs,
        "lcl_formal": lcl_formal,
        "cer_formal": cer_formal,
        "lcl_answers": lcl_answers,
        "cer_answers": cer_answers,
        "lcl_linkage": lcl_linkage,
        "cer_linkage": cer_linkage,
        "lcl_questions": lcl_questions,
        "cer_questions": cer_questions,
        "missingness": missingness,
        "duplicates": duplicates,
        "invalids": invalids,
        "group_d_summary": group_d_summary,
        "lcl_mapping_audit": lcl_mapping_audit,
        "route_audit": route_audit,
        "semantic_exceptions": semantic_exceptions,
        "comparability": comparison,
        "excel_inventory": excel_inventory,
        "allocation_summary": allocation_summary,
        "cer_sheet": cer_sheet,
        "cer_id_col": cer_id_col,
        "note": note,
        "status": status,
    }


def check_inputs(config_path: Path, root: Path) -> None:
    config = load_json(config_path)
    resolved = resolve_inputs(config, root)
    print(f"TASK: {ANALYSIS_NAME}")
    print(f"ROOT: {root}")
    print(f"PYTHON: {sys.executable}")
    print(f"PYTHON VERSION: {sys.version.split()[0]}")
    print(f"PANDAS VERSION: {pd.__version__}")
    print(f"NUMPY VERSION: {np.__version__}")
    try:
        import openpyxl  # type: ignore
        print(f"OPENPYXL VERSION: {openpyxl.__version__}")
    except Exception as exc:
        raise SurveyLinkageError(f"Required Excel reader openpyxl is unavailable: {exc}") from exc
    parquet_required = any(
        item.path.suffix.lower() in {".parquet", ".pq"}
        for item in resolved.values()
        if item.path and item.path.is_file()
    )
    if parquet_required:
        try:
            import pyarrow  # type: ignore
            print(f"PYARROW VERSION: {pyarrow.__version__}")
        except Exception as exc:
            raise SurveyLinkageError(f"Required Parquet reader pyarrow is unavailable: {exc}") from exc
    else:
        print("PYARROW VERSION: not required by resolved non-Parquet inputs")

    inventory = input_inventory(resolved)
    print(inventory[["role", "required", "resolved_path", "resolution", "suffix", "warning"]].to_string(index=False))
    analysis = compute_analysis(config, resolved)
    status = analysis["status"]
    allocation = analysis["allocation_summary"]
    print(f"CER PRE-TRIAL DATA SHEET: {analysis['cer_sheet']}")
    print(f"CER PRE-TRIAL ID COLUMN: {analysis['cer_id_col']}")
    print(f"CER ALLOCATION BEST SHEET: {allocation.get('allocation_best_sheet', '')}")
    print(f"CER ALLOCATION BEST ID COLUMN: {allocation.get('allocation_best_id_column', '')}")
    print(f"CER ALLOCATION FORMAL IDS FOUND: {allocation.get('allocation_best_linked_formal_n', 0)} / {config['expected']['cer_formal_meters']}")
    print(f"LCL FORMAL IDS: {len(analysis['lcl_formal'])}")
    print(f"CER FORMAL IDS: {len(analysis['cer_formal'])}")
    print(f"LCL RAW GROUP N UNIQUE IDS: {status['lcl_raw_group_n_unique_ids_n']}")
    print(f"LCL RAW GROUP N USABLE RESPONSE IDS: {status['lcl_raw_group_n_usable_response_ids_n']}")
    print(f"LCL RAW GROUP N ID-ONLY IDS: {status['lcl_raw_group_n_id_only_ids_n']}")
    print(f"LCL ANSWER/METADATA MAPPING: {status['lcl_answer_metadata_mapping_status']}")
    print(f"LCL VALID CONTENT COLUMNS: {status['lcl_valid_content_columns_n']}")
    print(f"CER RAW SURVEY UNIQUE IDS: {status['cer_raw_survey_unique_ids_n']}")
    print(f"CER RAW ID-ONLY IDS: {status['cer_raw_id_only_ids_n']}")
    print(f"LCL FORMAL USABLE LINKED: {status['lcl_linked_usable_response_n']}")
    print(f"LCL FORMAL ID-ONLY LINKED: {status['lcl_linked_id_only_no_response_n']}")
    print(f"LCL FORMAL NO SURVEY ROW: {status['lcl_no_survey_row_n']}")
    print(f"CER FORMAL USABLE LINKED: {status['cer_linked_usable_response_n']}")
    print(f"CER FORMAL NO SURVEY ROW: {status['cer_no_survey_row_n']}")
    print(f"CER GOVERNED ROUTING SOURCE COLUMNS: {status['cer_governed_routing_rules_n']}")
    print(f"CER ROUTING REVIEWS: {status['cer_routing_reviews_n']}")
    print(f"LCL QUESTIONS CSV ENCODING: {status['lcl_survey_questions_encoding']}")
    print(f"LCL ANSWERS CSV ENCODING: {status['lcl_survey_answers_encoding']}")
    print(f"CER PRE-TRIAL CSV ENCODING: {status['cer_pretrial_survey_encoding']}")
    print(f"PROVISIONAL ANALYSIS GATE: {status['analysis_gate_status']}")
    print("SURVEY LINKAGE INPUT CHECK: PASS")


def run_formal(config_path: Path, root: Path) -> None:
    config = load_json(config_path)
    resolved = resolve_inputs(config, root)
    analysis = compute_analysis(config, resolved)
    outputs = config["output_roots"]
    table_root = root / outputs["tables"]
    metadata_root = root / outputs["metadata"]
    decision_path = root / outputs["decision_note"]
    table_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    def write_csv(df: pd.DataFrame, name: str) -> Path:
        path = table_root / name
        df.to_csv(path, index=False)
        output_paths.append(path)
        return path

    write_csv(analysis["lcl_linkage"], "survey_linkage_lcl_survey_linkage_audit.csv")
    write_csv(analysis["cer_linkage"], "survey_linkage_cer_survey_linkage_audit.csv")
    write_csv(analysis["lcl_questions"], "survey_linkage_lcl_question_inventory.csv")
    write_csv(analysis["cer_questions"], "survey_linkage_cer_question_inventory.csv")
    write_csv(analysis["missingness"], "survey_linkage_missingness_by_dataset_group.csv")
    write_csv(analysis["input_inventory"], "survey_linkage_input_file_inventory.csv")
    write_csv(analysis["duplicates"], "survey_linkage_duplicate_id_audit.csv")
    write_csv(analysis["invalids"], "survey_linkage_invalid_id_rows.csv")
    write_csv(analysis["group_d_summary"], "survey_linkage_lcl_out_of_scope_group_d_summary.csv")
    write_csv(analysis["lcl_mapping_audit"], "survey_linkage_lcl_answer_metadata_mapping_audit.csv")
    write_csv(analysis["route_audit"], "survey_linkage_cer_routing_audit.csv")
    write_csv(analysis["semantic_exceptions"], "survey_linkage_semantic_exception_audit.csv")
    write_csv(analysis["comparability"], "survey_linkage_lcl_cer_comparability_map.csv")
    write_csv(analysis["excel_inventory"], "survey_linkage_excel_sheet_inventory.csv")

    decision_path.write_text(analysis["note"], encoding="utf-8")
    output_paths.append(decision_path)
    status_path = metadata_root / "survey_linkage_survey_inventory_linkage_status.json"
    status_path.write_text(json.dumps(analysis["status"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output_paths.append(status_path)
    output_inventory_path = metadata_root / "survey_linkage_output_inventory.csv"
    write_output_inventory(output_paths, output_inventory_path)

    status = analysis["status"]
    print("SURVEY LINKAGE CLOSURE FORMAL RUN SUMMARY")
    print(f"LCL formal/usable: {status['lcl_formal_n']} / {status['lcl_linked_usable_response_n']}")
    print(f"LCL formal ID-only: {status['lcl_linked_id_only_no_response_n']}")
    print(f"LCL formal no survey row: {status['lcl_no_survey_row_n']}")
    print(f"CER formal/usable: {status['cer_formal_n']} / {status['cer_linked_usable_response_n']}")
    print(f"CER formal no survey row: {status['cer_no_survey_row_n']}")
    print(f"CER routing reviews: {status['cer_routing_reviews_n']}")
    print(f"ANALYSIS GATE: {status['analysis_gate_status']}")
    print("TRAINING PERFORMED: False")
    print("PREDICTIONS MODIFIED: False")
    print("REGRESSION PERFORMED: False")
    print("OUTCOME-DEPENDENT VARIABLE SELECTION: False")
    print("SURVEY LINKAGE COMPUTATION COMPLETE — PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=ANALYSIS_NAME)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--mode", choices=["check-inputs", "run"], required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    root = args.root if args.root is not None else Path(config["root"])
    try:
        if args.mode == "check-inputs":
            check_inputs(args.config, root)
        else:
            run_formal(args.config, root)
    except SurveyLinkageError as exc:
        print(f"SURVEY LINKAGE ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"SURVEY LINKAGE UNEXPECTED ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
