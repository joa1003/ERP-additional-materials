#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve()
SOURCE = HERE.parents[1] / "src" / "survey_linkage_survey_inventory_linkage.py"
CONFIG = HERE.parents[1] / "configs" / "survey_linkage_survey_inventory_linkage.json"
spec = importlib.util.spec_from_file_location("survey_linkage_module", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load Survey linkage source")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def synthetic_lcl_question_meta() -> pd.DataFrame:
    rows = []
    for i in range(1, 211):
        rows.append({"Question_id": f"Q{i:02d}", "Survey": "attitudes", "Question": f"Attitude {i}"})
    for i in range(211, 345):
        wording = f"Appliance {i}"
        if i == 211:
            wording = "Format"
        elif i == 212:
            wording = "Smart Meter"
        elif i == 213:
            wording = "Energy Decision Maker"
        elif i == 214:
            wording = "Household Size"
        elif i == 231:
            wording = "Relationship to Others in household"
        elif i == 232:
            wording = "Newspapers - Printed"
        elif i == 233:
            wording = "Newspapers - Online"
        elif i == 234:
            wording = "Newspapers - None read regularly"
        elif i == 344:
            wording = "Likelihood to recommend"
        rows.append({"Question_id": f"Q{i}", "Survey": "appliance", "Question": wording})
    return pd.DataFrame(rows)


def main() -> None:
    require(module.normalise_lcl_any_id(" n0001 ") == ("N0001", "group_n"), "LCL Group N normalisation failed")
    require(module.normalise_lcl_any_id(" d0001 ") == ("D0001", "group_d_out_of_scope"), "LCL Group D scope failed")
    require(module.normalise_lcl_formal_id("D0001")[0] is None, "Group D must not enter formal Group N IDs")
    require(module.normalise_cer_id("1002.0") == (1002, "valid"), "CER Excel artefact normalisation failed")
    require(module.normalise_group("Group 4") == "G4", "group normalisation failed")

    meta_frame = synthetic_lcl_question_meta()
    meta, attitudes = module.lcl_question_metadata(meta_frame)
    mapping, content = module.lcl_answer_metadata_map(meta)
    mapping_df = pd.DataFrame(mapping)
    require(len(attitudes) == 210, "LCL attitude count failed")
    require(len(content) == 132, "LCL valid content-column count failed")
    require(mapping_df.loc[mapping_df["source_column"] == "Q211", "metadata_question_id"].iloc[0] == "Q212", "Q211 mapping failed")
    require(mapping_df.loc[mapping_df["source_column"] == "Q231", "scope_status"].iloc[0] == "excluded_property_ownership_validity_concern", "Q231 exclusion failed")
    require(mapping_df.loc[mapping_df["source_column"] == "Q233", "metadata_question_id"].iloc[0] == "Q234", "Q233 positional mapping failed")
    require(mapping_df.loc[mapping_df["source_column"] == "Q344", "metadata_question_id"].iloc[0] == "Q211", "Q344 Format mapping failed")
    require((mapping_df["scope_status"] == "missing_from_public_answers").sum() == 1, "missing metadata field audit failed")

    lcl_cols = {f"Q{i}": [None, None, None, None] for i in range(211, 345)}
    lcl = pd.DataFrame({"Household_id": ["N0217", "N0217", "N0002", "D0001"], **lcl_cols})
    lcl.loc[0, "Q211"] = "Yes"
    lcl.loc[0, "Q213"] = "2"
    lcl.loc[2, "Q211"] = None  # ID-only/no-response row
    lcl.loc[3, "Q211"] = "Yes"
    resolved, duplicate_audit, invalid, group_d = module.resolve_lcl_survey(lcl, "Household_id", content)
    require(len(invalid) == 0, "unexpected LCL invalid IDs")
    require(int(group_d.iloc[0]["row_n"]) == 1, "Group D out-of-scope count failed")
    n0217 = duplicate_audit.loc[duplicate_audit["entity_id"] == "N0217"].iloc[0]
    require(n0217["resolution_type"] == "empty_duplicate_collapsed", "N0217 empty duplicate resolution failed")
    require((resolved["_response_status"] == "usable_response").sum() == 1, "LCL usable response classification failed")
    require((resolved["_response_status"] == "id_only_no_response").sum() == 1, "LCL ID-only classification failed")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        csv_path = temp / "Smart meters Residential pre-trial survey data.csv"
        csv_path.write_bytes("ID,Question 410: Household,Comment\n1002,1,Worker’s home\n".encode("cp1252"))
        sheet, frame, id_col, inventory = module.select_cer_data_sheet(csv_path)
        require(sheet == "CSV_SINGLE_TABLE", "CER CSV source label failed")
        require(id_col == "ID", "CER CSV ID detection failed")
        require(frame.attrs.get("source_encoding") == "cp1252", "CER CP1252 detection failed")
        require(inventory.iloc[0]["text_encoding"] == "cp1252", "CER encoding inventory failed")

        xlsx_path = temp / "SME and Residential allocations.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            pd.DataFrame({"ID": [1002, 1003]}).to_excel(writer, sheet_name="Residential", index=False)
        _, allocation = module.allocation_inventory(xlsx_path, {1002, 1003})
        require(allocation["allocation_all_929_found"], "allocation reader failed")

    cer = pd.DataFrame(
        {
            "ID": [1, 2, 3],
            "Question 405: Internet": ["1", "2", "1"],
            "Question 406: Broadband": ["1", "", "2"],
            "Question 410: Household": ["1", "2", "3"],
            "Question 420: Adults": ["", "2", "2"],
            "Question 430: Adults daytime": ["", "1", "2"],
            "Question 43111: Children": ["", "", "1"],
            "Question 4312: Children daytime": ["", "", "8"],
            "Question 453: Build year": ["1990", "9999", "2000"],
            "Question 4531: Approx age": ["", "4", ""],
            "Question 6103: Floor area": ["100", "999999999", "200"],
            "Question 61031: Unit": ["1", "", "2"],
            "Question 4701: Water: Electric (immersion)": ["1", "0", "1"],
            "Question 4801: Immersion use": ["1", "", "2"],
            "Question 455: BER": ["2", "1", "3"],
            "Question 4551: BER rating": ["", "3", ""],
            "Question 471: Warm": ["1", "2", "1"],
            "Question 472: Reason: affordability": ["", "1", ""],
            "Question 473: No heat": ["2", "1", "2"],
            "Question 474: Consequence: cold day": ["", "1", ""],
            "Question 5414: Bill change": ["1", "2", "3"],
            "Question 5415: Increase amount": ["", "2", ""],
            "Question 54155: Decrease amount": ["", "", "3"],
            "Question 402: Income": ["1", None, "2"],
            "Question 4021: Income band": [None, "6", None],
            "Question 403: Unit": ["3", None, "3"],
            "Question 404: Tax": ["1", None, "1"],
        }
    )
    cer_resolved, cer_duplicates, cer_invalid = module.resolve_cer_survey(cer, "ID")
    require(len(cer_invalid) == 0 and len(cer_duplicates) == 3, "CER resolution failed")
    cer_resolved["profile_group"] = ["G1", "G2", "G3"]
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    questions, missing, route_audit = module.cer_question_inventory(
        cer_resolved,
        "ID",
        config["cer_main_routing_rules"],
        config["cer_special_code_rules"],
    )
    require((route_audit["route_consistency_status"] == "PASS").all(), "CER governed routing self-test failed")
    require(len(route_audit) == 13, "synthetic CER routed column count failed")
    require(not missing.duplicated(["dataset", "profile_group", "question_id"]).any(), "missingness key failed")
    exceptions = module.semantic_exception_audit(cer_resolved, "ID")
    require(set(exceptions["issue_id"]) == {"LCL_ANSWER_METADATA_POSITIONAL_MAP", "CER_INCOME_PUBLIC_RELEASE_SEMANTICS"}, "semantic exception audit failed")

    comp = module.comparability_map()
    require({"lcl_source_columns", "lcl_metadata_questions", "cer_questions"}.issubset(comp.columns), "comparability schema failed")
    require("cer_only" in set(comp["comparability_class"]), "comparability classes failed")

    print("SURVEY LINKAGE SELFTEST: PASS")


if __name__ == "__main__":
    main()
