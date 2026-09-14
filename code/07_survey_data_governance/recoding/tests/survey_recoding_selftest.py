#!/usr/bin/env python3
from __future__ import annotations
import importlib.util, json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

RELEASE = Path(__file__).resolve().parents[1]
MODULE_PATH = RELEASE / "src/survey_recoding_survey_recoding_harmonisation.py"
CONFIG_PATH = RELEASE / "configs/survey_recoding_survey_recoding_harmonisation.json"
spec = importlib.util.spec_from_file_location("survey_recoding", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load Survey recoding module")
module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

assert module.normalise_lcl_id(" n0217 ") == ("N0217", "valid_group_n")
assert module.normalise_lcl_id("D0001")[1] == "out_of_scope_group_d"
assert module.normalise_cer_id("123.0") == (123, "valid")

frame = pd.DataFrame({"Household_id": ["N0217", "N0217"], "Q213": ["2", np.nan], "Q240": ["Yes", np.nan]})
merged, resolution, conflicts = module.coalesce_nonconflicting(frame, ["Q213", "Q240"])
assert merged is not None and resolution == "coalesced_nonconflicting" and conflicts == []

lcl_row = pd.Series({
    "entity_id": "N0001", "profile_group": "G1", "survey_response_status": "usable_response",
    "Q213": "3", "Q222": "45-54", "Q223": "35-44", "Q224": "05-11", "Q225": np.nan, "Q226": np.nan, "Q227": np.nan, "Q228": np.nan, "Q229": np.nan,
    "Q235": "Semi-detached", "Q236": np.nan, "Q237": np.nan, "Q238": "5", "Q239": "3", "Q240": "Yes", "Q241": "Yes", "Q242": "No",
    "Q246": "Gas;Electric (including storage heaters)", "Q248": "Hot water storage tank with electric immersion heater",
    "Q293": "1", "Q295": "1", "Q297": "1", "Q298": "0", "Q299": "0", "Q300": "1", "Q301": "1", "Q303": "2",
})
lcl = module.lcl_recode_row(lcl_row, config)
assert lcl["household_composition_common"] == "adults_with_children_under16"
assert lcl["electric_thermal_load"] == 1.0

columns = {
    "Q410": "Question 410: people", "Q420": "Question 420: adults", "Q430": "Question 430: daytime adults", "Q43111": "Question 43111: children", "Q4312": "Question 4312: daytime children",
    "Q450": "Question 450: home", "Q460": "Question 460: bedrooms", "Q6103": "Question 6103: floor area", "Q61031": "Question 61031: units", "Q455": "Question 455: BER", "Q4551": "Question 4551: rating",
    "Q4906": "Question 4906: glazing", "Q4908": "Question 4908: attic", "Q4909": "Question 4909: walls", "Q4801": "Question 4801: immersion use", "Q401": "Question 401: SOCIAL CLASS", "Q452": "Question 452: tenure", "Q471": "Question 471: warm", "Q473": "Question 473: heating money",
}
extra = [
    "Question 470: Electricity (electric central heating/storage heating)", "Question 470: Electricity (plug in heaters)",
    "Question 4701: Electric (immersion)", "Question 4701: Electric (instantaneous heater)",
    "Question 472: I cannot afford to have the home as warm as I would like",
    "Question 49002: Washing machine", "Question 49002: Tumble dryer", "Question 49002: Dishwasher", "Question 49002: Electric cooker", "Question 49002: Electric heater (plug-in convector heaters)", "Question 49002: Electric shower (instant)", "Question 49002: Electric shower (electric pumped from hot tank)",
]
row_dict = {
    "entity_id": 1001, "profile_group": "G2", "survey_response_status": "usable_response",
    columns["Q410"]: "3", columns["Q420"]: "2", columns["Q430"]: "1", columns["Q43111"]: "2", columns["Q4312"]: "8", columns["Q450"]: "2",
    columns["Q460"]: "5", columns["Q6103"]: "1200", columns["Q61031"]: "2", columns["Q455"]: "2", columns["Q4551"]: "",
    columns["Q4906"]: "4", columns["Q4908"]: "2", columns["Q4909"]: "3", columns["Q4801"]: "1", columns["Q401"]: "2", columns["Q452"]: "4", columns["Q471"]: "2", columns["Q473"]: "2",
    extra[0]: "1", extra[1]: "0", extra[2]: "1", extra[3]: "0", extra[4]: "1", extra[5]: "2", extra[6]: "1", extra[7]: "2", extra[8]: "2", extra[9]: "1", extra[10]: "2", extra[11]: "1",
}
code_map = {}
for c in list(columns.values()) + extra:
    m = re.match(r"^Question ([0-9]+):", c)
    if m: code_map.setdefault("Q" + m.group(1), []).append(c)
cer = module.cer_recode_row(pd.Series(row_dict), config, code_map)
assert cer["household_composition_common"] == "adults_with_children_under15"
assert cer["daytime_presence_category"] == "one"
assert cer["bedroom_category"] == "5+"
assert cer["double_glazing_category"] == "about_three_quarters"
assert cer["attic_insulation_category"] == "yes_more_than_5_years_ago"
assert cer["wall_insulation_category"] == "dont_know"
assert np.isnan(cer["reported_thermal_feature_count"])
assert cer["electric_thermal_load"] == 1.0
assert cer["energy_vulnerability"] == 1.0

row_known = dict(row_dict); row_known[columns["Q4909"]] = "2"
cer_known = module.cer_recode_row(pd.Series(row_known), config, code_map)
assert cer_known["reported_thermal_feature_count"] == 2.0
assert cer_known["thermal_efficiency_proxy_count"] == 2.0

live_row = dict(row_dict); live_row[columns["Q410"]] = "1"; live_row[columns["Q420"]] = ""; live_row[columns["Q430"]] = ""; live_row[columns["Q43111"]] = ""; live_row[columns["Q4312"]] = ""
live = module.cer_recode_row(pd.Series(live_row), config, code_map)
assert pd.isna(live["daytime_presence_category"])
assert live["daytime_presence_category__missing_reason"] == "structurally_not_asked_live_alone"

cer_frame = pd.DataFrame([
    {"survey_response_status": "usable_response", "floor_area_m2": np.nan, "floor_area_m2__missing_reason": "special_missing_not_provided",
     "bedroom_category": "3", "bedroom_category__missing_reason": "available", "ber_rating": np.nan, "ber_rating__missing_reason": "not_applicable_no_ber",
     "double_glazing_category": "all", "double_glazing_category__missing_reason": "available", "attic_insulation_category": "yes_within_5_years", "attic_insulation_category__missing_reason": "available",
     "wall_insulation_category": "yes", "wall_insulation_category__missing_reason": "available", "social_class": "C1", "social_class__missing_reason": "available", "tenure": "own_mortgage", "tenure__missing_reason": "available"}
    for _ in range(10)
])
selected, audit = module.assign_dataset_level_fallbacks(cer_frame, config)
assert audit.set_index("domain").loc["dwelling_scale", "selected_variable"] == "bedroom_category"
assert audit.set_index("domain").loc["reported_thermal_features", "selected_variable"] == "double_glazing_category|attic_insulation_category|wall_insulation_category"
assert audit.set_index("domain").loc["economic_position", "selected_variable"] == "social_class"
assert selected["dwelling_scale_main"].eq("3").all()
assert "thermal_efficiency_main" not in selected.columns

lock = module.core_analysis_variable_lock()
cer_main = lock[lock["dataset"].eq("CER") & lock["analysis_role"].eq("main")]
assert cer_main["domain"].nunique() == 6 and len(cer_main) == 8
assert "thermal_efficiency_main" not in set(lock["analysis_variable"])
print("SURVEY RECODING SELFTEST: PASS")
