#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the CER residential control household manifest.")
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    source = root / "data/raw/ISSDA_CER/SurveySourceData_survey_original/SME and Residential allocations.xlsx"
    output = root / "outputs/tables/cer_residential_control_candidate_ids.csv"

    if not source.exists():
        raise FileNotFoundError(source)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output}. Use --overwrite to replace it.")

    df = pd.read_excel(source, sheet_name="Sheet1", usecols="A:E")
    residential = df.loc[df["Code"] == 1].copy()
    tariff = residential["Residential - Tariff allocation"].astype(str).str.strip()
    stimulus = residential["Residential - stimulus allocation"].astype(str).str.strip()
    selected = residential.loc[(tariff == "E") & (stimulus == "E"), [
        "ID", "Code", "Residential - Tariff allocation", "Residential - stimulus allocation"
    ]].copy()

    if selected["ID"].duplicated().any():
        raise AssertionError("Duplicate CER household IDs found in the selected cohort.")
    if len(selected) != 929:
        raise AssertionError(f"Expected 929 CER control households, found {len(selected)}.")

    output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output, index=False)
    print(f"CER control cohort: PASS ({len(selected)} households)")
    print(output)


if __name__ == "__main__":
    main()
