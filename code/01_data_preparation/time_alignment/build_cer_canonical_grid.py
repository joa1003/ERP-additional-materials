#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DAY_ONE = pd.Timestamp("2009-01-01")
FULL_START = pd.Timestamp("2009-07-14")
FULL_END = pd.Timestamp("2010-12-31")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the canonical 48-slot CER grid and non-standard-slot audit.")
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    raw_dir = root / "data/raw/ISSDA_CER"
    manifest = root / "outputs/tables/cer_residential_control_candidate_ids.csv"
    output = root / "data/processed/time_aligned/cer_canonical_local_grid.parquet"
    excluded_output = root / "data/processed/time_aligned/cer_excluded_nonstandard_slots.parquet"

    for path in (manifest,):
        if not path.exists():
            raise FileNotFoundError(path)
    for path in (output, excluded_output):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {path}. Use --overwrite to replace it.")
        if path.exists() and args.overwrite:
            path.unlink()

    ids = pd.read_csv(manifest)
    id_col = next((c for c in ("meter_id", "entity_id", "ID", "id") if c in ids.columns), None)
    if id_col is None:
        raise KeyError("CER household ID column not found in cohort manifest.")
    households = pd.to_numeric(ids[id_col], errors="raise").astype("int64").drop_duplicates().sort_values().to_numpy()
    if len(households) != 929:
        raise AssertionError(f"Expected 929 CER control households, found {len(households)}.")

    raw_files = [raw_dir / f"File{i}.txt" for i in range(1, 7)]
    missing = [str(p) for p in raw_files if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing CER reading files:\n" + "\n".join(missing))

    start_code = int((FULL_START - DAY_ONE).days + 1)
    end_code = int((FULL_END - DAY_ONE).days + 1)
    n_days = end_code - start_code + 1
    n_households = len(households)
    position = {int(h): i for i, h in enumerate(households)}
    household_set = set(position)

    values = np.full((n_households, n_days, 48), np.nan, dtype=np.float32)
    seen = np.zeros((n_households, n_days, 48), dtype=bool)
    values_flat = values.reshape(-1)
    seen_flat = seen.reshape(-1)
    excluded_parts: list[pd.DataFrame] = []
    found: set[int] = set()

    for raw_path in raw_files:
        for chunk in pd.read_csv(
            raw_path,
            sep=r"\s+",
            header=None,
            names=["meter_id", "daytime_code", "kwh"],
            dtype={"meter_id": "int32", "daytime_code": "int32", "kwh": "float32"},
            chunksize=1_000_000,
        ):
            chunk = chunk.loc[chunk["meter_id"].isin(household_set)].copy()
            if chunk.empty:
                continue
            found.update(chunk["meter_id"].unique().tolist())
            chunk["day_code"] = (chunk["daytime_code"] // 100).astype("int32")
            chunk["raw_slot"] = (chunk["daytime_code"] % 100).astype("int16")
            in_period = chunk["day_code"].between(start_code, end_code)
            standard = chunk["raw_slot"].between(1, 48)

            extra = chunk.loc[in_period & ~standard, ["meter_id", "daytime_code", "kwh", "day_code", "raw_slot"]].copy()
            if not extra.empty:
                extra["local_date"] = DAY_ONE + pd.to_timedelta(extra["day_code"] - 1, unit="D")
                excluded_parts.append(extra)

            standard_chunk = chunk.loc[in_period & standard, ["meter_id", "kwh", "day_code", "raw_slot"]].copy()
            if standard_chunk.empty:
                continue
            if standard_chunk["kwh"].isna().any():
                raise ValueError(f"Raw NaN kWh values found in {raw_path.name}.")

            hi = standard_chunk["meter_id"].map(position).to_numpy(dtype=np.int64)
            di = standard_chunk["day_code"].to_numpy(dtype=np.int64) - start_code
            si = standard_chunk["raw_slot"].to_numpy(dtype=np.int64) - 1
            cell = ((hi * n_days + di) * 48 + si)
            unique, counts = np.unique(cell, return_counts=True)
            if np.any(counts > 1) or seen_flat[unique].any():
                raise RuntimeError("Duplicate CER household/date/slot records found.")
            seen_flat[cell] = True
            values_flat[cell] = standard_chunk["kwh"].to_numpy(dtype=np.float32)

    if found != household_set:
        raise AssertionError(f"CER raw files contain {len(found)} of 929 selected households.")

    if excluded_parts:
        excluded = pd.concat(excluded_parts, ignore_index=True)
        excluded = excluded.sort_values(["meter_id", "day_code", "raw_slot"]).reset_index(drop=True)
    else:
        excluded = pd.DataFrame(columns=["meter_id", "daytime_code", "kwh", "day_code", "raw_slot", "local_date"])
    excluded_output.parent.mkdir(parents=True, exist_ok=True)
    excluded.to_parquet(excluded_output, index=False, compression="snappy")

    dates = pd.date_range(FULL_START, FULL_END, freq="D")
    slots = np.arange(1, 49, dtype=np.int8)
    base_dates = np.repeat(dates.to_numpy(), 48)
    base_slots = np.tile(slots, n_days)
    base_day_codes = ((pd.DatetimeIndex(base_dates) - DAY_ONE).days + 1).astype(np.int16)
    base_codes = base_day_codes.astype(np.int32) * 100 + base_slots.astype(np.int32)
    base_times = pd.DatetimeIndex(base_dates) + pd.to_timedelta((base_slots.astype(np.int16) - 1) * 30, unit="m")
    structural = ((pd.DatetimeIndex(base_dates) == pd.Timestamp("2010-03-28")) & np.isin(base_slots, [2, 3]))

    writer = None
    rows_per_household = n_days * 48
    written = 0
    try:
        for start in range(0, n_households, 25):
            end = min(start + 25, n_households)
            block_households = households[start:end]
            count = len(block_households)
            present = seen[start:end].reshape(-1)
            missing_mask = ~present
            structural_block = np.tile(structural, count)
            frame = pd.DataFrame({
                "meter_id": np.repeat(block_households.astype(np.int32), rows_per_household),
                "daytime_code": np.tile(base_codes, count),
                "day_code": np.tile(base_day_codes, count),
                "local_date": np.tile(base_dates, count),
                "local_slot": np.tile(base_slots, count),
                "timestamp_local": np.tile(base_times.to_numpy(), count),
                "kwh": values[start:end].reshape(-1),
                "raw_record_present": present,
                "is_missing": missing_mask,
                "is_structural_absence": structural_block,
                "is_observational_missing": missing_mask & ~structural_block,
            })
            frame["local_day_of_week"] = frame["timestamp_local"].dt.dayofweek.astype(np.int8)
            frame["local_weekend"] = (frame["local_day_of_week"] >= 5).astype(np.int8)
            frame["local_month"] = frame["timestamp_local"].dt.month.astype(np.int8)
            frame["timezone_conversion_applied"] = False
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                output.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(output, table.schema, compression="snappy")
            writer.write_table(table)
            written += len(frame)
            del frame, table
            gc.collect()
    finally:
        if writer is not None:
            writer.close()

    expected = n_households * n_days * 48
    if written != expected:
        raise AssertionError(f"Expected {expected:,} canonical rows, wrote {written:,}.")
    if int(structural.sum()) != 2:
        raise AssertionError("Expected two structural standard-slot absences per household.")

    print(f"CER canonical grid: PASS ({written:,} rows)")
    print(f"Non-standard rows retained: {len(excluded):,}")
    print(output)
    print(excluded_output)


if __name__ == "__main__":
    main()
