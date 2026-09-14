#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the LCL GMT to Europe/London local-clock map.")
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    source = root / "data/raw/LCL/csv/data_collection/data_tables/consumption_n.csv"
    output = root / "data/processed/time_aligned/lcl_full_local_clock_map.parquet"

    if not source.exists():
        raise FileNotFoundError(source)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output}. Use --overwrite to replace it.")

    raw = pd.read_csv(source, usecols=["GMT"])
    gmt = pd.DatetimeIndex(pd.to_datetime(raw["GMT"], errors="raise"))
    if gmt.hasnans or gmt.duplicated().any() or not gmt.is_monotonic_increasing:
        raise AssertionError("LCL GMT timestamps are not a complete unique increasing sequence.")
    expected = pd.date_range(gmt.min(), gmt.max(), freq="30min")
    if not gmt.equals(expected):
        raise AssertionError("LCL GMT timestamps do not match a complete 30-minute grid.")

    london = gmt.tz_localize("UTC").tz_convert("Europe/London")
    wall = london.tz_localize(None)
    wall_series = pd.Series(wall)
    occurrence = wall_series.groupby(wall_series, sort=False).cumcount().add(1).astype("int8")
    repeated = wall_series.duplicated(keep=False)
    offsets = np.fromiter(
        (int(ts.utcoffset().total_seconds() // 60) for ts in london.to_pydatetime()),
        dtype=np.int16,
        count=len(london),
    )
    is_dst = np.fromiter(
        (bool(ts.dst() is not None and ts.dst() != timedelta(0)) for ts in london.to_pydatetime()),
        dtype=bool,
        count=len(london),
    )

    result = pd.DataFrame({
        "raw_row_index": np.arange(len(gmt), dtype=np.int32),
        "timestamp_gmt": gmt,
        "timestamp_london": london,
        "timestamp_london_wall": wall,
        "local_date": wall.normalize(),
        "local_slot": (wall.hour * 2 + wall.minute // 30 + 1).astype(np.int8),
        "local_hour": wall.hour.astype(np.int8),
        "local_minute": wall.minute.astype(np.int8),
        "local_day_of_week": wall.dayofweek.astype(np.int8),
        "local_weekend": (wall.dayofweek >= 5).astype(np.int8),
        "local_month": wall.month.astype(np.int8),
        "local_year": wall.year.astype(np.int16),
        "utc_offset_minutes": offsets,
        "is_dst": is_dst,
        "local_occurrence": occurrence.to_numpy(),
        "is_repeated_local_time": repeated.to_numpy(),
        "physical_sequence_continuous": True,
    })

    if not result["local_slot"].between(1, 48).all():
        raise AssertionError("Unexpected LCL local slot.")
    if not set(result["utc_offset_minutes"].unique()).issubset({0, 60}):
        raise AssertionError("Unexpected Europe/London UTC offset.")

    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False, compression="snappy")
    print(f"LCL local-clock map: PASS ({len(result):,} rows)")
    print(output)


if __name__ == "__main__":
    main()
