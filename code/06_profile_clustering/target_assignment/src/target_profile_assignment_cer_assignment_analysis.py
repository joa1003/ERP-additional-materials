from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from target_profile_assignment_cer_assignment_common import (
    compute_dtw_assignments,
    ensure,
    load_config,
    load_fixed_prototypes,
    local_time_label,
    profile_zscore,
    resolve_path,
    sha256_file,
    write_json,
)

ANALYSIS_ID = "target_profile_assignment"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build CER training-period profiles and assign all 929 meters to fixed Profile clustering K=4 source prototypes."
    )
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def profile_period_mask(
    frame: pd.DataFrame,
    start_date: pd.Timestamp,
    start_slot: int,
    end_date: pd.Timestamp,
    end_slot: int,
) -> np.ndarray:
    dates = pd.to_datetime(frame["local_date"], errors="raise").dt.normalize()
    slots = pd.to_numeric(frame["local_slot"], errors="raise").astype(np.int16)
    lower = (dates > start_date) | ((dates == start_date) & (slots >= start_slot))
    upper = (dates < end_date) | ((dates == end_date) & (slots <= end_slot))
    return (lower & upper).to_numpy(dtype=bool)


def build_receiving_profiles(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    input_path = resolve_path(config, config["inputs"]["cer_canonical_parquet"])
    ensure(input_path.exists(), f"CER canonical parquet not found: {input_path}")

    profile_cfg = config["profile"]
    start_date = pd.Timestamp(profile_cfg["start_date"]).normalize()
    end_date = pd.Timestamp(profile_cfg["end_date"]).normalize()
    start_slot = int(profile_cfg["start_slot"])
    end_slot = int(profile_cfg["end_slot"])
    excluded_dates = {pd.Timestamp(value).normalize() for value in profile_cfg["excluded_local_dates"]}
    batch_size = int(profile_cfg.get("arrow_batch_size", 500_000))

    dataset = ds.dataset(str(input_path), format="parquet")
    required_columns = ["meter_id", "local_date", "local_slot", "kwh"]
    missing = [column for column in required_columns if column not in dataset.schema.names]
    ensure(not missing, f"CER canonical parquet missing columns: {missing}")

    scanner = dataset.scanner(columns=required_columns, batch_size=batch_size, use_threads=False)
    partials: list[pd.DataFrame] = []
    rows_scanned = 0
    rows_in_period_before_irregular_exclusion = 0
    rows_excluded_for_irregular_dates = 0
    rows_in_profile_grid = 0
    observed_load_rows = 0
    observed_dates_used: set[pd.Timestamp] = set()

    for batch_number, batch in enumerate(scanner.to_batches(), start=1):
        frame = batch.to_pandas(use_threads=False)
        rows_scanned += len(frame)
        ensure(frame["meter_id"].notna().all(), f"Missing meter_id in batch {batch_number}")
        ensure(frame["local_slot"].notna().all(), f"Missing local_slot in batch {batch_number}")

        period_mask = profile_period_mask(frame, start_date, start_slot, end_date, end_slot)
        period_frame = frame.loc[period_mask].copy()
        rows_in_period_before_irregular_exclusion += len(period_frame)
        if period_frame.empty:
            continue

        period_frame["local_date"] = pd.to_datetime(period_frame["local_date"], errors="raise").dt.normalize()
        irregular_mask = period_frame["local_date"].isin(excluded_dates)
        rows_excluded_for_irregular_dates += int(irregular_mask.sum())
        period_frame = period_frame.loc[~irregular_mask]
        if period_frame.empty:
            continue

        period_frame["local_slot"] = pd.to_numeric(period_frame["local_slot"], errors="raise").astype(np.int16)
        ensure(period_frame["local_slot"].between(1, 48).all(),
               f"Non-standard local slot entered profile construction in batch {batch_number}")
        period_frame["kwh"] = pd.to_numeric(period_frame["kwh"], errors="coerce")
        rows_in_profile_grid += len(period_frame)
        observed_load_rows += int(period_frame["kwh"].notna().sum())
        observed_dates_used.update(period_frame["local_date"].drop_duplicates().tolist())

        grouped = (
            period_frame.groupby(["meter_id", "local_slot"], sort=False, observed=True)
            .agg(
                load_sum=("kwh", "sum"),
                observed_days=("kwh", "count"),
                available_positions=("kwh", "size"),
            )
            .reset_index()
        )
        partials.append(grouped)

    ensure(partials, "No CER rows remained after profile-period and irregular-date filtering")
    combined = pd.concat(partials, ignore_index=True)
    aggregate = (
        combined.groupby(["meter_id", "local_slot"], sort=True, observed=True)
        .agg(
            load_sum=("load_sum", "sum"),
            observed_days=("observed_days", "sum"),
            available_positions=("available_positions", "sum"),
        )
        .reset_index()
    )

    expected_meters = int(config["expected"]["cer_meters"])
    meter_ids = sorted(aggregate["meter_id"].unique().tolist())
    ensure(len(meter_ids) == expected_meters,
           f"Expected {expected_meters} CER meters in profile period, found {len(meter_ids)}")
    ensure(len(aggregate) == expected_meters * 48,
           f"Expected {expected_meters * 48} meter-slot aggregates, found {len(aggregate)}")
    ensure((aggregate["observed_days"] > 0).all(), "At least one CER meter-slot has no observed load")
    ensure((aggregate["available_positions"] >= aggregate["observed_days"]).all(),
           "Observed-day count exceeds available-position count")

    aggregate["raw_mean_kwh"] = aggregate["load_sum"] / aggregate["observed_days"]
    ensure(np.isfinite(aggregate["raw_mean_kwh"].to_numpy(dtype=float)).all(),
           "At least one CER meter-slot mean is non-finite")

    raw_profile_frame = aggregate.pivot(index="meter_id", columns="local_slot", values="raw_mean_kwh")
    raw_profile_frame = raw_profile_frame.reindex(index=meter_ids, columns=list(range(1, 49)))
    ensure(not raw_profile_frame.isna().any().any(), "CER raw profile matrix contains missing values")
    raw_profiles = raw_profile_frame.to_numpy(dtype=np.float64)
    profile_z, profile_means, profile_stds = profile_zscore(raw_profiles)

    count_frame = aggregate.pivot(index="meter_id", columns="local_slot", values="observed_days")
    count_frame = count_frame.reindex(index=meter_ids, columns=list(range(1, 49)))
    position_frame = aggregate.pivot(index="meter_id", columns="local_slot", values="available_positions")
    position_frame = position_frame.reindex(index=meter_ids, columns=list(range(1, 49)))

    low_threshold = float(profile_cfg["daily_profile_variation_flag_threshold_kwh"])
    summary = pd.DataFrame(
        {
            "meter_id": meter_ids,
            "raw_profile_mean_kwh": profile_means,
            "raw_profile_std_kwh_ddof0": profile_stds,
            "low_daily_variation_lt_0_01": profile_stds < low_threshold,
            "min_observed_days_per_slot": count_frame.min(axis=1).to_numpy(dtype=int),
            "median_observed_days_per_slot": count_frame.median(axis=1).to_numpy(dtype=float),
            "max_observed_days_per_slot": count_frame.max(axis=1).to_numpy(dtype=int),
            "min_available_positions_per_slot": position_frame.min(axis=1).to_numpy(dtype=int),
            "max_available_positions_per_slot": position_frame.max(axis=1).to_numpy(dtype=int),
        }
    )

    long_rows: list[pd.DataFrame] = []
    for index, meter_id in enumerate(meter_ids):
        long_rows.append(
            pd.DataFrame(
                {
                    "meter_id": meter_id,
                    "local_slot": np.arange(1, 49, dtype=np.int16),
                    "local_time": [local_time_label(slot) for slot in range(1, 49)],
                    "raw_mean_kwh": raw_profiles[index],
                    "profile_zscore": profile_z[index],
                    "observed_days": count_frame.loc[meter_id].to_numpy(dtype=int),
                    "available_positions": position_frame.loc[meter_id].to_numpy(dtype=int),
                    "raw_profile_mean_kwh": profile_means[index],
                    "raw_profile_std_kwh_ddof0": profile_stds[index],
                    "low_daily_variation_lt_0_01": profile_stds[index] < low_threshold,
                }
            )
        )
    profiles_long = pd.concat(long_rows, ignore_index=True)

    audit = {
        "cer_rows_scanned": int(rows_scanned),
        "rows_in_profile_period_before_irregular_exclusion": int(rows_in_period_before_irregular_exclusion),
        "rows_excluded_for_irregular_dates": int(rows_excluded_for_irregular_dates),
        "rows_in_profile_grid_after_irregular_exclusion": int(rows_in_profile_grid),
        "observed_load_rows_used": int(observed_load_rows),
        "unique_local_dates_used": int(len(observed_dates_used)),
        "first_local_date_used": str(min(observed_dates_used).date()),
        "last_local_date_used": str(max(observed_dates_used).date()),
        "profile_period_start": f"{start_date.date()} {local_time_label(start_slot)}",
        "profile_period_end": f"{end_date.date()} {local_time_label(end_slot)}",
        "excluded_local_dates_configured": [str(value.date()) for value in sorted(excluded_dates)],
        "excluded_local_dates_present_inside_profile_period": [
            str(value.date()) for value in sorted(excluded_dates) if start_date <= value <= end_date
        ],
        "cer_meters": int(len(meter_ids)),
        "profile_rows": int(len(profiles_long)),
        "low_daily_variation_flag_count": int(summary["low_daily_variation_lt_0_01"].sum()),
        "zero_variation_profile_count": int((summary["raw_profile_std_kwh_ddof0"] == 0).sum()),
        "validation_rows_used": 0,
        "test_rows_used": 0,
        "forecasting_metrics_used": 0,
    }
    return profiles_long, summary, audit


def build_assignment_outputs(
    config: dict,
    profiles_long: pd.DataFrame,
    profile_summary: pd.DataFrame,
    prototype_matrix: np.ndarray,
    groups: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meter_ids = profile_summary["meter_id"].tolist()
    receiving_matrix = (
        profiles_long.pivot(index="meter_id", columns="local_slot", values="profile_zscore")
        .reindex(index=meter_ids, columns=list(range(1, 49)))
        .to_numpy(dtype=np.float64)
    )
    distances, nearest_group, second_group, nearest_distance, second_distance, margin, ratio = (
        compute_dtw_assignments(receiving_matrix, prototype_matrix, groups)
    )

    distance_matrix = pd.DataFrame({"meter_id": meter_ids})
    for index, group in enumerate(groups):
        distance_matrix[f"dtw_distance_group_{group}"] = distances[:, index]

    assignment = profile_summary.copy()
    assignment["assigned_group"] = nearest_group
    assignment["nearest_dtw_distance"] = nearest_distance
    assignment["second_nearest_group"] = second_group
    assignment["second_nearest_dtw_distance"] = second_distance
    assignment["distance_margin"] = margin
    assignment["distance_ratio"] = ratio
    assignment["profile_period_start"] = config["profile"]["period_start_label"]
    assignment["profile_period_end"] = config["profile"]["period_end_label"]
    assignment["prototype_k"] = int(config["expected"]["source_k"])
    assignment["prototype_seed_metadata"] = int(config["expected"]["source_seed"])
    assignment["source_prototypes_refitted"] = False
    assignment["receiving_profiles_used_in_k_selection"] = False
    assignment["forecasting_metrics_used_for_assignment"] = False

    group_summary = (
        assignment.groupby("assigned_group", sort=True)
        .agg(
            meter_count=("meter_id", "size"),
            nearest_distance_median=("nearest_dtw_distance", "median"),
            nearest_distance_q1=("nearest_dtw_distance", lambda values: values.quantile(0.25)),
            nearest_distance_q3=("nearest_dtw_distance", lambda values: values.quantile(0.75)),
            distance_margin_median=("distance_margin", "median"),
            distance_ratio_median=("distance_ratio", "median"),
            low_daily_variation_flag_count=("low_daily_variation_lt_0_01", "sum"),
        )
        .reindex(groups)
        .rename_axis("profile_group")
        .reset_index()
    )
    group_summary["meter_count"] = group_summary["meter_count"].fillna(0).astype(int)
    group_summary["low_daily_variation_flag_count"] = (
        group_summary["low_daily_variation_flag_count"].fillna(0).astype(int)
    )
    group_summary["meter_share"] = group_summary["meter_count"] / len(assignment)
    group_summary = group_summary[
        [
            "profile_group",
            "meter_count",
            "meter_share",
            "nearest_distance_median",
            "nearest_distance_q1",
            "nearest_distance_q3",
            "distance_margin_median",
            "distance_ratio_median",
            "low_daily_variation_flag_count",
        ]
    ]
    return assignment, distance_matrix, group_summary


def save_figures(
    config: dict,
    prototype_frame: pd.DataFrame,
    profiles_long: pd.DataFrame,
    assignment: pd.DataFrame,
    group_summary: pd.DataFrame,
) -> None:
    figure_dir = resolve_path(config, config["outputs"]["figure_dir"])
    figure_dir.mkdir(parents=True, exist_ok=True)

    merged = profiles_long.merge(assignment[["meter_id", "assigned_group"]], on="meter_id", how="left", validate="many_to_one")
    receiving_means = (
        merged.groupby(["assigned_group", "local_slot"], sort=True)["profile_zscore"]
        .mean()
        .reset_index()
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.8), constrained_layout=True)
    for group, ax in zip([1, 2, 3, 4], axes.ravel()):
        source = prototype_frame.loc[prototype_frame["profile_group"] == group].sort_values("local_slot")
        receiving = receiving_means.loc[receiving_means["assigned_group"] == group].sort_values("local_slot")
        count = int(group_summary.loc[group_summary["profile_group"] == group, "meter_count"].iloc[0])
        ax.plot(source["local_slot"], source["prototype_zscore"], linewidth=2.5, label="Fixed source profile")
        ax.plot(receiving["local_slot"], receiving["profile_zscore"], linewidth=2.2, linestyle="--", label="Receiving group mean")
        ax.axhline(0.0, linewidth=0.8, alpha=0.5)
        ax.set_title(f"Group {group}, receiving meters = {count}")
        ax.set_xlim(1, 48)
        ax.set_xticks([1, 9, 17, 25, 33, 41, 48], ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "23:30"])
        ax.set_xlabel("Local time")
        ax.set_ylabel("Standardised daily load shape")
        ax.grid(axis="y", alpha=0.20)
        if group == 1:
            ax.legend(frameon=True)
    fig.suptitle("Fixed source profiles and assigned receiving profiles")
    fig.savefig(figure_dir / "target_profile_assignment_fixed_source_and_receiving_profiles.png", dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(figure_dir / "target_profile_assignment_fixed_source_and_receiving_profiles.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9), constrained_layout=True)
    axes[0].bar(group_summary["profile_group"].astype(str), group_summary["meter_count"])
    axes[0].set_title("Receiving meter assignments")
    axes[0].set_xlabel("Source defined group")
    axes[0].set_ylabel("Meters")
    axes[0].grid(axis="y", alpha=0.20)
    for x, value in enumerate(group_summary["meter_count"]):
        axes[0].text(x, value, str(int(value)), ha="center", va="bottom", fontsize=10)

    box_data = [
        assignment.loc[assignment["assigned_group"] == group, "distance_ratio"].to_numpy(dtype=float)
        for group in [1, 2, 3, 4]
    ]
    axes[1].boxplot(box_data, showfliers=False)
    axes[1].set_xticks([1, 2, 3, 4], ["1", "2", "3", "4"])
    axes[1].set_title("Assignment distance ratio")
    axes[1].set_xlabel("Source defined group")
    axes[1].set_ylabel("Nearest distance divided by second distance")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(axis="y", alpha=0.20)
    fig.suptitle("Receiving meter assignments and distance confidence")
    fig.savefig(figure_dir / "target_profile_assignment_assignment_counts_and_distance_confidence.png", dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(figure_dir / "target_profile_assignment_assignment_counts_and_distance_confidence.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    table_dir = resolve_path(config, config["outputs"]["table_dir"])
    metadata_dir = resolve_path(config, config["outputs"]["metadata_dir"])
    decision_path = resolve_path(config, config["outputs"]["decision_md"])
    table_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    decision_path.parent.mkdir(parents=True, exist_ok=True)

    prototype_path = resolve_path(config, config["inputs"]["source_prototypes_csv"])
    prototype_hash_before = sha256_file(prototype_path)
    prototype_frame, prototype_matrix, groups = load_fixed_prototypes(config)
    profiles_long, profile_summary, profile_audit = build_receiving_profiles(config)
    assignment, distance_matrix, group_summary = build_assignment_outputs(
        config, profiles_long, profile_summary, prototype_matrix, groups
    )

    profiles_csv = table_dir / "target_profile_assignment_cer_training_profiles.csv"
    profiles_parquet = table_dir / "target_profile_assignment_cer_training_profiles.parquet"
    assignment_csv = table_dir / "target_profile_assignment_cer_source_defined_assignments.csv"
    assignment_parquet = table_dir / "target_profile_assignment_cer_source_defined_assignments.parquet"
    distance_csv = table_dir / "target_profile_assignment_cer_to_source_prototype_dtw_distances.csv"
    group_csv = table_dir / "target_profile_assignment_cer_assignment_group_summary.csv"
    audit_csv = table_dir / "target_profile_assignment_cer_profile_construction_audit.csv"

    profiles_long.to_csv(profiles_csv, index=False)
    profiles_long.to_parquet(profiles_parquet, index=False)
    assignment.to_csv(assignment_csv, index=False)
    assignment.to_parquet(assignment_parquet, index=False)
    distance_matrix.to_csv(distance_csv, index=False)
    group_summary.to_csv(group_csv, index=False)
    pd.DataFrame([profile_audit]).to_csv(audit_csv, index=False)


    prototype_hash_after = sha256_file(prototype_path)
    ensure(prototype_hash_before == prototype_hash_after, "Fixed source prototype file changed during Target profile assignment")

    status = {
        "analysis": "Target profile assignment fixed source prototype verification and CER assignment",
        "status": "ANALYSIS_COMPLETE_PENDING_VALIDATION",
        "analysis_id": ANALYSIS_ID,
        "config_path": config["_config_path"],
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "source_prototype_path": str(prototype_path),
        "source_prototype_sha256_before": prototype_hash_before,
        "source_prototype_sha256_after": prototype_hash_after,
        "source_k": int(config["expected"]["source_k"]),
        "source_seed_metadata": int(config["expected"]["source_seed"]),
        "source_prototypes_refitted": 0,
        "receiving_profiles_used_in_k_selection": 0,
        "forecasting_metrics_used": 0,
        "cer_meters": int(len(assignment)),
        "cer_assignments": int(assignment["meter_id"].nunique()),
        "unassigned_meters": int(assignment["assigned_group"].isna().sum()),
        "profile_rows": int(len(profiles_long)),
        "distance_ratio_definition": "nearest_dtw_distance / second_nearest_dtw_distance",
        "distance_margin_definition": "second_nearest_dtw_distance - nearest_dtw_distance",
        "profile_audit": profile_audit,
        "group_counts": {
            str(int(row.profile_group)): int(row.meter_count)
            for row in group_summary.itertuples(index=False)
        },
        "outputs": {
            "profiles_csv": str(profiles_csv),
            "profiles_parquet": str(profiles_parquet),
            "assignment_csv": str(assignment_csv),
            "assignment_parquet": str(assignment_parquet),
            "distance_csv": str(distance_csv),
            "group_summary_csv": str(group_csv),
            "profile_audit_csv": str(audit_csv),
        },
    }
    write_json(metadata_dir / "target_profile_assignment_cer_assignment_status.json", status)

    decision_text = f"""# Target profile assignment CER Assignment to Fixed Source Profiles\n\n## Status\n\n```text\nAnalysis completed; formal validation pending\nSource K = {config['expected']['source_k']}\nSource seed metadata = {config['expected']['source_seed']}\nSource prototypes refitted = no\nCER meters assigned = {len(assignment)}\n```\n\n## Locked design\n\n- Fixed source prototypes: `{prototype_path}`\n- CER profile period: `{config['profile']['period_start_label']}` to `{config['profile']['period_end_label']}`\n- Profile representation: 48 local half hour slot means.\n- Profile normalisation: per profile z score with `ddof = 0`.\n- Assignment: nearest fixed source prototype under DTW.\n- Distance margin: second nearest distance minus nearest distance.\n- Distance ratio: nearest distance divided by second nearest distance. Lower ratios indicate clearer separation from the second choice.\n- CER validation rows used: 0.\n- CER test rows used: 0.\n- Forecasting metrics used: 0.\n- Joint source and receiving clustering: no.\n\n## Outputs\n\n- `{assignment_csv}`\n- `{assignment_parquet}`\n- `{profiles_parquet}`\n- `{distance_csv}`\n- `{group_csv}`\n\nFormal validation assigns `COMPLETE_PASS` after exact row, distance and assignment checks pass.\n"""
    decision_path.write_text(decision_text, encoding="utf-8")

    print("=" * 96)
    print("TARGET PROFILE ASSIGNMENT CER ASSIGNMENT ANALYSIS COMPLETE")
    print("=" * 96)
    print(f"Analysis ID                  : {ANALYSIS_ID}")
    print(f"Fixed source K                : {config['expected']['source_k']}")
    print(f"Source seed metadata          : {config['expected']['source_seed']}")
    print("Source prototypes refitted    : NO")
    print(f"CER meters assigned           : {len(assignment)}")
    print(f"Profile rows                  : {len(profiles_long)}")
    print(f"Group counts                  : {status['group_counts']}")
    print(f"Low variation flags only      : {profile_audit['low_daily_variation_flag_count']}")
    print("Meters removed by this flag   : 0")
    print(f"Assignment table              : {assignment_csv}")


if __name__ == "__main__":
    main()
