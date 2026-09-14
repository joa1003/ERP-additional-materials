from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from profile_group_analysis_profile_group_common import (
    CRITERION_VALUE_COLUMNS,
    GAIN_COLUMNS,
    GAIN_LABELS,
    GROUP_DESCRIPTIONS,
    STRATEGY_LABELS,
    ensure,
    load_config,
    normalise_id,
    quantile_summary,
    read_json,
    resolve_path,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile group analysis profile-group analysis")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def save_figure(fig: plt.Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    for path in paths:
        path.unlink(missing_ok=True)
    fig.savefig(paths[0], dpi=300, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    for path in paths:
        ensure(path.is_file() and path.stat().st_size > 0, f"Figure not written: {path}")
    return paths


def load_and_validate(config: dict) -> tuple[Path, dict, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    project_root = Path(config["project_root"]).resolve()
    paths = {key: resolve_path(project_root, value) for key, value in config["inputs"].items()}
    for key, path in paths.items():
        ensure(path.is_file(), f"Missing formal input {key}: {path}")

    status8 = read_json(paths["household_analytical_status"])
    status9 = read_json(paths["target_profile_assignment_status"])
    ensure(status8.get("status") == "COMPLETE_PASS", "Household analytical table status is not COMPLETE_PASS")
    ensure(status9.get("status") == "COMPLETE_PASS", "Target profile assignment status is not COMPLETE_PASS")

    assignments = pd.read_csv(paths["target_profile_assignment_assignments"])
    metrics = pd.read_csv(paths["household_analytical_native_metrics"])
    gains = pd.read_csv(paths["household_analytical_native_gains"])
    shifts = pd.read_csv(paths["household_analytical_common_lead_shifts"])

    required_assign = {"meter_id", "assigned_group", "nearest_dtw_distance", "distance_margin", "distance_ratio"}
    required_metrics = {"entity_id", "strategy", "lead", "support_scope", "seed_count"} | set(CRITERION_VALUE_COLUMNS.values())
    required_gains = {"entity_id", "lead", "criterion", "support_scope", "seed_count"} | set(GAIN_COLUMNS.values())
    required_shifts = {"entity_id", "strategy", "criterion", "h12_minus_h1", "h48_minus_h12", "h48_minus_h1"}
    ensure(required_assign.issubset(assignments.columns), f"Target profile assignment assignment columns missing: {sorted(required_assign - set(assignments.columns))}")
    ensure(required_metrics.issubset(metrics.columns), f"Household analytical table metric columns missing: {sorted(required_metrics - set(metrics.columns))}")
    ensure(required_gains.issubset(gains.columns), f"Household analytical table gain columns missing: {sorted(required_gains - set(gains.columns))}")
    ensure(required_shifts.issubset(shifts.columns), f"Household analytical table lead-shift columns missing: {sorted(required_shifts - set(shifts.columns))}")

    assignments["meter_id"] = normalise_id(assignments["meter_id"])
    metrics["entity_id"] = normalise_id(metrics["entity_id"])
    gains["entity_id"] = normalise_id(gains["entity_id"])
    shifts["entity_id"] = normalise_id(shifts["entity_id"])

    expected = config["expected"]
    ensure(len(assignments) == expected["meters"], f"Expected {expected['meters']} assignment rows, found {len(assignments)}")
    ensure(assignments["meter_id"].nunique() == expected["meters"], "Target profile assignment assignment meter IDs are not unique")
    ensure(len(metrics) == expected["native_metric_rows"], f"Unexpected Household analytical table metric rows: {len(metrics)}")
    ensure(len(gains) == expected["native_gain_rows"], f"Unexpected Household analytical table gain rows: {len(gains)}")
    ensure(len(shifts) == expected["common_lead_shift_rows"], f"Unexpected Household analytical table lead-shift rows: {len(shifts)}")

    assignment_ids = set(assignments["meter_id"])
    ensure(set(metrics["entity_id"]) == assignment_ids, "Household analytical table metric and Target profile assignment assignment meter-ID sets differ")
    ensure(set(gains["entity_id"]) == assignment_ids, "Household analytical table gain and Target profile assignment assignment meter-ID sets differ")
    ensure(set(shifts["entity_id"]) == assignment_ids, "Household analytical table lead-shift and Target profile assignment assignment meter-ID sets differ")

    ensure(sorted(assignments["assigned_group"].astype(int).unique().tolist()) == expected["groups"], "Assigned groups are not exactly 1–4")
    actual_group_counts = assignments["assigned_group"].astype(int).value_counts().sort_index().to_dict()
    expected_group_counts = {int(k): int(v) for k, v in expected["group_counts"].items()}
    ensure(actual_group_counts == expected_group_counts, f"Group counts differ: expected={expected_group_counts}, actual={actual_group_counts}")

    ensure(sorted(metrics["lead"].astype(int).unique().tolist()) == expected["leads"], "Metric leads differ from expected")
    ensure(sorted(gains["lead"].astype(int).unique().tolist()) == expected["leads"], "Gain leads differ from expected")
    ensure(sorted(metrics["strategy"].astype(str).unique().tolist()) == sorted(expected["strategies"]), "Metric strategies differ from expected")
    ensure(sorted(gains["criterion"].astype(str).unique().tolist()) == sorted(expected["criteria"]), "Gain criteria differ from expected")
    ensure((metrics["seed_count"].astype(int) == expected["seed_count"]).all(), "Metric seed_count is not 4 throughout")
    ensure((gains["seed_count"].astype(int) == expected["seed_count"]).all(), "Gain seed_count is not 4 throughout")
    ensure(set(metrics["support_scope"].astype(str)) == {"native"}, "Primary metric input is not native support")
    ensure(set(gains["support_scope"].astype(str)) == {"native"}, "Primary gain input is not native support")

    return project_root, status8, status9, assignments, metrics, gains, shifts, paths


def build_group_strategy_summary(metrics_grouped: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (group, lead, strategy), part in metrics_grouped.groupby(["assigned_group", "lead", "strategy"], sort=True):
        for criterion, column in CRITERION_VALUE_COLUMNS.items():
            stats = quantile_summary(part[column])
            rows.append({
                "assigned_group": int(group),
                "group_description": GROUP_DESCRIPTIONS[int(group)],
                "lead": int(lead),
                "strategy": strategy,
                "strategy_label": STRATEGY_LABELS[strategy],
                "criterion": criterion,
                "unit": "kWh" if criterion in {"mae", "rmse"} else "percent",
                "meters": int(part["entity_id"].nunique()),
                "aggregation": "equal-meter",
                **stats,
            })
    return pd.DataFrame(rows)


def build_group_gain_summary(gains_grouped: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (group, lead, criterion), part in gains_grouped.groupby(["assigned_group", "lead", "criterion"], sort=True):
        for gain_name, column in GAIN_COLUMNS.items():
            values = pd.to_numeric(part[column], errors="coerce")
            stats = quantile_summary(values)
            rows.append({
                "assigned_group": int(group),
                "group_description": GROUP_DESCRIPTIONS[int(group)],
                "lead": int(lead),
                "criterion": criterion,
                "unit": "kWh" if criterion in {"mae", "rmse"} else "percentage points",
                "gain": gain_name,
                "gain_label": GAIN_LABELS[gain_name],
                "meters": int(part["entity_id"].nunique()),
                "aggregation": "equal-meter",
                **stats,
                "positive_meters": int((values > 0).sum()),
                "negative_meters": int((values < 0).sum()),
                "tie_meters": int((values == 0).sum()),
                "positive_share": float((values > 0).mean()),
                "negative_share": float((values < 0).mean()),
            })
    return pd.DataFrame(rows)


def build_lead_shift_summary(shifts_grouped: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (group, strategy, criterion), part in shifts_grouped.groupby(["assigned_group", "strategy", "criterion"], sort=True):
        for shift_name in ["h12_minus_h1", "h48_minus_h12", "h48_minus_h1"]:
            stats = quantile_summary(part[shift_name])
            rows.append({
                "assigned_group": int(group),
                "group_description": GROUP_DESCRIPTIONS[int(group)],
                "strategy": strategy,
                "strategy_label": STRATEGY_LABELS[strategy],
                "criterion": criterion,
                "unit": "kWh" if criterion in {"mae", "rmse"} else "percentage points",
                "shift": shift_name,
                "meters": int(part["entity_id"].nunique()),
                "aggregation": "equal-meter on strict common rows",
                **stats,
                "positive_share": float((pd.to_numeric(part[shift_name]) > 0).mean()),
            })
    return pd.DataFrame(rows)


def build_main_text_gain_table(group_gain: pd.DataFrame) -> pd.DataFrame:
    selected = group_gain[group_gain["gain"].isin(["source_transfer_gain", "fine_tuning_gain", "full_data_gain"])].copy()
    selected["median_iqr"] = selected.apply(lambda r: f"{r['median']:.4f} ({r['q25']:.4f}–{r['q75']:.4f})", axis=1)
    selected["positive_n_share"] = selected.apply(lambda r: f"{int(r['positive_meters'])} ({100*r['positive_share']:.1f}%)", axis=1)
    return selected[[
        "assigned_group", "group_description", "lead", "criterion", "unit", "gain", "gain_label", "meters", "median_iqr", "positive_n_share"
    ]].sort_values(["lead", "criterion", "assigned_group", "gain"])


def build_h12_strategy_table(group_strategy: pd.DataFrame, group_gain: pd.DataFrame) -> pd.DataFrame:
    strategy = group_strategy[group_strategy["lead"] == 12].copy()
    wide = strategy.pivot(index=["assigned_group", "group_description", "criterion", "unit", "meters"], columns="strategy", values="mean").reset_index()
    gain = group_gain[(group_gain["lead"] == 12) & (group_gain["gain"].isin(["source_transfer_gain", "fine_tuning_gain", "full_data_gain"]))]
    gain_wide = gain.pivot(index=["assigned_group", "criterion"], columns="gain", values="mean").reset_index()
    result = wide.merge(gain_wide, on=["assigned_group", "criterion"], how="left", validate="one_to_one")
    ordered = [
        "assigned_group", "group_description", "criterion", "unit", "meters",
        "direct_transfer", "fine_tuning", "cer_scratch_limited", "cer_scratch_full",
        "source_transfer_gain", "fine_tuning_gain", "full_data_gain",
    ]
    return result[ordered].sort_values(["criterion", "assigned_group"])


def plot_gain(group_gain: pd.DataFrame, gain_name: str, figure_dir: Path) -> list[Path]:
    frame = group_gain[group_gain["gain"] == gain_name].copy()
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8), constrained_layout=True)
    lead_offsets = {1: -0.18, 12: 0.0, 48: 0.18}
    x = np.arange(1, 5, dtype=float)
    for axis, criterion in zip(axes, ["mae", "rmse", "smape"]):
        part = frame[frame["criterion"] == criterion]
        for lead in [1, 12, 48]:
            lead_part = part[part["lead"] == lead].sort_values("assigned_group")
            y = lead_part["median"].to_numpy(float)
            lower = y - lead_part["q25"].to_numpy(float)
            upper = lead_part["q75"].to_numpy(float) - y
            axis.errorbar(x + lead_offsets[lead], y, yerr=np.vstack([lower, upper]), marker="o", capsize=3, linewidth=1.6, label=f"h={lead}")
        axis.axhline(0.0, linewidth=1.0, linestyle="--")
        axis.set_xticks(x, ["G1", "G2", "G3", "G4"])
        axis.set_title(criterion.upper(), loc="left", fontweight="bold")
        axis.set_xlabel("Source-defined profile group")
        axis.set_ylabel("Gain (kWh)" if criterion in {"mae", "rmse"} else "Gain (percentage points)")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    fig.suptitle(f"{GAIN_LABELS[gain_name]} by profile group and forecast lead", fontweight="bold")
    return save_figure(fig, figure_dir / f"profile_group_analysis_{gain_name}_by_group_and_lead")


def plot_strategy_mae(group_strategy: pd.DataFrame, figure_dir: Path) -> list[Path]:
    frame = group_strategy[group_strategy["criterion"] == "mae"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8), constrained_layout=True)
    x = np.arange(1, 5)
    for axis, lead in zip(axes, [1, 12, 48]):
        part = frame[frame["lead"] == lead]
        for strategy in ["direct_transfer", "fine_tuning", "cer_scratch_limited", "cer_scratch_full"]:
            row = part[part["strategy"] == strategy].sort_values("assigned_group")
            axis.plot(x, row["mean"].to_numpy(float), marker="o", linewidth=1.8, label=STRATEGY_LABELS[strategy])
        axis.set_xticks(x, ["G1", "G2", "G3", "G4"])
        axis.set_title(f"h={lead}", loc="left", fontweight="bold")
        axis.set_xlabel("Source-defined profile group")
        axis.set_ylabel("Equal-meter mean MAE (kWh)")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Strategy MAE by source-defined profile group", fontweight="bold")
    return save_figure(fig, figure_dir / "profile_group_analysis_strategy_mae_by_group_and_lead")


def write_decision_note(path: Path, group_gain: pd.DataFrame, group_strategy: pd.DataFrame, config: dict) -> None:
    lines = [
        "# Profile group analysis profile-group analysis decision note",
        "",
        "## Status",
        "",
        "Profile group analysis completed the equal-meter descriptive aggregation of Household analytical table meter outcomes within the fixed Target profile assignment K=4 source-defined groups.",
        "",
        "## Fixed analytical rules",
        "",
        "- Primary evidence uses native lead-specific meter results.",
        "- Cross-lead shifts use the strict common-row Household analytical table table as supporting evidence.",
        "- Every meter contributes equally within its assigned group.",
        "- No prototype was refitted and no meter was reassigned or excluded.",
        "- Positive gains follow the Household analytical table definitions recorded in the configuration.",
        "- Results are descriptive and non-causal.",
        "",
        "## Automatically identified group contrasts",
        "",
    ]
    selected = group_gain[group_gain["gain"].isin(["source_transfer_gain", "fine_tuning_gain", "full_data_gain"])]
    for lead in [1,12,48]:
        for criterion in ["mae","rmse","smape"]:
            part = selected[(selected["lead"] == lead) & (selected["criterion"] == criterion)]
            for gain in ["source_transfer_gain","fine_tuning_gain","full_data_gain"]:
                gpart = part[part["gain"] == gain]
                best = gpart.loc[gpart["median"].idxmax()]
                worst = gpart.loc[gpart["median"].idxmin()]
                lines.append(f"- h={lead}, {criterion.upper()}, {GAIN_LABELS[gain]}: highest median in G{int(best['assigned_group'])} ({best['median']:.6f}); lowest median in G{int(worst['assigned_group'])} ({worst['median']:.6f}).")
    lines.extend([
        "",
        "## Interpretation limit",
        "",
        "Group differences identify heterogeneity under the fixed study design. They do not establish that cluster membership causes forecasting or transfer outcomes. Continuous DTW distance and assignment-confidence relationships are reserved for Source similarity.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    project_root, status8, status9, assignments, metrics, gains, shifts, input_paths = load_and_validate(config)

    outputs = config["outputs"]
    table_dir = resolve_path(project_root, outputs["table_dir"])
    metadata_dir = resolve_path(project_root, outputs["metadata_dir"])
    decision_dir = resolve_path(project_root, outputs["decision_dir"])
    for path in [table_dir, metadata_dir, decision_dir]:
        path.mkdir(parents=True, exist_ok=True)

    membership = assignments.copy()
    membership["assigned_group"] = membership["assigned_group"].astype(int)
    membership["group_description"] = membership["assigned_group"].map(GROUP_DESCRIPTIONS)
    membership.to_csv(table_dir / "profile_group_analysis_meter_group_membership.csv", index=False)

    metric_join = metrics.merge(membership[["meter_id", "assigned_group", "group_description"]], left_on="entity_id", right_on="meter_id", how="left", validate="many_to_one").drop(columns=["meter_id"])
    gain_join = gains.merge(membership[["meter_id", "assigned_group", "group_description"]], left_on="entity_id", right_on="meter_id", how="left", validate="many_to_one").drop(columns=["meter_id"])
    shift_join = shifts.merge(membership[["meter_id", "assigned_group", "group_description"]], left_on="entity_id", right_on="meter_id", how="left", validate="many_to_one").drop(columns=["meter_id"])
    ensure(metric_join["assigned_group"].notna().all() and gain_join["assigned_group"].notna().all() and shift_join["assigned_group"].notna().all(), "Join produced unmatched meters")

    metric_join.to_csv(table_dir / "profile_group_analysis_native_meter_metrics_with_group.csv", index=False)
    gain_join.to_csv(table_dir / "profile_group_analysis_native_meter_gains_with_group.csv", index=False)

    group_strategy = build_group_strategy_summary(metric_join)
    group_gain = build_group_gain_summary(gain_join)
    lead_shift = build_lead_shift_summary(shift_join)
    main_gain = build_main_text_gain_table(group_gain)
    h12_table = build_h12_strategy_table(group_strategy, group_gain)

    group_strategy.to_csv(table_dir / "profile_group_analysis_group_strategy_metric_summary.csv", index=False)
    group_gain.to_csv(table_dir / "profile_group_analysis_group_gain_summary.csv", index=False)
    lead_shift.to_csv(table_dir / "profile_group_analysis_group_lead_shift_summary.csv", index=False)
    main_gain.to_csv(table_dir / "profile_group_analysis_main_text_group_gain_table.csv", index=False)
    h12_table.to_csv(table_dir / "profile_group_analysis_h12_profile_group_strategy_table.csv", index=False)

    decision_path = decision_dir / "profile_group_analysis_profile_group_analysis_decision.md"
    write_decision_note(decision_path, group_gain, group_strategy, config)

    inventory = []
    for path in sorted(list(table_dir.glob("profile_group_analysis_*.csv")) + [decision_path]):
        inventory.append({
            "path": path.relative_to(project_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "role": "table" if path.suffix == ".csv" else ("figure" if path.suffix in {".png", ".pdf"} else "technical decision"),
        })
    pd.DataFrame(inventory).to_csv(metadata_dir / "profile_group_analysis_output_inventory.csv", index=False)

    lock_path = resolve_path(project_root, config["experiment_lock"])
    status = {
        "status": "COMPLETE_PASS",
        "analysis_id": config["analysis_id"],
        "analysis": "Profile group analysis source-defined profile-group forecasting and gain analysis",
        "meters": int(membership["meter_id"].nunique()),
        "groups": sorted(membership["assigned_group"].unique().astype(int).tolist()),
        "group_counts": {str(k): int(v) for k, v in membership["assigned_group"].value_counts().sort_index().to_dict().items()},
        "leads": [1,12,48],
        "strategies": config["expected"]["strategies"],
        "criteria": config["expected"]["criteria"],
        "aggregation": config["analysis"]["aggregation"],
        "primary_support": config["analysis"]["primary_support"],
        "cross_lead_support": config["analysis"]["cross_lead_support"],
        "household_analytical_status": status8.get("status"),
        "target_profile_assignment_status": status9.get("status"),
        "meter_id_set_equality": True,
        "unmatched_meters": 0,
        "group_strategy_summary_rows": len(group_strategy),
        "group_gain_summary_rows": len(group_gain),
        "group_lead_shift_summary_rows": len(lead_shift),
        "main_text_group_gain_rows": len(main_gain),
        "h12_profile_group_strategy_rows": len(h12_table),
        "model_training_performed": 0,
        "prototypes_refitted": 0,
        "meters_reassigned": 0,
        "meters_excluded": 0,
        "forecasting_metrics_used_to_define_groups": 0,
        "continuous_dtw_relationship_analysis_performed": 0,
        "interpretation": config["analysis"]["interpretation"],
        "experiment_lock_path": str(lock_path),
        "experiment_lock_sha256": sha256_file(lock_path) if lock_path.is_file() else None,
        "input_sha256": {key: sha256_file(path) for key, path in input_paths.items()},
        "output_inventory": str((metadata_dir / "profile_group_analysis_output_inventory.csv").relative_to(project_root)),
    }
    (metadata_dir / "profile_group_analysis_profile_group_analysis_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    print("=" * 96)
    print("PROFILE GROUP ANALYSIS PROFILE-GROUP ANALYSIS COMPLETE")
    print("=" * 96)
    print(f"Meters joined                 : {status['meters']}")
    print(f"Group counts                  : {status['group_counts']}")
    print(f"Group strategy summary rows   : {len(group_strategy)}")
    print(f"Group gain summary rows       : {len(group_gain)}")
    print(f"Group lead-shift summary rows : {len(lead_shift)}")
    print("Aggregation                   : equal meter within source-defined group")
    print("Continuous DTW analysis       : NO — reserved for Source similarity")
    print(f"Status                        : {status['status']}")

if __name__ == "__main__":
    main()
