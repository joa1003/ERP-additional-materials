import os as _erp_os
from pathlib import Path as _ERPPath
_ERP_PROJECT_ROOT = _ERPPath(_erp_os.environ.get("ERP_PROJECT_ROOT", ".")).expanduser().resolve()
"""Render final report Figures 4.6 and 4.7 from reproduced load-range outputs.

Figure 4.6 shows D−T gain across observed load ranges at h = 12. Figure 4.7
shows the cumulative contribution of those ranges to the D−T net gain.
"""


import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ANALYSIS_NAME = "load_range_analysis_complete_cross_strategy_raw_load_composition_standardised_gain_decomposition"

DEFAULT_TABLE_DIR = Path(
    f"{_ERP_PROJECT_ROOT}/outputs/tables/"
    "load_range_analysis_complete_cross_strategy_raw_load_composition_standardised_gain_decomposition"
)
DEFAULT_OUTPUT_DIR = Path(
    f"{_ERP_PROJECT_ROOT}/outputs/figures/"
    "load_range_analysis_complete_cross_strategy_raw_load_composition_standardised_gain_decomposition"
)

METER_BIN_FILE = "load_range_analysis_h12_meter_four_bin_pairwise_gains_four_seed.csv"
GROUP_BIN_FILE = "load_range_analysis_h12_group_bin_conditional_gain_and_contribution.csv"
STANDARDISATION_FILE = "load_range_analysis_h12_group_observed_vs_common_composition_gain.csv"

# These are the canonical load-range index values. They are used only for stable ordering/filtering.
COMPARISON_ID = "fine_tuning"  # Fine Tuning over Direct Transfer (D-T)
FOCAL_STRATEGY = "fine_tuning"
COMPARATOR_STRATEGY = "direct_transfer"
OVERALL_GROUP = 0
OVERALL_POPULATION = "Overall CER"

RAW_BIN_ORDER = [
    "actual_le_0_1",
    "actual_gt_0_1_le_0_5",
    "actual_gt_0_5_le_1_0",
    "actual_gt_1_0",
]
RAW_BIN_DISPLAY = {
    "actual_le_0_1": "≤0.1 kWh",
    "actual_gt_0_1_le_0_5": "0.1-0.5 kWh",
    "actual_gt_0_5_le_1_0": "0.5-1.0 kWh",
    "actual_gt_1_0": ">1.0 kWh",
}

GROUP_ORDER = [1, 2, 3, 4]
GROUP_LABELS = {1: "G1", 2: "G2", 3: "G3", 4: "G4"}

METRICS = [
    {"key": "mae", "title": "MAE", "gain_col": "gain_mae", "unit": "kWh", "plain_unit": "kWh", "fmt": "+.4f"},
    {"key": "mse", "title": "MSE", "gain_col": "gain_mse", "unit": "kWh$^2$", "plain_unit": "kWh²", "fmt": "+.4f"},
    {"key": "smape", "title": "sMAPE", "gain_col": "gain_smape", "unit": "pp", "plain_unit": "pp", "fmt": "+.2f"},
]

# Thesis-like visual language: blue boxes, gold median/net, white diamond mean, blue no-gain line.
BOX_COLORS = ["#BFD3E6", "#7FA6C8", "#567FA7", "#2C3E50"]
BIN_POINT_COLORS = ["#BFD3E6", "#7FA6C8", "#567FA7", "#2C3E50"]
GROUP_COLORS = {1: "#3B6FA0", 2: "#3F8F86", 3: "#C6902F", 4: "#6B5B95"}
MEDIAN_COLOR = "#C88900"
NET_COLOR = "#C88900"
NO_GAIN_COLOR = "#6FA0E8"
LINE_COLOR = "#243447"
GRID_COLOR = "#E4E8EF"
TEXT_COLOR = "#253244"
OUTLIER_COLOR = "#B8BCC4"
AXIS_COLOR = "#666666"


def locate_table_dir(args: argparse.Namespace) -> Path:
    """Locate reproduced Load range analysis tables."""
    if args.table_dir:
        table_dir = Path(args.table_dir).expanduser().resolve()
        if not table_dir.exists():
            raise FileNotFoundError(f"--table-dir does not exist: {table_dir}")
        return table_dir

    if DEFAULT_TABLE_DIR.exists():
        return DEFAULT_TABLE_DIR

    raise FileNotFoundError(
        "Could not find the reproduced Load range analysis table directory. "
        f"Expected {DEFAULT_TABLE_DIR}. Run the load-range analysis first, "
        "or pass --table-dir /path/to/reproduced/tables."
    )


def require_columns(df: pd.DataFrame, columns: Iterable[str], file_name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {file_name}: {missing}")


def validate_strategy_filter(df: pd.DataFrame, file_name: str) -> pd.DataFrame:
    """Filter to D-T and validate focal/comparator labels when available."""
    require_columns(df, ["comparison_id"], file_name)
    out = df[df["comparison_id"].eq(COMPARISON_ID)].copy()
    if out.empty:
        raise ValueError(f"No D-T rows found in {file_name}: comparison_id == {COMPARISON_ID!r}")

    if "focal_strategy" in out.columns and not out["focal_strategy"].eq(FOCAL_STRATEGY).all():
        bad = out.loc[~out["focal_strategy"].eq(FOCAL_STRATEGY), "focal_strategy"].unique()
        raise ValueError(f"Unexpected focal_strategy values in D-T rows of {file_name}: {bad}")
    if "comparator_strategy" in out.columns and not out["comparator_strategy"].eq(COMPARATOR_STRATEGY).all():
        bad = out.loc[~out["comparator_strategy"].eq(COMPARATOR_STRATEGY), "comparator_strategy"].unique()
        raise ValueError(f"Unexpected comparator_strategy values in D-T rows of {file_name}: {bad}")
    return out


def load_data(table_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meter_path = table_dir / METER_BIN_FILE
    group_bin_path = table_dir / GROUP_BIN_FILE
    standard_path = table_dir / STANDARDISATION_FILE

    for p in [meter_path, group_bin_path, standard_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    meter = validate_strategy_filter(pd.read_csv(meter_path), METER_BIN_FILE)
    group_bin = validate_strategy_filter(pd.read_csv(group_bin_path), GROUP_BIN_FILE)
    standard = validate_strategy_filter(pd.read_csv(standard_path), STANDARDISATION_FILE)

    require_columns(
        meter,
        ["meter_id", "raw_bin", "row_share", "gain_mae", "gain_mse", "gain_smape"],
        METER_BIN_FILE,
    )
    require_columns(
        group_bin,
        [
            "assigned_group",
            "population",
            "raw_bin",
            "raw_bin_label",
            "metric",
            "unit",
            "mean_meter_row_share",
            "exposure_weighted_conditional_gain",
            "observed_gain_contribution",
        ],
        GROUP_BIN_FILE,
    )
    require_columns(
        standard,
        [
            "assigned_group",
            "metric",
            "unit",
            "observed_group_gain",
            "common_composition_standardised_gain",
            "standardisation_change",
        ],
        STANDARDISATION_FILE,
    )

    # Overall rows for Table J.2 logic.
    group_bin_overall = group_bin[
        group_bin["assigned_group"].eq(OVERALL_GROUP) | group_bin["population"].eq(OVERALL_POPULATION)
    ].copy()
    if group_bin_overall.empty:
        raise ValueError("No overall D-T rows found in group-bin table.")

    # Profile group rows for Table J.1 logic.
    standard_groups = standard[standard["assigned_group"].isin(GROUP_ORDER)].copy()
    if standard_groups.empty:
        raise ValueError("No profile-group D-T rows found in standardisation table.")

    # Check that the canonical raw-bin indices are present before plotting.
    for raw_bin in RAW_BIN_ORDER:
        if raw_bin not in set(meter["raw_bin"]):
            raise ValueError(f"Missing raw_bin in meter-level file: {raw_bin}")
        if raw_bin not in set(group_bin_overall["raw_bin"]):
            raise ValueError(f"Missing raw_bin in overall group-bin file: {raw_bin}")

    for metric in [m["key"] for m in METRICS]:
        if metric not in set(group_bin_overall["metric"]):
            raise ValueError(f"Missing metric in group-bin file: {metric}")
        if metric not in set(standard_groups["metric"]):
            raise ValueError(f"Missing metric in standardisation file: {metric}")

    meter["raw_bin"] = pd.Categorical(meter["raw_bin"], RAW_BIN_ORDER, ordered=True)
    group_bin_overall["raw_bin"] = pd.Categorical(group_bin_overall["raw_bin"], RAW_BIN_ORDER, ordered=True)
    standard_groups["assigned_group"] = pd.Categorical(standard_groups["assigned_group"], GROUP_ORDER, ordered=True)
    return meter, group_bin_overall, standard_groups


def clean_axes(ax, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS_COLOR)
    ax.spines["bottom"].set_color(AXIS_COLOR)
    ax.tick_params(axis="both", labelsize=9, colors="#444444")
    ax.grid(axis=grid_axis, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)


def format_value(value: float, metric: dict) -> str:
    return format(float(value), metric["fmt"])


def get_bin_labels(group_bin_overall: pd.DataFrame, metric_key: str = "mae") -> List[str]:
    rows = group_bin_overall[group_bin_overall["metric"].eq(metric_key)].sort_values("raw_bin")
    labels = []
    for _, row in rows.iterrows():
        raw_bin = str(row["raw_bin"])
        share = 100 * float(row["mean_meter_row_share"])
        labels.append(f"{RAW_BIN_DISPLAY[raw_bin]}\n({share:.0f}%)")
    return labels


def save_figure(fig, output_dir: Path, stem: str, formats: List[str]) -> List[Path]:
    written = []
    for ext in formats:
        path = output_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        written.append(path)
    plt.close(fig)
    return written


def plot_loadrange_boxplot(meter: pd.DataFrame, group_bin_overall: pd.DataFrame) -> plt.Figure:
    """Figure 1: household-level within-range distributions, based on the meter-level file."""
    xlabels = get_bin_labels(group_bin_overall, metric_key="mae")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=False)

    for ax, metric in zip(axes, METRICS):
        data = [
            meter.loc[meter["raw_bin"].eq(raw_bin), metric["gain_col"]].dropna().astype(float).to_numpy()
            for raw_bin in RAW_BIN_ORDER
        ]
        bp = ax.boxplot(
            data,
            positions=np.arange(1, len(RAW_BIN_ORDER) + 1),
            widths=0.55,
            patch_artist=True,
            showmeans=True,
            whis=1.5,
            meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor=LINE_COLOR, markersize=6, zorder=5),
            medianprops=dict(color=MEDIAN_COLOR, linewidth=2.2),
            whiskerprops=dict(color="#5A5A5A", linewidth=1.1),
            capprops=dict(color="#5A5A5A", linewidth=1.1),
            boxprops=dict(linewidth=1.1, edgecolor="#3A3A3A"),
            flierprops=dict(marker="o", markersize=2.2, markerfacecolor=OUTLIER_COLOR, markeredgecolor="none", alpha=0.45),
        )
        for patch, color in zip(bp["boxes"], BOX_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.88)

        ax.axhline(0, color=NO_GAIN_COLOR, linestyle="--", linewidth=1.1, zorder=1)
        ax.set_title(metric["title"], fontsize=13, fontweight="bold", color=TEXT_COLOR)
        ax.set_ylabel(f"Within range gain ({metric['unit']})", fontsize=10)
        ax.set_xticks(np.arange(1, len(xlabels) + 1))
        ax.set_xticklabels(xlabels, fontsize=8.6)
        clean_axes(ax, grid_axis="y")

    legend_elements = [
        Line2D([0], [0], color=MEDIAN_COLOR, lw=2.2, label="Median"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="white", markeredgecolor=LINE_COLOR, markersize=7, label="Mean"),
        Patch(facecolor=BOX_COLORS[1], alpha=0.88, edgecolor="#3A3A3A", label="Box = IQR (household gains within range)"),
        Line2D([0], [0], color=NO_GAIN_COLOR, linestyle="--", lw=1.1, label="No gain"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.055))
    fig.suptitle(
        "Fine Tuning over Direct Transfer gain by observed load range (h = 12)",
        fontsize=14,
        fontweight="bold",
        color=TEXT_COLOR,
        y=1.02,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    return fig


def plot_loadrange_cumline(group_bin_overall: pd.DataFrame) -> plt.Figure:
    """Figure 2: running total of observed-load-range contributions to net gain."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=False)

    for ax, metric in zip(axes, METRICS):
        rows = group_bin_overall[group_bin_overall["metric"].eq(metric["key"])].sort_values("raw_bin")
        contributions = rows["observed_gain_contribution"].astype(float).to_numpy()
        shares = rows["mean_meter_row_share"].astype(float).to_numpy()
        cumulative = np.concatenate([[0.0], np.cumsum(contributions)])
        xs = np.arange(len(cumulative))
        xlabels = ["Start"] + [f"{RAW_BIN_DISPLAY[str(rb)]}\n({100 * s:.0f}%)" for rb, s in zip(rows["raw_bin"], shares)]
        net = float(cumulative[-1])

        ax.plot(xs, cumulative, color=LINE_COLOR, linewidth=2.0, zorder=3)
        ax.scatter([xs[0]], [cumulative[0]], color="white", edgecolor=LINE_COLOR, linewidth=1.2, s=84, zorder=4)
        ax.scatter(xs[1:-1], cumulative[1:-1], color=BIN_POINT_COLORS[:-1], edgecolor=LINE_COLOR, linewidth=1.0, s=90, zorder=4)
        ax.scatter([xs[-1]], [cumulative[-1]], color=NET_COLOR, edgecolor=LINE_COLOR, linewidth=1.2, s=120, zorder=5)
        ax.axhline(0, color=NO_GAIN_COLOR, linestyle="--", linewidth=1.1, zorder=1)

        # Labels report the INCREMENT added by each observed load range. The point position is the running total.
        yrange = max(cumulative) - min(cumulative)
        if yrange == 0:
            yrange = max(abs(net), 1.0)
        for i, c in enumerate(contributions, start=1):
            va = "bottom" if c >= 0 else "top"
            dy = 9 if c >= 0 else -9
            ax.annotate(
                "Δ " + format_value(c, metric),
                (xs[i], cumulative[i]),
                textcoords="offset points",
                xytext=(0, dy),
                ha="center",
                va=va,
                fontsize=8,
                color=TEXT_COLOR,
            )

        ax.annotate(
            f"Net gain\n{format_value(net, metric)} {metric['plain_unit']}",
            (xs[-1], cumulative[-1]),
            textcoords="offset points",
            xytext=(30, 0),
            ha="left",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=TEXT_COLOR,
        )

        ax.set_title(metric["title"], fontsize=13, fontweight="bold", color=TEXT_COLOR)
        ax.set_ylabel(f"Cumulative contribution ({metric['unit']})", fontsize=9.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(xlabels, fontsize=8.4, rotation=0)
        clean_axes(ax, grid_axis="y")
        ax.set_xlim(-0.3, xs[-1] + 0.75)
        ax.set_ylim(min(cumulative) - 0.45 * yrange - 1e-12, max(cumulative) + 0.45 * yrange + 1e-12)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor=LINE_COLOR, markersize=7, label="Start (0)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=NET_COLOR, markeredgecolor=LINE_COLOR, markersize=8, label="Net gain point"),
        Line2D([0], [0], color=LINE_COLOR, lw=2.0, label="Running total across observed load ranges"),
        Line2D([0], [0], color=NO_GAIN_COLOR, linestyle="--", lw=1.1, label="No gain"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.055))
    fig.suptitle(
        "Cumulative contribution to Fine Tuning over Direct Transfer net gain (h = 12)",
        fontsize=14,
        fontweight="bold",
        color=TEXT_COLOR,
        y=1.03,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])
    return fig


def parse_formats(value: str) -> List[str]:
    formats = [v.strip().lower().lstrip(".") for v in value.split(",") if v.strip()]
    allowed = {"png", "pdf"}
    bad = [f for f in formats if f not in allowed]
    if bad:
        raise ValueError(f"Unsupported format(s): {bad}. Supported: png,pdf")
    return formats or ["png"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot h=12 D-T main-text figures from reproduced load-range analysis outputs.")
    parser.add_argument("--table-dir", default=None, help="Path to extracted Load range analysis tables directory. Optional.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory where figure files will be saved.")
    parser.add_argument(
        "--formats",
        default="png,pdf",
        help="Comma-separated output formats. Default: png,pdf. Use --formats png if PDF is not needed.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = parse_formats(args.formats)

    table_dir = locate_table_dir(args)
    print(f"Using table directory: {table_dir}")
    meter, group_bin_overall, _ = load_data(table_dir)

    written: List[Path] = []
    written.extend(save_figure(
        plot_loadrange_boxplot(meter, group_bin_overall),
        output_dir,
        "figure_4_6_dt_gain_by_load_range",
        formats,
    ))
    written.extend(save_figure(
        plot_loadrange_cumline(group_bin_overall),
        output_dir,
        "figure_4_7_dt_cumulative_load_range_contribution",
        formats,
    ))

    print("\nCreated figure files:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
