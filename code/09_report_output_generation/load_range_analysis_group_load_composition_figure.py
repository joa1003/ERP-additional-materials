#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os as _erp_os
from pathlib import Path as _ERPPath
_ERP_PROJECT_ROOT = _ERPPath(_erp_os.environ.get("ERP_PROJECT_ROOT", ".")).expanduser().resolve()

"""
Load range analysis | Section 4.3 observed load composition by group at h = 12

Creates ONE figure only:
    figure_4_3_ld_group_load_composition.png
    figure_4_3_ld_group_load_composition.pdf

The figure is built directly from the reproduced load-range analysis table:
    load_range_analysis_h12_group_bin_conditional_gain_and_contribution.csv

No result values are hard-coded.

The rightmost bar is the common composition reference used in the
standardisation analysis. Its annotation is:

    Common composition weights
    Reference load distribution for all groups

Default output directory:
    <ERP_PROJECT_ROOT>/outputs/figures/
    load_range_analysis_complete_cross_strategy_raw_load_composition_standardised_gain_decomposition
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Paths
# =============================================================================

ANALYSIS_NAME = (
    "load_range_analysis_complete_cross_strategy_raw_load_composition_"
    "standardised_gain_decomposition"
)

GROUP_BIN_FILE = "load_range_analysis_h12_group_bin_conditional_gain_and_contribution.csv"

DEFAULT_BUNDLE_TABLE_DIR = Path(
    f"{_ERP_PROJECT_ROOT}/outputs/tables/"
    "load_range_analysis_complete_cross_strategy_raw_load_composition_"
    "standardised_gain_decomposition"
)

DEFAULT_OUTPUT_TABLE_DIR = Path(
    f"{_ERP_PROJECT_ROOT}/outputs/tables/"
    "load_range_analysis_complete_cross_strategy_raw_load_composition_"
    "standardised_gain_decomposition"
)

ERP_ROOT = Path(f"{_ERP_PROJECT_ROOT}")

DEFAULT_OUTPUT_DIR = Path(
    f"{_ERP_PROJECT_ROOT}/outputs/figures/"
    "load_range_analysis_complete_cross_strategy_raw_load_composition_"
    "standardised_gain_decomposition"
)


# =============================================================================
# Formal Load range analysis identifiers
# =============================================================================

COMPARISON_ID = "source_transfer"
OVERALL_POPULATION = "Overall CER"

GROUP_ORDER = [1, 2, 3, 4]
GROUP_LABEL = {1: "G1", 2: "G2", 3: "G3", 4: "G4"}

RAW_BINS = [
    "actual_le_0_1",
    "actual_gt_0_1_le_0_5",
    "actual_gt_0_5_le_1_0",
    "actual_gt_1_0",
]

BIN_DISPLAY = {
    "actual_le_0_1": "≤0.1 kWh",
    "actual_gt_0_1_le_0_5": "0.1–0.5 kWh",
    "actual_gt_0_5_le_1_0": "0.5–1.0 kWh",
    "actual_gt_1_0": ">1.0 kWh",
}

METRICS = ["mae", "mse", "smape"]


# =============================================================================
# Visual style
# =============================================================================

NAVY = "#1C2A3A"
GRID = "#E2E6EC"
AXIS = "#666666"

BIN_COLOURS = [
    "#A9C2DA",
    "#6D93B8",
    "#3B6FA0",
    "#1C2A3A",
]


# =============================================================================
# File resolution
# =============================================================================

def resolve_group_bin_file(table_dir: Path | None) -> Path:
    """
    Locate the formal Load range analysis group-bin CSV.

    Priority:
      1) --table-dir supplied by user
      2) reproduced outputs/tables directory
      3) recursive search under <ERP_PROJECT_ROOT>
    """
    preferred_dirs = []

    if table_dir is not None:
        preferred_dirs.append(Path(table_dir).expanduser())

    preferred_dirs.extend([
        DEFAULT_OUTPUT_TABLE_DIR,
    ])

    for directory in preferred_dirs:
        candidate = directory / GROUP_BIN_FILE
        if candidate.is_file():
            print(f"[INPUT] {candidate}")
            return candidate

    matches = sorted(
        p for p in ERP_ROOT.rglob(GROUP_BIN_FILE)
        if p.is_file()
    )

    if len(matches) == 1:
        print(f"[INPUT] {matches[0]}")
        return matches[0]

    if len(matches) == 0:
        checked = "\n".join(
            f"  {d / GROUP_BIN_FILE}"
            for d in preferred_dirs
        )
        raise FileNotFoundError(
            f"Could not find {GROUP_BIN_FILE!r}.\n\n"
            f"Checked preferred locations:\n{checked}\n\n"
            f"Also searched recursively under:\n  {ERP_ROOT}\n\n"
            "Run the load-range analysis first or rerun with:\n"
            '  --table-dir "/path/to/the/formal/LoadRangeAnalysis/table/directory"'
        )


    formatted = "\n".join(
        f"  [{i+1}] {p}"
        for i, p in enumerate(matches)
    )
    raise RuntimeError(
        f"Multiple copies of {GROUP_BIN_FILE!r} were found.\n"
        "The script will not guess which copy is canonical.\n\n"
        f"Candidates:\n{formatted}\n\n"
        "Rerun with --table-dir pointing to the formal Load range analysis table directory."
    )


# =============================================================================
# Data loading and validation
# =============================================================================

def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(
            f"{label} is missing required columns: {missing}\n"
            f"Available columns:\n{list(df.columns)}"
        )


def normalise_group_value(value):
    if pd.isna(value):
        return np.nan

    s = str(value).strip().upper()

    for token in ["GROUP", "G", "_", " "]:
        s = s.replace(token, "")

    try:
        return int(float(s))
    except Exception:
        return np.nan


def load_load_range_analysis(table_dir: Path | None):
    path = resolve_group_bin_file(table_dir)
    df = pd.read_csv(path)

    require_columns(
        df,
        [
            "comparison_id",
            "population",
            "assigned_group",
            "raw_bin",
            "metric",
            "mean_meter_row_share",
        ],
        GROUP_BIN_FILE,
    )

    ld = df[df["comparison_id"].eq(COMPARISON_ID)].copy()

    if ld.empty:
        available = sorted(
            df["comparison_id"].dropna().astype(str).unique()
        )
        raise ValueError(
            f"No rows found for comparison_id={COMPARISON_ID!r}.\n"
            f"Available comparison_id values: {available}"
        )

    ld["_group_num"] = ld["assigned_group"].map(normalise_group_value)

    overall = ld[
        ld["population"].astype(str).str.strip().eq(OVERALL_POPULATION)
        | ld["_group_num"].eq(0)
    ].copy()

    groups = ld[
        ld["_group_num"].isin(GROUP_ORDER)
    ].copy()

    overall = overall[
        overall["raw_bin"].isin(RAW_BINS)
        & overall["metric"].isin(METRICS)
    ].copy()

    groups = groups[
        groups["raw_bin"].isin(RAW_BINS)
        & groups["metric"].isin(METRICS)
    ].copy()

    if overall.empty:
        raise ValueError(
            "No Overall CER rows were found inside the formal group-bin CSV."
        )

    if groups.empty:
        raise ValueError(
            "No G1-G4 rows were found inside the formal group-bin CSV."
        )

    return path, groups, overall


# =============================================================================
# Extract load-composition data
# =============================================================================

def get_group_composition(groups: pd.DataFrame) -> pd.DataFrame:
    """
    mean_meter_row_share is not metric-specific, so the same load-composition
    shares should be repeated for MAE, MSE and sMAPE rows.

    This function verifies that they agree before using one formal copy.
    """
    copies = {}

    for metric in METRICS:
        sub = groups[
            groups["metric"].eq(metric)
        ][
            ["_group_num", "raw_bin", "mean_meter_row_share"]
        ].copy()

        grouped = (
            sub.groupby(
                ["_group_num", "raw_bin"],
                as_index=False
            )["mean_meter_row_share"]
            .mean()
        )

        pivot = (
            grouped.pivot(
                index="_group_num",
                columns="raw_bin",
                values="mean_meter_row_share",
            )
            .reindex(
                index=GROUP_ORDER,
                columns=RAW_BINS
            )
            .astype(float)
        )

        copies[metric] = pivot

    for metric in ["mse", "smape"]:
        if not np.allclose(
            copies["mae"].to_numpy(),
            copies[metric].to_numpy(),
            atol=1e-10,
            rtol=0,
        ):
            raise ValueError(
                f"Group load-composition shares differ between MAE and {metric} rows."
            )

    composition = copies["mae"]

    sums = composition.sum(axis=1)

    if not np.allclose(
        sums.to_numpy(),
        1.0,
        atol=0.01,
        rtol=0
    ):
        raise ValueError(
            "One or more group load-composition rows do not sum to approximately 1.\n"
            f"{sums}"
        )

    return composition


def get_common_reference(overall: pd.DataFrame) -> pd.Series:
    """
    Extract the single common load distribution used in standardisation.

    The same reference load distribution is applied to all groups when
    calculating their common-composition standardised gains.
    """
    copies = {}

    for metric in METRICS:
        sub = overall[
            overall["metric"].eq(metric)
        ].copy()

        vals = (
            sub.groupby("raw_bin")["mean_meter_row_share"]
            .mean()
            .reindex(RAW_BINS)
            .astype(float)
        )

        copies[metric] = vals

    for metric in ["mse", "smape"]:
        if not np.allclose(
            copies["mae"].to_numpy(),
            copies[metric].to_numpy(),
            atol=1e-10,
            rtol=0,
        ):
            raise ValueError(
                f"Overall CER load shares differ between MAE and {metric} rows."
            )

    common = copies["mae"]

    if not np.isclose(
        common.sum(),
        1.0,
        atol=0.01
    ):
        raise ValueError(
            f"Common-composition weights sum to {common.sum():.6f}, "
            "not approximately 1."
        )

    return common


# =============================================================================
# Plot
# =============================================================================

def plot_group_load_composition(
    composition: pd.DataFrame,
    common: pd.Series,
) -> plt.Figure:

    fig, ax = plt.subplots(
        figsize=(12.5, 6.7)
    )

    # Leave a visual gap before the common composition reference.
    x = np.array([
        0.0,
        1.2,
        2.4,
        3.6,
        5.55,
    ])

    xlabels = [
        "G1",
        "G2",
        "G3",
        "G4",
        "Common\ncomposition",
    ]

    plot_matrix = np.vstack([
        composition.loc[1, RAW_BINS].to_numpy(),
        composition.loc[2, RAW_BINS].to_numpy(),
        composition.loc[3, RAW_BINS].to_numpy(),
        composition.loc[4, RAW_BINS].to_numpy(),
        common.reindex(RAW_BINS).to_numpy(),
    ])

    bottom = np.zeros(5)

    for j, raw_bin in enumerate(RAW_BINS):
        heights = 100.0 * plot_matrix[:, j]

        bars = ax.bar(
            x,
            heights,
            bottom=bottom,
            width=0.80,
            color=BIN_COLOURS[j],
            edgecolor="white",
            linewidth=0.8,
            label=BIN_DISPLAY[raw_bin],
        )

        for rect, h, base in zip(
            bars,
            heights,
            bottom
        ):
            if h >= 5.0:
                label_colour = (
                    "white"
                    if j >= 2
                    else NAVY
                )

                ax.text(
                    rect.get_x()
                    + rect.get_width() / 2,
                    base + h / 2,
                    f"{h:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color=label_colour,
                )

        bottom += heights

    # Visually separate the standardisation reference from G1-G4.
    separator_x = (
        x[3] + x[4]
    ) / 2

    ax.axvline(
        separator_x,
        color="#C8D0D8",
        linestyle="--",
        linewidth=1.2,
    )

    # Final agreed wording.
    ax.text(
        x[4],
        103.0,
        "Common composition weights\n"
        "Reference load distribution for all groups",
        ha="center",
        va="bottom",
        fontsize=10.2,
        color="#555555",
    )

    ax.set_title(
        "Observed load composition by group (h = 12)",
        fontsize=15,
        fontweight="bold",
        color=NAVY,
        pad=15,
    )

    ax.set_ylabel(
        "Share of h = 12 test observations (%)",
        fontsize=11,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        xlabels,
        fontsize=10.5,
    )

    ax.set_ylim(
        0,
        111
    )

    ax.set_yticks(
        np.arange(
            0,
            101,
            20
        )
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)

    ax.grid(
        axis="y",
        color=GRID,
        linewidth=0.8,
    )

    ax.set_axisbelow(True)

    ax.legend(
        title="Observed load range",
        ncol=4,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        fontsize=9.5,
        title_fontsize=10,
    )

    fig.tight_layout(
        rect=[
            0,
            0.06,
            1,
            1
        ]
    )

    return fig


# =============================================================================
# Save
# =============================================================================

def save_figure(
    fig: plt.Figure,
    output_dir: Path,
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    stem = (
        "figure_4_3_ld_group_load_composition"
    )

    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(f"[SAVED] {png}")
    print(f"[SAVED] {pdf}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create the Section 4.3 observed-load-composition figure "
            "from the formal Load range analysis package."
        )
    )

    parser.add_argument(
        "--table-dir",
        type=Path,
        default=None,
        help=(
            "Optional path to the formal Load range analysis table directory."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Output directory for PNG and PDF."
        ),
    )

    args = parser.parse_args()

    source_path, groups, overall = load_load_range_analysis(
        args.table_dir
    )

    composition = get_group_composition(
        groups
    )

    common = get_common_reference(
        overall
    )

    print("\n" + "=" * 80)
    print("FORMAL LOAD RANGE ANALYSIS INPUT")
    print("=" * 80)
    print(source_path)

    print("\nObserved load composition by group (%)")
    print(
        (
            100.0 * composition
        )
        .rename(
            index=GROUP_LABEL
        )
        .rename(
            columns=BIN_DISPLAY
        )
        .to_string(
            float_format=lambda x: f"{x:.2f}"
        )
    )

    print("\nCommon composition weights (%)")
    print(
        (
            100.0 * common
        )
        .rename(
            index=BIN_DISPLAY
        )
        .to_string(
            float_format=lambda x: f"{x:.2f}"
        )
    )

    fig = plot_group_load_composition(
        composition,
        common,
    )

    save_figure(
        fig,
        args.output_dir,
    )

    print("\n[DONE]")
    print(
        f"Output directory: "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()
