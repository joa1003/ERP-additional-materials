import os as _erp_os
from pathlib import Path as _ERPPath
_ERP_PROJECT_ROOT = _ERPPath(_erp_os.environ.get("ERP_PROJECT_ROOT", ".")).expanduser().resolve()
"""Render final report Figure 4.5 from reproduced load-range analysis outputs.

Figure 4.5 is the cumulative contribution of the four observed load ranges to the
L−D net gain at h = 12. Plotted values are read from the reproduced cross-strategy
load-range analysis table; no empirical result is hardcoded.
"""


import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =============================================================================
# Paths
# =============================================================================
ERP_ROOT = Path(f"{_ERP_PROJECT_ROOT}")
GROUP_BIN_FILE = "load_range_analysis_h12_group_bin_conditional_gain_and_contribution.csv"
DEFAULT_OUTPUT_DIR = Path(
    f"{_ERP_PROJECT_ROOT}/outputs/figures/"
    "load_range_analysis_complete_cross_strategy_raw_load_composition_standardised_gain_decomposition"
)

POPULATION = "Overall CER"


# =============================================================================
# Comparison definitions
# =============================================================================
COMPARISONS: Dict[str, Dict[str, str]] = {
    "ld": {
        "comparison_id": "source_transfer",
        "label": "Direct Transfer over Scratch Limited",
        "stem": "ld",
    },
    "dt": {
        "comparison_id": "fine_tuning",
        "label": "Fine Tuning over Direct Transfer",
        "stem": "dt",
    },
    "lf": {
        "comparison_id": "full_data",
        "label": "Scratch Full over Scratch Limited",
        "stem": "lf",
    },
    "fd": {
        "comparison_id": "direct_vs_full",
        "label": "Direct Transfer over Scratch Full",
        "stem": "fd",
    },
}

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
METRIC_TITLE = {"mae": "MAE", "mse": "MSE", "smape": "sMAPE"}
METRIC_UNIT = {"mae": "kWh", "mse": "kWh²", "smape": "pp"}
METRIC_FMT = {"mae": ".4f", "mse": ".4f", "smape": ".2f"}


# =============================================================================
# Visual language: aligned with existing Section 4.4 figures
# =============================================================================
BIN_COLORS = ["#A9C2DA", "#6D93B8", "#3B6FA0", "#1C2A3A"]
GOLD = "#B5820B"
ZERO_BLUE = "#2C6EBD"
GRID = "#E2E6EC"
NAVY = "#1C2A3A"
AXIS = "#666666"


# =============================================================================
# Input resolution and validation
# =============================================================================
def resolve_table(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Table not found: {path}")
        return path

    preferred = ERP_ROOT / "outputs" / "tables"
    candidates: List[Path] = []

    if preferred.is_dir():
        candidates.extend(sorted(p for p in preferred.rglob(GROUP_BIN_FILE) if p.is_file()))

    if ERP_ROOT.is_dir():
        for p in sorted(ERP_ROOT.rglob(GROUP_BIN_FILE)):
            if p.is_file() and p not in candidates:
                candidates.append(p)

    if not candidates:
        raise FileNotFoundError(
            f"Could not find {GROUP_BIN_FILE} under {ERP_ROOT}. "
            "Use --table to provide the exact file path."
        )

    if len(candidates) > 1:
        print("Multiple candidate tables found:")
        for p in candidates:
            print(f"  {p}")
        print("Using the first candidate listed above. Use --table if this is not the formal file.")

    return candidates[0]


def require_columns(df: pd.DataFrame) -> None:
    required = {
        "comparison_id",
        "population",
        "raw_bin",
        "metric",
        "mean_meter_row_share",
        "exposure_weighted_conditional_gain",
        "observed_gain_contribution",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    require_columns(df)
    return df


def comparison_rows(df: pd.DataFrame, key: str) -> pd.DataFrame:
    meta = COMPARISONS[key]
    out = df[
        df["comparison_id"].eq(meta["comparison_id"])
        & df["population"].eq(POPULATION)
    ].copy()

    if out.empty:
        available = sorted(df["comparison_id"].dropna().astype(str).unique().tolist())
        raise ValueError(
            f"No rows found for comparison_id={meta['comparison_id']!r}, "
            f"population={POPULATION!r}. Available comparison_id values: {available}"
        )

    for metric in METRICS:
        sub = out[out["metric"].eq(metric)]
        found = set(sub["raw_bin"].astype(str))
        missing = [rb for rb in RAW_BINS if rb not in found]
        if missing:
            raise ValueError(f"{key.upper()} {metric}: missing raw bins {missing}")

    return out


def ordered_metric_rows(comp_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    sub = comp_df[comp_df["metric"].eq(metric)].copy()
    sub["raw_bin"] = pd.Categorical(sub["raw_bin"], RAW_BINS, ordered=True)
    return sub.sort_values("raw_bin").set_index("raw_bin").reindex(RAW_BINS)


def xlabels_with_shares(comp_df: pd.DataFrame) -> List[str]:
    sub = ordered_metric_rows(comp_df, "mae")
    shares = sub["mean_meter_row_share"].astype(float).to_numpy()
    return [
        f"{BIN_DISPLAY[rb]}\n({100.0 * share:.0f}%)"
        for rb, share in zip(RAW_BINS, shares)
    ]


# =============================================================================
# Plot helpers
# =============================================================================
def clean_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def add_value_labels(ax, xs, ys, metric: str) -> None:
    y_min, y_max = min(ys), max(ys)
    span = y_max - y_min
    if span == 0:
        span = max(abs(y_max), 1.0)
    offset = span * 0.10

    for x, y in zip(xs, ys):
        dy = offset if y >= 0 else -offset
        va = "bottom" if y >= 0 else "top"
        ax.text(
            x,
            y + dy,
            format(float(y), METRIC_FMT[metric]),
            ha="center",
            va=va,
            fontsize=8.5,
            color=NAVY,
        )


# =============================================================================
# Figure 1: Appendix J.2 within-bin gain
# =============================================================================
def plot_cumulative_contribution(comp_df: pd.DataFrame, label: str) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    xlabels = ["Start"] + xlabels_with_shares(comp_df)

    for ax, metric in zip(axes, METRICS):
        sub = ordered_metric_rows(comp_df, metric)
        contrib = sub["observed_gain_contribution"].astype(float).to_numpy()
        cumulative = np.concatenate([[0.0], np.cumsum(contrib)])
        xs = np.arange(5)

        ax.axhline(0, color=ZERO_BLUE, linestyle="--", linewidth=1.1, zorder=1)
        ax.plot(xs, cumulative, color=NAVY, linewidth=2.0, zorder=2)
        ax.scatter([0], [0], s=85, facecolors="white", edgecolors=NAVY,
                   linewidths=1.1, zorder=3)
        ax.scatter(xs[1:-1], cumulative[1:-1], s=95, c=BIN_COLORS[:-1],
                   edgecolors=NAVY, linewidths=1.0, zorder=3)
        ax.scatter([xs[-1]], [cumulative[-1]], s=125, c=GOLD,
                   edgecolors=NAVY, linewidths=1.1, zorder=4)

        yrange = max(cumulative) - min(cumulative)
        if yrange == 0:
            yrange = max(abs(cumulative[-1]), 1.0)

        # Labels show the incremental contribution added at each step.
        for i, c in enumerate(contrib, start=1):
            va = "bottom" if c >= 0 else "top"
            dy = 8 if c >= 0 else -8
            ax.annotate(
                "Δ " + format(float(c), METRIC_FMT[metric]),
                (xs[i], cumulative[i]),
                xytext=(0, dy),
                textcoords="offset points",
                ha="center",
                va=va,
                fontsize=8.2,
                color=NAVY,
            )

        ax.annotate(
            f"Net\n{format(float(cumulative[-1]), METRIC_FMT[metric])} {METRIC_UNIT[metric]}",
            (xs[-1], cumulative[-1]),
            xytext=(26, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=NAVY,
        )

        ax.set_title(METRIC_TITLE[metric], fontsize=13, fontweight="bold", color=NAVY)
        ax.set_ylabel(f"Cumulative contribution ({METRIC_UNIT[metric]})", fontsize=10)
        ax.set_xticks(xs)
        ax.set_xticklabels(xlabels, fontsize=8.4)
        ax.set_xlim(-0.25, 4.72)
        clean_axes(ax)

        ymin, ymax = min(cumulative), max(cumulative)
        span = ymax - ymin
        if span == 0:
            span = max(abs(ymax), 1.0)
        ax.set_ylim(ymin - 0.42 * span, ymax + 0.42 * span)

    legend = [
        Line2D([0], [0], color=NAVY, lw=2.0, label="Running total"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=GOLD,
               markeredgecolor=NAVY, markersize=8, label="Net gain"),
        Line2D([0], [0], color=ZERO_BLUE, linestyle="--", lw=1.1, label="No gain"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.035))
    fig.suptitle(
        f"Cumulative contribution to {label} net gain (h = 12)",
        fontsize=14,
        fontweight="bold",
        color=NAVY,
        y=1.03,
    )
    fig.tight_layout(rect=[0, 0.07, 1, 0.94])
    return fig


# =============================================================================
# Save
# =============================================================================
def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Render final report Figure 4.5 (L−D cumulative load-range contribution).")
    parser.add_argument("--table", default=None, help="Optional exact path to load_range_analysis_h12_group_bin_conditional_gain_and_contribution.csv")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for PNG and PDF files.")
    args = parser.parse_args()

    table_path = resolve_table(args.table)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(table_path)
    comp_df = comparison_rows(df, "ld")
    save(plot_cumulative_contribution(comp_df, COMPARISONS["ld"]["label"]), output_dir, "figure_4_5_ld_cumulative_load_range_contribution")


if __name__ == "__main__":
    main()
