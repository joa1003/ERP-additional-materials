#!/usr/bin/env python3
"""Render submitted dissertation Figure 4.2 components from reproduced source-relevance data."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

GROUP_COLOURS = {1: "#3B6FA0", 2: "#3F8F86", 3: "#C6902F", 4: "#6B5B95"}
GROUP_LABELS = {1: "G1", 2: "G2", 3: "G3", 4: "G4"}
LINE_COLOUR = "#B5820B"
ZERO_COLOUR = "#2C6EBD"
GRID = "#E2E6EC"
SPINE = "#9AA4B2"
TEXT = "#1C2A3A"
METRICS = [
    ("mae", "MAE", "MAE L−D Gain (kWh)"),
    ("rmse", "RMSE", "RMSE L−D Gain (kWh)"),
    ("smape", "sMAPE", "sMAPE L−D Gain (pp)"),
]


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def quartile_medians(frame: pd.DataFrame, x: str) -> pd.DataFrame:
    work = frame[[x, "gain_source_transfer_absolute"]].dropna().copy()
    work["quartile"] = pd.qcut(work[x].rank(method="first"), 4, labels=False)
    return (work.groupby("quartile", observed=True)
            .agg(x=(x, "median"), y=("gain_source_transfer_absolute", "median"))
            .reset_index().sort_values("quartile"))


def style(ax: plt.Axes) -> None:
    ax.grid(True, color=GRID, alpha=0.72, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE)
    ax.spines["bottom"].set_color(SPINE)
    ax.tick_params(axis="both", labelsize=10, colors="#2A2F3A")


def render_one(data: pd.DataFrame, xcol: str, xlabel: str, title: str, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.02))
    fig.suptitle(title, fontsize=18, fontweight="bold", color=TEXT, y=0.985)

    for ax, (criterion, metric_title, ylabel) in zip(axes, METRICS):
        sub = data[data["criterion"].eq(criterion)].copy()
        ensure(len(sub) == 929, f"Figure 4.2 {criterion} panel expected 929 households; found {len(sub)}")
        for group in [1, 2, 3, 4]:
            g = sub[sub["assigned_group"].eq(group)]
            ax.scatter(
                g[xcol], g["gain_source_transfer_absolute"],
                s=18, alpha=0.34, color=GROUP_COLOURS[group],
                edgecolors="none", zorder=3,
            )
        med = quartile_medians(sub, xcol)
        ax.plot(
            med["x"], med["y"], marker="o", markersize=6.2,
            linewidth=2.6, color=LINE_COLOUR, zorder=5,
        )
        ax.axhline(0.0, color=ZERO_COLOUR, linewidth=1.25, linestyle="--", zorder=1)
        ax.set_title(metric_title, fontsize=15, fontweight="bold", loc="left", color=TEXT)
        ax.set_xlabel(xlabel, fontsize=11.5, color=TEXT)
        ax.set_ylabel(ylabel, fontsize=11.5, color=TEXT)
        style(ax)

    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=GROUP_COLOURS[g],
               label=GROUP_LABELS[g], markersize=7) for g in [1, 2, 3, 4]
    ]
    handles.append(
        Line2D([0], [0], marker="o", color=LINE_COLOUR, linewidth=2.6,
               label="Overall household quartile median", markersize=6)
    )
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 0.015), fontsize=11)
    fig.tight_layout(rect=[0.02, 0.10, 1, 0.92])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    input_path = root / "outputs/tables/local_source_support_complete_cross_strategy_forecasting_aligned_source_support/local_source_support_cross_strategy_profile_support_outcome_dataset.csv"
    ensure(input_path.is_file(), f"Missing generated source-relevance dataset: {input_path}")
    data = pd.read_csv(input_path)
    required = {"lead", "criterion", "assigned_group", "nearest_dtw_distance",
                "source_knn10_mean_rmse_z", "gain_source_transfer_absolute"}
    ensure(required.issubset(data.columns), f"Figure 4.2 input missing columns: {sorted(required-set(data.columns))}")
    data = data[data["lead"].eq(12)].copy()
    ensure(len(data) == 929 * 3, f"Figure 4.2 expected 2,787 h=12 rows; found {len(data)}")

    outdir = root / "outputs/figures/report"
    render_one(
        data,
        "nearest_dtw_distance",
        "Nearest DTW distance to assigned source prototype",
        "Source similarity versus transfer gain (h = 12)",
        outdir / "figure_4_2_source_similarity.png",
    )
    render_one(
        data,
        "source_knn10_mean_rmse_z",
        "Mean distance to 10 nearest source households",
        "Local source support versus transfer gain (h = 12)",
        outdir / "figure_4_2_local_source_support.png",
    )
    print("FIGURE 4.2: PASS")
    print(outdir / "figure_4_2_source_similarity.png")
    print(outdir / "figure_4_2_local_source_support.png")


if __name__ == "__main__":
    main()
