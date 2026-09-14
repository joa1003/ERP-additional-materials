#!/usr/bin/env python3
"""Generate the dissertation K-selection and final K=4 profile figures.

Inputs are the formal Profile clustering outputs under the reproduction workspace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


# Colour blind friendly palette.
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
DARK = "#1F2D3D"
SLATE = "#52616B"
PALE_BLUE = "#A9D6E5"
PALE_ORANGE = "#F4B183"
LIGHT_GRID = "#D9E1E8"
PLOT_FACE = "#FBFCFD"
LEGEND_FACE = "#FFFFFF"

PROFILE_COLOURS = [BLUE, ORANGE, GREEN, PURPLE]

ANALYSIS_ID = "profile_clustering_dissertation_figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
        help="ERP project root, for example <ERP_PROJECT_ROOT>",
    )
    return parser.parse_args()


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    ensure(not missing, f"{label} is missing required columns: {missing}")


def save_figure(fig: plt.Figure, png_path: Path, pdf_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def set_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12.5,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
            "axes.linewidth": 0.9,
            "lines.solid_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(PLOT_FACE)
    ax.grid(axis="y", color=LIGHT_GRID, linewidth=0.8, alpha=0.65)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#7A8793")
        spine.set_linewidth(0.8)


def annotate_candidate_points(ax: plt.Axes, x: np.ndarray, y: np.ndarray, selected_values=(4, 5)) -> None:
    for k, value in zip(x, y):
        if int(k) == selected_values[0]:
            colour = BLUE
            size = 72
            zorder = 5
        elif int(k) == selected_values[1]:
            colour = ORANGE
            size = 72
            zorder = 5
        else:
            colour = SLATE
            size = 40
            zorder = 4

        ax.scatter([k], [value], s=size, color=colour, edgecolor="white", linewidth=1.0, zorder=zorder)


def plot_candidate_k(k_table: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    frame = k_table.sort_values("k").copy()
    ensure(frame["k"].tolist() == list(range(2, 9)), "Candidate K table must contain K = 2 to K = 8")

    ks = frame["k"].to_numpy(dtype=int)
    inertia = frame["inertia_mean"].to_numpy(dtype=float)
    inertia_sd = frame["inertia_sample_std"].to_numpy(dtype=float)
    reduction = frame["relative_inertia_reduction_from_previous_k"].to_numpy(dtype=float)
    silhouette = frame["silhouette_mean"].to_numpy(dtype=float)
    silhouette_sd = frame["silhouette_sample_std"].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)
    fig.suptitle("Candidate values of K for source profile clustering")

    # Panel A
    ax = axes[0]
    style_axis(ax)
    ax.errorbar(
        ks,
        inertia,
        yerr=inertia_sd,
        color=DARK,
        linewidth=2.0,
        marker=None,
        capsize=3.5,
        elinewidth=1.0,
        zorder=3,
    )
    annotate_candidate_points(ax, ks, inertia)

    for k, y_value, reduction_value in zip(ks, inertia, reduction):
        if np.isfinite(reduction_value):
            ax.annotate(
                f"−{100 * reduction_value:.1f}%",
                xy=(k, y_value),
                xytext=(0, 11),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                color=DARK,
            )

    ax.set_title("DTW inertia")
    ax.set_xlabel("Number of groups, K")
    ax.set_ylabel("Mean DTW inertia")
    ax.set_xticks(ks)
    ax.text(
        0.02,
        0.04,
        "Percent labels show the reduction from the preceding value of K.",
        transform=ax.transAxes,
        fontsize=9,
        color=SLATE,
    )

    # Panel B
    ax = axes[1]
    style_axis(ax)
    ax.errorbar(
        ks,
        silhouette,
        yerr=silhouette_sd,
        color=DARK,
        linewidth=2.0,
        marker=None,
        capsize=3.5,
        elinewidth=1.0,
        zorder=3,
    )
    annotate_candidate_points(ax, ks, silhouette)
    ax.set_title("Full sample DTW silhouette")
    ax.set_xlabel("Number of groups, K")
    ax.set_ylabel("Mean silhouette")
    ax.set_xticks(ks)

    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor="white", markersize=8, label="K = 4"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=ORANGE, markeredgecolor="white", markersize=8, label="K = 5"),
    ]
    ax.legend(
        handles=handles,
        title="Focused comparison",
        loc="upper right",
        frameon=True,
        facecolor=LEGEND_FACE,
        edgecolor="#9AA7B2",
        framealpha=0.97,
    )

    save_figure(fig, png_path, pdf_path)


def paired_panel(ax: plt.Axes, frame: pd.DataFrame, left_column: str, right_column: str, title: str, percent: bool, win_text: str) -> None:
    style_axis(ax)

    left_values = frame[left_column].to_numpy(dtype=float)
    right_values = frame[right_column].to_numpy(dtype=float)

    for left_value, right_value in zip(left_values, right_values):
        colour = PALE_BLUE if left_value >= right_value else PALE_ORANGE
        ax.plot([0, 1], [left_value, right_value], color=colour, linewidth=1.25, alpha=0.95, zorder=1)
        ax.scatter([0], [left_value], s=18, color=BLUE, alpha=0.85, zorder=2)
        ax.scatter([1], [right_value], s=18, color=ORANGE, alpha=0.85, zorder=2)

    means = [left_values.mean(), right_values.mean()]
    ax.plot([0, 1], means, color=DARK, linewidth=2.8, zorder=4)
    ax.scatter([0], [means[0]], s=78, color=BLUE, edgecolor="white", linewidth=1.1, zorder=5)
    ax.scatter([1], [means[1]], s=78, color=ORANGE, edgecolor="white", linewidth=1.1, zorder=5)

    # Leave enough horizontal space for labels outside the mean markers,
    # while keeping the labels fully inside the panel frame.
    ax.set_xlim(-0.28, 1.28)
    ax.set_xticks([0, 1], ["K = 4", "K = 5"])
    ax.set_title(title)

    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    # Value labels placed left and right of the means so they do not overlap with the dark mean line.
    if percent:
        left_label = f"{100 * means[0]:.1f}%"
        right_label = f"{100 * means[1]:.1f}%"
    else:
        left_label = f"{means[0]:.3f}"
        right_label = f"{means[1]:.3f}"

    value_box = dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#B8C2CC", alpha=0.98)

    # Place each value directly beside its mean marker at the same y position.
    # The K = 4 label remains on the LEFT of the blue point.
    # The K = 5 label remains on the RIGHT of the orange point.
    # Fixed data-coordinate positions and wider x limits keep both boxes away
    # from the y axis and fully inside the panel frame.
    ax.text(
        -0.055,
        means[0],
        left_label,
        ha="right",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=DARK,
        bbox=value_box,
        zorder=6,
        clip_on=True,
    )
    ax.text(
        1.055,
        means[1],
        right_label,
        ha="left",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=DARK,
        bbox=value_box,
        zorder=6,
        clip_on=True,
    )

    ax.text(
        0.03,
        0.04,
        win_text,
        transform=ax.transAxes,
        fontsize=9.5,
        color=DARK,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#B8C2CC", "alpha": 0.96},
    )


def plot_k4_k5_paired_metrics(transitions: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    ensure(transitions["seed"].nunique() == 20, "Expected 20 paired initialisations")
    ensure(len(transitions) == 20, "Expected exactly one row per paired seed")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), constrained_layout=True)
    fig.suptitle("K = 4 and K = 5 comparison across 20 paired initialisations")

    paired_panel(
        axes[0],
        transitions,
        "k4_silhouette",
        "k5_silhouette",
        "Full sample DTW silhouette",
        False,
        "K = 4 is higher in 19 of 20 runs",
    )

    paired_panel(
        axes[1],
        transitions,
        "k4_minimum_group_share",
        "k5_minimum_group_share",
        "Minimum group share",
        True,
        "K = 4 is higher in 19 of 20 runs",
    )

    save_figure(fig, png_path, pdf_path)


def plot_final_prototypes(prototypes: pd.DataFrame, summary: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    require_columns(
        prototypes,
        ["k", "seed", "profile_group", "local_slot", "prototype_zscore", "local_time"],
        "Final prototype table",
    )
    require_columns(
        summary,
        ["k", "seed", "profile_group", "all_assigned_n", "peak_time"],
        "Final profile group table",
    )

    ensure(prototypes["k"].nunique() == 1 and int(prototypes["k"].iloc[0]) == 4, "Final prototype table must contain K = 4")
    ensure(summary["k"].nunique() == 1 and int(summary["k"].iloc[0]) == 4, "Final summary table must contain K = 4")
    ensure(prototypes["seed"].nunique() == 1 and int(prototypes["seed"].iloc[0]) == 71, "Final prototype run must use seed 71")
    ensure(summary["seed"].nunique() == 1 and int(summary["seed"].iloc[0]) == 71, "Final summary run must use seed 71")
    ensure(sorted(prototypes["profile_group"].unique().tolist()) == [1, 2, 3, 4], "Expected profile groups 1 to 4")

    fig, ax = plt.subplots(figsize=(12.2, 6.1), constrained_layout=True)
    ax.set_facecolor(PLOT_FACE)

    for group in [1, 2, 3, 4]:
        frame = prototypes[prototypes["profile_group"] == group].sort_values("local_slot")
        row = summary[summary["profile_group"] == group].iloc[0]
        label = f"Group {group}: peak at {row['peak_time']}, n = {int(row['all_assigned_n'])}"
        ax.plot(
            frame["local_slot"],
            frame["prototype_zscore"],
            linewidth=2.5,
            color=PROFILE_COLOURS[group - 1],
            label=label,
        )

    ax.axhline(0, color="#6C757D", linewidth=1.0, linestyle=(0, (4, 3)), zorder=0)
    ax.set_title("Final K = 4 daily load profiles defined with the source dataset")
    ax.set_ylabel("Standardised daily load shape (z score)")
    ax.set_xlabel("London local time")

    ticks = [1, 7, 13, 19, 25, 31, 37, 43, 48]
    time_map = prototypes[["local_slot", "local_time"]].drop_duplicates().set_index("local_slot")["local_time"].to_dict()
    ax.set_xticks(ticks, [time_map[slot] for slot in ticks])

    ax.grid(axis="y", color=LIGHT_GRID, linewidth=0.8, alpha=0.65)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#7A8793")
        spine.set_linewidth(0.8)

    legend = ax.legend(
        loc="lower right",
        ncol=2,
        frameon=True,
        fancybox=True,
        facecolor=LEGEND_FACE,
        edgecolor="#8C99A5",
        framealpha=0.96,
        borderpad=0.8,
        columnspacing=1.2,
        handlelength=3.0,
    )
    legend.get_frame().set_linewidth(0.9)

    save_figure(fig, png_path, pdf_path)


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()

    print(f"Running script : {Path(__file__).resolve()}")
    print(f"Analysis ID    : {ANALYSIS_ID}")
    print(f"Project root   : {root}")
    print("Label layout   : beside mean points at the same y position")
    print("Label positions: K = 4 left of blue point; K = 5 right of orange point")
    print("Panel x limits : -0.28 to 1.28")

    broad_table = root / "outputs" / "tables" / "profile_clustering_clustering_final_k_selection" / "profile_clustering_candidate_k_selection_table.csv"
    decision_dir = root / "outputs" / "tables" / "profile_clustering_clustering_final_decision_dissertation"
    figure_dir = root / "outputs" / "figures" / "profile_clustering_clustering_final_decision_dissertation"
    metadata_dir = root / "outputs" / "metadata" / "profile_clustering_clustering_final_decision_dissertation"

    transition_file = decision_dir / "profile_clustering_k45_transition_by_seed.csv"
    structural_file = decision_dir / "profile_clustering_k45_structural_decision_evidence.csv"
    prototype_file = decision_dir / "profile_clustering_final_selected_source_prototypes.csv"
    profile_summary_file = decision_dir / "profile_clustering_final_selected_profile_groups_dissertation.csv"

    input_paths = [broad_table, transition_file, structural_file, prototype_file, profile_summary_file]
    for path in input_paths:
        ensure(path.is_file(), f"Required input does not exist: {path}")

    k_table = pd.read_csv(broad_table)
    transitions = pd.read_csv(transition_file)
    structural_frame = pd.read_csv(structural_file)
    prototypes = pd.read_csv(prototype_file)
    profile_summary = pd.read_csv(profile_summary_file)

    require_columns(
        k_table,
        ["k", "inertia_mean", "inertia_sample_std", "relative_inertia_reduction_from_previous_k", "silhouette_mean", "silhouette_sample_std"],
        "Candidate K table",
    )
    require_columns(
        transitions,
        ["seed", "k4_silhouette", "k5_silhouette", "k4_minimum_group_share", "k5_minimum_group_share", "k4_inertia", "k5_inertia"],
        "K = 4 and K = 5 transition table",
    )
    require_columns(structural_frame, ["evidence", "value"], "Structural evidence table")

    structural = structural_frame.set_index("evidence")["value"].to_dict()
    required_structural = ["median_extra_group_share", "parent_mode_share", "median_origin_purity", "peak_band_mode_share", "recommended_k"]
    missing_structural = [key for key in required_structural if key not in structural]
    ensure(not missing_structural, f"Structural evidence is missing: {missing_structural}")
    ensure(int(float(structural["recommended_k"])) == 4, "The formal structural evidence does not recommend K = 4")

    ensure(transitions["seed"].nunique() == 20, "Expected 20 unique paired seeds")
    ensure(len(transitions) == 20, "Expected 20 rows in the paired transition table")

    set_publication_style()

    plot_candidate_k(
        k_table,
        figure_dir / "profile_clustering_candidate_k_selection_evidence_dissertation.png",
        figure_dir / "profile_clustering_candidate_k_selection_evidence_dissertation.pdf",
    )

    plot_k4_k5_paired_metrics(
        transitions,
        figure_dir / "profile_clustering_k4_vs_k5_paired_metric_comparison_dissertation.png",
        figure_dir / "profile_clustering_k4_vs_k5_paired_metric_comparison_dissertation.pdf",
    )


    plot_final_prototypes(
        prototypes,
        profile_summary,
        figure_dir / "profile_clustering_final_selected_source_prototypes_dissertation.png",
        figure_dir / "profile_clustering_final_selected_source_prototypes_dissertation.pdf",
    )


    metadata_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "status": "PASS",
        "analysis_id": ANALYSIS_ID,
        "script_path": str(Path(__file__).resolve()),
        "project_root": str(root),
        "selected_k": 4,
        "selected_seed": 71,
        "image_editing_used": False,
        "manual_post_processing_used": False,
        "paired_initialisations": int(transitions["seed"].nunique()),
        "source_inputs": {str(path.relative_to(root)): sha256(path) for path in input_paths},
        "outputs": [
            str((figure_dir / "profile_clustering_candidate_k_selection_evidence_dissertation.png").relative_to(root)),
            str((figure_dir / "profile_clustering_candidate_k_selection_evidence_dissertation.pdf").relative_to(root)),
            str((figure_dir / "profile_clustering_k4_vs_k5_paired_metric_comparison_dissertation.png").relative_to(root)),
            str((figure_dir / "profile_clustering_k4_vs_k5_paired_metric_comparison_dissertation.pdf").relative_to(root)),
            str((figure_dir / "profile_clustering_final_selected_source_prototypes_dissertation.png").relative_to(root)),
            str((figure_dir / "profile_clustering_final_selected_source_prototypes_dissertation.pdf").relative_to(root)),
        ],
    }
    status_path = metadata_dir / "profile_clustering_dissertation_figures_status.json"
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    expected_outputs = [root / relative_path for relative_path in status["outputs"]]
    missing_outputs = [str(path) for path in expected_outputs if not path.is_file()]
    ensure(not missing_outputs, f"Expected outputs were not created: {missing_outputs}")

    print("=" * 96)
    print("PROFILE CLUSTERING DISSERTATION FIGURES: PASS")
    print("=" * 96)
    print("Selected K       : 4")
    print("Selected seed    : 71")
    print("Image editing    : NO")
    print("Post processing  : NO")
    print("Figures written  :", figure_dir)
    print("Paired figure    :", figure_dir / "profile_clustering_k4_vs_k5_paired_metric_comparison_dissertation.png")
    print("Final profiles   :", figure_dir / "profile_clustering_final_selected_source_prototypes_dissertation.png")
    print("Status metadata  :", status_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
