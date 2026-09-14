from __future__ import annotations
import os as _erp_os
from pathlib import Path as _ERPPath
_ERP_PROJECT_ROOT = _ERPPath(_erp_os.environ.get("ERP_PROJECT_ROOT", ".")).expanduser().resolve()

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


DEFAULT_OUTPUT_DIR = Path(
    f"{_ERP_PROJECT_ROOT}/outputs/figures/group_heterogeneity_group_heterogeneity_practical_significance"
)

DEFAULT_INPUT = f"{_ERP_PROJECT_ROOT}/outputs/tables/group_heterogeneity_group_heterogeneity_practical_significance"

GROUP_IDS = [1, 2, 3, 4]
BOX_POSITIONS = [1.0, 2.6, 4.2, 5.8]
GROUP_LABELS: Dict[int, str] = {1: "G1", 2: "G2", 3: "G3", 4: "G4"}

GROUP_COLORS: Dict[int, str] = {
    1: "#3B6FA0",
    2: "#3F8F86",
    3: "#C6902F",
    4: "#6B5B95",
}
GROUP_FILL_COLORS: Dict[int, str] = {
    1: "#DCE8F2",
    2: "#DDEDEA",
    3: "#F5E8C7",
    4: "#E8E2F0",
}

# Figure 4.1 deliberately uses one restrained box colour for all groups.
# Group identity is already carried by the x-axis, so extra group colours would add
# visual meaning that is not needed and could imply an ordinal colour scale.
MAIN_BOX_EDGE_COLOR = "#455C70"
MAIN_BOX_FILL_COLOR = "#DCE4EB"
MAIN_BOX_FILL_ALPHA = 0.72

# Household cloud uses a separate slate blue-grey, deliberately different from
# the grey outlier markers used in Section 4.4 figures.
HOUSEHOLD_POINT_COLOR = "#6F8498"
HOUSEHOLD_POINT_ALPHA = 0.18
HOUSEHOLD_POINT_SIZE = 10
HOUSEHOLD_JITTER_WIDTH = 0.14
HOUSEHOLD_JITTER_SEED = 20260828

TEXT_COLOR = "#1C2A3A"
GRID_COLOR = "#E2E6EC"
ZERO_COLOR = "#2C6EBD"
MEDIAN_COLOR = "#B5820B"
MEAN_EDGE_COLOR = "#1C2A3A"
MEAN_LABEL_COLOR = "#30475E"
WHISKER_COLOR = "#617080"
ANNOTATION_COLOR = "#536273"
EFFECT_BOX_FACE = "#F6F8FA"
EFFECT_BOX_EDGE = "#CDD5DE"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.dpi": 150,
})


def load_group_heterogeneity_meter_dataset(input_path):
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    if path.is_file() and path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.is_dir():
        candidates = [
            path / "outputs/tables/group_heterogeneity_group_heterogeneity_practical_significance/group_heterogeneity_meter_transferability_dataset.csv",
            path / "group_heterogeneity_meter_transferability_dataset.csv",
        ]
        for csv_path in candidates:
            if csv_path.exists():
                return pd.read_csv(csv_path)
        matches = list(path.rglob("group_heterogeneity_meter_transferability_dataset.csv"))
        if len(matches) == 1:
            return pd.read_csv(matches[0])
        if len(matches) > 1:
            raise RuntimeError("Multiple group-heterogeneity datasets found; pass the exact CSV path.")
        raise FileNotFoundError("Could not find group_heterogeneity_meter_transferability_dataset.csv under the reproduced output directory.")
    raise ValueError("Input must be the reproduced group-heterogeneity table directory or CSV file.")


def subset_metric(df, lead=12, criterion="mae"):
    needed = [
        "entity_id", "lead", "criterion", "assigned_group",
        "source_transfer_gain_mean", "source_transfer_gain_relative_pct",
        "cer_scratch_limited_value_mean", "direct_transfer_value_mean",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataset is missing required columns: {missing}")
    sub = df[(df["lead"] == lead) & (df["criterion"] == criterion)].copy()
    if sub.empty:
        raise ValueError(f"No rows found for lead={lead}, criterion={criterion}.")
    return sub


def save_figure(fig, output_png, save_pdf=True):
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    if save_pdf:
        fig.savefig(output_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_png}")
    if save_pdf:
        print(f"Saved: {output_png.with_suffix('.pdf')}")


def apply_common_axis_style(ax):
    ax.grid(True, axis="both", color=GRID_COLOR, alpha=0.7, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#9AA4B2")
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(axis="both", labelsize=11, colors="#2A2F3A")


def get_group_arrays(sub, value_col):
    return [
        sub.loc[sub["assigned_group"] == gid, value_col].dropna().to_numpy()
        for gid in GROUP_IDS
    ]


def _draw_box_with_stats(ax, data, positions, decimals):
    """Draw box plots and put all statistic labels on the right side of each group.

    The labels are deliberately placed to the right of each box rather than on top
    of the marks. This keeps the labels readable when the mean and median are close.
    """
    bp = ax.boxplot(
        data, positions=positions, widths=0.52, patch_artist=True,
        whis=(0, 100), showfliers=False,
        medianprops=dict(color=MEDIAN_COLOR, linewidth=2.4),
        whiskerprops=dict(color=WHISKER_COLOR, linewidth=1.3),
        capprops=dict(color=WHISKER_COLOR, linewidth=1.3),
        boxprops=dict(linewidth=1.7),
    )
    for gid, patch in zip(GROUP_IDS, bp["boxes"]):
        patch.set_facecolor(GROUP_FILL_COLORS[gid])
        patch.set_edgecolor(GROUP_COLORS[gid])

    means = [float(np.nanmean(a)) for a in data]
    ax.scatter(positions, means, marker="D", s=58, facecolors="white",
               edgecolors=MEAN_EDGE_COLOR, linewidths=1.6, zorder=5)

    fmt = f"{{:.{decimals}f}}"
    bbox = dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.88)

    all_vals = np.concatenate([np.asarray(a, dtype=float) for a in data if len(a)])
    y_range = float(np.nanmax(all_vals) - np.nanmin(all_vals)) if len(all_vals) else 1.0
    if y_range <= 0:
        y_range = 1.0
    min_gap = 0.040 * y_range

    for pos, arr, mean_val in zip(positions, data, means):
        med = float(np.nanmedian(arr))
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))

        # All labels are on the right. Mean and median are staggered if close.
        mean_y = mean_val
        med_y = med
        if abs(mean_y - med_y) < min_gap:
            mid = (mean_y + med_y) / 2.0
            mean_y = mid + min_gap / 2.0
            med_y = mid - min_gap / 2.0

        label_x = pos + 0.34
        cap_x = pos + 0.25

        # Mean and median labels are deliberately emphasized.
        # Min and max stay lighter to keep them secondary.
        stat_fontsize = 7.5
        main_stat_fontsize = stat_fontsize + 1.0
        stat_color = WHISKER_COLOR
        ax.text(label_x, med_y, "median " + fmt.format(med), ha="left", va="center",
                fontsize=main_stat_fontsize, color=MEDIAN_COLOR, fontweight="bold", bbox=bbox, zorder=6)
        ax.text(label_x, mean_y, "mean " + fmt.format(mean_val), ha="left", va="center",
                fontsize=main_stat_fontsize, color=MEAN_LABEL_COLOR, fontweight="bold", bbox=bbox, zorder=6)
        ax.text(cap_x, hi, "max " + fmt.format(hi), fontsize=stat_fontsize, color=stat_color,
                ha="left", va="bottom", fontweight="normal", bbox=bbox, zorder=6)
        ax.text(cap_x, lo, "min " + fmt.format(lo), fontsize=stat_fontsize, color=stat_color,
                ha="left", va="top", fontweight="normal", bbox=bbox, zorder=6)
    return bp


def _eta_squared_oneway(sub, value_col):
    clean = sub[["assigned_group", value_col]].dropna().copy()
    y = clean[value_col].to_numpy(dtype=float)
    grand = float(np.mean(y))
    ss_total = float(np.sum((y - grand) ** 2))
    if ss_total == 0:
        return 0.0
    ss_between = 0.0
    for gid in GROUP_IDS:
        vals = clean.loc[clean["assigned_group"] == gid, value_col].to_numpy(dtype=float)
        if len(vals):
            ss_between += len(vals) * (float(np.mean(vals)) - grand) ** 2
    return ss_between / ss_total


def validate_h12_relative_mae(sub, lead):
    """Validate the plotting data against the formal Appendix E/F/G values.

    The expected values below are validation targets only. The figure itself is
    always drawn from the household-level Group heterogeneity dataset, never from these
    constants. The returned statistics are computed from the plotting data and are
    used only for display labels in the main-text figure.
    """
    stats = {}
    for gid in GROUP_IDS:
        vals = sub.loc[
            sub["assigned_group"] == gid,
            "source_transfer_gain_relative_pct",
        ].dropna().to_numpy(dtype=float)
        stats[gid] = {
            "n": int(len(vals)),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "positive_share": float(np.mean(vals > 0) * 100.0),
        }

    eta2 = _eta_squared_oneway(sub, "source_transfer_gain_relative_pct")

    if lead != 12:
        return stats, eta2

    expected_means = {1: 1.69, 2: 3.39, 3: 5.01, 4: 6.76}
    expected_positive = {1: 53.9, 2: 71.4, 3: 81.2, 4: 88.5}
    expected_eta2 = 0.0613

    failed = []
    for gid in GROUP_IDS:
        mean_val = stats[gid]["mean"]
        pos_share = stats[gid]["positive_share"]
        if abs(mean_val - expected_means[gid]) > 0.015:
            failed.append(
                f"G{gid} mean {mean_val:.4f}% != expected {expected_means[gid]:.2f}%"
            )
        if abs(pos_share - expected_positive[gid]) > 0.06:
            failed.append(
                f"G{gid} positive share {pos_share:.3f}% != expected {expected_positive[gid]:.1f}%"
            )

    if abs(eta2 - expected_eta2) > 0.00015:
        failed.append(f"eta^2 {eta2:.6f} != expected {expected_eta2:.4f}")

    print("\nValidation: h=12 household-level relative MAE L-D gain")
    print("group   n    mean_%   positive_%")
    for gid in GROUP_IDS:
        row = stats[gid]
        print(f"G{gid:<1}   {row['n']:>4d}   {row['mean']:>7.3f}   {row['positive_share']:>9.3f}")
    print(f"eta^2 = {eta2:.6f}")

    if failed:
        raise RuntimeError(
            "Formal-result validation failed. Figure was NOT saved.\n- "
            + "\n- ".join(failed)
        )
    print("Validation PASS: plotted household values match the formal Appendix results.\n")
    return stats, eta2


def _draw_household_cloud(ax, sub, positions, value_col):
    rng = np.random.default_rng(HOUSEHOLD_JITTER_SEED)
    for gid, pos in zip(GROUP_IDS, positions):
        vals = sub.loc[sub["assigned_group"] == gid, value_col].dropna().to_numpy(dtype=float)
        jitter = rng.uniform(-HOUSEHOLD_JITTER_WIDTH, HOUSEHOLD_JITTER_WIDTH, size=len(vals))
        ax.scatter(
            np.full(len(vals), pos) + jitter,
            vals,
            s=HOUSEHOLD_POINT_SIZE,
            alpha=HOUSEHOLD_POINT_ALPHA,
            color=HOUSEHOLD_POINT_COLOR,
            edgecolors="none",
            zorder=2.4,
        )


def _draw_clean_box(ax, data, positions):
    """Main-text boxplot summary over the household cloud.

    This preserves the Figure 4.1 boxplot convention: whiskers span the
    observed minimum and maximum (whis=(0, 100)). The household cloud is added
    only as an extra display layer and does not change the box statistics.
    All four boxes use the same restrained blue-grey treatment because the x-axis
    already identifies G1-G4.
    """
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.44,
        patch_artist=True,
        whis=(0, 100),
        showfliers=False,  # full min/max whiskers; every household is already shown in the cloud
        medianprops=dict(color=MEDIAN_COLOR, linewidth=2.5, zorder=4),
        whiskerprops=dict(color=WHISKER_COLOR, linewidth=1.25, zorder=3.5),
        capprops=dict(color=WHISKER_COLOR, linewidth=1.25, zorder=3.5),
        boxprops=dict(linewidth=1.6, zorder=3),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(MAIN_BOX_FILL_COLOR)
        patch.set_edgecolor(MAIN_BOX_EDGE_COLOR)
        patch.set_alpha(MAIN_BOX_FILL_ALPHA)

    means = [float(np.nanmean(a)) for a in data]
    ax.scatter(
        positions,
        means,
        marker="D",
        s=54,
        facecolors="white",
        edgecolors=MEAN_EDGE_COLOR,
        linewidths=1.5,
        zorder=5,
    )
    return bp, means


def _box_legend(include_households=False):
    handles = []
    if include_households:
        handles.append(
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=HOUSEHOLD_POINT_COLOR, markeredgecolor="none",
                   alpha=0.70, markersize=5.5, label="Household")
        )
    handles.extend([
        Line2D([0], [0], color=MEDIAN_COLOR, lw=2.5, label="Median"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="white",
               markeredgecolor=MEAN_EDGE_COLOR, markeredgewidth=1.5, markersize=7.5, label="Mean"),
    ])
    return handles


def plot_relative_box_main(sub, output_dir, lead=12, save_pdf=True):
    """Render submitted dissertation Figure 4.1 from reproduced household gains."""
    stats, _ = validate_h12_relative_mae(sub, lead=lead)
    value_col = "source_transfer_gain_relative_pct"
    data = get_group_arrays(sub, value_col)
    positions = [1.0, 2.0, 3.0, 4.0]

    fig, ax = plt.subplots(figsize=(12.18, 6.70))
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.32,
        patch_artist=True,
        whis=(0, 100),
        showfliers=False,
        medianprops=dict(color=MEDIAN_COLOR, linewidth=2.5),
        whiskerprops=dict(color=WHISKER_COLOR, linewidth=1.5),
        capprops=dict(color=WHISKER_COLOR, linewidth=1.5),
        boxprops=dict(linewidth=1.7),
    )
    for gid, patch in zip(GROUP_IDS, bp["boxes"]):
        patch.set_facecolor(GROUP_FILL_COLORS[gid])
        patch.set_edgecolor(GROUP_COLORS[gid])
        patch.set_alpha(0.95)

    means = [float(np.nanmean(a)) for a in data]
    medians = [float(np.nanmedian(a)) for a in data]
    mins = [float(np.nanmin(a)) for a in data]
    maxs = [float(np.nanmax(a)) for a in data]
    ax.scatter(
        positions, means, marker="D", s=70, facecolors="white",
        edgecolors="#1C2A3A", linewidths=1.6, zorder=5,
    )

    ax.axhline(0, color=ZERO_COLOR, linestyle="--", linewidth=1.4)
    ax.set_title(
        "Relative MAE gain of Direct Transfer over Scratch Limited by profile group",
        fontsize=18, fontweight="bold", color=TEXT_COLOR, pad=14,
    )
    ax.set_ylabel("Relative MAE gain L−D (%)", fontsize=13)
    ax.set_xlabel("Profile group", fontsize=13)
    ax.set_xticks(positions, ["G1", "G2", "G3", "G4"], fontsize=12)
    ax.set_xlim(0.62, 4.78)
    all_vals = np.concatenate(data)
    lower = 5 * np.floor((float(np.nanmin(all_vals)) - 1) / 5)
    upper = 5 * np.ceil((float(np.nanmax(all_vals)) + 1) / 5)
    ax.set_ylim(lower, upper)
    ax.grid(axis="both", color=GRID_COLOR, alpha=0.75, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#9AA4B2")

    # Submitted figure annotations: mean/median beside each box; min/max beside whiskers.
    for gid, x0, mean, med, lo, hi in zip(GROUP_IDS, positions, means, medians, mins, maxs):
        ax.text(x0 + 0.21, mean + 0.4, f"mean {mean:.1f}", ha="left", va="center",
                fontsize=10, fontweight="bold", color="#243B64")
        ax.text(x0 + 0.21, med - 0.4, f"median {med:.1f}", ha="left", va="center",
                fontsize=10, fontweight="bold", color=MEDIAN_COLOR)
        ax.text(x0 + 0.15, hi + 0.6, f"max {hi:.1f}", ha="left", va="center",
                fontsize=9, color="#5E6B7A")
        ax.text(x0 + 0.15, lo - 0.6, f"min {lo:.1f}", ha="left", va="center",
                fontsize=9, color="#5E6B7A")

    handles = [
        Line2D([0], [0], color=MEDIAN_COLOR, lw=2.5, label="Median"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="white",
               markeredgecolor="#1C2A3A", markeredgewidth=1.6, markersize=9, label="Mean"),
        Line2D([0], [0], color=WHISKER_COLOR, lw=1.5, label="Box = IQR"),
        Line2D([0], [0], color=ZERO_COLOR, lw=1.4, linestyle="--", label="No gain"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.95, fontsize=10.5)

    fig.tight_layout()
    save_figure(
        fig,
        output_dir / f"figure_4_1_relative_mae_ld_gain_by_group_h{lead}.png",
        save_pdf,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--lead", type=int, default=12)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--no-pdf", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    save_pdf = not args.no_pdf
    df = load_group_heterogeneity_meter_dataset(args.input)
    sub = subset_metric(df, lead=args.lead, criterion="mae")
    plot_relative_box_main(sub, output_dir=output_dir, lead=args.lead, save_pdf=save_pdf)


if __name__ == "__main__":
    main()
