import os as _erp_os
from pathlib import Path as _ERPPath
from pathlib import Path
_ERP_PROJECT_ROOT = _ERPPath(_erp_os.environ.get("ERP_PROJECT_ROOT", ".")).expanduser().resolve()
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DEFAULT_INPUT = f"{_ERP_PROJECT_ROOT}/outputs/tables/temporal_drift_analysis_complete_cross_strategy_target_span_representativeness_temporal_drift"
OUTDIR = f"{_ERP_PROJECT_ROOT}/outputs/figures/temporal_drift_analysis_complete_cross_strategy_target_span_representativeness_temporal_drift"

OVERALL_COLOR = "#1C2A3A"
GROUP_COLORS = {
    "Group 1": "#3B6FA0",
    "Group 2": "#3F8F86",
    "Group 3": "#C6902F",
    "Group 4": "#6B5B95",
}
GRID = "#E2E6EC"
ZERO = "#9AA4B2"
TEXT = "#1C2A3A"
SPINE = "#9AA4B2"

plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 150})


def norm_group(value):
    s = str(value).strip().lower().replace("_", " ")
    mapping = {"overall": "Overall", "all": "Overall",
               "g1": "Group 1", "group 1": "Group 1", "1": "Group 1",
               "g2": "Group 2", "group 2": "Group 2", "2": "Group 2",
               "g3": "Group 3", "group 3": "Group 3", "3": "Group 3",
               "g4": "Group 4", "group 4": "Group 4", "4": "Group 4"}
    return mapping.get(s, str(value))


def norm_metric(value):
    s = str(value).lower()
    if "smape" in s:
        return "sMAPE"
    if "rmse" in s:
        return "RMSE"
    if "mae" in s:
        return "MAE"
    return str(value)


def load_row_dataset(input_path):
    path = Path(input_path).expanduser().resolve()
    name = "temporal_drift_analysis_cross_strategy_drift_outcome_dataset.csv"
    if path.is_file() and path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.is_dir():
        candidate = path / name
        if candidate.exists():
            return pd.read_csv(candidate)
        matches = list(path.rglob(name))
        if len(matches) == 1:
            return pd.read_csv(matches[0])
        if len(matches) > 1:
            raise RuntimeError("Multiple temporal-drift datasets found; pass the exact CSV path.")
        raise FileNotFoundError(f"Could not find {name} under {path}")
    raise ValueError("Input must be the reproduced temporal-drift table directory or CSV file.")


def style_axis(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(SPINE)
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(axis="both", labelsize=9, colors="#2A2F3A")


def build_quartile_frame(row_df):
    h12 = row_df[row_df["lead"] == 12].copy()
    h12["Metric"] = h12["criterion"].map(norm_metric)
    h12["Group"] = h12["assigned_group"].map(norm_group)
    base_q = h12[h12["criterion"] == "mae"][["entity_id", "test_mean_abs_change_pct"]].drop_duplicates()
    q1, q2, q3 = base_q["test_mean_abs_change_pct"].quantile([0.25, 0.5, 0.75]).tolist()
    cuts = [-np.inf, q1, q2, q3, np.inf]
    base_q["Quartile"] = pd.cut(base_q["test_mean_abs_change_pct"], bins=cuts,
                                labels=["Q1", "Q2", "Q3", "Q4"], include_lowest=True)
    mins = base_q.groupby("Quartile", observed=True)["test_mean_abs_change_pct"].min()
    maxs = base_q.groupby("Quartile", observed=True)["test_mean_abs_change_pct"].max()
    qmap = {q: f"{q}\n{mins[q]:.0f}\u2013{maxs[q]:.0f}%" for q in ["Q1", "Q2", "Q3", "Q4"]}
    h12 = h12.merge(base_q[["entity_id", "Quartile"]], on="entity_id", how="left")
    h12["relative_gain"] = h12["gain_source_transfer_relative_pct"]
    gsum = h12.groupby(["Metric", "Group", "Quartile"], observed=True)["relative_gain"].mean().reset_index()
    osum = h12.groupby(["Metric", "Quartile"], observed=True)["relative_gain"].mean().reset_index()
    osum["Group"] = "Overall"
    plot_df = pd.concat([osum, gsum], ignore_index=True)
    plot_df["QOrder"] = plot_df["Quartile"].map({"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4})
    return plot_df, qmap


def make_main_figure(plot_df, qmap, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.2), sharex=True)
    for ax, metric in zip(axes, ["MAE", "RMSE", "sMAPE"]):
        sub = plot_df[plot_df["Metric"] == metric]
        # groups first (thin, muted), overall last (thick, dark, on top)
        for g in ["Group 1", "Group 2", "Group 3", "Group 4"]:
            sg = sub[sub["Group"] == g].sort_values("QOrder")
            ax.plot(sg["QOrder"], sg["relative_gain"], marker="o", markersize=3.5,
                    linewidth=1.3, color=GROUP_COLORS[g], alpha=0.55, zorder=3)
        so = sub[sub["Group"] == "Overall"].sort_values("QOrder")
        ax.plot(so["QOrder"], so["relative_gain"], marker="o", markersize=7,
                linewidth=3.2, color=OVERALL_COLOR, zorder=6,
                markeredgecolor="white", markeredgewidth=1.2)
        ax.axhline(0, color=ZERO, linewidth=1.0, linestyle="--", zorder=1)
        ax.set_title(metric, fontsize=12.5, fontweight="bold", color=TEXT, pad=8)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels([qmap[q] for q in ["Q1", "Q2", "Q3", "Q4"]], fontsize=8.5)
        style_axis(ax)
    axes[0].set_ylabel("Direct Transfer over Scratch Limited Gain (%)", fontsize=11, color=TEXT)
    fig.suptitle("Temporal drift and Direct Transfer over Scratch Limited Gain across drift quartiles (h = 12)",
                 fontsize=14, fontweight="bold", color=TEXT, y=0.99)
    fig.supxlabel("Temporal drift in mean load between the first 30 days of the training period and the test period",
                  fontsize=10.5, color=TEXT, y=0.075)
    handles = [Line2D([0], [0], color=OVERALL_COLOR, marker="o", linewidth=3.2, markersize=7, label="Overall")]
    handles += [Line2D([0], [0], color=GROUP_COLORS[g], marker="o", linewidth=1.3, markersize=4,
                       alpha=0.7, label=g) for g in ["Group 1", "Group 2", "Group 3", "Group 4"]]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=10)
    fig.tight_layout(rect=[0, 0.11, 1, 0.94])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Saved:", output_path)


def make_appendix_figure(row_df, output_path, metric="MAE"):
    h12 = row_df[row_df["lead"] == 12].copy()
    h12["Metric"] = h12["criterion"].map(norm_metric)
    h12["Group"] = h12["assigned_group"].map(norm_group)
    h12["relative_gain"] = h12["gain_source_transfer_relative_pct"]
    app = h12[h12["Metric"] == metric]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, g in zip(axes, ["Group 1", "Group 2", "Group 3", "Group 4"]):
        gdf = app[app["Group"] == g]
        ax.scatter(gdf["test_mean_abs_change_pct"], gdf["relative_gain"], s=13, alpha=0.25,
                   color=GROUP_COLORS[g], edgecolors="none", zorder=3)
        bins = pd.qcut(gdf["test_mean_abs_change_pct"], q=8, duplicates="drop")
        med = gdf.groupby(bins, observed=True).agg(x=("test_mean_abs_change_pct", "median"),
                                                   y=("relative_gain", "median")).dropna()
        ax.plot(med["x"], med["y"], color=GROUP_COLORS[g], linewidth=2.4, marker="o", markersize=4.5, zorder=5)
        ax.set_xscale("log")
        ax.axhline(0, color=ZERO, linewidth=1.0, linestyle="--", zorder=1)
        ax.set_title(g, fontsize=12, fontweight="bold", color=TEXT, pad=6)
        ax.grid(color=GRID, linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        for side in ["left", "bottom"]:
            ax.spines[side].set_color(SPINE)
    fig.suptitle(f"Household-level temporal drift and Direct Transfer over Scratch Limited relative {metric} Gain by group (h = 12)",
                 fontsize=13.5, fontweight="bold", color=TEXT, y=0.98)
    fig.supxlabel(
    "Temporal drift in mean load between the first 30 days of the training period and the test period (%)",
    fontsize=10.5,
    color=TEXT,
    x=0.56,
    y=0.055
)
    fig.supylabel(f"Direct Transfer over Scratch Limited relative {metric} Gain (%)", fontsize=10.5, color=TEXT, x=0.055)
    fig.tight_layout(rect=[0.06, 0.06, 1, 0.95])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Saved:", output_path)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=OUTDIR)
    args = parser.parse_args()
    row_df = load_row_dataset(args.input)
    plot_df, qmap = build_quartile_frame(row_df)
    make_main_figure(plot_df, qmap, os.path.join(args.output_dir, "figure_4_4_ld_relative_gain_by_temporal_drift_quartile.png"))
    make_appendix_figure(row_df, os.path.join(args.output_dir, "appendix_k_figure_k1_ld_relative_mae_gain_by_group.png"), metric="MAE")
    make_appendix_figure(row_df, os.path.join(args.output_dir, "appendix_k_figure_k2_ld_relative_rmse_gain_by_group.png"), metric="RMSE")
    make_appendix_figure(row_df, os.path.join(args.output_dir, "appendix_k_figure_k3_ld_relative_smape_gain_by_group.png"), metric="sMAPE")


if __name__ == "__main__":
    main()
