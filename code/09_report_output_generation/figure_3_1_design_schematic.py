#!/usr/bin/env python3
"""Render submitted dissertation Figure 3.1 strategy-design schematic."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch

GREY = "#C7D1DC"
SOURCE = "#C99A2E"
TARGET = "#7D89B6"
STRATEGY = "#65AAA4"
TEXT = "#000000"


def rounded(ax, xy, w, h, fc, radius=0.025, text=None, fontsize=18, weight="bold"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=fc, edgecolor="none"
    )
    ax.add_patch(patch)
    if text:
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize, fontweight=weight, color=TEXT)
    return patch


def hexbox(ax, xy, w, h, fc, text, fontsize=17):
    x, y = xy
    cut = 0.12 * w
    pts = [(x+cut,y),(x+w-cut,y),(x+w,y+h/2),(x+w-cut,y+h),(x+cut,y+h),(x,y+h/2)]
    p = Polygon(pts, closed=True, facecolor=fc, edgecolor="none")
    ax.add_patch(p)
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fontsize, fontweight="bold", color=TEXT)
    return p


def elbow(ax, start, end, mid_y=None, lw=2.2):
    sx, sy = start; ex, ey = end
    if mid_y is None:
        mid_y = (sy+ey)/2
    # vertical -> horizontal -> vertical with final arrowhead
    ax.plot([sx, sx], [sy, mid_y], color="black", lw=lw, solid_capstyle="round")
    ax.plot([sx, ex], [mid_y, mid_y], color="black", lw=lw, solid_capstyle="round")
    arr = FancyArrowPatch((ex, mid_y), (ex, ey), arrowstyle="-|>", mutation_scale=15,
                          linewidth=lw, color="black", shrinkA=0, shrinkB=0)
    ax.add_patch(arr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = root / "outputs/figures/report/figure_3_1_strategy_design.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6.75))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # Top architecture block.
    rounded(ax, (0.27, 0.865), 0.46, 0.105, GREY, radius=0.035,
            text="Same global CNN-LSTM architecture", fontsize=19)

    # Dataset blocks.
    hexbox(ax, (0.18, 0.675), 0.32, 0.12, SOURCE, "Source dataset", fontsize=18)
    hexbox(ax, (0.56, 0.675), 0.28, 0.12, TARGET, "Target dataset", fontsize=18)

    # Architecture connection to datasets.
    ax.plot([0.50,0.50],[0.865,0.825],color='black',lw=2.2)
    ax.plot([0.50,0.50],[0.825,0.825],color='black',lw=2.2)
    ax.plot([0.50,0.34],[0.825,0.825],color='black',lw=2.2)
    ax.plot([0.34,0.34],[0.825,0.795],color='black',lw=2.2)
    ax.plot([0.50,0.70],[0.825,0.825],color='black',lw=2.2)
    ax.plot([0.70,0.70],[0.825,0.795],color='black',lw=2.2)

    # Training-source/target-period blocks.
    hexbox(ax, (0.08, 0.475), 0.29, 0.13, SOURCE, "Source trained\nparameters $\\theta_S$", fontsize=17)
    hexbox(ax, (0.39, 0.475), 0.30, 0.13, TARGET, "First 30 days\ntraining period", fontsize=17)
    hexbox(ax, (0.71, 0.485), 0.25, 0.12, TARGET, "Full target\ntraining period", fontsize=17)

    # Dataset arrows.
    elbow(ax, (0.34,0.675), (0.22,0.605), mid_y=0.64)
    elbow(ax, (0.70,0.675), (0.54,0.605), mid_y=0.64)
    elbow(ax, (0.70,0.675), (0.835,0.605), mid_y=0.64)

    # Strategy blocks.
    rounded(ax, (0.045, 0.285), 0.20, 0.105, STRATEGY, radius=0.027,
            text="Direct Transfer\n(D)", fontsize=17)
    rounded(ax, (0.305, 0.285), 0.18, 0.105, STRATEGY, radius=0.027,
            text="Fine Tuning\n(T)", fontsize=17)
    rounded(ax, (0.525, 0.285), 0.205, 0.105, STRATEGY, radius=0.027,
            text="Scratch Limited\n(L)", fontsize=17)
    rounded(ax, (0.765, 0.300), 0.17, 0.105, STRATEGY, radius=0.027,
            text="Scratch Full\n(F)", fontsize=17)

    # Parameter/period arrows to strategies.
    elbow(ax, (0.22,0.475), (0.145,0.39), mid_y=0.44)
    elbow(ax, (0.22,0.475), (0.395,0.39), mid_y=0.44)
    elbow(ax, (0.54,0.475), (0.395,0.39), mid_y=0.44)
    elbow(ax, (0.54,0.475), (0.625,0.39), mid_y=0.44)
    ax.add_patch(FancyArrowPatch((0.835,0.485),(0.85,0.405),arrowstyle='-|>',mutation_scale=15,linewidth=2.2,color='black'))

    # Common target test period.
    rounded(ax, (0.30, 0.045), 0.40, 0.135, GREY, radius=0.035)
    ax.text(0.50,0.125,"Common target test period",ha='center',va='center',fontsize=18,fontweight='bold')
    ax.text(0.50,0.085,r"Forecast leads: $h = 1, 12, 48$",ha='center',va='center',fontsize=16)

    # Strategy outputs merge into common target test period.
    yjoin = 0.245
    for x in [0.145,0.395,0.625,0.85]:
        ax.plot([x,x],[0.285 if x != 0.85 else 0.300,yjoin],color='black',lw=2.2)
    ax.plot([0.145,0.50],[yjoin,yjoin],color='black',lw=2.2)
    ax.plot([0.395,0.50],[yjoin,yjoin],color='black',lw=2.2)
    ax.plot([0.625,0.50],[yjoin,yjoin],color='black',lw=2.2)
    ax.plot([0.85,0.50],[yjoin,yjoin],color='black',lw=2.2)
    ax.add_patch(FancyArrowPatch((0.50,yjoin),(0.50,0.18),arrowstyle='-|>',mutation_scale=15,linewidth=2.2,color='black'))

    fig.tight_layout(pad=0)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix('.pdf'), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("FIGURE 3.1: PASS")
    print(out)


if __name__ == "__main__":
    main()
