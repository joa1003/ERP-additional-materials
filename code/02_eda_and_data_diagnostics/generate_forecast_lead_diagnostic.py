from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

LOOKBACK = 48
LEADS = range(1, 49)
STD_FLOOR = 0.01
TRAIN_START = pd.Timestamp("2013-01-02 00:00")
TRAIN_END = pd.Timestamp("2013-09-13 18:30")
PURPLE = "#6F4E8C"
PURPLE_FILL = "#C8B7D4"
BLUE = "#1F77B4"
BLUE_FILL = "#B8D0E3"
ORANGE = "#E67E22"
GRID = "#DCE2E8"


def summ(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return np.nanmedian(v), np.nanquantile(v, .25), np.nanquantile(v, .75)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=Path.cwd())
    a = ap.parse_args()
    root = a.project_root.resolve()

    cohort = pd.read_csv(root / "outputs/tables/horizon_impact/lcl_p0_final_eligible_households_new.csv")
    hh = cohort.household_id.astype(str).tolist()
    tm = pd.read_parquet(root / "data/processed/time_aligned/lcl_full_local_clock_map.parquet", columns=["timestamp_gmt"])
    idx = pd.DatetimeIndex(pd.to_datetime(tm.timestamp_gmt))
    raw = pd.read_csv(
        root / "data/raw/LCL/csv/data_collection/data_tables/consumption_n.csv",
        usecols=hh, low_memory=False,
    ).apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    train = (idx >= TRAIN_START) & (idx <= TRAIN_END)
    tr = np.flatnonzero(train)
    mu = np.nanmean(raw[tr], axis=0)
    sd = np.maximum(np.nanstd(raw[tr], axis=0, ddof=0), STD_FLOOR)
    z = (raw - mu) / sd

    rows = []
    for h in LEADS:
        targets = tr[tr - (h + LOOKBACK - 1) >= 0]
        corr, pmae, smae = [], [], []
        for j in range(len(hh)):
            y = z[targets, j]
            origin = z[targets - h, j]
            seasonal = z[targets - 48, j]
            ok = np.isfinite(y) & np.isfinite(origin)
            ok2 = np.isfinite(y) & np.isfinite(seasonal)
            corr.append(np.corrcoef(origin[ok], y[ok])[0, 1] if ok.sum() > 1 and np.std(origin[ok]) > 0 and np.std(y[ok]) > 0 else np.nan)
            pmae.append(np.mean(np.abs(y[ok] - origin[ok])) if ok.any() else np.nan)
            smae.append(np.mean(np.abs(y[ok2] - seasonal[ok2])) if ok2.any() else np.nan)
        cm, cq1, cq3 = summ(corr)
        pm, pq1, pq3 = summ(pmae)
        sm, sq1, sq3 = summ(smae)
        rows.append({
            "horizon": h,
            "correlation_median": cm,
            "correlation_p25": cq1,
            "correlation_p75": cq3,
            "persistence_mae_scaled_median": pm,
            "persistence_mae_scaled_p25": pq1,
            "persistence_mae_scaled_p75": pq3,
            "daily_seasonal_mae_scaled_median": sm,
            "daily_seasonal_mae_scaled_p25": sq1,
            "daily_seasonal_mae_scaled_p75": sq3,
        })

    summary = pd.DataFrame(rows)
    td = root / "outputs/tables/horizon_impact"
    fd = root / "outputs/figures/horizon_impact"
    td.mkdir(parents=True, exist_ok=True)
    fd.mkdir(parents=True, exist_ok=True)
    summary.to_csv(td / "lcl_final3843_horizon_difficulty_summary_h1_h48.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14.36, 5.71))
    fig.suptitle("Forecast lead analysis", fontsize=18, y=0.995)
    x = summary.horizon.to_numpy(dtype=float)

    # Panel (a)
    ax = axes[0]
    ax.plot(x, summary.correlation_median, color=PURPLE, lw=2.5)
    ax.fill_between(x, summary.correlation_p25, summary.correlation_p75, color=PURPLE_FILL, alpha=.45)
    ax.axhline(0, color="#777777", lw=1.0, ls="--")
    selected = summary[summary.horizon.isin([1, 12, 48])]
    ax.scatter(selected.horizon, selected.correlation_median, color=PURPLE, s=64, zorder=5)
    for h, dx, dy in [(1, 1.8, 0.05), (12, 1.6, 0.06), (48, -6.0, 0.07)]:
        y = float(summary.loc[summary.horizon.eq(h), "correlation_median"].iloc[0])
        ax.text(h + dx, y + dy, rf"$h = {h}$", fontsize=13, color="black")
    ax.set_title("(a) Load dependence across forecast leads", fontsize=15, pad=10)
    ax.set_xlabel(r"Forecast lead $h$", fontsize=13)
    ax.set_ylabel("Correlation between forecast origin load and target", fontsize=13)
    ax.set_xlim(1, 48)
    ax.set_xticks([1, 6, 12, 18, 24, 30, 36, 42, 48])
    ax.grid(False)
    ax.legend(
        handles=[Line2D([0], [0], color=PURPLE, lw=2.5, label="Household median"),
                 Patch(facecolor=PURPLE_FILL, edgecolor=PURPLE_FILL, alpha=.45, label="Household IQR")],
        loc="upper right", frameon=False, fontsize=11,
    )

    # Panel (b)
    ax = axes[1]
    ax.plot(x, summary.persistence_mae_scaled_median, color=BLUE, lw=2.5)
    ax.fill_between(x, summary.persistence_mae_scaled_p25, summary.persistence_mae_scaled_p75, color=BLUE_FILL, alpha=.45)
    # Dissertation uses one daily-seasonal reference across the lead scan.
    seasonal_ref = float(np.nanmedian(summary.daily_seasonal_mae_scaled_median.to_numpy(dtype=float)))
    ax.axhline(seasonal_ref, color=ORANGE, lw=2.2, ls="--")
    ax.scatter(selected.horizon, selected.persistence_mae_scaled_median, color=BLUE, s=64, zorder=5)
    for h, dx, dy in [(1, 1.8, 0.04), (12, 1.6, 0.06), (48, -6.2, 0.03)]:
        y = float(summary.loc[summary.horizon.eq(h), "persistence_mae_scaled_median"].iloc[0])
        bbox = dict(facecolor="white", edgecolor="none", alpha=0.92, pad=1.5) if h in {12, 48} else None
        ax.text(h + dx, y + dy, rf"$h = {h}$", fontsize=13, color="black", bbox=bbox)
    ax.set_title("(b) Naive forecast error across forecast leads", fontsize=15, pad=10)
    ax.set_xlabel(r"Forecast lead $h$", fontsize=13)
    ax.set_ylabel("MAE / household training standard deviation", fontsize=13)
    ax.set_xlim(1, 48)
    ax.set_xticks([1, 6, 12, 18, 24, 30, 36, 42, 48])
    ax.grid(False)
    ax.legend(
        handles=[Line2D([0], [0], color=BLUE, lw=2.5, label="Persistence median"),
                 Patch(facecolor=BLUE_FILL, edgecolor=BLUE_FILL, alpha=.45, label="Persistence IQR"),
                 Line2D([0], [0], color=ORANGE, lw=2.2, ls="--", label="Daily seasonal naive")],
        loc="lower right", frameon=False, fontsize=11,
    )

    for ax in axes:
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color("#7A8793")
        ax.spines["bottom"].set_color("#7A8793")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fd / "forecast_lead_analysis.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("FORECAST LEAD DIAGNOSTIC: PASS")


if __name__ == "__main__":
    main()
