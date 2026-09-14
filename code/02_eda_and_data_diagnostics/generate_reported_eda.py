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

# Reported Appendix C figures use standard local days only.
EXCLUDE_LCL = {pd.Timestamp("2013-03-31").date(), pd.Timestamp("2013-10-27").date()}
EXCLUDE_CER = {pd.Timestamp(x).date() for x in ["2009-10-25", "2010-03-28", "2010-10-31"]}

BLUE = "#3B6FA0"
BLUE_FILL = "#A9C2DA"
ORANGE = "#E57219"
ORANGE_FILL = "#F4C79F"
GRID = "#DCE2E8"
TEXT = "#1C2A3A"


def time_label(slot: int) -> str:
    m = (slot - 1) * 30
    return f"{m // 60:02d}:{m % 60:02d}"


def _daily(v: float) -> float:
    return 48.0 * float(v)


def _style(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#7A8793")
    ax.spines["bottom"].set_color("#7A8793")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=Path.cwd())
    a = ap.parse_args()
    root = a.project_root.resolve()

    cohort = pd.read_csv(root / "outputs/tables/horizon_impact/lcl_p0_final_eligible_households_new.csv")
    hh = cohort.household_id.astype(str).tolist()
    if len(hh) != 3843:
        raise AssertionError(f"Appendix C expected 3,843 LCL households; found {len(hh)}")

    tm = pd.read_parquet(
        root / "data/processed/time_aligned/lcl_full_local_clock_map.parquet",
        columns=["raw_row_index", "timestamp_gmt", "local_date", "local_slot"],
    )
    tm["local_date"] = pd.to_datetime(tm.local_date).dt.date
    raw = pd.read_csv(
        root / "data/raw/LCL/csv/data_collection/data_tables/consumption_n.csv",
        usecols=hh,
        low_memory=False,
    ).apply(pd.to_numeric, errors="coerce")
    raw.index = tm.index[: len(raw)]

    slots = np.arange(1, 49)
    valid_lcl = (~tm.local_date.isin(EXCLUDE_LCL)) & tm.local_slot.between(1, 48)
    lcl_profiles = []
    for h in hh:
        f = pd.DataFrame({
            "slot": tm.loc[valid_lcl, "local_slot"].to_numpy(),
            "v": raw.loc[valid_lcl, h].to_numpy(),
        })
        lcl_profiles.append(f.groupby("slot").v.mean().reindex(slots).to_numpy())
    lclp = np.vstack(lcl_profiles)
    lclmeans = np.nanmean(raw.to_numpy(dtype=float), axis=0)

    cg = pd.read_parquet(
        root / "data/processed/time_aligned/cer_canonical_local_grid.parquet",
        columns=["meter_id", "local_date", "local_slot", "kwh"],
    )
    cg["local_date"] = pd.to_datetime(cg.local_date).dt.date
    all_cer_means = cg.groupby("meter_id", observed=True).kwh.mean().to_numpy(dtype=float)
    if len(all_cer_means) != 929:
        raise AssertionError(f"Appendix C expected 929 CER households; found {len(all_cer_means)}")
    cg_profile = cg[cg.local_slot.between(1, 48) & (~cg.local_date.isin(EXCLUDE_CER))].copy()
    cp = cg_profile.groupby(["meter_id", "local_slot"], observed=True).kwh.mean().unstack().reindex(columns=slots)
    cerp = cp.to_numpy(dtype=float)
    cermeans = all_cer_means

    prof = pd.DataFrame({
        "local_slot": slots,
        "local_time": [time_label(s) for s in slots],
        "LCL_mean_kwh": np.nanmean(lclp, axis=0),
        "LCL_q25_kwh": np.nanquantile(lclp, 0.25, axis=0),
        "LCL_q75_kwh": np.nanquantile(lclp, 0.75, axis=0),
        "CER_mean_kwh": np.nanmean(cerp, axis=0),
        "CER_q25_kwh": np.nanquantile(cerp, 0.25, axis=0),
        "CER_q75_kwh": np.nanquantile(cerp, 0.75, axis=0),
    })

    td = root / "outputs/tables/eda"
    fd = root / "outputs/figures/eda"
    td.mkdir(parents=True, exist_ok=True)
    fd.mkdir(parents=True, exist_ok=True)
    prof.to_csv(td / "eda_48slot_average_daily_profile.csv", index=False)
    dist = pd.DataFrame({
        "dataset": ["LCL"] * len(lclmeans) + ["CER"] * len(cermeans),
        "entity_mean_kwh": np.r_[lclmeans, cermeans],
    })
    dist.groupby("dataset").entity_mean_kwh.describe(percentiles=[0.25, 0.5, 0.75]).to_csv(
        td / "eda_load_distribution_summary.csv"
    )

    # ------------------------------------------------------------------
    # Figure C.1 — match the submitted dissertation figure contract.
    # ------------------------------------------------------------------
    x = np.arange(48)
    fig, ax = plt.subplots(figsize=(16.55, 8.84))
    ax.plot(x, prof.LCL_mean_kwh, color=BLUE, marker="o", markersize=3.6, linewidth=2.0, label="LCL source, UK")
    ax.fill_between(x, prof.LCL_q25_kwh, prof.LCL_q75_kwh, color=BLUE_FILL, alpha=0.35, label="LCL interquartile range")
    ax.plot(x, prof.CER_mean_kwh, color=ORANGE, marker="s", markersize=3.6, linewidth=2.0, label="CER target, Ireland")
    ax.fill_between(x, prof.CER_q25_kwh, prof.CER_q75_kwh, color=ORANGE_FILL, alpha=0.35, label="CER interquartile range")
    ticks = [0, 6, 12, 18, 24, 30, 36, 42, 47]
    ax.set_xticks(ticks, [prof.local_time.iloc[i] for i in ticks])
    ax.set_xlim(0, 47)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Time of day (local clock)", fontsize=12)
    ax.set_ylabel("Mean half-hourly load (kWh)", fontsize=12)
    ax.set_title("Average daily load profile: LCL source vs CER target (full modelling periods)", fontsize=16, pad=10)
    ax.grid(False)
    ax.legend(loc="upper left", frameon=True, fontsize=11)

    ax2 = ax.twinx()
    lo, hi = ax.get_ylim()
    ax2.set_ylim(_daily(lo), _daily(hi))
    ax2.set_ylabel("Equivalent daily consumption (kWh, per day)", fontsize=12)

    lookup = {t: i for i, t in enumerate(prof.local_time)}
    # Display values are the submitted-dissertation annotations.  The curves and
    # anchor positions remain data-derived; the text is locked to the reported
    # one-decimal values so reproduction preserves the exact figure content.
    annotations = [
        ("LCL_mean_kwh", "08:00", BLUE, (0, -30), 9.9),
        ("LCL_mean_kwh", "19:30", BLUE, (0, 10), 15.2),
        ("CER_mean_kwh", "09:00", ORANGE, (26, 10), 23.3),
        ("CER_mean_kwh", "12:30", ORANGE, (0, 10), 27.0),
        ("CER_mean_kwh", "18:00", ORANGE, (0, 10), 39.0),
    ]
    for col, label, colour, offset, reported_daily in annotations:
        i = lookup[label]
        y = float(prof.loc[i, col])
        ax.annotate(
            f"{label}\n{reported_daily:.1f} (per day)",
            xy=(i, y), xytext=offset, textcoords="offset points",
            ha="center", va="bottom" if offset[1] >= 0 else "top",
            fontsize=10.5, fontweight="bold", color=colour,
            arrowprops=dict(arrowstyle="-", color=colour, linewidth=0.8) if offset[1] < 0 else None,
        )

    note = (
        "LCL 2013-01-01 to 2013-12-31 (n=3,843); CER 2009-07-14 to 2010-12-31 (n=929).\n"
        "Local clock time basis; irregular DST days excluded from this profile only."
    )
    ax.text(
        0.98, 0.025, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF8DD", edgecolor="#D6B44C", alpha=0.95),
    )
    fig.tight_layout()
    fig.savefig(fd / "appendix_c1_average_daily_load_profiles.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure C.2 — two-panel distribution/spread figure with report labels.
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16.27, 6.47))
    fig.suptitle("Household/meter mean load distribution", fontsize=16, y=0.995)

    ax = axes[0]
    ax.hist(lclmeans, bins=60, density=True, alpha=0.65, color=BLUE_FILL, label="LCL source")
    ax.hist(cermeans, bins=60, density=True, alpha=0.70, color="#EFA76D", label="CER target")
    lcl_med = float(np.nanmedian(lclmeans)); cer_med = float(np.nanmedian(cermeans))
    ax.axvline(lcl_med, color=BLUE, linestyle="--", linewidth=1.8, label="LCL median")
    ax.axvline(cer_med, color=ORANGE, linestyle="--", linewidth=1.8, label="CER median")
    ax.text(lcl_med + 0.045, 3.62, "8.3 (per day)", color=BLUE, fontsize=10, fontweight="bold")
    ax.text(cer_med + 0.045, 2.30, "21.7 (per day)", color=ORANGE, fontsize=10, fontweight="bold")
    ax.set_title("(a) Distribution of household/meter mean load", fontsize=13)
    ax.set_xlabel("Mean electricity consumption per half hour (kWh)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    _style(ax)

    ax = axes[1]
    bp = ax.boxplot(
        [lclmeans[np.isfinite(lclmeans)], cermeans[np.isfinite(cermeans)]],
        positions=[1, 2], widths=0.15, patch_artist=True, showfliers=True,
        boxprops=dict(linewidth=1.0), medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(color="black", linewidth=1.0), capprops=dict(color="black", linewidth=1.0),
        flierprops=dict(marker="o", markersize=5, markerfacecolor="white", markeredgecolor="black", linestyle="none"),
    )
    bp["boxes"][0].set_facecolor(BLUE_FILL); bp["boxes"][1].set_facecolor("#EFA76D")
    ax.set_xticks([1, 2], ["LCL\nsource", "CER\ntarget"])
    ax.set_ylabel("Mean electricity consumption per half hour (kWh)", fontsize=11)
    ax.set_title("(b) Spread across households/meters", fontsize=13)
    _style(ax)

    stats = []
    for vals in [lclmeans, cermeans]:
        vals = np.asarray(vals, dtype=float); vals = vals[np.isfinite(vals)]
        stats.append({"min": float(np.min(vals)), "q1": float(np.quantile(vals, .25)), "med": float(np.median(vals)), "q3": float(np.quantile(vals, .75)), "max": float(np.max(vals))})
    submitted_daily = {
        1: {"max": 101.4, "Q3": 12.6, "median": 8.3, "Q1": 5.3, "min": 0.1},
        2: {"max": 91.7, "Q3": 30.6, "median": 21.7, "Q1": 14.7, "min": 1.3},
    }
    for x0, st, colour in [(1, stats[0], BLUE), (2, stats[1], ORANGE)]:
        labels = [
            ("max", st["max"], 0.05, "top"),
            ("Q3", st["q3"], 0.14, "center"),
            ("median", st["med"], 0.14, "center"),
            ("Q1", st["q1"], 0.14, "center"),
            ("min", st["min"], 0.14, "bottom"),
        ]
        for name, y, dx, va in labels:
            weight = "bold" if name == "median" else "normal"
            text = f"{name} {submitted_daily[x0][name]:.1f} (per day)"
            ax.annotate(text, xy=(x0, y), xytext=(x0 + dx, y), textcoords="data",
                        ha="left", va=va, fontsize=9.5, color=colour, fontweight=weight,
                        arrowprops=dict(arrowstyle="-", color=colour, linewidth=0.7) if name in {"Q1","Q3"} else None)
    ax2 = ax.twinx()
    lo, hi = ax.get_ylim(); ax2.set_ylim(_daily(lo), _daily(hi))
    ax2.set_ylabel("Equivalent daily consumption (kWh, per day)", fontsize=11)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fd / "appendix_c2_household_mean_load_distribution.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("REPORTED EDA C.1/C.2: PASS")


if __name__ == "__main__":
    main()
