import os as _erp_os
from pathlib import Path as _ERPPath
_ERP_PROJECT_ROOT = _ERPPath(_erp_os.environ.get("ERP_PROJECT_ROOT", ".")).expanduser().resolve()
"""
Scratch Full over Scratch Limited (L-F), h = 12
Produces three separate output figures for the submitted report:

  1. figure_4_8_lf_gain_by_load_range.png
     Within-bin gain distribution by observed load range (household level).
     Source: load_range_analysis complete cross strategy raw load composition
     standardised gain decomposition reproduced tables.

  2. figure_4_9_lf_cumulative_load_range_contribution.png
     Cumulative contribution to net gain across observed load ranges.
     Source: same load_range_analysis bundle, group-level (Overall CER) table.

  3. figure_4_10_lf_gain_by_drift_quartile.png
     L-F gain across temporal-drift quartiles (household level).
     Source: temporal_drift_analysis complete cross strategy target span representativeness
     temporal drift reproduced tables.

All underlying gain, contribution and temporal-drift values are read directly
from the validated reproduced tables. Figure-specific filtering, quartile assignment
and cumulative plotting operations are performed only for visualisation.
Observed-load percentages shown on Figures 4.8 and 4.9 are read from the
validated Overall CER mean_meter_row_share column rather than hardcoded.

Sign convention (verified against
load_range_analysis_complete_cross_strategy_raw_load_composition_standardised_gain_decomposition_decision.md):
    Gain_E(A,B) = Error_B - Error_A, positive means focal strategy A has
    lower error than comparator B. For "full_data", focal = Scratch Full,
    comparator = Scratch Limited, so:
        gain = Error(Scratch Limited) - Error(Scratch Full)
    Positive gain -> Scratch Full had the lower error.

Figure 4.9 annotation convention: each point label is prefixed with
"Delta" to make explicit that it shows the incremental contribution added
at that step, not the cumulative y-value of the point itself.

Validation note: the MAE and sMAPE endpoints of Figure 4.9 should
reproduce the corresponding h=12 net gains reported elsewhere for this
comparison. The MSE endpoint should be checked against the analysis table's own
additive MSE decomposition net gain only - it will NOT equal the squared
RMSE gain reported for this comparison, since sqrt(a) - sqrt(b) != sqrt(a-b).
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# The plotting script accepts the validated load-range and temporal-drift inputs
# are extracted under one exact hard-coded directory.
#
# It first checks an optional explicit directory (leave as None by default).
# If the file is not there, it searches ERP_ROOT recursively for the exact
# formal filename.  If exactly one copy is found, that copy is used.
# If multiple copies are found, execution stops rather than selecting one silently.
ERP_ROOT = Path(f"{_ERP_PROJECT_ROOT}")

LOAD_RANGE_ANALYSIS_TABLE_DIR = ERP_ROOT / "outputs/tables/load_range_analysis_complete_cross_strategy_raw_load_composition_standardised_gain_decomposition"
TEMPORAL_DRIFT_ANALYSIS_TABLE_DIR = ERP_ROOT / "outputs/tables/temporal_drift_analysis_complete_cross_strategy_target_span_representativeness_temporal_drift"

OUTPUT_DIR = Path(
    f"{_ERP_PROJECT_ROOT}/outputs/figures/"
    "load_range_analysis_complete_cross_strategy_raw_load_composition_standardised_gain_decomposition"
)

# load_range_analysis source files
METER_BIN_FILE = "load_range_analysis_h12_meter_four_bin_pairwise_gains_four_seed.csv"
GROUP_BIN_FILE = "load_range_analysis_h12_group_bin_conditional_gain_and_contribution.csv"

# temporal_drift_analysis source file
DRIFT_OUTCOME_FILE = "temporal_drift_analysis_cross_strategy_drift_outcome_dataset.csv"

COMPARISON_ID = "full_data"   # Scratch Full over Scratch Limited (L-F), load_range_analysis naming
POPULATION = "Overall CER"
RAW_BINS = ["actual_le_0_1", "actual_gt_0_1_le_0_5", "actual_gt_0_5_le_1_0", "actual_gt_1_0"]
BIN_AXIS_LABELS = ["\u22640.1 kWh", "0.1\u20130.5 kWh", "0.5\u20131.0 kWh", ">1.0 kWh"]
CUMLINE_AXIS_LABELS = ["\u22640.1\nkWh", "0.1\u20130.5\nkWh", "0.5\u20131.0\nkWh", ">1.0\nkWh"]

METRIC_COLUMN = {"mae": "gain_mae", "mse": "gain_mse", "smape": "gain_smape"}
METRIC_UNIT = {"mae": "kWh", "mse": "kWh\u00b2", "smape": "pp"}
METRIC_TITLE = {"mae": "MAE", "mse": "MSE", "smape": "sMAPE"}
METRIC_FMT = {"mae": "{:+.4f}", "mse": "{:+.4f}", "smape": "{:+.2f}"}

LEAD = 12
DRIFT_GAIN_COL = "gain_full_data_absolute"   # L-F, temporal_drift_analysis naming
DRIFT_PREDICTOR = "test_mean_abs_change_pct"
CRITERIA = ["mae", "rmse", "smape"]
CRITERION_TITLE = {"mae": "MAE", "rmse": "RMSE", "smape": "sMAPE"}
CRITERION_UNIT = {"mae": "kWh", "rmse": "kWh", "smape": "pp"}

BIN_COLOURS = ["#A9C2DA", "#6D93B8", "#3B6FA0", "#1C2A3A"]
QUARTILE_COLOURS = ["#A9C2DA", "#6D93B8", "#3B6FA0", "#1C2A3A"]
GOLD = "#B5820B"
ZERO_BLUE = "#2C6EBD"
GRID = "#E2E6EC"
NAVY = "#1C2A3A"



# ---------------------------------------------------------------------------
# Validation settings
# ---------------------------------------------------------------------------
EXPECTED_CER_METERS = 929
EXPECTED_DRIFT_QUARTILE_MEANS = {
    "mae": [0.0076, 0.0093, 0.0165, 0.0373],
    "rmse": [0.0205, 0.0311, 0.0483, 0.0975],
    "smape": [1.420, 1.172, 1.942, 3.869],
}
DRIFT_MEAN_TOLERANCE = {
    "mae": 5e-4,
    "rmse": 5e-4,
    "smape": 5e-3,
}


def _assert_unique_rows(df, keys, label):
    dup = df.duplicated(keys, keep=False)
    if dup.any():
        sample = df.loc[dup, keys].head(10).to_dict("records")
        raise ValueError(f"{label}: duplicate rows for keys {keys}. Examples: {sample}")


def validate_load_range_analysis_meter_table(df):
    required = {"comparison_id", "raw_bin", "gain_mae", "gain_mse", "gain_smape"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"load_range_analysis meter table missing columns: {sorted(missing)}")

    filtered = df[df["comparison_id"] == COMPARISON_ID]
    if filtered.empty:
        raise ValueError(f"No load_range_analysis meter rows found for comparison_id={COMPARISON_ID!r}.")

    missing_bins = [b for b in RAW_BINS if b not in set(filtered["raw_bin"].dropna())]
    if missing_bins:
        raise ValueError(f"load_range_analysis meter table missing raw bins: {missing_bins}")


def validate_load_range_analysis_group_table(df):
    required = {
        "comparison_id", "population", "raw_bin", "metric",
        "mean_meter_row_share",
        "observed_gain_contribution", "exposure_weighted_conditional_gain",
    }
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"load_range_analysis group table missing columns: {sorted(missing)}")

    filtered = df[
        (df["comparison_id"] == COMPARISON_ID) &
        (df["population"] == POPULATION)
    ].copy()

    if filtered.empty:
        raise ValueError(
            f"No load_range_analysis group rows found for comparison_id={COMPARISON_ID!r}, "
            f"population={POPULATION!r}."
        )

    _assert_unique_rows(filtered, ["metric", "raw_bin"], "load_range_analysis Overall CER table")

    expected_pairs = {(m, b) for m in ["mae", "mse", "smape"] for b in RAW_BINS}
    actual_pairs = set(zip(filtered["metric"], filtered["raw_bin"]))
    missing_pairs = sorted(expected_pairs.difference(actual_pairs))
    if missing_pairs:
        raise ValueError(f"load_range_analysis group table missing metric/bin rows: {missing_pairs}")

    # The four official contribution values must sum to the official decomposition net.
    # This is a consistency check within the load_range_analysis additive decomposition itself.
    for metric in ["mae", "mse", "smape"]:
        sub = filtered[filtered["metric"] == metric].set_index("raw_bin").reindex(RAW_BINS)
        if sub["observed_gain_contribution"].isna().any():
            raise ValueError(f"Missing observed_gain_contribution values for metric={metric}.")
        net = sub["observed_gain_contribution"].sum()
        print(f"load_range_analysis validation | {metric.upper()} contribution sum = {net:+.6f}")

    share_vectors = {}
    for metric in ["mae", "mse", "smape"]:
        sub = (
            filtered[filtered["metric"] == metric]
            .set_index("raw_bin")
            .reindex(RAW_BINS)
        )
        shares = sub["mean_meter_row_share"].astype(float).to_numpy()
        if pd.isna(shares).any():
            raise ValueError(f"Missing mean_meter_row_share values for metric={metric}.")
        if abs(shares.sum() - 1.0) > 0.01:
            raise ValueError(
                f"Observed-load shares for {metric} sum to {shares.sum():.6f}, "
                "which is not approximately 1."
            )
        share_vectors[metric] = shares

    reference = share_vectors["mae"]
    for metric in ["mse", "smape"]:
        if not np.allclose(reference, share_vectors[metric], atol=1e-10, rtol=0):
            raise ValueError(
                "Observed-load shares differ across metrics; the figure will not "
                "silently reuse one metric's shares."
            )

    print(
        "load_range_analysis validation | observed-load shares = "
        + ", ".join(f"{100*s:.2f}%" for s in reference)
    )


def prepare_drift_quartiles(df):
    """
    Assign temporal-drift quartiles once at the unique-household level, then merge
    that same assignment back to MAE, RMSE and sMAPE rows. This prevents the
    quartile boundaries from being re-estimated separately by criterion.
    """
    required = {
        "entity_id", "lead", "criterion", "support_scope",
        DRIFT_PREDICTOR, DRIFT_GAIN_COL,
    }
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"temporal_drift_analysis drift table missing columns: {sorted(missing)}")

    filtered = df[
        (df["lead"] == LEAD) &
        (df["support_scope"] == "native") &
        (df["criterion"].isin(CRITERIA))
    ].copy()

    if filtered.empty:
        raise ValueError("No temporal_drift_analysis h=12 native drift rows found.")

    # Each household's temporal-drift predictor should be identical across criteria.
    drift_nunique = filtered.groupby("entity_id")[DRIFT_PREDICTOR].nunique(dropna=False)
    bad = drift_nunique[drift_nunique != 1]
    if not bad.empty:
        raise ValueError(
            "Temporal-drift predictor is not constant across criteria for some households: "
            f"{bad.index[:10].tolist()}"
        )

    entity_drift = (
        filtered[["entity_id", DRIFT_PREDICTOR]]
        .drop_duplicates("entity_id")
        .sort_values(["entity_id"])
        .reset_index(drop=True)
    )

    if len(entity_drift) != EXPECTED_CER_METERS:
        raise ValueError(
            f"Expected {EXPECTED_CER_METERS} CER meters for drift quartiles, "
            f"found {len(entity_drift)}."
        )

    # labels=False avoids category-label edge cases; +1 gives Q1..Q4.
    entity_drift["quartile"] = (
        pd.qcut(entity_drift[DRIFT_PREDICTOR], 4, labels=False, duplicates="raise") + 1
    )

    out = filtered.merge(
        entity_drift[["entity_id", "quartile"]],
        on="entity_id",
        how="left",
        validate="many_to_one",
    )

    # Validate one row per entity per criterion.
    _assert_unique_rows(out, ["entity_id", "criterion"], "temporal_drift_analysis drift outcome table")

    # Validate quartile means against the formal Appendix L values.
    for criterion in CRITERIA:
        means = (
            out[out["criterion"] == criterion]
            .groupby("quartile")[DRIFT_GAIN_COL]
            .mean()
            .reindex([1, 2, 3, 4])
            .tolist()
        )
        expected = EXPECTED_DRIFT_QUARTILE_MEANS[criterion]
        tol = DRIFT_MEAN_TOLERANCE[criterion]
        for q, (observed, target) in enumerate(zip(means, expected), start=1):
            if abs(observed - target) > tol:
                raise ValueError(
                    f"Appendix L validation failed for {criterion.upper()} Q{q}: "
                    f"computed={observed:.6f}, expected≈{target:.6f}, tolerance={tol}"
                )

    print("temporal_drift_analysis validation | quartile means reproduce Appendix L within tolerance.")
    return out



# ---------------------------------------------------------------------------
# Robust input-file resolution
# ---------------------------------------------------------------------------
def resolve_formal_input_file(filename, table_dir):
    """Resolve a reproduced analysis table from its canonical output directory."""
    candidate = Path(table_dir) / filename
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Missing reproduced analysis table: {candidate}. "
            "Run the corresponding analysis stage before generating report figures."
        )
    print(f"Resolved input: {candidate}")
    return candidate


# ---------------------------------------------------------------------------
# load_range_analysis loaders
# ---------------------------------------------------------------------------
def load_meter_bin_table():
    path = resolve_formal_input_file(METER_BIN_FILE, LOAD_RANGE_ANALYSIS_TABLE_DIR)
    df = pd.read_csv(path)
    validate_load_range_analysis_meter_table(df)
    return df[df["comparison_id"] == COMPARISON_ID].copy()


def load_group_bin_table():
    path = resolve_formal_input_file(GROUP_BIN_FILE, LOAD_RANGE_ANALYSIS_TABLE_DIR)
    df = pd.read_csv(path)
    validate_load_range_analysis_group_table(df)
    return df[
        (df["comparison_id"] == COMPARISON_ID) &
        (df["population"] == POPULATION)
    ].copy()


def get_contributions(group_dt, metric):
    contribs = []
    for raw_bin in RAW_BINS:
        row = group_dt[(group_dt["raw_bin"] == raw_bin) & (group_dt["metric"] == metric)]
        contribs.append(row["observed_gain_contribution"].iloc[0])
    return contribs


def get_within_bin(group_dt, metric):
    vals = []
    for raw_bin in RAW_BINS:
        row = group_dt[(group_dt["raw_bin"] == raw_bin) & (group_dt["metric"] == metric)]
        vals.append(row["exposure_weighted_conditional_gain"].iloc[0])
    return vals


def get_observed_load_shares(group_dt):
    """Return the official Overall CER observed-load shares in RAW_BINS order."""
    shares = []
    for raw_bin in RAW_BINS:
        row = group_dt[
            (group_dt["raw_bin"] == raw_bin) &
            (group_dt["metric"] == "mae")
        ]
        if len(row) != 1:
            raise ValueError(
                f"Expected exactly one Overall CER MAE row for raw_bin={raw_bin!r}; "
                f"found {len(row)}."
            )
        shares.append(float(row["mean_meter_row_share"].iloc[0]))
    return shares


def get_boxplot_axis_labels(group_dt):
    shares = get_observed_load_shares(group_dt)
    return [
        f"{label}\n({100 * share:.0f}%)"
        for label, share in zip(BIN_AXIS_LABELS, shares)
    ]


def get_cumline_axis_labels(group_dt):
    shares = get_observed_load_shares(group_dt)
    return [
        f"{label}\n({100 * share:.0f}%)"
        for label, share in zip(CUMLINE_AXIS_LABELS, shares)
    ]


# ---------------------------------------------------------------------------
# temporal_drift_analysis loader
# ---------------------------------------------------------------------------
def load_drift_outcome_table():
    usecols = ["entity_id", "lead", "criterion", "support_scope", DRIFT_PREDICTOR, DRIFT_GAIN_COL]
    path = resolve_formal_input_file(DRIFT_OUTCOME_FILE, TEMPORAL_DRIFT_ANALYSIS_TABLE_DIR)
    df = pd.read_csv(path, usecols=usecols)
    return prepare_drift_quartiles(df)


# ---------------------------------------------------------------------------
# Figure 4.8: within-bin gain distribution
# ---------------------------------------------------------------------------
def build_boxplot_figure(meter_dt: pd.DataFrame, group_dt: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    xlabels = get_boxplot_axis_labels(group_dt)

    for ax, metric in zip(axes, ["mae", "mse", "smape"]):
        col = METRIC_COLUMN[metric]
        unit = METRIC_UNIT[metric]

        box_data = [meter_dt.loc[meter_dt["raw_bin"] == rb, col].dropna().values for rb in RAW_BINS]

        bp = ax.boxplot(
            box_data, positions=range(4), widths=0.55, patch_artist=True,
            showmeans=True, meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor=NAVY, markersize=6, zorder=5),
            medianprops=dict(color=GOLD, linewidth=2.2),
            whiskerprops=dict(color="#5A5A5A", linewidth=1.1),
            capprops=dict(color="#5A5A5A", linewidth=1.1),
            boxprops=dict(linewidth=1.1, edgecolor="#3A3A3A"),
            flierprops=dict(marker="o", markersize=2.5, markerfacecolor="#9A9A9A", markeredgecolor="none", alpha=0.5),
        )
        for patch, colour in zip(bp["boxes"], BIN_COLOURS):
            patch.set_facecolor(colour)
            patch.set_alpha(0.85)

        ax.axhline(0, color=ZERO_BLUE, linestyle="--", linewidth=1.1, zorder=1)
        ax.set_title(METRIC_TITLE[metric], fontsize=13, fontweight="bold", color=NAVY)
        ax.set_ylabel(f"Within range gain ({unit})", fontsize=10)
        ax.set_xticks(range(4))
        ax.set_xticklabels(xlabels, fontsize=8.8)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


    legend_elements = [
        plt.Line2D([0], [0], color=GOLD, lw=2.2, label="Median"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="white", markeredgecolor=NAVY, markersize=7, label="Mean"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#6D93B8", alpha=0.85, edgecolor="#3A3A3A", label="Box = IQR (household gains within range)"),
        plt.Line2D([0], [0], color=ZERO_BLUE, linestyle="--", lw=1.1, label="No gain"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Scratch Full over Scratch Limited gain by observed load range (h = 12)",
                 fontsize=14, fontweight="bold", color=NAVY, y=1.03)
    plt.tight_layout(rect=[0, 0.08, 1, 0.94])
    return fig


# ---------------------------------------------------------------------------
# Figure 4.9: cumulative contribution to net gain
# ---------------------------------------------------------------------------
def build_cumline_figure(group_dt: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    xlabels = ["Start"] + get_cumline_axis_labels(group_dt)

    for ax, metric in zip(axes, ["mae", "mse", "smape"]):
        unit = METRIC_UNIT[metric]
        fmt = METRIC_FMT[metric]

        contribs = get_contributions(group_dt, metric)
        net = sum(contribs)
        cum = [0.0]
        for c in contribs:
            cum.append(cum[-1] + c)

        xs = list(range(len(cum)))
        ax.plot(xs, cum, color=NAVY, linewidth=2, zorder=3)
        ax.scatter(xs[1:], cum[1:], color=BIN_COLOURS, edgecolor=NAVY, linewidth=1, s=90, zorder=4)
        ax.scatter([0], [0], color="white", edgecolor=NAVY, linewidth=1.3, s=90, zorder=4)
        ax.scatter([xs[-1]], [cum[-1]], color=GOLD, edgecolor=NAVY, linewidth=1.3, s=130, zorder=5)

        y_range = max(cum) - min(cum) or 1
        offset = y_range * 0.14
        for i in range(1, len(cum)):
            step = contribs[i - 1]
            va = "bottom" if step >= 0 else "top"
            # Prefix with a delta to make explicit this label is the
            # incremental contribution added at this step, not the
            # cumulative y-position of the point itself.
            ax.annotate("\u0394 " + fmt.format(step), (xs[i], cum[i]), textcoords="offset points",
                        xytext=(0, 9 if step >= 0 else -9), ha="center", va=va, fontsize=8, color=NAVY)

        ax.axhline(0, color=ZERO_BLUE, linestyle="--", linewidth=1.1, zorder=1)
        ax.set_title(METRIC_TITLE[metric], fontsize=13, fontweight="bold", color=NAVY)
        ax.set_ylabel(f"Cumulative contribution ({unit})", fontsize=9.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(xlabels, fontsize=9.5)
        ax.set_ylim(min(cum) - offset * 3, max(cum) + offset * 3)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.annotate(f"Net gain\n{fmt.format(net)} {unit}", (xs[-1], cum[-1]), textcoords="offset points",
                    xytext=(30, 0), ha="left", va="center", fontsize=8.6, fontweight="bold", color=NAVY)

    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor=NAVY, markersize=7, label="Start (0)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=GOLD, markeredgecolor=NAVY, markersize=8, label="Net gain (running total)"),
        plt.Line2D([0], [0], color=NAVY, lw=2, label="Running total across load ranges"),
        plt.Line2D([0], [0], color=ZERO_BLUE, linestyle="--", lw=1.1, label="No gain"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Cumulative contribution to Scratch Full over Scratch Limited net gain (h = 12)",
                 fontsize=14, fontweight="bold", color=NAVY, y=1.04)
    plt.tight_layout(rect=[0, 0.08, 1, 0.93])
    return fig


# ---------------------------------------------------------------------------
# Figure 4.10: L-F gain across temporal-drift quartiles
# ---------------------------------------------------------------------------
def build_drift_quartile_figure(drift_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))

    for ax, criterion in zip(axes, CRITERIA):
        unit = CRITERION_UNIT[criterion]
        part = drift_df[drift_df["criterion"] == criterion].copy()

        box_data = []
        quartile_labels = []
        quartile_ranges = []
        for q in [1, 2, 3, 4]:
            qdf = part[part["quartile"] == q]
            box_data.append(qdf[DRIFT_GAIN_COL].dropna().values)
            lo = qdf[DRIFT_PREDICTOR].min()
            hi = qdf[DRIFT_PREDICTOR].max()
            quartile_labels.append(f"Q{q}")
            quartile_ranges.append(f"Q{q}: {lo:.1f}\u2013{hi:.1f}%")

        bp = ax.boxplot(
            box_data, positions=range(4), widths=0.55, patch_artist=True,
            showmeans=True, meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor=NAVY, markersize=6, zorder=5),
            medianprops=dict(color=GOLD, linewidth=2.2),
            whiskerprops=dict(color="#5A5A5A", linewidth=1.1),
            capprops=dict(color="#5A5A5A", linewidth=1.1),
            boxprops=dict(linewidth=1.1, edgecolor="#3A3A3A"),
            flierprops=dict(marker="o", markersize=2.5, markerfacecolor="#9A9A9A", markeredgecolor="none", alpha=0.5),
        )
        for patch, colour in zip(bp["boxes"], QUARTILE_COLOURS):
            patch.set_facecolor(colour)
            patch.set_alpha(0.85)

        ax.axhline(0, color=ZERO_BLUE, linestyle="--", linewidth=1.1, zorder=1)
        ax.set_title(CRITERION_TITLE[criterion], fontsize=13, fontweight="bold", color=NAVY)
        ax.set_ylabel(f"L\u2212F absolute gain ({unit})", fontsize=10)
        ax.set_xticks(range(4))
        ax.set_xticklabels(quartile_labels, fontsize=10)
        ax.set_xlabel("Temporal drift quartile", fontsize=9.5)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    legend_elements = [
        plt.Line2D([0], [0], color=GOLD, lw=2.2, label="Median"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="white", markeredgecolor=NAVY, markersize=7, label="Mean"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#6D93B8", alpha=0.85, edgecolor="#3A3A3A", label="Box = IQR (household gains within quartile)"),
        plt.Line2D([0], [0], color=ZERO_BLUE, linestyle="--", lw=1.1, label="No gain"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Scratch Full over Scratch Limited gain across temporal-drift quartiles (h = 12)",
                 fontsize=14, fontweight="bold", color=NAVY, y=1.03)
    plt.tight_layout(rect=[0, 0.1, 1, 0.94])
    print("Quartile boundaries (for caption/note text):", quartile_ranges)
    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Figure 4.8 & 4.4E: load_range_analysis ---
    meter_dt = load_meter_bin_table()
    group_dt = load_group_bin_table()

    fig_d = build_boxplot_figure(meter_dt, group_dt)
    path_d = OUTPUT_DIR / "figure_4_8_lf_gain_by_load_range.png"
    fig_d.savefig(path_d, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {path_d}")

    fig_e = build_cumline_figure(group_dt)
    path_e = OUTPUT_DIR / "figure_4_9_lf_cumulative_load_range_contribution.png"
    fig_e.savefig(path_e, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {path_e}")

    # --- Figure 4.10: temporal_drift_analysis ---
    drift_df = load_drift_outcome_table()
    fig_f = build_drift_quartile_figure(drift_df)
    path_f = OUTPUT_DIR / "figure_4_10_lf_gain_by_drift_quartile.png"
    fig_f.savefig(path_f, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {path_f}")


if __name__ == "__main__":
    main()
