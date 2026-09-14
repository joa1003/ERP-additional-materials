import os as _erp_os
from pathlib import Path as _ERPPath
_ERP_PROJECT_ROOT = _ERPPath(_erp_os.environ.get("ERP_PROJECT_ROOT", ".")).expanduser().resolve()
"""
the submitted report — Direct Transfer over Scratch Full (F−D), h = 12

FINAL MAIN-TEXT FIGURE SCRIPT
-----------------------------
Produces THREE figures only:

Figure 4.11
    figure_4_11_fd_gain_by_load_range.png/.pdf
    Household-level F−D Gain distributions within observed load ranges.
    Metrics: MAE, MSE, sMAPE.
    Percentages shown under the x-axis categories are the h=12 target-test
    observation shares in the four observed-load ranges.

Figure 4.12
    figure_4_12_fd_cumulative_load_range_contribution.png/.pdf
    Overall CER cumulative contribution of observed-load ranges to net F−D Gain.
    Metrics: MAE, MSE, sMAPE.
    Percentages again refer to observed-load-range shares.

Figure 4.13
    figure_4_13_fd_gain_by_drift_quartile.png/.pdf
    Household-level ABSOLUTE F−D Gain distributions across temporal-drift quartiles.
    Metrics: MAE, RMSE, sMAPE.
    The x-axis shows Q1–Q4 only, matching the the L−F figures temporal-drift figure.
    Temporal-drift percentage boundaries are printed to the terminal for caption/note
    checking but are not drawn on the figure.

The relative-gain results remain in Appendix L tables and supporting text.

SIGN CONVENTION
---------------
F−D = Direct Transfer over Scratch Full
Gain = Error(Scratch Full) − Error(Direct Transfer)

Positive Gain -> Direct Transfer has lower error.
Negative Gain -> Scratch Full has lower error.

ANALYSIS LEVEL
--------------
Figures 4.11 and 4.13 are household-level distribution figures.
Figure 4.12 is intentionally an Overall CER additive contribution figure.

DISPLAY-ONLY OUTLIER RULE
-------------------------
Figures 4.11 and 4.13 display outlier markers, matching the the L−F figures boxplot
rule so that visual grammar remains consistent across the L−F and F−D figures.
All households are retained in the calculation of quartiles, whiskers, medians,
means, and outlier markers.

METRIC NOTE
-----------
Load range analysis uses MSE instead of RMSE because squared-error contributions are additive
across observed-load ranges. Temporal drift analysis uses the formally reported MAE/RMSE/sMAPE.

All plotted empirical values are read from the formal Load range analysis / Temporal drift analysis CSV files.
Numerical anchors below are validation checks only; they are not used to draw the
figures.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# =============================================================================
# Paths
# =============================================================================
ERP_ROOT = Path(f"{_ERP_PROJECT_ROOT}")

# Leave as None for automatic resolution.
# If multiple copies exist at the same priority level, set the corresponding
# variable to the exact validated table directory.
LOAD_RANGE_ANALYSIS_TABLE_DIR = None
TEMPORAL_DRIFT_ANALYSIS_TABLE_DIR = None

OUTPUT_DIR = Path(
    f"{_ERP_PROJECT_ROOT}/outputs/figures/"
    "load_range_analysis_complete_cross_strategy_raw_load_composition_standardised_gain_decomposition"
)

METER_BIN_FILE = "load_range_analysis_h12_meter_four_bin_pairwise_gains_four_seed.csv"
GROUP_BIN_FILE = "load_range_analysis_h12_group_bin_conditional_gain_and_contribution.csv"
DRIFT_OUTCOME_FILE = "temporal_drift_analysis_cross_strategy_drift_outcome_dataset.csv"


# =============================================================================
# Formal comparison settings
# =============================================================================
COMPARISON_ID = "direct_vs_full"
POPULATION = "Overall CER"
LEAD = 12
EXPECTED_CER_METERS = 929

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

CUMLINE_DISPLAY = {
    "actual_le_0_1": "≤0.1\nkWh",
    "actual_gt_0_1_le_0_5": "0.1–0.5\nkWh",
    "actual_gt_0_5_le_1_0": "0.5–1.0\nkWh",
    "actual_gt_1_0": ">1.0\nkWh",
}

LOAD_METRICS = ["mae", "mse", "smape"]
LOAD_GAIN_COL = {
    "mae": "gain_mae",
    "mse": "gain_mse",
    "smape": "gain_smape",
}
LOAD_TITLE = {
    "mae": "MAE",
    "mse": "MSE",
    "smape": "sMAPE",
}
LOAD_UNIT = {
    "mae": "kWh",
    "mse": "kWh²",
    "smape": "pp",
}
LOAD_FMT = {
    "mae": "{:+.4f}",
    "mse": "{:+.4f}",
    "smape": "{:+.2f}",
}

DRIFT_PREDICTOR = "test_mean_abs_change_pct"
DRIFT_CRITERIA = ["mae", "rmse", "smape"]
DRIFT_TITLE = {
    "mae": "MAE",
    "rmse": "RMSE",
    "smape": "sMAPE",
}
DRIFT_UNIT = {
    "mae": "kWh",
    "rmse": "kWh",
    "smape": "pp",
}

PREFERRED_ABS_GAIN_COLS = [
    "gain_direct_vs_full_absolute",
    "direct_vs_full_gain_absolute",
]


# =============================================================================
# Visual language
#
# Locked to the the L−F figures L−F figure script:
#   blue boxes
#   gold median
#   white diamond mean
#   ZERO_BLUE = #2C6EBD
#   dashed no-gain line, linewidth 1.1
# =============================================================================
BIN_COLOURS = ["#A9C2DA", "#6D93B8", "#3B6FA0", "#1C2A3A"]
QUARTILE_COLOURS = ["#A9C2DA", "#6D93B8", "#3B6FA0", "#1C2A3A"]

GOLD = "#B5820B"
ZERO_BLUE = "#2C6EBD"
GRID = "#E2E6EC"
NAVY = "#1C2A3A"
WHISKER = "#5A5A5A"
OUTLIER = "#9A9A9A"


# =============================================================================
# Formal validation anchors — NOT plotting data
# =============================================================================
EXPECTED_LOAD_RANGE_ANALYSIS = {
    "mae": {
        "within": [0.0041, 0.0198, 0.0162, -0.0913],
        "contrib": [0.0009, 0.0094, 0.0026, -0.0139],
        "net": -0.0010,
    },
    "mse": {
        "within": [0.0114, 0.0234, 0.0191, -0.2999],
        "contrib": [0.0024, 0.0111, 0.0031, -0.0456],
        "net": -0.0290,
    },
    "smape": {
        "within": [-2.21, 1.40, -0.48, -8.45],
        "contrib": [-0.47, 0.67, -0.08, -1.28],
        "net": -1.16,
    },
}

EXPECTED_DRIFT_ABS_MEANS = {
    "mae": [-0.0044, -0.0002, 0.0015, -0.0008],
    "rmse": [-0.0025, -0.0057, -0.0111, -0.0409],
    "smape": [-2.416, -1.351, -0.881, 0.001],
}

LOAD_RANGE_ANALYSIS_WITHIN_TOL = {
    "mae": 8e-4,
    "mse": 1.5e-3,
    "smape": 0.05,
}
LOAD_RANGE_ANALYSIS_CONTRIB_TOL = {
    "mae": 8e-4,
    "mse": 1.5e-3,
    "smape": 0.05,
}
LOAD_RANGE_ANALYSIS_NET_TOL = {
    "mae": 8e-4,
    "mse": 1.5e-3,
    "smape": 0.05,
}
DRIFT_ABS_TOL = {
    "mae": 7e-4,
    "rmse": 7e-4,
    "smape": 0.03,
}


# =============================================================================
# Safe input resolution
# =============================================================================
def resolve_input(filename: str, explicit_dir=None) -> Path:
    """
    Resolution priority:
      1. explicit_dir / filename
      2. ERP_ROOT / outputs / tables recursively
      3. ERP_ROOT recursively

    The script never silently chooses among multiple files at the same
    priority level.
    """
    if explicit_dir is not None:
        candidate = Path(explicit_dir).expanduser().resolve() / filename
        if candidate.is_file():
            print(f"Resolved explicit input: {candidate}")
            return candidate
        raise FileNotFoundError(f"Explicit input not found: {candidate}")

    preferred_root = ERP_ROOT / "outputs" / "tables"

    if preferred_root.is_dir():
        preferred = sorted(
            p for p in preferred_root.rglob(filename)
            if p.is_file()
        )

        if len(preferred) == 1:
            print(f"Auto-resolved preferred input: {preferred[0]}")
            return preferred[0]

        if len(preferred) > 1:
            formatted = "\n".join(
                f"  [{i + 1}] {p}"
                for i, p in enumerate(preferred)
            )
            raise RuntimeError(
                f"Multiple copies of {filename!r} found under {preferred_root}.\n"
                "Set LOAD_RANGE_ANALYSIS_TABLE_DIR or TEMPORAL_DRIFT_ANALYSIS_TABLE_DIR explicitly.\n"
                f"Candidates:\n{formatted}"
            )

    matches = sorted(
        p for p in ERP_ROOT.rglob(filename)
        if p.is_file()
    )

    if len(matches) == 1:
        print(f"Auto-resolved input: {matches[0]}")
        return matches[0]

    if len(matches) == 0:
        raise FileNotFoundError(
            f"Could not find {filename!r} under {ERP_ROOT}.\n"
            f"Check with:\nfind '{ERP_ROOT}' -name '{filename}' -print"
        )

    formatted = "\n".join(
        f"  [{i + 1}] {p}"
        for i, p in enumerate(matches)
    )

    raise RuntimeError(
        f"Multiple copies of {filename!r} found under {ERP_ROOT}.\n"
        "The script will not guess which copy is canonical.\n"
        f"Candidates:\n{formatted}"
    )


def require_columns(df: pd.DataFrame, columns, label: str):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(
            f"{label} missing required columns: {missing}"
        )


def assert_unique(df: pd.DataFrame, keys, label: str):
    dup = df.duplicated(keys, keep=False)
    if dup.any():
        sample = (
            df.loc[dup, keys]
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            f"{label}: duplicate rows for {keys}. "
            f"Examples: {sample}"
        )


# =============================================================================
# Load range analysis
# =============================================================================
def load_meter_bin_table():
    path = resolve_input(
        METER_BIN_FILE,
        LOAD_RANGE_ANALYSIS_TABLE_DIR,
    )
    df = pd.read_csv(path)

    require_columns(
        df,
        [
            "comparison_id",
            "raw_bin",
            "gain_mae",
            "gain_mse",
            "gain_smape",
        ],
        METER_BIN_FILE,
    )

    out = df[
        df["comparison_id"].eq(COMPARISON_ID)
    ].copy()

    if out.empty:
        raise ValueError(
            f"No rows for comparison_id={COMPARISON_ID!r}."
        )

    for raw_bin in RAW_BINS:
        if raw_bin not in set(out["raw_bin"].dropna()):
            raise ValueError(
                f"Missing raw_bin={raw_bin!r} "
                f"in {METER_BIN_FILE}."
            )

    return out


def load_group_bin_table():
    path = resolve_input(
        GROUP_BIN_FILE,
        LOAD_RANGE_ANALYSIS_TABLE_DIR,
    )
    df = pd.read_csv(path)

    require_columns(
        df,
        [
            "comparison_id",
            "population",
            "raw_bin",
            "metric",
            "mean_meter_row_share",
            "exposure_weighted_conditional_gain",
            "observed_gain_contribution",
        ],
        GROUP_BIN_FILE,
    )

    out = df[
        df["comparison_id"].eq(COMPARISON_ID)
        & df["population"].eq(POPULATION)
    ].copy()

    if out.empty:
        raise ValueError(
            f"No rows for comparison_id={COMPARISON_ID!r}, "
            f"population={POPULATION!r}."
        )

    assert_unique(
        out,
        ["metric", "raw_bin"],
        "Load range analysis Overall CER table",
    )

    validate_load_range_analysis(out)
    return out


def validate_load_range_analysis(group_df):
    expected_pairs = {
        (metric, raw_bin)
        for metric in LOAD_METRICS
        for raw_bin in RAW_BINS
    }

    actual_pairs = set(
        zip(
            group_df["metric"],
            group_df["raw_bin"],
        )
    )

    missing = sorted(
        expected_pairs - actual_pairs
    )

    if missing:
        raise ValueError(
            f"Load range analysis missing metric/bin pairs: {missing}"
        )

    share_vectors = {}

    for metric in LOAD_METRICS:
        sub = (
            group_df[
                group_df["metric"].eq(metric)
            ]
            .set_index("raw_bin")
            .reindex(RAW_BINS)
        )

        within = (
            sub["exposure_weighted_conditional_gain"]
            .astype(float)
            .to_numpy()
        )

        contrib = (
            sub["observed_gain_contribution"]
            .astype(float)
            .to_numpy()
        )

        shares = (
            sub["mean_meter_row_share"]
            .astype(float)
            .to_numpy()
        )

        net = float(contrib.sum())

        for i, (obs, exp) in enumerate(
            zip(
                within,
                EXPECTED_LOAD_RANGE_ANALYSIS[metric]["within"],
            ),
            start=1,
        ):
            if abs(obs - exp) > LOAD_RANGE_ANALYSIS_WITHIN_TOL[metric]:
                raise ValueError(
                    "Appendix J within-range validation failed: "
                    f"{metric.upper()} bin {i}, "
                    f"observed={obs:+.6f}, "
                    f"expected≈{exp:+.6f}"
                )

        for i, (obs, exp) in enumerate(
            zip(
                contrib,
                EXPECTED_LOAD_RANGE_ANALYSIS[metric]["contrib"],
            ),
            start=1,
        ):
            if abs(obs - exp) > LOAD_RANGE_ANALYSIS_CONTRIB_TOL[metric]:
                raise ValueError(
                    "Appendix J contribution validation failed: "
                    f"{metric.upper()} bin {i}, "
                    f"observed={obs:+.6f}, "
                    f"expected≈{exp:+.6f}"
                )

        if (
            abs(
                net
                - EXPECTED_LOAD_RANGE_ANALYSIS[metric]["net"]
            )
            > LOAD_RANGE_ANALYSIS_NET_TOL[metric]
        ):
            raise ValueError(
                "Appendix J net validation failed: "
                f"{metric.upper()}, "
                f"observed={net:+.6f}, "
                f"expected≈"
                f"{EXPECTED_LOAD_RANGE_ANALYSIS[metric]['net']:+.6f}"
            )

        if abs(float(shares.sum()) - 1.0) > 0.01:
            raise ValueError(
                f"Observed-load shares for {metric} "
                f"sum to {shares.sum():.6f}, "
                "not approximately 1."
            )

        share_vectors[metric] = shares

        print(
            f"LoadRangeAnalysis validation | "
            f"{metric.upper()} contribution sum "
            f"= {net:+.6f}"
        )

    for metric in ["mse", "smape"]:
        if not np.allclose(
            share_vectors["mae"],
            share_vectors[metric],
            atol=1e-10,
            rtol=0,
        ):
            raise ValueError(
                "Observed-load shares differ across metrics."
            )

    print(
        "LoadRangeAnalysis validation | observed-load shares = "
        + ", ".join(
            f"{100 * x:.2f}%"
            for x in share_vectors["mae"]
        )
    )


def get_load_shares(group_df):
    sub = (
        group_df[
            group_df["metric"].eq("mae")
        ]
        .set_index("raw_bin")
        .reindex(RAW_BINS)
    )

    return (
        sub["mean_meter_row_share"]
        .astype(float)
        .to_numpy()
    )


def get_contributions(group_df, metric):
    sub = (
        group_df[
            group_df["metric"].eq(metric)
        ]
        .set_index("raw_bin")
        .reindex(RAW_BINS)
    )

    return (
        sub["observed_gain_contribution"]
        .astype(float)
        .to_numpy()
    )


def get_loadrange_labels(group_df):
    """
    Match earlier Section 4.4 load-range figures:
      label
      (share%)

    Shares are read from the formal Load range analysis Overall CER table.
    They are rounded to whole percentages for the figure, while the exact
    percentages are printed during validation.
    """
    shares = get_load_shares(group_df)

    return [
        f"{BIN_DISPLAY[raw_bin]}\n({100 * share:.0f}%)"
        for raw_bin, share in zip(
            RAW_BINS,
            shares,
        )
    ]


def get_cumline_labels(group_df):
    shares = get_load_shares(group_df)

    return ["Start"] + [
        f"{CUMLINE_DISPLAY[raw_bin]}\n"
        f"({100 * share:.0f}%)"
        for raw_bin, share in zip(
            RAW_BINS,
            shares,
        )
    ]


# =============================================================================
# Temporal drift analysis
# =============================================================================
def normalise_name(name: str) -> str:
    return re.sub(
        r"_+",
        "_",
        re.sub(
            r"[^a-z0-9]+",
            "_",
            str(name).lower(),
        ),
    ).strip("_")


def resolve_fd_absolute_gain_column(columns):
    for name in PREFERRED_ABS_GAIN_COLS:
        if name in columns:
            return name

    candidates = []

    for col in columns:
        norm = normalise_name(col)

        if (
            "gain" in norm
            and "direct" in norm
            and "full" in norm
            and "absolute" in norm
        ):
            candidates.append(col)

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) == 0:
        raise KeyError(
            "Could not resolve F−D absolute gain column.\n"
            f"Available columns:\n{list(columns)}"
        )

    raise RuntimeError(
        "Multiple candidate F−D absolute gain columns "
        f"found: {candidates}"
    )


def load_drift_outcome_table():
    path = resolve_input(
        DRIFT_OUTCOME_FILE,
        TEMPORAL_DRIFT_ANALYSIS_TABLE_DIR,
    )

    df = pd.read_csv(path)

    gain_col = resolve_fd_absolute_gain_column(
        df.columns
    )

    require_columns(
        df,
        [
            "entity_id",
            "lead",
            "criterion",
            "support_scope",
            DRIFT_PREDICTOR,
            gain_col,
        ],
        DRIFT_OUTCOME_FILE,
    )

    out = df[
        df["lead"].eq(LEAD)
        & df["support_scope"].eq("native")
        & df["criterion"].isin(DRIFT_CRITERIA)
    ].copy()

    if out.empty:
        raise ValueError(
            "No h=12 native Temporal drift analysis rows found."
        )

    for col in [
        DRIFT_PREDICTOR,
        gain_col,
    ]:
        out[col] = pd.to_numeric(
            out[col],
            errors="coerce",
        )

    if out[
        [
            DRIFT_PREDICTOR,
            gain_col,
        ]
    ].isna().any().any():
        raise ValueError(
            "Missing/non-numeric Temporal drift analysis "
            "drift or gain values."
        )

    # Temporal drift is a household property.
    # Assign quartiles ONCE at unique-household level.
    predictor_count = (
        out.groupby("entity_id")[
            DRIFT_PREDICTOR
        ]
        .nunique(dropna=False)
    )

    bad = predictor_count[
        predictor_count.ne(1)
    ]

    if not bad.empty:
        raise ValueError(
            "Temporal-drift predictor is not "
            "identical across criteria for: "
            f"{bad.index[:10].tolist()}"
        )

    entity_drift = (
        out[
            [
                "entity_id",
                DRIFT_PREDICTOR,
            ]
        ]
        .drop_duplicates("entity_id")
        .sort_values("entity_id")
        .reset_index(drop=True)
    )

    if len(entity_drift) != EXPECTED_CER_METERS:
        raise ValueError(
            f"Expected {EXPECTED_CER_METERS} households, "
            f"found {len(entity_drift)}."
        )

    entity_drift["quartile"] = (
        pd.qcut(
            entity_drift[DRIFT_PREDICTOR],
            4,
            labels=False,
            duplicates="raise",
        )
        + 1
    )

    out = out.merge(
        entity_drift[
            [
                "entity_id",
                "quartile",
            ]
        ],
        on="entity_id",
        how="left",
        validate="many_to_one",
    )

    assert_unique(
        out,
        ["entity_id", "criterion"],
        "Temporal drift analysis F−D drift table",
    )

    validate_temporal_drift_analysis(
        out,
        gain_col,
    )

    print_drift_quartile_boundaries(out)

    return out, gain_col


def validate_temporal_drift_analysis(df, gain_col):
    for criterion in DRIFT_CRITERIA:
        part = df[
            df["criterion"].eq(criterion)
        ]

        observed_means = (
            part.groupby("quartile")[gain_col]
            .mean()
            .reindex([1, 2, 3, 4])
            .to_numpy()
        )

        for q, (obs, exp) in enumerate(
            zip(
                observed_means,
                EXPECTED_DRIFT_ABS_MEANS[criterion],
            ),
            start=1,
        ):
            if (
                abs(obs - exp)
                > DRIFT_ABS_TOL[criterion]
            ):
                raise ValueError(
                    "Appendix L absolute validation failed: "
                    f"{criterion.upper()} Q{q}, "
                    f"observed={obs:+.6f}, "
                    f"expected≈{exp:+.6f}"
                )

    print(
        "TemporalDriftAnalysis validation | "
        f"absolute gain column = {gain_col}"
    )
    print(
        "TemporalDriftAnalysis validation | "
        "Appendix L absolute quartile means reproduced."
    )


def print_drift_quartile_boundaries(drift_df):
    """
    Boundaries are printed for checking/caption use only.
    They are intentionally NOT placed on Figure 4.13, matching the L−F figures.
    """
    unique = (
        drift_df[
            [
                "entity_id",
                DRIFT_PREDICTOR,
                "quartile",
            ]
        ]
        .drop_duplicates("entity_id")
    )

    boundaries = []

    for q in [1, 2, 3, 4]:
        qdf = unique[
            unique["quartile"].eq(q)
        ]

        lo = float(
            qdf[DRIFT_PREDICTOR].min()
        )
        hi = float(
            qdf[DRIFT_PREDICTOR].max()
        )

        boundaries.append(
            f"Q{q}: {lo:.1f}–{hi:.1f}%"
        )

    print(
        "Temporal-drift quartile boundaries "
        "(caption/note only): "
        + "; ".join(boundaries)
    )


# =============================================================================
# Plot helpers
# =============================================================================
def clean_axis(ax):
    ax.grid(
        axis="y",
        color=GRID,
        linewidth=0.8,
        zorder=0,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_household_boxplot(
    ax,
    data,
    colours,
):
    """
    Outlier markers are displayed, matching the the L−F figures boxplot rule.
    All observations remain in the data passed to matplotlib; mean/median/IQR
    are calculated from the full household distribution.
    """
    bp = ax.boxplot(
        data,
        positions=range(4),
        widths=0.55,
        patch_artist=True,
        showmeans=True,
        showfliers=True,
        whis=1.5,
        meanprops=dict(
            marker="D",
            markerfacecolor="white",
            markeredgecolor=NAVY,
            markersize=6,
            zorder=5,
        ),
        medianprops=dict(
            color=GOLD,
            linewidth=2.2,
        ),
        whiskerprops=dict(
            color=WHISKER,
            linewidth=1.1,
        ),
        capprops=dict(
            color=WHISKER,
            linewidth=1.1,
        ),
        boxprops=dict(
            linewidth=1.1,
            edgecolor="#3A3A3A",
        ),
        flierprops=dict(
            marker="o",
            markersize=2.5,
            markerfacecolor=OUTLIER,
            markeredgecolor="none",
            alpha=0.5,
        ),
    )

    for patch, colour in zip(
        bp["boxes"],
        colours,
    ):
        patch.set_facecolor(colour)
        patch.set_alpha(0.85)

    return bp


def add_box_legend(
    fig,
    context,
):
    if context == "range":
        box_label = (
            "Box = IQR "
            "(household gains within range)"
        )
    else:
        box_label = (
            "Box = IQR "
            "(household gains within quartile)"
        )

    handles = [
        Line2D(
            [0],
            [0],
            color=GOLD,
            lw=2.2,
            label="Median",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="white",
            markeredgecolor=NAVY,
            markersize=7,
            label="Mean",
        ),
        Patch(
            facecolor="#6D93B8",
            alpha=0.85,
            edgecolor="#3A3A3A",
            label=box_label,
        ),
        Line2D(
            [0],
            [0],
            color=ZERO_BLUE,
            linestyle="--",
            lw=1.1,
            label="No gain",
        ),
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.055),
    )


def add_no_gain_line(ax):
    """
    Exact the L−F figures no-gain visual:
      color #2C6EBD
      dashed
      linewidth 1.1
    """
    ax.axhline(
        0,
        color=ZERO_BLUE,
        linestyle="--",
        linewidth=1.1,
        zorder=1,
    )


def save_figure(
    fig,
    stem,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = (
        OUTPUT_DIR
        / f"{stem}.png"
    )

    pdf_path = (
        OUTPUT_DIR
        / f"{stem}.pdf"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


# =============================================================================
# Figure 4.11
# Household-level F−D gain by observed load range
# =============================================================================
def build_loadrange_boxplot(
    meter_df,
    group_df,
):
    xlabels = get_loadrange_labels(
        group_df
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14, 4.8),
    )

    for ax, metric in zip(
        axes,
        LOAD_METRICS,
    ):
        col = LOAD_GAIN_COL[metric]

        data = [
            pd.to_numeric(
                meter_df.loc[
                    meter_df["raw_bin"].eq(
                        raw_bin
                    ),
                    col,
                ],
                errors="coerce",
            )
            .dropna()
            .to_numpy()
            for raw_bin in RAW_BINS
        ]

        if any(
            len(values) == 0
            for values in data
        ):
            raise ValueError(
                "Empty household-level "
                f"load-range data for {metric}."
            )

        draw_household_boxplot(
            ax,
            data,
            BIN_COLOURS,
        )

        add_no_gain_line(ax)

        ax.set_title(
            LOAD_TITLE[metric],
            fontsize=13,
            fontweight="bold",
            color=NAVY,
        )

        ax.set_ylabel(
            "Within range gain "
            f"({LOAD_UNIT[metric]})",
            fontsize=10,
        )

        ax.set_xticks(
            range(4)
        )

        ax.set_xticklabels(
            xlabels,
            fontsize=8.8,
        )

        clean_axis(ax)

    add_box_legend(
        fig,
        "range",
    )

    fig.suptitle(
        "Direct Transfer over Scratch Full gain "
        "by observed load range (h = 12)",
        fontsize=14,
        fontweight="bold",
        color=NAVY,
        y=1.03,
    )

    fig.tight_layout(
        rect=[
            0,
            0.09,
            1,
            0.94,
        ]
    )

    return fig


# =============================================================================
# Figure 4.12
# Overall CER cumulative contribution
# =============================================================================
def build_cumulative_figure(
    group_df,
):
    xlabels = get_cumline_labels(
        group_df
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17, 4.8),
    )

    for ax, metric in zip(
        axes,
        LOAD_METRICS,
    ):
        contributions = get_contributions(
            group_df,
            metric,
        )

        cumulative = np.concatenate(
            [
                [0.0],
                np.cumsum(
                    contributions
                ),
            ]
        )

        xs = np.arange(5)
        fmt = LOAD_FMT[metric]
        unit = LOAD_UNIT[metric]

        ax.plot(
            xs,
            cumulative,
            color=NAVY,
            linewidth=2.0,
            zorder=3,
        )

        ax.scatter(
            [0],
            [0],
            facecolor="white",
            edgecolor=NAVY,
            linewidth=1.3,
            s=90,
            zorder=4,
        )

        ax.scatter(
            xs[1:-1],
            cumulative[1:-1],
            color=BIN_COLOURS[:-1],
            edgecolor=NAVY,
            linewidth=1.0,
            s=90,
            zorder=4,
        )

        ax.scatter(
            [xs[-1]],
            [cumulative[-1]],
            color=GOLD,
            edgecolor=NAVY,
            linewidth=1.3,
            s=130,
            zorder=5,
        )

        add_no_gain_line(ax)

        for i, step in enumerate(
            contributions,
            start=1,
        ):
            ax.annotate(
                "Δ "
                + fmt.format(
                    float(step)
                ),
                (
                    xs[i],
                    cumulative[i],
                ),
                textcoords="offset points",
                xytext=(
                    0,
                    14 if (metric == "smape" and i == 3) else (9 if step >= 0 else -9),
                ),
                ha="center",
                va=(
                    "bottom"
                    if step >= 0
                    else "top"
                ),
                fontsize=8,
                color=NAVY,
            )

        net = float(
            cumulative[-1]
        )

        ax.annotate(
            "Net gain\n"
            f"{fmt.format(net)} {unit}",
            (
                xs[-1],
                cumulative[-1],
            ),
            textcoords="offset points",
            xytext=(30, 0),
            ha="left",
            va="center",
            fontsize=8.6,
            fontweight="bold",
            color=NAVY,
        )

        yrange = float(
            cumulative.max()
            - cumulative.min()
        )

        if yrange == 0:
            yrange = max(
                abs(net),
                1.0,
            )

        ax.set_title(
            LOAD_TITLE[metric],
            fontsize=13,
            fontweight="bold",
            color=NAVY,
        )

        ax.set_ylabel(
            "Cumulative contribution "
            f"({unit})",
            fontsize=9.5,
        )

        ax.set_xticks(
            xs
        )

        ax.set_xticklabels(
            xlabels,
            fontsize=9.0,
            rotation=0,
        )

        ax.set_xlim(
            -0.3,
            4.75,
        )

        ax.set_ylim(
            float(
                cumulative.min()
            )
            - 0.45 * yrange
            - 1e-12,
            float(
                cumulative.max()
            )
            + 0.45 * yrange
            + 1e-12,
        )

        clean_axis(ax)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="white",
            markeredgecolor=NAVY,
            markersize=7,
            label="Start (0)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=GOLD,
            markeredgecolor=NAVY,
            markersize=8,
            label="Net gain (running total)",
        ),
        Line2D(
            [0],
            [0],
            color=NAVY,
            lw=2.0,
            label=(
                "Running total across "
                "observed load ranges"
            ),
        ),
        Line2D(
            [0],
            [0],
            color=ZERO_BLUE,
            linestyle="--",
            lw=1.1,
            label="No gain",
        ),
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.055),
    )

    fig.suptitle(
        "Cumulative contribution to "
        "Direct Transfer over Scratch Full "
        "net gain (h = 12)",
        fontsize=14,
        fontweight="bold",
        color=NAVY,
        y=1.04,
    )

    fig.tight_layout(
        rect=[
            0,
            0.09,
            1,
            0.93,
        ]
    )

    return fig


# =============================================================================
# Figure 4.13
# Household-level absolute F−D gain across temporal-drift quartiles
# =============================================================================
def build_drift_boxplot(
    drift_df,
    gain_col,
):
    # Match the L−F figures:
    # show Q1–Q4 only.
    qlabels = [
        "Q1",
        "Q2",
        "Q3",
        "Q4",
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14, 4.8),
    )

    for ax, criterion in zip(
        axes,
        DRIFT_CRITERIA,
    ):
        part = drift_df[
            drift_df["criterion"].eq(
                criterion
            )
        ]

        data = [
            part.loc[
                part["quartile"].eq(q),
                gain_col,
            ]
            .dropna()
            .to_numpy()
            for q in [1, 2, 3, 4]
        ]

        if any(
            len(values) == 0
            for values in data
        ):
            raise ValueError(
                "Empty household-level drift data "
                f"for {criterion}."
            )

        draw_household_boxplot(
            ax,
            data,
            QUARTILE_COLOURS,
        )

        add_no_gain_line(ax)

        ax.set_title(
            DRIFT_TITLE[criterion],
            fontsize=13,
            fontweight="bold",
            color=NAVY,
        )

        ax.set_ylabel(
            "F−D absolute gain "
            f"({DRIFT_UNIT[criterion]})",
            fontsize=10,
        )

        ax.set_xticks(
            range(4)
        )

        ax.set_xticklabels(
            qlabels,
            fontsize=10,
        )

        clean_axis(ax)

    # One shared x-axis label, rather than repeating the longer definition
    # three times. The metric definition remains in the figure caption/note.
    fig.supxlabel(
        "Temporal drift quartile",
        fontsize=9.5,
        y=0.095,
    )

    add_box_legend(
        fig,
        "quartile",
    )

    fig.suptitle(
        "Direct Transfer over Scratch Full gain "
        "across temporal drift quartiles (h = 12)",
        fontsize=14,
        fontweight="bold",
        color=NAVY,
        y=1.03,
    )

    fig.tight_layout(
        rect=[
            0,
            0.13,
            1,
            0.94,
        ]
    )

    return fig


# =============================================================================
# Main
# =============================================================================
def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 88)
    print(
        "F−D | "
        "Direct Transfer over Scratch Full | h=12"
    )
    print(
        "Positive F−D Gain = "
        "Direct Transfer lower error."
    )
    print(
        "Negative F−D Gain = "
        "Scratch Full lower error."
    )
    print(
        "Display rule: boxplot flier markers are shown, matching the L−F figures; "
        "all households remain in calculations."
    )
    print("=" * 88)

    meter_df = load_meter_bin_table()
    group_df = load_group_bin_table()

    save_figure(
        build_loadrange_boxplot(
            meter_df,
            group_df,
        ),
        "figure_4_11_fd_gain_by_load_range",
    )

    save_figure(
        build_cumulative_figure(
            group_df
        ),
        "figure_4_12_fd_cumulative_load_range_contribution",
    )

    drift_df, abs_gain_col = (
        load_drift_outcome_table()
    )

    save_figure(
        build_drift_boxplot(
            drift_df,
            abs_gain_col,
        ),
        "figure_4_13_fd_gain_by_drift_quartile",
    )

    print(
        "\nAll THREE main-text F−D figures completed."
    )
    print(
        "is intentionally omitted."
    )
    print(
        f"Output directory: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
