from pathlib import Path
import os
_ERP_PROJECT_ROOT = Path(os.environ.get("ERP_PROJECT_ROOT", ".")).expanduser().resolve()
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# LCL R1-R4 eligibility and cohort assertion
# Scope: LCL eligibility.
#
# Output files:
#   lcl_p0_final_r3_r4_selection_summary().csv
#   lcl_p0_final_r3_r4_exclusion_detail().csv
#   lcl_p0_final_eligible_households().csv
# ============================================================

PROJECT_ROOT = Path(str(_ERP_PROJECT_ROOT))

PATHS = {
    "design_lock": (
        PROJECT_ROOT / "documentation" / "reproducibility_design_contract.md"
    ),
    "lcl_raw": (
        PROJECT_ROOT / "data" / "raw" / "LCL" / "csv"
        / "data_collection" / "data_tables" / "consumption_n.csv"
    ),
    "lcl_time_map": (
        PROJECT_ROOT / "data" / "processed" / "time_aligned"
        / "lcl_full_local_clock_map.parquet"
    ),
}

OUT_DIR_TABLES = PROJECT_ROOT / "outputs" / "tables" / "horizon_impact"
OUT_DIR_META = PROJECT_ROOT / "outputs" / "metadata" / "sample_construction"

OUT_SELECTION_SUMMARY = (
    OUT_DIR_TABLES / "lcl_p0_final_r3_r4_selection_summary.csv"
)
OUT_EXCLUSION_DETAIL = (
    OUT_DIR_TABLES / "lcl_p0_final_r3_r4_exclusion_detail.csv"
)
OUT_FINAL_ELIGIBLE = (
    OUT_DIR_TABLES / "lcl_p0_final_eligible_households_new.csv"
)
OUT_R3_LOWEST30 = OUT_DIR_TABLES / "lcl_rule3_lowest_30_training_window_counts.csv"
OUT_R4_TRADEOFF = OUT_DIR_TABLES / "lcl_rule4_missing_window_threshold_comparison.csv"
OUT_R3_FIGURE = PROJECT_ROOT / "outputs" / "figures" / "eligibility" / "lcl_rule3_training_window_counts.png"
OUT_META = OUT_DIR_META / "DataPreparation_r1_r4_confirmation.md"

for p in (OUT_SELECTION_SUMMARY, OUT_EXCLUSION_DETAIL, OUT_FINAL_ELIGIBLE, OUT_R3_LOWEST30, OUT_R4_TRADEOFF, OUT_R3_FIGURE, OUT_META):
    if p.exists():
        raise FileExistsError(
            "Data preparation stopped because a required  output already "
            f"exists and must not be overwritten:\n{p}"
        )

# ------------------------------------------------------------
# Constants (locked)
# ------------------------------------------------------------
LOOKBACK = 48
HORIZONS = [1, 12, 48]
COVERAGE_MIN = 0.90
ROUND_DP = 3
R3_THRESHOLD = 100
R4_THRESHOLD = 0.05

P0_START = pd.Timestamp("2013-01-01 00:00:00")
P0_END = pd.Timestamp("2013-12-31 23:30:00")

SPLIT_BOUNDS = {
    "train": (
        pd.Timestamp("2013-01-02 00:00:00"),
        pd.Timestamp("2013-09-13 18:30:00"),
    ),
    "val": (
        pd.Timestamp("2013-09-13 19:00:00"),
        pd.Timestamp("2013-10-20 04:00:00"),
    ),
    "test": (
        pd.Timestamp("2013-10-20 04:30:00"),
        pd.Timestamp("2013-12-31 23:30:00"),
    ),
}

EXPECTED = {
    "raw_households": 4173,
    "p0_positions": 17520,
    "r1_pass": 3887,
    "r1_r2_pass": 3885,
    "r2_fail_ids": {"N0498", "N1603"},
    "r3_fail_ids": {"N2035", "N2840", "N3080", "N3458", "N3962"},
    "r4_fail_count": 37,
    "overlap_count": 0,
    "final_count": 3843,
}

R4_FAIL_IDS_EXPECTED = {
    "N0321", "N0471", "N0501", "N0522", "N0543", "N0600", "N0800",
    "N0817", "N0959", "N1077", "N1490", "N1746", "N1818", "N1851",
    "N2041", "N2076", "N2114", "N2124", "N2161", "N2274", "N2402",
    "N2419", "N2678", "N2805", "N2831", "N2895", "N2943", "N3044",
    "N3214", "N3246", "N3281", "N3361", "N3381", "N3517", "N3601",
    "N3730", "N4084",
}

NATIVE_TRAIN_CANDIDATES = {1: 12230, 12: 12219, 48: 12183}

# ------------------------------------------------------------
# 1. Canonical P0 GMT grid, using explicit raw_row_index
# ------------------------------------------------------------
print("=" * 78)
print("STEP 1 — CANONICAL P0 GRID (EXPLICIT raw_row_index)")
print("=" * 78)

time_map = pd.read_parquet(
    PATHS["lcl_time_map"], columns=["raw_row_index", "timestamp_gmt"]
)
gmt_all = pd.to_datetime(time_map["timestamp_gmt"], errors="raise")
p0_mask_full = (gmt_all >= P0_START) & (gmt_all <= P0_END)

p0_raw_indices = (
    time_map.loc[p0_mask_full, "raw_row_index"].astype(np.int64).to_numpy()
)
GRID = pd.DatetimeIndex(gmt_all.to_numpy()[p0_mask_full.to_numpy()])

if len(GRID) != EXPECTED["p0_positions"]:
    raise AssertionError(
        f"Expected {EXPECTED['p0_positions']:,} P0 positions, "
        f"found {len(GRID):,}."
    )
if GRID.has_duplicates or not GRID.is_monotonic_increasing:
    raise ValueError("Canonical P0 GMT grid is not unique/ordered.")
grid_step = GRID.to_series().diff().dropna().unique()
if not (len(grid_step) == 1 and grid_step[0] == pd.Timedelta("30min")):
    raise ValueError(f"Non-uniform P0 grid step: {grid_step}")

if not np.array_equal(
    p0_raw_indices, np.arange(p0_raw_indices[0], p0_raw_indices[-1] + 1)
):
    raise ValueError(
        "P0 raw_row_index values are not contiguous; cannot use a "
        "single skiprows/nrows block read."
    )

N_POSITIONS = len(GRID)
FIRST_RAW_ROW = int(p0_raw_indices[0])
print(f"P0 positions confirmed  : {N_POSITIONS:,}")
print(f"P0 raw_row_index range  : {p0_raw_indices[0]} to {p0_raw_indices[-1]}")

# ------------------------------------------------------------
# 2. LCL raw households — population identity
# ------------------------------------------------------------
print("\n" + "=" * 78)
print("STEP 2 — LCL RAW HOUSEHOLDS (GROUP N SOURCE POPULATION)")
print("=" * 78)

raw_header = pd.read_csv(PATHS["lcl_raw"], nrows=0)
households = [c for c in raw_header.columns if c != "GMT"]
if len(households) != EXPECTED["raw_households"]:
    raise AssertionError(
        f"Expected {EXPECTED['raw_households']:,} LCL households, "
        f"found {len(households):,}."
    )
N_HOUSEHOLDS = len(households)
print(f"LCL population source : {PATHS['lcl_raw'].name} (UK Data Service Group N)")
print(f"Household columns     : {N_HOUSEHOLDS:,}")

# ------------------------------------------------------------
# 3. Load only the P0 block via explicit skiprows/nrows
# ------------------------------------------------------------
print("\n" + "=" * 78)
print("STEP 3 — LOAD P0 BLOCK ONLY (raw_row_index-driven skiprows/nrows)")
print("=" * 78)

p0_block = pd.read_csv(
    PATHS["lcl_raw"],
    usecols=households,
    skiprows=range(1, FIRST_RAW_ROW + 1),
    nrows=N_POSITIONS,
    low_memory=False,
)
if len(p0_block) != N_POSITIONS:
    raise AssertionError(
        f"P0 block read {len(p0_block):,} rows, expected {N_POSITIONS:,}."
    )

# Numeric coercion is kept (rather than a raw dtype=float32 read) to
# tolerate stray non-numeric artefacts safely; the intermediate copy
# is released immediately after conversion to limit peak memory.
matrix = p0_block.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
present = np.isfinite(matrix)
del p0_block
print(f"Loaded matrix : {matrix.shape}")

# ------------------------------------------------------------
# 4. R1 — full-period coverage
# ------------------------------------------------------------
print("\n" + "=" * 78)
print("STEP 4 — R1 (FULL-PERIOD COVERAGE >= 90%)")
print("=" * 78)

overall_coverage = present.mean(axis=0)
r1_pass = overall_coverage >= COVERAGE_MIN

static = pd.DataFrame(
    {
        "household_id": households,
        "overall_coverage": overall_coverage,
        "R1_pass": r1_pass,
    }
)

if int(r1_pass.sum()) != EXPECTED["r1_pass"]:
    raise AssertionError(
        f"R1 reproduction failed. Expected {EXPECTED['r1_pass']:,}, "
        f"found {int(r1_pass.sum()):,}."
    )
print(f"R1 pass: {int(r1_pass.sum()):,}")

# ------------------------------------------------------------
# 5. R2 — training-period observed variation
# ------------------------------------------------------------
print("\n" + "=" * 78)
print("STEP 5 — R2 (TRAINING-PERIOD VARIATION > 1 DISTINCT VALUE)")
print("=" * 78)

train_mask = (GRID >= SPLIT_BOUNDS["train"][0]) & (GRID <= SPLIT_BOUNDS["train"][1])
train_rows = np.flatnonzero(train_mask)

train_distinct = np.zeros(N_HOUSEHOLDS, dtype=np.int32)
for j in range(N_HOUSEHOLDS):
    vals = matrix[train_rows, j]
    obs = vals[np.isfinite(vals)]
    if obs.size:
        train_distinct[j] = np.unique(np.round(obs, ROUND_DP)).size

static["train_distinct_values"] = train_distinct
static["R2_pass"] = train_distinct > 1
static["R1_R2_pass"] = static["R1_pass"] & static["R2_pass"]

r2_fail_ids = set(
    static.loc[static["R1_pass"] & ~static["R2_pass"], "household_id"]
)
if r2_fail_ids != EXPECTED["r2_fail_ids"]:
    raise AssertionError(
        f"R2 failure-ID mismatch.\nExpected: {sorted(EXPECTED['r2_fail_ids'])}\n"
        f"Found: {sorted(r2_fail_ids)}"
    )
if int(static["R1_R2_pass"].sum()) != EXPECTED["r1_r2_pass"]:
    raise AssertionError(
        f"R1+R2 reproduction failed. Expected {EXPECTED['r1_r2_pass']:,}, "
        f"found {int(static['R1_R2_pass'].sum()):,}."
    )
print(f"R2 failures  : {sorted(r2_fail_ids)}")
print(f"R1+R2 pass   : {int(static['R1_R2_pass'].sum()):,}")

base_mask = static["R1_R2_pass"].to_numpy()

# ------------------------------------------------------------
# 6. R3 and R4 — computed unconditionally and independently
# ------------------------------------------------------------
print("\n" + "=" * 78)
print("STEP 6 — R3/R4 MULTI-HORIZON WINDOW AUDIT (INDEPENDENT)")
print("=" * 78)


def target_indices_for_horizon(h, split_start, split_end):
    first_target = LOOKBACK - 1 + h
    all_targets = np.arange(first_target, N_POSITIONS, dtype=np.int64)
    ts = GRID[all_targets]
    mask = (ts >= split_start) & (ts <= split_end)
    return all_targets[mask]


candidate_counts = {}
target_cache = {}
input_start_cache = {}
for h in HORIZONS:
    targets = target_indices_for_horizon(h, *SPLIT_BOUNDS["train"])
    starts = targets - h - (LOOKBACK - 1)
    if np.any(starts < 0):
        raise AssertionError(f"Negative input start for h={h}.")
    target_cache[h] = targets
    input_start_cache[h] = starts
    candidate_counts[h] = len(targets)

for h in HORIZONS:
    if candidate_counts[h] != NATIVE_TRAIN_CANDIDATES[h]:
        raise AssertionError(
            f"Native train candidate mismatch at h={h}: "
            f"expected {NATIVE_TRAIN_CANDIDATES[h]:,}, "
            f"observed {candidate_counts[h]:,}."
        )

useful_h = {h: np.zeros(N_HOUSEHOLDS, dtype=np.int64) for h in HORIZONS}
skip_rate_h = {h: np.zeros(N_HOUSEHOLDS, dtype=np.float64) for h in HORIZONS}

BLOCK = 256
for block_start in range(0, N_HOUSEHOLDS, BLOCK):
    block_end = min(block_start + BLOCK, N_HOUSEHOLDS)
    block_slice = slice(block_start, block_end)
    block_present = present[:, block_slice]
    block_values = matrix[:, block_slice]
    width = block_end - block_start

    cum_present = np.vstack(
        [np.zeros((1, width), dtype=np.int32),
         np.cumsum(block_present, axis=0, dtype=np.int32)]
    )
    adjacent_change = (
        block_present[:-1, :] & block_present[1:, :]
        & (block_values[:-1, :] != block_values[1:, :])
    )
    cum_change = np.vstack(
        [np.zeros((1, width), dtype=np.int32),
         np.cumsum(adjacent_change, axis=0, dtype=np.int32)]
    )

    for h in HORIZONS:
        targets = target_cache[h]
        starts = input_start_cache[h]

        input_obs = cum_present[starts + LOOKBACK, :] - cum_present[starts, :]
        input_ok = input_obs == LOOKBACK
        target_ok = block_present[targets, :]

        input_change = cum_change[starts + LOOKBACK - 1, :] - cum_change[starts, :]
        input_varying = input_ok & (input_change > 0)
        target_nonzero = target_ok & (block_values[targets, :] > 0)
        useful = input_varying & target_nonzero

        candidate = candidate_counts[h]
        valid_count = (input_ok & target_ok).sum(axis=0).astype(np.int64)
        lost_count = candidate - valid_count

        useful_h[h][block_slice] = useful.sum(axis=0).astype(np.int64)
        skip_rate_h[h][block_slice] = lost_count / candidate

    print(f"  processed households {block_start + 1:,}-{block_end:,}")

for h in HORIZONS:
    static[f"train_useful_transitions_h{h}"] = useful_h[h]
    static[f"train_skip_rate_h{h}"] = skip_rate_h[h]

useful_matrix = np.column_stack([useful_h[h] for h in HORIZONS])
skip_matrix = np.column_stack([skip_rate_h[h] for h in HORIZONS])
horizon_array = np.array(HORIZONS)

min_useful = useful_matrix.min(axis=1)
min_useful_horizon = horizon_array[useful_matrix.argmin(axis=1)]
max_skip = skip_matrix.max(axis=1)
max_skip_horizon = horizon_array[skip_matrix.argmax(axis=1)]

# Column names follow the public reproducibility design contract.
static["minimum_useful_transitions_formal_horizons"] = min_useful
static["minimum_useful_transition_horizon"] = min_useful_horizon
static["maximum_train_skip_rate_formal_horizons"] = max_skip
static["maximum_train_skip_rate_horizon"] = max_skip_horizon

static["R3_pass"] = (
    static["minimum_useful_transitions_formal_horizons"] >= R3_THRESHOLD
)
static["R4_pass"] = (
    static["maximum_train_skip_rate_formal_horizons"] <= R4_THRESHOLD
)

# ------------------------------------------------------------
# 7. Independent R3 fail set, independent R4 fail set, real overlap
# ------------------------------------------------------------
print("\n" + "=" * 78)
print("STEP 7 — INDEPENDENT R3/R4 FAIL SETS AND GENUINE OVERLAP")
print("=" * 78)

r3_fail_mask = base_mask & ~static["R3_pass"].to_numpy()
r4_fail_mask = base_mask & ~static["R4_pass"].to_numpy()

r3_fail_ids = set(static.loc[r3_fail_mask, "household_id"])
r4_fail_ids = set(static.loc[r4_fail_mask, "household_id"])

if r3_fail_ids != EXPECTED["r3_fail_ids"]:
    raise AssertionError(
        f"R3 failure-ID mismatch.\nExpected: {sorted(EXPECTED['r3_fail_ids'])}\n"
        f"Found: {sorted(r3_fail_ids)}"
    )
if len(r4_fail_ids) != EXPECTED["r4_fail_count"] or r4_fail_ids != R4_FAIL_IDS_EXPECTED:
    missing = R4_FAIL_IDS_EXPECTED - r4_fail_ids
    extra = r4_fail_ids - R4_FAIL_IDS_EXPECTED
    raise AssertionError(
        f"R4 failure-ID mismatch.\nMissing: {sorted(missing)}\nExtra: {sorted(extra)}"
    )

overlap_ids = r3_fail_ids & r4_fail_ids
if len(overlap_ids) != EXPECTED["overlap_count"]:
    raise AssertionError(
        f"R3/R4 overlap mismatch. Expected {EXPECTED['overlap_count']}, "
        f"found {len(overlap_ids)}: {sorted(overlap_ids)}"
    )

union_fail_ids = r3_fail_ids | r4_fail_ids
final_mask = base_mask & static["R3_pass"].to_numpy() & static["R4_pass"].to_numpy()
static["final_eligible"] = final_mask
final_count = int(final_mask.sum())

if final_count != EXPECTED["final_count"]:
    raise AssertionError(
        f"Final cohort mismatch. Expected {EXPECTED['final_count']:,}, "
        f"found {final_count:,}."
    )

identity_check = (
    EXPECTED["r1_r2_pass"] - len(r3_fail_ids) - len(r4_fail_ids) + len(overlap_ids)
)
if identity_check != final_count:
    raise AssertionError(
        f"Identity check failed: {EXPECTED['r1_r2_pass']} - {len(r3_fail_ids)} "
        f"- {len(r4_fail_ids)} + {len(overlap_ids)} = {identity_check}, "
        f"expected {final_count}."
    )

print(f"R2 failures       : n={len(r2_fail_ids)} -> {sorted(r2_fail_ids)}")
print(f"R3 failures       : n={len(r3_fail_ids)} -> {sorted(r3_fail_ids)}")
print(f"R4 failures       : n={len(r4_fail_ids)} -> {sorted(r4_fail_ids)}")
print(f"R3 ∩ R4 overlap   : n={len(overlap_ids)} (genuinely computed)")
print(f"R3 ∪ R4 exclusions: n={len(union_fail_ids)}")
print(f"Final cohort      : {final_count:,}")
print(
    f"Identity: {EXPECTED['r1_r2_pass']} - {len(r3_fail_ids)} - "
    f"{len(r4_fail_ids)} + {len(overlap_ids)} = {final_count}  OK"
)

# ------------------------------------------------------------
# 8. Write outputs
# ------------------------------------------------------------
print("\n" + "=" * 78)
print("STEP 9 — WRITE OUTPUTS")
print("=" * 78)

OUT_DIR_TABLES.mkdir(parents=True, exist_ok=True)
OUT_DIR_META.mkdir(parents=True, exist_ok=True)

# Selection flow summary — sequential input/removed/retained
r1_removed = N_HOUSEHOLDS - int(r1_pass.sum())
r2_removed = int(r1_pass.sum()) - int(base_mask.sum())
r3_removed = len(r3_fail_ids)
r4_removed = len(r4_fail_ids)

selection_summary = pd.DataFrame(
    [
        {
            "stage": "raw", "criterion": "LCL Group N source population",
            "input_households": N_HOUSEHOLDS, "removed_at_stage": 0,
            "retained_households": N_HOUSEHOLDS,
        },
        {
            "stage": "R1", "criterion": "full-period coverage >= 90%",
            "input_households": N_HOUSEHOLDS, "removed_at_stage": r1_removed,
            "retained_households": int(r1_pass.sum()),
        },
        {
            "stage": "R2", "criterion": "training-period distinct values > 1",
            "input_households": int(r1_pass.sum()), "removed_at_stage": r2_removed,
            "retained_households": int(base_mask.sum()),
        },
        {
            "stage": "R3", "criterion": "min useful transitions across h={1,12,48} >= 100",
            "input_households": int(base_mask.sum()), "removed_at_stage": r3_removed,
            "retained_households": int(base_mask.sum()) - r3_removed,
        },
        {
            "stage": "R4", "criterion": "max training skip rate across h={1,12,48} <= 5%",
            "input_households": int(base_mask.sum()) - r3_removed,
            "removed_at_stage": r4_removed - len(overlap_ids),
            "retained_households": final_count,
        },
    ]
)
selection_summary.to_csv(OUT_SELECTION_SUMMARY, index=False)

# Exclusion detail
r1_fail_ids = set(static.loc[~static["R1_pass"], "household_id"])
r2_fail_ids_only = set(
    static.loc[static["R1_pass"] & ~static["R2_pass"], "household_id"]
)

exclusion_rows = []
for hid in sorted(r1_fail_ids):
    row = static.loc[static["household_id"] == hid].iloc[0]
    exclusion_rows.append({
        "household_id": hid, "excluded_by": "R1",
        "overall_coverage": row["overall_coverage"],
        "train_distinct_values": np.nan,
        "minimum_useful_transitions_formal_horizons": np.nan,
        "maximum_train_skip_rate_formal_horizons": np.nan,
    })
for hid in sorted(r2_fail_ids_only):
    row = static.loc[static["household_id"] == hid].iloc[0]
    exclusion_rows.append({
        "household_id": hid, "excluded_by": "R2",
        "overall_coverage": row["overall_coverage"],
        "train_distinct_values": row["train_distinct_values"],
        "minimum_useful_transitions_formal_horizons": np.nan,
        "maximum_train_skip_rate_formal_horizons": np.nan,
    })
for hid in sorted(union_fail_ids):
    row = static.loc[static["household_id"] == hid].iloc[0]
    rule = "R3_and_R4" if hid in overlap_ids else "R3" if hid in r3_fail_ids else "R4"
    exclusion_rows.append({
        "household_id": hid, "excluded_by": rule,
        "overall_coverage": row["overall_coverage"],
        "train_distinct_values": row["train_distinct_values"],
        "minimum_useful_transitions_formal_horizons": row[
            "minimum_useful_transitions_formal_horizons"
        ],
        "maximum_train_skip_rate_formal_horizons": row[
            "maximum_train_skip_rate_formal_horizons"
        ],
    })

exclusion_detail = pd.DataFrame(exclusion_rows)
exclusion_detail.to_csv(OUT_EXCLUSION_DETAIL, index=False)

# Final eligible households — canonical schema, sorted
final_columns = [
    "household_id",
    "minimum_useful_transitions_formal_horizons",
    "minimum_useful_transition_horizon",
    "maximum_train_skip_rate_formal_horizons",
    "maximum_train_skip_rate_horizon",
] + [f"train_useful_transitions_h{h}" for h in HORIZONS] + [
    f"train_skip_rate_h{h}" for h in HORIZONS
]
final_eligible = (
    static.loc[final_mask, final_columns]
    .sort_values("household_id")
    .reset_index(drop=True)
)
final_eligible.to_csv(OUT_FINAL_ELIGIBLE, index=False)

# Appendix B Figure B.1 — reported Rule 3 diagnostic at h = 1
#
# The dissertation figure visualises the h = 1 learnable-window counts for the
# 30 lowest-count R1/R2 households. Formal R3 eligibility is still evaluated
# independently at h = 1, 12 and 48 above; this display is the reported
# threshold-separation diagnostic and must not be substituted with the
# cross-horizon minimum.
r3_review = (
    static.loc[base_mask, ["household_id", "train_useful_transitions_h1"]]
    .sort_values(["train_useful_transitions_h1", "household_id"])
    .head(30)
    .reset_index(drop=True)
)
r3_review = r3_review.rename(columns={"train_useful_transitions_h1": "learnable_training_windows_h1"})
r3_review.to_csv(OUT_R3_LOWEST30, index=False)

expected_low = [2, 4, 27, 29, 41]
actual_low = r3_review.loc[:4, "learnable_training_windows_h1"].astype(int).tolist()
if actual_low != expected_low:
    raise AssertionError(
        "Appendix B.1 check failed: expected the five reported h=1 low counts "
        f"{expected_low}, found {actual_low}."
    )
if int(r3_review.loc[5, "learnable_training_windows_h1"]) != 661:
    raise AssertionError(
        "Appendix B.1 check failed: expected the next lowest h=1 household to have 661 windows."
    )

OUT_R3_FIGURE.parent.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(12.69, 10.59))
values = r3_review["learnable_training_windows_h1"].to_numpy(dtype=float)
labels = [str(v) for v in r3_review["household_id"]]
ypos = np.arange(len(values))
excluded = values < R3_THRESHOLD
colours = np.where(excluded, "#C63A2B", "#4C78A8")

ax.barh(ypos, values, color=colours, edgecolor="#2F3E4E", linewidth=0.8, height=0.72)
ax.invert_yaxis()
ax.set_xscale("log")
ax.set_xlim(1, 3.0e4)
ax.set_title("Learnable training windows across LCL source households", fontsize=18, pad=16)
ax.set_xlabel("Learnable training windows per household (logarithmic scale)", fontsize=14)
ax.grid(axis="x", color="#D9DEE5", linewidth=0.9, alpha=0.85)
ax.set_axisbelow(True)

# First five rows show household IDs; retained rows are displayed by rank exactly
# as in the submitted dissertation figure.
yticklabels = []
for i, hid in enumerate(labels):
    yticklabels.append(hid if i < 5 else f"rank {i+1}")
ax.set_yticks(ypos)
ax.set_yticklabels(yticklabels, fontsize=10)
for i, tick in enumerate(ax.get_yticklabels()):
    if i < 5:
        tick.set_color("#C63A2B")
        tick.set_fontweight("bold")

# Threshold line and label.
ax.axvline(R3_THRESHOLD, color="black", linestyle="--", linewidth=1.7, zorder=5)
ax.annotate(
    "Exclusion threshold\n100 windows",
    xy=(R3_THRESHOLD, 0.1), xytext=(170, -0.7),
    textcoords="data", ha="left", va="center", fontsize=12,
    arrowprops=dict(arrowstyle="-", color="black", linewidth=1.2),
)

# Explicit values for the five excluded households.
for i in range(5):
    ax.text(values[i] * 1.25, i, f"{int(values[i])}", va="center", ha="left",
            fontsize=11, color="#C63A2B")

ratio = int(round(values[5] / values[4]))
ax.annotate(
    f"{ratio}-fold separation\n{int(values[4])} → {int(values[5])} windows",
    xy=(180, 4.5), xytext=(2200, 2.2),
    ha="left", va="center", fontsize=13,
    bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="black"),
    arrowprops=dict(arrowstyle="->", color="black", linewidth=1.4,
                    connectionstyle="arc3,rad=-0.16"),
)

from matplotlib.patches import Patch
ax.legend(
    handles=[
        Patch(facecolor="#C63A2B", edgecolor="#2F3E4E", label="Excluded from training pool (n = 5)"),
        Patch(facecolor="#4C78A8", edgecolor="#2F3E4E", label="Retained in training pool"),
    ],
    loc="lower right", frameon=True, framealpha=0.96, fontsize=11,
)

for side in ["top", "right"]:
    ax.spines[side].set_visible(False)

note = (
    "A learnable training window is a supervised sample whose 48-step input sequence contains at least one change in value\n"
    "and whose target is non-zero. Cohort: 3,885 LCL source households (2013 coverage ≥ 90%, excluding two all-zero series).\n"
    "Training period: 2013-01-02 to 2013-09-13, giving 12,230 candidate windows per household. Only the 30 lowest-ranked households are shown."
)
fig.text(0.5, 0.012, note, ha="center", va="bottom", fontsize=9.5)
fig.subplots_adjust(left=0.12, right=0.985, top=0.94, bottom=0.15)
fig.savefig(OUT_R3_FIGURE, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

# Appendix B Table B.2 — Rule 4 threshold trade-off after R1-R3
r3_base = base_mask & static["R3_pass"].to_numpy()
all_useful = float(useful_matrix[r3_base].sum())
candidate_by_h = np.array([candidate_counts[h] for h in HORIZONS], dtype=float)
expected_b2 = {
    0.00: (1346, 34.93, 0.00),
    0.02: (3679, 94.97, 0.48),
    0.05: (3843, 99.13, 0.56),
    0.10: (3865, 99.66, 0.59),
}
b2_rows = []
for threshold, expected_vals in expected_b2.items():
    keep = r3_base & (max_skip <= threshold + 1e-12)
    retained = int(keep.sum())
    useful_pct = 100.0 * float(useful_matrix[keep].sum()) / all_useful
    # Candidate counts differ slightly by lead; aggregate row-level skip rate exactly.
    lost = float((skip_matrix[keep] * candidate_by_h).sum())
    candidates = float(keep.sum() * candidate_by_h.sum())
    overall_skip_pct = 100.0 * lost / candidates if candidates else np.nan
    b2_rows.append({
        "r4_threshold_pct": 100 * threshold,
        "households_retained": retained,
        "useful_transitions_retained_pct": useful_pct,
        "overall_skip_rate_pct": overall_skip_pct,
    })
    exp_n, exp_useful, exp_skip = expected_vals
    if retained != exp_n or round(useful_pct, 2) != exp_useful or round(overall_skip_pct, 2) != exp_skip:
        raise AssertionError(
            f"Appendix B.2 mismatch at threshold {threshold:.0%}: "
            f"observed n={retained}, useful={useful_pct:.2f}%, skip={overall_skip_pct:.2f}%; "
            f"expected n={exp_n}, useful={exp_useful:.2f}%, skip={exp_skip:.2f}%."
        )
pd.DataFrame(b2_rows).to_csv(OUT_R4_TRADEOFF, index=False)

# Metadata
meta_lines = [
    "---",
    "design_status: current",
    "valid_for_formal_results: yes",
    "analysis_status: reported",
    f"source_lock_path: {PATHS['design_lock']}",
    f"raw_load_path: {PATHS['lcl_raw']}",
    f"time_map_path: {PATHS['lcl_time_map']}",
    "dataset: LCL",
    "household_or_meter_subset: R1-R4 recomputed from raw 4,173 Group N households",
    f"P0_period: {P0_START} to {P0_END}",
    f"training_period: {SPLIT_BOUNDS['train'][0]} to {SPLIT_BOUNDS['train'][1]}",
    f"formal_horizons: {HORIZONS}",
    f"R1_threshold: coverage >= {COVERAGE_MIN}",
    f"R2_threshold: distinct training values > 1 (rounded to {ROUND_DP} dp)",
    f"R3_threshold: min useful transitions across formal horizons >= {R3_THRESHOLD}",
    f"R4_threshold: max training skip rate across formal horizons <= {R4_THRESHOLD}",
    "missing_handling: complete-case; skip affected sample only",
    "input_imputation: false",
    "target_imputation: false",
    "population_lock_lcl: consumption_n.csv (UK Data Service Group N)",
    "population_lock_cer: confirmed in Project setup, not re-checked here",
    "time_alignment: canonical GMT physical sequence via raw_row_index "
    "in lcl_full_local_clock_map.parquet",
    "created_for: Exploratory data analysis+ formal EDA, clustering and sample-construction input",
    "valid_for_methodology: yes",
    "---",
    "",
    "# Data preparation — R1-R4 Recomputation and Cohort Confirmation",
    "",
    f"- Raw LCL households: `{N_HOUSEHOLDS}`",
    f"- P0 positions: `{N_POSITIONS}`",
    f"- R1 pass: `{int(r1_pass.sum())}`",
    f"- R2 failures: `{sorted(r2_fail_ids_only)}`",
    f"- R1+R2 pass: `{int(base_mask.sum())}`",
    f"- R3 failures: `{sorted(r3_fail_ids)}`",
    f"- R4 failures (n={len(r4_fail_ids)}): `{sorted(r4_fail_ids)}`",
    f"- R3 ∩ R4 overlap (genuinely computed): `{len(overlap_ids)}`",
    f"- R3 ∪ R4 union exclusions: `{len(union_fail_ids)}`",
    f"- Final eligible cohort: `{final_count}`",
    f"- Identity: {EXPECTED['r1_r2_pass']} - {len(r3_fail_ids)} - "
    f"{len(r4_fail_ids)} + {len(overlap_ids)} = {final_count}",
    "",
    "## Output files",
    "",
    f"- `{OUT_SELECTION_SUMMARY.name}`",
    f"- `{OUT_EXCLUSION_DETAIL.name}`",
    f"- `{OUT_FINAL_ELIGIBLE.name}`",
    f"- `{OUT_R3_LOWEST30.name}`",
    f"- `{OUT_R4_TRADEOFF.name}`",
    f"- `{OUT_R3_FIGURE.name}`",
    "",
    "## Result",
    "",
    "- Data preparation gate: `PASS`",
    "- Cohort rebuilt directly from raw LCL data and canonical time alignment.",
    "",
]
OUT_META.write_text("\n".join(meta_lines), encoding="utf-8")

print(f"Selection summary : {OUT_SELECTION_SUMMARY}")
print(f"Exclusion detail   : {OUT_EXCLUSION_DETAIL}")
print(f"Final eligible     : {OUT_FINAL_ELIGIBLE}")
print(f"Rule 3 table       : {OUT_R3_LOWEST30}")
print(f"Rule 3 figure      : {OUT_R3_FIGURE}")
print(f"Rule 4 trade-off   : {OUT_R4_TRADEOFF}")
print(f"Metadata           : {OUT_META}")
print("\nDATA PREPARATION result: PASS")
