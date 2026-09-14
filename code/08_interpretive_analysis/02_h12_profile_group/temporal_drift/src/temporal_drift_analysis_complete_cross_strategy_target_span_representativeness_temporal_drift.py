#!/usr/bin/env python3
"""Temporal drift analysis: complete cross-strategy target-span representativeness and temporal drift.

Direct read-only extension of Temporal drift features. It reuses the formally validated Temporal drift features
meter-level temporal-drift features, attaches all four strategy errors from the
Profile group analysis native meter table, derives all six unique pairwise strategy gains, and
examines criterion-specific relationships at h={1,12,48}, with h=12 primary.

No model is trained, no prediction is changed, no prototype is refitted, no
meter is reassigned and no meter is excluded. Findings are descriptive and
non-causal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

ANALYSIS_ID = "temporal_drift_analysis"
ANALYSIS = "temporal_drift_analysis_complete_cross_strategy_target_span_representativeness_temporal_drift"
UPSTREAM_ANALYSIS = "temporal_drift_features_target_span_representativeness_temporal_drift"
GROUPS = [1, 2, 3, 4]
GROUP_COUNTS = {1: 141, 2: 325, 3: 324, 4: 139}
LEADS = [1, 12, 48]
CRITERIA = ["mae", "rmse", "smape"]
PRIMARY_LEAD = 12
PRIMARY_PREDICTORS = [
    "test_z_clock_rmse",
    "test_correlation_distance",
    "test_mean_abs_change_pct",
    "test_std_abs_change_pct",
    "test_low_share_abs_change_pp",
    "test_high_share_abs_change_pp",
    "test_peak_shift_slots",
    "validation_z_clock_rmse",
    "later_train_z_clock_rmse",
]
QUARTILE_PREDICTORS = [
    "test_z_clock_rmse",
    "test_mean_abs_change_pct",
    "test_low_share_abs_change_pp",
    "test_high_share_abs_change_pp",
]
STRATEGY_VALUE_COLUMNS = {
    "direct_transfer": "direct_transfer_value_mean",
    "fine_tuning": "fine_tuning_value_mean",
    "cer_scratch_limited": "cer_scratch_limited_value_mean",
    "cer_scratch_full": "cer_scratch_full_value_mean",
}
COMPARISONS = [
    ("source_transfer", "direct_transfer", "cer_scratch_limited", "primary_lock_controlled"),
    ("fine_tuning", "fine_tuning", "direct_transfer", "primary_lock_controlled"),
    ("full_data", "cer_scratch_full", "cer_scratch_limited", "primary_lock_controlled"),
    ("direct_vs_full", "direct_transfer", "cer_scratch_full", "primary_lock_controlled"),
    ("fine_tuning_vs_limited", "fine_tuning", "cer_scratch_limited", "supplementary_strategy_closure"),
    ("full_vs_fine_tuning", "cer_scratch_full", "fine_tuning", "supplementary_strategy_closure"),
]
PROFILE_GROUP_ANALYSIS_PRIMARY_GAIN_COLUMNS = {
    "source_transfer": "source_transfer_gain_mean",
    "fine_tuning": "fine_tuning_gain_mean",
    "full_data": "full_data_gain_mean",
    "direct_vs_full": "direct_vs_full_gain_mean",
}


@dataclass(frozen=True)
class Paths:
    root: Path
    profile_group_analysis_gains: Path
    upstream_tables: Path
    upstream_metadata: Path
    tables: Path
    metadata: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--bootstraps", type=int, default=500)
    return p.parse_args()


def make_paths(root: Path) -> Paths:
    root = root.expanduser().absolute()
    return Paths(
        root=root,
        profile_group_analysis_gains=root / "outputs/tables/profile_group_analysis_profile_group_analysis/profile_group_analysis_native_meter_gains_with_group.csv",
        upstream_tables=root / f"outputs/tables/{UPSTREAM_ANALYSIS}",
        upstream_metadata=root / f"outputs/metadata/{UPSTREAM_ANALYSIS}",
        tables=root / f"outputs/tables/{ANALYSIS}",
        metadata=root / f"outputs/metadata/{ANALYSIS}",
    )


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    ensure(not missing, f"{label} missing columns: {missing}")


def norm_id(value) -> str:
    s = str(value).strip()
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


def safe_relative(absolute_gain, comparator_error):
    num = np.asarray(absolute_gain, dtype=float)
    den = np.asarray(comparator_error, dtype=float)
    out = np.full(np.broadcast_shapes(num.shape, den.shape), np.nan, dtype=float)
    np.divide(100.0 * num, den, out=out, where=np.abs(den) > 1e-12)
    return out


def bh_adjust(values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(values), dtype=float)
    n = len(p)
    order = np.argsort(p)
    out = np.empty(n, dtype=float)
    previous = 1.0
    for rank, idx in reversed(list(enumerate(order, start=1))):
        previous = min(previous, p[idx] * n / rank)
        out[idx] = previous
    return np.clip(out, 0.0, 1.0)


def apply_grouped_bh(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out["q_value_bh"] = np.nan
    for _, idx in out.groupby(group_cols, sort=False, observed=True).groups.items():
        idx = list(idx)
        out.loc[idx, "q_value_bh"] = bh_adjust(out.loc[idx, "p_value"].to_numpy(float))
    return out


def rank_residual(values, groups) -> np.ndarray:
    y = pd.Series(values).rank(method="average").to_numpy(float)
    X = pd.get_dummies(pd.Series(groups).astype(int), drop_first=False).to_numpy(float)
    X = np.column_stack([np.ones(len(X)), X[:, 1:]])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return y - X @ beta


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_upstream_inventory(paths: Paths) -> int:
    status_path = paths.upstream_metadata / f"{UPSTREAM_ANALYSIS}_status.json"
    inventory_path = paths.upstream_metadata / "temporal_drift_features_output_inventory.csv"
    ensure(status_path.is_file(), f"Missing Temporal drift features status: {status_path}")
    ensure(inventory_path.is_file(), f"Missing Temporal drift features inventory: {inventory_path}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    ensure(status.get("analysis_id") == "temporal_drift_features", "Unexpected Temporal drift features analysis ID")
    ensure(status.get("status") == "COMPLETE_PASS", "Temporal drift features status is not COMPLETE_PASS")
    ensure(int(status.get("meters", -1)) == 929, "Temporal drift features status meter count mismatch")
    inv = pd.read_csv(inventory_path)
    require_columns(inv, ["relative_path", "bytes", "sha256"], "Temporal drift features inventory")
    for row in inv.itertuples(index=False):
        file_path = paths.root / str(row.relative_path)
        ensure(file_path.is_file(), f"Temporal drift features inventory file missing: {file_path}")
        ensure(file_path.stat().st_size == int(row.bytes), f"Temporal drift features inventory size mismatch: {file_path}")
        ensure(sha256_file(file_path) == str(row.sha256), f"Temporal drift features inventory SHA mismatch: {file_path}")
    return len(inv)


def load_inputs(paths: Paths):
    ensure(paths.profile_group_analysis_gains.is_file(), f"Missing Profile group analysis gains: {paths.profile_group_analysis_gains}")
    upstream_inventory_count = verify_upstream_inventory(paths)

    profile_group_analysis = pd.read_csv(paths.profile_group_analysis_gains)
    required = [
        "entity_id", "lead", "criterion", "support_scope", "assigned_group",
        "group_description", "seed_count",
        *STRATEGY_VALUE_COLUMNS.values(),
        *PROFILE_GROUP_ANALYSIS_PRIMARY_GAIN_COLUMNS.values(),
    ]
    require_columns(profile_group_analysis, required, "Profile group analysis gains")
    ensure(len(profile_group_analysis) == 8361, f"Expected 8,361 Profile group analysis rows, found {len(profile_group_analysis):,}")
    profile_group_analysis["meter_id"] = profile_group_analysis["entity_id"].map(norm_id)
    profile_group_analysis["lead"] = pd.to_numeric(profile_group_analysis["lead"], errors="raise").astype(int)
    profile_group_analysis["assigned_group"] = pd.to_numeric(profile_group_analysis["assigned_group"], errors="raise").astype(int)
    profile_group_analysis["criterion"] = profile_group_analysis["criterion"].astype(str).str.lower()
    ensure(set(profile_group_analysis["lead"].unique()) == set(LEADS), f"Unexpected leads: {sorted(profile_group_analysis.lead.unique())}")
    ensure(set(profile_group_analysis["criterion"].unique()) == set(CRITERIA), f"Unexpected criteria: {sorted(profile_group_analysis.criterion.unique())}")
    ensure(profile_group_analysis["support_scope"].eq("native").all(), "Native support required")
    ensure(profile_group_analysis["seed_count"].eq(4).all(), "Four formal seeds required")
    ensure(profile_group_analysis["meter_id"].nunique() == 929, "Expected 929 Profile group analysis meters")
    ensure(not profile_group_analysis.duplicated(["meter_id", "lead", "criterion"]).any(), "Duplicate Profile group analysis meter-lead-criterion rows")
    for lead in LEADS:
        for criterion in CRITERIA:
            n = len(profile_group_analysis[(profile_group_analysis.lead == lead) & (profile_group_analysis.criterion == criterion)])
            ensure(n == 929, f"Expected 929 rows for lead={lead}, criterion={criterion}; found {n}")
    counts = (
        profile_group_analysis[["meter_id", "assigned_group"]]
        .drop_duplicates()
        .assigned_group.value_counts()
        .sort_index()
        .to_dict()
    )
    ensure(counts == GROUP_COUNTS, f"Unexpected group counts: {counts}")

    drift_path = paths.upstream_tables / "temporal_drift_features_meter_temporal_drift_features.csv"
    upstream_outcome_path = paths.upstream_tables / "temporal_drift_features_h12_drift_outcome_dataset.csv"
    upstream_overall_path = paths.upstream_tables / "temporal_drift_features_overall_spearman_summary.csv"
    upstream_adjusted_path = paths.upstream_tables / "temporal_drift_features_group_adjusted_rank_summary.csv"
    upstream_within_path = paths.upstream_tables / "temporal_drift_features_within_group_spearman_summary.csv"
    upstream_primary_path = paths.upstream_tables / "temporal_drift_features_h12_primary_drift_evidence.csv"
    upstream_quartile_path = paths.upstream_tables / "temporal_drift_features_drift_quartile_summary.csv"
    upstream_group_path = paths.upstream_tables / "temporal_drift_features_group_drift_summary.csv"
    for path in [
        drift_path, upstream_outcome_path, upstream_overall_path,
        upstream_adjusted_path, upstream_within_path, upstream_primary_path,
        upstream_quartile_path, upstream_group_path,
    ]:
        ensure(path.is_file(), f"Missing Temporal drift features upstream table: {path}")

    drift = pd.read_csv(drift_path)
    drift["meter_id"] = drift["meter_id"].map(norm_id)
    require_columns(drift, ["meter_id", *PRIMARY_PREDICTORS], "Temporal drift features drift features")
    ensure(len(drift) == 929 and drift.meter_id.nunique() == 929, "Expected 929 unique Temporal drift features drift rows")
    ensure(set(drift.meter_id) == set(profile_group_analysis.meter_id), "Profile group analysis and Temporal drift features meter sets differ")

    upstream = {
        "outcome": pd.read_csv(upstream_outcome_path),
        "overall": pd.read_csv(upstream_overall_path),
        "adjusted": pd.read_csv(upstream_adjusted_path),
        "within": pd.read_csv(upstream_within_path),
        "primary": pd.read_csv(upstream_primary_path),
        "quartile": pd.read_csv(upstream_quartile_path),
        "group": pd.read_csv(upstream_group_path),
    }
    ensure(len(upstream["outcome"]) == 2787, "Temporal drift features h=12 outcome row count mismatch")
    ensure(len(upstream["overall"]) == 108, "Temporal drift features overall row count mismatch")
    ensure(len(upstream["adjusted"]) == 108, "Temporal drift features adjusted row count mismatch")
    ensure(len(upstream["within"]) == 432, "Temporal drift features within-group row count mismatch")
    ensure(len(upstream["primary"]) == 27, "Temporal drift features primary row count mismatch")
    ensure(len(upstream["quartile"]) == 48, "Temporal drift features quartile row count mismatch")
    ensure(len(upstream["group"]) == 36, "Temporal drift features group-drift row count mismatch")
    return profile_group_analysis, drift, upstream, upstream_inventory_count


def build_outcome(profile_group_analysis: pd.DataFrame, drift: pd.DataFrame) -> pd.DataFrame:
    base = profile_group_analysis.merge(drift, on="meter_id", how="inner", validate="many_to_one")
    ensure(len(base) == 8361, "Cross-strategy drift outcome row count mismatch")

    derived: dict[str, np.ndarray | pd.Series] = {}
    for strategy, source_column in STRATEGY_VALUE_COLUMNS.items():
        derived[f"strategy_error_{strategy}"] = (
            pd.to_numeric(base[source_column], errors="raise").astype(float).to_numpy()
        )

    for comparison_id, focal, comparator, _ in COMPARISONS:
        focal_error = np.asarray(derived[f"strategy_error_{focal}"], dtype=float)
        comparator_error = np.asarray(derived[f"strategy_error_{comparator}"], dtype=float)
        absolute = comparator_error - focal_error
        derived[f"gain_{comparison_id}_absolute"] = absolute
        derived[f"gain_{comparison_id}_relative_pct"] = safe_relative(absolute, comparator_error)
        derived[f"gain_{comparison_id}_positive"] = absolute > 0
        derived[f"gain_{comparison_id}_focal_error"] = focal_error
        derived[f"gain_{comparison_id}_comparator_error"] = comparator_error

    derived_frame = pd.DataFrame(derived, index=base.index)
    return pd.concat([base, derived_frame], axis=1)


def gain_definition_table() -> pd.DataFrame:
    rows = []
    for comparison_id, focal, comparator, role in COMPARISONS:
        rows.append({
            "comparison_id": comparison_id,
            "focal_strategy": focal,
            "comparator_strategy": comparator,
            "role": role,
            "absolute_gain_formula": f"error_{comparator} - error_{focal}",
            "positive_means": f"{focal} lower error than {comparator}",
            "relative_gain_denominator": comparator,
        })
    return pd.DataFrame(rows)


def outcome_specs() -> list[dict]:
    specs = []
    for strategy in STRATEGY_VALUE_COLUMNS:
        specs.append({
            "outcome_family": "strategy_error",
            "outcome_id": strategy,
            "outcome_scale": "absolute_error",
            "column": f"strategy_error_{strategy}",
            "strategy": strategy,
            "comparison_id": "",
        })
    for comparison_id, _, _, _ in COMPARISONS:
        specs.append({
            "outcome_family": "gain_absolute",
            "outcome_id": comparison_id,
            "outcome_scale": "absolute_gain",
            "column": f"gain_{comparison_id}_absolute",
            "strategy": "",
            "comparison_id": comparison_id,
        })
        specs.append({
            "outcome_family": "gain_relative",
            "outcome_id": comparison_id,
            "outcome_scale": "relative_gain_pct",
            "column": f"gain_{comparison_id}_relative_pct",
            "strategy": "",
            "comparison_id": comparison_id,
        })
    return specs


def relationship_tables(outcome: pd.DataFrame):
    overall_rows = []
    adjusted_rows = []
    within_rows = []
    specs = outcome_specs()
    for lead in LEADS:
        for criterion in CRITERIA:
            d = outcome[(outcome.lead == lead) & (outcome.criterion == criterion)].reset_index(drop=True)
            ensure(len(d) == 929, f"Expected 929 rows for lead={lead}, criterion={criterion}")
            for predictor in PRIMARY_PREDICTORS:
                x = d[predictor].to_numpy(float)
                for spec in specs:
                    y = d[spec["column"]].to_numpy(float)
                    valid = np.isfinite(x) & np.isfinite(y)
                    ensure(valid.sum() >= 900, f"Too few finite rows for {lead}/{criterion}/{predictor}/{spec['outcome_id']}")
                    rho, p_value = stats.spearmanr(x[valid], y[valid])
                    base = {
                        "lead": lead,
                        "criterion": criterion,
                        "predictor": predictor,
                        "outcome_family": spec["outcome_family"],
                        "outcome_id": spec["outcome_id"],
                        "outcome_scale": spec["outcome_scale"],
                        "strategy": spec["strategy"],
                        "comparison_id": spec["comparison_id"],
                        "n": int(valid.sum()),
                        "rho": float(rho),
                        "p_value": float(p_value),
                    }
                    overall_rows.append(base)
                    rx = rank_residual(x[valid], d.loc[valid, "assigned_group"])
                    ry = rank_residual(y[valid], d.loc[valid, "assigned_group"])
                    adjusted_rho, adjusted_p = stats.pearsonr(rx, ry)
                    adjusted_rows.append({**base, "rho": float(adjusted_rho), "p_value": float(adjusted_p)})
                    for group in GROUPS:
                        group_mask = valid & d.assigned_group.eq(group).to_numpy()
                        group_rho, group_p = stats.spearmanr(x[group_mask], y[group_mask])
                        within_rows.append({
                            **base,
                            "assigned_group": group,
                            "n": int(group_mask.sum()),
                            "rho": float(group_rho),
                            "p_value": float(group_p),
                        })
    overall = apply_grouped_bh(pd.DataFrame(overall_rows), ["lead", "criterion", "outcome_family"])
    adjusted = apply_grouped_bh(pd.DataFrame(adjusted_rows), ["lead", "criterion", "outcome_family"])
    within = apply_grouped_bh(pd.DataFrame(within_rows), ["lead", "criterion", "assigned_group", "outcome_family"])
    return overall, adjusted, within


def bootstrap_primary(outcome: pd.DataFrame, bootstraps: int) -> pd.DataFrame:
    """Temporal drift features-aligned bootstrap for h=12 relative gains.

    The same bootstrap resample is used across all six comparisons for one
    criterion-predictor cell. scipy.rankdata is applied column-wise so the
    resampled Spearman ranks are recomputed exactly while avoiding six separate
    scipy.spearmanr calls per resample.
    """
    rows = []
    rng = np.random.default_rng(20260728)
    comparison_ids = [item[0] for item in COMPARISONS]
    for criterion in CRITERIA:
        d = outcome[(outcome.lead == PRIMARY_LEAD) & (outcome.criterion == criterion)].reset_index(drop=True)
        Y = d[[f"gain_{comparison_id}_relative_pct" for comparison_id in comparison_ids]].to_numpy(float)
        for predictor in PRIMARY_PREDICTORS:
            x = d[predictor].to_numpy(float)
            valid = np.isfinite(x) & np.isfinite(Y).all(axis=1)
            xv = x[valid]
            Yv = Y[valid]
            values = np.empty((bootstraps, len(comparison_ids)), dtype=float)
            for b in range(bootstraps):
                idx = rng.integers(0, len(xv), len(xv))
                rx = stats.rankdata(xv[idx], method="average")
                ry = stats.rankdata(Yv[idx], method="average", axis=0)
                rx = rx - rx.mean()
                ry = ry - ry.mean(axis=0)
                denominator = np.sqrt(np.sum(rx * rx) * np.sum(ry * ry, axis=0))
                values[b] = np.divide(rx @ ry, denominator, out=np.full(len(comparison_ids), np.nan), where=denominator > 0)
            for j, comparison_id in enumerate(comparison_ids):
                rho, p_value = stats.spearmanr(xv, Yv[:, j])
                rows.append({
                    "lead": PRIMARY_LEAD,
                    "criterion": criterion,
                    "predictor": predictor,
                    "outcome_family": "gain_relative",
                    "comparison_id": comparison_id,
                    "outcome_scale": "relative_gain_pct",
                    "n": int(valid.sum()),
                    "rho": float(rho),
                    "p_value": float(p_value),
                    "bootstrap_ci_low": float(np.nanquantile(values[:, j], 0.025)),
                    "bootstrap_ci_high": float(np.nanquantile(values[:, j], 0.975)),
                    "bootstraps": int(bootstraps),
                })
    return apply_grouped_bh(pd.DataFrame(rows), ["criterion", "outcome_family"])


def quartile_summary(outcome: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lead in LEADS:
        for criterion in CRITERIA:
            d0 = outcome[(outcome.lead == lead) & (outcome.criterion == criterion)].copy()
            for predictor in QUARTILE_PREDICTORS:
                d = d0.copy()
                d["quartile"] = pd.qcut(d[predictor].rank(method="first"), 4, labels=[1, 2, 3, 4])
                for comparison_id, focal, comparator, _ in COMPARISONS:
                    for quartile, q in d.groupby("quartile", observed=True):
                        absolute = q[f"gain_{comparison_id}_absolute"]
                        relative = q[f"gain_{comparison_id}_relative_pct"]
                        rows.append({
                            "lead": lead,
                            "criterion": criterion,
                            "predictor": predictor,
                            "comparison_id": comparison_id,
                            "focal_strategy": focal,
                            "comparator_strategy": comparator,
                            "quartile": int(quartile),
                            "meters": len(q),
                            "predictor_mean": float(q[predictor].mean()),
                            "focal_error_mean": float(q[f"strategy_error_{focal}"].mean()),
                            "comparator_error_mean": float(q[f"strategy_error_{comparator}"].mean()),
                            "absolute_gain_mean": float(absolute.mean()),
                            "absolute_gain_median": float(absolute.median()),
                            "relative_gain_mean_pct": float(relative.mean()),
                            "relative_gain_median_pct": float(relative.median()),
                            "positive_gain_share": float((absolute > 0).mean()),
                        })
    return pd.DataFrame(rows)


def primary_component_evidence(
    primary_bootstrap: pd.DataFrame,
    overall: pd.DataFrame,
    adjusted: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for row in primary_bootstrap[primary_bootstrap.outcome_family.eq("gain_relative")].itertuples(index=False):
        comparison = next(item for item in COMPARISONS if item[0] == row.comparison_id)
        _, focal, comparator, role = comparison
        selector = (
            (overall.lead == PRIMARY_LEAD)
            & (overall.criterion == row.criterion)
            & (overall.predictor == row.predictor)
        )
        gain_row = overall[
            selector
            & overall.outcome_family.eq("gain_relative")
            & overall.outcome_id.eq(row.comparison_id)
        ].iloc[0]
        focal_row = overall[
            selector
            & overall.outcome_family.eq("strategy_error")
            & overall.outcome_id.eq(focal)
        ].iloc[0]
        comparator_row = overall[
            selector
            & overall.outcome_family.eq("strategy_error")
            & overall.outcome_id.eq(comparator)
        ].iloc[0]
        gain_adjusted = adjusted[
            selector
            & adjusted.outcome_family.eq("gain_relative")
            & adjusted.outcome_id.eq(row.comparison_id)
        ].iloc[0]
        focal_adjusted = adjusted[
            selector
            & adjusted.outcome_family.eq("strategy_error")
            & adjusted.outcome_id.eq(focal)
        ].iloc[0]
        comparator_adjusted = adjusted[
            selector
            & adjusted.outcome_family.eq("strategy_error")
            & adjusted.outcome_id.eq(comparator)
        ].iloc[0]
        rows.append({
            "lead": PRIMARY_LEAD,
            "criterion": row.criterion,
            "predictor": row.predictor,
            "comparison_id": row.comparison_id,
            "comparison_role": role,
            "focal_strategy": focal,
            "comparator_strategy": comparator,
            "relative_gain_rho": gain_row.rho,
            "relative_gain_q_value_bh": gain_row.q_value_bh,
            "relative_gain_bootstrap_ci_low": row.bootstrap_ci_low,
            "relative_gain_bootstrap_ci_high": row.bootstrap_ci_high,
            "group_adjusted_relative_gain_rho": gain_adjusted.rho,
            "group_adjusted_relative_gain_q_value_bh": gain_adjusted.q_value_bh,
            "focal_error_rho": focal_row.rho,
            "focal_error_q_value_bh": focal_row.q_value_bh,
            "comparator_error_rho": comparator_row.rho,
            "comparator_error_q_value_bh": comparator_row.q_value_bh,
            "group_adjusted_focal_error_rho": focal_adjusted.rho,
            "group_adjusted_focal_error_q_value_bh": focal_adjusted.q_value_bh,
            "group_adjusted_comparator_error_rho": comparator_adjusted.rho,
            "group_adjusted_comparator_error_q_value_bh": comparator_adjusted.q_value_bh,
        })
    return pd.DataFrame(rows)


def algebraic_audits(outcome: pd.DataFrame):
    profile_group_analysis_rows = []
    identity_rows = []
    for lead in LEADS:
        for criterion in CRITERIA:
            d = outcome[(outcome.lead == lead) & (outcome.criterion == criterion)]
            for comparison_id, focal, comparator, _ in COMPARISONS:
                expected = d[f"strategy_error_{comparator}"] - d[f"strategy_error_{focal}"]
                actual = d[f"gain_{comparison_id}_absolute"]
                max_diff = float(np.max(np.abs(expected - actual)))
                identity_rows.append({
                    "lead": lead,
                    "criterion": criterion,
                    "comparison_id": comparison_id,
                    "identity": "comparator_error_minus_focal_error",
                    "max_abs_difference": max_diff,
                    "pass_reproduction_tolerance": max_diff <= 1e-4,
                })
                if comparison_id in PROFILE_GROUP_ANALYSIS_PRIMARY_GAIN_COLUMNS:
                    source_col = PROFILE_GROUP_ANALYSIS_PRIMARY_GAIN_COLUMNS[comparison_id]
                    source_diff = float(np.max(np.abs(d[source_col] - actual)))
                    profile_group_analysis_rows.append({
                        "lead": lead,
                        "criterion": criterion,
                        "comparison_id": comparison_id,
                        "profile_group_analysis_column": source_col,
                        "max_abs_difference": source_diff,
                        "pass_reproduction_tolerance": source_diff <= 1e-4,
                    })
            closure_checks = {
                "fine_tuning_vs_limited_equals_source_plus_fine":
                    d["gain_fine_tuning_vs_limited_absolute"]
                    - (d["gain_source_transfer_absolute"] + d["gain_fine_tuning_absolute"]),
                "direct_vs_full_equals_source_minus_full":
                    d["gain_direct_vs_full_absolute"]
                    - (d["gain_source_transfer_absolute"] - d["gain_full_data_absolute"]),
                "full_vs_fine_equals_full_minus_fine_vs_limited":
                    d["gain_full_vs_fine_tuning_absolute"]
                    - (d["gain_full_data_absolute"] - d["gain_fine_tuning_vs_limited_absolute"]),
            }
            for name, residual in closure_checks.items():
                max_diff = float(np.max(np.abs(residual)))
                identity_rows.append({
                    "lead": lead,
                    "criterion": criterion,
                    "comparison_id": name,
                    "identity": "pairwise_gain_algebraic_closure",
                    "max_abs_difference": max_diff,
                    "pass_reproduction_tolerance": max_diff <= 1e-4,
                })
    return pd.DataFrame(profile_group_analysis_rows), pd.DataFrame(identity_rows)


def upstream_original_relationships(outcome_h12: pd.DataFrame):
    outcomes = [
        "source_transfer_gain_mean",
        "source_transfer_gain_relative_pct",
        "direct_transfer_value_mean",
        "cer_scratch_limited_value_mean",
    ]
    overall_rows = []
    adjusted_rows = []
    within_rows = []
    for criterion in CRITERIA:
        d = outcome_h12[outcome_h12.criterion.eq(criterion)].copy()
        for predictor in PRIMARY_PREDICTORS:
            for y_col in outcomes:
                x = d[predictor].to_numpy(float)
                y = d[y_col].to_numpy(float)
                valid = np.isfinite(x) & np.isfinite(y)
                rho, p_value = stats.spearmanr(x[valid], y[valid])
                overall_rows.append({
                    "criterion": criterion,
                    "predictor": predictor,
                    "outcome": y_col,
                    "n": int(valid.sum()),
                    "rho": float(rho),
                    "p_value": float(p_value),
                })
                rx = rank_residual(x[valid], d.loc[valid, "assigned_group"])
                ry = rank_residual(y[valid], d.loc[valid, "assigned_group"])
                adjusted_rho, adjusted_p = stats.pearsonr(rx, ry)
                adjusted_rows.append({
                    "criterion": criterion,
                    "predictor": predictor,
                    "outcome": y_col,
                    "n": int(valid.sum()),
                    "rho": float(adjusted_rho),
                    "p_value": float(adjusted_p),
                })
                for group in GROUPS:
                    group_mask = valid & d.assigned_group.eq(group).to_numpy()
                    within_rho, within_p = stats.spearmanr(x[group_mask], y[group_mask])
                    within_rows.append({
                        "criterion": criterion,
                        "predictor": predictor,
                        "outcome": y_col,
                        "assigned_group": group,
                        "n": int(group_mask.sum()),
                        "rho": float(within_rho),
                        "p_value": float(within_p),
                    })
    for rows in (overall_rows, adjusted_rows, within_rows):
        q = bh_adjust([row["p_value"] for row in rows])
        for row, q_value in zip(rows, q):
            row["q_value_bh"] = float(q_value)
    return pd.DataFrame(overall_rows), pd.DataFrame(adjusted_rows), pd.DataFrame(within_rows)


def upstream_original_quartiles(outcome_h12: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for criterion in CRITERIA:
        base = outcome_h12[outcome_h12.criterion.eq(criterion)].copy()
        for predictor in QUARTILE_PREDICTORS:
            d = base.copy()
            d["quartile"] = pd.qcut(d[predictor].rank(method="first"), 4, labels=[1, 2, 3, 4])
            for quartile, q in d.groupby("quartile", observed=True):
                y = q.source_transfer_gain_relative_pct
                rows.append({
                    "criterion": criterion,
                    "predictor": predictor,
                    "quartile": int(quartile),
                    "meters": len(q),
                    "predictor_mean": q[predictor].mean(),
                    "gain_mean_pct": y.mean(),
                    "gain_median_pct": y.median(),
                    "positive_share": (y > 0).mean(),
                })
    return pd.DataFrame(rows)


def upstream_reproduction(outcome: pd.DataFrame, upstream: dict) -> pd.DataFrame:
    h12 = outcome[outcome.lead.eq(12)].copy()
    h12["source_transfer_gain_relative_pct"] = h12["gain_source_transfer_relative_pct"]
    audit_rows = []

    keys = ["meter_id", "criterion"]
    old_outcome = upstream["outcome"].copy()
    old_outcome["meter_id"] = old_outcome["meter_id"].map(norm_id)
    old_outcome["criterion"] = old_outcome["criterion"].astype(str).str.lower()
    compare_columns = [
        "source_transfer_gain_mean",
        "source_transfer_gain_relative_pct",
        "direct_transfer_value_mean",
        "cer_scratch_limited_value_mean",
        *PRIMARY_PREDICTORS,
    ]
    merged = h12[keys + compare_columns].merge(
        old_outcome[keys + compare_columns],
        on=keys,
        suffixes=("", "_old"),
        validate="one_to_one",
    )
    outcome_max = 0.0
    for col in compare_columns:
        outcome_max = max(outcome_max, float(np.nanmax(np.abs(merged[f"{col}"] - merged[f"{col}_old"]))))
    audit_rows.append({
        "upstream_output": "temporal_drift_features_h12_drift_outcome_dataset.csv",
        "rows_expected": 2787,
        "rows_reproduced": len(merged),
        "max_abs_difference": outcome_max,
        "pass_reproduction_tolerance": len(merged) == 2787 and outcome_max <= 1e-4,
    })

    new_overall, new_adjusted, new_within = upstream_original_relationships(h12)
    new_quartile = upstream_original_quartiles(h12)
    table_pairs = [
        ("temporal_drift_features_overall_spearman_summary.csv", new_overall, upstream["overall"], ["criterion", "predictor", "outcome"], ["n", "rho", "p_value", "q_value_bh"]),
        ("temporal_drift_features_group_adjusted_rank_summary.csv", new_adjusted, upstream["adjusted"], ["criterion", "predictor", "outcome"], ["n", "rho", "p_value", "q_value_bh"]),
        ("temporal_drift_features_within_group_spearman_summary.csv", new_within, upstream["within"], ["criterion", "predictor", "outcome", "assigned_group"], ["n", "rho", "p_value", "q_value_bh"]),
        ("temporal_drift_features_drift_quartile_summary.csv", new_quartile, upstream["quartile"], ["criterion", "predictor", "quartile"], ["meters", "predictor_mean", "gain_mean_pct", "gain_median_pct", "positive_share"]),
    ]
    for name, new_df, old_df, merge_keys, numeric_cols in table_pairs:
        merged_table = new_df.merge(old_df, on=merge_keys, suffixes=("", "_old"), validate="one_to_one")
        max_diff = 0.0
        for col in numeric_cols:
            max_diff = max(max_diff, float(np.nanmax(np.abs(merged_table[f"{col}"] - merged_table[f"{col}_old"]))))
        audit_rows.append({
            "upstream_output": name,
            "rows_expected": len(old_df),
            "rows_reproduced": len(merged_table),
            "max_abs_difference": max_diff,
            "pass_reproduction_tolerance": len(merged_table) == len(old_df) and max_diff <= 1e-4,
        })

    ensure(all(row["pass_reproduction_tolerance"] for row in audit_rows), "Temporal drift features upstream reproduction failed")
    return pd.DataFrame(audit_rows)


def group_drift_summary(drift: pd.DataFrame, outcome: pd.DataFrame) -> pd.DataFrame:
    membership = outcome[["meter_id", "assigned_group", "group_description"]].drop_duplicates()
    merged = drift.merge(membership, on="meter_id", validate="one_to_one")
    rows = []
    for predictor in PRIMARY_PREDICTORS:
        for group, q in merged.groupby("assigned_group", observed=True):
            rows.append({
                "predictor": predictor,
                "assigned_group": int(group),
                "group_description": q.group_description.iloc[0],
                "meters": len(q),
                "mean": q[predictor].mean(),
                "median": q[predictor].median(),
                "q25": q[predictor].quantile(0.25),
                "q75": q[predictor].quantile(0.75),
                "sample_std": q[predictor].std(ddof=1),
            })
    return pd.DataFrame(rows)


def write_outputs(
    paths: Paths,
    drift: pd.DataFrame,
    outcome: pd.DataFrame,
    definitions: pd.DataFrame,
    overall: pd.DataFrame,
    adjusted: pd.DataFrame,
    within: pd.DataFrame,
    primary_bootstrap: pd.DataFrame,
    primary_components: pd.DataFrame,
    quartiles: pd.DataFrame,
    group_drift: pd.DataFrame,
    profile_group_analysis_audit: pd.DataFrame,
    algebra_audit: pd.DataFrame,
    upstream_audit: pd.DataFrame,
    upstream_inventory_count: int,
    elapsed: float,
) -> None:
    paths.tables.mkdir(parents=True, exist_ok=True)
    paths.metadata.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []

    def write_csv(df: pd.DataFrame, name: str) -> Path:
        path = paths.tables / name
        df.to_csv(path, index=False)
        files.append(path)
        return path

    write_csv(drift, "temporal_drift_analysis_meter_temporal_drift_features_reused.csv")
    write_csv(outcome, "temporal_drift_analysis_cross_strategy_drift_outcome_dataset.csv")
    write_csv(definitions, "temporal_drift_analysis_pairwise_gain_definitions.csv")
    write_csv(group_drift, "temporal_drift_analysis_group_drift_summary_reproduced.csv")
    write_csv(overall, "temporal_drift_analysis_overall_spearman_summary.csv")
    write_csv(adjusted, "temporal_drift_analysis_group_adjusted_rank_summary.csv")
    write_csv(within, "temporal_drift_analysis_within_group_spearman_summary.csv")
    write_csv(primary_bootstrap, "temporal_drift_analysis_h12_primary_bootstrap_relationships.csv")
    write_csv(primary_components, "temporal_drift_analysis_h12_primary_component_relationship_evidence.csv")
    write_csv(quartiles, "temporal_drift_analysis_drift_quartile_summary.csv")
    write_csv(profile_group_analysis_audit, "temporal_drift_analysis_profile_group_analysis_gain_reproduction_audit.csv")
    write_csv(algebra_audit, "temporal_drift_analysis_pairwise_gain_algebraic_identity_audit.csv")
    write_csv(upstream_audit, "temporal_drift_analysis_temporal_drift_features_upstream_reproduction_audit.csv")

    status = {
        "analysis_id": ANALYSIS_ID,
        "status": "COMPLETE_PASS",
        "meters": 929,
        "leads": LEADS,
        "criteria": CRITERIA,
        "strategies": 4,
        "pairwise_comparisons": 6,
        "outcome_rows": len(outcome),
        "overall_relationships": len(overall),
        "group_adjusted_relationships": len(adjusted),
        "within_group_relationships": len(within),
        "primary_bootstrap_relationships": len(primary_bootstrap),
        "primary_component_rows": len(primary_components),
        "quartile_rows": len(quartiles),
        "profile_group_analysis_audit_rows": len(profile_group_analysis_audit),
        "algebraic_audit_rows": len(algebra_audit),
        "temporal_drift_features_upstream_audit_rows": len(upstream_audit),
        "temporal_drift_features_inventory_files_verified": upstream_inventory_count,
        "elapsed_seconds": elapsed,
        "training_performed": False,
        "predictions_modified": False,
        "prototypes_refitted": False,
        "meters_reassigned": False,
        "meters_excluded": False,
        "interpretation": "descriptive and non-causal",
    }
    status_path = paths.metadata / f"{ANALYSIS}_status.json"
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    files.append(status_path)

    inventory_rows = []
    for file_path in files:
        inventory_rows.append({
            "relative_path": str(file_path.relative_to(paths.root)),
            "bytes": file_path.stat().st_size,
            "sha256": sha256_file(file_path),
        })
    pd.DataFrame(inventory_rows).to_csv(paths.metadata / "temporal_drift_analysis_output_inventory.csv", index=False)


def main() -> None:
    args = parse_args()
    paths = make_paths(args.root)
    profile_group_analysis, drift, upstream, upstream_inventory_count = load_inputs(paths)
    print("=" * 112)
    print("TEMPORAL DRIFT ANALYSIS COMPLETE CROSS-STRATEGY TARGET-SPAN REPRESENTATIVENESS AND TEMPORAL DRIFT")
    print("=" * 112)
    print(f"Meters                     : {profile_group_analysis.meter_id.nunique()}")
    print(f"Profile group analysis rows              : {len(profile_group_analysis):,}")
    print(f"Temporal drift features drift rows        : {len(drift):,}")
    print(f"Temporal drift features inventory files  : {upstream_inventory_count}")
    print("Leads                      : 1, 12, 48")
    print("Strategies                 : 4")
    print("Pairwise comparisons       : 6")
    print("Mode                       : read-only; direct Temporal drift features extension; descriptive and non-causal")
    if args.check_only:
        print("INPUT CHECK: PASS")
        return

    start = time.time()
    outcome = build_outcome(profile_group_analysis, drift)
    definitions = gain_definition_table()
    profile_group_analysis_audit, algebra_audit = algebraic_audits(outcome)
    ensure(profile_group_analysis_audit.pass_reproduction_tolerance.all(), "Profile group analysis gain reproduction failed")
    ensure(algebra_audit.pass_reproduction_tolerance.all(), "Pairwise gain algebraic identity failed")
    upstream_audit = upstream_reproduction(outcome, upstream)
    overall, adjusted, within = relationship_tables(outcome)
    primary_bootstrap = bootstrap_primary(outcome, args.bootstraps)
    quartiles = quartile_summary(outcome)
    primary_components = primary_component_evidence(primary_bootstrap, overall, adjusted)
    group_drift = group_drift_summary(drift, outcome)
    elapsed = time.time() - start

    ensure(len(outcome) == 8361, "Expected 8,361 outcome rows")
    ensure(len(overall) == 1296, "Expected 1,296 overall relationships")
    ensure(len(adjusted) == 1296, "Expected 1,296 group-adjusted relationships")
    ensure(len(within) == 5184, "Expected 5,184 within-group relationships")
    ensure(len(primary_bootstrap) == 162, "Expected 162 h=12 primary bootstrap relationships")
    ensure(len(primary_components) == 162, "Expected 162 h=12 primary component rows")
    ensure(len(quartiles) == 864, "Expected 864 quartile rows")
    ensure(len(profile_group_analysis_audit) == 36, "Expected 36 Profile group analysis audit rows")
    ensure(len(algebra_audit) == 81, "Expected 81 algebraic audit rows")
    ensure(len(upstream_audit) == 5, "Expected 5 Temporal drift features upstream audit rows")

    write_outputs(
        paths, drift, outcome, definitions, overall, adjusted, within,
        primary_bootstrap, primary_components, quartiles, group_drift,
        profile_group_analysis_audit, algebra_audit, upstream_audit,
        upstream_inventory_count, elapsed,
    )
    print("=" * 112)
    print("TEMPORAL DRIFT ANALYSIS COMPLETE — PASS")
    print("=" * 112)
    print("Meters retained                    : 929")
    print("Cross-strategy outcome rows        : 8,361")
    print("Overall relationship rows          : 1,296")
    print("Group-adjusted relationship rows   : 1,296")
    print("Within-group relationship rows     : 5,184")
    print("h=12 primary bootstrap rows        : 162")
    print("h=12 primary component rows        : 162")
    print("Drift-quartile summary rows        : 864")
    print("Profile group analysis reproduction              : PASS")
    print("Temporal drift features upstream reproduction  : PASS")
    print(f"Elapsed seconds                    : {elapsed:.1f}")
    print("Training performed                 : False")
    print("Predictions modified               : False")
    print("Final status                       : COMPLETE_PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"TEMPORAL DRIFT ANALYSIS FAILED: {exc}", file=__import__("sys").stderr)
        raise
