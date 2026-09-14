#!/usr/bin/env python3
"""Group heterogeneity: group heterogeneity and practical-significance audit.

Read-only post-hoc analysis using completed Profile group analysis meter-level gains and
Raw load structure meter-level raw-load structure. No forecasting model is trained,
no prediction is changed, no source prototype is refitted, and no receiving
meter is reassigned or excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ANALYSIS_ID = "group_heterogeneity"
ANALYSIS_NAME = "group_heterogeneity_group_heterogeneity_practical_significance"
RNG_SEED = 20260728
GROUP_ORDER = [1, 2, 3, 4]
GROUP_LABELS = {
    1: "G1 Morning–evening double peak",
    2: "G2 Early evening peak",
    3: "G3 Late evening peak",
    4: "G4 Late night peak",
}
GAIN_LABELS = {
    "source_transfer_gain": "Source Transfer Gain",
    "fine_tuning_gain": "Fine-Tuning Gain",
    "full_data_gain": "Full-Data Gain",
    "direct_vs_full_gain": "Direct-versus-Full Gain",
}
CRITERION_UNITS = {"mae": "kWh", "rmse": "kWh", "smape": "percentage points"}


@dataclass(frozen=True)
class Paths:
    root: Path
    gains: Path
    metrics: Path
    raw: Path
    tables: Path
    figures: Path
    metadata: Path
    decision: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--permutations", type=int, default=1999)
    parser.add_argument("--bootstraps", type=int, default=1000)
    parser.add_argument("--cv-repeats", type=int, default=50)
    parser.add_argument("--cv-folds", type=int, default=10)
    return parser.parse_args()


def resolve_paths(root: Path) -> Paths:
    root = root.expanduser().absolute()
    return Paths(
        root=root,
        gains=root / "outputs/tables/profile_group_analysis_profile_group_analysis/profile_group_analysis_native_meter_gains_with_group.csv",
        metrics=root / "outputs/tables/profile_group_analysis_profile_group_analysis/profile_group_analysis_native_meter_metrics_with_group.csv",
        raw=root / "outputs/tables/raw_load_structure_h12_profile_group_peak_decomposition/raw_load_structure_h12_meter_raw_load_structure.csv",
        tables=root / f"outputs/tables/{ANALYSIS_NAME}",
        figures=root / f"outputs/figures/{ANALYSIS_NAME}",
        metadata=root / f"outputs/metadata/{ANALYSIS_NAME}",
        decision=root / f"documentation/technical_decisions/{ANALYSIS_NAME}_decision.md",
    )


def require_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def load_and_validate(paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for p in (paths.gains, paths.metrics, paths.raw):
        if not p.is_file():
            raise FileNotFoundError(f"Required input not found: {p}")

    gains = pd.read_csv(paths.gains)
    metrics = pd.read_csv(paths.metrics)
    raw = pd.read_csv(paths.raw)

    require_columns(
        gains,
        [
            "entity_id", "lead", "criterion", "support_scope", "assigned_group",
            "group_description", "seed_count",
            "source_transfer_gain_mean", "source_transfer_gain_sample_std",
            "source_transfer_gain_minimum", "source_transfer_gain_maximum",
            "fine_tuning_gain_mean", "fine_tuning_gain_sample_std",
            "fine_tuning_gain_minimum", "fine_tuning_gain_maximum",
            "full_data_gain_mean", "full_data_gain_sample_std",
            "full_data_gain_minimum", "full_data_gain_maximum",
            "direct_vs_full_gain_mean", "direct_vs_full_gain_sample_std",
            "direct_vs_full_gain_minimum", "direct_vs_full_gain_maximum",
            "direct_transfer_value_mean", "cer_scratch_limited_value_mean",
            "cer_scratch_full_value_mean",
        ],
        "Profile group analysis gains",
    )
    require_columns(
        metrics,
        ["strategy", "lead", "entity_id", "actual_mean_kwh", "actual_std_kwh", "actual_zero_rate", "assigned_group"],
        "Profile group analysis metrics",
    )
    require_columns(
        raw,
        [
            "meter_id", "assigned_group", "group_description", "test_actual_mean_kwh",
            "test_actual_std_kwh_ddof0", "actual_eq_0_row_share",
            "actual_gt_0_le_0_1_row_share", "actual_gt_0_1_le_0_5_row_share",
            "actual_gt_0_5_le_1_0_row_share", "actual_gt_1_0_row_share",
            "top_10pct_row_share", "top_5pct_row_share", "top_1pct_row_share",
        ],
        "Raw load structure raw-load structure",
    )

    if gains.shape[0] != 8361:
        raise ValueError(f"Expected 8,361 Profile group analysis gain rows, found {len(gains):,}")
    if raw.shape[0] != 929:
        raise ValueError(f"Expected 929 Raw load structure raw rows, found {len(raw):,}")
    if gains["entity_id"].nunique() != 929:
        raise ValueError("Profile group analysis gain input does not contain 929 unique meters")
    if raw["meter_id"].nunique() != 929:
        raise ValueError("Raw load structure raw input does not contain 929 unique meters")
    if sorted(gains["lead"].unique().tolist()) != [1, 12, 48]:
        raise ValueError("Expected leads [1, 12, 48]")
    if sorted(gains["criterion"].unique().tolist()) != ["mae", "rmse", "smape"]:
        raise ValueError("Expected criteria [mae, rmse, smape]")
    if not gains["support_scope"].eq("native").all():
        raise ValueError("Group heterogeneity requires native-support meter gains only")
    if not gains["seed_count"].eq(4).all():
        raise ValueError("Expected four formal seeds for every Profile group analysis meter outcome")

    group_counts = gains[["entity_id", "assigned_group"]].drop_duplicates()["assigned_group"].value_counts().sort_index().to_dict()
    expected_counts = {1: 141, 2: 325, 3: 324, 4: 139}
    if group_counts != expected_counts:
        raise ValueError(f"Unexpected group counts: {group_counts}; expected {expected_counts}")

    raw_counts = raw["assigned_group"].value_counts().sort_index().to_dict()
    if raw_counts != expected_counts:
        raise ValueError(f"Raw load structure group counts differ: {raw_counts}; expected {expected_counts}")

    gain_ids = set(gains["entity_id"].astype(str).unique())
    raw_ids = set(raw["meter_id"].astype(str).unique())
    if gain_ids != raw_ids:
        raise ValueError(f"Profile-group input meter sets differ: gains-only={len(gain_ids-raw_ids)}, raw-only={len(raw_ids-gain_ids)}")

    # Cross-input audit: profile-group and raw-load data must describe the same h=12 support.
    h12_actual = metrics[(metrics["lead"] == 12) & (metrics["strategy"] == "direct_transfer")][
        ["entity_id", "actual_mean_kwh", "actual_std_kwh"]
    ].copy()
    if len(h12_actual) != 929 or h12_actual["entity_id"].nunique() != 929:
        raise ValueError("Profile group analysis h=12 Direct Transfer actual-support audit does not contain 929 meters")
    audit = h12_actual.merge(
        raw[["meter_id", "test_actual_mean_kwh", "test_actual_std_kwh_ddof0"]],
        left_on="entity_id", right_on="meter_id", how="inner", validate="one_to_one",
    )
    if len(audit) != 929:
        raise ValueError("Profile-group h=12 actual-support audit did not match all 929 households")
    mean_diff = float(np.max(np.abs(audit["actual_mean_kwh"] - audit["test_actual_mean_kwh"])))
    std_diff = float(np.max(np.abs(audit["actual_std_kwh"] - audit["test_actual_std_kwh_ddof0"])))
    if mean_diff > 1e-6 or std_diff > 1e-6:
        raise ValueError(f"Profile-group h=12 actual-support mismatch: mean={mean_diff:.3e}, std={std_diff:.3e}")

    return gains, metrics, raw


def safe_relative(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce").astype(float)
    den = pd.to_numeric(denominator, errors="coerce").astype(float)
    out = np.full(len(num), np.nan, dtype=float)
    valid = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > 1e-12)
    out[valid] = 100.0 * num.to_numpy()[valid] / den.to_numpy()[valid]
    return pd.Series(out, index=numerator.index)


def build_transferability_dataset(gains: pd.DataFrame) -> pd.DataFrame:
    df = gains.copy()
    for gain in GAIN_LABELS:
        mean_col = f"{gain}_mean"
        std_col = f"{gain}_sample_std"
        min_col = f"{gain}_minimum"
        max_col = f"{gain}_maximum"
        df[f"{gain}_all_seeds_positive"] = pd.to_numeric(df[min_col], errors="coerce") > 0
        df[f"{gain}_all_seeds_negative"] = pd.to_numeric(df[max_col], errors="coerce") < 0
        df[f"{gain}_mean_positive"] = pd.to_numeric(df[mean_col], errors="coerce") > 0
        df[f"{gain}_seed_cv_abs"] = pd.to_numeric(df[std_col], errors="coerce") / np.maximum(
            np.abs(pd.to_numeric(df[mean_col], errors="coerce")), 1e-12
        )

    df["source_transfer_gain_relative_pct"] = safe_relative(
        df["source_transfer_gain_mean"], df["cer_scratch_limited_value_mean"]
    )
    df["fine_tuning_gain_relative_pct"] = safe_relative(
        df["fine_tuning_gain_mean"], df["direct_transfer_value_mean"]
    )
    df["full_data_gain_relative_pct"] = safe_relative(
        df["full_data_gain_mean"], df["cer_scratch_limited_value_mean"]
    )
    df["direct_vs_full_gain_relative_pct"] = safe_relative(
        df["direct_vs_full_gain_mean"], df["cer_scratch_full_value_mean"]
    )
    return df


def one_way_effect(y: np.ndarray, g: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y)
    y = np.asarray(y[mask], dtype=float)
    g = np.asarray(g[mask])
    levels = np.array(sorted(pd.unique(g)))
    n = len(y)
    k = len(levels)
    grand = float(np.mean(y))
    ss_between = 0.0
    ss_within = 0.0
    for level in levels:
        vals = y[g == level]
        if len(vals) == 0:
            continue
        ss_between += len(vals) * (float(np.mean(vals)) - grand) ** 2
        ss_within += float(np.sum((vals - np.mean(vals)) ** 2))
    ss_total = ss_between + ss_within
    df_between = k - 1
    df_within = n - k
    ms_between = ss_between / df_between if df_between > 0 else np.nan
    ms_within = ss_within / df_within if df_within > 0 else np.nan
    f_stat = ms_between / ms_within if np.isfinite(ms_within) and ms_within > 0 else np.nan
    eta2 = ss_between / ss_total if ss_total > 0 else 0.0
    omega2 = (ss_between - df_between * ms_within) / (ss_total + ms_within) if ss_total + ms_within > 0 else 0.0
    return {
        "n": float(n), "groups": float(k), "grand_mean": grand,
        "ss_between": ss_between, "ss_within": ss_within, "ss_total": ss_total,
        "f_statistic": f_stat, "eta_squared": eta2, "omega_squared": omega2,
    }


def permutation_pvalue_one_way(y: np.ndarray, g: np.ndarray, permutations: int, rng: np.random.Generator) -> float:
    y = np.asarray(y, dtype=float)
    g = np.asarray(g)
    mask = np.isfinite(y)
    y = y[mask]
    g = g[mask]
    obs = one_way_effect(y, g)["f_statistic"]
    if not np.isfinite(obs):
        return np.nan
    count = 0
    for _ in range(permutations):
        perm = rng.permutation(y)
        stat = one_way_effect(perm, g)["f_statistic"]
        if np.isfinite(stat) and stat >= obs - 1e-15:
            count += 1
    return (count + 1.0) / (permutations + 1.0)


def bootstrap_one_way_ci(y: np.ndarray, g: np.ndarray, bootstraps: int, rng: np.random.Generator) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    g = np.asarray(g)
    mask = np.isfinite(y)
    y = y[mask]
    g = g[mask]
    levels = np.array(sorted(pd.unique(g)))
    eta = np.empty(bootstraps, dtype=float)
    omega = np.empty(bootstraps, dtype=float)
    for b in range(bootstraps):
        ys: list[np.ndarray] = []
        gs: list[np.ndarray] = []
        for level in levels:
            vals = y[g == level]
            draw = rng.choice(vals, size=len(vals), replace=True)
            ys.append(draw)
            gs.append(np.full(len(draw), level))
        effect = one_way_effect(np.concatenate(ys), np.concatenate(gs))
        eta[b] = effect["eta_squared"]
        omega[b] = effect["omega_squared"]
    return {
        "eta_squared_ci_low": float(np.quantile(eta, 0.025)),
        "eta_squared_ci_high": float(np.quantile(eta, 0.975)),
        "omega_squared_ci_low": float(np.quantile(omega, 0.025)),
        "omega_squared_ci_high": float(np.quantile(omega, 0.975)),
    }


def bh_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    q = np.full(len(p), np.nan, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return q
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    back = np.empty(m, dtype=float)
    back[order] = adjusted
    q[valid] = back
    return q


def eta_label(value: float) -> str:
    if not np.isfinite(value):
        return "not estimable"
    if value < 0.01:
        return "negligible by common heuristic"
    if value < 0.06:
        return "small by common heuristic"
    if value < 0.14:
        return "moderate by common heuristic"
    return "large by common heuristic"


def g_label(value: float) -> str:
    a = abs(value)
    if not np.isfinite(a):
        return "not estimable"
    if a < 0.20:
        return "negligible by common heuristic"
    if a < 0.50:
        return "small by common heuristic"
    if a < 0.80:
        return "moderate by common heuristic"
    return "large by common heuristic"


def group_descriptive(df: pd.DataFrame, value_col: str, gain: str, scale: str, lead: int, criterion: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    overall = pd.to_numeric(df[value_col], errors="coerce")
    overall_mean = float(overall.mean())
    overall_sd = float(overall.std(ddof=1))
    for group in GROUP_ORDER:
        sub = df[df["assigned_group"] == group]
        vals = pd.to_numeric(sub[value_col], errors="coerce").dropna()
        gain_mean_col = f"{gain}_mean"
        gain_std_col = f"{gain}_sample_std"
        rows.append({
            "gain": gain,
            "gain_label": GAIN_LABELS[gain],
            "lead": lead,
            "criterion": criterion,
            "scale": scale,
            "unit": "percent" if scale == "relative_pct" else CRITERION_UNITS[criterion],
            "assigned_group": group,
            "group_label": GROUP_LABELS[group],
            "meters": int(len(vals)),
            "overall_mean": overall_mean,
            "overall_sample_std": overall_sd,
            "mean": float(vals.mean()),
            "sample_std": float(vals.std(ddof=1)),
            "median": float(vals.median()),
            "q25": float(vals.quantile(0.25)),
            "q75": float(vals.quantile(0.75)),
            "minimum": float(vals.min()),
            "maximum": float(vals.max()),
            "group_minus_overall": float(vals.mean() - overall_mean),
            "group_mean_standardised_vs_overall_sd": float((vals.mean() - overall_mean) / overall_sd) if overall_sd > 0 else np.nan,
            "mean_positive_share": float((pd.to_numeric(sub[gain_mean_col], errors="coerce") > 0).mean()),
            "all_seeds_positive_share": float((pd.to_numeric(sub[f"{gain}_minimum"], errors="coerce") > 0).mean()),
            "mean_seed_sample_std": float(pd.to_numeric(sub[gain_std_col], errors="coerce").mean()),
        })
    return rows


def omnibus_row(df: pd.DataFrame, value_col: str, gain: str, scale: str, lead: int, criterion: str,
                 permutations: int, bootstraps: int, rng: np.random.Generator) -> dict[str, object]:
    y = pd.to_numeric(df[value_col], errors="coerce").to_numpy(float)
    g = df["assigned_group"].to_numpy()
    effect = one_way_effect(y, g)
    p = permutation_pvalue_one_way(y, g, permutations, rng)
    ci = bootstrap_one_way_ci(y, g, bootstraps, rng)
    return {
        "gain": gain,
        "gain_label": GAIN_LABELS[gain],
        "lead": lead,
        "criterion": criterion,
        "scale": scale,
        "unit": "percent" if scale == "relative_pct" else CRITERION_UNITS[criterion],
        **effect,
        **ci,
        "permutation_count": permutations,
        "permutation_p_value": p,
        "eta_effect_label": eta_label(effect["eta_squared"]),
        "omega_effect_label": eta_label(max(effect["omega_squared"], 0.0)),
    }


def pairwise_mean_contrasts(df: pd.DataFrame, value_col: str, gain: str, scale: str, criterion: str,
                            permutations: int, bootstraps: int, rng: np.random.Generator) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ga, gb in combinations(GROUP_ORDER, 2):
        a = pd.to_numeric(df.loc[df["assigned_group"] == ga, value_col], errors="coerce").dropna().to_numpy(float)
        b = pd.to_numeric(df.loc[df["assigned_group"] == gb, value_col], errors="coerce").dropna().to_numpy(float)
        diff = float(np.mean(b) - np.mean(a))
        pooled_num = (len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)
        pooled_den = len(a) + len(b) - 2
        pooled_sd = math.sqrt(pooled_num / pooled_den) if pooled_den > 0 and pooled_num > 0 else np.nan
        d = diff / pooled_sd if np.isfinite(pooled_sd) and pooled_sd > 0 else np.nan
        correction = 1.0 - 3.0 / (4.0 * (len(a) + len(b)) - 9.0)
        hedges_g = correction * d if np.isfinite(d) else np.nan

        boot_diff = np.empty(bootstraps, dtype=float)
        boot_g = np.empty(bootstraps, dtype=float)
        for i in range(bootstraps):
            aa = rng.choice(a, size=len(a), replace=True)
            bb = rng.choice(b, size=len(b), replace=True)
            bd = float(np.mean(bb) - np.mean(aa))
            boot_diff[i] = bd
            pnum = (len(aa) - 1) * np.var(aa, ddof=1) + (len(bb) - 1) * np.var(bb, ddof=1)
            psd = math.sqrt(pnum / pooled_den) if pooled_den > 0 and pnum > 0 else np.nan
            boot_g[i] = correction * bd / psd if np.isfinite(psd) and psd > 0 else np.nan

        pooled = np.concatenate([a, b])
        count = 0
        for _ in range(permutations):
            perm = rng.permutation(pooled)
            pa = perm[: len(a)]
            pb = perm[len(a):]
            pdiff = float(np.mean(pb) - np.mean(pa))
            if abs(pdiff) >= abs(diff) - 1e-15:
                count += 1
        p = (count + 1.0) / (permutations + 1.0)
        rows.append({
            "gain": gain,
            "gain_label": GAIN_LABELS[gain],
            "lead": 12,
            "criterion": criterion,
            "scale": scale,
            "unit": "percent" if scale == "relative_pct" else CRITERION_UNITS[criterion],
            "group_a": ga,
            "group_b": gb,
            "contrast": f"G{gb} - G{ga}",
            "n_a": len(a),
            "n_b": len(b),
            "mean_a": float(np.mean(a)),
            "mean_b": float(np.mean(b)),
            "mean_difference_b_minus_a": diff,
            "mean_difference_ci_low": float(np.quantile(boot_diff, 0.025)),
            "mean_difference_ci_high": float(np.quantile(boot_diff, 0.975)),
            "hedges_g": hedges_g,
            "hedges_g_ci_low": float(np.nanquantile(boot_g, 0.025)),
            "hedges_g_ci_high": float(np.nanquantile(boot_g, 0.975)),
            "hedges_g_effect_label": g_label(hedges_g),
            "permutation_count": permutations,
            "permutation_p_value": p,
        })
    return rows


def positive_share_contrasts(df: pd.DataFrame, gain: str, criterion: str, bootstraps: int,
                             rng: np.random.Generator) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    col = f"{gain}_mean"
    for ga, gb in combinations(GROUP_ORDER, 2):
        a = (pd.to_numeric(df.loc[df["assigned_group"] == ga, col], errors="coerce") > 0).astype(int).to_numpy()
        b = (pd.to_numeric(df.loc[df["assigned_group"] == gb, col], errors="coerce") > 0).astype(int).to_numpy()
        pa = float(np.mean(a))
        pb = float(np.mean(b))
        diff = pb - pa
        boot = np.empty(bootstraps, dtype=float)
        for i in range(bootstraps):
            boot[i] = float(np.mean(rng.choice(b, size=len(b), replace=True)) - np.mean(rng.choice(a, size=len(a), replace=True)))
        table = np.array([[int(a.sum()), int(len(a)-a.sum())], [int(b.sum()), int(len(b)-b.sum())]])
        _, p = stats.fisher_exact(table, alternative="two-sided")
        rows.append({
            "gain": gain,
            "gain_label": GAIN_LABELS[gain],
            "lead": 12,
            "criterion": criterion,
            "group_a": ga,
            "group_b": gb,
            "contrast": f"G{gb} - G{ga}",
            "n_a": len(a),
            "n_b": len(b),
            "positive_share_a": pa,
            "positive_share_b": pb,
            "risk_difference_b_minus_a": diff,
            "risk_difference_ci_low": float(np.quantile(boot, 0.025)),
            "risk_difference_ci_high": float(np.quantile(boot, 0.975)),
            "fisher_exact_p_value": float(p),
        })
    return rows


def stratified_folds(groups: np.ndarray, folds: int, rng: np.random.Generator) -> list[np.ndarray]:
    buckets: list[list[int]] = [[] for _ in range(folds)]
    for level in sorted(pd.unique(groups)):
        idx = np.flatnonzero(groups == level)
        idx = rng.permutation(idx)
        for j, value in enumerate(idx):
            buckets[j % folds].append(int(value))
    return [np.array(sorted(bucket), dtype=int) for bucket in buckets]


def binary_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    pos = y == 1
    n1 = int(pos.sum())
    n0 = int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = stats.rankdata(score, method="average")
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def cv_group_utility(y: np.ndarray, groups: np.ndarray, repeats: int, folds: int,
                     rng: np.random.Generator) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    mask = np.isfinite(y)
    y = y[mask]
    groups = groups[mask]
    r2_values: list[float] = []
    auc_values: list[float] = []
    brier_skill_values: list[float] = []
    binary = (y > 0).astype(int)

    for _ in range(repeats):
        fold_idx = stratified_folds(groups, folds, rng)
        pred = np.full(len(y), np.nan)
        base = np.full(len(y), np.nan)
        prob = np.full(len(y), np.nan)
        prob_base = np.full(len(y), np.nan)
        for test_idx in fold_idx:
            train_mask = np.ones(len(y), dtype=bool)
            train_mask[test_idx] = False
            y_train = y[train_mask]
            g_train = groups[train_mask]
            b_train = binary[train_mask]
            overall_mean = float(np.mean(y_train))
            overall_prob = float(np.mean(b_train))
            group_means = {level: float(np.mean(y_train[g_train == level])) for level in pd.unique(groups)}
            group_probs = {level: float(np.mean(b_train[g_train == level])) for level in pd.unique(groups)}
            for idx in test_idx:
                pred[idx] = group_means[groups[idx]]
                base[idx] = overall_mean
                prob[idx] = group_probs[groups[idx]]
                prob_base[idx] = overall_prob
        sse_group = float(np.sum((y - pred) ** 2))
        sse_base = float(np.sum((y - base) ** 2))
        r2_values.append(1.0 - sse_group / sse_base if sse_base > 0 else np.nan)
        auc_values.append(binary_auc(binary, prob))
        brier_group = float(np.mean((binary - prob) ** 2))
        brier_base = float(np.mean((binary - prob_base) ** 2))
        brier_skill_values.append(1.0 - brier_group / brier_base if brier_base > 0 else np.nan)

    def summary(values: Sequence[float], prefix: str) -> dict[str, float]:
        arr = np.asarray(values, dtype=float)
        return {
            f"{prefix}_mean": float(np.nanmean(arr)),
            f"{prefix}_median": float(np.nanmedian(arr)),
            f"{prefix}_ci_low": float(np.nanquantile(arr, 0.025)),
            f"{prefix}_ci_high": float(np.nanquantile(arr, 0.975)),
        }

    return {
        **summary(r2_values, "cv_r_squared"),
        **summary(auc_values, "positive_gain_auc"),
        **summary(brier_skill_values, "positive_gain_brier_skill"),
    }


def raw_effects(raw: pd.DataFrame, variables: list[str], permutations: int, bootstraps: int,
                rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    effect_rows: list[dict[str, object]] = []
    for var in variables:
        vals_all = pd.to_numeric(raw[var], errors="coerce")
        overall_mean = float(vals_all.mean())
        overall_sd = float(vals_all.std(ddof=1))
        summary_rows.append({
            "variable": var, "population": "overall_cer", "assigned_group": 0,
            "meters": int(vals_all.notna().sum()), "mean": overall_mean,
            "sample_std": overall_sd, "median": float(vals_all.median()),
            "q25": float(vals_all.quantile(0.25)), "q75": float(vals_all.quantile(0.75)),
            "group_minus_overall": 0.0, "group_mean_standardised_vs_overall_sd": 0.0,
        })
        for group in GROUP_ORDER:
            vals = pd.to_numeric(raw.loc[raw["assigned_group"] == group, var], errors="coerce").dropna()
            summary_rows.append({
                "variable": var, "population": GROUP_LABELS[group], "assigned_group": group,
                "meters": len(vals), "mean": float(vals.mean()), "sample_std": float(vals.std(ddof=1)),
                "median": float(vals.median()), "q25": float(vals.quantile(0.25)), "q75": float(vals.quantile(0.75)),
                "group_minus_overall": float(vals.mean() - overall_mean),
                "group_mean_standardised_vs_overall_sd": float((vals.mean()-overall_mean)/overall_sd) if overall_sd > 0 else np.nan,
            })
        y = vals_all.to_numpy(float)
        g = raw["assigned_group"].to_numpy()
        effect = one_way_effect(y, g)
        p = permutation_pvalue_one_way(y, g, permutations, rng)
        ci = bootstrap_one_way_ci(y, g, bootstraps, rng)
        effect_rows.append({
            "variable": var,
            **effect,
            **ci,
            "permutation_count": permutations,
            "permutation_p_value": p,
            "eta_effect_label": eta_label(effect["eta_squared"]),
            "omega_effect_label": eta_label(max(effect["omega_squared"], 0.0)),
        })
    effects = pd.DataFrame(effect_rows)
    effects["bh_q_value"] = bh_adjust(effects["permutation_p_value"])
    return pd.DataFrame(summary_rows), effects


def markdown_table(df: pd.DataFrame, columns: Sequence[str], formats: dict[str, str] | None = None) -> str:
    formats = formats or {}
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, divider]
    for row in df.loc[:, columns].itertuples(index=False, name=None):
        values: list[str] = []
        for col, value in zip(columns, row):
            if pd.isna(value):
                values.append("")
            elif col in formats:
                values.append(format(value, formats[col]))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_decision_note(path: Path, primary: pd.DataFrame, omnibus: pd.DataFrame,
                        pairwise: pd.DataFrame, utility: pd.DataFrame,
                        raw_effect: pd.DataFrame, lead_shift: pd.DataFrame) -> None:
    h12_mae_abs = primary[(primary["lead"] == 12) & (primary["criterion"] == "mae") & (primary["scale"] == "absolute")].copy()
    h12_mae_abs = h12_mae_abs.sort_values("assigned_group")
    om = omnibus[(omnibus["gain"] == "source_transfer_gain") & (omnibus["lead"] == 12) &
                 (omnibus["criterion"] == "mae") & (omnibus["scale"] == "absolute")].iloc[0]
    pr = pairwise[(pairwise["criterion"] == "mae") & (pairwise["scale"] == "absolute") &
                  (pairwise["group_a"] == 1) & (pairwise["group_b"] == 4)].iloc[0]
    util = utility[(utility["criterion"] == "mae") & (utility["scale"] == "relative_pct")].iloc[0]

    text = f"""# Group heterogeneity group heterogeneity and practical-significance audit

## Governance

- Analysis ID: `{ANALYSIS_ID}`
- Analysis is read-only and post-hoc.
- All 929 receiving meters are retained in their fixed source-defined K=4 groups.
- No forecasting model is retrained, no prediction is modified, no source prototype is refitted, and no meter is reassigned.
- Findings are descriptive and associational. Group membership is not treated as a causal intervention or a transfer-decision rule.
- No smallest effect size of interest was pre-specified; therefore this analysis does not claim equivalence from a non-significant contrast.

## Primary h=12 Source Transfer MAE evidence

{markdown_table(h12_mae_abs, ["assigned_group", "meters", "mean", "sample_std", "median", "mean_positive_share", "all_seeds_positive_share"], {"mean": ".6f", "sample_std": ".6f", "median": ".6f", "mean_positive_share": ".3f", "all_seeds_positive_share": ".3f"})}

The one-way group effect for absolute MAE Source Transfer Gain was eta-squared `{om['eta_squared']:.4f}` (bootstrap 95% CI `{om['eta_squared_ci_low']:.4f}` to `{om['eta_squared_ci_high']:.4f}`) and omega-squared `{om['omega_squared']:.4f}`. The permutation p-value was `{om['permutation_p_value']:.6g}`. This quantifies the proportion of meter-level outcome variation associated with the four fixed group means; it does not indicate that the groups explain the remaining within-group heterogeneity.

The G4 minus G1 mean contrast was `{pr['mean_difference_b_minus_a']:.6f}` kWh (bootstrap 95% CI `{pr['mean_difference_ci_low']:.6f}` to `{pr['mean_difference_ci_high']:.6f}`), with Hedges' g `{pr['hedges_g']:.3f}`. Adjacent-group contrasts must be read separately because an ordered set of group means does not imply four sharply separated distributions.

## Individual-level practical utility

For meter-level relative MAE Source Transfer Gain, a group-only repeated stratified cross-validation produced mean R-squared `{util['cv_r_squared_mean']:.4f}` (95% repeat interval `{util['cv_r_squared_ci_low']:.4f}` to `{util['cv_r_squared_ci_high']:.4f}`). The positive-gain AUC was `{util['positive_gain_auc_mean']:.3f}` and the Brier skill relative to an overall-prevalence predictor was `{util['positive_gain_brier_skill_mean']:.4f}`. These quantities assess whether group membership alone is useful for individual-meter prediction, rather than merely whether group means differ.

## Raw-load group characterisation

Raw-load effect sizes compare the same 929 meters over the h=12 test support and use equal-meter aggregation. Because clustering used independently standardised daily profiles, small raw-load eta-squared values are compatible with groups that primarily reflect shape and timing rather than magnitude.

{markdown_table(raw_effect, ["variable", "eta_squared", "omega_squared", "eta_squared_ci_low", "eta_squared_ci_high", "bh_q_value"], {"eta_squared": ".4f", "omega_squared": ".4f", "eta_squared_ci_low": ".4f", "eta_squared_ci_high": ".4f", "bh_q_value": ".4g"})}

## Lead sensitivity

Lead sensitivity is compact and supporting. It tests whether group differences in within-meter gain shifts differ across the fixed groups; it does not reopen the primary h=12 interpretation or select a lead after observing the results.

{markdown_table(lead_shift, ["criterion", "scale", "lead_shift", "eta_squared", "omega_squared", "permutation_p_value", "bh_q_value"], {"eta_squared": ".4f", "omega_squared": ".4f", "permutation_p_value": ".4g", "bh_q_value": ".4g"})}

## Claim boundary

This audit can establish whether fixed groups are associated with statistically structured and practically sized differences in observed transfer benefit. It cannot establish that a profile type causes the gain difference. Causal mechanism claims would require a new intervention, such as controlled profile-matched versus profile-mismatched source pretraining.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_figures(transfer: pd.DataFrame, omnibus: pd.DataFrame, raw_summary: pd.DataFrame, figures: Path) -> list[Path]:
    figures.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    # Figure 1: primary relative MAE gain distributions.
    sub = transfer[(transfer["lead"] == 12) & (transfer["criterion"] == "mae")].copy()
    data = [sub.loc[sub["assigned_group"] == g, "source_transfer_gain_relative_pct"].dropna().to_numpy() for g in GROUP_ORDER]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.boxplot(data, tick_labels=[f"G{g}" for g in GROUP_ORDER], showfliers=False)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_title("h=12 meter-level relative MAE Source Transfer Gain by fixed profile group")
    ax.set_xlabel("Fixed source-defined profile group")
    ax.set_ylabel("Relative MAE Transfer Gain (%)")
    fig.tight_layout()
    p = figures / "group_heterogeneity_h12_relative_mae_source_transfer_gain_by_group.png"
    fig.savefig(p, dpi=220)
    plt.close(fig)
    outputs.append(p)

    # Figure 2: eta-squared across leads and criteria, absolute Source Transfer Gain.
    eff = omnibus[(omnibus["gain"] == "source_transfer_gain") & (omnibus["scale"] == "absolute")].copy()
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    x = np.arange(len(CRITERION_UNITS))
    width = 0.24
    for j, lead in enumerate([1, 12, 48]):
        vals = [float(eff[(eff["lead"] == lead) & (eff["criterion"] == c)]["eta_squared"].iloc[0]) for c in ["mae", "rmse", "smape"]]
        ax.bar(x + (j - 1) * width, vals, width=width, label=f"h={lead}")
    ax.set_xticks(x)
    ax.set_xticklabels(["MAE", "RMSE", "SMAPE"])
    ax.set_ylabel("Eta-squared: variation associated with profile group")
    ax.set_title("Profile-group effect size for Source Transfer Gain across leads")
    ax.legend()
    fig.tight_layout()
    p = figures / "group_heterogeneity_source_transfer_group_eta_squared_across_leads.png"
    fig.savefig(p, dpi=220)
    plt.close(fig)
    outputs.append(p)

    # Figure 3: positive share by group and lead for MAE.
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for lead in [1, 12, 48]:
        s = transfer[(transfer["lead"] == lead) & (transfer["criterion"] == "mae")]
        shares = [float((s.loc[s["assigned_group"] == g, "source_transfer_gain_mean"] > 0).mean()) for g in GROUP_ORDER]
        ax.plot(GROUP_ORDER, shares, marker="o", label=f"h={lead}")
    ax.set_xticks(GROUP_ORDER)
    ax.set_xticklabels([f"G{g}" for g in GROUP_ORDER])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of meters with positive MAE Source Transfer Gain")
    ax.set_title("Consistency of Source Transfer benefit by group and lead")
    ax.legend()
    fig.tight_layout()
    p = figures / "group_heterogeneity_source_transfer_positive_share_by_group_and_lead.png"
    fig.savefig(p, dpi=220)
    plt.close(fig)
    outputs.append(p)

    # Figure 4: selected raw characteristics, group mean relative to overall CER mean.
    selected = ["test_actual_mean_kwh", "test_actual_std_kwh_ddof0", "actual_gt_1_0_row_share", "top_1pct_row_share"]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    x = np.arange(len(selected))
    width = 0.18
    for j, group in enumerate(GROUP_ORDER):
        vals = []
        for var in selected:
            row = raw_summary[(raw_summary["variable"] == var) & (raw_summary["assigned_group"] == group)].iloc[0]
            vals.append(float(row["group_mean_standardised_vs_overall_sd"]))
        ax.bar(x + (j - 1.5) * width, vals, width=width, label=f"G{group}")
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(["Mean load", "Load SD", ">1 kWh share", "Top 1% share"])
    ax.set_ylabel("Group mean deviation from overall CER mean (overall SD units)")
    ax.set_title("Raw-load group characterisation on the h=12 test support")
    ax.legend()
    fig.tight_layout()
    p = figures / "group_heterogeneity_raw_load_group_deviation_from_overall_cer.png"
    fig.savefig(p, dpi=220)
    plt.close(fig)
    outputs.append(p)
    return outputs


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_inventory(root: Path, files: list[Path], path: Path) -> pd.DataFrame:
    rows = []
    for file in sorted(files):
        rows.append({
            "relative_path": str(file.relative_to(root)),
            "bytes": file.stat().st_size,
            "sha256": sha256(file),
        })
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args.root)
    start = time.time()
    gains, metrics, raw = load_and_validate(paths)

    print("=" * 100)
    print("GROUP HETEROGENEITY GROUP HETEROGENEITY AND PRACTICAL-SIGNIFICANCE AUDIT")
    print("=" * 100)
    print(f"Project root       : {paths.root}")
    print(f"Meters            : {gains['entity_id'].nunique()}")
    print(f"Profile group analysis gain rows: {len(gains):,}")
    print(f"Raw load structure raw rows : {len(raw):,}")
    print(f"Permutations      : {args.permutations:,}")
    print(f"Bootstraps        : {args.bootstraps:,}")
    print(f"CV repeats/folds  : {args.cv_repeats}/{args.cv_folds}")
    print("Mode              : read-only; descriptive and non-causal")

    if args.check_only:
        print("\nINPUT CHECK: PASS")
        print(f"  {paths.gains}")
        print(f"  {paths.metrics}")
        print(f"  {paths.raw}")
        return

    for directory in (paths.tables, paths.metadata, paths.decision.parent):
        directory.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RNG_SEED)
    transfer = build_transferability_dataset(gains)

    source_summary_rows: list[dict[str, object]] = []
    source_omnibus_rows: list[dict[str, object]] = []
    for lead in [1, 12, 48]:
        for criterion in ["mae", "rmse", "smape"]:
            subset = transfer[(transfer["lead"] == lead) & (transfer["criterion"] == criterion)].copy()
            for scale, col in [
                ("absolute", "source_transfer_gain_mean"),
                ("relative_pct", "source_transfer_gain_relative_pct"),
            ]:
                source_summary_rows.extend(group_descriptive(subset, col, "source_transfer_gain", scale, lead, criterion))
                source_omnibus_rows.append(omnibus_row(
                    subset, col, "source_transfer_gain", scale, lead, criterion,
                    args.permutations, args.bootstraps, rng,
                ))
    source_summary = pd.DataFrame(source_summary_rows)
    source_omnibus = pd.DataFrame(source_omnibus_rows)
    source_omnibus["bh_q_value"] = bh_adjust(source_omnibus["permutation_p_value"])

    # h=12 pairwise evidence for Source Transfer Gain.
    pair_rows: list[dict[str, object]] = []
    pos_rows: list[dict[str, object]] = []
    utility_rows: list[dict[str, object]] = []
    for criterion in ["mae", "rmse", "smape"]:
        subset = transfer[(transfer["lead"] == 12) & (transfer["criterion"] == criterion)].copy()
        for scale, col in [("absolute", "source_transfer_gain_mean"), ("relative_pct", "source_transfer_gain_relative_pct")]:
            pair_rows.extend(pairwise_mean_contrasts(
                subset, col, "source_transfer_gain", scale, criterion,
                args.permutations, args.bootstraps, rng,
            ))
            utility = cv_group_utility(
                pd.to_numeric(subset[col], errors="coerce").to_numpy(float),
                subset["assigned_group"].to_numpy(),
                args.cv_repeats, args.cv_folds, rng,
            )
            utility_rows.append({
                "gain": "source_transfer_gain", "lead": 12, "criterion": criterion,
                "scale": scale, "unit": "percent" if scale == "relative_pct" else CRITERION_UNITS[criterion],
                "meters": len(subset), "cv_repeats": args.cv_repeats, "cv_folds": args.cv_folds,
                **utility,
            })
        pos_rows.extend(positive_share_contrasts(subset, "source_transfer_gain", criterion, args.bootstraps, rng))
    pairwise = pd.DataFrame(pair_rows)
    pairwise["bh_q_value"] = bh_adjust(pairwise["permutation_p_value"])
    positive = pd.DataFrame(pos_rows)
    positive["bh_q_value"] = bh_adjust(positive["fisher_exact_p_value"])
    utility_df = pd.DataFrame(utility_rows)

    # Other h=12 strategy contrasts, absolute and relative.
    other_rows: list[dict[str, object]] = []
    for gain in ["fine_tuning_gain", "full_data_gain", "direct_vs_full_gain"]:
        for criterion in ["mae", "rmse", "smape"]:
            subset = transfer[(transfer["lead"] == 12) & (transfer["criterion"] == criterion)].copy()
            for scale, col in [("absolute", f"{gain}_mean"), ("relative_pct", f"{gain}_relative_pct")]:
                other_rows.append(omnibus_row(
                    subset, col, gain, scale, 12, criterion,
                    args.permutations, args.bootstraps, rng,
                ))
    other_omnibus = pd.DataFrame(other_rows)
    other_omnibus["bh_q_value"] = bh_adjust(other_omnibus["permutation_p_value"])

    # Lead-shift group effects for Source Transfer Gain.
    shift_rows: list[dict[str, object]] = []
    for criterion in ["mae", "rmse", "smape"]:
        s = transfer[transfer["criterion"] == criterion].copy()
        for scale, col in [("absolute", "source_transfer_gain_mean"), ("relative_pct", "source_transfer_gain_relative_pct")]:
            wide = s.pivot(index=["entity_id", "assigned_group"], columns="lead", values=col).reset_index()
            for a, b in [(1, 12), (12, 48), (1, 48)]:
                y = pd.to_numeric(wide[b], errors="coerce") - pd.to_numeric(wide[a], errors="coerce")
                effect = one_way_effect(y.to_numpy(float), wide["assigned_group"].to_numpy())
                p = permutation_pvalue_one_way(y.to_numpy(float), wide["assigned_group"].to_numpy(), args.permutations, rng)
                ci = bootstrap_one_way_ci(y.to_numpy(float), wide["assigned_group"].to_numpy(), args.bootstraps, rng)
                shift_rows.append({
                    "gain": "source_transfer_gain", "criterion": criterion, "scale": scale,
                    "unit": "percent" if scale == "relative_pct" else CRITERION_UNITS[criterion],
                    "lead_from": a, "lead_to": b, "lead_shift": f"h={b} minus h={a}",
                    **effect, **ci, "permutation_count": args.permutations, "permutation_p_value": p,
                    "eta_effect_label": eta_label(effect["eta_squared"]),
                })
    lead_shift = pd.DataFrame(shift_rows)
    lead_shift["bh_q_value"] = bh_adjust(lead_shift["permutation_p_value"])

    # Formal raw-load group characterisation on the matched h=12 test support.
    raw_variables = [
        "test_actual_mean_kwh",
        "test_actual_std_kwh_ddof0",
        "actual_eq_0_row_share",
        "actual_gt_0_le_0_1_row_share",
        "actual_gt_0_1_le_0_5_row_share",
        "actual_gt_0_5_le_1_0_row_share",
        "actual_gt_1_0_row_share",
        "top_10pct_row_share",
        "top_5pct_row_share",
        "top_1pct_row_share",
    ]
    raw_summary, raw_effect = raw_effects(raw, raw_variables, args.permutations, args.bootstraps, rng)

    # Primary h=12 evidence table.
    primary = source_summary[(source_summary["lead"] == 12) & (source_summary["criterion"] == "mae")].copy()

    outputs: list[Path] = []
    table_map = {
        "group_heterogeneity_meter_transferability_dataset.csv": transfer,
        "group_heterogeneity_source_transfer_group_descriptive_summary.csv": source_summary,
        "group_heterogeneity_source_transfer_omnibus_effects.csv": source_omnibus,
        "group_heterogeneity_h12_source_transfer_pairwise_contrasts.csv": pairwise,
        "group_heterogeneity_h12_source_transfer_positive_share_contrasts.csv": positive,
        "group_heterogeneity_h12_group_predictive_utility.csv": utility_df,
        "group_heterogeneity_h12_other_gain_omnibus_effects.csv": other_omnibus,
        "group_heterogeneity_source_transfer_lead_shift_group_effects.csv": lead_shift,
        "group_heterogeneity_raw_group_characterisation_summary.csv": raw_summary,
        "group_heterogeneity_raw_group_effects.csv": raw_effect,
        "group_heterogeneity_h12_primary_evidence_table.csv": primary,
    }
    for name, frame in table_map.items():
        p = paths.tables / name
        frame.to_csv(p, index=False)
        outputs.append(p)

    write_decision_note(paths.decision, primary, source_omnibus, pairwise, utility_df, raw_effect, lead_shift)
    outputs.append(paths.decision)

    elapsed = time.time() - start
    status = {
        "analysis_id": ANALYSIS_ID,
        "analysis": ANALYSIS_NAME,
        "status": "COMPLETE_PASS",
        "meters": 929,
        "profile_group_analysis_gain_rows": int(len(gains)),
        "raw_load_structure_raw_rows": int(len(raw)),
        "source_transfer_omnibus_rows": int(len(source_omnibus)),
        "h12_pairwise_rows": int(len(pairwise)),
        "positive_share_contrast_rows": int(len(positive)),
        "predictive_utility_rows": int(len(utility_df)),
        "other_gain_omnibus_rows": int(len(other_omnibus)),
        "lead_shift_rows": int(len(lead_shift)),
        "raw_effect_rows": int(len(raw_effect)),
        "permutations": args.permutations,
        "bootstraps": args.bootstraps,
        "cv_repeats": args.cv_repeats,
        "cv_folds": args.cv_folds,
        "training_performed": False,
        "predictions_modified": False,
        "prototypes_refitted": False,
        "meters_reassigned": False,
        "meters_excluded": False,
        "causal_claims": False,
        "elapsed_seconds": elapsed,
    }
    status_path = paths.metadata / f"{ANALYSIS_NAME}_status.json"
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    outputs.append(status_path)

    inventory_path = paths.metadata / "group_heterogeneity_output_inventory.csv"
    inventory = write_inventory(paths.root, outputs, inventory_path)

    print("\n" + "=" * 100)
    print("GROUP HETEROGENEITY COMPLETE — PASS")
    print("=" * 100)
    print(f"Meters retained                 : 929")
    print(f"Source-transfer omnibus rows   : {len(source_omnibus)}")
    print(f"h=12 pairwise contrast rows    : {len(pairwise)}")
    print(f"Positive-share contrast rows   : {len(positive)}")
    print(f"Predictive-utility rows        : {len(utility_df)}")
    print(f"Raw-characteristic effects     : {len(raw_effect)}")
    print(f"Output inventory rows          : {len(inventory)}")
    print(f"Elapsed seconds                : {elapsed:.1f}")
    print(f"Tables                         : {paths.tables}")
    print(f"Decision note                  : {paths.decision}")
    print("Training performed              : False")
    print("Predictions modified            : False")
    print("Final status                    : COMPLETE_PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"GROUP HETEROGENEITY FAILED: {exc}", file=sys.stderr)
        raise
