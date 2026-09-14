from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

ANALYSIS_SLUG = "cross_strategy_profile_group_cross_strategy_profile_group_appendix_closure"
TABLE_ROOT_REL = Path("outputs/tables") / ANALYSIS_SLUG
META_ROOT_REL = Path("outputs/metadata") / ANALYSIS_SLUG
DECISION_REL = Path("documentation/technical_decisions/cross_strategy_profile_group_cross_strategy_profile_group_appendix_closure_decision.md")

STRATEGIES = ["direct_transfer", "fine_tuning", "cer_scratch_limited", "cer_scratch_full"]
COMPARISONS = {
    "direct_over_limited": ("direct_transfer", "cer_scratch_limited"),
    "fine_tuning_over_direct": ("fine_tuning", "direct_transfer"),
    "full_over_limited": ("cer_scratch_full", "cer_scratch_limited"),
    "direct_over_full": ("direct_transfer", "cer_scratch_full"),
    "fine_tuning_over_limited": ("fine_tuning", "cer_scratch_limited"),
    "full_over_fine_tuning": ("cer_scratch_full", "fine_tuning"),
}
STRATEGY_DISPLAY = {
    "direct_transfer": "Direct Transfer",
    "fine_tuning": "Fine Tuning",
    "cer_scratch_limited": "Scratch Limited",
    "cer_scratch_full": "Scratch Full",
}

# Rounded reported profile-group anchors. Tolerances account for reported rounding.
LD_DESC_ANCHORS = {
    "G1": {"mae": 0.0082, "relative_mae": 1.69, "mae_pos": 53.9, "rmse": 0.0242, "rmse_pos": 96.5, "smape": -0.27, "smape_pos": 39.0},
    "G2": {"mae": 0.0153, "relative_mae": 3.39, "mae_pos": 71.4, "rmse": 0.0343, "rmse_pos": 94.8, "smape": 0.57, "smape_pos": 53.5},
    "G3": {"mae": 0.0184, "relative_mae": 5.01, "mae_pos": 81.2, "rmse": 0.0367, "rmse_pos": 99.4, "smape": 1.18, "smape_pos": 60.8},
    "G4": {"mae": 0.0247, "relative_mae": 6.76, "mae_pos": 88.5, "rmse": 0.0388, "rmse_pos": 99.3, "smape": 2.47, "smape_pos": 73.4},
}
LD_EFFECT_ANCHORS = {
    ("MAE", "absolute"): (0.0307, 0.0275),
    ("MAE", "relative"): (0.0613, 0.0582),
    ("RMSE", "absolute"): (0.0171, 0.0139),
    ("RMSE", "relative"): (0.0474, 0.0442),
    ("SMAPE", "absolute"): (0.0484, 0.0453),
    ("SMAPE", "relative"): (0.0509, 0.0478),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def norm(s: str) -> str:
    return "".join(ch.lower() for ch in str(s) if ch.isalnum())


def choose_column(columns: Iterable[str], exact_aliases: List[str], token_sets: List[Tuple[str, ...]], exclude_tokens: Tuple[str, ...] = ()) -> str:
    cols = list(columns)
    by_norm = {norm(c): c for c in cols}
    for alias in exact_aliases:
        if norm(alias) in by_norm:
            return by_norm[norm(alias)]
    candidates = []
    for c in cols:
        n = norm(c)
        if any(t in n for t in exclude_tokens):
            continue
        for toks in token_sets:
            if all(t in n for t in toks):
                score = sum(len(t) for t in toks)
                if "error" in n:
                    score += 30
                if "metric" in n:
                    score += 5
                candidates.append((score, len(n), c))
                break
    if not candidates:
        raise KeyError(f"Cannot resolve required column. aliases={exact_aliases}; tokens={token_sets}; available={cols}")
    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
    return candidates[0][2]


def resolve_schema(df: pd.DataFrame) -> dict:
    cols = list(df.columns)
    schema = {}
    schema["meter"] = choose_column(cols, ["meter_id", "entity_id"], [("meter", "id"), ("entity", "id")])
    schema["lead"] = choose_column(cols, ["lead", "horizon", "h"], [("lead",), ("horizon",)])
    schema["metric"] = choose_column(cols, ["metric", "criterion", "error_metric"], [("metric",), ("criterion",)])
    schema["group"] = choose_column(cols, ["profile_group", "group", "assigned_group", "group_id"], [("profile", "group"), ("assigned", "group"), ("group",)], exclude_tokens=("gain", "error"))

    aliases = {
        "direct_transfer": ["direct_transfer_error", "direct_error", "error_direct_transfer"],
        "fine_tuning": ["fine_tuning_error", "finetuning_error", "error_fine_tuning"],
        "cer_scratch_limited": ["cer_scratch_limited_error", "scratch_limited_error", "limited_error", "error_cer_scratch_limited"],
        "cer_scratch_full": ["cer_scratch_full_error", "scratch_full_error", "full_error", "error_cer_scratch_full"],
    }
    tokens = {
        "direct_transfer": [("direct", "transfer", "error"), ("direct", "error")],
        "fine_tuning": [("fine", "tuning", "error"), ("finetuning", "error")],
        "cer_scratch_limited": [("scratch", "limited", "error"), ("limited", "error")],
        "cer_scratch_full": [("scratch", "full", "error"), ("full", "error")],
    }
    for strategy in STRATEGIES:
        schema[strategy] = choose_column(cols, aliases[strategy], tokens[strategy], exclude_tokens=("gain", "relative", "focal", "comparator"))
    if len(set(schema.values())) != len(schema.values()):
        raise RuntimeError(f"Resolved schema reused a column unexpectedly: {schema}")
    return schema


def canonical_group(x) -> str:
    s = str(x).strip().upper()
    if s.startswith("G"):
        digits = "".join(ch for ch in s if ch.isdigit())
    else:
        try:
            digits = str(int(float(s)))
        except Exception:
            digits = "".join(ch for ch in s if ch.isdigit())
    if digits not in {"1", "2", "3", "4"}:
        raise ValueError(f"Unexpected profile group value: {x!r}")
    return f"G{digits}"


def canonical_metric(x) -> str:
    s = str(x).strip().upper().replace("S-MAPE", "SMAPE").replace("SMAPE%", "SMAPE")
    if "SMAPE" in s:
        return "SMAPE"
    if "RMSE" in s:
        return "RMSE"
    if "MAE" in s:
        return "MAE"
    raise ValueError(f"Unexpected metric value: {x!r}")


def load_and_validate_input(project_root: Path, config: dict) -> Tuple[pd.DataFrame, dict, dict]:
    outcome_path = project_root / config["local_source_support_outcome_path"]
    status_path = project_root / config["local_source_support_status_path"]
    if not outcome_path.exists():
        raise FileNotFoundError(f"Mandatory Local source support outcome file missing: {outcome_path}")
    if not status_path.exists():
        raise FileNotFoundError(f"Mandatory Local source support status file missing: {status_path}")
    actual_sha = sha256_file(outcome_path)
    status_text = status_path.read_text(encoding="utf-8", errors="replace")
    if "COMPLETE_PASS" not in status_text and "COMPLETE" not in status_text:
        raise RuntimeError(f"Local source support status does not contain COMPLETE_PASS/COMPLETE: {status_path}")

    df = pd.read_csv(outcome_path)
    if len(df) != int(config["expected_local_source_support_rows"]):
        raise RuntimeError(f"Local source support row count mismatch: expected {config['expected_local_source_support_rows']}; actual {len(df)}")
    schema = resolve_schema(df)
    work = df.copy()
    work["_meter_id"] = work[schema["meter"]].astype(str)
    work["_lead"] = pd.to_numeric(work[schema["lead"]], errors="raise").astype(int)
    work["_metric"] = work[schema["metric"]].map(canonical_metric)
    work["_group"] = work[schema["group"]].map(canonical_group)
    for strategy in STRATEGIES:
        work[f"_{strategy}_error"] = pd.to_numeric(work[schema[strategy]], errors="raise").astype(float)

    leads = sorted(work["_lead"].unique().tolist())
    if leads != [1, 12, 48]:
        raise RuntimeError(f"Unexpected Local source support leads: {leads}")
    metrics = sorted(work["_metric"].unique().tolist())
    if metrics != ["MAE", "RMSE", "SMAPE"]:
        raise RuntimeError(f"Unexpected metrics: {metrics}")
    if work["_meter_id"].nunique() != int(config["expected_meters"]):
        raise RuntimeError("Local source support unique meter count mismatch.")
    if work.duplicated(["_meter_id", "_lead", "_metric"]).any():
        raise RuntimeError("Duplicate meter × lead × metric rows in Local source support outcome input.")
    if work[[f"_{s}_error" for s in STRATEGIES]].isna().any().any():
        raise RuntimeError("Non-finite strategy errors found in Local source support outcome input.")

    h12 = work.loc[work["_lead"] == int(config["formal_lead"])].copy()
    if len(h12) != int(config["expected_meters"]) * 3:
        raise RuntimeError(f"h=12 row count mismatch: {len(h12)}")
    for metric in ["MAE", "RMSE", "SMAPE"]:
        sub = h12.loc[h12["_metric"] == metric]
        counts = sub.groupby("_group")["_meter_id"].nunique().to_dict()
        expected = {k: int(v) for k, v in config["expected_group_counts"].items()}
        if counts != expected:
            raise RuntimeError(f"h=12 {metric} group count mismatch: expected {expected}; actual {counts}")

    record = {
        "local_source_support_outcome_path": str(outcome_path),
        "local_source_support_outcome_sha256": actual_sha,
        "local_source_support_status_path": str(status_path),
        "input_rows": int(len(work)),
        "unique_meters": int(work["_meter_id"].nunique()),
        "leads": leads,
        "metrics": metrics,
        "resolved_schema": schema,
    }
    return work, schema, record


def construct_h12_long(work: pd.DataFrame, config: dict) -> pd.DataFrame:
    h12 = work.loc[work["_lead"] == int(config["formal_lead"])].copy()
    rows = []
    for _, d in h12.iterrows():
        errors = {s: float(d[f"_{s}_error"]) for s in STRATEGIES}
        for code in config["comparison_order"]:
            focal, comparator = COMPARISONS[code]
            focal_error = errors[focal]
            comparator_error = errors[comparator]
            absolute_gain = comparator_error - focal_error
            if comparator_error == 0.0:
                relative_gain = np.nan
            else:
                relative_gain = 100.0 * absolute_gain / comparator_error
            rows.append({
                "meter_id": str(d["_meter_id"]),
                "lead": int(d["_lead"]),
                "profile_group": d["_group"],
                "metric": d["_metric"],
                "comparison_code": code,
                "comparison": config["comparison_names"][code],
                "focal_strategy": STRATEGY_DISPLAY[focal],
                "comparator_strategy": STRATEGY_DISPLAY[comparator],
                "focal_error": focal_error,
                "comparator_error": comparator_error,
                "absolute_gain": absolute_gain,
                "relative_gain_pct": relative_gain,
                "positive_gain": bool(absolute_gain > 0.0),
            })
    out = pd.DataFrame(rows)
    expected_rows = int(config["expected_meters"]) * 3 * 6
    if len(out) != expected_rows:
        raise RuntimeError(f"Cross strategy profile group h12 long rows expected {expected_rows}; got {len(out)}")
    if out["relative_gain_pct"].isna().any():
        bad = int(out["relative_gain_pct"].isna().sum())
        raise RuntimeError(f"Relative gain denominator was zero for {bad} rows; cannot reproduce governed relative gains.")
    # Exact algebraic closure across absolute gains, by meter/metric.
    wide = out.pivot(index=["meter_id", "metric"], columns="comparison_code", values="absolute_gain")
    checks = {
        "L-D + D-T = L-T": np.max(np.abs(wide["direct_over_limited"] + wide["fine_tuning_over_direct"] - wide["fine_tuning_over_limited"])),
        "L-F + F-D = L-D": np.max(np.abs(wide["full_over_limited"] + wide["direct_over_full"] - wide["direct_over_limited"])),
        "D-T + T-F = D-F": np.max(np.abs(wide["fine_tuning_over_direct"] + wide["full_over_fine_tuning"] + wide["direct_over_full"])),
    }
    if max(checks.values()) > 1e-4:
        raise RuntimeError(f"Pairwise absolute-gain algebraic identity failed: {checks}")
    out.attrs["algebraic_checks"] = checks
    return out


def one_way_effect(y: np.ndarray, groups: np.ndarray) -> Tuple[float, float]:
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    mask = np.isfinite(y)
    y, groups = y[mask], groups[mask]
    levels = np.unique(groups)
    n = len(y)
    k = len(levels)
    if n <= k or k < 2:
        return np.nan, np.nan
    grand = float(np.mean(y))
    ss_between = 0.0
    ss_within = 0.0
    for g in levels:
        vals = y[groups == g]
        m = float(np.mean(vals))
        ss_between += len(vals) * (m - grand) ** 2
        ss_within += float(np.sum((vals - m) ** 2))
    ss_total = ss_between + ss_within
    if ss_total <= 0:
        return 0.0, 0.0
    eta2 = ss_between / ss_total
    df_between = k - 1
    df_within = n - k
    ms_within = ss_within / df_within if df_within > 0 else np.nan
    omega_raw = (ss_between - df_between * ms_within) / (ss_total + ms_within) if np.isfinite(ms_within) else np.nan
    omega2 = max(0.0, float(omega_raw)) if np.isfinite(omega_raw) else np.nan
    return float(eta2), omega2


def bootstrap_effect(y: np.ndarray, groups: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float, float, float]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    levels = np.unique(groups)
    eta_vals = np.empty(n_boot, dtype=float)
    omega_vals = np.empty(n_boot, dtype=float)
    idx_by_group = [np.flatnonzero(groups == g) for g in levels]
    for b in range(n_boot):
        pieces = [rng.choice(idx, size=len(idx), replace=True) for idx in idx_by_group]
        idx = np.concatenate(pieces)
        eta_vals[b], omega_vals[b] = one_way_effect(y[idx], groups[idx])
    return (
        float(np.nanquantile(eta_vals, 0.025)),
        float(np.nanquantile(eta_vals, 0.975)),
        float(np.nanquantile(omega_vals, 0.025)),
        float(np.nanquantile(omega_vals, 0.975)),
    )


def permutation_p(y: np.ndarray, groups: np.ndarray, observed_eta: float, n_perm: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(groups)
        eta, _ = one_way_effect(y, perm)
        if eta >= observed_eta - 1e-15:
            ge += 1
    return float((ge + 1) / (n_perm + 1))


def bh_adjust(pvals: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(pvals), dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q_ranked = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        q_ranked[i] = min(1.0, prev)
    q = np.empty(n, dtype=float)
    q[order] = q_ranked
    return q


def build_k1a(long_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []
    groups = ["G1", "G2", "G3", "G4"]
    outcome_specs = [
        ("MAE", "absolute_gain", "mean", "MAE Gain (kWh)"),
        ("MAE", "relative_gain_pct", "mean", "Relative MAE Gain (%)"),
        ("MAE", "positive_gain", "share", "Positive MAE households (%)"),
        ("RMSE", "absolute_gain", "mean", "RMSE Gain (kWh)"),
        ("RMSE", "positive_gain", "share", "Positive RMSE households (%)"),
        ("SMAPE", "absolute_gain", "mean", "sMAPE Gain (pp)"),
        ("SMAPE", "positive_gain", "share", "Positive sMAPE households (%)"),
    ]
    for code in config["comparison_order"]:
        comp = config["comparison_names"][code]
        for metric, col, kind, label in outcome_specs:
            sub = long_df[(long_df["comparison_code"] == code) & (long_df["metric"] == metric)]
            row = {"comparison": comp, "comparison_code": code, "outcome": label}
            for g in groups:
                vals = sub.loc[sub["profile_group"] == g, col]
                if len(vals) == 0:
                    raise RuntimeError(f"No rows for {code} {metric} {g}")
                value = float(vals.astype(float).mean())
                if kind == "share":
                    value *= 100.0
                row[g] = value
            rows.append(row)
    out = pd.DataFrame(rows)
    if len(out) != 42:
        raise RuntimeError(f"K.1A must have 42 rows; got {len(out)}")
    return out


def build_effect_table(long_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []
    n_boot = int(config["bootstrap_resamples"])
    n_perm = int(config["permutation_resamples"])
    base_seed = int(config["random_seed"])
    run_index = 0
    for code in config["comparison_order"]:
        for metric in ["MAE", "RMSE", "SMAPE"]:
            sub = long_df[(long_df["comparison_code"] == code) & (long_df["metric"] == metric)].sort_values("meter_id")
            groups = sub["profile_group"].to_numpy()
            for scale, col in [("absolute", "absolute_gain"), ("relative", "relative_gain_pct")]:
                y = sub[col].to_numpy(dtype=float)
                eta, omega = one_way_effect(y, groups)
                seed = base_seed + 10007 * run_index
                eta_lo, eta_hi, om_lo, om_hi = bootstrap_effect(y, groups, n_boot, seed)
                p = permutation_p(y, groups, eta, n_perm, seed + 7919)
                rows.append({
                    "comparison": config["comparison_names"][code],
                    "comparison_code": code,
                    "metric": metric,
                    "gain_scale": scale,
                    "n_meters": int(len(sub)),
                    "eta_squared": eta,
                    "eta_squared_ci_low": eta_lo,
                    "eta_squared_ci_high": eta_hi,
                    "omega_squared": omega,
                    "omega_squared_ci_low": om_lo,
                    "omega_squared_ci_high": om_hi,
                    "permutation_p": p,
                    "bootstrap_resamples": n_boot,
                    "permutation_resamples": n_perm,
                    "random_seed_stream": seed,
                })
                run_index += 1
    out = pd.DataFrame(rows)
    if len(out) != 36:
        raise RuntimeError(f"Complete effect table must have 36 rows; got {len(out)}")
    out["bh_q_all_36"] = bh_adjust(out["permutation_p"])
    out["bh_q_within_gain_scale_18"] = np.nan
    for scale in ["absolute", "relative"]:
        idx = out.index[out["gain_scale"] == scale]
        out.loc[idx, "bh_q_within_gain_scale_18"] = bh_adjust(out.loc[idx, "permutation_p"])
    out["bh_q"] = out["bh_q_all_36"]
    return out


def reproduction_audit(k1a: pd.DataFrame, effects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    def add(check, expected, actual, tol):
        passed = bool(abs(float(actual) - float(expected)) <= tol)
        rows.append({"check": check, "expected": expected, "actual": actual, "tolerance": tol, "pass": passed})

    ld = k1a[k1a["comparison_code"] == "direct_over_limited"].set_index("outcome")
    mapping = {
        "MAE Gain (kWh)": ("mae", 0.00011),
        "Relative MAE Gain (%)": ("relative_mae", 0.011),
        "Positive MAE households (%)": ("mae_pos", 0.11),
        "RMSE Gain (kWh)": ("rmse", 0.00011),
        "Positive RMSE households (%)": ("rmse_pos", 0.11),
        "sMAPE Gain (pp)": ("smape", 0.011),
        "Positive sMAPE households (%)": ("smape_pos", 0.11),
    }
    for outcome, (anchor_key, tol) in mapping.items():
        for g in ["G1", "G2", "G3", "G4"]:
            add(f"ProfileGroupAnalysis/D L-D descriptive {outcome} {g}", LD_DESC_ANCHORS[g][anchor_key], ld.loc[outcome, g], tol)

    ld_effect = effects[effects["comparison_code"] == "direct_over_limited"]
    for (metric, scale), (eta_anchor, om_anchor) in LD_EFFECT_ANCHORS.items():
        hit = ld_effect[(ld_effect["metric"] == metric) & (ld_effect["gain_scale"] == scale)]
        if len(hit) != 1:
            raise RuntimeError(f"Missing L-D effect row {metric} {scale}")
        r = hit.iloc[0]
        add(f"GroupHeterogeneity L-D eta2 {metric} {scale}", eta_anchor, r["eta_squared"], 0.00011)
        add(f"GroupHeterogeneity L-D omega2 {metric} {scale}", om_anchor, r["omega_squared"], 0.00011)

    out = pd.DataFrame(rows)
    if not bool(out["pass"].all()):
        failed = out.loc[~out["pass"]]
        raise RuntimeError("Group heterogeneity reproduction gate failed:\n" + failed.to_string(index=False))
    return out


def make_presentation_tables(effects: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    def make(scale: str) -> pd.DataFrame:
        x = effects[effects["gain_scale"] == scale].copy()
        x["eta_squared_95_ci"] = x.apply(lambda r: f"{r.eta_squared_ci_low:.6f}–{r.eta_squared_ci_high:.6f}", axis=1)
        x["omega_squared_95_ci"] = x.apply(lambda r: f"{r.omega_squared_ci_low:.6f}–{r.omega_squared_ci_high:.6f}", axis=1)
        return x[[
            "comparison", "metric", "eta_squared", "eta_squared_95_ci",
            "omega_squared", "omega_squared_95_ci", "permutation_p", "bh_q"
        ]].reset_index(drop=True)
    return make("absolute"), make("relative")


def write_outputs(project_root: Path, config: dict, input_record: dict, long_df: pd.DataFrame, k1a: pd.DataFrame, effects: pd.DataFrame, audit: pd.DataFrame) -> dict:
    table_root = project_root / TABLE_ROOT_REL
    meta_root = project_root / META_ROOT_REL
    decision_path = project_root / DECISION_REL
    table_root.mkdir(parents=True, exist_ok=True)
    meta_root.mkdir(parents=True, exist_ok=True)
    decision_path.parent.mkdir(parents=True, exist_ok=True)

    paths = {
        "household_long": table_root / "cross_strategy_profile_group_h12_cross_strategy_household_gain_long.csv",
        "k1a": table_root / "cross_strategy_profile_group_table_k1a_profile_group_gains.csv",
        "k1b_complete": table_root / "cross_strategy_profile_group_table_k1b_complete_effect_size_tests.csv",
        "k1b_absolute": table_root / "cross_strategy_profile_group_table_k1b_absolute_gain_presentation.csv",
        "k1b_relative": table_root / "cross_strategy_profile_group_table_k1b_relative_gain_supplement.csv",
        "reproduction_audit": table_root / "cross_strategy_profile_group_group_heterogeneity_reproduction_audit.csv",
        "input_record": meta_root / "cross_strategy_profile_group_input_record.json",
        "status": meta_root / "cross_strategy_profile_group_analysis_status.json",
        "inventory": meta_root / "cross_strategy_profile_group_output_inventory.csv",
        "decision": decision_path,
    }
    long_df.to_csv(paths["household_long"], index=False)
    k1a.to_csv(paths["k1a"], index=False)
    effects.to_csv(paths["k1b_complete"], index=False)
    abs_pres, rel_pres = make_presentation_tables(effects)
    abs_pres.to_csv(paths["k1b_absolute"], index=False)
    rel_pres.to_csv(paths["k1b_relative"], index=False)
    audit.to_csv(paths["reproduction_audit"], index=False)
    save_json_atomic(paths["input_record"], input_record)

    # Human-readable decision note is generated from the actual run, not from pre-filled results.
    min_q = float(effects["bh_q"].min())
    decision = "# Cross strategy profile group Cross-Strategy Profile-Group Appendix Closure — Decision Record\n\n"
    decision += f"Status: COMPLETE_PASS\n\n"
    decision += f"Formal lead: h={config['formal_lead']}\n\n"
    decision += f"Meters: {config['expected_meters']}\n\n"
    decision += "This analysis uses the checksum-verified household-level cross-strategy outcome table and does not train models or alter profile assignments.\n\n"
    decision += "Six pairwise comparisons were evaluated with the sign convention: positive gain favours the first named strategy.\n\n"
    decision += f"Complete K.1B rows: {len(effects)} (6 comparisons × 3 metrics × 2 gain scales).\n\n"
    decision += f"Bootstrap resamples per test: {config['bootstrap_resamples']}; permutation resamples per test: {config['permutation_resamples']}.\n\n"
    decision += f"Primary BH family: all 36 h=12 omnibus tests; minimum q = {min_q:.6g}.\n\n"
    decision += "The Direct-over-Limited descriptive and effect-size checks passed. Bootstrap intervals and permutation p-values are deterministic under the specified random seed.\n"
    paths["decision"].write_text(decision, encoding="utf-8")

    algebra = long_df.attrs.get("algebraic_checks", {})
    status = {
        "analysis": ANALYSIS_SLUG,
        "analysis_id": config["analysis_id"],
        "status": "COMPLETE_PASS",
        "formal_lead": int(config["formal_lead"]),
        "meters": int(config["expected_meters"]),
        "group_counts": config["expected_group_counts"],
        "comparisons": 6,
        "metrics": 3,
        "gain_scales": 2,
        "household_gain_long_rows": int(len(long_df)),
        "table_k1a_rows": int(len(k1a)),
        "table_k1b_complete_rows": int(len(effects)),
        "table_k1b_absolute_rows": int(len(abs_pres)),
        "table_k1b_relative_rows": int(len(rel_pres)),
        "group_heterogeneity_reproduction_checks": int(len(audit)),
        "group_heterogeneity_reproduction_all_pass": bool(audit["pass"].all()),
        "pairwise_algebraic_max_abs_diff": float(max(algebra.values())) if algebra else None,
        "bootstrap_resamples": int(config["bootstrap_resamples"]),
        "permutation_resamples": int(config["permutation_resamples"]),
        "random_seed": int(config["random_seed"]),
        "bh_primary_family": config["bh_primary_family"],
        "training_performed": False,
        "predictions_modified": False,
        "prototypes_refitted": False,
        "meters_reassigned_or_excluded": False,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    save_json_atomic(paths["status"], status)

    inv_rows = []
    for role, p in paths.items():
        if role == "inventory":
            continue
        inv_rows.append({
            "role": role,
            "path": str(p.relative_to(project_root)),
            "exists": p.exists(),
            "size_bytes": p.stat().st_size if p.exists() else -1,
            "sha256": sha256_file(p) if p.exists() else "",
        })
    pd.DataFrame(inv_rows).to_csv(paths["inventory"], index=False)
    return {k: str(v) for k, v in paths.items()}


def run_self_test() -> None:
    rng = np.random.default_rng(123)
    groups = np.repeat(np.array(["G1", "G2", "G3", "G4"]), [10, 10, 10, 10])
    y = np.concatenate([rng.normal(loc=i, scale=0.5, size=10) for i in range(4)])
    eta, omega = one_way_effect(y, groups)
    if not (0 < omega <= eta < 1):
        raise RuntimeError(f"Synthetic effect-size self-test failed: eta={eta}, omega={omega}")
    lo, hi, olo, ohi = bootstrap_effect(y, groups, 100, 9)
    if not (0 <= lo <= hi <= 1 and 0 <= olo <= ohi <= 1):
        raise RuntimeError("Bootstrap self-test failed.")
    p = permutation_p(y, groups, eta, 199, 10)
    if not (0 < p <= 1):
        raise RuntimeError("Permutation p self-test failed.")
    q = bh_adjust([0.01, 0.02, 0.5])
    if np.any(q < np.array([0.01, 0.02, 0.5])):
        raise RuntimeError("BH self-test failed.")
    # Gain sign identities.
    D, T, L, F = 2.0, 2.4, 3.0, 1.8
    gains = {
        "direct_over_limited": L-D,
        "fine_tuning_over_direct": D-T,
        "full_over_limited": L-F,
        "direct_over_full": F-D,
        "fine_tuning_over_limited": L-T,
        "full_over_fine_tuning": T-F,
    }
    if abs(gains["direct_over_limited"] + gains["fine_tuning_over_direct"] - gains["fine_tuning_over_limited"]) > 1e-12:
        raise RuntimeError("Gain identity self-test failed.")
    print("CROSS_STRATEGY_PROFILE_GROUP SYNTHETIC SELF-TEST: PASS")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--input-gate-only", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        run_self_test()
        return

    project_root = Path(args.project_root).expanduser().resolve()
    config = load_json(Path(args.config).expanduser().resolve())
    work, schema, input_record = load_and_validate_input(project_root, config)
    print("CROSS_STRATEGY_PROFILE_GROUP INPUT GATE: PASS")
    print("LocalSourceSupport SHA256:", input_record["local_source_support_outcome_sha256"])
    print("Resolved schema:", json.dumps(schema, indent=2))
    if args.input_gate_only:
        return

    long_df = construct_h12_long(work, config)
    k1a = build_k1a(long_df, config)
    effects = build_effect_table(long_df, config)
    audit = reproduction_audit(k1a, effects)
    paths = write_outputs(project_root, config, input_record, long_df, k1a, effects, audit)
    print("CROSS_STRATEGY_PROFILE_GROUP ANALYSIS COMPLETE — PASS")
    print("K1A rows:", len(k1a))
    print("K1B complete rows:", len(effects))
    print("Reproduction checks:", len(audit), "all pass")
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
