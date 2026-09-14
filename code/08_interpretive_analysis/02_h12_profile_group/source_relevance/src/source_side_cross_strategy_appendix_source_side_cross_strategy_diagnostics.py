#!/usr/bin/env python3
"""Generate the reported h=12 source-similarity and local-support association tables.

The analysis uses the generated household-level source-support outcome dataset. It
reports nearest-prototype DTW distance, assignment margin/ratio, and mean distance
to the ten nearest same-group source households. Associations are descriptive and
post-hoc; profile groups, prototypes, predictions and strategy errors are unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ANALYSIS_ID = "source_side_cross_strategy"
STRATEGY_COLS = {
    "direct_transfer": "strategy_error_direct_transfer",
    "fine_tuning": "strategy_error_fine_tuning",
    "cer_scratch_limited": "strategy_error_cer_scratch_limited",
    "cer_scratch_full": "strategy_error_cer_scratch_full",
}
GAIN_PREFIX = {
    "source_transfer": "gain_source_transfer",
    "fine_tuning": "gain_fine_tuning",
    "full_data": "gain_full_data",
    "direct_vs_full": "gain_direct_vs_full",
    "fine_tuning_vs_limited": "gain_fine_tuning_vs_limited",
    "full_vs_fine_tuning": "gain_full_vs_fine_tuning",
}


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bh_adjust(values: np.ndarray) -> np.ndarray:
    p = np.asarray(values, float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty(len(p), float)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def rank_residual(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    ranks = pd.Series(values).rank(method="average").to_numpy(float)
    indicators = pd.get_dummies(pd.Series(groups).astype(int), drop_first=True).to_numpy(float)
    design = np.column_stack([np.ones(len(ranks)), indicators])
    beta = np.linalg.lstsq(design, ranks, rcond=None)[0]
    return ranks - design @ beta


def association(x: np.ndarray, y: np.ndarray, groups: np.ndarray, adjustment: str) -> tuple[float, float]:
    if adjustment == "None":
        rho, p = stats.spearmanr(x, y)
    else:
        rho, p = stats.pearsonr(rank_residual(x, groups), rank_residual(y, groups))
    return float(rho), float(p)


def centered_rank(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = stats.rankdata(values, method="average").astype(float)
    for group in np.unique(groups):
        mask = groups == group
        out[mask] -= out[mask].mean()
    return out


def correlation_fast(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    ac = a - a.mean(); bc = b - b.mean()
    den = np.sqrt(np.dot(ac, ac) * np.dot(bc, bc))
    return float(np.dot(ac, bc) / den) if den > 0 else np.nan


def gain_column(comparison_id: str, scale: str) -> str:
    suffix = "absolute" if scale == "absolute" else "relative_pct"
    return f"{GAIN_PREFIX[comparison_id]}_{suffix}"


def validate(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    d = df[df["lead"].eq(int(cfg["primary_lead"]))].copy()
    ensure(set(d["criterion"]) == set(cfg["criteria"]), "Unexpected h=12 criteria")
    measures = {item["column"] for item in cfg["source_measures"]}
    needed = {"meter_id", "lead", "criterion", "assigned_group", *measures}
    for comp in cfg["comparisons"]:
        needed.update({STRATEGY_COLS[comp["focal"]], STRATEGY_COLS[comp["comparator"]]})
        needed.update({gain_column(comp["id"], "absolute"), gain_column(comp["id"], "relative")})
    ensure(needed.issubset(d.columns), f"Outcome dataset missing columns: {sorted(needed-set(d.columns))}")
    expected_groups = {int(k): int(v) for k, v in cfg["expected_group_counts"].items()}
    for criterion in cfg["criteria"]:
        z = d[d["criterion"].eq(criterion)]
        ensure(len(z) == int(cfg["expected_meters"]), f"{criterion}: expected 929 rows")
        ensure(z["meter_id"].nunique() == int(cfg["expected_meters"]), f"{criterion}: household IDs not unique")
        ensure(z["assigned_group"].astype(int).value_counts().sort_index().to_dict() == expected_groups,
               f"{criterion}: group counts differ from the fixed assignment")
    ensure(np.isfinite(d[list(measures)].to_numpy(float)).all(), "Non-finite source-side measures")
    return d


def gain_formula_audit(d: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    for comp in cfg["comparisons"]:
        focal = STRATEGY_COLS[comp["focal"]]
        comparator = STRATEGY_COLS[comp["comparator"]]
        calc_abs = d[comparator].to_numpy(float) - d[focal].to_numpy(float)
        existing_abs = d[gain_column(comp["id"], "absolute")].to_numpy(float)
        calc_rel = np.full(len(d), np.nan, float)
        denom = d[comparator].to_numpy(float)
        np.divide(100.0 * calc_abs, denom, out=calc_rel, where=denom != 0)
        existing_rel = d[gain_column(comp["id"], "relative")].to_numpy(float)
        finite = np.isfinite(calc_rel) & np.isfinite(existing_rel)
        da = float(np.max(np.abs(calc_abs-existing_abs)))
        dr = float(np.max(np.abs(calc_rel[finite]-existing_rel[finite]))) if finite.any() else 0.0
        rows.append({"comparison_id": comp["id"], "max_absolute_gain_difference": da,
                     "max_relative_gain_difference": dr, "status": "PASS" if da <= 1e-4 and dr <= 1e-4 else "FAIL"})
    out = pd.DataFrame(rows)
    ensure(out["status"].eq("PASS").all(), "Gain formula audit failed")
    return out


def build_associations(d: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    for measure in cfg["source_measures"]:
        for adjustment in measure["adjustments"]:
            for comp in cfg["comparisons"]:
                for scale in cfg["gain_scales"]:
                    for criterion in cfg["criteria"]:
                        z = d[d["criterion"].eq(criterion)].sort_values("meter_id")
                        x = z[measure["column"]].to_numpy(float)
                        y = z[gain_column(comp["id"], scale)].to_numpy(float)
                        groups = z["assigned_group"].to_numpy(int)
                        ensure(np.isfinite(x).all() and np.isfinite(y).all(), "Non-finite association values")
                        rho, p = association(x, y, groups, adjustment)
                        rows.append({
                            "evidence_type": measure["evidence_type"], "measure": measure["measure"],
                            "measure_column": measure["column"], "adjustment": adjustment,
                            "comparison_id": comp["id"], "comparison": comp["label"], "formula": comp["formula"],
                            "gain_scale": scale, "criterion": criterion, "n": len(z), "rho": rho, "p_value": p,
                        })
    out = pd.DataFrame(rows)
    ensure(len(out) == 216, f"Expected 216 association rows; found {len(out)}")
    out["q_value_bh"] = np.nan
    out["bh_family"] = ""
    for key, indices in out.groupby(["adjustment", "criterion", "gain_scale"], sort=False).groups.items():
        indices = list(indices)
        expected = 12 if key[0] == "None" else 24
        ensure(len(indices) == expected, f"Unexpected BH family size for {key}: {len(indices)}")
        out.loc[indices, "q_value_bh"] = bh_adjust(out.loc[indices, "p_value"].to_numpy(float))
        out.loc[indices, "bh_family"] = "__".join(map(str, key))
    return out


def add_bootstrap_cis(d: pd.DataFrame, associations: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    repetitions = int(cfg["association"]["bootstrap_repetitions"])
    seed = int(cfg["association"]["seed"])
    confidence = float(cfg["association"].get("bootstrap_confidence", 0.95))
    ensure(repetitions >= 1000, "At least 1,000 household bootstrap repetitions are required")
    meters = sorted(d[d["criterion"].eq(cfg["criteria"][0])]["meter_id"].tolist())
    aligned = {c: d[d["criterion"].eq(c)].set_index("meter_id").loc[meters].reset_index() for c in cfg["criteria"]}
    base = aligned[cfg["criteria"][0]]
    groups = base["assigned_group"].to_numpy(int)
    n = len(base)
    source = {m["column"]: base[m["column"]].to_numpy(float) for m in cfg["source_measures"]}
    outcomes = {(criterion, comp["id"], scale): aligned[criterion][gain_column(comp["id"], scale)].to_numpy(float)
                for criterion in cfg["criteria"] for comp in cfg["comparisons"] for scale in cfg["gain_scales"]}
    row_map = {(r.measure_column, r.adjustment, r.comparison_id, r.gain_scale, r.criterion): i
               for i, r in associations.reset_index(drop=True).iterrows()}
    boot = np.full((len(associations), repetitions), np.nan, float)
    rng = np.random.default_rng(seed)
    for repetition in range(repetitions):
        idx = rng.integers(0, n, size=n)
        sampled_groups = groups[idx]
        x_cache = {}
        for measure in cfg["source_measures"]:
            column = measure["column"]
            ranked = stats.rankdata(source[column][idx], method="average")
            if "None" in measure["adjustments"]:
                x_cache[(column, "None")] = ranked
            if "Group adjusted" in measure["adjustments"]:
                x_cache[(column, "Group adjusted")] = centered_rank(source[column][idx], sampled_groups)
        for (criterion, comp_id, scale), y in outcomes.items():
            y_none = stats.rankdata(y[idx], method="average")
            y_adjusted = centered_rank(y[idx], sampled_groups)
            for measure in cfg["source_measures"]:
                for adjustment in measure["adjustments"]:
                    row = row_map[(measure["column"], adjustment, comp_id, scale, criterion)]
                    boot[row, repetition] = correlation_fast(
                        x_cache[(measure["column"], adjustment)],
                        y_none if adjustment == "None" else y_adjusted,
                    )
    ensure(np.isfinite(boot).all(), "Non-finite bootstrap correlations")
    alpha = (1-confidence)/2
    out = associations.reset_index(drop=True).copy()
    out["rho_bootstrap_ci_lower"] = np.quantile(boot, alpha, axis=1)
    out["rho_bootstrap_ci_upper"] = np.quantile(boot, 1-alpha, axis=1)
    out["bootstrap_repetitions"] = repetitions
    return out


def report_tables(associations: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Appendix G: L-D absolute gain, six requested measure/adjustment rows, metrics across columns.
    g = associations[(associations["comparison_id"].eq("source_transfer")) & (associations["gain_scale"].eq("absolute"))].copy()
    key = ["evidence_type", "measure", "measure_column", "adjustment"]
    g_table = g[key].drop_duplicates().copy()
    for criterion in cfg["criteria"]:
        z = g[g["criterion"].eq(criterion)][key + ["rho", "q_value_bh"]].rename(
            columns={"rho": f"{criterion}_rho", "q_value_bh": f"{criterion}_bh_q"})
        g_table = g_table.merge(z, on=key, how="left", validate="one_to_one")

    # Appendix H: each comparison × gain scale, with the six source measure columns and CIs/q.
    base_keys = ["comparison_id", "comparison", "formula", "gain_scale", "criterion"]
    h_rows = associations[base_keys].drop_duplicates().copy()
    for measure in cfg["source_measures"]:
        for adjustment in measure["adjustments"]:
            label = measure["column"] + ("_group_adjusted" if adjustment == "Group adjusted" else "_none")
            z = associations[(associations["measure_column"].eq(measure["column"])) & (associations["adjustment"].eq(adjustment))]
            z = z[base_keys + ["rho", "rho_bootstrap_ci_lower", "rho_bootstrap_ci_upper", "q_value_bh"]].rename(columns={
                "rho": f"{label}_rho", "rho_bootstrap_ci_lower": f"{label}_ci_lower",
                "rho_bootstrap_ci_upper": f"{label}_ci_upper", "q_value_bh": f"{label}_bh_q",
            })
            h_rows = h_rows.merge(z, on=base_keys, how="left", validate="one_to_one")
    comp_order = {c["id"]: i for i, c in enumerate(cfg["comparisons"])}
    scale_order = {"absolute": 0, "relative": 1}; criterion_order={c:i for i,c in enumerate(cfg["criteria"])}
    h_rows["_c"] = h_rows["comparison_id"].map(comp_order); h_rows["_s"] = h_rows["gain_scale"].map(scale_order); h_rows["_m"] = h_rows["criterion"].map(criterion_order)
    h_rows = h_rows.sort_values(["_c","_s","_m"]).drop(columns=["_c","_s","_m"]).reset_index(drop=True)
    ensure(len(g_table) == 6, f"Appendix G table expected 6 rows; found {len(g_table)}")
    ensure(len(h_rows) == 36, f"Appendix H table expected 36 rows; found {len(h_rows)}")
    return g_table, h_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    config_path = args.config or root / "code/08_interpretive_analysis/02_h12_profile_group/source_relevance/configs/source_side_cross_strategy_appendix_source_side_cross_strategy_diagnostics_config.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    input_path = root / cfg["inputs"]["local_source_support_outcome"]
    ensure(input_path.is_file(), f"Missing generated source-support outcome: {input_path}")
    d = validate(pd.read_csv(input_path), cfg)
    gain_audit = gain_formula_audit(d, cfg)
    associations = add_bootstrap_cis(d, build_associations(d, cfg), cfg)
    table_g, table_h = report_tables(associations, cfg)

    table_dir = root / cfg["outputs"]["table_dir"]
    metadata_dir = root / cfg["outputs"]["metadata_dir"]
    table_dir.mkdir(parents=True, exist_ok=True); metadata_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name, frame in [
        ("source_side_associations_long.csv", associations),
        ("appendix_g_ld_source_similarity_support.csv", table_g),
        ("appendix_h_cross_strategy_source_similarity_support.csv", table_h),
        ("source_side_gain_formula_audit.csv", gain_audit),
    ]:
        path = table_dir / name; frame.to_csv(path, index=False); outputs.append(path)
    status = {
        "analysis_id": ANALYSIS_ID, "status": "COMPLETE_PASS", "lead": int(cfg["primary_lead"]),
        "households": int(cfg["expected_meters"]), "association_rows": len(associations),
        "appendix_g_rows": len(table_g), "appendix_h_rows": len(table_h),
        "bootstrap_repetitions": int(cfg["association"]["bootstrap_repetitions"]),
        "training_performed": False, "predictions_modified": False,
        "prototypes_refitted": False, "meters_reassigned": False,
        "interpretation": "post-hoc descriptive source-similarity and local-support associations",
    }
    status_path = metadata_dir / "source_side_cross_strategy_status.json"
    status_path.write_text(json.dumps(status, indent=2)+"\n", encoding="utf-8"); outputs.append(status_path)
    inventory = pd.DataFrame([{"path": str(p.relative_to(root)), "size_bytes": p.stat().st_size, "sha256": sha256(p)} for p in outputs])
    inventory.to_csv(metadata_dir / "source_side_cross_strategy_output_inventory.csv", index=False)
    print("SOURCE SIDE CROSS STRATEGY: PASS")
    print(f"Association rows: {len(associations)}")
    print(f"Appendix G rows: {len(table_g)}")
    print(f"Appendix H rows: {len(table_h)}")


if __name__ == "__main__":
    main()
