from __future__ import annotations

import argparse
import itertools
import math
import multiprocessing as mp
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, silhouette_score
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import cdist_dtw

from profile_clustering_final_k_common import (
    ProfileClusteringFinalKError,
    ensure,
    load_yaml,
    resolve_path,
    sample_std,
    slot_columns,
    write_json,
)


_GLOBAL: dict[str, Any] = {}


def _extract_low_signal_mask(frame: pd.DataFrame) -> np.ndarray:
    """Return the governed prototype-fit exclusion mask across schema variants."""

    exclusion_columns = (
        "low_signal_prototype_fit_exclusion",
        "prototype_fit_excluded",
        "prototype_fit_exclusion",
        "excluded_from_prototype_fit",
        "low_signal_flag",
        "is_low_signal",
        "low_signal",
    )

    eligible_columns = (
        "prototype_fit_eligible",
        "fit_eligible",
        "prototype_fit_included",
        "used_for_prototype_fit",
    )

    std_columns = (
        "raw_profile_std_kwh_ddof0",
        "raw_profile_std_kwh",
        "raw_profile_sd_kwh",
        "profile_std_kwh",
        "raw_std_kwh",
    )

    def as_bool(series: pd.Series) -> np.ndarray:
        if pd.api.types.is_bool_dtype(series):
            return series.to_numpy(dtype=bool)

        if pd.api.types.is_numeric_dtype(series):
            values = pd.to_numeric(series, errors="raise")
            unique = set(values.dropna().unique().tolist())
            if not unique.issubset({0, 1, 0.0, 1.0}):
                raise ProfileClusteringFinalKError(
                    f"Boolean-like column contains non-binary values: {unique}"
                )
            return values.astype(bool).to_numpy()

        values = series.astype(str).str.strip().str.lower()
        mapping = {
            "true": True, "false": False,
            "1": True, "0": False,
            "yes": True, "no": False,
        }
        unexpected = sorted(set(values.unique()) - set(mapping))
        if unexpected:
            raise ProfileClusteringFinalKError(
                f"Boolean-like column contains unexpected values: {unexpected}"
            )
        return values.map(mapping).to_numpy(dtype=bool)

    for column in exclusion_columns:
        if column in frame.columns:
            return as_bool(frame[column])

    for column in eligible_columns:
        if column in frame.columns:
            return ~as_bool(frame[column])

    # Formal locked rule: raw 48-slot profile SD < 0.01 kWh.
    for column in std_columns:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
            return values < 0.01

    raise ProfileClusteringFinalKError(
        "Cannot derive prototype-fit exclusion mask. "
        f"Available columns: {frame.columns.tolist()}"
    )



@dataclass(frozen=True)
class FitRequest:
    k: int
    seed: int
    run_scope: str


def _worker_init(
    zscore_path: str,
    low_signal_path: str,
    distance_path: str,
    expected_households: int,
    expected_fit_households: int,
) -> None:
    profile_frame = pd.read_csv(zscore_path)
    slots = slot_columns(profile_frame.columns.tolist())
    low_signal = pd.read_csv(low_signal_path)

    ensure(len(profile_frame) == expected_households, "Worker profile household count mismatch")
    ensure(len(low_signal) == expected_households, "Worker low-signal household count mismatch")
    ensure(
        profile_frame["household_id"].astype(str).tolist()
        == low_signal["household_id"].astype(str).tolist(),
        "Worker profile and low-signal household order mismatch",
    )

    all_values = profile_frame[slots].to_numpy(dtype=np.float64)
    low_mask = _extract_low_signal_mask(low_signal)
    fit_mask = ~low_mask
    ensure(int(fit_mask.sum()) == expected_fit_households, "Worker fit household count mismatch")
    ensure(np.isfinite(all_values).all(), "Worker profiles contain non-finite values")

    _GLOBAL["household_ids"] = profile_frame["household_id"].astype(str).to_numpy()
    _GLOBAL["all_values"] = np.ascontiguousarray(all_values[:, :, None])
    _GLOBAL["fit_values"] = np.ascontiguousarray(all_values[fit_mask, :, None])
    _GLOBAL["fit_mask"] = fit_mask
    _GLOBAL["low_mask"] = low_mask
    _GLOBAL["distance_path"] = distance_path


def _fit_one(request: FitRequest, settings: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    k = int(request.k)
    seed = int(request.seed)

    model = TimeSeriesKMeans(
        n_clusters=k,
        metric="dtw",
        max_iter=int(settings["max_iter"]),
        max_iter_barycenter=int(settings["max_iter_barycenter"]),
        tol=float(settings["tol"]),
        n_init=1,
        n_jobs=1,
        dtw_inertia=True,
        random_state=seed,
        init="k-means++",
        verbose=0,
    )

    labels_fit = model.fit_predict(_GLOBAL["fit_values"]).astype(np.int16, copy=False)
    labels_all = model.predict(_GLOBAL["all_values"]).astype(np.int16, copy=False)
    prototypes = np.asarray(model.cluster_centers_, dtype=np.float64).squeeze(-1)

    ensure(prototypes.shape == (k, 48), f"Unexpected prototype shape for K={k}, seed={seed}: {prototypes.shape}")
    ensure(len(np.unique(labels_fit)) == k, f"Fit did not produce all {k} clusters for seed {seed}")
    ensure(len(np.unique(labels_all)) == k, f"All-household assignment did not produce all {k} clusters for seed {seed}")

    distance_matrix = np.load(_GLOBAL["distance_path"], mmap_mode="r")
    silhouette = float(silhouette_score(distance_matrix, labels_fit, metric="precomputed"))

    fit_sizes = np.bincount(labels_fit, minlength=k).astype(np.int64)
    all_sizes = np.bincount(labels_all, minlength=k).astype(np.int64)
    low_sizes = np.bincount(labels_all[_GLOBAL["low_mask"]], minlength=k).astype(np.int64)
    low_shares = np.divide(
        low_sizes,
        all_sizes,
        out=np.zeros(k, dtype=np.float64),
        where=all_sizes > 0,
    )

    elapsed = time.perf_counter() - started
    n_iter = int(getattr(model, "n_iter_", -1))
    max_iter = int(settings["max_iter"])

    return {
        "k": k,
        "seed": seed,
        "run_scope": request.run_scope,
        "inertia_dtw": float(model.inertia_),
        "full_sample_dtw_silhouette": silhouette,
        "fit_seconds": float(elapsed),
        "iterations": n_iter,
        "max_iter_configured": max_iter,
        "max_iter_barycenter_configured": int(settings["max_iter_barycenter"]),
        "tolerance": float(settings["tol"]),
        "reached_iteration_cap": bool(n_iter >= max_iter),
        "converged_proxy": bool(n_iter >= 0 and n_iter < max_iter),
        "labels_fit_raw": labels_fit,
        "labels_all_raw": labels_all,
        "prototypes_raw": prototypes,
        "fit_sizes_raw": fit_sizes,
        "all_sizes_raw": all_sizes,
        "low_signal_sizes_raw": low_sizes,
        "low_signal_shares_raw": low_shares,
    }


def _make_requests(config: dict[str, Any]) -> list[FitRequest]:
    broad_k = [int(value) for value in config["clustering"]["broad_k"]]
    broad_seeds = [int(value) for value in config["clustering"]["broad_seeds"]]
    focused_k = [int(value) for value in config["clustering"]["focused_k"]]
    focused_seeds = [int(value) for value in config["clustering"]["focused_seeds"]]

    requests: dict[tuple[int, int], FitRequest] = {}
    for k in broad_k:
        for seed in broad_seeds:
            requests[(k, seed)] = FitRequest(k=k, seed=seed, run_scope="broad")
    for k in focused_k:
        for seed in focused_seeds:
            scope = "broad_and_focused" if (k, seed) in requests else "focused"
            requests[(k, seed)] = FitRequest(k=k, seed=seed, run_scope=scope)

    return [requests[key] for key in sorted(requests)]


def _align_results(results: list[dict[str, Any]], reference_seed: int) -> None:
    by_k: dict[int, list[dict[str, Any]]] = {}
    for result in results:
        by_k.setdefault(int(result["k"]), []).append(result)

    for k, group in by_k.items():
        references = [result for result in group if int(result["seed"]) == reference_seed]
        ensure(len(references) == 1, f"Expected one reference seed {reference_seed} for K={k}")
        reference = references[0]
        reference_prototypes = reference["prototypes_raw"]

        for result in group:
            if int(result["seed"]) == reference_seed:
                raw_to_aligned = {raw: raw for raw in range(k)}
                matched_distances = np.zeros(k, dtype=np.float64)
            else:
                costs = cdist_dtw(
                    reference_prototypes[:, :, None],
                    result["prototypes_raw"][:, :, None],
                    n_jobs=1,
                )
                ref_indices, raw_indices = linear_sum_assignment(costs)
                ensure(len(ref_indices) == k, f"Prototype matching failed for K={k}, seed={result['seed']}")
                raw_to_aligned = {int(raw): int(ref) for ref, raw in zip(ref_indices, raw_indices)}
                matched_distances = np.full(k, np.nan, dtype=np.float64)
                for ref, raw in zip(ref_indices, raw_indices):
                    matched_distances[int(ref)] = float(costs[int(ref), int(raw)])

            labels_fit_aligned = np.array(
                [raw_to_aligned[int(label)] for label in result["labels_fit_raw"]],
                dtype=np.int16,
            )
            labels_all_aligned = np.array(
                [raw_to_aligned[int(label)] for label in result["labels_all_raw"]],
                dtype=np.int16,
            )
            prototypes_aligned = np.empty_like(result["prototypes_raw"])
            for raw, aligned in raw_to_aligned.items():
                prototypes_aligned[aligned] = result["prototypes_raw"][raw]

            result["labels_fit_aligned"] = labels_fit_aligned
            result["labels_all_aligned"] = labels_all_aligned
            result["prototypes_aligned"] = prototypes_aligned
            result["matched_prototype_dtw_to_reference"] = matched_distances
            result["fit_sizes_aligned"] = np.bincount(labels_fit_aligned, minlength=k).astype(np.int64)
            result["all_sizes_aligned"] = np.bincount(labels_all_aligned, minlength=k).astype(np.int64)
            low_mask = _GLOBAL.get("low_mask")
            if low_mask is None:
                # Parent process does not use worker globals; reconstructed later.
                result["low_signal_sizes_aligned"] = None
                result["low_signal_shares_aligned"] = None


def _parent_low_signal_metrics(
    results: list[dict[str, Any]],
    low_mask: np.ndarray,
) -> None:
    for result in results:
        k = int(result["k"])
        labels = result["labels_all_aligned"]
        all_sizes = np.bincount(labels, minlength=k).astype(np.int64)
        low_sizes = np.bincount(labels[low_mask], minlength=k).astype(np.int64)
        low_shares = np.divide(
            low_sizes,
            all_sizes,
            out=np.zeros(k, dtype=np.float64),
            where=all_sizes > 0,
        )
        result["all_sizes_aligned"] = all_sizes
        result["low_signal_sizes_aligned"] = low_sizes
        result["low_signal_shares_aligned"] = low_shares


def _pairwise_ari_table(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for k, group_iter in itertools.groupby(sorted(results, key=lambda item: (item["k"], item["seed"])), key=lambda item: item["k"]):
        group = list(group_iter)
        for left, right in itertools.combinations(group, 2):
            rows.append({
                "k": int(k),
                "left_seed": int(left["seed"]),
                "right_seed": int(right["seed"]),
                "left_run_scope": left["run_scope"],
                "right_run_scope": right["run_scope"],
                "adjusted_rand_index": float(adjusted_rand_score(left["labels_fit_raw"], right["labels_fit_raw"])),
            })
    return pd.DataFrame(rows)


def _diagnostic_frames(
    results: list[dict[str, Any]],
    household_ids: np.ndarray,
    low_mask: np.ndarray,
    reference_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    prototype_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []

    n_fit = int((~low_mask).sum())
    n_all = int(len(low_mask))

    for result in sorted(results, key=lambda item: (item["k"], item["seed"])):
        k = int(result["k"])
        fit_sizes = result["fit_sizes_aligned"]
        all_sizes = result["all_sizes_aligned"]
        low_sizes = result["low_signal_sizes_aligned"]
        low_shares = result["low_signal_shares_aligned"]

        run_rows.append({
            "k": k,
            "seed": int(result["seed"]),
            "run_scope": result["run_scope"],
            "reference_seed": reference_seed,
            "inertia_dtw": result["inertia_dtw"],
            "full_sample_dtw_silhouette": result["full_sample_dtw_silhouette"],
            "minimum_cluster_size_fit": int(fit_sizes.min()),
            "maximum_cluster_size_fit": int(fit_sizes.max()),
            "minimum_cluster_share_fit": float(fit_sizes.min() / n_fit),
            "maximum_cluster_share_fit": float(fit_sizes.max() / n_fit),
            "cluster_size_cv_fit": float(np.std(fit_sizes, ddof=0) / np.mean(fit_sizes)),
            "minimum_cluster_size_all_assigned": int(all_sizes.min()),
            "maximum_cluster_size_all_assigned": int(all_sizes.max()),
            "minimum_cluster_share_all_assigned": float(all_sizes.min() / n_all),
            "maximum_cluster_share_all_assigned": float(all_sizes.max() / n_all),
            "cluster_size_cv_all_assigned": float(np.std(all_sizes, ddof=0) / np.mean(all_sizes)),
            "maximum_low_signal_count_in_one_group": int(low_sizes.max()),
            "maximum_within_group_low_signal_share": float(low_shares.max()),
            "fit_seconds": result["fit_seconds"],
            "iterations": result["iterations"],
            "max_iter_configured": result["max_iter_configured"],
            "max_iter_barycenter_configured": result["max_iter_barycenter_configured"],
            "tolerance": result["tolerance"],
            "reached_iteration_cap": result["reached_iteration_cap"],
            "converged_proxy": result["converged_proxy"],
        })

        for cluster_index in range(k):
            cluster_rows.append({
                "k": k,
                "seed": int(result["seed"]),
                "run_scope": result["run_scope"],
                "cluster_id_aligned_to_reference": cluster_index + 1,
                "fit_eligible_n": int(fit_sizes[cluster_index]),
                "fit_eligible_share": float(fit_sizes[cluster_index] / n_fit),
                "all_assigned_n": int(all_sizes[cluster_index]),
                "all_assigned_share": float(all_sizes[cluster_index] / n_all),
                "low_signal_n": int(low_sizes[cluster_index]),
                "within_group_low_signal_share": float(low_shares[cluster_index]),
                "matched_prototype_dtw_to_reference": float(result["matched_prototype_dtw_to_reference"][cluster_index]),
            })
            for slot_index, value in enumerate(result["prototypes_aligned"][cluster_index], start=1):
                prototype_rows.append({
                    "k": k,
                    "seed": int(result["seed"]),
                    "run_scope": result["run_scope"],
                    "reference_seed": reference_seed,
                    "cluster_id_aligned_to_reference": cluster_index + 1,
                    "local_slot": slot_index,
                    "prototype_zscore": float(value),
                    "fit_eligible_n": int(fit_sizes[cluster_index]),
                    "all_assigned_n": int(all_sizes[cluster_index]),
                })

        for household_id, label, is_low in zip(household_ids, result["labels_all_aligned"], low_mask):
            assignment_rows.append({
                "household_id": str(household_id),
                "k": k,
                "seed": int(result["seed"]),
                "run_scope": result["run_scope"],
                "cluster_id_aligned_to_reference": int(label) + 1,
                "low_signal_prototype_fit_exclusion": bool(is_low),
                "prototype_fit_eligible": bool(not is_low),
            })

    return (
        pd.DataFrame(run_rows),
        pd.DataFrame(cluster_rows),
        pd.DataFrame(prototype_rows),
        pd.DataFrame(assignment_rows),
    )


def _summarise_k(
    run_df: pd.DataFrame,
    pairwise_ari: pd.DataFrame,
    cluster_df: pd.DataFrame,
    broad_k: list[int],
    broad_seeds: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    previous_inertia: float | None = None
    for k in broad_k:
        subset = run_df[(run_df["k"] == k) & (run_df["seed"].isin(broad_seeds))].copy()
        ensure(len(subset) == len(broad_seeds), f"Broad K={k} does not have all broad seeds")
        ari = pairwise_ari[
            (pairwise_ari["k"] == k)
            & (pairwise_ari["left_seed"].isin(broad_seeds))
            & (pairwise_ari["right_seed"].isin(broad_seeds))
        ]
        prototype_distances = cluster_df[
            (cluster_df["k"] == k)
            & (cluster_df["seed"].isin(broad_seeds))
            & (cluster_df["seed"] != broad_seeds[0])
        ]["matched_prototype_dtw_to_reference"]
        inertia_mean = float(subset["inertia_dtw"].mean())
        relative_reduction = None if previous_inertia is None else (previous_inertia - inertia_mean) / previous_inertia
        previous_inertia = inertia_mean
        rows.append({
            "k": k,
            "runs": len(subset),
            "inertia_mean": inertia_mean,
            "inertia_sample_std": float(subset["inertia_dtw"].std(ddof=1)),
            "relative_inertia_reduction_from_previous_k": relative_reduction,
            "silhouette_mean": float(subset["full_sample_dtw_silhouette"].mean()),
            "silhouette_sample_std": float(subset["full_sample_dtw_silhouette"].std(ddof=1)),
            "minimum_cluster_share_fit_minimum": float(subset["minimum_cluster_share_fit"].min()),
            "minimum_cluster_share_fit_mean": float(subset["minimum_cluster_share_fit"].mean()),
            "maximum_within_group_low_signal_share_maximum": float(subset["maximum_within_group_low_signal_share"].max()),
            "maximum_within_group_low_signal_share_mean": float(subset["maximum_within_group_low_signal_share"].mean()),
            "cluster_size_cv_fit_mean": float(subset["cluster_size_cv_fit"].mean()),
            "pairwise_ari_mean": float(ari["adjusted_rand_index"].mean()),
            "pairwise_ari_minimum": float(ari["adjusted_rand_index"].min()),
            "pairwise_ari_maximum": float(ari["adjusted_rand_index"].max()),
            "matched_prototype_dtw_mean": float(prototype_distances.mean()),
            "matched_prototype_dtw_maximum": float(prototype_distances.max()),
            "convergence_proxy_rate": float(subset["converged_proxy"].mean()),
            "iteration_cap_hit_count": int(subset["reached_iteration_cap"].sum()),
            "iterations_mean": float(subset["iterations"].mean()),
            "iterations_maximum": int(subset["iterations"].max()),
            "fit_seconds_mean": float(subset["fit_seconds"].mean()),
            "reported_k_selected": bool(k == 4),
            "reported_k": 4,
            "selection_status": "REPORTED_K_SELECTION_EVIDENCE",
        })
    return pd.DataFrame(rows)


def _summarise_focused(
    run_df: pd.DataFrame,
    pairwise_ari: pd.DataFrame,
    cluster_df: pd.DataFrame,
    focused_k: list[int],
    focused_seeds: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for k in focused_k:
        subset = run_df[(run_df["k"] == k) & (run_df["seed"].isin(focused_seeds))].copy()
        ensure(len(subset) == len(focused_seeds), f"Focused K={k} does not have all focused seeds")
        ari = pairwise_ari[
            (pairwise_ari["k"] == k)
            & (pairwise_ari["left_seed"].isin(focused_seeds))
            & (pairwise_ari["right_seed"].isin(focused_seeds))
        ]
        prototype_distances = cluster_df[
            (cluster_df["k"] == k)
            & (cluster_df["seed"].isin(focused_seeds))
            & (cluster_df["seed"] != focused_seeds[0])
        ]["matched_prototype_dtw_to_reference"]
        rows.append({
            "k": k,
            "independent_initialisations": len(subset),
            "inertia_mean": float(subset["inertia_dtw"].mean()),
            "inertia_sample_std": float(subset["inertia_dtw"].std(ddof=1)),
            "inertia_minimum": float(subset["inertia_dtw"].min()),
            "inertia_maximum": float(subset["inertia_dtw"].max()),
            "silhouette_mean": float(subset["full_sample_dtw_silhouette"].mean()),
            "silhouette_sample_std": float(subset["full_sample_dtw_silhouette"].std(ddof=1)),
            "minimum_cluster_share_fit_minimum": float(subset["minimum_cluster_share_fit"].min()),
            "minimum_cluster_share_fit_mean": float(subset["minimum_cluster_share_fit"].mean()),
            "maximum_within_group_low_signal_share_maximum": float(subset["maximum_within_group_low_signal_share"].max()),
            "maximum_within_group_low_signal_share_mean": float(subset["maximum_within_group_low_signal_share"].mean()),
            "pairwise_ari_mean": float(ari["adjusted_rand_index"].mean()),
            "pairwise_ari_minimum": float(ari["adjusted_rand_index"].min()),
            "pairwise_ari_maximum": float(ari["adjusted_rand_index"].max()),
            "matched_prototype_dtw_mean": float(prototype_distances.mean()),
            "matched_prototype_dtw_maximum": float(prototype_distances.max()),
            "convergence_proxy_rate": float(subset["converged_proxy"].mean()),
            "iteration_cap_hit_count": int(subset["reached_iteration_cap"].sum()),
            "iterations_mean": float(subset["iterations"].mean()),
            "iterations_maximum": int(subset["iterations"].max()),
            "fit_seconds_mean": float(subset["fit_seconds"].mean()),
            "selection_status": "INITIALISATION_SENSITIVITY_REPORTED",
        })
    return pd.DataFrame(rows)


def _plot_k_diagnostics(k_summary: pd.DataFrame, output_path: Path) -> None:
    ks = k_summary["k"].to_numpy()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    fig.suptitle("Candidate-K diagnostics for train-period LCL source profiles", fontsize=18)

    axes[0, 0].errorbar(ks, k_summary["inertia_mean"], yerr=k_summary["inertia_sample_std"], marker="o", capsize=4)
    axes[0, 0].set_title("(a) Mean DTW inertia")
    axes[0, 0].set_xlabel("Number of clusters (K)")
    axes[0, 0].set_ylabel("DTW inertia")
    for _, row in k_summary.iterrows():
        if pd.notna(row["relative_inertia_reduction_from_previous_k"]):
            axes[0, 0].annotate(
                f"−{100 * row['relative_inertia_reduction_from_previous_k']:.1f}%",
                (row["k"], row["inertia_mean"]),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=8,
            )

    axes[0, 1].errorbar(ks, k_summary["silhouette_mean"], yerr=k_summary["silhouette_sample_std"], marker="o", capsize=4)
    axes[0, 1].set_title("(b) Full-sample DTW silhouette")
    axes[0, 1].set_xlabel("Number of clusters (K)")
    axes[0, 1].set_ylabel("Silhouette")

    axes[0, 2].plot(ks, 100 * k_summary["minimum_cluster_share_fit_mean"], marker="o", label="Mean smallest-cluster share")
    axes[0, 2].plot(ks, 100 * k_summary["minimum_cluster_share_fit_minimum"], marker="o", label="Minimum over broad seeds")
    axes[0, 2].plot(ks, 100 * k_summary["maximum_within_group_low_signal_share_maximum"], marker="o", label="Highest low-signal concentration")
    axes[0, 2].set_title("(c) Balance and low-signal concentration")
    axes[0, 2].set_xlabel("Number of clusters (K)")
    axes[0, 2].set_ylabel("Percent")
    axes[0, 2].legend(fontsize=8)

    axes[1, 0].plot(ks, k_summary["pairwise_ari_mean"], marker="o", label="Mean pairwise ARI")
    axes[1, 0].plot(ks, k_summary["pairwise_ari_minimum"], marker="o", label="Minimum pairwise ARI")
    axes[1, 0].set_title("(d) Broad-seed assignment stability")
    axes[1, 0].set_xlabel("Number of clusters (K)")
    axes[1, 0].set_ylabel("Adjusted Rand Index")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(ks, k_summary["matched_prototype_dtw_mean"], marker="o", label="Mean matched-prototype DTW")
    axes[1, 1].plot(ks, k_summary["matched_prototype_dtw_maximum"], marker="o", label="Maximum matched-prototype DTW")
    axes[1, 1].set_title("(e) Prototype stability relative to seed 42")
    axes[1, 1].set_xlabel("Number of clusters (K)")
    axes[1, 1].set_ylabel("DTW distance")
    axes[1, 1].legend(fontsize=8)

    axes[1, 2].plot(ks, 100 * k_summary["convergence_proxy_rate"], marker="o", label="Convergence proxy rate (%)")
    axes[1, 2].set_title("(f) Iteration-cap evidence")
    axes[1, 2].set_xlabel("Number of clusters (K)")
    axes[1, 2].set_ylabel("Percent")
    twin = axes[1, 2].twinx()
    twin.plot(ks, k_summary["iterations_mean"], marker="s", linestyle="--", label="Mean iterations")
    twin.set_ylabel("Iterations")
    lines, labels = axes[1, 2].get_legend_handles_labels()
    lines2, labels2 = twin.get_legend_handles_labels()
    axes[1, 2].legend(lines + lines2, labels + labels2, fontsize=8, loc="best")

    for axis in axes.flat:
        axis.axvline(4, linestyle="--", linewidth=1)
        axis.grid(alpha=0.25)
        axis.set_xticks(ks)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_focused_sensitivity(run_df: pd.DataFrame, focused_k: list[int], focused_seeds: list[int], output_path: Path) -> None:
    subset = run_df[(run_df["k"].isin(focused_k)) & (run_df["seed"].isin(focused_seeds))].copy()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle("Independent-initialisation sensitivity for K=4, K=5 and K=6", fontsize=17)
    metrics = [
        ("inertia_dtw", "DTW inertia"),
        ("full_sample_dtw_silhouette", "Full-sample DTW silhouette"),
        ("minimum_cluster_share_fit", "Smallest-cluster share"),
        ("maximum_within_group_low_signal_share", "Highest within-group low-signal share"),
    ]
    for axis, (column, title) in zip(axes.flat, metrics):
        values = [subset.loc[subset["k"] == k, column].to_numpy() for k in focused_k]
        axis.boxplot(values, tick_labels=[f"K={k}" for k in focused_k], showmeans=True)
        axis.set_title(title)
        axis.grid(alpha=0.25)
        if "share" in column:
            axis.yaxis.set_major_formatter(lambda value, position: f"{100 * value:.0f}%")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_prototypes(prototype_df: pd.DataFrame, reference_seed: int, ks: list[int], output_path: Path) -> None:
    fig, axes = plt.subplots(len(ks), 1, figsize=(16, 3.4 * len(ks)), sharex=True, constrained_layout=True)
    if len(ks) == 1:
        axes = [axes]
    fig.suptitle("Reference-seed candidate source prototypes on the London local half-hour clock", fontsize=17)
    for axis, k in zip(axes, ks):
        subset = prototype_df[(prototype_df["k"] == k) & (prototype_df["seed"] == reference_seed)]
        ensure(len(subset) == k * 48, f"Missing reference-seed prototype rows for K={k}")
        for cluster_id in range(1, k + 1):
            group = subset[subset["cluster_id_aligned_to_reference"] == cluster_id].sort_values("local_slot")
            n = int(group["all_assigned_n"].iloc[0])
            axis.plot(group["local_slot"], group["prototype_zscore"], marker="o", markersize=2.5, label=f"C{cluster_id} (all assigned n={n})")
        axis.axhline(0, linewidth=0.8)
        axis.set_title(f"K={k}")
        axis.set_ylabel("DTW barycentre z-score")
        axis.grid(alpha=0.25)
        axis.legend(ncol=min(k, 4), fontsize=8)
    axes[-1].set_xlabel("London local half-hour slot")
    axes[-1].set_xticks([1, 7, 13, 19, 25, 31, 37, 43, 48])
    axes[-1].set_xticklabels(["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00", "23:30"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_selection_summary(
    path: Path,
    k_summary: pd.DataFrame,
    focused_summary: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    """Write the reported K-selection evidence summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    k_text = k_summary.to_string(index=False)
    focused_text = focused_summary.to_string(index=False)
    text = (
        "# Source profile K-selection evidence\n\n"
        "Reported K: **4**\n\n"
        "The selection evidence uses source-dataset training profiles only. "
        "Target-dataset assignments, forecasting performance and survey variables are not used.\n\n"
        "## Candidate K diagnostics\n\n"
        f"```text\n{k_text}\n```\n\n"
        "## Initialisation sensitivity for K = 4, 5 and 6\n\n"
        f"```text\n{focused_text}\n```\n"
    )
    path.write_text(text, encoding="utf-8")


def run_analysis(config_path: Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    project_root = Path(config["project_root"]).resolve()
    paths = config["paths"]
    output_tables = resolve_path(project_root, paths["output_tables"])
    output_metadata = resolve_path(project_root, paths["output_metadata"])
    decision_path = resolve_path(project_root, paths["decision_document"])

    for output_dir in (output_tables, output_metadata):
        output_dir.mkdir(parents=True, exist_ok=True)

    zscore_path = resolve_path(project_root, paths["profiles_zscore"])
    low_signal_path = resolve_path(project_root, paths["low_signal_audit"])
    distance_path = resolve_path(project_root, paths["distance_matrix"])

    profiles = pd.read_csv(zscore_path)
    slots = slot_columns(profiles.columns.tolist())
    low_signal = pd.read_csv(low_signal_path)
    ensure(profiles["household_id"].astype(str).tolist() == low_signal["household_id"].astype(str).tolist(), "Profile and low-signal ID order mismatch")
    household_ids = profiles["household_id"].astype(str).to_numpy()
    low_mask = _extract_low_signal_mask(low_signal)

    requests = _make_requests(config)
    clustering_settings = config["clustering"]
    allocated_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    workers = min(int(clustering_settings["max_parallel_workers"]), max(1, allocated_cpus), len(requests))

    print("=" * 100, flush=True)
    print("PROFILE CLUSTERING FINAL K-SELECTION FITS", flush=True)
    print("=" * 100, flush=True)
    print(f"Runs                  : {len(requests)}", flush=True)
    print(f"Parallel workers      : {workers}", flush=True)
    print(f"Broad K               : {clustering_settings['broad_k']}", flush=True)
    print(f"Focused K             : {clustering_settings['focused_k']}", flush=True)
    print(f"Focused initialisations: {len(clustering_settings['focused_seeds'])}", flush=True)

    results: list[dict[str, Any]] = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_worker_init,
        initargs=(
            str(zscore_path),
            str(low_signal_path),
            str(distance_path),
            int(config["expected"]["source_households"]),
            int(config["expected"]["fit_eligible_households"]),
        ),
    ) as executor:
        futures = {executor.submit(_fit_one, request, clustering_settings): request for request in requests}
        for future in as_completed(futures):
            request = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                for pending in futures:
                    pending.cancel()
                raise ProfileClusteringFinalKError(f"Fit failed for K={request.k}, seed={request.seed}: {exc}") from exc
            results.append(result)
            print(
                f"COMPLETE K={result['k']} seed={result['seed']} "
                f"inertia={result['inertia_dtw']:.6f} "
                f"silhouette={result['full_sample_dtw_silhouette']:.6f} "
                f"iterations={result['iterations']} "
                f"seconds={result['fit_seconds']:.1f}",
                flush=True,
            )

    reference_seed = int(clustering_settings["reference_seed"])
    _align_results(results, reference_seed)
    _parent_low_signal_metrics(results, low_mask)

    run_df, cluster_df, prototype_df, assignment_df = _diagnostic_frames(
        results,
        household_ids,
        low_mask,
        reference_seed,
    )
    pairwise_ari = _pairwise_ari_table(results)

    broad_k = [int(value) for value in clustering_settings["broad_k"]]
    broad_seeds = [int(value) for value in clustering_settings["broad_seeds"]]
    focused_k = [int(value) for value in clustering_settings["focused_k"]]
    focused_seeds = [int(value) for value in clustering_settings["focused_seeds"]]

    k_summary = _summarise_k(run_df, pairwise_ari, cluster_df, broad_k, broad_seeds)
    focused_summary = _summarise_focused(run_df, pairwise_ari, cluster_df, focused_k, focused_seeds)

    run_df.to_csv(output_tables / "profile_clustering_final_k_all_run_diagnostics.csv", index=False)
    cluster_df.to_csv(output_tables / "profile_clustering_final_k_cluster_composition_all_runs.csv", index=False)
    prototype_df.to_csv(output_tables / "profile_clustering_final_k_candidate_prototypes_all_runs.csv", index=False)
    assignment_df.to_parquet(output_tables / "profile_clustering_final_k_all_source_assignments_all_runs.parquet", index=False)
    pairwise_ari.to_csv(output_tables / "profile_clustering_final_k_pairwise_ari_all_runs.csv", index=False)
    k_summary.to_csv(output_tables / "profile_clustering_candidate_k_selection_table.csv", index=False)
    focused_summary.to_csv(output_tables / "profile_clustering_initialisation_sensitivity_k4_k5_k6.csv", index=False)


    _write_selection_summary(decision_path, k_summary, focused_summary, config)

    metadata = {
        "status": "COMPLETE",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "Profile clustering focused final K selection",
        "project_root": str(project_root),
        "source_households": int(config["expected"]["source_households"]),
        "fit_eligible_households": int(config["expected"]["fit_eligible_households"]),
        "low_signal_households": int(config["expected"]["low_signal_households"]),
        "broad_k": broad_k,
        "broad_seeds": broad_seeds,
        "focused_k": focused_k,
        "focused_seeds": focused_seeds,
        "total_independent_fits": len(results),
        "reference_seed": reference_seed,
        "max_iter": int(clustering_settings["max_iter"]),
        "max_iter_barycenter": int(clustering_settings["max_iter_barycenter"]),
        "tol": float(clustering_settings["tol"]),
        "parallel_workers": workers,
        "final_k_decision": 4,
        "cer_rows_used": 0,
        "forecasting_metrics_used": 0,
        "survey_variables_used": 0,
        "time_basis": {
            "canonical_time_map": str(resolve_path(project_root, paths["time_map"])),
            "profile_clock": "London local_slot 1-48",
            "raw_gmt_hour_used_as_local_time": False,
            "excluded_local_dates": config["time_rules"]["excluded_local_dates"],
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "outputs": {
            "tables": str(output_tables),
            "metadata": str(output_metadata),
            "decision_document": str(decision_path),
        },
    }
    write_json(output_metadata / "profile_clustering_final_k_selection_status.json", metadata)

    summary_lines = [
        "# Source profile K-selection summary",
        "",
        "- Status: **COMPLETE**",
        f"- Broad K scan: **{broad_k}**",
        f"- Broad diagnostic seeds: **{broad_seeds}**",
        f"- Focused initialisation sensitivity: **K={focused_k}, {len(focused_seeds)} independent initialisations each**",
        f"- Total independent fits: **{len(results)}**",
        "- Reported K: **4**",
        "- CER rows used: **0**",
        "- Forecasting metrics used: **0**",
        "- Survey variables used: **0**",
        "- Profile time basis: **canonical London local half-hour slot; raw GMT hour not used as local time**",
        "",
        "The retained diagnostics document the reported K = 4 source-profile selection.",
        "",
    ]
    (output_metadata / "profile_clustering_final_k_selection_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    metadata = run_analysis(args.config)
    print("=" * 100)
    print("PROFILE CLUSTERING FOCUSED FINAL K SELECTION COMPLETE")
    print("=" * 100)
    print(f"Status          : {metadata['status']}")
    print(f"Independent fits: {metadata['total_independent_fits']}")
    print("Reported K      : 4")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProfileClusteringFinalKError as exc:
        print(f"PROFILE CLUSTERING FINAL K ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
