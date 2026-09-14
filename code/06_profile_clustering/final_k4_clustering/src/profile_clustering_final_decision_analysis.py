from __future__ import annotations

import argparse
import itertools
import math
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from profile_clustering_final_decision_common import (
    ProfileClusteringFinalDecisionError, ensure, load_json, local_time_from_slot,
    peak_band_2h, resolve, sha256_file, write_json,
)

PALETTE = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00']


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    previous = np.full(len(b) + 1, np.inf, dtype=float)
    previous[0] = 0.0
    for x in a:
        current = np.full(len(b) + 1, np.inf, dtype=float)
        for j, y in enumerate(b, start=1):
            cost = (x - y) ** 2
            current[j] = cost + min(current[j - 1], previous[j], previous[j - 1])
        previous = current
    return float(math.sqrt(previous[-1]))


def read_assignments(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == '.parquet':
        return pd.read_parquet(path)
    return pd.read_csv(path)


def write_assignments(frame: pd.DataFrame, path: Path, parquet_enabled: bool) -> Path:
    if parquet_enabled:
        frame.to_parquet(path, index=False)
        return path
    csv_path = path.with_suffix('.csv')
    frame.to_csv(csv_path, index=False)
    return csv_path


def best_injective_mapping(contingency: np.ndarray) -> tuple[dict[int, int], int, int]:
    # Map four K=4 rows to four distinct K=5 columns, maximising overlap.
    ensure(contingency.shape == (4, 5), f'Unexpected contingency shape: {contingency.shape}')
    best_score = -1
    best_mapping: dict[int, int] | None = None
    for cols in itertools.permutations(range(5), 4):
        score = sum(int(contingency[row, col]) for row, col in enumerate(cols))
        if score > best_score:
            best_score = score
            best_mapping = {row: col for row, col in enumerate(cols)}
    ensure(best_mapping is not None, 'Could not determine K=4 to K=5 mapping')
    extra = next(iter(set(range(5)) - set(best_mapping.values())))
    return best_mapping, int(extra), int(best_score)


def prototype_vector(prototypes: pd.DataFrame, k: int, seed: int, cluster_id: int) -> np.ndarray:
    frame = prototypes[
        (prototypes['k'] == k)
        & (prototypes['seed'] == seed)
        & (prototypes['cluster_id_aligned_to_reference'] == cluster_id)
    ].sort_values('local_slot')
    ensure(len(frame) == 48, f'Missing prototype K={k}, seed={seed}, cluster={cluster_id}')
    return frame['prototype_zscore'].to_numpy(dtype=float)


def candidate_run(run_df: pd.DataFrame, k: int, seeds: list[int], rule: dict[str, Any]) -> pd.Series:
    frame = run_df[(run_df['k'] == k) & (run_df['seed'].isin(seeds))].copy()
    ensure(len(frame) == len(seeds), f'K={k} candidate run count mismatch')
    median_silhouette = float(frame['full_sample_dtw_silhouette'].median())
    eligible = frame[
        frame['converged_proxy'].astype(bool)
        & (~frame['reached_iteration_cap'].astype(bool))
        & (frame['minimum_cluster_share_all_assigned'] >= float(rule['minimum_all_assigned_share']))
        & (frame['maximum_within_group_low_signal_share'] <= float(rule['maximum_within_group_low_signal_share']))
    ].copy()
    if bool(rule['require_silhouette_at_or_above_candidate_median']):
        eligible = eligible[eligible['full_sample_dtw_silhouette'] >= median_silhouette]
    ensure(len(eligible) > 0, f'No diagnostically acceptable run for K={k}')
    return eligible.sort_values(['inertia_dtw', 'seed']).iloc[0]


def transition_for_seed(
    seed: int,
    assignments: pd.DataFrame,
    prototypes: pd.DataFrame,
    run_df: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    base = assignments[(assignments['seed'] == seed) & (assignments['k'].isin([4, 5]))].copy()
    ensure(len(base) == 2 * base['household_id'].nunique(), f'K4/K5 assignment count mismatch for seed {seed}')
    left = base[base['k'] == 4][['household_id', 'cluster_id_aligned_to_reference', 'prototype_fit_eligible']].rename(
        columns={'cluster_id_aligned_to_reference': 'k4_cluster', 'prototype_fit_eligible': 'fit_k4'}
    )
    right = base[base['k'] == 5][['household_id', 'cluster_id_aligned_to_reference', 'prototype_fit_eligible']].rename(
        columns={'cluster_id_aligned_to_reference': 'k5_cluster', 'prototype_fit_eligible': 'fit_k5'}
    )
    joined = left.merge(right, on='household_id', how='inner', validate='one_to_one')
    ensure((joined['fit_k4'] == joined['fit_k5']).all(), f'Fit flag mismatch for seed {seed}')
    fit = joined[joined['fit_k4'].astype(bool)].copy()
    contingency = pd.crosstab(fit['k4_cluster'], fit['k5_cluster']).reindex(index=[1,2,3,4], columns=[1,2,3,4,5], fill_value=0)
    mapping0, extra0, overlap = best_injective_mapping(contingency.to_numpy(dtype=int))
    mapping = {k4 + 1: k5 + 1 for k4, k5 in mapping0.items()}
    extra = extra0 + 1
    extra_column = contingency[extra]
    parent = int(extra_column.idxmax())
    retained = int(mapping[parent])
    extra_n = int(extra_column.sum())
    parent_n = int(contingency.loc[parent].sum())
    origin_from_parent = int(contingency.loc[parent, extra])
    retained_from_parent = int(contingency.loc[parent, retained])
    origin_purity = origin_from_parent / extra_n
    parent_split_share = origin_from_parent / parent_n
    two_child_coverage = (origin_from_parent + retained_from_parent) / parent_n
    origin_probs = extra_column.to_numpy(dtype=float) / extra_n
    positive = origin_probs[origin_probs > 0]
    origin_entropy = float(-(positive * np.log(positive)).sum() / np.log(4))

    p4 = prototype_vector(prototypes, 4, seed, parent)
    p5_extra = prototype_vector(prototypes, 5, seed, extra)
    p5_retained = prototype_vector(prototypes, 5, seed, retained)
    all_k5 = [prototype_vector(prototypes, 5, seed, cid) for cid in range(1, 6)]
    pairwise = [dtw_distance(all_k5[i], all_k5[j]) for i in range(5) for j in range(i + 1, 5)]
    extra_sibling_dtw = dtw_distance(p5_extra, p5_retained)
    distinctness_ratio = extra_sibling_dtw / float(np.median(pairwise))

    peak_parent = int(np.argmax(p4) + 1)
    peak_extra = int(np.argmax(p5_extra) + 1)
    peak_retained = int(np.argmax(p5_retained) + 1)
    run4 = run_df[(run_df['k'] == 4) & (run_df['seed'] == seed)].iloc[0]
    run5 = run_df[(run_df['k'] == 5) & (run_df['seed'] == seed)].iloc[0]

    transition_long = contingency.reset_index().melt(id_vars='k4_cluster', var_name='k5_cluster', value_name='fit_n')
    inverse_mapping = {k5: k4 for k4, k5 in mapping.items()}
    transition_long['mapped_k5_role'] = transition_long['k5_cluster'].map(
        lambda value: 'extra_group' if int(value) == extra else f'retained_child_of_k4_pg{inverse_mapping[int(value)]}'
    )
    transition_long['seed'] = seed
    transition_long['fit_row_share'] = transition_long.apply(lambda row: row['fit_n'] / contingency.loc[int(row['k4_cluster'])].sum(), axis=1)
    transition_long['fit_column_share'] = transition_long.apply(lambda row: row['fit_n'] / contingency[int(row['k5_cluster'])].sum(), axis=1)

    summary = {
        'seed': seed,
        'k4_to_k5_overlap_n': overlap,
        'k4_to_k5_overlap_share': overlap / len(fit),
        'extra_k5_cluster_aligned': extra,
        'dominant_parent_k4_cluster_aligned': parent,
        'retained_sibling_k5_cluster_aligned': retained,
        'extra_fit_n': extra_n,
        'extra_fit_share': extra_n / len(fit),
        'extra_origin_from_parent_n': origin_from_parent,
        'extra_origin_purity': origin_purity,
        'parent_split_share_to_extra': parent_split_share,
        'parent_two_child_coverage': two_child_coverage,
        'extra_origin_normalised_entropy': origin_entropy,
        'parent_peak_slot_k4': peak_parent,
        'parent_peak_time_k4': local_time_from_slot(peak_parent),
        'retained_peak_slot_k5': peak_retained,
        'retained_peak_time_k5': local_time_from_slot(peak_retained),
        'extra_peak_slot_k5': peak_extra,
        'extra_peak_time_k5': local_time_from_slot(peak_extra),
        'extra_peak_band_2h': peak_band_2h(peak_extra),
        'parent_to_extra_dtw': dtw_distance(p4, p5_extra),
        'parent_to_retained_dtw': dtw_distance(p4, p5_retained),
        'extra_to_retained_dtw': extra_sibling_dtw,
        'extra_distinctness_ratio_to_k5_median_pairwise': distinctness_ratio,
        'k4_inertia': float(run4['inertia_dtw']),
        'k5_inertia': float(run5['inertia_dtw']),
        'k5_minus_k4_inertia': float(run5['inertia_dtw'] - run4['inertia_dtw']),
        'k4_silhouette': float(run4['full_sample_dtw_silhouette']),
        'k5_silhouette': float(run5['full_sample_dtw_silhouette']),
        'k5_minus_k4_silhouette': float(run5['full_sample_dtw_silhouette'] - run4['full_sample_dtw_silhouette']),
        'k4_minimum_group_share': float(run4['minimum_cluster_share_all_assigned']),
        'k5_minimum_group_share': float(run5['minimum_cluster_share_all_assigned']),
        'k5_minus_k4_minimum_group_share': float(run5['minimum_cluster_share_all_assigned'] - run4['minimum_cluster_share_all_assigned']),
        'k4_iterations': int(run4['iterations']),
        'k5_iterations': int(run5['iterations']),
    }
    return summary, transition_long


def create_decision_summary(transitions: pd.DataFrame, run_df: pd.DataFrame, seeds: list[int], rule: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    parent_counts = Counter(transitions['dominant_parent_k4_cluster_aligned'].astype(int))
    parent_mode, parent_mode_n = parent_counts.most_common(1)[0]
    band_counts = Counter(transitions['extra_peak_band_2h'].astype(str))
    band_mode, band_mode_n = band_counts.most_common(1)[0]
    k4 = run_df[(run_df['k'] == 4) & (run_df['seed'].isin(seeds))].sort_values('seed')
    k5 = run_df[(run_df['k'] == 5) & (run_df['seed'].isin(seeds))].sort_values('seed')
    merged = k4.merge(k5, on='seed', suffixes=('_k4', '_k5'), validate='one_to_one')
    silhouette_loss = float((merged['full_sample_dtw_silhouette_k4'] - merged['full_sample_dtw_silhouette_k5']).mean())
    min_share_loss = float((merged['minimum_cluster_share_all_assigned_k4'] - merged['minimum_cluster_share_all_assigned_k5']).mean())
    structural = {
        'parent_mode_k4_cluster': int(parent_mode),
        'parent_mode_runs': int(parent_mode_n),
        'parent_mode_share': parent_mode_n / len(seeds),
        'peak_band_mode': band_mode,
        'peak_band_mode_runs': int(band_mode_n),
        'peak_band_mode_share': band_mode_n / len(seeds),
        'median_origin_purity': float(transitions['extra_origin_purity'].median()),
        'median_extra_group_share': float(transitions['extra_fit_share'].median()),
        'median_two_child_parent_coverage': float(transitions['parent_two_child_coverage'].median()),
        'median_distinctness_ratio': float(transitions['extra_distinctness_ratio_to_k5_median_pairwise'].median()),
        'mean_silhouette_loss_k4_minus_k5': silhouette_loss,
        'mean_minimum_share_loss_k4_minus_k5': min_share_loss,
        'k4_silhouette_wins': int((merged['full_sample_dtw_silhouette_k4'] > merged['full_sample_dtw_silhouette_k5']).sum()),
        'k4_minimum_share_wins': int((merged['minimum_cluster_share_all_assigned_k4'] > merged['minimum_cluster_share_all_assigned_k5']).sum()),
        'k5_inertia_wins': int((merged['inertia_dtw_k5'] < merged['inertia_dtw_k4']).sum()),
        'minimum_k5_group_share_observed': float(merged['minimum_cluster_share_all_assigned_k5'].min()),
    }
    gates = {
        'stable_parent_gate': structural['parent_mode_share'] >= float(rule['parent_mode_share_minimum']),
        'origin_purity_gate': structural['median_origin_purity'] >= float(rule['median_origin_purity_minimum']),
        'extra_size_gate': structural['median_extra_group_share'] >= float(rule['median_extra_group_share_minimum']),
        'peak_band_gate': structural['peak_band_mode_share'] >= float(rule['peak_band_mode_share_minimum']),
        'two_child_coverage_gate': structural['median_two_child_parent_coverage'] >= float(rule['median_two_child_parent_coverage_minimum']),
        'distinctness_gate': structural['median_distinctness_ratio'] >= float(rule['median_distinctness_ratio_minimum']),
        'silhouette_tradeoff_gate': structural['mean_silhouette_loss_k4_minus_k5'] <= float(rule['mean_silhouette_loss_maximum']),
        'minimum_share_tradeoff_gate': structural['mean_minimum_share_loss_k4_minus_k5'] <= float(rule['mean_minimum_share_loss_maximum']),
        'k5_floor_gate': structural['minimum_k5_group_share_observed'] >= float(rule['minimum_k5_group_share_floor']),
    }
    recommend_k5 = all(gates.values())
    recommendation = 5 if recommend_k5 else 4
    rows = []
    for name, value in structural.items():
        rows.append({'evidence': name, 'value': value})
    for name, value in gates.items():
        rows.append({'evidence': name, 'value': bool(value)})
    rows.extend([
        {'evidence': 'recommended_k', 'value': recommendation},
        {'evidence': 'selection_status', 'value': 'REPORTED_K_CONFIRMED'},
    ])
    payload = {'structural_summary': structural, 'decision_gates': gates, 'recommended_k': recommendation, 'reported_k': 4, 'status': 'COMPLETE'}
    return pd.DataFrame(rows), payload


def materialise_candidate(k: int, selected_seed: int, assignments: pd.DataFrame, prototypes: pd.DataFrame, composition: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, int]]:
    proto = prototypes[(prototypes['k'] == k) & (prototypes['seed'] == selected_seed)].copy()
    ensure(len(proto) == k * 48, f'Candidate prototype row count mismatch for K={k}')
    peaks = proto.loc[proto.groupby('cluster_id_aligned_to_reference')['prototype_zscore'].idxmax(), ['cluster_id_aligned_to_reference', 'local_slot']]
    order = peaks.sort_values(['local_slot', 'cluster_id_aligned_to_reference'])['cluster_id_aligned_to_reference'].astype(int).tolist()
    mapping = {old: new for new, old in enumerate(order, start=1)}
    proto['profile_group'] = proto['cluster_id_aligned_to_reference'].map(mapping)
    proto['local_time'] = proto['local_slot'].map(local_time_from_slot)
    proto = proto.sort_values(['profile_group', 'local_slot'])
    assn = assignments[(assignments['k'] == k) & (assignments['seed'] == selected_seed)].copy()
    assn['profile_group'] = assn['cluster_id_aligned_to_reference'].map(mapping)
    assn = assn.sort_values('household_id')
    comp = composition[(composition['k'] == k) & (composition['seed'] == selected_seed)].copy()
    comp['profile_group'] = comp['cluster_id_aligned_to_reference'].map(mapping)
    peak_map = {mapping[int(row.cluster_id_aligned_to_reference)]: int(row.local_slot) for row in peaks.itertuples()}
    comp['peak_slot'] = comp['profile_group'].map(peak_map)
    comp['peak_time'] = comp['peak_slot'].map(local_time_from_slot)
    comp = comp.sort_values('profile_group')
    return assn, proto, mapping


def plot_candidate_k(k_table: pd.DataFrame, selected_k: int, path_png: Path, path_pdf: Path) -> None:
    ks = k_table['k'].to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), constrained_layout=True)
    fig.suptitle('Candidate-K evidence for source-profile clustering', fontsize=15)
    panels = [
        ('inertia_mean', 'DTW inertia', 'Lower is better'),
        ('silhouette_mean', 'Full-sample DTW silhouette', 'Higher is better'),
        ('minimum_cluster_share_fit_minimum', 'Worst-seed minimum group share', 'Higher is better'),
        ('pairwise_ari_mean', 'Mean pairwise assignment stability (ARI)', 'Higher is better'),
    ]
    for ax, (column, title, note) in zip(axes.flat, panels):
        values = k_table[column].to_numpy(dtype=float)
        ax.plot(ks, values, linewidth=1.8, marker='o', markersize=4.5, color='#333333')
        ax.axvspan(3.75, 5.25, color='#D9D9D9', alpha=0.35, zorder=0)
        ax.axvline(selected_k, color='#0072B2', linewidth=1.5, linestyle='--')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Number of groups (K)')
        ax.text(0.02, 0.03, note, transform=ax.transAxes, fontsize=8, color='#555555')
        ax.grid(axis='y', alpha=0.22)
        ax.set_xticks(ks)
        if 'share' in column:
            ax.yaxis.set_major_formatter(lambda value, pos: f'{100*value:.0f}%')
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=400, bbox_inches='tight')
    fig.savefig(path_pdf, bbox_inches='tight')
    plt.close(fig)


def plot_paired_decision(transitions: pd.DataFrame, selected_k: int, path_png: Path, path_pdf: Path) -> None:
    seeds = transitions['seed'].astype(int).to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), constrained_layout=True)
    fig.suptitle('K=4 versus K=5: paired initialisation and structural evidence', fontsize=15)
    for ax, left, right, title, percent in [
        (axes[0,0], 'k4_silhouette', 'k5_silhouette', 'Full-sample silhouette', False),
        (axes[0,1], 'k4_minimum_group_share', 'k5_minimum_group_share', 'Minimum group share', True),
    ]:
        for _, row in transitions.iterrows():
            ax.plot([4,5], [row[left], row[right]], color='#B0B0B0', linewidth=0.7, alpha=0.6)
        means = [transitions[left].mean(), transitions[right].mean()]
        ax.plot([4,5], means, color='#111111', linewidth=2.2, marker='o', markersize=6)
        ax.set_xticks([4,5], ['K=4','K=5'])
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.22)
        if percent:
            ax.yaxis.set_major_formatter(lambda value, pos: f'{100*value:.0f}%')
    axes[1,0].boxplot(
        [transitions['extra_origin_purity'], transitions['extra_fit_share'], transitions['parent_two_child_coverage']],
        tick_labels=['Origin purity', 'Extra-group share', 'Two-child coverage'], showmeans=True
    )
    axes[1,0].set_title('Structure of the additional K=5 group')
    axes[1,0].yaxis.set_major_formatter(lambda value, pos: f'{100*value:.0f}%')
    axes[1,0].grid(axis='y', alpha=0.22)
    bands = transitions['extra_peak_band_2h'].value_counts().sort_index()
    axes[1,1].barh(bands.index.astype(str), bands.values, color='#666666')
    axes[1,1].set_title('Peak-time band of the additional K=5 group')
    axes[1,1].set_xlabel('Number of matched initialisations')
    axes[1,1].grid(axis='x', alpha=0.22)
    fig.text(0.5, 0.005, 'Reported source-profile selection: K=4.', ha='center', fontsize=9)
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=400, bbox_inches='tight')
    fig.savefig(path_pdf, bbox_inches='tight')
    plt.close(fig)


def plot_prototypes(proto: pd.DataFrame, summary: pd.DataFrame, title: str, path_png: Path, path_pdf: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.8, 5.8), constrained_layout=True)
    for group in sorted(proto['profile_group'].unique()):
        frame = proto[proto['profile_group'] == group].sort_values('local_slot')
        row = summary[summary['profile_group'] == group].iloc[0]
        label = f'PG{group} — peak {row.peak_time} (n={int(row.all_assigned_n)})'
        ax.plot(frame['local_slot'], frame['prototype_zscore'], linewidth=2.15, color=PALETTE[(int(group)-1)%len(PALETTE)], label=label)
    ax.axhline(0, color='#777777', linewidth=0.9, linestyle='--', zorder=0)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel('Standardised daily load shape (z-score)')
    ax.set_xlabel('London local time')
    ticks = [1,7,13,19,25,31,37,43,48]
    ax.set_xticks(ticks, [local_time_from_slot(slot) for slot in ticks])
    ax.grid(axis='y', alpha=0.22)
    ax.legend(frameon=False, ncol=2, loc='upper left')
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=400, bbox_inches='tight')
    fig.savefig(path_pdf, bbox_inches='tight')
    plt.close(fig)


def plot_transition_heatmap(matrix: pd.DataFrame, path_png: Path, path_pdf: Path) -> None:
    roles=[f'retained_child_of_k4_pg{i}' for i in range(1,5)]+['extra_group']
    frame=matrix.set_index('k4_cluster').reindex(index=[1,2,3,4],columns=roles)
    values=frame.to_numpy(dtype=float)
    fig,ax=plt.subplots(figsize=(9.2,5.2),constrained_layout=True)
    im=ax.imshow(values,aspect='auto',vmin=0,vmax=1,cmap='Blues')
    ax.set_xticks(range(5),['Retained PG1','Retained PG2','Retained PG3','Retained PG4','Additional group'],rotation=20,ha='right')
    ax.set_yticks(range(4),['K=4 PG1','K=4 PG2','K=4 PG3','K=4 PG4'])
    ax.set_title('Mean K=4 to K=5 membership transitions across 20 paired initialisations')
    for i in range(4):
        for j in range(5):
            val=values[i,j]
            ax.text(j,i,f'{100*val:.1f}%',ha='center',va='center',color='white' if val>0.55 else 'black',fontsize=9)
    cbar=fig.colorbar(im,ax=ax); cbar.set_label('Mean row share')
    path_png.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path_png,dpi=400,bbox_inches='tight'); fig.savefig(path_pdf,bbox_inches='tight'); plt.close(fig)


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns=[str(c) for c in df.columns]
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for row in df.itertuples(index=False,name=None):
        values=[]
        for value in row:
            if pd.isna(value): text=''
            elif isinstance(value,float): text=f'{value:.6g}'
            else: text=str(value)
            values.append(text.replace('|','\\|'))
        lines.append('| '+' | '.join(values)+' |')
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run(config_path: Path) -> dict[str, Any]:
    cfg = load_json(config_path)
    root = Path(cfg['project_root']).resolve()
    p = cfg['paths']
    out_t = resolve(root, p['output_tables']); out_m = resolve(root, p['output_metadata'])
    for d in (out_t, out_m): d.mkdir(parents=True, exist_ok=True)
    parquet_enabled = bool(cfg.get('write_parquet_outputs', True))
    run_df = pd.read_csv(resolve(root, p['run_diagnostics']))
    k_table = pd.read_csv(resolve(root, p['candidate_k_table']))
    focused = pd.read_csv(resolve(root, p['focused_summary']))
    composition = pd.read_csv(resolve(root, p['cluster_composition']))
    prototypes = pd.read_csv(resolve(root, p['candidate_prototypes']))
    assignments = read_assignments(resolve(root, p['all_assignments']))
    seeds = [int(x) for x in cfg['expected']['focused_seeds']]

    audit_rows = []
    for k in [4,5]:
        frame = run_df[(run_df['k']==k)&(run_df['seed'].isin(seeds))].sort_values('seed')
        ensure(len(frame)==20, f'Expected 20 runs for K={k}')
        ensure(frame['seed'].astype(int).tolist()==sorted(seeds), f'Seed set mismatch K={k}')
        ensure((frame['max_iter_configured']==100).all(), 'max_iter mismatch')
        ensure((frame['max_iter_barycenter_configured']==100).all(), 'max_iter_barycenter mismatch')
        ensure(np.allclose(frame['tolerance'], 1e-6), 'tolerance mismatch')
        ensure((~frame['reached_iteration_cap'].astype(bool)).all(), 'Iteration cap hit detected')
        ensure(frame['converged_proxy'].astype(bool).all(), 'Convergence proxy failure')
        for row in frame.itertuples():
            audit_rows.append({
                'k': k, 'seed': int(row.seed), 'independent_initialisation': True,
                'n_init_per_run': 1, 'metric': 'dtw', 'initialisation': 'k-means++',
                'max_iter': int(row.max_iter_configured), 'max_iter_barycenter': int(row.max_iter_barycenter_configured),
                'tolerance': float(row.tolerance), 'iterations_observed': int(row.iterations),
                'iteration_cap_hit': bool(row.reached_iteration_cap), 'convergence_proxy': bool(row.converged_proxy),
                'inertia_dtw': float(row.inertia_dtw), 'full_sample_dtw_silhouette': float(row.full_sample_dtw_silhouette),
                'minimum_group_share_all_assigned': float(row.minimum_cluster_share_all_assigned),
            })
    audit_df = pd.DataFrame(audit_rows).sort_values(['seed','k'])
    audit_df.to_csv(out_t/'profile_clustering_k45_iteration_initialisation_audit.csv', index=False)

    transitions=[]; transition_cells=[]
    for seed in seeds:
        s, cells = transition_for_seed(seed, assignments, prototypes, run_df)
        transitions.append(s); transition_cells.append(cells)
    transition_df = pd.DataFrame(transitions).sort_values('seed')
    transition_long = pd.concat(transition_cells, ignore_index=True)
    transition_df.to_csv(out_t/'profile_clustering_k45_transition_by_seed.csv', index=False)
    transition_long.to_csv(out_t/'profile_clustering_k45_transition_cells.csv', index=False)
    role_order=[f'retained_child_of_k4_pg{i}' for i in range(1,5)]+['extra_group']
    mapped=(transition_long.groupby(['seed','k4_cluster','mapped_k5_role'],as_index=False)['fit_row_share'].sum()
            .groupby(['k4_cluster','mapped_k5_role'],as_index=False)['fit_row_share'].mean())
    matrix=mapped.pivot(index='k4_cluster',columns='mapped_k5_role',values='fit_row_share').reindex(index=[1,2,3,4],columns=role_order,fill_value=0).reset_index()
    matrix.to_csv(out_t/'profile_clustering_k45_optimally_mapped_transition_matrix.csv',index=False)

    decision_table, decision = create_decision_summary(transition_df, run_df, seeds, cfg['k5_structural_rule'])
    decision_table.to_csv(out_t/'profile_clustering_k45_structural_decision_evidence.csv', index=False)
    recommended_k = int(decision['recommended_k'])
    reported_k = 4
    ensure(recommended_k == reported_k, f'K-selection diagnostics do not reproduce reported K={reported_k}')

    candidate_rows=[]; candidate_material={}
    for k in [4,5]:
        selected = candidate_run(run_df, k, seeds, cfg['candidate_run_rule'])
        seed=int(selected['seed'])
        assn, proto, mapping = materialise_candidate(k, seed, assignments, prototypes, composition)
        candidate_material[k]=(assn,proto,mapping,seed)
        candidate_rows.append({
            'k': k, 'selected_seed_internal_reproducibility_only': seed,
            'selection_rule': 'diagnostic gates; silhouette >= candidate median; minimum inertia; lowest seed tie-break',
            'inertia_dtw': float(selected['inertia_dtw']), 'full_sample_dtw_silhouette': float(selected['full_sample_dtw_silhouette']),
            'minimum_group_share_all_assigned': float(selected['minimum_cluster_share_all_assigned']),
            'maximum_within_group_low_signal_share': float(selected['maximum_within_group_low_signal_share']),
            'iterations': int(selected['iterations']), 'iteration_cap_hit': bool(selected['reached_iteration_cap'])
        })
        write_assignments(assn, out_t/f'profile_clustering_k{k}_candidate_source_assignments.parquet', parquet_enabled)
        proto.to_csv(out_t/f'profile_clustering_k{k}_candidate_prototypes.csv', index=False)
        comp = composition[(composition['k']==k)&(composition['seed']==seed)].copy()
        comp['profile_group']=comp['cluster_id_aligned_to_reference'].map(mapping)
        peaks=proto.loc[proto.groupby('profile_group')['prototype_zscore'].idxmax(),['profile_group','local_slot']]
        comp=comp.merge(peaks,on='profile_group',how='left'); comp['peak_time']=comp['local_slot'].map(local_time_from_slot)
        comp.sort_values('profile_group').to_csv(out_t/f'profile_clustering_k{k}_candidate_profile_group_summary.csv',index=False)
    candidate_df=pd.DataFrame(candidate_rows)
    candidate_df.to_csv(out_t/'profile_clustering_k45_candidate_run_selection_audit.csv',index=False)

    assn, proto, mapping, final_seed = candidate_material[reported_k]
    final_comp = composition[(composition['k']==reported_k)&(composition['seed']==final_seed)].copy()
    final_comp['profile_group']=final_comp['cluster_id_aligned_to_reference'].map(mapping)
    peaks=proto.loc[proto.groupby('profile_group')['prototype_zscore'].idxmax(),['profile_group','local_slot']]
    final_comp=final_comp.merge(peaks,on='profile_group',how='left'); final_comp['peak_time']=final_comp['local_slot'].map(local_time_from_slot)
    final_comp=final_comp.sort_values('profile_group')
    write_assignments(assn, out_t/'profile_clustering_final_selected_source_assignments.parquet', parquet_enabled)
    proto.to_csv(out_t/'profile_clustering_final_selected_source_prototypes.csv', index=False)
    final_comp.to_csv(out_t/'profile_clustering_final_selected_profile_groups_dissertation.csv', index=False)
    write_markdown_table(final_comp[['profile_group','all_assigned_n','all_assigned_share','fit_eligible_n','low_signal_n','peak_time']], out_t/'profile_clustering_final_selected_profile_groups_dissertation.md')

    main_decision = pd.DataFrame([
        {'criterion':'Mean DTW inertia (20 initialisations)','k4':float(run_df[(run_df.k==4)&run_df.seed.isin(seeds)].inertia_dtw.mean()),'k5':float(run_df[(run_df.k==5)&run_df.seed.isin(seeds)].inertia_dtw.mean()),'evidence_direction':'Lower favours K=5'},
        {'criterion':'Mean full-sample silhouette (20 initialisations)','k4':float(run_df[(run_df.k==4)&run_df.seed.isin(seeds)].full_sample_dtw_silhouette.mean()),'k5':float(run_df[(run_df.k==5)&run_df.seed.isin(seeds)].full_sample_dtw_silhouette.mean()),'evidence_direction':'Higher favours K=4'},
        {'criterion':'Mean minimum group share (20 initialisations)','k4':float(run_df[(run_df.k==4)&run_df.seed.isin(seeds)].minimum_cluster_share_all_assigned.mean()),'k5':float(run_df[(run_df.k==5)&run_df.seed.isin(seeds)].minimum_cluster_share_all_assigned.mean()),'evidence_direction':'Higher favours K=4'},
        {'criterion':'Mean pairwise ARI','k4':float(focused[focused.k==4].pairwise_ari_mean.iloc[0]),'k5':float(focused[focused.k==5].pairwise_ari_mean.iloc[0]),'evidence_direction':'Higher favours K=5'},
        {'criterion':'Median extra-group origin purity','k4':np.nan,'k5':float(transition_df.extra_origin_purity.median()),'evidence_direction':'Higher supports structural K=5 split'},
        {'criterion':'Dominant-parent recurrence share','k4':np.nan,'k5':float(decision['structural_summary']['parent_mode_share']),'evidence_direction':'Higher supports structural K=5 split'},
        {'criterion':'Extra-group peak-band recurrence share','k4':np.nan,'k5':float(decision['structural_summary']['peak_band_mode_share']),'evidence_direction':'Higher supports structural K=5 split'},
        {'criterion':'Operational recommendation','k4':'SELECTED' if recommended_k==4 else '', 'k5':'SELECTED' if recommended_k==5 else '', 'evidence_direction':'Reported source-profile selection'},
    ])
    main_decision.to_csv(out_t/'profile_clustering_k4_vs_k5_decision_table_dissertation.csv',index=False)
    write_markdown_table(main_decision, out_t/'profile_clustering_k4_vs_k5_decision_table_dissertation.md')

    status={
        'status':'COMPLETE',
        'reported_k':reported_k,
        'final_candidate_seed_internal_only':final_seed,
        'source_households':int(assignments.household_id.nunique()),
        'focused_seeds':seeds,
        'kmeans_rerun':False,
        'cer_used':False,
        'forecasting_metrics_used':False,
        'survey_used':False,
        'training_period_only':True,
        'decision':decision,
    }
    write_json(out_m/'profile_clustering_final_decision_dissertation_status.json',status)

    decision_path=resolve(root,p['decision_document']); decision_path.parent.mkdir(parents=True,exist_ok=True)
    s=decision['structural_summary']; g=decision['decision_gates']
    decision_path.write_text(
        '# Profile clustering Final K Decision and Dissertation Outputs\n\n'
        '## Status\n\n'
        f'```text\nREPORTED K: {reported_k}\nK-MEANS RERUN: NO\nTARGET DATASET USED: NO\n```\n\n'
        '## Exact initialisation and iteration evidence\n\n'
        f'- Paired initialisations: {seeds}\n- Independent run setting: n_init=1 per seed and K.\n'
        '- Metric: DTW; initialisation: k-means++; max_iter=100; max_iter_barycenter=100; tolerance=1e-6.\n'
        '- The same 20 seeds were used for K=4 and K=5. No run hit the iteration cap.\n\n'
        '## K=5 structural transition evidence\n\n'
        f'- Dominant K=4 parent recurrence: {s["parent_mode_runs"]}/20 ({100*s["parent_mode_share"]:.1f}%).\n'
        f'- Median origin purity of the additional K=5 group: {100*s["median_origin_purity"]:.1f}%.\n'
        f'- Median extra-group share: {100*s["median_extra_group_share"]:.1f}%.\n'
        f'- Modal two-hour peak band: {s["peak_band_mode"]}, observed in {s["peak_band_mode_runs"]}/20 runs.\n'
        f'- Median two-child coverage of the parent group: {100*s["median_two_child_parent_coverage"]:.1f}%.\n'
        f'- Median extra-versus-sibling DTW distinctness ratio: {s["median_distinctness_ratio"]:.3f}.\n\n'
        '## Decision rule\n\n'
        'The retained diagnostics compare K=4 and K=5 using source-dataset structure, stability and fit evidence. The reported source-profile selection is K=4.\n\n'
        '## Gate results\n\n' + '\n'.join(f'- {name}: **{"PASS" if value else "FAIL"}**' for name,value in g.items()) + '\n\n'
        'The selected seed is retained only as a reproducibility parameter and is not part of the analytical interpretation.\n',
        encoding='utf-8')
    print('='*100)
    print('PROFILE CLUSTERING FINAL DECISION AND DISSERTATION OUTPUTS COMPLETE')
    print('='*100)
    print('Reported K         :',reported_k)
    print('K-Means rerun      : NO')
    return status


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True,type=Path); args=parser.parse_args()
    run(args.config)
    return 0

if __name__=='__main__':
    try: raise SystemExit(main())
    except ProfileClusteringFinalDecisionError as exc:
        print(f'PROFILE CLUSTERING FINAL DECISION FAILED: {exc}', flush=True)
        raise SystemExit(2)
