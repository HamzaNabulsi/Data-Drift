#!/usr/bin/env python3
"""
Tests for Page's L trend test implementation in batch_analysis.py.

Validates:
1. Monotonic increasing trend detected
2. Monotonic decreasing trend detected
3. No trend (flat) not falsely significant
4. Two-sided detection works
5. Edge case: 2 periods falls back to Mann-Whitney
6. Edge case: too few replicates handled gracefully
7. FDR correction reduces false positives
8. Backward compatibility: legacy columns exist
"""
import sys
import os
from pathlib import Path

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Patch sys.argv before importing batch_analysis (it parses args at module level)
sys.argv = [sys.argv[0]]

import numpy as np
import pandas as pd
import pytest

import batch_analysis
batch_analysis.N_BOOTSTRAP = 50


def _make_results_df(aucs_by_period, subgroup_type='Overall', subgroup='All'):
    """Helper: create a results_df from a dict of {period: auc}."""
    rows = []
    for period, auc in aucs_by_period.items():
        rows.append({
            'subgroup_type': subgroup_type,
            'subgroup': subgroup,
            'time_period': str(period),
            'auc': auc,
            'auc_ci_lower': auc - 0.02,
            'auc_ci_upper': auc + 0.02,
            'n': 500,
            'n_deaths': 50,
            'mortality_rate': 0.1,
        })
    return pd.DataFrame(rows)


def _make_bootstrap_store(periods, center_aucs, n_replicates=50,
                          subgroup_type='Overall', subgroup='All', noise=0.02):
    """Helper: create bootstrap replicates centered around given AUCs."""
    rng = np.random.RandomState(42)
    store = {}
    for period, center in zip(periods, center_aucs):
        replicates = rng.normal(center, noise, n_replicates).tolist()
        # Clamp to [0, 1]
        replicates = [max(0.0, min(1.0, r)) for r in replicates]
        store[(subgroup_type, subgroup, str(period))] = replicates
    return store


def test_increasing_trend():
    """Monotonically increasing AUC should be detected as significant."""
    periods = [2010, 2012, 2014, 2016, 2018]
    aucs = [0.65, 0.68, 0.72, 0.76, 0.80]

    results_df = _make_results_df(dict(zip(periods, aucs)))
    bootstrap_store = _make_bootstrap_store(periods, aucs)

    deltas = batch_analysis.compute_drift_trend_test(results_df, bootstrap_store)

    assert len(deltas) == 1
    row = deltas.iloc[0]
    assert row['trend_direction'] == 'increasing'
    assert row['p_value_trend'] < 0.05
    assert row['delta'] == pytest.approx(0.80 - 0.65, abs=0.001)


def test_decreasing_trend():
    """Monotonically decreasing AUC should be detected as significant."""
    periods = [2010, 2012, 2014, 2016, 2018]
    aucs = [0.80, 0.76, 0.72, 0.68, 0.65]

    results_df = _make_results_df(dict(zip(periods, aucs)))
    bootstrap_store = _make_bootstrap_store(periods, aucs)

    deltas = batch_analysis.compute_drift_trend_test(results_df, bootstrap_store)

    assert len(deltas) == 1
    row = deltas.iloc[0]
    assert row['trend_direction'] == 'decreasing'
    assert row['p_value_trend'] < 0.05
    assert row['delta'] == pytest.approx(0.65 - 0.80, abs=0.001)


def test_flat_not_significant():
    """Flat AUC across periods should NOT be significant."""
    periods = [2010, 2012, 2014, 2016, 2018]
    aucs = [0.72, 0.72, 0.72, 0.72, 0.72]

    results_df = _make_results_df(dict(zip(periods, aucs)))
    bootstrap_store = _make_bootstrap_store(periods, aucs)

    deltas = batch_analysis.compute_drift_trend_test(results_df, bootstrap_store)

    assert len(deltas) == 1
    row = deltas.iloc[0]
    assert row['p_value_trend'] > 0.05
    assert not row['significant']


def test_two_period_fallback():
    """With only 2 periods, should fall back to Mann-Whitney U test."""
    periods = [2010, 2020]
    aucs = [0.65, 0.80]

    results_df = _make_results_df(dict(zip(periods, aucs)))
    bootstrap_store = _make_bootstrap_store(periods, aucs)

    deltas = batch_analysis.compute_drift_trend_test(results_df, bootstrap_store)

    assert len(deltas) == 1
    row = deltas.iloc[0]
    assert pd.notna(row['p_value_trend'])
    assert row['n_periods'] == 2


def test_too_few_replicates():
    """With < 3 replicates per period, should still produce output (with NaN p-value)."""
    periods = [2010, 2012, 2014]
    aucs = [0.65, 0.70, 0.75]

    results_df = _make_results_df(dict(zip(periods, aucs)))
    # Only 2 replicates per period
    bootstrap_store = _make_bootstrap_store(periods, aucs, n_replicates=2)

    deltas = batch_analysis.compute_drift_trend_test(results_df, bootstrap_store)

    assert len(deltas) == 1
    # Should handle gracefully (NaN p-value)
    row = deltas.iloc[0]
    assert pd.isna(row['p_value_trend']) or isinstance(row['p_value_trend'], float)


def test_fdr_correction_applied():
    """FDR correction should produce p_value_trend_fdr >= p_value_trend."""
    periods = [2010, 2012, 2014, 2016]

    # Create multiple subgroups
    rows = []
    store = {}
    rng = np.random.RandomState(123)
    for i, subgroup in enumerate(['Male', 'Female', 'Young', 'Old']):
        aucs = [0.65 + 0.03 * j + rng.normal(0, 0.01) for j in range(len(periods))]
        for p, auc in zip(periods, aucs):
            rows.append({
                'subgroup_type': 'Test',
                'subgroup': subgroup,
                'time_period': str(p),
                'auc': auc,
                'auc_ci_lower': auc - 0.02,
                'auc_ci_upper': auc + 0.02,
                'n': 500,
                'n_deaths': 50,
                'mortality_rate': 0.1,
            })
            replicates = rng.normal(auc, 0.02, 50).tolist()
            store[('Test', subgroup, str(p))] = replicates

    results_df = pd.DataFrame(rows)
    deltas = batch_analysis.compute_drift_trend_test(results_df, store)

    # FDR-adjusted p-values should be >= raw p-values
    valid = deltas.dropna(subset=['p_value_trend', 'p_value_trend_fdr'])
    for _, row in valid.iterrows():
        assert row['p_value_trend_fdr'] >= row['p_value_trend'] - 1e-10


def test_backward_compatibility_columns():
    """Legacy columns (p_value_delong, z_statistic, p_value_permutation) should exist."""
    periods = [2010, 2012, 2014, 2016]
    aucs = [0.65, 0.70, 0.75, 0.80]

    results_df = _make_results_df(dict(zip(periods, aucs)))
    bootstrap_store = _make_bootstrap_store(periods, aucs)

    deltas = batch_analysis.compute_drift_trend_test(results_df, bootstrap_store)

    assert 'p_value_delong' in deltas.columns
    assert 'z_statistic' in deltas.columns
    assert 'p_value_permutation' in deltas.columns
    assert 'significant' in deltas.columns
    assert 'delta' in deltas.columns

    # New columns should also exist
    assert 'p_value_trend' in deltas.columns
    assert 'p_value_trend_fdr' in deltas.columns
    assert 'trend_direction' in deltas.columns
    assert 'n_periods' in deltas.columns
    assert 'page_L_statistic' in deltas.columns


def test_between_group_different_trends():
    """Two groups with opposite trends should have significant between-group difference."""
    periods = [2010, 2012, 2014, 2016, 2018]
    # Group A: improving
    aucs_a = [0.65, 0.68, 0.72, 0.76, 0.80]
    # Group B: degrading
    aucs_b = [0.80, 0.76, 0.72, 0.68, 0.65]

    # Build results_df with both groups
    rows = []
    for p, auc_a, auc_b in zip(periods, aucs_a, aucs_b):
        rows.append({
            'subgroup_type': 'Race', 'subgroup': 'GroupA',
            'time_period': str(p), 'auc': auc_a,
            'auc_ci_lower': auc_a - 0.02, 'auc_ci_upper': auc_a + 0.02,
            'n': 500, 'n_deaths': 50, 'mortality_rate': 0.1,
        })
        rows.append({
            'subgroup_type': 'Race', 'subgroup': 'GroupB',
            'time_period': str(p), 'auc': auc_b,
            'auc_ci_lower': auc_b - 0.02, 'auc_ci_upper': auc_b + 0.02,
            'n': 500, 'n_deaths': 50, 'mortality_rate': 0.1,
        })
        # Overall
        rows.append({
            'subgroup_type': 'Overall', 'subgroup': 'All',
            'time_period': str(p), 'auc': 0.72,
            'auc_ci_lower': 0.70, 'auc_ci_upper': 0.74,
            'n': 1000, 'n_deaths': 100, 'mortality_rate': 0.1,
        })

    results_df = pd.DataFrame(rows)

    # Build bootstrap store
    rng = np.random.RandomState(42)
    store = {}
    for p, auc_a, auc_b in zip(periods, aucs_a, aucs_b):
        store[('Race', 'GroupA', str(p))] = rng.normal(auc_a, 0.02, 50).tolist()
        store[('Race', 'GroupB', str(p))] = rng.normal(auc_b, 0.02, 50).tolist()
        store[('Overall', 'All', str(p))] = rng.normal(0.72, 0.02, 50).tolist()

    # First compute deltas (needed as input)
    deltas = batch_analysis.compute_drift_trend_test(results_df, store)

    # Now compute between-group comparison
    bg = batch_analysis.compute_between_group_drift_comparison(deltas, store)

    assert len(bg) > 0

    # Find the GroupA vs GroupB pairwise comparison
    pairwise = bg[(bg['group_a'] == 'GroupA') & (bg['group_b'] == 'GroupB')]
    assert len(pairwise) == 1
    row = pairwise.iloc[0]

    # Should be significant (opposite trends)
    assert row['p_value'] < 0.05
    # delta_diff should be positive (GroupA improved, GroupB degraded)
    assert row['delta_diff'] > 0

    # Also check vs Overall comparisons exist
    vs_overall = bg[bg['group_b'] == 'Overall (All)']
    assert len(vs_overall) >= 2  # GroupA vs Overall and GroupB vs Overall


def test_between_group_same_trends():
    """Two groups with identical trends should NOT have significant difference."""
    periods = [2010, 2012, 2014, 2016]
    aucs = [0.70, 0.72, 0.74, 0.76]

    rows = []
    store = {}
    rng = np.random.RandomState(99)
    for p, auc in zip(periods, aucs):
        for sg in ['GroupA', 'GroupB']:
            rows.append({
                'subgroup_type': 'Gender', 'subgroup': sg,
                'time_period': str(p), 'auc': auc,
                'auc_ci_lower': auc - 0.02, 'auc_ci_upper': auc + 0.02,
                'n': 500, 'n_deaths': 50, 'mortality_rate': 0.1,
            })
            store[('Gender', sg, str(p))] = rng.normal(auc, 0.02, 50).tolist()
        rows.append({
            'subgroup_type': 'Overall', 'subgroup': 'All',
            'time_period': str(p), 'auc': auc,
            'auc_ci_lower': auc - 0.02, 'auc_ci_upper': auc + 0.02,
            'n': 1000, 'n_deaths': 100, 'mortality_rate': 0.1,
        })
        store[('Overall', 'All', str(p))] = rng.normal(auc, 0.02, 50).tolist()

    results_df = pd.DataFrame(rows)
    deltas = batch_analysis.compute_drift_trend_test(results_df, store)
    bg = batch_analysis.compute_between_group_drift_comparison(deltas, store)

    assert len(bg) > 0
    # Pairwise comparison should NOT be significant
    pairwise = bg[(bg['group_a'] == 'GroupA') & (bg['group_b'] == 'GroupB')]
    assert len(pairwise) == 1
    assert pairwise.iloc[0]['p_value'] > 0.05


def test_bootstrap_split_trend_uses_first_half():
    """Verify that trend tests use the first half of bootstrap replicates."""
    periods = [2010, 2012, 2014, 2016, 2018]
    aucs = [0.65, 0.68, 0.72, 0.76, 0.80]

    results_df = _make_results_df(dict(zip(periods, aucs)))

    rng = np.random.RandomState(42)
    store = {}
    n_reps = 100
    for p, center in zip(periods, aucs):
        # First half: centered at the actual AUC (strong trend signal)
        first_half = rng.normal(center, 0.01, n_reps // 2).tolist()
        # Second half: very different values (noise centered at 0.5)
        second_half = rng.normal(0.5, 0.01, n_reps // 2).tolist()
        store[('Overall', 'All', str(p))] = first_half + second_half

    deltas = batch_analysis.compute_drift_trend_test(results_df, store)

    row = deltas.iloc[0]
    # If trend test uses first half (clean signal), it should detect increasing trend
    assert row['trend_direction'] == 'increasing'
    assert row['p_value_trend'] < 0.05, (
        "Trend test should be significant when first half has clean increasing signal"
    )


def test_bootstrap_split_between_group_uses_second_half():
    """Verify that between-group comparisons use the second half of bootstrap replicates."""
    periods = [2010, 2012, 2014, 2016, 2018]
    aucs_a = [0.65, 0.68, 0.72, 0.76, 0.80]
    aucs_b = [0.80, 0.76, 0.72, 0.68, 0.65]

    rows = []
    store = {}
    rng = np.random.RandomState(42)
    n_reps = 100

    for p, auc_a, auc_b in zip(periods, aucs_a, aucs_b):
        for sg, auc in [('GroupA', auc_a), ('GroupB', auc_b), ('All', (auc_a + auc_b) / 2)]:
            stype = 'Race' if sg != 'All' else 'Overall'
            rows.append({
                'subgroup_type': stype, 'subgroup': sg,
                'time_period': str(p), 'auc': auc,
                'auc_ci_lower': auc - 0.02, 'auc_ci_upper': auc + 0.02,
                'n': 500, 'n_deaths': 50, 'mortality_rate': 0.1,
            })
            # First half: noise at 0.5 (no signal for between-group)
            first_half = rng.normal(0.5, 0.01, n_reps // 2).tolist()
            # Second half: actual signal
            second_half = rng.normal(auc, 0.02, n_reps // 2).tolist()
            store[(stype, sg, str(p))] = first_half + second_half

    results_df = pd.DataFrame(rows)
    deltas = batch_analysis.compute_drift_trend_test(results_df, store)
    bg = batch_analysis.compute_between_group_drift_comparison(deltas, store)

    # Between-group should detect difference using second half (clean signal)
    pairwise = bg[(bg['group_a'] == 'GroupA') & (bg['group_b'] == 'GroupB')]
    assert len(pairwise) == 1
    assert pairwise.iloc[0]['p_value'] < 0.05, (
        "Between-group comparison should be significant when second half has clean signal"
    )


def test_split_results_differ_from_nosplit():
    """Results with split bootstrap should differ from hypothetical non-split.

    We verify this indirectly: the split implementation uses only the first
    half of replicates (50 out of 100). We confirm that the L statistic
    from the split differs from the L statistic using all 100 replicates,
    proving that different data subsets are being used.
    """
    from scipy.stats import page_trend_test

    periods = [2010, 2012, 2014, 2016, 2018]
    # Use weak trend with high noise so p-values stay in distinguishable range
    aucs = [0.72, 0.72, 0.73, 0.73, 0.74]

    results_df = _make_results_df(dict(zip(periods, aucs)))
    rng = np.random.RandomState(42)
    n_reps = 100

    store = {}
    for p, center in zip(periods, aucs):
        store[('Overall', 'All', str(p))] = rng.normal(center, 0.05, n_reps).tolist()

    # Split result (first half only, 50 replicates)
    deltas_split = batch_analysis.compute_drift_trend_test(results_df, store)
    L_split = deltas_split.iloc[0]['page_L_statistic']

    # Non-split: manually compute using all 100 replicates
    replicate_lists = []
    for p in periods:
        replicate_lists.append(store[('Overall', 'All', str(p))])
    min_len = min(len(r) for r in replicate_lists)
    matrix_full = np.column_stack([np.array(r[:min_len]) for r in replicate_lists])
    res_full = page_trend_test(matrix_full, method='asymptotic')
    L_full = res_full.statistic

    # The L statistics must differ because they use different numbers of replicates
    # (50 vs 100 rows in the matrix). Page's L scales with number of rows.
    assert L_split != L_full, (
        f"Split L={L_split} should differ from full L={L_full} "
        "to confirm different data subsets are being used"
    )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
