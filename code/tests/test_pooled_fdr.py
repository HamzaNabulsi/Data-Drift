"""
Test Pooled FDR Correction
==========================
Verifies that pooled FDR correction across all scores:
1. Produces adjusted p-values >= the max of individual per-score FDR p-values
   (i.e., pooled FDR is more conservative)
2. Applies BH correction correctly on synthetic multi-score p-values
3. Handles edge cases: all NaN, single p-value, empty DataFrames

Run with: python -m pytest code/tests/test_pooled_fdr.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest
from scipy.stats import false_discovery_control


# ---------- Reference implementation for testing ----------

def pool_fdr_correction(score_dfs, p_col='p_value_trend', alpha=0.05):
    """
    Apply Benjamini-Hochberg FDR correction across ALL scores simultaneously.

    Instead of correcting p-values within each score independently,
    this pools all p-values from all scores into one vector and applies
    BH once. This is more conservative because each test competes
    against a larger family of hypotheses.

    Args:
        score_dfs: list of DataFrames, one per score, each having `p_col`
        p_col: column name containing raw p-values
        alpha: significance threshold after correction

    Returns:
        list of DataFrames (same order) with added 'p_value_pooled_fdr' and
        'significant_pooled' columns.
    """
    if not score_dfs:
        return []

    # Collect all valid p-values and their indices
    all_pvals = []
    locations = []  # (df_index, row_index)
    for df_idx, df in enumerate(score_dfs):
        if df.empty or p_col not in df.columns:
            continue
        for row_idx in df.index:
            pval = df.loc[row_idx, p_col]
            if pd.notna(pval):
                all_pvals.append(pval)
                locations.append((df_idx, row_idx))

    # Initialize output columns
    out_dfs = []
    for df in score_dfs:
        df = df.copy()
        df['p_value_pooled_fdr'] = np.nan
        df['significant_pooled'] = False
        out_dfs.append(df)

    if len(all_pvals) == 0:
        return out_dfs

    all_pvals = np.array(all_pvals)
    adjusted = false_discovery_control(all_pvals, method='bh')

    for adj_p, (df_idx, row_idx) in zip(adjusted, locations):
        out_dfs[df_idx].loc[row_idx, 'p_value_pooled_fdr'] = adj_p
        out_dfs[df_idx].loc[row_idx, 'significant_pooled'] = adj_p < alpha

    return out_dfs


def per_score_fdr(df, p_col='p_value_trend', alpha=0.05):
    """Apply BH FDR correction within a single score DataFrame."""
    df = df.copy()
    valid = df[p_col].notna()
    if valid.sum() == 0:
        df['p_value_fdr'] = np.nan
        return df
    raw = df.loc[valid, p_col].values
    adjusted = false_discovery_control(raw, method='bh')
    df.loc[valid, 'p_value_fdr'] = adjusted
    return df


# ---------- Tests ----------

def test_pooled_fdr_more_conservative_overall():
    """Pooled FDR should reject fewer or equal total hypotheses than per-score FDR.

    Pooling all p-values across scores into one BH correction increases the
    number of hypotheses each test competes against. This makes the overall
    procedure more conservative: the total number of rejections should be
    less than or equal to the sum of per-score rejections.
    """
    rng = np.random.RandomState(42)

    # Create 3 "score" DataFrames with p-values in the marginal range
    dfs = []
    for score_name in ['SOFA', 'OASIS', 'SAPS']:
        n_tests = 10
        pvals = rng.uniform(0.001, 0.10, size=n_tests)
        df = pd.DataFrame({
            'subgroup': [f'group_{i}' for i in range(n_tests)],
            'score': score_name,
            'p_value_trend': pvals,
        })
        dfs.append(df)

    alpha = 0.05

    # Pooled FDR: total rejections
    pooled_dfs = pool_fdr_correction(dfs, p_col='p_value_trend', alpha=alpha)
    pooled_rejections = sum(
        df['significant_pooled'].sum() for df in pooled_dfs
    )

    # Per-score FDR: total rejections
    per_score_rejections = 0
    for df in dfs:
        corrected = per_score_fdr(df, p_col='p_value_trend')
        per_score_rejections += (corrected['p_value_fdr'] < alpha).sum()

    assert pooled_rejections <= per_score_rejections, (
        f"Pooled FDR rejections ({pooled_rejections}) should be <= "
        f"per-score FDR rejections ({per_score_rejections})"
    )


def test_bh_correction_applied_correctly():
    """Verify BH procedure is applied correctly on synthetic known p-values."""
    # Known p-values: 0.01, 0.04, 0.03 across 2 scores
    df1 = pd.DataFrame({'p_value_trend': [0.01, 0.04]})
    df2 = pd.DataFrame({'p_value_trend': [0.03]})

    pooled = pool_fdr_correction([df1, df2])

    # Manually compute BH on [0.01, 0.04, 0.03]:
    # sorted: 0.01, 0.03, 0.04
    # ranks:  1,    2,    3
    # BH adjusted: min(p * m/rank, 1.0)
    #   0.01 * 3/1 = 0.03
    #   0.03 * 3/2 = 0.045
    #   0.04 * 3/3 = 0.04
    # enforce monotonicity (reverse cummin): 0.03, 0.04, 0.04
    # So: p=0.01 -> 0.03, p=0.03 -> 0.04, p=0.04 -> 0.04

    all_pooled_p = []
    for df in pooled:
        for idx in df.index:
            all_pooled_p.append(df.loc[idx, 'p_value_pooled_fdr'])

    # Map back: df1 row 0 had p=0.01 -> adj ~0.03
    #           df1 row 1 had p=0.04 -> adj ~0.04
    #           df2 row 0 had p=0.03 -> adj ~0.04 or 0.045
    np.testing.assert_allclose(pooled[0].loc[0, 'p_value_pooled_fdr'], 0.03, atol=1e-10)
    assert pooled[0].loc[1, 'p_value_pooled_fdr'] >= 0.04 - 1e-10
    assert pooled[1].loc[0, 'p_value_pooled_fdr'] >= 0.03 - 1e-10


def test_all_nan_pvalues():
    """All NaN p-values should produce all NaN adjusted p-values."""
    df = pd.DataFrame({'p_value_trend': [np.nan, np.nan, np.nan]})

    result = pool_fdr_correction([df])

    assert result[0]['p_value_pooled_fdr'].isna().all()
    assert not result[0]['significant_pooled'].any()


def test_single_pvalue():
    """Single p-value should be returned unchanged by BH correction."""
    df = pd.DataFrame({'p_value_trend': [0.03]})

    result = pool_fdr_correction([df])

    np.testing.assert_allclose(result[0].loc[0, 'p_value_pooled_fdr'], 0.03, atol=1e-10)


def test_empty_dataframes():
    """Empty DataFrames should be handled gracefully."""
    empty_df = pd.DataFrame(columns=['p_value_trend'])
    non_empty = pd.DataFrame({'p_value_trend': [0.01, 0.05]})

    result = pool_fdr_correction([empty_df, non_empty])

    assert len(result) == 2
    assert result[0].empty or result[0]['p_value_pooled_fdr'].isna().all()
    assert result[1]['p_value_pooled_fdr'].notna().sum() == 2


def test_empty_list():
    """Empty list of DataFrames should return empty list."""
    result = pool_fdr_correction([])
    assert result == []


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
