"""
Test Clinical Significance (Effect Size Threshold)
===================================================
Verifies that `clinically_significant` column is True only when BOTH:
1. Statistically significant (FDR-corrected p < 0.05)
2. |delta| > MIN_CLINICALLY_SIGNIFICANT_DELTA (0.05 AUROC)

Run with: python -m pytest code/tests/test_effect_size.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest


# ---------- Reference implementation for testing ----------

MIN_CLINICALLY_SIGNIFICANT_DELTA = 0.05


def apply_clinical_significance(df, delta_col='delta', p_col='p_value_trend_fdr',
                                alpha=0.05,
                                min_delta=MIN_CLINICALLY_SIGNIFICANT_DELTA):
    """
    Add a 'clinically_significant' column that requires both statistical
    significance AND a minimum effect size.

    Args:
        df: DataFrame with drift results
        delta_col: column containing the AUC delta (last - first)
        p_col: column containing the FDR-corrected p-value
        alpha: significance threshold
        min_delta: minimum |delta| for clinical significance

    Returns:
        DataFrame with added 'clinically_significant' column
    """
    df = df.copy()
    stat_sig = df[p_col].notna() & (df[p_col] < alpha)
    effect_sig = df[delta_col].abs() > min_delta
    df['clinically_significant'] = stat_sig & effect_sig
    return df


# ---------- Tests ----------

def test_below_threshold_not_clinically_significant():
    """Delta = 0.03 (below 0.05 threshold) should NOT be clinically significant even if p < 0.05."""
    df = pd.DataFrame({
        'subgroup': ['group_a'],
        'delta': [0.03],
        'p_value_trend_fdr': [0.001],
    })

    result = apply_clinical_significance(df)

    assert not result.loc[0, 'clinically_significant'], (
        "Delta 0.03 < 0.05 threshold: should NOT be clinically significant"
    )


def test_above_threshold_and_significant():
    """Delta = 0.08 (above 0.05) with p < 0.05 should be clinically significant."""
    df = pd.DataFrame({
        'subgroup': ['group_a'],
        'delta': [0.08],
        'p_value_trend_fdr': [0.01],
    })

    result = apply_clinical_significance(df)

    assert result.loc[0, 'clinically_significant'], (
        "Delta 0.08 > 0.05 threshold AND p < 0.05: should be clinically significant"
    )


def test_above_threshold_but_not_stat_significant():
    """Delta = 0.08 (above threshold) but p > 0.05 should NOT be clinically significant."""
    df = pd.DataFrame({
        'subgroup': ['group_a'],
        'delta': [0.08],
        'p_value_trend_fdr': [0.15],
    })

    result = apply_clinical_significance(df)

    assert not result.loc[0, 'clinically_significant'], (
        "p = 0.15 > 0.05: should NOT be clinically significant despite large delta"
    )


def test_negative_delta_above_threshold():
    """Negative delta with |delta| > threshold and p < 0.05 should be clinically significant."""
    df = pd.DataFrame({
        'subgroup': ['group_a'],
        'delta': [-0.10],
        'p_value_trend_fdr': [0.001],
    })

    result = apply_clinical_significance(df)

    assert result.loc[0, 'clinically_significant'], (
        "|delta| = 0.10 > 0.05 threshold AND p < 0.05: should be clinically significant"
    )


def test_exact_threshold_boundary():
    """Delta exactly at 0.05 should NOT be clinically significant (strict inequality)."""
    df = pd.DataFrame({
        'subgroup': ['group_a'],
        'delta': [0.05],
        'p_value_trend_fdr': [0.01],
    })

    result = apply_clinical_significance(df)

    assert not result.loc[0, 'clinically_significant'], (
        "Delta exactly 0.05 should NOT pass strict > 0.05 threshold"
    )


def test_nan_pvalue():
    """NaN p-value should result in clinically_significant = False."""
    df = pd.DataFrame({
        'subgroup': ['group_a'],
        'delta': [0.10],
        'p_value_trend_fdr': [np.nan],
    })

    result = apply_clinical_significance(df)

    assert not result.loc[0, 'clinically_significant'], (
        "NaN p-value should NOT be clinically significant"
    )


def test_multiple_rows_mixed():
    """Test with multiple rows of mixed significance."""
    df = pd.DataFrame({
        'subgroup': ['a', 'b', 'c', 'd', 'e'],
        'delta': [0.03, 0.08, 0.08, -0.12, 0.05],
        'p_value_trend_fdr': [0.001, 0.01, 0.15, 0.001, 0.01],
    })

    result = apply_clinical_significance(df)

    expected = [False, True, False, True, False]
    actual = result['clinically_significant'].tolist()
    assert actual == expected, f"Expected {expected}, got {actual}"


def test_zero_delta():
    """Zero delta should NOT be clinically significant."""
    df = pd.DataFrame({
        'subgroup': ['group_a'],
        'delta': [0.0],
        'p_value_trend_fdr': [0.001],
    })

    result = apply_clinical_significance(df)

    assert not result.loc[0, 'clinically_significant']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
