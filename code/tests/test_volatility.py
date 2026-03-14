"""
Test Volatility Indicators
===========================
Verifies compute_volatility_indicators() with:
1. Monotonically increasing AUC -> low trend reversal count, specific CV
2. Flat AUC -> CV near 0, no drawdown
3. Oscillating AUC -> high trend reversal count
4. Single period -> handles gracefully

Run with: python -m pytest code/tests/test_volatility.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest


# ---------- Reference implementation for testing ----------

def compute_volatility_indicators(auc_series):
    """
    Compute volatility indicators for a series of AUC values across time periods.

    Args:
        auc_series: array-like of AUC values ordered by time period

    Returns:
        dict with keys:
            - cv: coefficient of variation (std / mean)
            - max_drawdown: largest peak-to-trough decline
            - trend_reversal_count: number of direction changes
    """
    auc = np.array(auc_series, dtype=float)

    # Remove NaN values
    auc = auc[~np.isnan(auc)]

    if len(auc) == 0:
        return {'cv': np.nan, 'max_drawdown': np.nan, 'trend_reversal_count': np.nan}

    if len(auc) == 1:
        return {'cv': 0.0, 'max_drawdown': 0.0, 'trend_reversal_count': 0}

    # Coefficient of variation
    mean_auc = np.mean(auc)
    if mean_auc == 0:
        cv = np.nan
    else:
        cv = np.std(auc, ddof=1) / abs(mean_auc)

    # Max drawdown: largest peak-to-trough decline
    running_max = np.maximum.accumulate(auc)
    drawdowns = running_max - auc
    max_drawdown = np.max(drawdowns)

    # Trend reversal count: number of direction changes
    if len(auc) < 3:
        trend_reversal_count = 0
    else:
        diffs = np.diff(auc)
        signs = np.sign(diffs)
        # Remove zeros (no change)
        signs = signs[signs != 0]
        if len(signs) < 2:
            trend_reversal_count = 0
        else:
            reversals = np.diff(signs)
            trend_reversal_count = int(np.sum(reversals != 0))

    return {
        'cv': cv,
        'max_drawdown': max_drawdown,
        'trend_reversal_count': trend_reversal_count,
    }


# ---------- Tests ----------

def test_monotonically_increasing():
    """Monotonically increasing AUC should have 0 trend reversals and positive CV."""
    aucs = [0.65, 0.68, 0.72, 0.76, 0.80]
    result = compute_volatility_indicators(aucs)

    assert result['trend_reversal_count'] == 0, (
        f"Expected 0 reversals for monotonic increase, got {result['trend_reversal_count']}"
    )
    assert result['cv'] > 0, "CV should be positive for varying AUC"
    # Max drawdown should be 0 for monotonically increasing
    np.testing.assert_allclose(result['max_drawdown'], 0.0, atol=1e-10)


def test_flat_auc():
    """Flat AUC should have CV near 0 and no drawdown."""
    aucs = [0.75, 0.75, 0.75, 0.75, 0.75]
    result = compute_volatility_indicators(aucs)

    np.testing.assert_allclose(result['cv'], 0.0, atol=1e-10,
                               err_msg="CV should be ~0 for flat AUC")
    np.testing.assert_allclose(result['max_drawdown'], 0.0, atol=1e-10,
                               err_msg="Max drawdown should be 0 for flat AUC")
    assert result['trend_reversal_count'] == 0


def test_oscillating_auc():
    """Oscillating AUC should have high trend reversal count."""
    # Alternating up/down pattern: 4 direction changes
    aucs = [0.70, 0.75, 0.68, 0.76, 0.65]
    result = compute_volatility_indicators(aucs)

    assert result['trend_reversal_count'] >= 3, (
        f"Expected >= 3 reversals for oscillating AUC, got {result['trend_reversal_count']}"
    )
    assert result['cv'] > 0
    assert result['max_drawdown'] > 0, "Oscillating AUC should have positive drawdown"


def test_single_period():
    """Single period should handle gracefully with zero volatility."""
    aucs = [0.80]
    result = compute_volatility_indicators(aucs)

    assert result['cv'] == 0.0
    assert result['max_drawdown'] == 0.0
    assert result['trend_reversal_count'] == 0


def test_empty_input():
    """Empty input should return NaN for all indicators."""
    result = compute_volatility_indicators([])

    assert np.isnan(result['cv'])
    assert np.isnan(result['max_drawdown'])
    assert np.isnan(result['trend_reversal_count'])


def test_two_periods():
    """Two periods should have 0 reversals but valid CV and drawdown."""
    aucs = [0.70, 0.80]
    result = compute_volatility_indicators(aucs)

    assert result['trend_reversal_count'] == 0
    assert result['cv'] > 0
    assert result['max_drawdown'] == 0.0  # increasing, so no drawdown


def test_two_periods_decreasing():
    """Decreasing two periods should have positive drawdown."""
    aucs = [0.80, 0.70]
    result = compute_volatility_indicators(aucs)

    np.testing.assert_allclose(result['max_drawdown'], 0.10, atol=1e-10)


def test_nan_values_handled():
    """NaN values in the AUC series should be ignored."""
    aucs = [0.70, np.nan, 0.75, 0.80]
    result = compute_volatility_indicators(aucs)

    assert not np.isnan(result['cv'])
    assert result['trend_reversal_count'] == 0  # monotonically increasing after NaN removal


def test_monotonically_decreasing():
    """Monotonically decreasing AUC should have 0 reversals and max drawdown = total drop."""
    aucs = [0.85, 0.80, 0.75, 0.70, 0.65]
    result = compute_volatility_indicators(aucs)

    assert result['trend_reversal_count'] == 0
    np.testing.assert_allclose(result['max_drawdown'], 0.20, atol=1e-10)
    assert result['cv'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
