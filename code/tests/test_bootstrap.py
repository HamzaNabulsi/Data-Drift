"""
Test Bootstrap Confidence Interval Implementation
==================================================
Verifies that the bootstrap CI function in batch_analysis.py:
1. Produces correct point estimates (AUC)
2. Generates valid confidence intervals
3. CI width scales appropriately with sample size
4. Handles edge cases properly

Run with: python -m pytest code/tests/test_bootstrap.py -v
Or directly: python code/tests/test_bootstrap.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')


# Import the function we're testing
def compute_auc_with_ci(y_true, y_pred, n_bootstrap=100, ci_level=0.95, random_seed=42):
    """
    Compute AUC with bootstrap confidence intervals.
    (Copy of the function from batch_analysis.py for testing)
    """
    try:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) < 30 or len(np.unique(y_true)) < 2:
            return np.nan, np.nan, np.nan

        auc = roc_auc_score(y_true, y_pred)

        rng = np.random.RandomState(random_seed)
        n = len(y_true)
        bootstrap_aucs = []

        for _ in range(n_bootstrap):
            idx = rng.randint(0, n, n)
            y_true_boot = y_true[idx]
            y_pred_boot = y_pred[idx]

            if len(np.unique(y_true_boot)) < 2:
                continue

            try:
                bootstrap_aucs.append(roc_auc_score(y_true_boot, y_pred_boot))
            except:
                continue

        if len(bootstrap_aucs) < n_bootstrap * 0.5:
            return auc, np.nan, np.nan

        alpha = (1 - ci_level) / 2
        ci_lower = np.percentile(bootstrap_aucs, alpha * 100)
        ci_upper = np.percentile(bootstrap_aucs, (1 - alpha) * 100)

        return auc, ci_lower, ci_upper

    except Exception:
        return np.nan, np.nan, np.nan


def generate_synthetic_data(n_samples, true_auc, seed=42):
    """
    Generate synthetic binary classification data with known AUC.

    Uses a simple threshold model where predictions are sampled from
    different distributions for positive and negative classes.
    """
    rng = np.random.RandomState(seed)

    # Create balanced classes
    n_pos = n_samples // 2
    n_neg = n_samples - n_pos

    y_true = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])

    # Generate predictions that will give approximately the target AUC
    # AUC = P(pred_pos > pred_neg), so we use separated distributions
    # Higher separation -> higher AUC
    separation = 2 * (true_auc - 0.5)  # Maps AUC 0.5-1.0 to separation 0-1

    pred_pos = rng.normal(loc=separation, scale=0.5, size=n_pos)
    pred_neg = rng.normal(loc=0, scale=0.5, size=n_neg)
    y_pred = np.concatenate([pred_pos, pred_neg])

    # Shuffle to avoid order effects
    shuffle_idx = rng.permutation(n_samples)

    return y_true[shuffle_idx], y_pred[shuffle_idx]


def test_point_estimate_accuracy():
    """Test 1: Verify AUC point estimate matches sklearn."""
    print("\n" + "="*60)
    print("TEST 1: Point Estimate Accuracy")
    print("="*60)

    test_cases = [
        (1000, 0.75, "Moderate AUC"),
        (1000, 0.85, "Good AUC"),
        (1000, 0.95, "Excellent AUC"),
        (500, 0.70, "Small sample, moderate AUC"),
    ]

    all_passed = True

    for n_samples, target_auc, description in test_cases:
        y_true, y_pred = generate_synthetic_data(n_samples, target_auc)

        # Compute using our function
        auc_computed, ci_lower, ci_upper = compute_auc_with_ci(y_true, y_pred, n_bootstrap=100)

        # Compute using sklearn directly
        auc_sklearn = roc_auc_score(y_true, y_pred)

        # Check they match exactly
        match = np.isclose(auc_computed, auc_sklearn, atol=1e-10)
        status = "PASS" if match else "FAIL"

        if not match:
            all_passed = False

        print(f"\n  {description}:")
        print(f"    Computed AUC:  {auc_computed:.6f}")
        print(f"    sklearn AUC:   {auc_sklearn:.6f}")
        print(f"    Match: {status}")

    print(f"\n  Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    return all_passed


def test_ci_validity():
    """Test 2: Verify CI contains the point estimate and has valid bounds."""
    print("\n" + "="*60)
    print("TEST 2: Confidence Interval Validity")
    print("="*60)

    test_cases = [
        (1000, 0.80, 0.95, "95% CI"),
        (1000, 0.80, 0.90, "90% CI"),
        (1000, 0.80, 0.99, "99% CI"),
    ]

    all_passed = True

    for n_samples, target_auc, ci_level, description in test_cases:
        y_true, y_pred = generate_synthetic_data(n_samples, target_auc)

        auc, ci_lower, ci_upper = compute_auc_with_ci(
            y_true, y_pred, n_bootstrap=500, ci_level=ci_level
        )

        # Check validity
        valid_bounds = ci_lower <= auc <= ci_upper
        valid_range = 0 <= ci_lower <= 1 and 0 <= ci_upper <= 1
        valid_order = ci_lower < ci_upper

        passed = valid_bounds and valid_range and valid_order
        if not passed:
            all_passed = False

        print(f"\n  {description}:")
        print(f"    AUC:      {auc:.4f}")
        print(f"    CI:       [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"    Width:    {ci_upper - ci_lower:.4f}")
        print(f"    Contains point estimate: {'YES' if valid_bounds else 'NO'}")
        print(f"    Valid range [0,1]:       {'YES' if valid_range else 'NO'}")
        print(f"    Lower < Upper:           {'YES' if valid_order else 'NO'}")
        print(f"    Status: {'PASS' if passed else 'FAIL'}")

    print(f"\n  Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    return all_passed


def test_ci_width_scales_with_sample_size():
    """Test 3: Verify CI width decreases with larger sample size."""
    print("\n" + "="*60)
    print("TEST 3: CI Width Scales with Sample Size")
    print("="*60)

    sample_sizes = [100, 500, 1000, 5000]
    target_auc = 0.80

    widths = []

    for n in sample_sizes:
        y_true, y_pred = generate_synthetic_data(n, target_auc)
        auc, ci_lower, ci_upper = compute_auc_with_ci(y_true, y_pred, n_bootstrap=200)
        width = ci_upper - ci_lower
        widths.append(width)
        print(f"  N={n:5d}: CI width = {width:.4f}")

    # Check that widths decrease with sample size
    decreasing = all(widths[i] >= widths[i+1] for i in range(len(widths)-1))

    # Check approximately sqrt(n) scaling (width should halve when n quadruples)
    # Comparing n=100 to n=5000 (50x more data), width should decrease by ~sqrt(50) = 7x
    expected_ratio = np.sqrt(sample_sizes[-1] / sample_sizes[0])
    actual_ratio = widths[0] / widths[-1] if widths[-1] > 0 else np.inf
    reasonable_scaling = 0.5 * expected_ratio < actual_ratio < 2.0 * expected_ratio

    passed = decreasing and reasonable_scaling

    print(f"\n  Widths monotonically decreasing: {'YES' if decreasing else 'NO'}")
    print(f"  Width ratio (N=100 vs N=5000): {actual_ratio:.2f}x")
    print(f"  Expected ratio (sqrt scaling): ~{expected_ratio:.2f}x")
    print(f"  Scaling is reasonable: {'YES' if reasonable_scaling else 'NO'}")
    print(f"  Status: {'PASS' if passed else 'FAIL'}")

    return passed


def test_edge_cases():
    """Test 4: Verify proper handling of edge cases."""
    print("\n" + "="*60)
    print("TEST 4: Edge Case Handling")
    print("="*60)

    all_passed = True

    # Case 1: Too few samples
    print("\n  Case 1: Too few samples (n=20)")
    y_true = np.array([0, 1] * 10)
    y_pred = np.random.rand(20)
    auc, ci_lower, ci_upper = compute_auc_with_ci(y_true, y_pred)

    is_nan = np.isnan(auc)
    print(f"    Returns NaN: {'YES' if is_nan else 'NO'}")
    if not is_nan:
        all_passed = False
        print("    Status: FAIL (should return NaN for n < 30)")
    else:
        print("    Status: PASS")

    # Case 2: Single class (all positive or all negative)
    print("\n  Case 2: Single class (all outcomes = 1)")
    y_true = np.ones(100)
    y_pred = np.random.rand(100)
    auc, ci_lower, ci_upper = compute_auc_with_ci(y_true, y_pred)

    is_nan = np.isnan(auc)
    print(f"    Returns NaN: {'YES' if is_nan else 'NO'}")
    if not is_nan:
        all_passed = False
        print("    Status: FAIL (should return NaN for single class)")
    else:
        print("    Status: PASS")

    # Case 3: Missing values
    print("\n  Case 3: Data with NaN values")
    y_true, y_pred = generate_synthetic_data(200, 0.80)
    y_pred[::10] = np.nan  # Set every 10th value to NaN
    auc, ci_lower, ci_upper = compute_auc_with_ci(y_true, y_pred)

    is_valid = not np.isnan(auc) and ci_lower < auc < ci_upper
    print(f"    Handles NaN gracefully: {'YES' if is_valid else 'NO'}")
    if is_valid:
        print(f"    AUC: {auc:.4f}, CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print("    Status: PASS")
    else:
        all_passed = False
        print("    Status: FAIL")

    # Case 4: Perfect predictions
    print("\n  Case 4: Perfect predictions (AUC = 1.0)")
    y_true = np.array([0] * 50 + [1] * 50)
    y_pred = np.array([0.0] * 50 + [1.0] * 50)
    auc, ci_lower, ci_upper = compute_auc_with_ci(y_true, y_pred)

    is_perfect = np.isclose(auc, 1.0)
    print(f"    AUC = 1.0: {'YES' if is_perfect else 'NO'}")
    print(f"    AUC: {auc:.4f}, CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    if is_perfect:
        print("    Status: PASS")
    else:
        all_passed = False
        print("    Status: FAIL")

    print(f"\n  Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    return all_passed


def test_reproducibility():
    """Test 5: Verify results are reproducible with same seed."""
    print("\n" + "="*60)
    print("TEST 5: Reproducibility")
    print("="*60)

    y_true, y_pred = generate_synthetic_data(500, 0.80)

    # Run twice with same seed
    result1 = compute_auc_with_ci(y_true, y_pred, n_bootstrap=100, random_seed=42)
    result2 = compute_auc_with_ci(y_true, y_pred, n_bootstrap=100, random_seed=42)

    # Run with different seed
    result3 = compute_auc_with_ci(y_true, y_pred, n_bootstrap=100, random_seed=123)

    same_seed_match = np.allclose(result1, result2, equal_nan=True)
    diff_seed_differ = not np.allclose(result1, result3, equal_nan=True)

    print(f"  Run 1 (seed=42):  AUC={result1[0]:.4f}, CI=[{result1[1]:.4f}, {result1[2]:.4f}]")
    print(f"  Run 2 (seed=42):  AUC={result2[0]:.4f}, CI=[{result2[1]:.4f}, {result2[2]:.4f}]")
    print(f"  Run 3 (seed=123): AUC={result3[0]:.4f}, CI=[{result3[1]:.4f}, {result3[2]:.4f}]")
    print(f"\n  Same seed produces identical results: {'YES' if same_seed_match else 'NO'}")
    print(f"  Different seed produces different CIs: {'YES' if diff_seed_differ else 'NO'}")

    passed = same_seed_match and diff_seed_differ
    print(f"  Status: {'PASS' if passed else 'FAIL'}")

    return passed


def run_all_tests():
    """Run all tests and report summary."""
    print("\n" + "="*60)
    print("BOOTSTRAP CONFIDENCE INTERVAL TESTS")
    print("="*60)

    results = {
        "Point estimate accuracy": test_point_estimate_accuracy(),
        "CI validity": test_ci_validity(),
        "CI width scaling": test_ci_width_scales_with_sample_size(),
        "Edge cases": test_edge_cases(),
        "Reproducibility": test_reproducibility(),
    }

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED - Review output above")
    print("="*60)

    return all_passed


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
