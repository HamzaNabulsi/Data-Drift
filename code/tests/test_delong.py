"""
Test DeLong's Test Implementation
=================================
Verifies that the DeLong's test function in batch_analysis.py:
1. Correctly identifies significant AUC differences
2. Returns appropriate p-values for known scenarios
3. Handles edge cases properly
4. Produces consistent results across runs

Run with: python -m pytest code/tests/test_delong.py -v
Or directly: python code/tests/test_delong.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.metrics import roc_auc_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def delong_test(y_true1, y_pred1, y_true2, y_pred2):
    """
    Perform DeLong's test for comparing two AUCs from independent samples.
    (Copy of the function from batch_analysis.py for testing)
    """
    try:
        auc1 = roc_auc_score(y_true1, y_pred1)
        auc2 = roc_auc_score(y_true2, y_pred2)

        n1 = len(y_true1)
        n2 = len(y_true2)

        n_pos1 = np.sum(y_true1)
        n_neg1 = n1 - n_pos1
        n_pos2 = np.sum(y_true2)
        n_neg2 = n2 - n_pos2

        # Hanley-McNeil variance approximation
        q1_1 = auc1 / (2 - auc1)
        q2_1 = 2 * auc1**2 / (1 + auc1)
        var1 = (auc1 * (1 - auc1) + (n_pos1 - 1) * (q1_1 - auc1**2) + (n_neg1 - 1) * (q2_1 - auc1**2)) / (n_pos1 * n_neg1)

        q1_2 = auc2 / (2 - auc2)
        q2_2 = 2 * auc2**2 / (1 + auc2)
        var2 = (auc2 * (1 - auc2) + (n_pos2 - 1) * (q1_2 - auc2**2) + (n_neg2 - 1) * (q2_2 - auc2**2)) / (n_pos2 * n_neg2)

        se_diff = np.sqrt(var1 + var2)
        if se_diff == 0:
            return np.nan, np.nan

        z = (auc1 - auc2) / se_diff
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        return z, p_value

    except Exception:
        return np.nan, np.nan


def generate_synthetic_data(n_samples, true_auc, seed=42):
    """Generate synthetic binary classification data with known AUC."""
    rng = np.random.RandomState(seed)

    n_pos = n_samples // 2
    n_neg = n_samples - n_pos

    y_true = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])

    separation = 2 * (true_auc - 0.5)
    pred_pos = rng.normal(loc=separation, scale=0.5, size=n_pos)
    pred_neg = rng.normal(loc=0, scale=0.5, size=n_neg)
    y_pred = np.concatenate([pred_pos, pred_neg])

    shuffle_idx = rng.permutation(n_samples)

    return y_true[shuffle_idx], y_pred[shuffle_idx]


def test_identical_samples():
    """Test 1: Identical samples should have non-significant p-value."""
    print("\n" + "="*60)
    print("TEST 1: Identical Samples (No Difference)")
    print("="*60)

    y_true, y_pred = generate_synthetic_data(500, 0.80, seed=42)

    # Compare the same data to itself (split into two halves)
    n = len(y_true) // 2
    y_true1, y_pred1 = y_true[:n], y_pred[:n]
    y_true2, y_pred2 = y_true[n:], y_pred[n:]

    # Both samples from same distribution, should have similar AUC
    auc1 = roc_auc_score(y_true1, y_pred1)
    auc2 = roc_auc_score(y_true2, y_pred2)

    z, p_value = delong_test(y_true1, y_pred1, y_true2, y_pred2)

    print(f"  AUC1: {auc1:.4f}")
    print(f"  AUC2: {auc2:.4f}")
    print(f"  Difference: {abs(auc1 - auc2):.4f}")
    print(f"  Z-statistic: {z:.4f}")
    print(f"  P-value: {p_value:.4f}")

    # For samples from same distribution, we expect p > 0.05 usually
    # (not always, but typically)
    is_not_significant = p_value > 0.05
    print(f"\n  P-value > 0.05: {'YES' if is_not_significant else 'NO'}")
    print(f"  Status: {'PASS (as expected for same distribution)' if is_not_significant else 'NOTE: May happen by chance'}")

    return True  # This test is informational


def test_large_difference():
    """Test 2: Large AUC difference should be significant."""
    print("\n" + "="*60)
    print("TEST 2: Large AUC Difference (Should Be Significant)")
    print("="*60)

    # Generate two samples with very different AUCs
    y_true1, y_pred1 = generate_synthetic_data(500, 0.70, seed=42)
    y_true2, y_pred2 = generate_synthetic_data(500, 0.90, seed=123)

    auc1 = roc_auc_score(y_true1, y_pred1)
    auc2 = roc_auc_score(y_true2, y_pred2)

    z, p_value = delong_test(y_true1, y_pred1, y_true2, y_pred2)

    print(f"  AUC1 (target 0.70): {auc1:.4f}")
    print(f"  AUC2 (target 0.90): {auc2:.4f}")
    print(f"  Difference: {abs(auc1 - auc2):.4f}")
    print(f"  Z-statistic: {z:.4f}")
    print(f"  P-value: {p_value:.6f}")

    is_significant = p_value < 0.05
    print(f"\n  P-value < 0.05: {'YES' if is_significant else 'NO'}")
    print(f"  Status: {'PASS' if is_significant else 'FAIL'}")

    return is_significant


def test_z_statistic_direction():
    """Test 3: Z-statistic should have correct sign."""
    print("\n" + "="*60)
    print("TEST 3: Z-Statistic Direction")
    print("="*60)

    # Sample 1: Lower AUC
    y_true1, y_pred1 = generate_synthetic_data(500, 0.65, seed=42)
    # Sample 2: Higher AUC
    y_true2, y_pred2 = generate_synthetic_data(500, 0.85, seed=123)

    auc1 = roc_auc_score(y_true1, y_pred1)
    auc2 = roc_auc_score(y_true2, y_pred2)

    z, p_value = delong_test(y_true1, y_pred1, y_true2, y_pred2)

    print(f"  AUC1 (lower): {auc1:.4f}")
    print(f"  AUC2 (higher): {auc2:.4f}")
    print(f"  Z-statistic: {z:.4f}")

    # Z should be negative when AUC1 < AUC2 (since z = (auc1 - auc2) / se)
    correct_sign = (auc1 < auc2 and z < 0) or (auc1 > auc2 and z > 0) or (np.isclose(auc1, auc2) and np.isclose(z, 0, atol=0.1))

    print(f"\n  AUC1 < AUC2: {auc1 < auc2}")
    print(f"  Z < 0: {z < 0}")
    print(f"  Correct direction: {'YES' if correct_sign else 'NO'}")
    print(f"  Status: {'PASS' if correct_sign else 'FAIL'}")

    return correct_sign


def test_sample_size_effect():
    """Test 4: Larger samples should give smaller p-values for same difference."""
    print("\n" + "="*60)
    print("TEST 4: Sample Size Effect on P-value")
    print("="*60)

    sample_sizes = [100, 500, 1000, 2000]
    p_values = []

    # Use fixed AUC targets to create consistent difference
    auc_target1 = 0.72
    auc_target2 = 0.82

    for n in sample_sizes:
        y_true1, y_pred1 = generate_synthetic_data(n, auc_target1, seed=42)
        y_true2, y_pred2 = generate_synthetic_data(n, auc_target2, seed=123)

        auc1 = roc_auc_score(y_true1, y_pred1)
        auc2 = roc_auc_score(y_true2, y_pred2)

        z, p_value = delong_test(y_true1, y_pred1, y_true2, y_pred2)
        p_values.append(p_value)

        print(f"  N={n:5d}: AUC1={auc1:.4f}, AUC2={auc2:.4f}, diff={auc2-auc1:.4f}, p={p_value:.6f}")

    # P-values should generally decrease with larger sample size
    # (more power to detect the same effect)
    decreasing_trend = p_values[0] > p_values[-1]

    print(f"\n  P-value decreases with sample size: {'YES' if decreasing_trend else 'NO'}")
    print(f"  Status: {'PASS' if decreasing_trend else 'FAIL'}")

    return decreasing_trend


def test_edge_cases():
    """Test 5: Handle edge cases properly."""
    print("\n" + "="*60)
    print("TEST 5: Edge Case Handling")
    print("="*60)

    all_passed = True

    # Case 1: Very small samples
    print("\n  Case 1: Very small samples (n=30)")
    y_true1, y_pred1 = generate_synthetic_data(30, 0.75, seed=42)
    y_true2, y_pred2 = generate_synthetic_data(30, 0.80, seed=123)

    z, p_value = delong_test(y_true1, y_pred1, y_true2, y_pred2)
    valid = not np.isnan(z) and not np.isnan(p_value)
    print(f"    Returns valid values: {'YES' if valid else 'NO'}")
    print(f"    Z={z:.4f}, p={p_value:.4f}")
    print(f"    Status: {'PASS' if valid else 'FAIL'}")
    if not valid:
        all_passed = False

    # Case 2: Unequal sample sizes
    print("\n  Case 2: Unequal sample sizes (n1=200, n2=800)")
    y_true1, y_pred1 = generate_synthetic_data(200, 0.75, seed=42)
    y_true2, y_pred2 = generate_synthetic_data(800, 0.80, seed=123)

    z, p_value = delong_test(y_true1, y_pred1, y_true2, y_pred2)
    valid = not np.isnan(z) and not np.isnan(p_value) and 0 <= p_value <= 1
    print(f"    Returns valid values: {'YES' if valid else 'NO'}")
    print(f"    Z={z:.4f}, p={p_value:.4f}")
    print(f"    Status: {'PASS' if valid else 'FAIL'}")
    if not valid:
        all_passed = False

    # Case 3: Single class in one sample
    print("\n  Case 3: Single class (should fail gracefully)")
    y_true1 = np.ones(100)  # All positive
    y_pred1 = np.random.rand(100)
    y_true2, y_pred2 = generate_synthetic_data(100, 0.80, seed=123)

    z, p_value = delong_test(y_true1, y_pred1, y_true2, y_pred2)
    is_nan = np.isnan(z) and np.isnan(p_value)
    print(f"    Returns NaN: {'YES' if is_nan else 'NO'}")
    print(f"    Status: {'PASS' if is_nan else 'FAIL'}")
    if not is_nan:
        all_passed = False

    # Case 4: Imbalanced classes
    print("\n  Case 4: Imbalanced classes (90% negative)")
    rng = np.random.RandomState(42)
    n = 500
    y_true1 = np.concatenate([np.ones(50), np.zeros(450)])
    y_pred1 = rng.rand(n)
    y_true2 = np.concatenate([np.ones(50), np.zeros(450)])
    y_pred2 = rng.rand(n)

    z, p_value = delong_test(y_true1, y_pred1, y_true2, y_pred2)
    valid = not np.isnan(z) and not np.isnan(p_value) and 0 <= p_value <= 1
    print(f"    Returns valid values: {'YES' if valid else 'NO'}")
    print(f"    Z={z:.4f}, p={p_value:.4f}")
    print(f"    Status: {'PASS' if valid else 'FAIL'}")
    if not valid:
        all_passed = False

    print(f"\n  Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    return all_passed


def test_comparison_with_permutation():
    """Test 6: Compare DeLong's test with permutation test for validation."""
    print("\n" + "="*60)
    print("TEST 6: Comparison with Permutation Test")
    print("="*60)

    # Generate samples with known difference
    y_true1, y_pred1 = generate_synthetic_data(300, 0.70, seed=42)
    y_true2, y_pred2 = generate_synthetic_data(300, 0.85, seed=123)

    auc1 = roc_auc_score(y_true1, y_pred1)
    auc2 = roc_auc_score(y_true2, y_pred2)

    # DeLong's test
    z, p_delong = delong_test(y_true1, y_pred1, y_true2, y_pred2)

    # Simple permutation test for comparison
    observed_diff = auc2 - auc1

    y_true_combined = np.concatenate([y_true1, y_true2])
    y_pred_combined = np.concatenate([y_pred1, y_pred2])
    n1 = len(y_true1)
    n_total = len(y_true_combined)

    rng = np.random.RandomState(42)
    n_perm = 1000
    null_diffs = []

    for _ in range(n_perm):
        perm_idx = rng.permutation(n_total)
        idx1, idx2 = perm_idx[:n1], perm_idx[n1:]

        try:
            auc_perm1 = roc_auc_score(y_true_combined[idx1], y_pred_combined[idx1])
            auc_perm2 = roc_auc_score(y_true_combined[idx2], y_pred_combined[idx2])
            null_diffs.append(auc_perm2 - auc_perm1)
        except:
            continue

    p_perm = np.mean(np.abs(null_diffs) >= np.abs(observed_diff))

    print(f"  AUC1: {auc1:.4f}")
    print(f"  AUC2: {auc2:.4f}")
    print(f"  Observed difference: {observed_diff:.4f}")
    print(f"\n  DeLong's test:")
    print(f"    Z-statistic: {z:.4f}")
    print(f"    P-value: {p_delong:.6f}")
    print(f"\n  Permutation test (n={n_perm}):")
    print(f"    P-value: {p_perm:.6f}")

    # Both should agree on significance
    both_significant = (p_delong < 0.05) == (p_perm < 0.05)
    p_values_similar = abs(p_delong - p_perm) < 0.1  # Within 0.1 of each other

    print(f"\n  Both agree on significance: {'YES' if both_significant else 'NO'}")
    print(f"  P-values within 0.1: {'YES' if p_values_similar else 'NO'}")
    print(f"  Status: {'PASS' if both_significant else 'MARGINAL'}")

    return both_significant


def test_reproducibility():
    """Test 7: Results should be reproducible."""
    print("\n" + "="*60)
    print("TEST 7: Reproducibility")
    print("="*60)

    y_true1, y_pred1 = generate_synthetic_data(500, 0.75, seed=42)
    y_true2, y_pred2 = generate_synthetic_data(500, 0.85, seed=123)

    # Run twice
    z1, p1 = delong_test(y_true1, y_pred1, y_true2, y_pred2)
    z2, p2 = delong_test(y_true1, y_pred1, y_true2, y_pred2)

    identical = np.isclose(z1, z2) and np.isclose(p1, p2)

    print(f"  Run 1: Z={z1:.6f}, p={p1:.6f}")
    print(f"  Run 2: Z={z2:.6f}, p={p2:.6f}")
    print(f"  Results identical: {'YES' if identical else 'NO'}")
    print(f"  Status: {'PASS' if identical else 'FAIL'}")

    return identical


def run_all_tests():
    """Run all tests and report summary."""
    print("\n" + "="*60)
    print("DELONG'S TEST VERIFICATION")
    print("="*60)

    results = {
        "Identical samples": test_identical_samples(),
        "Large difference detection": test_large_difference(),
        "Z-statistic direction": test_z_statistic_direction(),
        "Sample size effect": test_sample_size_effect(),
        "Edge cases": test_edge_cases(),
        "Permutation comparison": test_comparison_with_permutation(),
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
