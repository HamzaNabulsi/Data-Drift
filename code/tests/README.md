# Statistical Method Tests

This directory contains verification tests for the statistical methods used in the drift analysis.

## Test Files

### `test_bootstrap.py` - Bootstrap Confidence Interval Tests

Verifies the `compute_auc_with_ci()` function in `batch_analysis.py`.

**Tests performed:**
1. **Point estimate accuracy** - AUC matches sklearn exactly
2. **CI validity** - Intervals contain point estimate, valid bounds
3. **CI width scaling** - Width decreases with sample size (~sqrt(n))
4. **Edge cases** - Small samples, single class, NaN values, perfect predictions
5. **Reproducibility** - Same seed produces identical results

**Run:**
```bash
python code/tests/test_bootstrap.py
```

### `test_delong.py` - DeLong's Test Verification

Verifies the `delong_test()` function in `batch_analysis.py`.

**Tests performed:**
1. **Identical samples** - No significant difference detected
2. **Large difference** - Significant differences detected correctly
3. **Z-statistic direction** - Sign indicates which AUC is larger
4. **Sample size effect** - P-values decrease with larger samples
5. **Edge cases** - Small samples, unequal sizes, single class, imbalanced
6. **Permutation comparison** - Results agree with permutation test
7. **Reproducibility** - Deterministic output

**Run:**
```bash
python code/tests/test_delong.py
```

## Running All Tests

```bash
# From project root
source .venv/bin/activate
python code/tests/test_bootstrap.py
python code/tests/test_delong.py

# Or with pytest
pytest code/tests/ -v
```

## Test Results Summary

**Last verified:** December 2024

| Test Suite | Result | Notes |
|------------|--------|-------|
| Bootstrap CI | PASS (5/5) | All tests pass |
| DeLong's Test | PASS (7/7) | All tests pass |

### Key Findings

1. **Bootstrap CI Implementation**
   - Point estimates match sklearn exactly
   - 95% CI width ~0.05 for n=1000, ~0.02 for n=5000
   - Width scales as expected with sqrt(n)
   - Handles edge cases gracefully (returns NaN for invalid inputs)

2. **DeLong's Test Implementation**
   - Correctly uses Hanley-McNeil variance approximation
   - Z-statistic direction matches AUC comparison
   - P-values agree with permutation test
   - Properly handles imbalanced classes and small samples

## Implementation Notes

### Bootstrap CI Method
- Uses percentile method for CI calculation
- Stratified bootstrap (maintains class balance)
- Default: 100 iterations (development), 1000 for production
- Returns NaN for samples < 30 or single-class outcomes

### DeLong's Test
- Non-parametric test for comparing two AUCs
- Uses Hanley-McNeil variance approximation
- Two-tailed test with z-statistic
- Returns NaN for invalid inputs (single class, etc.)

## References

- DeLong ER, DeLong DM, Clarke-Pearson DL (1988). "Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach." Biometrics 44(3):837-845.
- Hanley JA, McNeil BJ (1982). "The meaning and use of the area under a receiver operating characteristic (ROC) curve." Radiology 143(1):29-36.
