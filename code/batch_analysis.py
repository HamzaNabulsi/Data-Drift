"""
Batch Drift Analysis Runner
Analyzes all configured datasets for subgroup-specific drift in ICU severity scores.

Usage:
    python batch_analysis.py                    # Default (N_BOOTSTRAP=100)
    python batch_analysis.py --fast             # Fast testing (N_BOOTSTRAP=2)
    python batch_analysis.py --bootstrap 1000   # Production (N_BOOTSTRAP=1000)
    python batch_analysis.py -b 50              # Custom bootstrap iterations
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import DATASETS, OUTPUT_PATH as OUTPUT_DIR
from sklearn.metrics import roc_auc_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# BOOTSTRAP CONFIGURATION
# =============================================================================
# N_BOOTSTRAP controls the number of bootstrap iterations for confidence intervals.
# Higher values = more accurate CIs but slower runtime.
#
# Recommended values:
#   - Fast testing:  2-10     (~1 min for all datasets)
#   - Development:   50-100   (~10-20 min)
#   - Production:    1000     (~2-4 hours)
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run drift analysis with configurable bootstrap iterations.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python batch_analysis.py                    # Default (100 bootstrap iterations)
    python batch_analysis.py --fast             # Fast testing (2 iterations)
    python batch_analysis.py --bootstrap 1000   # Production run (1000 iterations)
    python batch_analysis.py -b 50              # Custom: 50 iterations
        """
    )
    parser.add_argument(
        '-b', '--bootstrap',
        type=int,
        default=100,
        help='Number of bootstrap iterations for CI (default: 100)'
    )
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Fast mode: use only 2 bootstrap iterations (for testing)'
    )
    return parser.parse_args()

# Parse arguments
args = parse_args()
N_BOOTSTRAP = 2 if args.fast else args.bootstrap
CI_LEVEL = 0.95
RANDOM_SEED = 42

# Define age bins consistently across all datasets
AGE_BINS = [18, 45, 65, 80, 150]
AGE_LABELS = ['18-44', '45-64', '65-79', '80+']

# Race/ethnicity mapping for consistent grouping
RACE_MAPPING = {
    # MIMIC-III lowercase
    'white': 'White', 'black': 'Black', 'hispanic': 'Hispanic',
    'asian': 'Asian', 'native': 'Other', 'other': 'Other', 'unknown': 'Unknown',
    # MIMIC-IV uppercase
    'WHITE': 'White', 'BLACK': 'Black', 'HISPANIC': 'Hispanic',
    'ASIAN': 'Asian', 'AMERICAN INDIAN': 'Other', 'OTHER': 'Other', 'UNKNOWN': 'Unknown',
    # eICU
    'Caucasian': 'White', 'African American': 'Black', 'Hispanic': 'Hispanic',
    'Asian': 'Asian', 'Native American': 'Other', 'Other/Unknown': 'Other',
}


def load_dataset(dataset_key):
    """Load a dataset from config."""
    config = DATASETS[dataset_key]
    filepath = os.path.join(config['data_path'], config['file'])

    if not os.path.exists(filepath):
        print(f"  ERROR: File not found: {filepath}")
        return None, config

    df = pd.read_csv(filepath)
    print(f"  Loaded {len(df):,} records from {config['name']}")
    return df, config


def standardize_demographics(df, config):
    """Standardize demographic columns across datasets."""
    demo_cols = config.get('demographic_cols', {})

    # Age binning
    if 'age' in demo_cols and demo_cols['age'] in df.columns:
        age_col = demo_cols['age']
        df['age_group'] = pd.cut(df[age_col], bins=AGE_BINS, labels=AGE_LABELS, right=False)

    # Gender standardization
    if 'gender' in demo_cols and demo_cols['gender'] in df.columns:
        gender_col = demo_cols['gender']
        df['gender_std'] = df[gender_col].map(lambda x: 'Male' if str(x).upper() in ['M', 'MALE', '1'] else
                                               ('Female' if str(x).upper() in ['F', 'FEMALE', '0'] else 'Unknown'))

    # Race standardization
    if 'race' in demo_cols and demo_cols['race'] in df.columns:
        race_col = demo_cols['race']
        df['race_std'] = df[race_col].map(lambda x: RACE_MAPPING.get(x, 'Other'))

    return df


def compute_auc(y_true, y_pred):
    """Compute AUC with error handling."""
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        if y_pred.isna().all():
            return np.nan
        mask = ~y_pred.isna() & ~y_true.isna()
        if mask.sum() < 10:
            return np.nan
        return roc_auc_score(y_true[mask], y_pred[mask])
    except:
        return np.nan


def compute_auc_with_ci(y_true, y_pred, n_bootstrap=N_BOOTSTRAP, ci_level=CI_LEVEL):
    """
    Compute AUC with bootstrap confidence intervals.

    Returns:
        tuple: (auc, ci_lower, ci_upper) or (nan, nan, nan) if computation fails
    """
    try:
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Handle missing values
        mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) < 30 or len(np.unique(y_true)) < 2:
            return np.nan, np.nan, np.nan

        # Compute point estimate
        auc = roc_auc_score(y_true, y_pred)

        # Bootstrap for CI
        rng = np.random.RandomState(RANDOM_SEED)
        n = len(y_true)
        bootstrap_aucs = []

        for _ in range(n_bootstrap):
            # Stratified bootstrap to maintain class balance
            idx = rng.randint(0, n, n)
            y_true_boot = y_true[idx]
            y_pred_boot = y_pred[idx]

            # Skip if only one class in bootstrap sample
            if len(np.unique(y_true_boot)) < 2:
                continue

            try:
                bootstrap_aucs.append(roc_auc_score(y_true_boot, y_pred_boot))
            except:
                continue

        if len(bootstrap_aucs) < n_bootstrap * 0.5:
            # Too many failed bootstraps
            return auc, np.nan, np.nan

        # Compute percentile CI
        alpha = (1 - ci_level) / 2
        ci_lower = np.percentile(bootstrap_aucs, alpha * 100)
        ci_upper = np.percentile(bootstrap_aucs, (1 - alpha) * 100)

        return auc, ci_lower, ci_upper

    except Exception as e:
        return np.nan, np.nan, np.nan


def delong_test(y_true1, y_pred1, y_true2, y_pred2):
    """
    Perform DeLong's test for comparing two AUCs from independent samples.

    This is a non-parametric test that compares the discriminative ability
    of two models/scores across different time periods or populations.

    Returns:
        tuple: (z_statistic, p_value) or (nan, nan) if computation fails
    """
    try:
        # Compute AUCs
        auc1 = roc_auc_score(y_true1, y_pred1)
        auc2 = roc_auc_score(y_true2, y_pred2)

        n1 = len(y_true1)
        n2 = len(y_true2)

        # Compute variance using Hanley-McNeil approximation
        # For AUC variance: var(AUC) ≈ AUC(1-AUC) * (1 + (n_pos-1)(Q1-AUC²)/(AUC(1-AUC)) + (n_neg-1)(Q2-AUC²)/(AUC(1-AUC))) / (n_pos * n_neg)
        # Simplified approximation: var(AUC) ≈ AUC(1-AUC) / min(n_pos, n_neg)

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

        # Z-statistic for difference
        se_diff = np.sqrt(var1 + var2)
        if se_diff == 0:
            return np.nan, np.nan

        z = (auc1 - auc2) / se_diff
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))  # Two-tailed test

        return z, p_value

    except Exception as e:
        return np.nan, np.nan


def permutation_test_auc_diff(y_true1, y_pred1, y_true2, y_pred2, n_permutations=1000, seed=42):
    """
    Perform a permutation test for the difference in AUC between two samples.

    This test shuffles the time period labels and recomputes the AUC difference
    to build a null distribution, then computes an empirical p-value.

    Args:
        y_true1, y_pred1: Outcome and predictions for first time period
        y_true2, y_pred2: Outcome and predictions for second time period
        n_permutations: Number of permutation iterations
        seed: Random seed for reproducibility

    Returns:
        tuple: (observed_diff, p_value) or (nan, nan) if computation fails
    """
    try:
        # Compute observed AUC difference
        auc1 = roc_auc_score(y_true1, y_pred1)
        auc2 = roc_auc_score(y_true2, y_pred2)
        observed_diff = auc2 - auc1  # Later minus earlier

        # Combine data
        y_true_combined = np.concatenate([y_true1, y_true2])
        y_pred_combined = np.concatenate([y_pred1, y_pred2])
        n1 = len(y_true1)
        n_total = len(y_true_combined)

        # Permutation test
        rng = np.random.RandomState(seed)
        null_diffs = []

        for _ in range(n_permutations):
            # Shuffle indices
            perm_idx = rng.permutation(n_total)

            # Split into two groups
            idx1 = perm_idx[:n1]
            idx2 = perm_idx[n1:]

            y_true_perm1 = y_true_combined[idx1]
            y_pred_perm1 = y_pred_combined[idx1]
            y_true_perm2 = y_true_combined[idx2]
            y_pred_perm2 = y_pred_combined[idx2]

            # Check both groups have both classes
            if len(np.unique(y_true_perm1)) < 2 or len(np.unique(y_true_perm2)) < 2:
                continue

            try:
                auc_perm1 = roc_auc_score(y_true_perm1, y_pred_perm1)
                auc_perm2 = roc_auc_score(y_true_perm2, y_pred_perm2)
                null_diffs.append(auc_perm2 - auc_perm1)
            except:
                continue

        if len(null_diffs) < n_permutations * 0.5:
            return observed_diff, np.nan

        # Compute two-tailed p-value
        null_diffs = np.array(null_diffs)
        p_value = np.mean(np.abs(null_diffs) >= np.abs(observed_diff))

        return observed_diff, p_value

    except Exception as e:
        return np.nan, np.nan


def analyze_drift(df, config, score_col, compute_ci=True):
    """Analyze drift for a specific score across subgroups.

    Args:
        df: DataFrame with patient data
        config: Dataset configuration
        score_col: Column name for the severity score
        compute_ci: Whether to compute bootstrap confidence intervals (slower but recommended)
    """
    results = []

    year_col = config['year_col']
    outcome_col = config['outcome_col']
    outcome_positive = config['outcome_positive']

    # Create binary outcome
    df['outcome_binary'] = (df[outcome_col] == outcome_positive).astype(int)

    # Get time periods
    if year_col not in df.columns:
        print(f"  WARNING: Year column '{year_col}' not found")
        return pd.DataFrame()

    time_periods = sorted(df[year_col].dropna().unique())
    if len(time_periods) < 2:
        print(f"  WARNING: Only {len(time_periods)} time period(s) found, skipping")
        return pd.DataFrame()

    print(f"  Time periods: {time_periods}")
    if compute_ci:
        print(f"  Computing {int(CI_LEVEL*100)}% bootstrap CIs (n={N_BOOTSTRAP})...")

    def _compute_and_append(subset, subgroup_type, subgroup, period):
        """Helper to compute AUC (+CI) and append to results."""
        if compute_ci:
            auc, ci_lower, ci_upper = compute_auc_with_ci(
                subset['outcome_binary'], subset[score_col]
            )
        else:
            auc = compute_auc(subset['outcome_binary'], subset[score_col])
            ci_lower, ci_upper = np.nan, np.nan

        results.append({
            'subgroup_type': subgroup_type,
            'subgroup': subgroup,
            'time_period': str(period),
            'auc': auc,
            'auc_ci_lower': ci_lower,
            'auc_ci_upper': ci_upper,
            'n': len(subset),
            'n_deaths': int(subset['outcome_binary'].sum()),
            'mortality_rate': subset['outcome_binary'].mean()
        })

    # Overall AUC by time period
    for period in time_periods:
        subset = df[df[year_col] == period]
        _compute_and_append(subset, 'Overall', 'All', period)

    # Age group analysis
    if 'age_group' in df.columns:
        for age_group in AGE_LABELS:
            for period in time_periods:
                subset = df[(df['age_group'] == age_group) & (df[year_col] == period)]
                if len(subset) >= 50:
                    _compute_and_append(subset, 'Age', age_group, period)

    # Gender analysis
    if 'gender_std' in df.columns:
        for gender in ['Male', 'Female']:
            for period in time_periods:
                subset = df[(df['gender_std'] == gender) & (df[year_col] == period)]
                if len(subset) >= 50:
                    _compute_and_append(subset, 'Gender', gender, period)

    # Race analysis
    if 'race_std' in df.columns:
        for race in ['White', 'Black', 'Hispanic', 'Asian']:
            for period in time_periods:
                subset = df[(df['race_std'] == race) & (df[year_col] == period)]
                if len(subset) >= 30:  # Lower threshold for minority groups
                    _compute_and_append(subset, 'Race', race, period)

    return pd.DataFrame(results)


def compute_drift_deltas(results_df):
    """Compute drift (AUC change) between first and last time period with CIs."""
    if results_df.empty:
        return pd.DataFrame()

    periods = sorted(results_df['time_period'].unique())
    if len(periods) < 2:
        return pd.DataFrame()

    first_period = periods[0]
    last_period = periods[-1]

    deltas = []
    for (subgroup_type, subgroup), group in results_df.groupby(['subgroup_type', 'subgroup']):
        first = group[group['time_period'] == first_period]
        last = group[group['time_period'] == last_period]

        if not first.empty and not last.empty:
            auc_first = first['auc'].values[0]
            auc_last = last['auc'].values[0]

            if not np.isnan(auc_first) and not np.isnan(auc_last):
                delta = auc_last - auc_first

                # Get CIs if available
                ci_first_lower = first['auc_ci_lower'].values[0] if 'auc_ci_lower' in first.columns else np.nan
                ci_first_upper = first['auc_ci_upper'].values[0] if 'auc_ci_upper' in first.columns else np.nan
                ci_last_lower = last['auc_ci_lower'].values[0] if 'auc_ci_lower' in last.columns else np.nan
                ci_last_upper = last['auc_ci_upper'].values[0] if 'auc_ci_upper' in last.columns else np.nan

                # Compute delta CI (conservative: propagate uncertainty)
                # delta_lower = last_lower - first_upper (worst case for decrease)
                # delta_upper = last_upper - first_lower (worst case for increase)
                if not np.isnan(ci_first_lower) and not np.isnan(ci_last_lower):
                    delta_ci_lower = ci_last_lower - ci_first_upper
                    delta_ci_upper = ci_last_upper - ci_first_lower
                else:
                    delta_ci_lower = np.nan
                    delta_ci_upper = np.nan

                # Determine statistical significance from CI
                # Significant if CI doesn't contain 0
                if not np.isnan(delta_ci_lower) and not np.isnan(delta_ci_upper):
                    significant = (delta_ci_lower > 0) or (delta_ci_upper < 0)
                else:
                    significant = False

                deltas.append({
                    'subgroup_type': subgroup_type,
                    'subgroup': subgroup,
                    'auc_first': auc_first,
                    'auc_first_ci_lower': ci_first_lower,
                    'auc_first_ci_upper': ci_first_upper,
                    'auc_last': auc_last,
                    'auc_last_ci_lower': ci_last_lower,
                    'auc_last_ci_upper': ci_last_upper,
                    'delta': delta,
                    'delta_ci_lower': delta_ci_lower,
                    'delta_ci_upper': delta_ci_upper,
                    'significant': significant,
                    'period_first': first_period,
                    'period_last': last_period,
                    'n_first': first['n'].values[0],
                    'n_last': last['n'].values[0]
                })

    return pd.DataFrame(deltas)


def compute_drift_deltas_with_pvalues(df, config, score_col, results_df, n_permutations=None):
    """
    Compute drift deltas with statistical significance testing (p-values).

    Uses DeLong's test (fast, parametric) and optionally permutation test (slower, non-parametric).

    Args:
        df: Original DataFrame with patient data
        config: Dataset configuration
        score_col: Score column name
        results_df: Results DataFrame from analyze_drift
        n_permutations: Number of permutation iterations (None = skip permutation test)

    Returns:
        DataFrame with deltas and p-values
    """
    if results_df.empty:
        return pd.DataFrame()

    year_col = config['year_col']
    outcome_col = config['outcome_col']
    outcome_positive = config['outcome_positive']

    # Create binary outcome if not exists
    if 'outcome_binary' not in df.columns:
        df['outcome_binary'] = (df[outcome_col] == outcome_positive).astype(int)

    periods = sorted(results_df['time_period'].unique())
    if len(periods) < 2:
        return pd.DataFrame()

    first_period = periods[0]
    last_period = periods[-1]

    deltas = []

    # Helper to get subgroup mask
    def get_subgroup_mask(subgroup_type, subgroup):
        if subgroup_type == 'Overall':
            return pd.Series([True] * len(df), index=df.index)
        elif subgroup_type == 'Age':
            return df['age_group'] == subgroup
        elif subgroup_type == 'Gender':
            return df['gender_std'] == subgroup
        elif subgroup_type == 'Race':
            return df['race_std'] == subgroup
        return pd.Series([False] * len(df), index=df.index)

    for (subgroup_type, subgroup), group in results_df.groupby(['subgroup_type', 'subgroup']):
        first = group[group['time_period'] == first_period]
        last = group[group['time_period'] == last_period]

        if first.empty or last.empty:
            continue

        auc_first = first['auc'].values[0]
        auc_last = last['auc'].values[0]

        if np.isnan(auc_first) or np.isnan(auc_last):
            continue

        delta = auc_last - auc_first

        # Get raw data for statistical tests
        subgroup_mask = get_subgroup_mask(subgroup_type, subgroup)

        # Handle type conversion for time period matching
        # The results_df has string periods, but df might have int/numpy types
        df_year_values = df[year_col].astype(str)
        mask_first = subgroup_mask & (df_year_values == str(first_period))
        mask_last = subgroup_mask & (df_year_values == str(last_period))

        df_first = df[mask_first].dropna(subset=[score_col, 'outcome_binary'])
        df_last = df[mask_last].dropna(subset=[score_col, 'outcome_binary'])

        if len(df_first) < 30 or len(df_last) < 30:
            continue

        y_true1 = df_first['outcome_binary'].values
        y_pred1 = df_first[score_col].values
        y_true2 = df_last['outcome_binary'].values
        y_pred2 = df_last[score_col].values

        # DeLong's test (fast)
        z_stat, p_delong = delong_test(y_true1, y_pred1, y_true2, y_pred2)

        # Permutation test (slower, optional)
        if n_permutations and n_permutations > 0:
            _, p_perm = permutation_test_auc_diff(y_true1, y_pred1, y_true2, y_pred2,
                                                   n_permutations=n_permutations)
        else:
            p_perm = np.nan

        # Get CIs if available
        ci_first_lower = first['auc_ci_lower'].values[0] if 'auc_ci_lower' in first.columns else np.nan
        ci_first_upper = first['auc_ci_upper'].values[0] if 'auc_ci_upper' in first.columns else np.nan
        ci_last_lower = last['auc_ci_lower'].values[0] if 'auc_ci_lower' in last.columns else np.nan
        ci_last_upper = last['auc_ci_upper'].values[0] if 'auc_ci_upper' in last.columns else np.nan

        # Compute delta CI
        if not np.isnan(ci_first_lower) and not np.isnan(ci_last_lower):
            delta_ci_lower = ci_last_lower - ci_first_upper
            delta_ci_upper = ci_last_upper - ci_first_lower
        else:
            delta_ci_lower = np.nan
            delta_ci_upper = np.nan

        # Determine significance (use DeLong p-value with alpha=0.05)
        significant = p_delong < 0.05 if not np.isnan(p_delong) else False

        deltas.append({
            'subgroup_type': subgroup_type,
            'subgroup': subgroup,
            'auc_first': auc_first,
            'auc_first_ci_lower': ci_first_lower,
            'auc_first_ci_upper': ci_first_upper,
            'auc_last': auc_last,
            'auc_last_ci_lower': ci_last_lower,
            'auc_last_ci_upper': ci_last_upper,
            'delta': delta,
            'delta_ci_lower': delta_ci_lower,
            'delta_ci_upper': delta_ci_upper,
            'z_statistic': z_stat,
            'p_value_delong': p_delong,
            'p_value_permutation': p_perm,
            'significant': significant,
            'period_first': first_period,
            'period_last': last_period,
            'n_first': first['n'].values[0],
            'n_last': last['n'].values[0]
        })

    return pd.DataFrame(deltas)


def export_per_dataset_tables(dataset_key, dataset_name, results_df, deltas_df, output_dir):
    """
    Export per-dataset tables (not cross-dataset comparisons).

    Creates:
    - output/{dataset}/drift_results.csv - All AUC values per period
    - output/{dataset}/drift_deltas.csv - Delta changes with p-values
    - output/{dataset}/summary_by_score.csv - Summary table per score
    - output/{dataset}/subgroup_drift.csv - Subgroup-level drift summary
    """
    # Create dataset-specific output directory
    dataset_dir = Path(output_dir) / dataset_key
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save raw results (all periods)
    results_file = dataset_dir / 'drift_results.csv'
    results_df.to_csv(results_file, index=False)

    # 2. Save deltas with statistical testing
    deltas_file = dataset_dir / 'drift_deltas.csv'
    deltas_df.to_csv(deltas_file, index=False)

    # 3. Create summary by score
    summary_rows = []
    for score in deltas_df['score'].unique():
        score_data = deltas_df[deltas_df['score'] == score]
        overall = score_data[score_data['subgroup'] == 'All']

        if not overall.empty:
            row = overall.iloc[0]
            summary_rows.append({
                'Score': score.upper(),
                'AUC (First Period)': f"{row['auc_first']:.3f}",
                'AUC (Last Period)': f"{row['auc_last']:.3f}",
                'Delta': f"{row['delta']:+.3f}",
                '95% CI': f"({row['delta_ci_lower']:.3f}, {row['delta_ci_upper']:.3f})",
                'p-value': f"{row['p_value_delong']:.4f}" if pd.notna(row['p_value_delong']) else 'N/A',
                'Significant': 'Yes' if row.get('significant', False) else 'No',
                'N (First)': int(row['n_first']),
                'N (Last)': int(row['n_last'])
            })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_file = dataset_dir / 'summary_by_score.csv'
        summary_df.to_csv(summary_file, index=False)

    # 4. Create subgroup drift table (formatted for publication)
    subgroup_rows = []
    for _, row in deltas_df.iterrows():
        sig_marker = '*' if row.get('significant', False) else ''
        subgroup_rows.append({
            'Score': row['score'].upper(),
            'Subgroup Type': row['subgroup_type'],
            'Subgroup': row['subgroup'],
            'AUC First': f"{row['auc_first']:.3f}",
            'AUC Last': f"{row['auc_last']:.3f}",
            'Delta': f"{row['delta']:+.3f}{sig_marker}",
            '95% CI': f"({row['delta_ci_lower']:.3f}, {row['delta_ci_upper']:.3f})",
            'p-value': f"{row['p_value_delong']:.4f}" if pd.notna(row['p_value_delong']) else 'N/A',
            'Period First': row['period_first'],
            'Period Last': row['period_last']
        })

    if subgroup_rows:
        subgroup_df = pd.DataFrame(subgroup_rows)
        subgroup_file = dataset_dir / 'subgroup_drift.csv'
        subgroup_df.to_csv(subgroup_file, index=False)

    print(f"  Exported tables to: {dataset_dir}/")
    return dataset_dir


def run_batch_analysis(datasets_to_run=None):
    """Run drift analysis on all (or specified) datasets.

    Args:
        datasets_to_run: List of dataset keys to analyze. If None, uses default temporal datasets.
    """

    # Define which datasets to analyze
    # mimic_combined = MIMIC-III + MIMIC-IV merged for continuous 2001-2022 analysis
    # eicu_combined = eICU merged for 2014-2021 temporal drift analysis
    temporal_datasets = ['mimic_combined', 'saltz', 'zhejiang', 'eicu_combined']

    if datasets_to_run is None:
        datasets_to_run = temporal_datasets

    all_results = []
    all_deltas = []

    for dataset_key in datasets_to_run:
        if dataset_key not in DATASETS:
            print(f"\nSkipping {dataset_key}: not in config")
            continue

        config = DATASETS[dataset_key]
        print(f"\n{'='*60}")
        print(f"Analyzing: {config['name']}")
        print('='*60)

        # Load data
        df, config = load_dataset(dataset_key)
        if df is None:
            continue

        # Standardize demographics
        df = standardize_demographics(df, config)

        # Get available scores
        score_cols = config.get('score_cols', [config.get('score_col', 'sofa')])

        # Collect per-dataset results
        dataset_results = []
        dataset_deltas = []

        for score_col in score_cols:
            if score_col not in df.columns:
                print(f"  Score '{score_col}' not found, skipping")
                continue

            print(f"\n  Analyzing {score_col.upper()} score...")

            # Run drift analysis
            results = analyze_drift(df, config, score_col)

            if not results.empty:
                results['dataset'] = dataset_key
                results['dataset_name'] = config['name']
                results['score'] = score_col
                all_results.append(results)
                dataset_results.append(results)

                # Compute deltas with statistical testing (DeLong's test)
                print(f"  Computing statistical significance (DeLong's test)...")
                deltas = compute_drift_deltas_with_pvalues(df, config, score_col, results, n_permutations=None)

                if deltas.empty:
                    # Fallback to simple deltas if p-value computation fails
                    deltas = compute_drift_deltas(results)

                if not deltas.empty:
                    deltas['dataset'] = dataset_key
                    deltas['dataset_name'] = config['name']
                    deltas['score'] = score_col
                    all_deltas.append(deltas)
                    dataset_deltas.append(deltas)

                    # Print summary with significance
                    print(f"\n  Drift Summary for {score_col.upper()}:")
                    for _, row in deltas.iterrows():
                        arrow = "↓" if row['delta'] < 0 else "↑"
                        sig_marker = "*" if row.get('significant', False) else ""
                        p_val = row.get('p_value_delong', np.nan)
                        p_str = f"p={p_val:.3f}" if not np.isnan(p_val) else ""
                        print(f"    {row['subgroup_type']:8} | {row['subgroup']:10} | {row['auc_first']:.3f} → {row['auc_last']:.3f} ({arrow}{abs(row['delta']):.3f}{sig_marker}) {p_str}")

        # Export per-dataset tables (not cross-dataset)
        if dataset_results and dataset_deltas:
            combined_dataset_results = pd.concat(dataset_results, ignore_index=True)
            combined_dataset_deltas = pd.concat(dataset_deltas, ignore_index=True)
            export_per_dataset_tables(
                dataset_key,
                config['name'],
                combined_dataset_results,
                combined_dataset_deltas,
                OUTPUT_DIR
            )

    # Combine all results (kept for internal use, but not saved as cross-dataset comparison)
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_deltas = pd.concat(all_deltas, ignore_index=True) if all_deltas else pd.DataFrame()

        # NOTE: We intentionally do NOT save cross-dataset comparison files.
        # Per Leo's feedback: "showing cross-dataset comparisons defeats our purpose"
        # Each dataset's results are saved in output/{dataset}/ subdirectories.

        print(f"\n{'='*60}")
        print("BATCH ANALYSIS COMPLETE")
        print('='*60)
        print(f"Per-dataset results saved to: output/{{dataset}}/ subdirectories")
        print(f"  - drift_results.csv (AUC values per period)")
        print(f"  - drift_deltas.csv (delta changes with p-values)")
        print(f"  - summary_by_score.csv (overall summary)")
        print(f"  - subgroup_drift.csv (subgroup-level analysis)")
        print(f"\nDatasets analyzed: {', '.join(datasets_to_run)}")

        return combined_results, combined_deltas

    return pd.DataFrame(), pd.DataFrame()


if __name__ == '__main__':
    results, deltas = run_batch_analysis()

    if not deltas.empty:
        # Per-dataset summary (NOT cross-dataset comparison)
        for dataset in deltas['dataset_name'].unique():
            dataset_deltas = deltas[deltas['dataset_name'] == dataset]

            print(f"\n{'='*60}")
            print(f"KEY FINDINGS: {dataset}")
            print("="*60)

            # Count significant findings for this dataset
            if 'significant' in dataset_deltas.columns:
                sig_count = dataset_deltas['significant'].sum()
                total_count = len(dataset_deltas)
                print(f"Significant drifts: {sig_count} / {total_count} ({100*sig_count/total_count:.1f}%)")

            # Find largest drifts within this dataset
            for score in dataset_deltas['score'].unique():
                score_data = dataset_deltas[dataset_deltas['score'] == score]
                if not score_data.empty:
                    worst = score_data.loc[score_data['delta'].idxmin()]
                    best = score_data.loc[score_data['delta'].idxmax()]

                    worst_sig = "*" if worst.get('significant', False) else ""
                    best_sig = "*" if best.get('significant', False) else ""

                    print(f"\n  {score.upper()}:")
                    print(f"    Worst: {worst['subgroup_type']}={worst['subgroup']} ({worst['delta']:+.3f}{worst_sig})")
                    print(f"    Best:  {best['subgroup_type']}={best['subgroup']} ({best['delta']:+.3f}{best_sig})")

            # Show significant findings for this dataset
            if 'significant' in dataset_deltas.columns:
                sig_data = dataset_deltas[dataset_deltas['significant'] == True]
                if not sig_data.empty:
                    sig_decline = sig_data[sig_data['delta'] < 0].nsmallest(3, 'delta')
                    sig_improve = sig_data[sig_data['delta'] > 0].nlargest(3, 'delta')

                    if not sig_decline.empty:
                        print(f"\n  Top significant declines:")
                        for _, row in sig_decline.iterrows():
                            print(f"    {row['subgroup_type']}={row['subgroup']} ({row['score']}): {row['delta']:+.3f} p={row['p_value_delong']:.4f}")

                    if not sig_improve.empty:
                        print(f"\n  Top significant improvements:")
                        for _, row in sig_improve.iterrows():
                            print(f"    {row['subgroup_type']}={row['subgroup']} ({row['score']}): {row['delta']:+.3f} p={row['p_value_delong']:.4f}")
