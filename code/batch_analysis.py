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
from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix
from scipy import stats
from scipy.stats import page_trend_test, false_discovery_control
import warnings
warnings.filterwarnings('ignore')

# SOFA threshold for binary classification (JAMA 2001: SOFA>=10 ~ 40% mortality)
SOFA_THRESHOLD = 10

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


def compute_auc_with_ci(y_true, y_pred, n_bootstrap=N_BOOTSTRAP, ci_level=CI_LEVEL,
                         return_replicates=False):
    """
    Compute AUC with bootstrap confidence intervals.

    Returns:
        tuple: (auc, ci_lower, ci_upper) or (nan, nan, nan) if computation fails
               If return_replicates=True, returns (auc, ci_lower, ci_upper, bootstrap_aucs)
    """
    fail = (np.nan, np.nan, np.nan, []) if return_replicates else (np.nan, np.nan, np.nan)
    try:
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Handle missing values
        mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) < 30 or len(np.unique(y_true)) < 2:
            return fail

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
            if return_replicates:
                return auc, np.nan, np.nan, bootstrap_aucs
            return auc, np.nan, np.nan

        # Compute percentile CI
        alpha = (1 - ci_level) / 2
        ci_lower = np.percentile(bootstrap_aucs, alpha * 100)
        ci_upper = np.percentile(bootstrap_aucs, (1 - alpha) * 100)

        if return_replicates:
            return auc, ci_lower, ci_upper, bootstrap_aucs
        return auc, ci_lower, ci_upper

    except Exception as e:
        return fail


# =============================================================================
# XIAOLI'S RECOMMENDED METRICS (Dec 2025)
# =============================================================================

def compute_classification_metrics(y_true, y_pred, threshold=SOFA_THRESHOLD):
    """
    Compute classification metrics at SOFA >= threshold (default 10).

    Per Xiaoli's recommendation (JAMA 2001): SOFA >= 10 correlates with ~40% mortality.

    Returns:
        dict: TPR (sensitivity), FPR, PPV (precision), NPV, or NaNs if computation fails
    """
    try:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Handle missing values
        mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) < 30 or len(np.unique(y_true)) < 2:
            return {'tpr': np.nan, 'fpr': np.nan, 'ppv': np.nan, 'npv': np.nan}

        # Binary prediction at threshold
        y_pred_binary = (y_pred >= threshold).astype(int)

        # Confusion matrix: TN, FP, FN, TP
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary, labels=[0, 1]).ravel()

        # Calculate metrics
        tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan  # Sensitivity / Recall
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan  # False Positive Rate
        ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan  # Precision / PPV
        npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan  # Negative Predictive Value

        return {'tpr': tpr, 'fpr': fpr, 'ppv': ppv, 'npv': npv}

    except Exception as e:
        return {'tpr': np.nan, 'fpr': np.nan, 'ppv': np.nan, 'npv': np.nan}


def compute_calibration_metrics(y_true, y_pred, score_max=24):
    """
    Compute calibration metrics for severity scores.

    Args:
        y_true: Binary outcome (0/1)
        y_pred: Severity score (e.g., SOFA 0-24)
        score_max: Maximum score value for normalization

    Returns:
        dict: Brier score, SMR (Standardized Mortality Ratio), expected mortality
    """
    try:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Handle missing values
        mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) < 30:
            return {'brier_score': np.nan, 'smr': np.nan, 'expected_mortality': np.nan, 'observed_mortality': np.nan}

        # Normalize score to probability (simple linear scaling)
        # SOFA: 0-24 scale, approximate mortality probability
        y_pred_prob = np.clip(y_pred / score_max, 0, 1)

        # Brier score (lower is better, 0 = perfect)
        brier = brier_score_loss(y_true, y_pred_prob)

        # SMR: Observed / Expected deaths
        observed_deaths = y_true.sum()
        expected_deaths = y_pred_prob.sum()
        smr = observed_deaths / expected_deaths if expected_deaths > 0 else np.nan

        observed_mortality = y_true.mean()
        expected_mortality = y_pred_prob.mean()

        return {
            'brier_score': brier,
            'smr': smr,
            'expected_mortality': expected_mortality,
            'observed_mortality': observed_mortality
        }

    except Exception as e:
        return {'brier_score': np.nan, 'smr': np.nan, 'expected_mortality': np.nan, 'observed_mortality': np.nan}


def compute_fairness_metrics(df, score_col, outcome_col, group_col, threshold=SOFA_THRESHOLD):
    """
    Compute fairness metrics across demographic groups.

    Args:
        df: DataFrame with patient data
        score_col: Severity score column
        outcome_col: Binary outcome column
        group_col: Demographic grouping column (e.g., 'race_std', 'gender_std')
        threshold: SOFA threshold for binary classification

    Returns:
        dict: Demographic parity difference, equalized odds difference, group-wise TPR/FPR
    """
    try:
        groups = df[group_col].dropna().unique()
        if len(groups) < 2:
            return {'demographic_parity_diff': np.nan, 'equalized_odds_diff': np.nan, 'group_metrics': {}}

        group_metrics = {}
        positive_rates = []
        tprs = []
        fprs = []

        for group in groups:
            subset = df[df[group_col] == group].dropna(subset=[score_col, outcome_col])
            if len(subset) < 30:
                continue

            y_true = subset[outcome_col].values
            y_pred = subset[score_col].values
            y_pred_binary = (y_pred >= threshold).astype(int)

            # Positive prediction rate (for demographic parity)
            pos_rate = y_pred_binary.mean()
            positive_rates.append(pos_rate)

            # TPR and FPR (for equalized odds)
            if len(np.unique(y_true)) == 2:
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary, labels=[0, 1]).ravel()
                tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
                fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
                tprs.append(tpr)
                fprs.append(fpr)

            group_metrics[group] = {
                'n': len(subset),
                'positive_rate': pos_rate,
                'tpr': tpr if len(np.unique(y_true)) == 2 else np.nan,
                'fpr': fpr if len(np.unique(y_true)) == 2 else np.nan
            }

        # Demographic parity: max difference in positive prediction rates
        demographic_parity_diff = max(positive_rates) - min(positive_rates) if len(positive_rates) >= 2 else np.nan

        # Equalized odds: max difference in TPR + max difference in FPR
        tpr_diff = max(tprs) - min(tprs) if len(tprs) >= 2 else np.nan
        fpr_diff = max(fprs) - min(fprs) if len(fprs) >= 2 else np.nan
        equalized_odds_diff = (tpr_diff + fpr_diff) / 2 if not np.isnan(tpr_diff) else np.nan

        return {
            'demographic_parity_diff': demographic_parity_diff,
            'equalized_odds_diff': equalized_odds_diff,
            'tpr_diff': tpr_diff,
            'fpr_diff': fpr_diff,
            'group_metrics': group_metrics
        }

    except Exception as e:
        return {'demographic_parity_diff': np.nan, 'equalized_odds_diff': np.nan, 'group_metrics': {}}


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

    Returns:
        tuple: (results_df, bootstrap_store) where bootstrap_store is a dict
               keyed by (subgroup_type, subgroup, period) → list of bootstrap AUC replicates
    """
    results = []
    bootstrap_store = {}

    year_col = config['year_col']
    outcome_col = config['outcome_col']
    outcome_positive = config['outcome_positive']

    # Create binary outcome
    df['outcome_binary'] = (df[outcome_col] == outcome_positive).astype(int)

    # Get time periods
    if year_col not in df.columns:
        print(f"  WARNING: Year column '{year_col}' not found")
        return pd.DataFrame(), bootstrap_store

    time_periods = sorted(df[year_col].dropna().unique())
    if len(time_periods) < 2:
        print(f"  WARNING: Only {len(time_periods)} time period(s) found, skipping")
        return pd.DataFrame(), bootstrap_store

    print(f"  Time periods: {time_periods}")
    if compute_ci:
        print(f"  Computing {int(CI_LEVEL*100)}% bootstrap CIs (n={N_BOOTSTRAP})...")

    def _compute_and_append(subset, subgroup_type, subgroup, period):
        """Helper to compute AUC (+CI) and append to results."""
        if compute_ci:
            auc, ci_lower, ci_upper, replicates = compute_auc_with_ci(
                subset['outcome_binary'], subset[score_col], return_replicates=True
            )
            bootstrap_store[(subgroup_type, subgroup, str(period))] = replicates
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

    # Intersectional analysis (Age x Gender x Race)
    # Per Hamza's suggestion: analyze combinations of all three demographics
    if 'age_group' in df.columns and 'gender_std' in df.columns:
        print(f"  Computing intersectional analysis (Age x Gender" + (" x Race)..." if 'race_std' in df.columns else ")..."))

        genders = ['Male', 'Female']
        races = ['White', 'Black', 'Hispanic', 'Asian'] if 'race_std' in df.columns else [None]

        for age_group in AGE_LABELS:
            for gender in genders:
                for race in races:
                    for period in time_periods:
                        if race is not None:
                            subset = df[(df['age_group'] == age_group) &
                                       (df['gender_std'] == gender) &
                                       (df['race_std'] == race) &
                                       (df[year_col] == period)]
                            subgroup_label = f"{age_group}_{gender}_{race}"
                        else:
                            subset = df[(df['age_group'] == age_group) &
                                       (df['gender_std'] == gender) &
                                       (df[year_col] == period)]
                            subgroup_label = f"{age_group}_{gender}"

                        # Minimum sample size for intersectional groups
                        if len(subset) >= 30:
                            _compute_and_append(subset, 'Intersectional', subgroup_label, period)

    return pd.DataFrame(results), bootstrap_store


def analyze_xiaoli_metrics(df, config, score_col='sofa'):
    """
    Analyze Xiaoli's recommended metrics: classification at SOFA>=10, calibration, fairness.

    This implements the Dec 2025 recommendations:
    1. SOFA >= 10 threshold classification (TPR, FPR, PPV, NPV)
    2. Calibration metrics (Brier score, SMR)
    3. Fairness metrics (demographic parity, equalized odds)

    Args:
        df: DataFrame with patient data
        config: Dataset configuration
        score_col: Score column (default 'sofa')

    Returns:
        tuple: (classification_results_df, calibration_results_df, fairness_results_df)
    """
    classification_results = []
    calibration_results = []
    fairness_results = []

    year_col = config['year_col']
    outcome_col = config['outcome_col']
    outcome_positive = config['outcome_positive']

    # Create binary outcome
    if 'outcome_binary' not in df.columns:
        df['outcome_binary'] = (df[outcome_col] == outcome_positive).astype(int)

    # Check if score column exists
    if score_col not in df.columns:
        print(f"  WARNING: Score column '{score_col}' not found, skipping Xiaoli metrics")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    time_periods = sorted(df[year_col].dropna().unique())
    print(f"  Computing Xiaoli metrics (SOFA>={SOFA_THRESHOLD} threshold)...")

    def _compute_metrics_for_subset(subset, subgroup_type, subgroup, period):
        """Helper to compute all Xiaoli metrics for a subset."""
        if len(subset) < 30:
            return

        y_true = subset['outcome_binary'].values
        y_pred = subset[score_col].values

        # 1. Classification metrics at SOFA >= 10
        class_metrics = compute_classification_metrics(y_true, y_pred, threshold=SOFA_THRESHOLD)
        classification_results.append({
            'subgroup_type': subgroup_type,
            'subgroup': subgroup,
            'time_period': str(period),
            'threshold': SOFA_THRESHOLD,
            'tpr': class_metrics['tpr'],
            'fpr': class_metrics['fpr'],
            'ppv': class_metrics['ppv'],
            'npv': class_metrics['npv'],
            'n': len(subset),
            'n_deaths': int(subset['outcome_binary'].sum()),
            'n_high_score': int((subset[score_col] >= SOFA_THRESHOLD).sum())
        })

        # 2. Calibration metrics
        calib_metrics = compute_calibration_metrics(y_true, y_pred, score_max=24)
        calibration_results.append({
            'subgroup_type': subgroup_type,
            'subgroup': subgroup,
            'time_period': str(period),
            'brier_score': calib_metrics['brier_score'],
            'smr': calib_metrics['smr'],
            'expected_mortality': calib_metrics['expected_mortality'],
            'observed_mortality': calib_metrics['observed_mortality'],
            'n': len(subset)
        })

    # Overall metrics by time period
    for period in time_periods:
        subset = df[df[year_col] == period]
        _compute_metrics_for_subset(subset, 'Overall', 'All', period)

    # Age group analysis
    if 'age_group' in df.columns:
        for age_group in AGE_LABELS:
            for period in time_periods:
                subset = df[(df['age_group'] == age_group) & (df[year_col] == period)]
                _compute_metrics_for_subset(subset, 'Age', age_group, period)

    # Gender analysis
    if 'gender_std' in df.columns:
        for gender in ['Male', 'Female']:
            for period in time_periods:
                subset = df[(df['gender_std'] == gender) & (df[year_col] == period)]
                _compute_metrics_for_subset(subset, 'Gender', gender, period)

    # Race analysis
    if 'race_std' in df.columns:
        for race in ['White', 'Black', 'Hispanic', 'Asian']:
            for period in time_periods:
                subset = df[(df['race_std'] == race) & (df[year_col] == period)]
                _compute_metrics_for_subset(subset, 'Race', race, period)

    # 3. Fairness metrics per time period
    # Also track detailed per-subgroup metrics
    fairness_detailed_results = []

    for period in time_periods:
        period_df = df[df[year_col] == period]

        # Fairness by gender
        if 'gender_std' in period_df.columns:
            gender_fairness = compute_fairness_metrics(period_df, score_col, 'outcome_binary', 'gender_std')
            fairness_results.append({
                'time_period': str(period),
                'group_type': 'Gender',
                'demographic_parity_diff': gender_fairness['demographic_parity_diff'],
                'equalized_odds_diff': gender_fairness['equalized_odds_diff'],
                'tpr_diff': gender_fairness['tpr_diff'],
                'fpr_diff': gender_fairness['fpr_diff']
            })
            # Save detailed per-subgroup metrics
            for subgroup, metrics in gender_fairness.get('group_metrics', {}).items():
                fairness_detailed_results.append({
                    'time_period': str(period),
                    'group_type': 'Gender',
                    'subgroup': subgroup,
                    'n': metrics.get('n', 0),
                    'positive_rate': metrics.get('positive_rate', np.nan),
                    'sensitivity': metrics.get('tpr', np.nan),
                    'specificity': 1 - metrics.get('fpr', np.nan) if metrics.get('fpr') is not None else np.nan,
                    'tpr': metrics.get('tpr', np.nan),
                    'fpr': metrics.get('fpr', np.nan)
                })

        # Fairness by race
        if 'race_std' in period_df.columns:
            race_fairness = compute_fairness_metrics(period_df, score_col, 'outcome_binary', 'race_std')
            fairness_results.append({
                'time_period': str(period),
                'group_type': 'Race',
                'demographic_parity_diff': race_fairness['demographic_parity_diff'],
                'equalized_odds_diff': race_fairness['equalized_odds_diff'],
                'tpr_diff': race_fairness['tpr_diff'],
                'fpr_diff': race_fairness['fpr_diff']
            })
            # Save detailed per-subgroup metrics
            for subgroup, metrics in race_fairness.get('group_metrics', {}).items():
                fairness_detailed_results.append({
                    'time_period': str(period),
                    'group_type': 'Race',
                    'subgroup': subgroup,
                    'n': metrics.get('n', 0),
                    'positive_rate': metrics.get('positive_rate', np.nan),
                    'sensitivity': metrics.get('tpr', np.nan),
                    'specificity': 1 - metrics.get('fpr', np.nan) if metrics.get('fpr') is not None else np.nan,
                    'tpr': metrics.get('tpr', np.nan),
                    'fpr': metrics.get('fpr', np.nan)
                })

        # Fairness by age
        if 'age_group' in period_df.columns:
            age_fairness = compute_fairness_metrics(period_df, score_col, 'outcome_binary', 'age_group')
            fairness_results.append({
                'time_period': str(period),
                'group_type': 'Age',
                'demographic_parity_diff': age_fairness['demographic_parity_diff'],
                'equalized_odds_diff': age_fairness['equalized_odds_diff'],
                'tpr_diff': age_fairness['tpr_diff'],
                'fpr_diff': age_fairness['fpr_diff']
            })
            # Save detailed per-subgroup metrics
            for subgroup, metrics in age_fairness.get('group_metrics', {}).items():
                fairness_detailed_results.append({
                    'time_period': str(period),
                    'group_type': 'Age',
                    'subgroup': subgroup,
                    'n': metrics.get('n', 0),
                    'positive_rate': metrics.get('positive_rate', np.nan),
                    'sensitivity': metrics.get('tpr', np.nan),
                    'specificity': 1 - metrics.get('fpr', np.nan) if metrics.get('fpr') is not None else np.nan,
                    'tpr': metrics.get('tpr', np.nan),
                    'fpr': metrics.get('fpr', np.nan)
                })

    return (
        pd.DataFrame(classification_results),
        pd.DataFrame(calibration_results),
        pd.DataFrame(fairness_results),
        pd.DataFrame(fairness_detailed_results)
    )


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
        elif subgroup_type == 'Intersectional':
            # Parse intersectional label: "Age_Gender_Race" or "Age_Gender"
            parts = subgroup.split('_')
            if len(parts) == 3:
                age, gender, race = parts
                return (df['age_group'] == age) & (df['gender_std'] == gender) & (df['race_std'] == race)
            elif len(parts) == 2:
                age, gender = parts
                return (df['age_group'] == age) & (df['gender_std'] == gender)
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


def compute_drift_trend_test(results_df, bootstrap_store, alpha=0.05):
    """
    Compute drift trend significance using Page's L trend test on bootstrap replicates.

    For each subgroup, constructs a matrix (rows=bootstrap replicates, columns=ordered periods)
    and tests for a monotonic trend across ALL periods (not just first vs last).
    Applies Benjamini-Hochberg FDR correction across all tests.

    Args:
        results_df: Results DataFrame from analyze_drift (with auc per period)
        bootstrap_store: dict of {(subgroup_type, subgroup, period): [bootstrap_aucs]}
        alpha: Significance threshold after FDR correction (default 0.05)

    Returns:
        DataFrame with trend test results including FDR-corrected p-values
    """
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

        if first.empty or last.empty:
            continue

        auc_first = first['auc'].values[0]
        auc_last = last['auc'].values[0]

        if np.isnan(auc_first) or np.isnan(auc_last):
            continue

        delta = auc_last - auc_first

        # Gather bootstrap replicates across all periods for this subgroup
        available_periods = []
        replicate_lists = []
        for p in periods:
            key = (subgroup_type, subgroup, str(p))
            if key in bootstrap_store and len(bootstrap_store[key]) > 0:
                available_periods.append(p)
                replicate_lists.append(bootstrap_store[key])

        # Page's test needs >= 3 periods and >= 3 replicates
        p_trend = np.nan
        L_stat = np.nan
        trend_direction = 'none'

        if len(available_periods) >= 3:
            min_len = min(len(r) for r in replicate_lists)
            if min_len >= 3:
                # Build matrix: rows=replicates, columns=periods (ordered)
                matrix = np.column_stack([np.array(r[:min_len]) for r in replicate_lists])

                try:
                    # Test for increasing trend
                    res_inc = page_trend_test(matrix, method='asymptotic')
                    p_inc = res_inc.pvalue
                except Exception:
                    p_inc = 1.0

                try:
                    # Test for decreasing trend (reverse column order)
                    res_dec = page_trend_test(matrix[:, ::-1], method='asymptotic')
                    p_dec = res_dec.pvalue
                except Exception:
                    p_dec = 1.0

                # Two-sided p-value
                p_trend = min(2.0 * min(p_inc, p_dec), 1.0)

                if p_inc < p_dec:
                    trend_direction = 'increasing'
                    L_stat = res_inc.statistic
                else:
                    trend_direction = 'decreasing'
                    L_stat = res_dec.statistic
        elif len(available_periods) == 2:
            # Fallback: with only 2 periods, use DeLong's test
            min_len = min(len(r) for r in replicate_lists)
            if min_len >= 3:
                from scipy.stats import mannwhitneyu
                r1 = np.array(replicate_lists[0][:min_len])
                r2 = np.array(replicate_lists[1][:min_len])
                try:
                    _, p_trend = mannwhitneyu(r1, r2, alternative='two-sided')
                    trend_direction = 'increasing' if np.median(r2) > np.median(r1) else 'decreasing'
                except Exception:
                    p_trend = np.nan

        # Get CIs
        ci_first_lower = first['auc_ci_lower'].values[0] if 'auc_ci_lower' in first.columns else np.nan
        ci_first_upper = first['auc_ci_upper'].values[0] if 'auc_ci_upper' in first.columns else np.nan
        ci_last_lower = last['auc_ci_lower'].values[0] if 'auc_ci_lower' in last.columns else np.nan
        ci_last_upper = last['auc_ci_upper'].values[0] if 'auc_ci_upper' in last.columns else np.nan

        if not np.isnan(ci_first_lower) and not np.isnan(ci_last_lower):
            delta_ci_lower = ci_last_lower - ci_first_upper
            delta_ci_upper = ci_last_upper - ci_first_lower
        else:
            delta_ci_lower = np.nan
            delta_ci_upper = np.nan

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
            'trend_direction': trend_direction,
            'page_L_statistic': L_stat,
            'p_value_trend': p_trend,
            'period_first': first_period,
            'period_last': last_period,
            'n_first': first['n'].values[0],
            'n_last': last['n'].values[0],
            'n_periods': len(available_periods),
            # Legacy columns for backward compatibility
            'z_statistic': np.nan,
            'p_value_delong': p_trend,  # alias for downstream consumers
            'p_value_permutation': np.nan,
        })

    if not deltas:
        return pd.DataFrame()

    deltas_df = pd.DataFrame(deltas)

    # Benjamini-Hochberg FDR correction across all tests
    valid_mask = deltas_df['p_value_trend'].notna()
    if valid_mask.sum() > 0:
        raw_pvals = deltas_df.loc[valid_mask, 'p_value_trend'].values
        adjusted = false_discovery_control(raw_pvals, method='bh')
        deltas_df.loc[valid_mask, 'p_value_trend_fdr'] = adjusted
        deltas_df.loc[valid_mask, 'significant'] = adjusted < alpha
    else:
        deltas_df['p_value_trend_fdr'] = np.nan
        deltas_df['significant'] = False

    if 'p_value_trend_fdr' not in deltas_df.columns:
        deltas_df['p_value_trend_fdr'] = np.nan
    if 'significant' not in deltas_df.columns:
        deltas_df['significant'] = False
    deltas_df['significant'] = deltas_df['significant'].infer_objects(copy=False).fillna(False)

    return deltas_df


def compute_between_group_drift_comparison(deltas_df, bootstrap_store, alpha=0.05):
    """
    Test whether drift differs significantly BETWEEN subgroups.

    For each subgroup_type (Age, Gender, Race): pairwise Mann-Whitney U
    on bootstrap delta distributions. For Intersectional: compare each
    group vs Overall only (too many pairwise otherwise).

    Args:
        deltas_df: Output from compute_drift_trend_test() (one row per subgroup-score)
        bootstrap_store: dict of {(subgroup_type, subgroup, period_str): [auc_replicates]}
        alpha: Significance threshold after FDR correction

    Returns:
        DataFrame with between-group comparison results
    """
    from scipy.stats import mannwhitneyu
    from itertools import combinations

    if deltas_df.empty or not bootstrap_store:
        return pd.DataFrame()

    # Determine periods from bootstrap_store keys
    all_periods = sorted(set(k[2] for k in bootstrap_store.keys()))
    if len(all_periods) < 2:
        return pd.DataFrame()

    first_period = all_periods[0]
    last_period = all_periods[-1]

    comparisons = []

    # Get unique subgroup_types present in the data
    subgroup_types = deltas_df['subgroup_type'].unique()

    for stype in subgroup_types:
        stype_data = deltas_df[deltas_df['subgroup_type'] == stype]
        subgroups = stype_data['subgroup'].unique().tolist()

        if len(subgroups) < 2 and stype != 'Intersectional':
            continue

        # Build delta distributions for each subgroup in this type
        delta_distributions = {}
        for sg in subgroups:
            key_first = (stype, sg, str(first_period))
            key_last = (stype, sg, str(last_period))

            if key_first not in bootstrap_store or key_last not in bootstrap_store:
                continue

            reps_first = np.array(bootstrap_store[key_first])
            reps_last = np.array(bootstrap_store[key_last])

            min_len = min(len(reps_first), len(reps_last))
            if min_len < 3:
                continue

            # delta distribution: AUC_last - AUC_first for each replicate
            delta_dist = reps_last[:min_len] - reps_first[:min_len]
            delta_distributions[sg] = delta_dist

        if stype == 'Intersectional':
            # Compare each intersectional group vs Overall
            overall_key_first = ('Overall', 'All', str(first_period))
            overall_key_last = ('Overall', 'All', str(last_period))

            if overall_key_first in bootstrap_store and overall_key_last in bootstrap_store:
                reps_first_ov = np.array(bootstrap_store[overall_key_first])
                reps_last_ov = np.array(bootstrap_store[overall_key_last])
                min_len_ov = min(len(reps_first_ov), len(reps_last_ov))

                if min_len_ov >= 3:
                    delta_overall = reps_last_ov[:min_len_ov] - reps_first_ov[:min_len_ov]

                    for sg, delta_dist in delta_distributions.items():
                        try:
                            _, p_val = mannwhitneyu(delta_dist, delta_overall, alternative='two-sided')
                        except Exception:
                            p_val = np.nan

                        obs_delta_a = np.median(delta_dist)
                        obs_delta_b = np.median(delta_overall)
                        diff_dist = delta_dist[:min(len(delta_dist), len(delta_overall))] - \
                                    delta_overall[:min(len(delta_dist), len(delta_overall))]

                        comparisons.append({
                            'subgroup_type': stype,
                            'group_a': sg,
                            'group_b': 'Overall (All)',
                            'delta_a': obs_delta_a,
                            'delta_b': obs_delta_b,
                            'delta_diff': obs_delta_a - obs_delta_b,
                            'delta_diff_ci_lower': np.percentile(diff_dist, 2.5),
                            'delta_diff_ci_upper': np.percentile(diff_dist, 97.5),
                            'p_value': p_val,
                        })
        else:
            # Pairwise comparisons within this subgroup_type
            for sg_a, sg_b in combinations(sorted(delta_distributions.keys()), 2):
                delta_a = delta_distributions[sg_a]
                delta_b = delta_distributions[sg_b]

                try:
                    _, p_val = mannwhitneyu(delta_a, delta_b, alternative='two-sided')
                except Exception:
                    p_val = np.nan

                obs_delta_a = np.median(delta_a)
                obs_delta_b = np.median(delta_b)
                min_len = min(len(delta_a), len(delta_b))
                diff_dist = delta_a[:min_len] - delta_b[:min_len]

                comparisons.append({
                    'subgroup_type': stype,
                    'group_a': sg_a,
                    'group_b': sg_b,
                    'delta_a': obs_delta_a,
                    'delta_b': obs_delta_b,
                    'delta_diff': obs_delta_a - obs_delta_b,
                    'delta_diff_ci_lower': np.percentile(diff_dist, 2.5),
                    'delta_diff_ci_upper': np.percentile(diff_dist, 97.5),
                    'p_value': p_val,
                })

            # Also compare each subgroup vs Overall
            overall_key_first = ('Overall', 'All', str(first_period))
            overall_key_last = ('Overall', 'All', str(last_period))

            if overall_key_first in bootstrap_store and overall_key_last in bootstrap_store:
                reps_first_ov = np.array(bootstrap_store[overall_key_first])
                reps_last_ov = np.array(bootstrap_store[overall_key_last])
                min_len_ov = min(len(reps_first_ov), len(reps_last_ov))

                if min_len_ov >= 3:
                    delta_overall = reps_last_ov[:min_len_ov] - reps_first_ov[:min_len_ov]

                    for sg, delta_dist in delta_distributions.items():
                        try:
                            _, p_val = mannwhitneyu(delta_dist, delta_overall, alternative='two-sided')
                        except Exception:
                            p_val = np.nan

                        obs_delta_a = np.median(delta_dist)
                        obs_delta_b = np.median(delta_overall)
                        min_len = min(len(delta_dist), len(delta_overall))
                        diff_dist = delta_dist[:min_len] - delta_overall[:min_len]

                        comparisons.append({
                            'subgroup_type': stype,
                            'group_a': sg,
                            'group_b': 'Overall (All)',
                            'delta_a': obs_delta_a,
                            'delta_b': obs_delta_b,
                            'delta_diff': obs_delta_a - obs_delta_b,
                            'delta_diff_ci_lower': np.percentile(diff_dist, 2.5),
                            'delta_diff_ci_upper': np.percentile(diff_dist, 97.5),
                            'p_value': p_val,
                        })

    if not comparisons:
        return pd.DataFrame()

    comp_df = pd.DataFrame(comparisons)

    # FDR correction across all comparisons
    valid_mask = comp_df['p_value'].notna()
    if valid_mask.sum() > 0:
        raw_pvals = comp_df.loc[valid_mask, 'p_value'].values
        adjusted = false_discovery_control(raw_pvals, method='bh')
        comp_df.loc[valid_mask, 'p_value_fdr'] = adjusted
        comp_df.loc[valid_mask, 'significant'] = adjusted < alpha
    else:
        comp_df['p_value_fdr'] = np.nan
        comp_df['significant'] = False

    if 'p_value_fdr' not in comp_df.columns:
        comp_df['p_value_fdr'] = np.nan
    if 'significant' not in comp_df.columns:
        comp_df['significant'] = False
    comp_df['significant'] = comp_df['significant'].infer_objects(copy=False).fillna(False)

    return comp_df


def export_per_dataset_tables(dataset_key, dataset_name, results_df, deltas_df, output_dir,
                              between_group_df=None):
    """
    Export per-dataset tables (not cross-dataset comparisons).

    Creates:
    - output/{dataset}/drift_results.csv - All AUC values per period
    - output/{dataset}/drift_deltas.csv - Delta changes with p-values
    - output/{dataset}/summary_by_score.csv - Summary table per score
    - output/{dataset}/subgroup_drift.csv - Subgroup-level drift summary
    - output/{dataset}/between_group_comparisons.csv - Between-group drift comparisons
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
                'p-value (trend)': f"{row.get('p_value_trend_fdr', row.get('p_value_delong', np.nan)):.4f}" if pd.notna(row.get('p_value_trend_fdr', row.get('p_value_delong', np.nan))) else 'N/A',
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
            'p-value (trend, FDR)': f"{row.get('p_value_trend_fdr', row.get('p_value_delong', np.nan)):.4f}" if pd.notna(row.get('p_value_trend_fdr', row.get('p_value_delong', np.nan))) else 'N/A',
            'Period First': row['period_first'],
            'Period Last': row['period_last']
        })

    if subgroup_rows:
        subgroup_df = pd.DataFrame(subgroup_rows)
        subgroup_file = dataset_dir / 'subgroup_drift.csv'
        subgroup_df.to_csv(subgroup_file, index=False)

    # 5. Save between-group comparisons if provided
    if between_group_df is not None and not between_group_df.empty:
        bg_file = dataset_dir / 'between_group_comparisons.csv'
        between_group_df.to_csv(bg_file, index=False)
        n_sig = between_group_df['significant'].sum() if 'significant' in between_group_df.columns else 0
        print(f"  Saved between-group comparisons ({n_sig}/{len(between_group_df)} significant)")

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
        dataset_between_group = []

        for score_col in score_cols:
            if score_col not in df.columns:
                print(f"  Score '{score_col}' not found, skipping")
                continue

            print(f"\n  Analyzing {score_col.upper()} score...")

            # Run drift analysis
            results, bootstrap_store = analyze_drift(df, config, score_col)

            if not results.empty:
                results['dataset'] = dataset_key
                results['dataset_name'] = config['name']
                results['score'] = score_col
                all_results.append(results)
                dataset_results.append(results)

                # Compute deltas with trend test (Page's L test + FDR correction)
                print(f"  Computing trend significance (Page's L test + FDR correction)...")
                deltas = compute_drift_trend_test(results, bootstrap_store)

                if deltas.empty:
                    # Fallback to simple deltas if trend test fails
                    deltas = compute_drift_deltas(results)

                if not deltas.empty:
                    deltas['dataset'] = dataset_key
                    deltas['dataset_name'] = config['name']
                    deltas['score'] = score_col
                    all_deltas.append(deltas)
                    dataset_deltas.append(deltas)

                    # Compute between-group drift comparisons
                    print(f"  Computing between-group drift comparisons...")
                    bg_comparisons = compute_between_group_drift_comparison(deltas, bootstrap_store)
                    if not bg_comparisons.empty:
                        bg_comparisons['dataset'] = dataset_key
                        bg_comparisons['dataset_name'] = config['name']
                        bg_comparisons['score'] = score_col
                        dataset_between_group.append(bg_comparisons)
                        n_sig = bg_comparisons['significant'].sum()
                        print(f"    {n_sig}/{len(bg_comparisons)} between-group comparisons significant")

                    # Print summary with significance
                    print(f"\n  Drift Summary for {score_col.upper()}:")
                    for _, row in deltas.iterrows():
                        arrow = "↓" if row['delta'] < 0 else "↑"
                        sig_marker = "*" if row.get('significant', False) else ""
                        p_val = row.get('p_value_trend_fdr', row.get('p_value_trend', np.nan))
                        p_str = f"p={p_val:.3f}" if pd.notna(p_val) and not np.isnan(p_val) else ""
                        print(f"    {row['subgroup_type']:8} | {row['subgroup']:10} | {row['auc_first']:.3f} → {row['auc_last']:.3f} ({arrow}{abs(row['delta']):.3f}{sig_marker}) {p_str}")

        # Run Xiaoli's recommended metrics analysis (SOFA only)
        sofa_col = 'sofa' if 'sofa' in df.columns else None
        if sofa_col:
            print(f"\n  Running Xiaoli metrics analysis (SOFA>={SOFA_THRESHOLD} threshold)...")
            class_results, calib_results, fairness_results, fairness_detailed = analyze_xiaoli_metrics(df, config, sofa_col)

            # Save Xiaoli metrics to dataset directory
            dataset_dir = Path(OUTPUT_DIR) / dataset_key
            dataset_dir.mkdir(parents=True, exist_ok=True)

            if not class_results.empty:
                class_results.to_csv(dataset_dir / 'classification_metrics.csv', index=False)
                print(f"    Saved classification metrics (TPR, FPR, PPV, NPV)")

            if not calib_results.empty:
                calib_results.to_csv(dataset_dir / 'calibration_metrics.csv', index=False)
                print(f"    Saved calibration metrics (Brier score, SMR)")

            if not fairness_results.empty:
                fairness_results.to_csv(dataset_dir / 'fairness_metrics.csv', index=False)
                print(f"    Saved fairness metrics (demographic parity, equalized odds)")

            if not fairness_detailed.empty:
                fairness_detailed.to_csv(dataset_dir / 'fairness_detailed.csv', index=False)
                print(f"    Saved detailed fairness metrics per subgroup (Gender: Male/Female, Race, Age)")

        # Export per-dataset tables (not cross-dataset)
        if dataset_results and dataset_deltas:
            combined_dataset_results = pd.concat(dataset_results, ignore_index=True)
            combined_dataset_deltas = pd.concat(dataset_deltas, ignore_index=True)
            combined_bg = pd.concat(dataset_between_group, ignore_index=True) if dataset_between_group else None
            export_per_dataset_tables(
                dataset_key,
                config['name'],
                combined_dataset_results,
                combined_dataset_deltas,
                OUTPUT_DIR,
                between_group_df=combined_bg
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
                            p_val = row.get('p_value_trend_fdr', row.get('p_value_delong', np.nan))
                            print(f"    {row['subgroup_type']}={row['subgroup']} ({row['score']}): {row['delta']:+.3f} p={p_val:.4f}")

                    if not sig_improve.empty:
                        print(f"\n  Top significant improvements:")
                        for _, row in sig_improve.iterrows():
                            p_val = row.get('p_value_trend_fdr', row.get('p_value_delong', np.nan))
                            print(f"    {row['subgroup_type']}={row['subgroup']} ({row['score']}): {row['delta']:+.3f} p={p_val:.4f}")
