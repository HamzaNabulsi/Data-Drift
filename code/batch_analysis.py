"""
Batch Drift Analysis Runner
Analyzes all configured datasets for subgroup-specific drift in ICU severity scores.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import DATASETS, OUTPUT_PATH as OUTPUT_DIR
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

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


def analyze_drift(df, config, score_col):
    """Analyze drift for a specific score across subgroups."""
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

    # Overall AUC by time period
    for period in time_periods:
        subset = df[df[year_col] == period]
        auc = compute_auc(subset['outcome_binary'], subset[score_col])
        results.append({
            'subgroup_type': 'Overall',
            'subgroup': 'All',
            'time_period': str(period),
            'auc': auc,
            'n': len(subset),
            'n_deaths': subset['outcome_binary'].sum(),
            'mortality_rate': subset['outcome_binary'].mean()
        })

    # Age group analysis
    if 'age_group' in df.columns:
        for age_group in AGE_LABELS:
            for period in time_periods:
                subset = df[(df['age_group'] == age_group) & (df[year_col] == period)]
                if len(subset) >= 50:
                    auc = compute_auc(subset['outcome_binary'], subset[score_col])
                    results.append({
                        'subgroup_type': 'Age',
                        'subgroup': age_group,
                        'time_period': str(period),
                        'auc': auc,
                        'n': len(subset),
                        'n_deaths': subset['outcome_binary'].sum(),
                        'mortality_rate': subset['outcome_binary'].mean()
                    })

    # Gender analysis
    if 'gender_std' in df.columns:
        for gender in ['Male', 'Female']:
            for period in time_periods:
                subset = df[(df['gender_std'] == gender) & (df[year_col] == period)]
                if len(subset) >= 50:
                    auc = compute_auc(subset['outcome_binary'], subset[score_col])
                    results.append({
                        'subgroup_type': 'Gender',
                        'subgroup': gender,
                        'time_period': str(period),
                        'auc': auc,
                        'n': len(subset),
                        'n_deaths': subset['outcome_binary'].sum(),
                        'mortality_rate': subset['outcome_binary'].mean()
                    })

    # Race analysis
    if 'race_std' in df.columns:
        for race in ['White', 'Black', 'Hispanic', 'Asian']:
            for period in time_periods:
                subset = df[(df['race_std'] == race) & (df[year_col] == period)]
                if len(subset) >= 30:  # Lower threshold for minority groups
                    auc = compute_auc(subset['outcome_binary'], subset[score_col])
                    results.append({
                        'subgroup_type': 'Race',
                        'subgroup': race,
                        'time_period': str(period),
                        'auc': auc,
                        'n': len(subset),
                        'n_deaths': subset['outcome_binary'].sum(),
                        'mortality_rate': subset['outcome_binary'].mean()
                    })

    return pd.DataFrame(results)


def compute_drift_deltas(results_df):
    """Compute drift (AUC change) between first and last time period."""
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
                deltas.append({
                    'subgroup_type': subgroup_type,
                    'subgroup': subgroup,
                    'auc_first': auc_first,
                    'auc_last': auc_last,
                    'delta': auc_last - auc_first,
                    'period_first': first_period,
                    'period_last': last_period,
                    'n_first': first['n'].values[0],
                    'n_last': last['n'].values[0]
                })

    return pd.DataFrame(deltas)


def run_batch_analysis(datasets_to_run=None):
    """Run drift analysis on all (or specified) datasets."""

    # Define which datasets to analyze (those with temporal data)
    temporal_datasets = ['mimiciv', 'amsterdam_icu', 'zhejiang', 'eicu', 'eicu_new']

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

                # Compute deltas
                deltas = compute_drift_deltas(results)
                if not deltas.empty:
                    deltas['dataset'] = dataset_key
                    deltas['dataset_name'] = config['name']
                    deltas['score'] = score_col
                    all_deltas.append(deltas)

                    # Print summary
                    print(f"\n  Drift Summary for {score_col.upper()}:")
                    for _, row in deltas.iterrows():
                        arrow = "↓" if row['delta'] < 0 else "↑"
                        print(f"    {row['subgroup_type']:8} | {row['subgroup']:10} | {row['auc_first']:.3f} → {row['auc_last']:.3f} ({arrow}{abs(row['delta']):.3f})")

    # Combine all results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_deltas = pd.concat(all_deltas, ignore_index=True) if all_deltas else pd.DataFrame()

        # Save results
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(exist_ok=True)

        results_file = output_dir / 'all_datasets_drift_results.csv'
        deltas_file = output_dir / 'all_datasets_drift_deltas.csv'

        combined_results.to_csv(results_file, index=False)
        combined_deltas.to_csv(deltas_file, index=False)

        print(f"\n{'='*60}")
        print("BATCH ANALYSIS COMPLETE")
        print('='*60)
        print(f"Results saved to: {results_file}")
        print(f"Deltas saved to: {deltas_file}")
        print(f"\nTotal records: {len(combined_results):,}")
        print(f"Total delta comparisons: {len(combined_deltas):,}")

        return combined_results, combined_deltas

    return pd.DataFrame(), pd.DataFrame()


if __name__ == '__main__':
    results, deltas = run_batch_analysis()

    if not deltas.empty:
        print("\n" + "="*60)
        print("KEY FINDINGS - LARGEST DRIFT BY SUBGROUP")
        print("="*60)

        # Find largest drifts
        for score in deltas['score'].unique():
            score_deltas = deltas[deltas['score'] == score]
            if not score_deltas.empty:
                worst = score_deltas.loc[score_deltas['delta'].idxmin()]
                best = score_deltas.loc[score_deltas['delta'].idxmax()]

                print(f"\n{score.upper()}:")
                print(f"  Worst drift: {worst['dataset_name']} - {worst['subgroup']} ({worst['delta']:+.3f})")
                print(f"  Best drift:  {best['dataset_name']} - {best['subgroup']} ({best['delta']:+.3f})")
