"""
MIMIC-IV Subsets: SOFA Drift + Care Frequency Analysis
=======================================================
Analyzes MIMIC-IV mouthcare and mechanical ventilation cohorts.

These cohorts have:
- Pre-computed SOFA scores
- Care frequency data (mouthcare/turning intervals)
- Race/ethnicity data

Usage:
    python supplementary_analysis.py                    # Default (N_BOOTSTRAP=100)
    python supplementary_analysis.py --fast             # Fast testing (N_BOOTSTRAP=2)
    python supplementary_analysis.py --bootstrap 1000   # Production (N_BOOTSTRAP=1000)

Output:
- output/mimic_sofa_results.csv - Full results by subgroup
- output/mimic_sofa_deltas.csv - Drift deltas
- figures/figS1_mimic_mouthcare.png - Mouthcare cohort figure
- figures/figS2_mimic_mechvent.png - Mechanical ventilation cohort figure
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# BOOTSTRAP CONFIGURATION (see batch_analysis.py for details)
# =============================================================================
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='MIMIC-IV SOFA + Care Frequency Analysis')
    parser.add_argument('-b', '--bootstrap', type=int, default=100,
                       help='Number of bootstrap iterations for CI (default: 100)')
    parser.add_argument('--fast', action='store_true',
                       help='Fast mode: use only 2 bootstrap iterations')
    return parser.parse_args()

args = parse_args()
N_BOOTSTRAP = 2 if args.fast else args.bootstrap
CI_LEVEL = 0.95
RANDOM_SEED = 42

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data' / 'mimic_iv_lc'  # MIMIC-IV Long-term Care cohort data
OUTPUT_DIR = BASE_DIR / 'output'
FIGURES_DIR = BASE_DIR / 'figures'
SUPPLEMENTARY_FIGURES_DIR = FIGURES_DIR / 'supplementary'
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
SUPPLEMENTARY_FIGURES_DIR.mkdir(exist_ok=True)

# Age bins
AGE_BINS = [18, 45, 65, 80, 150]
AGE_LABELS = ['18-44', '45-64', '65-79', '80+']

# Figure styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.dpi'] = 150
plt.rcParams['figure.facecolor'] = 'white'

# Colors
SUBGROUP_COLORS = {
    '18-44': '#4c72b0', '45-64': '#55a868', '65-79': '#c44e52', '80+': '#8172b3',
    'Male': '#64b5cd', 'Female': '#dd8452',
    'White': '#4c72b0', 'Black': '#55a868', 'Hispanic': '#c44e52', 'Asian': '#8172b3',
    'Q1 (High)': '#d62728', 'Q2': '#ff7f0e', 'Q3': '#2ca02c', 'Q4 (Low)': '#1f77b4',
    'All': '#666666'
}


def compute_auc(y_true, y_pred):
    """Compute AUC with error handling."""
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
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

    except Exception as e:
        return np.nan, np.nan, np.nan


def load_and_prepare(filepath, care_freq_col):
    """Load data and prepare for analysis."""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} records from {filepath.name}")

    # Binary outcome
    df['outcome_binary'] = (df['outcome'] == 'Deceased').astype(int)

    # Age groups
    df['age_group'] = pd.cut(df['admission_age'], bins=AGE_BINS, labels=AGE_LABELS, right=False)

    # Gender standardization
    df['gender_std'] = df['gender'].map({'Male': 'Male', 'Female': 'Female'})

    # Race standardization
    race_map = {'White': 'White', 'Black': 'Black', 'Hispanic': 'Hispanic',
                'Asian': 'Asian', 'Other': 'Other', 'Unknown': 'Unknown'}
    df['race_std'] = df['race'].map(lambda x: race_map.get(x, 'Other'))

    # Care frequency quartiles (Q1=highest frequency, Q4=lowest)
    df['care_quartile'] = pd.qcut(df[care_freq_col], q=4, labels=['Q1 (High)', 'Q2', 'Q3', 'Q4 (Low)'])

    return df


def analyze_sofa_drift(df, dataset_name, compute_ci=True):
    """Analyze SOFA drift across all subgroups with optional CIs."""
    results = []
    year_col = 'admission_year'
    score_col = 'sofa'

    time_periods = sorted(df[year_col].dropna().unique())
    print(f"Time periods: {time_periods}")
    if compute_ci:
        print(f"Computing {int(CI_LEVEL*100)}% bootstrap CIs (n={N_BOOTSTRAP})...")

    # Define subgroup analyses
    analyses = [
        ('Overall', 'All', None, None),
    ]

    # Age groups
    for ag in AGE_LABELS:
        analyses.append(('Age', ag, 'age_group', ag))

    # Gender
    for g in ['Male', 'Female']:
        analyses.append(('Gender', g, 'gender_std', g))

    # Race
    for r in ['White', 'Black', 'Hispanic', 'Asian']:
        analyses.append(('Race', r, 'race_std', r))

    # Care frequency quartiles
    for q in ['Q1 (High)', 'Q2', 'Q3', 'Q4 (Low)']:
        analyses.append(('Care Frequency', q, 'care_quartile', q))

    # Run analyses
    for subgroup_type, subgroup, col, val in analyses:
        for period in time_periods:
            if col is None:
                subset = df[df[year_col] == period]
            else:
                subset = df[(df[col] == val) & (df[year_col] == period)]

            if len(subset) >= 30:
                if compute_ci:
                    auc, ci_lower, ci_upper = compute_auc_with_ci(
                        subset['outcome_binary'].values, subset[score_col].values
                    )
                else:
                    auc = compute_auc(subset['outcome_binary'].values, subset[score_col].values)
                    ci_lower, ci_upper = np.nan, np.nan

                results.append({
                    'dataset': dataset_name,
                    'subgroup_type': subgroup_type,
                    'subgroup': subgroup,
                    'time_period': period,
                    'auc': auc,
                    'auc_ci_lower': ci_lower,
                    'auc_ci_upper': ci_upper,
                    'n': len(subset),
                    'n_deaths': int(subset['outcome_binary'].sum()),
                    'mortality_rate': subset['outcome_binary'].mean()
                })

    return pd.DataFrame(results)


def compute_deltas(results_df):
    """Compute AUC change between first and last period with CIs."""
    periods = sorted(results_df['time_period'].unique())
    if len(periods) < 2:
        return pd.DataFrame()

    first, last = periods[0], periods[-1]
    deltas = []

    for (subgroup_type, subgroup), group in results_df.groupby(['subgroup_type', 'subgroup']):
        first_row = group[group['time_period'] == first]
        last_row = group[group['time_period'] == last]

        if not first_row.empty and not last_row.empty:
            auc_first = first_row['auc'].values[0]
            auc_last = last_row['auc'].values[0]

            if not np.isnan(auc_first) and not np.isnan(auc_last):
                delta = auc_last - auc_first

                # Get CIs if available
                ci_first_lower = first_row['auc_ci_lower'].values[0] if 'auc_ci_lower' in first_row.columns else np.nan
                ci_first_upper = first_row['auc_ci_upper'].values[0] if 'auc_ci_upper' in first_row.columns else np.nan
                ci_last_lower = last_row['auc_ci_lower'].values[0] if 'auc_ci_lower' in last_row.columns else np.nan
                ci_last_upper = last_row['auc_ci_upper'].values[0] if 'auc_ci_upper' in last_row.columns else np.nan

                # Compute delta CI (conservative propagation)
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
                    'period_first': first,
                    'period_last': last
                })

    return pd.DataFrame(deltas)


def generate_supplementary_figure(results_df, deltas_df, dataset_name, output_name):
    """Generate a 4-panel supplementary figure with confidence intervals."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    time_periods = sorted(results_df['time_period'].unique())
    x_ticks = range(len(time_periods))

    has_ci = 'auc_ci_lower' in results_df.columns
    has_delta_ci = 'delta_ci_lower' in deltas_df.columns if deltas_df is not None and not deltas_df.empty else False

    # Panel A: Overall + Age
    ax1 = axes[0, 0]
    for subgroup in ['All'] + AGE_LABELS:
        if subgroup == 'All':
            data = results_df[(results_df['subgroup_type'] == 'Overall')]
        else:
            data = results_df[(results_df['subgroup_type'] == 'Age') & (results_df['subgroup'] == subgroup)]

        if not data.empty:
            data = data.sort_values('time_period')
            color = SUBGROUP_COLORS.get(subgroup, '#666')
            lw = 2.5 if subgroup == 'All' else 1.5
            x = x_ticks[:len(data)]
            ax1.plot(x, data['auc'], 'o-', label=subgroup, color=color, linewidth=lw, markersize=6)

            # Add CI shading
            if has_ci and not data['auc_ci_lower'].isna().all():
                ax1.fill_between(x, data['auc_ci_lower'], data['auc_ci_upper'], color=color, alpha=0.15)

    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(time_periods, rotation=45, ha='right')
    ax1.set_ylabel('SOFA AUC')
    ax1.set_title('A. SOFA Performance by Age Group' + (' (95% CI)' if has_ci else ''))
    ax1.legend(loc='lower right', fontsize=8)
    ax1.set_ylim(0.5, 0.95)

    # Panel B: Race
    ax2 = axes[0, 1]
    for race in ['White', 'Black', 'Hispanic', 'Asian']:
        data = results_df[(results_df['subgroup_type'] == 'Race') & (results_df['subgroup'] == race)]
        if not data.empty:
            data = data.sort_values('time_period')
            color = SUBGROUP_COLORS.get(race, '#666')
            x = x_ticks[:len(data)]
            ax2.plot(x, data['auc'], 'o-', label=race, color=color, linewidth=1.5, markersize=6)

            if has_ci and not data['auc_ci_lower'].isna().all():
                ax2.fill_between(x, data['auc_ci_lower'], data['auc_ci_upper'], color=color, alpha=0.15)

    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(time_periods, rotation=45, ha='right')
    ax2.set_ylabel('SOFA AUC')
    ax2.set_title('B. SOFA Performance by Race' + (' (95% CI)' if has_ci else ''))
    ax2.legend(loc='lower right', fontsize=8)
    ax2.set_ylim(0.5, 0.95)

    # Panel C: Care Frequency
    ax3 = axes[1, 0]
    for q in ['Q1 (High)', 'Q2', 'Q3', 'Q4 (Low)']:
        data = results_df[(results_df['subgroup_type'] == 'Care Frequency') & (results_df['subgroup'] == q)]
        if not data.empty:
            data = data.sort_values('time_period')
            color = SUBGROUP_COLORS.get(q, '#666')
            x = x_ticks[:len(data)]
            ax3.plot(x, data['auc'], 'o-', label=q, color=color, linewidth=1.5, markersize=6)

            if has_ci and not data['auc_ci_lower'].isna().all():
                ax3.fill_between(x, data['auc_ci_lower'], data['auc_ci_upper'], color=color, alpha=0.15)

    ax3.set_xticks(x_ticks)
    ax3.set_xticklabels(time_periods, rotation=45, ha='right')
    ax3.set_ylabel('SOFA AUC')
    ax3.set_title('C. SOFA Performance by Care Frequency' + (' (95% CI)' if has_ci else ''))
    ax3.legend(loc='lower right', fontsize=8)
    ax3.set_ylim(0.5, 0.95)

    # Panel D: Delta summary with error bars
    ax4 = axes[1, 1]
    if not deltas_df.empty:
        # Sort by delta
        plot_data = deltas_df.sort_values('delta')
        colors = [SUBGROUP_COLORS.get(s, '#666') for s in plot_data['subgroup']]

        if has_delta_ci:
            xerr = np.array([
                plot_data['delta'].values - plot_data['delta_ci_lower'].values,
                plot_data['delta_ci_upper'].values - plot_data['delta'].values
            ])
            xerr = np.nan_to_num(xerr, nan=0)
            xerr = np.clip(xerr, 0, None)  # Error bars must be non-negative
            bars = ax4.barh(range(len(plot_data)), plot_data['delta'], color=colors,
                           xerr=xerr, capsize=3, error_kw={'elinewidth': 1})
        else:
            bars = ax4.barh(range(len(plot_data)), plot_data['delta'], color=colors)

        ax4.set_yticks(range(len(plot_data)))
        ax4.set_yticklabels(plot_data['subgroup'], fontsize=8)
        ax4.axvline(x=0, color='black', linewidth=0.5)
        ax4.set_xlabel('AUC Change')
        ax4.set_title('D. SOFA AUC Change by Subgroup' + (' (95% CI)' if has_delta_ci else ''))

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, plot_data['delta'])):
            ax4.text(val + 0.005 if val >= 0 else val - 0.005, i,
                    f'{val:+.3f}', va='center', ha='left' if val >= 0 else 'right', fontsize=7)

    plt.suptitle(f'{dataset_name} - SOFA Drift Analysis', fontsize=14, y=0.98)

    # Save to supplementary figures folder
    output_path = SUPPLEMENTARY_FIGURES_DIR / output_name
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: supplementary/{output_name}")

    return output_path


def run_mimic_sofa_analysis():
    """Run SOFA + care frequency analysis for MIMIC-IV subsets."""
    print("=" * 60)
    print("MIMIC-IV SUBSETS: SOFA + CARE FREQUENCY ANALYSIS")
    print("=" * 60)

    all_results = []
    all_deltas = []

    # Dataset configurations
    datasets = [
        {
            'name': 'MIMIC-IV Mouthcare',
            'file': 'mouthcare_interval_frequency.csv',
            'care_col': 'mouthcare_interval_frequency',
            'output_fig': 'figS1_mimic_mouthcare.png',
            'key': 'mouthcare'
        },
        {
            'name': 'MIMIC-IV Mechanical Ventilation',
            'file': 'turning_interval_frequency.csv',
            'care_col': 'turning_interval_frequency',
            'output_fig': 'figS2_mimic_mechvent.png',
            'key': 'mechvent'
        }
    ]

    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"Analyzing: {ds['name']}")
        print("=" * 60)

        filepath = DATA_DIR / ds['file']
        if not filepath.exists():
            print(f"ERROR: File not found: {filepath}")
            continue

        # Load and prepare
        df = load_and_prepare(filepath, ds['care_col'])

        # Analyze
        results = analyze_sofa_drift(df, ds['name'])
        deltas = compute_deltas(results)

        # Print summary
        print(f"\nDrift Summary for SOFA:")
        for _, row in deltas.iterrows():
            arrow = "^" if row['delta'] >= 0 else "v"
            print(f"  {row['subgroup_type']:15} | {row['subgroup']:12} | {row['auc_first']:.3f} -> {row['auc_last']:.3f} ({arrow}{abs(row['delta']):.3f})")

        # Generate figure
        generate_supplementary_figure(results, deltas, ds['name'], ds['output_fig'])

        # Store
        results['dataset_key'] = ds['key']
        deltas['dataset_key'] = ds['key']
        deltas['dataset_name'] = ds['name']
        all_results.append(results)
        all_deltas.append(deltas)

    # Save combined results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_deltas = pd.concat(all_deltas, ignore_index=True)

        combined_results.to_csv(OUTPUT_DIR / 'mimic_sofa_results.csv', index=False)
        combined_deltas.to_csv(OUTPUT_DIR / 'mimic_sofa_deltas.csv', index=False)

        print(f"\n{'='*60}")
        print("MIMIC-IV SOFA ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"Results saved to: output/mimic_sofa_results.csv")
        print(f"Deltas saved to: output/mimic_sofa_deltas.csv")
        print(f"Figures saved to: figures/supplementary/figS1_mimic_mouthcare.png, figS2_mimic_mechvent.png")

        return combined_results, combined_deltas

    return pd.DataFrame(), pd.DataFrame()


if __name__ == '__main__':
    results, deltas = run_mimic_sofa_analysis()

    if not deltas.empty:
        print("\n" + "=" * 60)
        print("KEY FINDINGS - MIMIC-IV SOFA ANALYSIS")
        print("=" * 60)

        for dataset in deltas['dataset_name'].unique():
            ds_deltas = deltas[deltas['dataset_name'] == dataset]
            print(f"\n{dataset}:")

            # Best and worst
            worst = ds_deltas.loc[ds_deltas['delta'].idxmin()]
            best = ds_deltas.loc[ds_deltas['delta'].idxmax()]

            print(f"  Worst drift: {worst['subgroup']} ({worst['delta']:+.3f})")
            print(f"  Best drift:  {best['subgroup']} ({best['delta']:+.3f})")

            # Care frequency impact
            care_deltas = ds_deltas[ds_deltas['subgroup_type'] == 'Care Frequency']
            if not care_deltas.empty:
                q1 = care_deltas[care_deltas['subgroup'] == 'Q1 (High)']['delta'].values
                q4 = care_deltas[care_deltas['subgroup'] == 'Q4 (Low)']['delta'].values
                if len(q1) > 0 and len(q4) > 0:
                    print(f"  Care frequency gap: Q4 ({q4[0]:+.3f}) vs Q1 ({q1[0]:+.3f}) = {q4[0]-q1[0]:+.3f}")
