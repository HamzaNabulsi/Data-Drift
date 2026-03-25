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
from scipy.stats import chi2_contingency
from sklearn.metrics import roc_auc_score
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

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


def _cramers_v(confusion_matrix):
    """Compute Cramér's V statistic for a contingency table.

    Args:
        confusion_matrix: A 2-D array-like contingency table.

    Returns:
        float: Cramér's V effect size (0 = no association, 1 = perfect).
    """
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    min_dim = min(confusion_matrix.shape[0], confusion_matrix.shape[1]) - 1
    if min_dim == 0 or n == 0:
        return np.nan
    return np.sqrt(chi2 / (n * min_dim))


def analyze_care_demographics_correlation(df, config):
    """Cross-tabulate care frequency quartiles with demographics and test independence.

    Performs the following (L6 analysis):
    - Cross-tabulates care quartiles with age, gender, and race individually
    - Computes intersectional cross-tabulation: care quartile × age × gender × race
    - For each intersection, reports sample size, mean care frequency, and AUC drift
    - Runs chi-squared test and Cramér's V for each demographic dimension
    - Saves results CSV and heatmap figure

    Args:
        df: DataFrame with columns care_quartile, age_group, gender_std, race_std,
            outcome_binary, sofa, and the care frequency column.
        config: Dict with keys 'name', 'care_col', 'dataset_key'.

    Returns:
        pd.DataFrame: Intersectional summary rows.
    """
    dataset_key = config.get('dataset_key', config.get('key', 'unknown'))
    care_col = config['care_col']
    dataset_output_dir = OUTPUT_DIR / dataset_key
    dataset_output_dir.mkdir(exist_ok=True)

    demographic_cols = {
        'Age': 'age_group',
        'Gender': 'gender_std',
        'Race': 'race_std',
    }

    # ------------------------------------------------------------------
    # 1. Chi-squared / Cramér's V for each demographic vs care quartile
    # ------------------------------------------------------------------
    stat_rows = []
    for demo_label, demo_col in demographic_cols.items():
        ct = pd.crosstab(df['care_quartile'], df[demo_col])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue
        chi2, p, dof, _ = chi2_contingency(ct)
        cv = _cramers_v(ct)
        stat_rows.append({
            'demographic': demo_label,
            'chi2': chi2,
            'p_value': p,
            'dof': dof,
            'cramers_v': cv,
        })
    stats_df = pd.DataFrame(stat_rows)

    # ------------------------------------------------------------------
    # 2. Intersectional breakdown: quartile × age × gender × race
    # ------------------------------------------------------------------
    intersect_rows = []
    grouped = df.groupby(['care_quartile', 'age_group', 'gender_std', 'race_std'], observed=True)
    for (q, age, gender, race), grp in grouped:
        row = {
            'care_quartile': q,
            'age_group': age,
            'gender': gender,
            'race': race,
            'n': len(grp),
            'mean_care_frequency': grp[care_col].mean(),
        }
        # AUC drift proxy: compute AUC if enough data
        auc_val = compute_auc(grp['outcome_binary'].values, grp['sofa'].values) if len(grp) >= 30 else np.nan
        row['auc'] = auc_val
        intersect_rows.append(row)
    intersect_df = pd.DataFrame(intersect_rows)

    # ------------------------------------------------------------------
    # 3. Combine and save
    # ------------------------------------------------------------------
    # Append chi-sq stats as a separate section in the CSV
    combined_path = dataset_output_dir / 'care_demographics_correlation.csv'
    with open(combined_path, 'w') as fh:
        fh.write('# Chi-squared tests: care quartile vs demographics\n')
        stats_df.to_csv(fh, index=False)
        fh.write('\n# Intersectional breakdown: quartile x age x gender x race\n')
        intersect_df.to_csv(fh, index=False)
    print(f"Saved: {combined_path.relative_to(BASE_DIR)}")

    # ------------------------------------------------------------------
    # 4. Heatmap: mean care frequency by care quartile × race
    # ------------------------------------------------------------------
    pivot = intersect_df.groupby(['care_quartile', 'race'])['mean_care_frequency'].mean().unstack(fill_value=np.nan)
    if not pivot.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax, linewidths=0.5)
        ax.set_title(f'{config["name"]} — Mean Care Frequency\nby Care Quartile × Race')
        ax.set_ylabel('Care Quartile')
        ax.set_xlabel('Race')
        fig_path = SUPPLEMENTARY_FIGURES_DIR / f'{dataset_key}_care_demographics_heatmap.png'
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {fig_path.relative_to(BASE_DIR)}")

    return intersect_df


def generate_care_phenotype_demographic_figure(results_df, df, config):
    """Generate care phenotype figures focused on race x quartile drift (L6).

    Answers Leo's L6 question: does SOFA drift differ by race WITHIN each care
    quartile, and does demographic composition explain quartile-level drift?

    Produces:
    - Panel A: Grouped bar chart — AUROC delta by care quartile, bars per race,
               red 'All' line overlay (Xiaoli format). Saved as main figure.
    - Panel B: Heatmap — demographic composition (% by race) per care quartile.

    Outputs:
    - figures/supplementary/{key}_care_phenotype_demographics.png
    - output/{key}/care_quartile_race_drift_raw.csv   (race x quartile x time)
    - output/{key}/care_quartile_race_drift_deltas.csv (delta per race x quartile)

    Args:
        results_df: DataFrame from analyze_sofa_drift.
        df: Original prepared DataFrame with individual-level data.
        config: Dict with keys 'name', 'care_col', 'key'.
    """
    dataset_key = config.get('dataset_key', config.get('key', 'unknown'))
    out_dir = OUTPUT_DIR / dataset_key
    out_dir.mkdir(exist_ok=True)

    races = ['White', 'Black', 'Hispanic', 'Asian']
    quartiles = ['Q1 (High)', 'Q2', 'Q3', 'Q4 (Low)']
    time_periods = sorted(df['admission_year'].unique())

    # ------------------------------------------------------------------
    # Step 1: Compute AUC by (race x quartile x time) — the missing link
    # ------------------------------------------------------------------
    within_rows = []
    for race in races:
        for q in quartiles:
            for period in time_periods:
                subset = df[(df['race_std'] == race) &
                            (df['care_quartile'] == q) &
                            (df['admission_year'] == period)]
                if len(subset) >= 30:
                    auc_val = compute_auc(subset['outcome_binary'].values,
                                         subset['sofa'].values)
                    within_rows.append({
                        'race': race,
                        'care_quartile': q,
                        'time_period': period,
                        'auc': auc_val,
                        'n': len(subset),
                        'phenotype': dataset_key,
                    })
    within_df = pd.DataFrame(within_rows)

    # Save raw temporal table
    raw_path = out_dir / 'care_quartile_race_drift_raw.csv'
    within_df.to_csv(raw_path, index=False)
    print(f"Saved: {raw_path.relative_to(BASE_DIR)}")

    # ------------------------------------------------------------------
    # Step 2: Compute first-to-last delta per (race x quartile)
    # ------------------------------------------------------------------
    if len(time_periods) >= 2:
        first_period, last_period = time_periods[0], time_periods[-1]
        delta_rows = []
        for (race, q), grp in within_df.groupby(['race', 'care_quartile']):
            r_first = grp[grp['time_period'] == first_period]
            r_last = grp[grp['time_period'] == last_period]
            if r_first.empty or r_last.empty:
                continue
            auc_first = r_first['auc'].values[0]
            auc_last = r_last['auc'].values[0]
            delta_rows.append({
                'race': race,
                'care_quartile': q,
                'auc_first': auc_first,
                'auc_last': auc_last,
                'delta': auc_last - auc_first,
                'n_first': r_first['n'].values[0],
                'n_last': r_last['n'].values[0],
                'phenotype': dataset_key,
            })
        delta_df = pd.DataFrame(delta_rows)
        delta_path = out_dir / 'care_quartile_race_drift_deltas.csv'
        delta_df.to_csv(delta_path, index=False)
        print(f"Saved: {delta_path.relative_to(BASE_DIR)}")
    else:
        delta_df = pd.DataFrame()

    # ------------------------------------------------------------------
    # Step 3: Compute overall (all-race) delta per quartile for 'All' overlay
    # ------------------------------------------------------------------
    overall_rows = []
    for q in quartiles:
        for period in time_periods:
            subset = df[(df['care_quartile'] == q) & (df['admission_year'] == period)]
            if len(subset) >= 30:
                overall_rows.append({
                    'care_quartile': q,
                    'time_period': period,
                    'auc': compute_auc(subset['outcome_binary'].values,
                                      subset['sofa'].values),
                    'n': len(subset),
                })
    overall_df = pd.DataFrame(overall_rows)
    all_deltas = {}
    if len(time_periods) >= 2:
        for q in quartiles:
            grp = overall_df[overall_df['care_quartile'] == q]
            r_first = grp[grp['time_period'] == time_periods[0]]
            r_last = grp[grp['time_period'] == time_periods[-1]]
            if not r_first.empty and not r_last.empty:
                all_deltas[q] = r_last['auc'].values[0] - r_first['auc'].values[0]

    # ------------------------------------------------------------------
    # Step 4: Compute demographic composition per quartile (for Panel B)
    # ------------------------------------------------------------------
    comp_rows = []
    for q in quartiles:
        qtl_df = df[df['care_quartile'] == q]
        total = len(qtl_df)
        for race in races:
            n_race = (qtl_df['race_std'] == race).sum()
            comp_rows.append({
                'care_quartile': q,
                'race': race,
                'pct': 100 * n_race / total if total > 0 else 0,
            })
    comp_df = pd.DataFrame(comp_rows)

    # ------------------------------------------------------------------
    # Step 5: Figure — Panel A: grouped bar chart (Xiaoli format)
    #                  Panel B: composition heatmap
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Panel A: AUROC delta by (quartile x race), 'All' overlay ---
    ax_a = axes[0]
    if not delta_df.empty:
        x = np.arange(len(quartiles))
        n_races = len(races)
        bar_w = 0.18
        offsets = np.linspace(-(n_races - 1) / 2, (n_races - 1) / 2, n_races) * bar_w
        race_colors = {r: SUBGROUP_COLORS.get(r, '#888') for r in races}

        for i, race in enumerate(races):
            vals = []
            for q in quartiles:
                row = delta_df[(delta_df['race'] == race) &
                               (delta_df['care_quartile'] == q)]
                vals.append(row['delta'].values[0] if not row.empty else np.nan)
            ax_a.bar(x + offsets[i], vals, width=bar_w,
                     color=race_colors[race], label=race, alpha=0.85, zorder=3)

        # 'All' overlay line
        if all_deltas:
            all_vals = [all_deltas.get(q, np.nan) for q in quartiles]
            ax_a.plot(x, all_vals, color='red', marker='D', linewidth=2,
                      markersize=7, label='All (overall)', zorder=5)
            ax_a.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)

        ax_a.set_xticks(x)
        ax_a.set_xticklabels(quartiles, fontsize=10)
        ax_a.set_ylabel('AUROC Delta (last - first period)')
        ax_a.set_title('A. SOFA Drift by Care Quartile and Race\n'
                       '(red = overall quartile delta)', fontsize=11)
        ax_a.legend(fontsize=9, loc='upper left')
        ax_a.set_ylim(-0.35, 0.35)

    # --- Panel B: demographic composition heatmap ---
    ax_b = axes[1]
    if not comp_df.empty:
        pivot = comp_df.pivot(index='care_quartile', columns='race', values='pct')
        pivot = pivot.reindex(index=quartiles, columns=races)
        im = ax_b.imshow(pivot.values, aspect='auto', cmap='YlOrRd', vmin=0, vmax=25)
        ax_b.set_xticks(range(len(races)))
        ax_b.set_xticklabels(races, fontsize=10)
        ax_b.set_yticks(range(len(quartiles)))
        ax_b.set_yticklabels(quartiles, fontsize=10)
        for i in range(len(quartiles)):
            for j in range(len(races)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax_b.text(j, i, f'{val:.1f}%', ha='center', va='center',
                              fontsize=9,
                              color='black' if val < 15 else 'white')
        plt.colorbar(im, ax=ax_b, label='% of quartile')
        ax_b.set_title('B. Demographic Composition per Care Quartile\n'
                       '(% of patients by race)', fontsize=11)

    fig.suptitle(
        f'{config["name"]} — Care Quartile Drift by Race (L6)\n'
        'MIMIC only',
        fontsize=12,
    )
    fig.tight_layout()
    fig_path = SUPPLEMENTARY_FIGURES_DIR / f'{dataset_key}_care_phenotype_demographics.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {fig_path.relative_to(BASE_DIR)}")


def analyze_eicu_regional(df, config):
    """Analyze eICU drift broken down by hospital region and teaching status (X11/L5).

    If the dataset is eICU and hospital metadata columns (hospitalregion,
    teaching) are present, this function:
    - Groups hospitals by region (Midwest, Northeast, South, West)
    - Groups by teaching vs non-teaching status
    - Runs the standard SOFA drift analysis within each region/hospital type
    - Compares drift patterns across regions (especially for Black patients)
    - Saves results to output/eicu_combined/regional_breakdown.csv

    If the required columns are missing the function logs a warning and returns.

    Args:
        df: Prepared eICU DataFrame.
        config: Dict with keys 'name', 'dataset_key', etc.

    Returns:
        pd.DataFrame or None: Regional breakdown results, or None if skipped.
    """
    required_cols = {'hospitalregion', 'teaching'}
    available = set(df.columns)
    missing = required_cols - available
    if missing:
        logger.warning(
            "eICU regional analysis skipped — missing columns: %s. "
            "Ensure hospital metadata is merged into the input data.",
            ', '.join(sorted(missing)),
        )
        print(f"WARNING: eICU regional analysis skipped — missing columns: {', '.join(sorted(missing))}")
        return None

    eicu_output_dir = OUTPUT_DIR / 'eicu_combined'
    eicu_output_dir.mkdir(exist_ok=True)

    regions = df['hospitalregion'].dropna().unique()
    teaching_statuses = df['teaching'].dropna().unique()

    all_rows = []

    # ------------------------------------------------------------------
    # 1. Drift by region
    # ------------------------------------------------------------------
    for region in sorted(regions):
        region_df = df[df['hospitalregion'] == region]
        if len(region_df) < 50:
            continue
        results = analyze_sofa_drift(region_df, f'eICU_{region}', compute_ci=False)
        results['region'] = region
        results['teaching_status'] = 'All'
        deltas = compute_deltas(results)
        if not deltas.empty:
            deltas['region'] = region
            deltas['teaching_status'] = 'All'
            all_rows.append(deltas)

    # ------------------------------------------------------------------
    # 2. Drift by teaching status
    # ------------------------------------------------------------------
    for ts in sorted(teaching_statuses):
        ts_df = df[df['teaching'] == ts]
        ts_label = 'Teaching' if ts else 'Non-Teaching'
        if len(ts_df) < 50:
            continue
        results = analyze_sofa_drift(ts_df, f'eICU_{ts_label}', compute_ci=False)
        results['region'] = 'All'
        results['teaching_status'] = ts_label
        deltas = compute_deltas(results)
        if not deltas.empty:
            deltas['region'] = 'All'
            deltas['teaching_status'] = ts_label
            all_rows.append(deltas)

    # ------------------------------------------------------------------
    # 3. Drift by region × teaching (focus on Black patients per L5)
    # ------------------------------------------------------------------
    for region in sorted(regions):
        for ts in sorted(teaching_statuses):
            sub = df[(df['hospitalregion'] == region) & (df['teaching'] == ts)]
            ts_label = 'Teaching' if ts else 'Non-Teaching'
            if len(sub) < 50:
                continue
            results = analyze_sofa_drift(sub, f'eICU_{region}_{ts_label}', compute_ci=False)
            results['region'] = region
            results['teaching_status'] = ts_label
            deltas = compute_deltas(results)
            if not deltas.empty:
                deltas['region'] = region
                deltas['teaching_status'] = ts_label
                all_rows.append(deltas)

    if not all_rows:
        print("WARNING: No regional breakdown rows produced (insufficient data).")
        return None

    regional_df = pd.concat(all_rows, ignore_index=True)
    out_path = eicu_output_dir / 'regional_breakdown.csv'
    regional_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path.relative_to(BASE_DIR)}")

    # Highlight Black-patient drift across regions
    black_rows = regional_df[regional_df['subgroup'] == 'Black']
    if not black_rows.empty:
        print("\n  Black-patient AUC drift by region:")
        for _, row in black_rows.iterrows():
            region_label = row.get('region', '?')
            ts_label = row.get('teaching_status', '?')
            print(f"    Region={region_label}, Teaching={ts_label}: delta={row['delta']:+.3f}")

    return regional_df


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

        # L6: Care quartile vs demographics correlation analysis
        analyze_care_demographics_correlation(df, ds)

        # X13: Care phenotype figures focused on race-age-gender interactions
        generate_care_phenotype_demographic_figure(results, df, ds)

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
