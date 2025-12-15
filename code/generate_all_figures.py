"""
Cross-Dataset Drift Analysis Figures
Generates publication-quality visualizations comparing drift across all datasets.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Set style for clear, publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.dpi'] = 150
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Paths
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / 'output'
FIGURES_DIR = BASE_DIR / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)

# Colors
DATASET_COLORS = {
    'mimiciv': '#1f77b4',
    'amsterdam_icu': '#ff7f0e',
    'zhejiang': '#2ca02c',
    'eicu': '#d62728',
    'eicu_new': '#9467bd'
}

SUBGROUP_COLORS = {
    '18-44': '#4c72b0',
    '45-64': '#55a868',
    '65-79': '#c44e52',
    '80+': '#8172b3',
    'Male': '#64b5cd',
    'Female': '#dd8452',
    'White': '#4c72b0',
    'Black': '#55a868',
    'Hispanic': '#c44e52',
    'Asian': '#8172b3',
    'All': '#666666'
}


def load_data():
    """Load drift analysis results."""
    results_file = OUTPUT_DIR / 'all_datasets_drift_results.csv'
    deltas_file = OUTPUT_DIR / 'all_datasets_drift_deltas.csv'

    if not results_file.exists():
        print("ERROR: Results file not found. Run batch_analysis.py first.")
        return None, None

    results = pd.read_csv(results_file)
    deltas = pd.read_csv(deltas_file) if deltas_file.exists() else None

    print(f"Loaded {len(results):,} result rows")
    if deltas is not None:
        print(f"Loaded {len(deltas):,} delta comparisons")

    return results, deltas


def fig1_overall_drift_by_dataset(results, deltas):
    """Figure 1: Overall drift comparison across datasets and scores."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: AUC over time for each dataset (OASIS as primary score)
    ax1 = axes[0]
    oasis_overall = results[(results['score'] == 'oasis') & (results['subgroup'] == 'All')].copy()

    for dataset in oasis_overall['dataset'].unique():
        df_sub = oasis_overall[oasis_overall['dataset'] == dataset].sort_values('time_period')
        color = DATASET_COLORS.get(dataset, '#666666')
        label = df_sub['dataset_name'].iloc[0].split(' (')[0]
        ax1.plot(range(len(df_sub)), df_sub['auc'], 'o-', label=label,
                color=color, markersize=8, linewidth=2)

    ax1.set_xlabel('Time Period (Ordered)')
    ax1.set_ylabel('AUC')
    ax1.set_title('A. Overall OASIS Performance Over Time')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.set_ylim(0.6, 0.9)
    ax1.axhline(y=0.75, color='gray', linestyle='--', alpha=0.5, label='Reference (0.75)')

    # Panel B: Delta comparison heatmap
    ax2 = axes[1]
    if deltas is not None:
        # Filter for Overall group only
        overall_deltas = deltas[deltas['subgroup'] == 'All'].copy()
        pivot = overall_deltas.pivot(index='dataset_name', columns='score', values='delta')

        # Reorder columns by score type
        score_order = ['oasis', 'sapsii', 'apsiii', 'sofa', 'apachescore']
        pivot = pivot[[c for c in score_order if c in pivot.columns]]

        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                   ax=ax2, cbar_kws={'label': 'AUC Change'}, vmin=-0.1, vmax=0.1)
        ax2.set_title('B. Overall AUC Change by Dataset and Score')
        ax2.set_xlabel('Severity Score')
        ax2.set_ylabel('')

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig1_overall_drift_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: fig1_overall_drift_comparison.png")


def fig2_age_stratified_drift(results, deltas):
    """Figure 2: Age-stratified drift across datasets."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Get datasets with age data
    age_data = results[(results['subgroup_type'] == 'Age') & (results['score'] == 'oasis')].copy()
    datasets = age_data['dataset'].unique()

    for idx, dataset in enumerate(datasets):
        row, col = divmod(idx, 3)
        ax = axes[row, col]

        df_sub = age_data[age_data['dataset'] == dataset]
        dataset_name = df_sub['dataset_name'].iloc[0].split(' (')[0]

        for age_group in ['18-44', '45-64', '65-79', '80+']:
            age_df = df_sub[df_sub['subgroup'] == age_group].sort_values('time_period')
            if not age_df.empty:
                color = SUBGROUP_COLORS.get(age_group, '#666666')
                ax.plot(range(len(age_df)), age_df['auc'], 'o-', label=age_group,
                       color=color, markersize=6, linewidth=1.5)

        ax.set_title(f'{dataset_name}')
        ax.set_xlabel('Time Period')
        ax.set_ylabel('AUC')
        ax.legend(title='Age', loc='lower right', fontsize=8)
        ax.set_ylim(0.4, 1.0)

    # Hide unused axes
    for idx in range(len(datasets), 6):
        row, col = divmod(idx, 3)
        axes[row, col].axis('off')

    plt.suptitle('OASIS Performance by Age Group Over Time', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig2_age_stratified_drift.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: fig2_age_stratified_drift.png")


def fig3_race_disparities(results, deltas):
    """Figure 3: Race/ethnicity disparities in US datasets."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Filter for datasets with race data
    race_data = results[(results['subgroup_type'] == 'Race') & (results['score'] == 'oasis')].copy()
    race_datasets = ['mimiciv', 'eicu', 'eicu_new']

    for idx, dataset in enumerate(race_datasets):
        ax = axes[idx]
        df_sub = race_data[race_data['dataset'] == dataset]

        if df_sub.empty:
            ax.axis('off')
            continue

        dataset_name = df_sub['dataset_name'].iloc[0].split(' (')[0]

        for race in ['White', 'Black', 'Hispanic', 'Asian']:
            race_df = df_sub[df_sub['subgroup'] == race].sort_values('time_period')
            if not race_df.empty:
                color = SUBGROUP_COLORS.get(race, '#666666')
                ax.plot(range(len(race_df)), race_df['auc'], 'o-', label=race,
                       color=color, markersize=6, linewidth=1.5)

        ax.set_title(f'{dataset_name}')
        ax.set_xlabel('Time Period')
        ax.set_ylabel('AUC')
        ax.legend(title='Race', loc='lower right', fontsize=8)
        ax.set_ylim(0.6, 0.9)

    plt.suptitle('OASIS Performance by Race/Ethnicity Over Time', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig3_race_disparities.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: fig3_race_disparities.png")


def fig4_drift_delta_summary(deltas):
    """Figure 4: Summary of drift deltas by subgroup type."""
    if deltas is None or deltas.empty:
        print("No delta data available for fig4")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.35, wspace=0.35)

    # Panel A: Overall deltas by dataset and score
    ax1 = axes[0, 0]
    overall_deltas = deltas[deltas['subgroup'] == 'All'].copy()

    if not overall_deltas.empty:
        pivot = overall_deltas.pivot(index='dataset_name', columns='score', values='delta')
        pivot = pivot.reindex(columns=[c for c in ['oasis', 'sapsii', 'apsiii', 'sofa', 'apachescore'] if c in pivot.columns])
        pivot.plot(kind='bar', ax=ax1, width=0.8)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_title('A. Overall AUC Change by Dataset')
        ax1.set_xlabel('')
        ax1.set_ylabel('AUC Change')
        ax1.legend(title='Score', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax1.set_xticklabels([x.get_text().split(' (')[0] for x in ax1.get_xticklabels()], rotation=45, ha='right')

    # Panel B: Age subgroup deltas (OASIS)
    ax2 = axes[0, 1]
    age_deltas = deltas[(deltas['subgroup_type'] == 'Age') & (deltas['score'] == 'oasis')].copy()

    if not age_deltas.empty:
        pivot = age_deltas.pivot(index='dataset_name', columns='subgroup', values='delta')
        pivot = pivot.reindex(columns=[c for c in ['18-44', '45-64', '65-79', '80+'] if c in pivot.columns])
        pivot.plot(kind='bar', ax=ax2, width=0.8, color=[SUBGROUP_COLORS.get(c, '#666') for c in pivot.columns])
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_title('B. OASIS AUC Change by Age Group')
        ax2.set_xlabel('')
        ax2.set_ylabel('AUC Change')
        ax2.legend(title='Age', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax2.set_xticklabels([x.get_text().split(' (')[0] for x in ax2.get_xticklabels()], rotation=45, ha='right')

    # Panel C: Gender deltas
    ax3 = axes[1, 0]
    gender_deltas = deltas[(deltas['subgroup_type'] == 'Gender') & (deltas['score'] == 'oasis')].copy()

    if not gender_deltas.empty:
        pivot = gender_deltas.pivot(index='dataset_name', columns='subgroup', values='delta')
        colors = [SUBGROUP_COLORS.get(c, '#666') for c in pivot.columns]
        pivot.plot(kind='bar', ax=ax3, width=0.8, color=colors)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_title('C. OASIS AUC Change by Gender')
        ax3.set_xlabel('')
        ax3.set_ylabel('AUC Change')
        ax3.legend(title='Gender', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax3.set_xticklabels([x.get_text().split(' (')[0] for x in ax3.get_xticklabels()], rotation=45, ha='right')

    # Panel D: Race deltas (US datasets only)
    ax4 = axes[1, 1]
    race_deltas = deltas[(deltas['subgroup_type'] == 'Race') & (deltas['score'] == 'oasis')].copy()

    if not race_deltas.empty:
        pivot = race_deltas.pivot(index='dataset_name', columns='subgroup', values='delta')
        pivot = pivot.reindex(columns=[c for c in ['White', 'Black', 'Hispanic', 'Asian'] if c in pivot.columns])
        colors = [SUBGROUP_COLORS.get(c, '#666') for c in pivot.columns]
        pivot.plot(kind='bar', ax=ax4, width=0.8, color=colors)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4.set_title('D. OASIS AUC Change by Race')
        ax4.set_xlabel('')
        ax4.set_ylabel('AUC Change')
        ax4.legend(title='Race', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax4.set_xticklabels([x.get_text().split(' (')[0] for x in ax4.get_xticklabels()], rotation=45, ha='right')

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig4_drift_delta_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: fig4_drift_delta_summary.png")


def fig5_score_comparison_heatmap(deltas):
    """Figure 5: Comprehensive heatmap of all subgroup drifts."""
    if deltas is None or deltas.empty:
        print("No delta data available for fig5")
        return

    fig, ax = plt.subplots(figsize=(14, 12))

    # Create combined index
    deltas_copy = deltas.copy()
    deltas_copy['row_label'] = deltas_copy['dataset_name'].str.split(r' \(').str[0] + ' - ' + deltas_copy['subgroup']

    # Pivot for heatmap
    pivot = deltas_copy.pivot(index='row_label', columns='score', values='delta')
    pivot = pivot.reindex(columns=[c for c in ['oasis', 'sapsii', 'apsiii', 'sofa', 'apachescore'] if c in pivot.columns])

    # Sort by average delta
    pivot['avg'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('avg')
    pivot = pivot.drop('avg', axis=1)

    # Rename columns for clarity
    pivot.columns = [c.upper() for c in pivot.columns]

    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
               ax=ax, cbar_kws={'label': 'AUC Change', 'shrink': 0.8},
               vmin=-0.2, vmax=0.2, annot_kws={'size': 9})

    ax.set_title('AUC Change Across All Datasets, Scores, and Subgroups', fontsize=14, pad=15)
    ax.set_xlabel('Severity Score', fontsize=12)
    ax.set_ylabel('')
    ax.tick_params(axis='y', labelsize=9)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig5_comprehensive_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: fig5_comprehensive_heatmap.png")


def fig6_covid_era_comparison(results, deltas):
    """Figure 6: Pre-COVID vs COVID era comparison (eICU vs eICU-New)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Get eICU and eICU-New data
    eicu_data = deltas[deltas['dataset'].isin(['eicu', 'eicu_new'])].copy()

    if eicu_data.empty:
        print("No eICU comparison data available for fig6")
        return

    # Panel A: Overall comparison by score
    ax1 = axes[0]
    overall = eicu_data[eicu_data['subgroup'] == 'All']

    if not overall.empty:
        pivot = overall.pivot(index='score', columns='dataset_name', values='delta')
        pivot.plot(kind='bar', ax=ax1, width=0.7)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_title('A. Overall AUC Change')
        ax1.set_xlabel('Score')
        ax1.set_ylabel('AUC Change')
        ax1.legend(title='Era')
        ax1.tick_params(axis='x', rotation=45)

    # Panel B: Age comparison (OASIS)
    ax2 = axes[1]
    age = eicu_data[(eicu_data['subgroup_type'] == 'Age') & (eicu_data['score'] == 'oasis')]

    if not age.empty:
        pivot = age.pivot(index='subgroup', columns='dataset_name', values='delta')
        pivot = pivot.reindex(['18-44', '45-64', '65-79', '80+'])
        pivot.plot(kind='bar', ax=ax2, width=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_title('B. OASIS AUC Change by Age')
        ax2.set_xlabel('Age Group')
        ax2.set_ylabel('AUC Change')
        ax2.legend(title='Era')
        ax2.tick_params(axis='x', rotation=0)

    # Panel C: Race comparison (OASIS)
    ax3 = axes[2]
    race = eicu_data[(eicu_data['subgroup_type'] == 'Race') & (eicu_data['score'] == 'oasis')]

    if not race.empty:
        pivot = race.pivot(index='subgroup', columns='dataset_name', values='delta')
        pivot = pivot.reindex([r for r in ['White', 'Black', 'Hispanic', 'Asian'] if r in pivot.index])
        pivot.plot(kind='bar', ax=ax3, width=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_title('C. OASIS AUC Change by Race')
        ax3.set_xlabel('Race')
        ax3.set_ylabel('AUC Change')
        ax3.legend(title='Era')
        ax3.tick_params(axis='x', rotation=45)

    plt.suptitle('Pre-COVID (2014-15) vs COVID Era (2020-21) ICU Score Performance', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig6_covid_era_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: fig6_covid_era_comparison.png")


def fig7_money_figure(results, deltas):
    """Figure 7: The 'money figure' - key findings summary."""
    fig = plt.figure(figsize=(18, 14))

    # Create grid with more spacing
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4)

    # Panel A: Diverging slopes - age (top left, spans 2 cols)
    ax1 = fig.add_subplot(gs[0, :2])
    age_data = results[(results['subgroup_type'] == 'Age') & (results['score'] == 'oasis')].copy()

    for dataset in ['mimiciv', 'amsterdam_icu', 'zhejiang']:
        df_sub = age_data[age_data['dataset'] == dataset]
        if df_sub.empty:
            continue

        # Get first and last periods
        periods = sorted(df_sub['time_period'].unique())
        if len(periods) < 2:
            continue

        first_period, last_period = periods[0], periods[-1]
        first_data = df_sub[df_sub['time_period'] == first_period]
        last_data = df_sub[df_sub['time_period'] == last_period]

        color = DATASET_COLORS.get(dataset, '#666666')
        label = df_sub['dataset_name'].iloc[0].split(' (')[0]

        for age_group in ['18-44', '45-64', '65-79', '80+']:
            first_auc = first_data[first_data['subgroup'] == age_group]['auc'].values
            last_auc = last_data[last_data['subgroup'] == age_group]['auc'].values

            if len(first_auc) > 0 and len(last_auc) > 0:
                alpha = 1.0 if age_group in ['18-44', '80+'] else 0.3
                lw = 2.5 if age_group in ['18-44', '80+'] else 1.0
                ax1.plot([0, 1], [first_auc[0], last_auc[0]], 'o-',
                        color=color, alpha=alpha, linewidth=lw,
                        label=f'{label} {age_group}' if age_group in ['18-44', '80+'] else '')

    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['First Period', 'Last Period'])
    ax1.set_ylabel('OASIS AUC')
    ax1.set_title('A. Age Groups Diverge: Young Improve, Elderly Decline')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    ax1.set_ylim(0.5, 0.95)

    # Panel B: Race disparities (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    race_deltas = deltas[(deltas['subgroup_type'] == 'Race') & (deltas['score'] == 'oasis')].copy()

    if not race_deltas.empty:
        # Group by race, get mean delta
        race_avg = race_deltas.groupby('subgroup')['delta'].mean().sort_values()
        colors = [SUBGROUP_COLORS.get(r, '#666') for r in race_avg.index]
        bars = ax2.barh(race_avg.index, race_avg.values, color=colors)
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Average AUC Change')
        ax2.set_title('B. OASIS Drift by Race')

        # Add value labels
        for bar, val in zip(bars, race_avg.values):
            ax2.text(val + 0.002 if val >= 0 else val - 0.002,
                    bar.get_y() + bar.get_height()/2,
                    f'{val:+.3f}', va='center', ha='left' if val >= 0 else 'right', fontsize=9)

    # Panel C: COVID impact (middle row, spans 2 cols)
    ax3 = fig.add_subplot(gs[1, :2])

    # Compare eICU (pre-COVID) vs eICU-New (COVID era)
    pre_covid = results[(results['dataset'] == 'eicu') & (results['subgroup'] == 'All')].copy()
    covid = results[(results['dataset'] == 'eicu_new') & (results['subgroup'] == 'All')].copy()

    if not pre_covid.empty and not covid.empty:
        scores = pre_covid['score'].unique()
        x = np.arange(len(scores))
        width = 0.35

        pre_means = [pre_covid[pre_covid['score'] == s]['auc'].mean() for s in scores]
        covid_means = [covid[covid['score'] == s]['auc'].mean() for s in scores]

        ax3.bar(x - width/2, pre_means, width, label='Pre-COVID (2014-15)', color='#1f77b4')
        ax3.bar(x + width/2, covid_means, width, label='COVID Era (2020-21)', color='#d62728')

        ax3.set_xticks(x)
        ax3.set_xticklabels([s.upper() for s in scores])
        ax3.set_ylabel('Mean AUC')
        ax3.set_title('C. COVID Era Impact on Score Performance')
        ax3.legend()
        ax3.set_ylim(0.6, 0.9)

    # Panel D: Key statistics (middle right)
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')

    # Calculate key stats
    stats_text = "KEY FINDINGS\n" + "="*30 + "\n\n"

    if deltas is not None and not deltas.empty:
        worst_overall = deltas[deltas['subgroup'] == 'All'].nsmallest(3, 'delta')
        best_overall = deltas[deltas['subgroup'] == 'All'].nlargest(3, 'delta')

        stats_text += "Largest Improvements:\n"
        for _, row in best_overall.iterrows():
            stats_text += f"  {row['dataset_name'].split(' (')[0]} ({row['score'].upper()}): {row['delta']:+.3f}\n"

        stats_text += "\nLargest Declines:\n"
        for _, row in worst_overall.iterrows():
            stats_text += f"  {row['dataset_name'].split(' (')[0]} ({row['score'].upper()}): {row['delta']:+.3f}\n"

        # Subgroup extremes
        worst_subgroup = deltas.nsmallest(1, 'delta').iloc[0]
        best_subgroup = deltas.nlargest(1, 'delta').iloc[0]

        stats_text += f"\nMost Vulnerable Subgroup:\n"
        stats_text += f"  {worst_subgroup['dataset_name'].split(' (')[0]}\n"
        stats_text += f"  {worst_subgroup['subgroup']} ({worst_subgroup['score'].upper()})\n"
        stats_text += f"  AUC Change: {worst_subgroup['delta']:+.3f}\n"

    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel E: Comprehensive heatmap (bottom row)
    ax5 = fig.add_subplot(gs[2, :])

    if deltas is not None and not deltas.empty:
        # Filter for OASIS only to keep it readable
        oasis_deltas = deltas[deltas['score'] == 'oasis'].copy()
        oasis_deltas['label'] = oasis_deltas['dataset_name'].str.split(r' \(').str[0] + '\n' + oasis_deltas['subgroup']

        pivot = oasis_deltas.pivot_table(index='subgroup', columns='dataset_name', values='delta', aggfunc='mean')
        pivot.columns = [c.split(' (')[0] for c in pivot.columns]

        # Reorder rows
        row_order = ['All', '18-44', '45-64', '65-79', '80+', 'Male', 'Female', 'White', 'Black', 'Hispanic', 'Asian']
        pivot = pivot.reindex([r for r in row_order if r in pivot.index])

        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                   ax=ax5, cbar_kws={'label': 'OASIS AUC Change'}, vmin=-0.15, vmax=0.15)
        ax5.set_title('D. OASIS Performance Change Across All Datasets and Subgroups')
        ax5.set_xlabel('Dataset')
        ax5.set_ylabel('Subgroup')

    plt.suptitle('Multi-Dataset Analysis of ICU Score Drift by Subgroup', fontsize=16, y=0.98)
    fig.savefig(FIGURES_DIR / 'fig7_money_figure.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: fig7_money_figure.png")


def create_summary_table(results, deltas):
    """Create summary tables for supplementary materials."""
    if deltas is None or deltas.empty:
        print("No delta data for summary table")
        return

    # Table 1: Overall drift by dataset and score
    overall = deltas[deltas['subgroup'] == 'All'].copy()
    table1 = overall.pivot(index='dataset_name', columns='score', values='delta')
    table1 = table1.round(3)
    table1.to_csv(OUTPUT_DIR / 'table1_overall_drift.csv')
    print("Saved: table1_overall_drift.csv")

    # Table 2: Subgroup drift summary (OASIS)
    oasis_deltas = deltas[deltas['score'] == 'oasis'].copy()
    table2 = oasis_deltas.pivot(index='subgroup', columns='dataset_name', values='delta')
    table2 = table2.round(3)
    table2.to_csv(OUTPUT_DIR / 'table2_oasis_subgroup_drift.csv')
    print("Saved: table2_oasis_subgroup_drift.csv")

    # Table 3: Sample sizes
    sizes = results[results['score'] == 'oasis'][['dataset_name', 'subgroup', 'time_period', 'n']].copy()
    sizes_pivot = sizes.groupby(['dataset_name', 'subgroup'])['n'].sum().unstack(fill_value=0)
    sizes_pivot.to_csv(OUTPUT_DIR / 'table3_sample_sizes.csv')
    print("Saved: table3_sample_sizes.csv")


def main():
    """Generate all figures."""
    print("="*60)
    print("GENERATING CROSS-DATASET DRIFT ANALYSIS FIGURES")
    print("="*60)

    # Load data
    results, deltas = load_data()
    if results is None:
        return

    # Generate figures
    print("\nGenerating figures...")

    fig1_overall_drift_by_dataset(results, deltas)
    fig2_age_stratified_drift(results, deltas)
    fig3_race_disparities(results, deltas)
    fig4_drift_delta_summary(deltas)
    fig5_score_comparison_heatmap(deltas)
    fig6_covid_era_comparison(results, deltas)
    fig7_money_figure(results, deltas)

    # Create summary tables
    print("\nGenerating summary tables...")
    create_summary_table(results, deltas)

    print("\n" + "="*60)
    print("ALL FIGURES GENERATED SUCCESSFULLY")
    print("="*60)
    print(f"\nFigures saved to: {FIGURES_DIR}")
    print(f"Tables saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
