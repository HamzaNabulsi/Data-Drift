"""
Subgroup-Specific Drift Analysis Figures
=========================================
Generates publication-quality visualizations showing NON-UNIFORM model drift
across different patient subgroups.

Per Leo's feedback (Dec 21, 2025):
- Each dataset must be analyzed SEPARATELY (no cross-dataset comparisons in main)
- Main figures: One comprehensive figure per dataset showing subgroup drift
- Supplementary: Cross-dataset comparisons moved here

Main Figures (PER-DATASET analysis):
- fig1_mimic_combined: MIMIC Combined subgroup drift analysis
- fig2_eicu_combined: eICU Combined subgroup drift analysis
- fig3_saltz: Saltz (Netherlands) subgroup drift analysis
- fig4_zhejiang: Zhejiang (China) subgroup drift analysis
- fig5_money_figure: Summary of key findings (kept for visual impact)

Supplementary Figures (figures/supplementary/):
- figS1: MIMIC mouthcare cohort (care phenotypes)
- figS2: MIMIC mechanical ventilation cohort (care phenotypes)
- figS3: Overall drift comparison (cross-dataset)
- figS4: Age-stratified drift comparison (cross-dataset)
- figS5: Race disparities comparison (cross-dataset)
- figS6: Forest plot of significant findings (cross-dataset)
- figS7: Gender drift patterns (cross-dataset)

Output:
- figures/fig1_mimic_combined.png
- figures/fig2_eicu_combined.png
- figures/fig3_saltz.png
- figures/fig4_zhejiang.png
- figures/fig5_money_figure.png
- figures/supplementary/figS1_mimic_mouthcare.png
- figures/supplementary/figS2_mimic_mechvent.png
- figures/supplementary/figS3_overall_drift_comparison.png
- figures/supplementary/figS4_age_comparison.png
- figures/supplementary/figS5_race_comparison.png
- figures/supplementary/figS6_significance_forest.png
- figures/supplementary/figS7_gender_comparison.png
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
import shutil

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
SUPPLEMENTARY_FIGURES_DIR = FIGURES_DIR / 'supplementary'
SUPPLEMENTARY_FIGURES_DIR.mkdir(exist_ok=True)

# Colors
DATASET_COLORS = {
    'mimiciii': '#17becf',    # Cyan - historical baseline
    'mimiciv': '#1f77b4',     # Blue
    'mimic_combined': '#1f77b4',  # Blue - same as MIMIC-IV (combined dataset)
    'saltz': '#ff7f0e',
    'zhejiang': '#2ca02c',
    'eicu': '#d62728',
    'eicu_new': '#9467bd',
    'eicu_combined': '#d62728'  # Red - same as eICU (combined dataset)
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
    """Load drift analysis results from per-dataset files.

    Data is stored per-dataset (not as cross-dataset comparisons) per Leo's feedback.
    This function combines them for figure generation only.
    """
    # Load from per-dataset subdirectories
    datasets = ['mimic_combined', 'saltz', 'zhejiang', 'eicu_combined']
    all_results = []
    all_deltas = []

    for dataset in datasets:
        results_file = OUTPUT_DIR / dataset / 'drift_results.csv'
        deltas_file = OUTPUT_DIR / dataset / 'drift_deltas.csv'

        if results_file.exists():
            df = pd.read_csv(results_file)
            all_results.append(df)

        if deltas_file.exists():
            df = pd.read_csv(deltas_file)
            all_deltas.append(df)

    if not all_results:
        print("ERROR: No per-dataset results found. Run batch_analysis.py first.")
        return None, None

    results = pd.concat(all_results, ignore_index=True)
    deltas = pd.concat(all_deltas, ignore_index=True) if all_deltas else None

    print(f"Loaded {len(results):,} result rows from {len(all_results)} datasets")
    if deltas is not None:
        print(f"Loaded {len(deltas):,} delta comparisons")

    return results, deltas


def figS3_overall_drift_by_dataset(results, deltas):
    """Supplementary Figure S3: Overall drift comparison across datasets and scores.

    NOTE: This is moved to supplementary per Leo's feedback:
    "We don't want overall, we want it to show that across different subgroups,
    we showed non-uniform model drift."
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: AUC over time for each dataset (OASIS as primary score)
    ax1 = axes[0]
    oasis_overall = results[(results['score'] == 'oasis') & (results['subgroup'] == 'All')].copy()

    has_ci = 'auc_ci_lower' in oasis_overall.columns

    for dataset in oasis_overall['dataset'].unique():
        df_sub = oasis_overall[oasis_overall['dataset'] == dataset].sort_values('time_period')
        color = DATASET_COLORS.get(dataset, '#666666')
        label = df_sub['dataset_name'].iloc[0].split(' (')[0]
        x = range(len(df_sub))

        # Plot line with markers
        ax1.plot(x, df_sub['auc'], 'o-', label=label,
                color=color, markersize=8, linewidth=2)

        # Add shaded CI region if available
        if has_ci and not df_sub['auc_ci_lower'].isna().all():
            ax1.fill_between(x, df_sub['auc_ci_lower'], df_sub['auc_ci_upper'],
                           color=color, alpha=0.15)

    ax1.set_xlabel('Time Period (Ordered)')
    ax1.set_ylabel('AUC')
    ax1.set_title('A. Overall OASIS Performance Over Time' + (' (95% CI)' if has_ci else ''))
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
    # Save to SUPPLEMENTARY folder
    fig.savefig(SUPPLEMENTARY_FIGURES_DIR / 'figS3_overall_drift_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: supplementary/figS3_overall_drift_comparison.png")


def create_per_dataset_figure(results, deltas, dataset_key, fig_num):
    """Create a comprehensive figure for a SINGLE dataset showing non-uniform subgroup drift.

    Per Leo's feedback (Dec 21, 2025): Each dataset must be analyzed separately.
    This is the MAIN figure format - showing subgroup drift within one dataset.

    Layout:
    - Panel A: Age group drift (line plot over time or bar chart)
    - Panel B: Gender drift
    - Panel C: Race/ethnicity drift (if available)
    - Panel D: Subgroup delta summary (bar chart)
    """
    # Filter data for this dataset
    ds_results = results[results['dataset'] == dataset_key].copy()
    ds_deltas = deltas[deltas['dataset'] == dataset_key].copy() if deltas is not None else None

    if ds_results.empty:
        print(f"No data for dataset: {dataset_key}")
        return

    dataset_name = ds_results['dataset_name'].iloc[0].split(' (')[0]
    has_ci = 'auc_ci_lower' in ds_results.columns

    # Determine layout based on available data
    has_race = 'Race' in ds_results['subgroup_type'].unique()

    if has_race:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Use OASIS as primary score (or first available)
    primary_score = 'oasis' if 'oasis' in ds_results['score'].unique() else ds_results['score'].iloc[0]
    score_data = ds_results[ds_results['score'] == primary_score]

    time_periods = sorted(score_data['time_period'].unique())
    n_periods = len(time_periods)

    # ========== Panel A: Age Group Drift ==========
    ax_age = axes[0, 0]
    age_data = score_data[score_data['subgroup_type'] == 'Age']

    if not age_data.empty:
        if n_periods > 1:
            # Line plot for multi-period
            for age_group in ['18-44', '45-64', '65-79', '80+']:
                age_df = age_data[age_data['subgroup'] == age_group].sort_values('time_period')
                if not age_df.empty:
                    color = SUBGROUP_COLORS.get(age_group, '#666666')
                    x = range(len(age_df))
                    ax_age.plot(x, age_df['auc'], 'o-', label=age_group,
                               color=color, markersize=8, linewidth=2)
                    if has_ci and not age_df['auc_ci_lower'].isna().all():
                        ax_age.fill_between(x, age_df['auc_ci_lower'], age_df['auc_ci_upper'],
                                           color=color, alpha=0.15)
            # Use actual year labels on x-axis
            ax_age.set_xticks(range(n_periods))
            ax_age.set_xticklabels(time_periods, rotation=45, ha='right', fontsize=8)
            ax_age.set_xlabel('Year')
            ax_age.legend(title='Age Group', loc='best', fontsize=9)
        else:
            # Bar chart for single period
            age_groups = ['18-44', '45-64', '65-79', '80+']
            aucs = [age_data[age_data['subgroup'] == ag]['auc'].values[0]
                    if not age_data[age_data['subgroup'] == ag].empty else np.nan
                    for ag in age_groups]
            colors = [SUBGROUP_COLORS.get(ag, '#666') for ag in age_groups]
            ax_age.bar(age_groups, aucs, color=colors)
            ax_age.set_xlabel('Age Group')

        ax_age.set_ylabel(f'{primary_score.upper()} AUC')
        ax_age.set_title('A. Performance by Age Group' + (' (95% CI)' if has_ci and n_periods > 1 else ''))
        ax_age.set_ylim(0.5, 1.0)
    else:
        ax_age.text(0.5, 0.5, 'No age data available', ha='center', va='center')
        ax_age.set_title('A. Performance by Age Group')

    # ========== Panel B: Gender Drift ==========
    ax_gender = axes[0, 1]
    gender_data = score_data[score_data['subgroup_type'] == 'Gender']

    if not gender_data.empty:
        if n_periods > 1:
            for gender in ['Male', 'Female']:
                gender_df = gender_data[gender_data['subgroup'] == gender].sort_values('time_period')
                if not gender_df.empty:
                    color = SUBGROUP_COLORS.get(gender, '#666666')
                    x = range(len(gender_df))
                    ax_gender.plot(x, gender_df['auc'], 'o-', label=gender,
                                  color=color, markersize=8, linewidth=2)
                    if has_ci and not gender_df['auc_ci_lower'].isna().all():
                        ax_gender.fill_between(x, gender_df['auc_ci_lower'], gender_df['auc_ci_upper'],
                                              color=color, alpha=0.15)
            # Use actual year labels on x-axis
            ax_gender.set_xticks(range(n_periods))
            ax_gender.set_xticklabels(time_periods, rotation=45, ha='right', fontsize=8)
            ax_gender.set_xlabel('Year')
            ax_gender.legend(title='Gender', loc='best', fontsize=9)
        else:
            genders = ['Male', 'Female']
            aucs = [gender_data[gender_data['subgroup'] == g]['auc'].values[0]
                    if not gender_data[gender_data['subgroup'] == g].empty else np.nan
                    for g in genders]
            colors = [SUBGROUP_COLORS.get(g, '#666') for g in genders]
            ax_gender.bar(genders, aucs, color=colors)
            ax_gender.set_xlabel('Gender')

        ax_gender.set_ylabel(f'{primary_score.upper()} AUC')
        ax_gender.set_title('B. Performance by Gender' + (' (95% CI)' if has_ci and n_periods > 1 else ''))
        ax_gender.set_ylim(0.5, 1.0)
    else:
        ax_gender.text(0.5, 0.5, 'No gender data available', ha='center', va='center')
        ax_gender.set_title('B. Performance by Gender')

    # ========== Panel C: Race/Ethnicity Drift (or Overall) ==========
    ax_race = axes[1, 0]
    race_data = score_data[score_data['subgroup_type'] == 'Race']

    if not race_data.empty and has_race:
        if n_periods > 1:
            for race in ['White', 'Black', 'Hispanic', 'Asian']:
                race_df = race_data[race_data['subgroup'] == race].sort_values('time_period')
                if not race_df.empty:
                    color = SUBGROUP_COLORS.get(race, '#666666')
                    x = range(len(race_df))
                    ax_race.plot(x, race_df['auc'], 'o-', label=race,
                                color=color, markersize=8, linewidth=2)
                    if has_ci and not race_df['auc_ci_lower'].isna().all():
                        ax_race.fill_between(x, race_df['auc_ci_lower'], race_df['auc_ci_upper'],
                                            color=color, alpha=0.15)
            # Use actual year labels on x-axis
            ax_race.set_xticks(range(n_periods))
            ax_race.set_xticklabels(time_periods, rotation=45, ha='right', fontsize=8)
            ax_race.set_xlabel('Year')
            ax_race.legend(title='Race/Ethnicity', loc='best', fontsize=9)
        else:
            races = ['White', 'Black', 'Hispanic', 'Asian']
            aucs = [race_data[race_data['subgroup'] == r]['auc'].values[0]
                    if not race_data[race_data['subgroup'] == r].empty else np.nan
                    for r in races]
            colors = [SUBGROUP_COLORS.get(r, '#666') for r in races]
            ax_race.bar(races, aucs, color=colors)
            ax_race.set_xlabel('Race/Ethnicity')

        ax_race.set_ylabel(f'{primary_score.upper()} AUC')
        ax_race.set_title('C. Performance by Race/Ethnicity' + (' (95% CI)' if has_ci and n_periods > 1 else ''))
        ax_race.set_ylim(0.5, 1.0)
    else:
        # Show overall performance if no race data
        overall_data = score_data[score_data['subgroup'] == 'All'].sort_values('time_period')
        if not overall_data.empty and n_periods > 1:
            x = range(len(overall_data))
            ax_race.plot(x, overall_data['auc'], 'o-', color='#666666', markersize=8, linewidth=2)
            if has_ci and not overall_data['auc_ci_lower'].isna().all():
                ax_race.fill_between(x, overall_data['auc_ci_lower'], overall_data['auc_ci_upper'],
                                    color='#666666', alpha=0.15)
            # Use actual year labels on x-axis
            ax_race.set_xticks(range(n_periods))
            ax_race.set_xticklabels(time_periods, rotation=45, ha='right', fontsize=8)
            ax_race.set_xlabel('Year')
        ax_race.set_ylabel(f'{primary_score.upper()} AUC')
        ax_race.set_title('C. Overall Performance' + (' (95% CI)' if has_ci else '') + ' (No race data)')
        ax_race.set_ylim(0.5, 1.0)

    # ========== Panel D: Subgroup Delta Summary (Bar Chart) ==========
    ax_delta = axes[1, 1]

    if ds_deltas is not None and not ds_deltas.empty:
        # Filter for primary score and exclude 'All'
        score_deltas = ds_deltas[(ds_deltas['score'] == primary_score) & (ds_deltas['subgroup'] != 'All')].copy()

        if not score_deltas.empty:
            # Sort by delta
            score_deltas = score_deltas.sort_values('delta')

            # Create labels
            score_deltas['label'] = score_deltas['subgroup_type'] + ': ' + score_deltas['subgroup']

            # Color by significance
            colors = []
            for _, row in score_deltas.iterrows():
                if row.get('significant', False):
                    colors.append('#2ca02c' if row['delta'] > 0 else '#d62728')
                else:
                    colors.append('#999999')

            y_pos = np.arange(len(score_deltas))
            bars = ax_delta.barh(y_pos, score_deltas['delta'], color=colors, alpha=0.8)
            ax_delta.set_yticks(y_pos)
            ax_delta.set_yticklabels(score_deltas['label'], fontsize=9)
            ax_delta.axvline(x=0, color='black', linewidth=1)
            ax_delta.set_xlabel(f'{primary_score.upper()} AUC Change (First → Last Period)')
            ax_delta.set_title('D. Subgroup Drift Summary (* = p<0.05)')

            # Add significance markers
            for i, (_, row) in enumerate(score_deltas.iterrows()):
                if row.get('significant', False):
                    x_pos = row['delta'] + 0.005 if row['delta'] >= 0 else row['delta'] - 0.005
                    ha = 'left' if row['delta'] >= 0 else 'right'
                    ax_delta.text(x_pos, i, '*', va='center', ha=ha, fontsize=12, fontweight='bold')
        else:
            ax_delta.text(0.5, 0.5, 'No delta data available', ha='center', va='center')
            ax_delta.set_title('D. Subgroup Drift Summary')
    else:
        ax_delta.text(0.5, 0.5, 'No delta data available', ha='center', va='center')
        ax_delta.set_title('D. Subgroup Drift Summary')

    # Finalize
    plt.suptitle(f'{dataset_name}: Non-Uniform Subgroup Drift Analysis', fontsize=16, y=1.02)
    plt.tight_layout()

    # Save
    output_name = f'fig{fig_num}_{dataset_key}.png'
    fig.savefig(FIGURES_DIR / output_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_name}")


def figS4_age_stratified_comparison(results, deltas):
    """Supplementary Figure S4: Age-stratified drift comparison across datasets.

    NOTE: Moved to supplementary per Leo's feedback - cross-dataset comparisons
    should not be in main figures.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Get datasets with age data
    age_data = results[(results['subgroup_type'] == 'Age') & (results['score'] == 'oasis')].copy()
    # Include all main analysis datasets
    dataset_order = ['mimic_combined', 'saltz', 'zhejiang', 'eicu_combined']
    datasets = [d for d in dataset_order if d in age_data['dataset'].unique()]
    has_ci = 'auc_ci_lower' in age_data.columns

    for idx, dataset in enumerate(datasets):
        row, col = divmod(idx, 3)
        ax = axes[row, col]

        df_sub = age_data[age_data['dataset'] == dataset]
        dataset_name = df_sub['dataset_name'].iloc[0].split(' (')[0]
        time_periods = sorted(df_sub['time_period'].unique())

        # For single-period datasets, show bar chart
        if len(time_periods) == 1:
            age_groups = ['18-44', '45-64', '65-79', '80+']
            aucs = []
            ci_lower = []
            ci_upper = []
            colors = []

            for ag in age_groups:
                age_df = df_sub[df_sub['subgroup'] == ag]
                if not age_df.empty:
                    aucs.append(age_df['auc'].values[0])
                    if has_ci:
                        ci_lower.append(age_df['auc_ci_lower'].values[0])
                        ci_upper.append(age_df['auc_ci_upper'].values[0])
                    colors.append(SUBGROUP_COLORS.get(ag, '#666'))
                else:
                    aucs.append(np.nan)
                    ci_lower.append(np.nan)
                    ci_upper.append(np.nan)
                    colors.append('#666')

            x = np.arange(len(age_groups))
            if has_ci and ci_lower:
                yerr = np.array([np.array(aucs) - np.array(ci_lower),
                                np.array(ci_upper) - np.array(aucs)])
                yerr = np.nan_to_num(yerr, nan=0)
                yerr = np.clip(yerr, 0, None)  # Error bars must be non-negative
                ax.bar(x, aucs, color=colors, yerr=yerr, capsize=4)
            else:
                ax.bar(x, aucs, color=colors)
            ax.set_xticks(x)
            ax.set_xticklabels(age_groups)
            ax.set_title(f'{dataset_name} (Single Period)')
        else:
            # Multi-period: line chart
            for age_group in ['18-44', '45-64', '65-79', '80+']:
                age_df = df_sub[df_sub['subgroup'] == age_group].sort_values('time_period')
                if not age_df.empty:
                    color = SUBGROUP_COLORS.get(age_group, '#666666')
                    x = range(len(age_df))
                    ax.plot(x, age_df['auc'], 'o-', label=age_group,
                           color=color, markersize=6, linewidth=1.5)

                    # Add CI shading
                    if has_ci and not age_df['auc_ci_lower'].isna().all():
                        ax.fill_between(x, age_df['auc_ci_lower'], age_df['auc_ci_upper'],
                                       color=color, alpha=0.15)

            ax.set_xlabel('Time Period')
            ax.legend(title='Age', loc='lower right', fontsize=8)
            ax.set_title(f'{dataset_name}')

        ax.set_ylabel('AUC')
        ax.set_ylim(0.4, 1.0)

    # Hide unused axes
    for idx in range(len(datasets), 6):
        row, col = divmod(idx, 3)
        axes[row, col].axis('off')

    plt.suptitle('OASIS Performance by Age Group (Cross-Dataset Comparison)' + (' (95% CI)' if has_ci else ''), fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(SUPPLEMENTARY_FIGURES_DIR / 'figS4_age_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: supplementary/figS4_age_comparison.png")


def figS5_race_comparison(results, deltas):
    """Supplementary Figure S5: Race/ethnicity disparities comparison across datasets.

    NOTE: Moved to supplementary per Leo's feedback - cross-dataset comparisons
    should not be in main figures.
    """
    # Only use datasets with race data AND multiple time periods for meaningful drift analysis
    race_data = results[(results['subgroup_type'] == 'Race') & (results['score'] == 'oasis')].copy()

    # Filter to datasets with >1 time period
    race_datasets_multi = []
    for dataset in ['mimic_combined', 'saltz', 'zhejiang', 'eicu_combined']:
        df_sub = race_data[race_data['dataset'] == dataset]
        if not df_sub.empty and len(df_sub['time_period'].unique()) > 1:
            race_datasets_multi.append(dataset)

    # Create appropriate grid based on number of datasets
    n_datasets = len(race_datasets_multi)
    if n_datasets == 0:
        print("No multi-period race data available for fig3")
        return
    elif n_datasets <= 2:
        fig, axes = plt.subplots(1, n_datasets, figsize=(7 * n_datasets, 6))
        if n_datasets == 1:
            axes = [axes]
    else:
        # Use 1 row with MIMIC-IV spanning top, eICU datasets on bottom
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
        axes = [
            fig.add_subplot(gs[0, :]),  # MIMIC-IV spans full top row
            fig.add_subplot(gs[1, 0]),   # eICU bottom left
            fig.add_subplot(gs[1, 1])    # eICU-New bottom right
        ]

    has_ci = 'auc_ci_lower' in race_data.columns

    for idx, dataset in enumerate(race_datasets_multi):
        ax = axes[idx]
        df_sub = race_data[race_data['dataset'] == dataset]

        if df_sub.empty:
            ax.axis('off')
            continue

        dataset_name = df_sub['dataset_name'].iloc[0].split(' (')[0]

        # Multi-period: line chart
        for race in ['White', 'Black', 'Hispanic', 'Asian']:
            race_df = df_sub[df_sub['subgroup'] == race].sort_values('time_period')
            if not race_df.empty:
                color = SUBGROUP_COLORS.get(race, '#666666')
                x = range(len(race_df))
                ax.plot(x, race_df['auc'], 'o-', label=race,
                       color=color, markersize=6, linewidth=1.5)

                # Add CI shading
                if has_ci and not race_df['auc_ci_lower'].isna().all():
                    ax.fill_between(x, race_df['auc_ci_lower'], race_df['auc_ci_upper'],
                                   color=color, alpha=0.15)

        ax.set_title(f'{dataset_name}')
        ax.set_xlabel('Time Period')
        ax.set_ylabel('AUC')
        ax.legend(title='Race', loc='lower right', fontsize=8)
        ax.set_ylim(0.6, 0.9)

    plt.suptitle('OASIS Performance by Race/Ethnicity (Cross-Dataset Comparison)' + (' (95% CI)' if has_ci else ''), fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(SUPPLEMENTARY_FIGURES_DIR / 'figS5_race_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: supplementary/figS5_race_comparison.png")


def _plot_bar_with_ci(ax, data, x_col, y_col, hue_col, colors, yerr_lower=None, yerr_upper=None):
    """Helper to plot grouped bar chart with optional error bars."""
    pivot = data.pivot(index=x_col, columns=hue_col, values=y_col)

    # Get error data if available
    has_ci = yerr_lower is not None and yerr_upper is not None
    if has_ci:
        pivot_lower = data.pivot(index=x_col, columns=hue_col, values=yerr_lower)
        pivot_upper = data.pivot(index=x_col, columns=hue_col, values=yerr_upper)

    n_groups = len(pivot.index)
    n_bars = len(pivot.columns)
    bar_width = 0.8 / n_bars
    x = np.arange(n_groups)

    for i, col in enumerate(pivot.columns):
        offset = (i - n_bars/2 + 0.5) * bar_width
        values = pivot[col].values
        color = colors[i] if i < len(colors) else '#666666'

        if has_ci and col in pivot_lower.columns:
            # Compute asymmetric error bars
            lower = pivot_lower[col].values
            upper = pivot_upper[col].values
            yerr = np.array([values - lower, upper - values])
            yerr = np.nan_to_num(yerr, nan=0)
            yerr = np.clip(yerr, 0, None)  # Error bars must be non-negative
            ax.bar(x + offset, values, bar_width, label=col, color=color,
                   yerr=yerr, capsize=3, error_kw={'elinewidth': 1, 'capthick': 1})
        else:
            ax.bar(x + offset, values, bar_width, label=col, color=color)

    ax.set_xticks(x)
    return pivot


def figS8_drift_delta_summary(deltas):
    """Supplementary Figure S8: Summary of drift deltas by subgroup type (cross-dataset).

    NOTE: Moved to supplementary per Leo's feedback.
    """
    if deltas is None or deltas.empty:
        print("No delta data available for fig3")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.35, wspace=0.35)

    has_ci = 'delta_ci_lower' in deltas.columns

    # Panel A: Overall deltas by dataset and score
    ax1 = axes[0, 0]
    overall_deltas = deltas[deltas['subgroup'] == 'All'].copy()

    if not overall_deltas.empty:
        score_order = [c for c in ['oasis', 'sapsii', 'apsiii', 'sofa', 'apachescore'] if c in overall_deltas['score'].unique()]
        overall_deltas = overall_deltas[overall_deltas['score'].isin(score_order)]

        pivot = _plot_bar_with_ci(
            ax1, overall_deltas, 'dataset_name', 'delta', 'score',
            colors=[plt.cm.tab10(i) for i in range(len(score_order))],
            yerr_lower='delta_ci_lower' if has_ci else None,
            yerr_upper='delta_ci_upper' if has_ci else None
        )
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_title('A. Overall AUC Change by Dataset' + (' (95% CI)' if has_ci else ''))
        ax1.set_xlabel('')
        ax1.set_ylabel('AUC Change')
        ax1.legend(title='Score', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax1.set_xticklabels([x.split(' (')[0] for x in pivot.index], rotation=45, ha='right')

    # Panel B: Age subgroup deltas (OASIS)
    ax2 = axes[0, 1]
    age_deltas = deltas[(deltas['subgroup_type'] == 'Age') & (deltas['score'] == 'oasis')].copy()

    if not age_deltas.empty:
        age_order = ['18-44', '45-64', '65-79', '80+']
        age_deltas = age_deltas[age_deltas['subgroup'].isin(age_order)]

        pivot = _plot_bar_with_ci(
            ax2, age_deltas, 'dataset_name', 'delta', 'subgroup',
            colors=[SUBGROUP_COLORS.get(c, '#666') for c in age_order],
            yerr_lower='delta_ci_lower' if has_ci else None,
            yerr_upper='delta_ci_upper' if has_ci else None
        )
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_title('B. OASIS AUC Change by Age Group' + (' (95% CI)' if has_ci else ''))
        ax2.set_xlabel('')
        ax2.set_ylabel('AUC Change')
        ax2.legend(title='Age', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax2.set_xticklabels([x.split(' (')[0] for x in pivot.index], rotation=45, ha='right')

    # Panel C: Gender deltas
    ax3 = axes[1, 0]
    gender_deltas = deltas[(deltas['subgroup_type'] == 'Gender') & (deltas['score'] == 'oasis')].copy()

    if not gender_deltas.empty:
        gender_order = ['Male', 'Female']
        pivot = _plot_bar_with_ci(
            ax3, gender_deltas, 'dataset_name', 'delta', 'subgroup',
            colors=[SUBGROUP_COLORS.get(c, '#666') for c in gender_order],
            yerr_lower='delta_ci_lower' if has_ci else None,
            yerr_upper='delta_ci_upper' if has_ci else None
        )
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_title('C. OASIS AUC Change by Gender' + (' (95% CI)' if has_ci else ''))
        ax3.set_xlabel('')
        ax3.set_ylabel('AUC Change')
        ax3.legend(title='Gender', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax3.set_xticklabels([x.split(' (')[0] for x in pivot.index], rotation=45, ha='right')

    # Panel D: Race deltas (US datasets only)
    ax4 = axes[1, 1]
    race_deltas = deltas[(deltas['subgroup_type'] == 'Race') & (deltas['score'] == 'oasis')].copy()

    if not race_deltas.empty:
        race_order = ['White', 'Black', 'Hispanic', 'Asian']
        race_deltas = race_deltas[race_deltas['subgroup'].isin(race_order)]

        pivot = _plot_bar_with_ci(
            ax4, race_deltas, 'dataset_name', 'delta', 'subgroup',
            colors=[SUBGROUP_COLORS.get(c, '#666') for c in race_order],
            yerr_lower='delta_ci_lower' if has_ci else None,
            yerr_upper='delta_ci_upper' if has_ci else None
        )
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4.set_title('D. OASIS AUC Change by Race' + (' (95% CI)' if has_ci else ''))
        ax4.set_xlabel('')
        ax4.set_ylabel('AUC Change')
        ax4.legend(title='Race', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax4.set_xticklabels([x.split(' (')[0] for x in pivot.index], rotation=45, ha='right')

    plt.tight_layout()
    fig.savefig(SUPPLEMENTARY_FIGURES_DIR / 'figS8_drift_delta_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: supplementary/figS8_drift_delta_summary.png")


def figS9_comprehensive_heatmap(deltas):
    """Supplementary Figure S9: Comprehensive heatmap of all subgroup drifts (cross-dataset).

    NOTE: Moved to supplementary per Leo's feedback.
    """
    if deltas is None or deltas.empty:
        print("No delta data available for fig4")
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
    fig.savefig(SUPPLEMENTARY_FIGURES_DIR / 'figS9_comprehensive_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: supplementary/figS9_comprehensive_heatmap.png")


def fig5_money_figure(results, deltas):
    """Figure 5: The 'money figure' - key findings summary with confidence intervals."""
    fig = plt.figure(figsize=(16, 12))

    # Create grid with more spacing (2x2 layout, bottom row spans full width)
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.35, height_ratios=[1, 1.2])

    has_ci = 'auc_ci_lower' in results.columns

    # Panel A: Diverging slopes - age (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    age_data = results[(results['subgroup_type'] == 'Age') & (results['score'] == 'oasis')].copy()

    # Use mimic_combined instead of mimiciv for main analysis datasets
    for dataset in ['mimic_combined', 'saltz', 'zhejiang']:
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
            first_row = first_data[first_data['subgroup'] == age_group]
            last_row = last_data[last_data['subgroup'] == age_group]

            if len(first_row) > 0 and len(last_row) > 0:
                first_auc = first_row['auc'].values[0]
                last_auc = last_row['auc'].values[0]

                alpha = 1.0 if age_group in ['18-44', '80+'] else 0.3
                lw = 2.5 if age_group in ['18-44', '80+'] else 1.0
                ax1.plot([0, 1], [first_auc, last_auc], 'o-',
                        color=color, alpha=alpha, linewidth=lw,
                        label=f'{label} {age_group}' if age_group in ['18-44', '80+'] else '')

                # Add error bars for CIs if available (only for key age groups)
                if has_ci and age_group in ['18-44', '80+']:
                    first_ci = [first_row['auc_ci_lower'].values[0], first_row['auc_ci_upper'].values[0]]
                    last_ci = [last_row['auc_ci_lower'].values[0], last_row['auc_ci_upper'].values[0]]

                    if not np.isnan(first_ci[0]):
                        yerr_first = [[max(0, first_auc - first_ci[0])], [max(0, first_ci[1] - first_auc)]]
                        ax1.errorbar([0], [first_auc], yerr=yerr_first,
                                    color=color, alpha=alpha, capsize=4, capthick=1.5, fmt='none')
                    if not np.isnan(last_ci[0]):
                        yerr_last = [[max(0, last_auc - last_ci[0])], [max(0, last_ci[1] - last_auc)]]
                        ax1.errorbar([1], [last_auc], yerr=yerr_last,
                                    color=color, alpha=alpha, capsize=4, capthick=1.5, fmt='none')

    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['First Period', 'Last Period'])
    ax1.set_ylabel('OASIS AUC')
    ax1.set_title('A. Age Groups Diverge: Young Improve, Elderly Decline' + (' (95% CI)' if has_ci else ''))
    ax1.legend(loc='upper left', fontsize=7, framealpha=0.9)
    ax1.set_ylim(0.45, 0.95)

    # Panel B: Race disparities (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    race_deltas = deltas[(deltas['subgroup_type'] == 'Race') & (deltas['score'] == 'oasis')].copy()
    has_delta_ci = 'delta_ci_lower' in deltas.columns

    if not race_deltas.empty:
        # Group by race, get mean delta and CI
        race_avg = race_deltas.groupby('subgroup')['delta'].mean().sort_values()
        colors = [SUBGROUP_COLORS.get(r, '#666') for r in race_avg.index]

        # Compute error bars from CI if available
        if has_delta_ci:
            race_ci_lower = race_deltas.groupby('subgroup')['delta_ci_lower'].mean()
            race_ci_upper = race_deltas.groupby('subgroup')['delta_ci_upper'].mean()
            xerr = np.array([
                race_avg.values - race_ci_lower.loc[race_avg.index].values,
                race_ci_upper.loc[race_avg.index].values - race_avg.values
            ])
            xerr = np.nan_to_num(xerr, nan=0)
            xerr = np.clip(xerr, 0, None)  # Error bars must be non-negative
            bars = ax2.barh(race_avg.index, race_avg.values, color=colors,
                           xerr=xerr, capsize=3, error_kw={'elinewidth': 1})
        else:
            bars = ax2.barh(race_avg.index, race_avg.values, color=colors)

        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Average AUC Change')
        ax2.set_title('B. OASIS Drift by Race' + (' (95% CI)' if has_delta_ci else ''))

        # Add value labels
        for bar, val in zip(bars, race_avg.values):
            ax2.text(val + 0.002 if val >= 0 else val - 0.002,
                    bar.get_y() + bar.get_height()/2,
                    f'{val:+.3f}', va='center', ha='left' if val >= 0 else 'right', fontsize=9)

    # Panel C: Comprehensive heatmap (bottom row, spans full width)
    ax3 = fig.add_subplot(gs[1, :])

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
                   ax=ax3, cbar_kws={'label': 'OASIS AUC Change'}, vmin=-0.15, vmax=0.15)
        ax3.set_title('C. OASIS Performance Change Across All Datasets and Subgroups')
        ax3.set_xlabel('Dataset')
        ax3.set_ylabel('Subgroup')

    plt.suptitle('Multi-Dataset Analysis of ICU Score Drift by Subgroup', fontsize=14, y=0.98)
    fig.savefig(FIGURES_DIR / 'fig5_money_figure.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: fig5_money_figure.png")


def figS6_significance_forest_plot(deltas):
    """Supplementary Figure S6: Forest plot of significant drift findings (cross-dataset).

    Shows only statistically significant findings (p < 0.05) ranked by effect size.
    NOTE: Moved to supplementary per Leo's feedback.
    """
    if deltas is None or deltas.empty:
        print("No delta data available for fig6")
        return

    # Filter for significant findings only
    sig_deltas = deltas[deltas['significant'] == True].copy()

    if sig_deltas.empty:
        print("No significant findings for fig6")
        return

    # Create label combining dataset, subgroup, and score
    sig_deltas['label'] = (sig_deltas['dataset_name'].str.split(' \\(').str[0] +
                           ' | ' + sig_deltas['subgroup'] +
                           ' (' + sig_deltas['score'].str.upper() + ')')

    # Sort by delta magnitude (most negative to most positive)
    sig_deltas = sig_deltas.sort_values('delta')

    # Limit to top 30 for readability
    if len(sig_deltas) > 30:
        # Take top 15 improvements and top 15 declines
        top_improve = sig_deltas.nlargest(15, 'delta')
        top_decline = sig_deltas.nsmallest(15, 'delta')
        sig_deltas = pd.concat([top_decline, top_improve]).drop_duplicates()
        sig_deltas = sig_deltas.sort_values('delta')

    fig, ax = plt.subplots(figsize=(12, max(8, len(sig_deltas) * 0.35)))

    y_pos = np.arange(len(sig_deltas))

    # Color by direction: green for improvement, red for decline
    colors = ['#2ca02c' if d > 0 else '#d62728' for d in sig_deltas['delta']]

    # Plot horizontal bars
    bars = ax.barh(y_pos, sig_deltas['delta'], color=colors, alpha=0.7, height=0.7)

    # Add error bars if CI available
    has_ci = 'delta_ci_lower' in sig_deltas.columns
    if has_ci:
        xerr = np.array([
            sig_deltas['delta'].values - sig_deltas['delta_ci_lower'].values,
            sig_deltas['delta_ci_upper'].values - sig_deltas['delta'].values
        ])
        xerr = np.nan_to_num(xerr, nan=0)
        xerr = np.clip(xerr, 0, None)
        ax.errorbar(sig_deltas['delta'], y_pos, xerr=xerr, fmt='none',
                   color='black', capsize=2, capthick=1, elinewidth=1)

    # Add vertical line at zero
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)

    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sig_deltas['label'], fontsize=9)
    ax.set_xlabel('AUC Change (First → Last Period)', fontsize=11)
    ax.set_title('Statistically Significant Drift (p < 0.05, DeLong\'s Test)' +
                 (' with 95% CI' if has_ci else ''), fontsize=13, pad=15)

    # Add p-value annotations
    for i, (_, row) in enumerate(sig_deltas.iterrows()):
        p_val = row['p_value_delong']
        if p_val < 0.001:
            p_str = 'p<0.001'
        else:
            p_str = f'p={p_val:.3f}'

        x_pos = row['delta'] + 0.01 if row['delta'] >= 0 else row['delta'] - 0.01
        ha = 'left' if row['delta'] >= 0 else 'right'
        ax.text(x_pos, i, p_str, va='center', ha=ha, fontsize=7, style='italic')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ca02c', alpha=0.7, label='Improvement'),
        Patch(facecolor='#d62728', alpha=0.7, label='Decline')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    fig.savefig(SUPPLEMENTARY_FIGURES_DIR / 'figS6_significance_forest_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: supplementary/figS6_significance_forest_plot.png")


def figS10_score_comparison_by_age(deltas):
    """Supplementary Figure S10: Score comparison by age across datasets.

    Shows that the non-uniform drift pattern is consistent across different scores.
    NOTE: Moved to supplementary per Leo's feedback.
    """
    if deltas is None or deltas.empty:
        print("No delta data available for fig7")
        return

    # Filter for age subgroups only
    age_deltas = deltas[deltas['subgroup_type'] == 'Age'].copy()

    if age_deltas.empty:
        print("No age delta data for fig7")
        return

    # Create figure with one subplot per dataset
    datasets = age_deltas['dataset'].unique()
    n_datasets = len(datasets)

    fig, axes = plt.subplots(1, n_datasets, figsize=(5 * n_datasets, 6), sharey=True)
    if n_datasets == 1:
        axes = [axes]

    age_order = ['18-44', '45-64', '65-79', '80+']
    score_colors = {'oasis': '#1f77b4', 'sapsii': '#ff7f0e', 'apsiii': '#2ca02c', 'sofa': '#d62728', 'apachescore': '#9467bd'}

    has_ci = 'delta_ci_lower' in age_deltas.columns

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        df_sub = age_deltas[age_deltas['dataset'] == dataset]
        dataset_name = df_sub['dataset_name'].iloc[0].split(' (')[0]

        scores = [s for s in ['oasis', 'sapsii', 'apsiii', 'sofa', 'apachescore'] if s in df_sub['score'].unique()]

        x = np.arange(len(age_order))
        bar_width = 0.8 / len(scores)

        for i, score in enumerate(scores):
            offset = (i - len(scores)/2 + 0.5) * bar_width
            score_data = df_sub[df_sub['score'] == score]

            deltas_vals = []
            ci_lower = []
            ci_upper = []

            for age in age_order:
                age_row = score_data[score_data['subgroup'] == age]
                if not age_row.empty:
                    deltas_vals.append(age_row['delta'].values[0])
                    if has_ci:
                        ci_lower.append(age_row['delta_ci_lower'].values[0])
                        ci_upper.append(age_row['delta_ci_upper'].values[0])
                else:
                    deltas_vals.append(np.nan)
                    ci_lower.append(np.nan)
                    ci_upper.append(np.nan)

            color = score_colors.get(score, '#666666')

            if has_ci and ci_lower:
                yerr = np.array([
                    np.array(deltas_vals) - np.array(ci_lower),
                    np.array(ci_upper) - np.array(deltas_vals)
                ])
                yerr = np.nan_to_num(yerr, nan=0)
                yerr = np.clip(yerr, 0, None)
                ax.bar(x + offset, deltas_vals, bar_width, label=score.upper(),
                       color=color, yerr=yerr, capsize=2)
            else:
                ax.bar(x + offset, deltas_vals, bar_width, label=score.upper(), color=color)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(age_order, rotation=45, ha='right')
        ax.set_title(dataset_name, fontsize=12)
        ax.set_xlabel('Age Group')

        if idx == 0:
            ax.set_ylabel('AUC Change')

        if idx == n_datasets - 1:
            ax.legend(title='Score', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

    plt.suptitle('Drift Patterns by Age Group Across All Scores (Cross-Dataset)' + (' (95% CI)' if has_ci else ''),
                 fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(SUPPLEMENTARY_FIGURES_DIR / 'figS10_score_comparison_by_age.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: supplementary/figS10_score_comparison_by_age.png")


def figS7_gender_comparison(results, deltas):
    """Supplementary Figure S7: Gender-specific drift patterns across datasets.

    Demonstrates that gender drift varies by geographic region.
    NOTE: Moved to supplementary per Leo's feedback.
    """
    if deltas is None or deltas.empty:
        print("No delta data available for fig8")
        return

    # Filter for gender subgroups
    gender_deltas = deltas[(deltas['subgroup_type'] == 'Gender') & (deltas['score'] == 'oasis')].copy()

    if gender_deltas.empty:
        print("No gender delta data for fig8")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    has_ci = 'delta_ci_lower' in gender_deltas.columns

    # Panel A: Bar chart comparing Male vs Female drift by dataset
    ax1 = axes[0]

    datasets = gender_deltas['dataset_name'].unique()
    x = np.arange(len(datasets))
    bar_width = 0.35

    male_data = gender_deltas[gender_deltas['subgroup'] == 'Male']
    female_data = gender_deltas[gender_deltas['subgroup'] == 'Female']

    male_deltas = []
    female_deltas = []
    male_ci = []
    female_ci = []

    for ds in datasets:
        m_row = male_data[male_data['dataset_name'] == ds]
        f_row = female_data[female_data['dataset_name'] == ds]

        male_deltas.append(m_row['delta'].values[0] if not m_row.empty else np.nan)
        female_deltas.append(f_row['delta'].values[0] if not f_row.empty else np.nan)

        if has_ci:
            if not m_row.empty:
                male_ci.append([m_row['delta_ci_lower'].values[0], m_row['delta_ci_upper'].values[0]])
            else:
                male_ci.append([np.nan, np.nan])
            if not f_row.empty:
                female_ci.append([f_row['delta_ci_lower'].values[0], f_row['delta_ci_upper'].values[0]])
            else:
                female_ci.append([np.nan, np.nan])

    if has_ci:
        male_yerr = np.array([
            np.array(male_deltas) - np.array([c[0] for c in male_ci]),
            np.array([c[1] for c in male_ci]) - np.array(male_deltas)
        ])
        female_yerr = np.array([
            np.array(female_deltas) - np.array([c[0] for c in female_ci]),
            np.array([c[1] for c in female_ci]) - np.array(female_deltas)
        ])
        male_yerr = np.nan_to_num(male_yerr, nan=0)
        female_yerr = np.nan_to_num(female_yerr, nan=0)
        male_yerr = np.clip(male_yerr, 0, None)
        female_yerr = np.clip(female_yerr, 0, None)

        ax1.bar(x - bar_width/2, male_deltas, bar_width, label='Male',
                color=SUBGROUP_COLORS['Male'], yerr=male_yerr, capsize=3)
        ax1.bar(x + bar_width/2, female_deltas, bar_width, label='Female',
                color=SUBGROUP_COLORS['Female'], yerr=female_yerr, capsize=3)
    else:
        ax1.bar(x - bar_width/2, male_deltas, bar_width, label='Male', color=SUBGROUP_COLORS['Male'])
        ax1.bar(x + bar_width/2, female_deltas, bar_width, label='Female', color=SUBGROUP_COLORS['Female'])

    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([d.split(' (')[0] for d in datasets], rotation=45, ha='right')
    ax1.set_ylabel('OASIS AUC Change')
    ax1.set_title('A. Gender-Specific Drift by Dataset' + (' (95% CI)' if has_ci else ''))
    ax1.legend(title='Gender')

    # Panel B: Gender gap (Male - Female) by dataset
    ax2 = axes[1]

    gender_gap = np.array(male_deltas) - np.array(female_deltas)
    colors = ['#2ca02c' if g > 0 else '#d62728' for g in gender_gap]

    bars = ax2.barh(range(len(datasets)), gender_gap, color=colors, alpha=0.7)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax2.set_yticks(range(len(datasets)))
    ax2.set_yticklabels([d.split(' (')[0] for d in datasets])
    ax2.set_xlabel('Gender Gap (Male - Female AUC Change)')
    ax2.set_title('B. Gender Gap in Drift')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, gender_gap)):
        if not np.isnan(val):
            x_pos = val + 0.002 if val >= 0 else val - 0.002
            ha = 'left' if val >= 0 else 'right'
            ax2.text(x_pos, i, f'{val:+.3f}', va='center', ha=ha, fontsize=9)

    plt.suptitle('Gender-Specific Drift Patterns in OASIS Performance (Cross-Dataset)', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(SUPPLEMENTARY_FIGURES_DIR / 'figS7_gender_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: supplementary/figS7_gender_comparison.png")


def figS11_temporal_trajectory(results):
    """Supplementary Figure S11: Temporal AUC trajectories across datasets.

    Plots the actual AUC values over all time periods for key subgroups.
    NOTE: Moved to supplementary per Leo's feedback.
    """
    if results is None or results.empty:
        print("No results data for fig9")
        return

    # Focus on OASIS score
    oasis_data = results[results['score'] == 'oasis'].copy()

    # Get datasets with multiple time periods
    multi_period_datasets = []
    for dataset in oasis_data['dataset'].unique():
        df_sub = oasis_data[oasis_data['dataset'] == dataset]
        if len(df_sub['time_period'].unique()) > 2:
            multi_period_datasets.append(dataset)

    if not multi_period_datasets:
        print("No multi-period datasets for fig9")
        return

    n_datasets = len(multi_period_datasets)
    fig, axes = plt.subplots(2, n_datasets, figsize=(5 * n_datasets, 10))

    if n_datasets == 1:
        axes = axes.reshape(-1, 1)

    has_ci = 'auc_ci_lower' in oasis_data.columns

    for idx, dataset in enumerate(multi_period_datasets):
        df_sub = oasis_data[oasis_data['dataset'] == dataset]
        dataset_name = df_sub['dataset_name'].iloc[0].split(' (')[0]
        time_periods = sorted(df_sub['time_period'].unique())

        # Top row: Age trajectories
        ax_age = axes[0, idx]
        age_data = df_sub[df_sub['subgroup_type'] == 'Age']

        for age_group in ['18-44', '45-64', '65-79', '80+']:
            age_df = age_data[age_data['subgroup'] == age_group].sort_values('time_period')
            if not age_df.empty:
                color = SUBGROUP_COLORS.get(age_group, '#666666')
                x = range(len(age_df))
                ax_age.plot(x, age_df['auc'], 'o-', label=age_group,
                           color=color, markersize=6, linewidth=2)

                if has_ci and not age_df['auc_ci_lower'].isna().all():
                    ax_age.fill_between(x, age_df['auc_ci_lower'], age_df['auc_ci_upper'],
                                       color=color, alpha=0.15)

        ax_age.set_title(f'{dataset_name} - Age Groups')
        ax_age.set_xlabel('Time Period')
        ax_age.set_ylabel('OASIS AUC')
        ax_age.legend(title='Age', loc='lower right', fontsize=8)
        ax_age.set_ylim(0.55, 0.95)

        # Bottom row: Race trajectories (if available)
        ax_race = axes[1, idx]
        race_data = df_sub[df_sub['subgroup_type'] == 'Race']

        if not race_data.empty:
            for race in ['White', 'Black', 'Hispanic', 'Asian']:
                race_df = race_data[race_data['subgroup'] == race].sort_values('time_period')
                if not race_df.empty:
                    color = SUBGROUP_COLORS.get(race, '#666666')
                    x = range(len(race_df))
                    ax_race.plot(x, race_df['auc'], 'o-', label=race,
                               color=color, markersize=6, linewidth=2)

                    if has_ci and not race_df['auc_ci_lower'].isna().all():
                        ax_race.fill_between(x, race_df['auc_ci_lower'], race_df['auc_ci_upper'],
                                           color=color, alpha=0.15)

            ax_race.set_title(f'{dataset_name} - Race/Ethnicity')
            ax_race.legend(title='Race', loc='lower right', fontsize=8)
        else:
            # If no race data, show gender
            gender_data = df_sub[df_sub['subgroup_type'] == 'Gender']
            for gender in ['Male', 'Female']:
                gender_df = gender_data[gender_data['subgroup'] == gender].sort_values('time_period')
                if not gender_df.empty:
                    color = SUBGROUP_COLORS.get(gender, '#666666')
                    x = range(len(gender_df))
                    ax_race.plot(x, gender_df['auc'], 'o-', label=gender,
                               color=color, markersize=6, linewidth=2)

                    if has_ci and not gender_df['auc_ci_lower'].isna().all():
                        ax_race.fill_between(x, gender_df['auc_ci_lower'], gender_df['auc_ci_upper'],
                                           color=color, alpha=0.15)
            ax_race.set_title(f'{dataset_name} - Gender')
            ax_race.legend(title='Gender', loc='lower right', fontsize=8)

        ax_race.set_xlabel('Time Period')
        ax_race.set_ylabel('OASIS AUC')
        ax_race.set_ylim(0.55, 0.95)

    plt.suptitle('OASIS Performance Trajectories Over Time (Cross-Dataset)' + (' (95% CI)' if has_ci else ''),
                 fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(SUPPLEMENTARY_FIGURES_DIR / 'figS11_temporal_trajectory.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: supplementary/figS11_temporal_trajectory.png")


def create_summary_table(results, deltas):
    """Create per-dataset summary tables (NOT cross-dataset comparisons).

    Per Leo's feedback: cross-dataset comparisons "defeat our purpose".
    Tables are saved to each dataset's output subdirectory.
    """
    if deltas is None or deltas.empty:
        print("No delta data for summary table")
        return

    # Create per-dataset summary tables
    for dataset in deltas['dataset'].unique():
        dataset_dir = OUTPUT_DIR / dataset
        dataset_dir.mkdir(exist_ok=True)

        dataset_deltas = deltas[deltas['dataset'] == dataset]
        dataset_results = results[results['dataset'] == dataset]

        # Table: Summary by score for this dataset
        overall = dataset_deltas[dataset_deltas['subgroup'] == 'All'].copy()
        if not overall.empty:
            summary = overall[['score', 'auc_first', 'auc_last', 'delta', 'p_value_delong', 'significant']].copy()
            summary['score'] = summary['score'].str.upper()
            summary = summary.rename(columns={
                'auc_first': 'AUC (First)',
                'auc_last': 'AUC (Last)',
                'delta': 'Delta',
                'p_value_delong': 'p-value',
                'significant': 'Significant'
            })
            summary.to_csv(dataset_dir / 'summary_by_score.csv', index=False)

    print(f"Per-dataset summary tables updated in output/{{dataset}}/ subdirectories")


def copy_supplementary_figures():
    """Verify supplementary figures exist (generated by supplementary_analysis.py)."""
    print("\nVerifying supplementary figures...")

    # These figures are generated directly to figures/supplementary/ by supplementary_analysis.py
    supplementary_figures = [
        SUPPLEMENTARY_FIGURES_DIR / 'figS1_mimic_mouthcare.png',
        SUPPLEMENTARY_FIGURES_DIR / 'figS2_mimic_mechvent.png',
    ]

    for fig_path in supplementary_figures:
        if fig_path.exists():
            print(f"Found: {fig_path.name}")
        else:
            print(f"Warning: {fig_path.name} not found (run supplementary_analysis.py first)")


def main():
    """Generate all figures from per-dataset results.

    Per Leo's feedback (Dec 21, 2025):
    - MAIN figures: One comprehensive figure per dataset (NO cross-dataset comparisons)
    - SUPPLEMENTARY figures: Cross-dataset comparisons moved here
    """
    print("="*60)
    print("GENERATING DRIFT ANALYSIS FIGURES")
    print("Per Leo's feedback: Each dataset analyzed SEPARATELY")
    print("="*60)

    # Load data
    results, deltas = load_data()
    if results is None:
        return

    # ============================================================
    # MAIN FIGURES: Per-dataset analysis (NO cross-dataset comparisons)
    # ============================================================
    print("\n" + "-"*50)
    print("MAIN FIGURES: Per-Dataset Subgroup Drift Analysis")
    print("-"*50)

    # Fig 1-4: One comprehensive figure per dataset
    dataset_configs = [
        ('mimic_combined', 1),
        ('eicu_combined', 2),
        ('saltz', 3),
        ('zhejiang', 4),
    ]

    for dataset_key, fig_num in dataset_configs:
        create_per_dataset_figure(results, deltas, dataset_key, fig_num)

    # Fig 5: Money figure (summary of key findings - kept for visual impact)
    print("\nGenerating summary money figure...")
    fig5_money_figure(results, deltas)

    # ============================================================
    # SUPPLEMENTARY FIGURES: Cross-dataset comparisons
    # ============================================================
    print("\n" + "-"*50)
    print("SUPPLEMENTARY FIGURES: Cross-Dataset Comparisons")
    print("-"*50)

    # Move all cross-dataset comparison figures to supplementary
    # Rename functions to figS* pattern and save to supplementary folder

    # S3: Overall drift comparison (cross-dataset)
    figS3_overall_drift_by_dataset(results, deltas)

    # S4: Age-stratified drift comparison (cross-dataset)
    figS4_age_stratified_comparison(results, deltas)

    # S5: Race disparities comparison (cross-dataset)
    figS5_race_comparison(results, deltas)

    # S6: Forest plot of significant findings (cross-dataset)
    figS6_significance_forest_plot(deltas)

    # S7: Gender drift patterns (cross-dataset)
    figS7_gender_comparison(results, deltas)

    # S8: Drift delta summary (cross-dataset)
    figS8_drift_delta_summary(deltas)

    # S9: Comprehensive heatmap (cross-dataset)
    figS9_comprehensive_heatmap(deltas)

    # S10: Score comparison by age (cross-dataset)
    figS10_score_comparison_by_age(deltas)

    # S11: Temporal trajectory (cross-dataset)
    figS11_temporal_trajectory(results)

    # Create summary tables
    print("\nGenerating summary tables...")
    create_summary_table(results, deltas)

    # Verify care phenotype supplementary figures
    copy_supplementary_figures()

    print("\n" + "="*60)
    print("ALL FIGURES GENERATED SUCCESSFULLY")
    print("="*60)
    print(f"\nMain figures (1-5): {FIGURES_DIR}")
    print(f"Supplementary figures (S1-S11): {SUPPLEMENTARY_FIGURES_DIR}")
    print(f"Tables: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
