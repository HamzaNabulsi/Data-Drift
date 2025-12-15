"""
Generate publication-quality figures for Data Drift analysis.
Creates diverging slopes plots, heatmaps, and comparison figures.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300

# Paths
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / 'output'
FIGURES_DIR = BASE_DIR / 'figures'

# Color palettes
DATASET_COLORS = {
    'MIMIC Mouthcare': '#E63946',
    'Amsterdam ICU': '#457B9D',
}

AGE_COLORS = {
    '<50': '#2E86AB',
    '50-65': '#A23B72',
    '65-80': '#F18F01',
    '80+': '#C73E1D'
}

GENDER_COLORS = {
    'male': '#1D3557',
    'female': '#E63946',
    'Male': '#1D3557',
    'Female': '#E63946'
}

RACE_COLORS = {
    'White': '#457B9D',
    'Black': '#E63946',
    'Hispanic': '#2A9D8F',
    'Asian': '#E9C46A',
    'Other': '#F4A261',
    'Unknown': '#8D99AE'
}

def load_data():
    """Load all performance CSV files."""
    data = {}

    # MIMIC Mouthcare
    mimic_dir = OUTPUT_DIR / 'mimic_mouthcare'
    if mimic_dir.exists():
        data['mimic_mouthcare'] = {
            'yearly': pd.read_csv(mimic_dir / 'mimic_mouthcare_yearly_performance.csv'),
            'age': pd.read_csv(mimic_dir / 'mimic_mouthcare_age_performance.csv'),
            'gender': pd.read_csv(mimic_dir / 'mimic_mouthcare_gender_performance.csv'),
            'race': pd.read_csv(mimic_dir / 'mimic_mouthcare_race_performance.csv'),
            'care': pd.read_csv(mimic_dir / 'mimic_mouthcare_care_performance.csv'),
        }

    # Amsterdam ICU
    amsterdam_dir = OUTPUT_DIR / 'amsterdam_icu'
    if amsterdam_dir.exists():
        data['amsterdam_icu'] = {
            'yearly': pd.read_csv(amsterdam_dir / 'amsterdam_icu_yearly_performance.csv'),
            'age': pd.read_csv(amsterdam_dir / 'amsterdam_icu_age_performance.csv'),
            'gender': pd.read_csv(amsterdam_dir / 'amsterdam_icu_gender_performance.csv'),
        }

    return data


def create_diverging_slopes_age(data, save_path):
    """
    Create diverging slopes plot showing AUC trajectories by age group.
    Each line represents a subgroup, showing them diverging over time.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    datasets = [
        ('mimic_mouthcare', 'MIMIC Mouthcare (2008-2019)', axes[0]),
        ('amsterdam_icu', 'Amsterdam ICU (2013-2021)', axes[1])
    ]

    for dataset_key, title, ax in datasets:
        if dataset_key not in data:
            continue

        df = data[dataset_key]['age']

        # Get unique periods and age groups
        periods = df['Period'].unique()
        age_groups = ['<50', '50-65', '65-80', '80+']

        # Create numeric x-axis
        period_to_x = {p: i for i, p in enumerate(sorted(periods, key=str))}

        for age_group in age_groups:
            subset = df[df['Age_Group'] == age_group].copy()
            subset['x'] = subset['Period'].map(period_to_x)
            subset = subset.sort_values('x')

            ax.plot(subset['x'], subset['AUC'],
                   marker='o', linewidth=2.5, markersize=8,
                   color=AGE_COLORS.get(age_group, '#333'),
                   label=age_group, alpha=0.9)

            # Add endpoint annotations
            if len(subset) > 0:
                first = subset.iloc[0]
                last = subset.iloc[-1]
                change = last['AUC'] - first['AUC']
                sign = '+' if change > 0 else ''
                ax.annotate(f'{sign}{change:.3f}',
                           xy=(last['x'] + 0.15, last['AUC']),
                           fontsize=9, fontweight='bold',
                           color=AGE_COLORS.get(age_group, '#333'))

        ax.set_xticks(list(period_to_x.values()))
        ax.set_xticklabels(list(period_to_x.keys()), rotation=45, ha='right')
        ax.set_xlabel('Time Period')
        ax.set_ylabel('AUC (Discrimination)')
        ax.set_title(title, fontweight='bold')
        ax.legend(title='Age Group', loc='best')
        ax.set_ylim(0.5, 0.95)
        ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5, label='Acceptable (0.7)')

    plt.suptitle('Diverging Drift: Age Groups Show Unequal Performance Changes',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def create_diverging_slopes_gender(data, save_path):
    """Create diverging slopes plot for gender."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    datasets = [
        ('mimic_mouthcare', 'MIMIC Mouthcare (2008-2019)', axes[0]),
        ('amsterdam_icu', 'Amsterdam ICU (2013-2021)', axes[1])
    ]

    for dataset_key, title, ax in datasets:
        if dataset_key not in data:
            continue

        df = data[dataset_key]['gender']

        periods = df['Period'].unique()
        genders = df['Gender'].unique()

        period_to_x = {p: i for i, p in enumerate(sorted(periods, key=str))}

        for gender in genders:
            subset = df[df['Gender'] == gender].copy()
            subset['x'] = subset['Period'].map(period_to_x)
            subset = subset.sort_values('x')

            color = GENDER_COLORS.get(gender, '#333')
            ax.plot(subset['x'], subset['AUC'],
                   marker='o', linewidth=2.5, markersize=8,
                   color=color, label=gender.capitalize(), alpha=0.9)

            if len(subset) > 0:
                first = subset.iloc[0]
                last = subset.iloc[-1]
                change = last['AUC'] - first['AUC']
                sign = '+' if change > 0 else ''
                ax.annotate(f'{sign}{change:.3f}',
                           xy=(last['x'] + 0.1, last['AUC']),
                           fontsize=10, fontweight='bold',
                           color=color)

        ax.set_xticks(list(period_to_x.values()))
        ax.set_xticklabels(list(period_to_x.keys()), rotation=45, ha='right')
        ax.set_xlabel('Time Period')
        ax.set_ylabel('AUC (Discrimination)')
        ax.set_title(title, fontweight='bold')
        ax.legend(title='Gender', loc='best')
        ax.set_ylim(0.5, 0.85)

    plt.suptitle('Gender Disparity in Model Drift',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def create_drift_delta_heatmap(data, save_path):
    """
    Create heatmap showing AUC change (first to last period) across subgroups.
    Red = decline, Blue = improvement.
    """
    # Collect drift deltas for each subgroup
    results = []

    # MIMIC Mouthcare
    if 'mimic_mouthcare' in data:
        # Age groups
        df = data['mimic_mouthcare']['age']
        for age_group in df['Age_Group'].unique():
            subset = df[df['Age_Group'] == age_group].sort_values('Period')
            if len(subset) >= 2:
                delta = subset.iloc[-1]['AUC'] - subset.iloc[0]['AUC']
                results.append({
                    'Dataset': 'MIMIC\nMouthcare',
                    'Subgroup': f'Age: {age_group}',
                    'Delta': delta
                })

        # Gender
        df = data['mimic_mouthcare']['gender']
        for gender in df['Gender'].unique():
            subset = df[df['Gender'] == gender].sort_values('Period')
            if len(subset) >= 2:
                delta = subset.iloc[-1]['AUC'] - subset.iloc[0]['AUC']
                results.append({
                    'Dataset': 'MIMIC\nMouthcare',
                    'Subgroup': f'Gender: {gender}',
                    'Delta': delta
                })

        # Race
        df = data['mimic_mouthcare']['race']
        for race in ['White', 'Black', 'Hispanic', 'Asian']:
            subset = df[df['Race'] == race].sort_values('Period')
            if len(subset) >= 2:
                delta = subset.iloc[-1]['AUC'] - subset.iloc[0]['AUC']
                results.append({
                    'Dataset': 'MIMIC\nMouthcare',
                    'Subgroup': f'Race: {race}',
                    'Delta': delta
                })

    # Amsterdam ICU
    if 'amsterdam_icu' in data:
        # Age groups
        df = data['amsterdam_icu']['age']
        for age_group in df['Age_Group'].unique():
            subset = df[df['Age_Group'] == age_group].sort_values('Period')
            if len(subset) >= 2:
                delta = subset.iloc[-1]['AUC'] - subset.iloc[0]['AUC']
                results.append({
                    'Dataset': 'Amsterdam\nICU',
                    'Subgroup': f'Age: {age_group}',
                    'Delta': delta
                })

        # Gender
        df = data['amsterdam_icu']['gender']
        for gender in df['Gender'].unique():
            subset = df[df['Gender'] == gender].sort_values('Period')
            if len(subset) >= 2:
                delta = subset.iloc[-1]['AUC'] - subset.iloc[0]['AUC']
                results.append({
                    'Dataset': 'Amsterdam\nICU',
                    'Subgroup': f'Gender: {gender}',
                    'Delta': delta
                })

    # Create DataFrame and pivot
    results_df = pd.DataFrame(results)
    pivot = results_df.pivot(index='Subgroup', columns='Dataset', values='Delta')

    # Sort subgroups logically
    subgroup_order = [
        'Age: <50', 'Age: 50-65', 'Age: 65-80', 'Age: 80+',
        'Gender: Female', 'Gender: Male',
        'Race: White', 'Race: Black', 'Race: Hispanic', 'Race: Asian'
    ]
    pivot = pivot.reindex([s for s in subgroup_order if s in pivot.index])

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Custom diverging colormap (red for negative, blue for positive)
    cmap = sns.diverging_palette(10, 240, s=80, l=55, as_cmap=True)

    # Determine symmetric color range
    vmax = max(abs(pivot.min().min()), abs(pivot.max().max()))

    sns.heatmap(pivot, annot=True, fmt='.3f', cmap=cmap,
                center=0, vmin=-vmax, vmax=vmax,
                linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'AUC Change', 'shrink': 0.8},
                ax=ax)

    ax.set_title('Drift Magnitude Heatmap: AUC Change from First to Last Period\n(Red = Decline, Blue = Improvement)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Rotate y-axis labels
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def create_overall_comparison(data, save_path):
    """Create overall AUC trend comparison across datasets."""
    fig, ax = plt.subplots(figsize=(12, 6))

    datasets_info = [
        ('mimic_mouthcare', 'MIMIC Mouthcare', '#E63946'),
        ('amsterdam_icu', 'Amsterdam ICU', '#457B9D'),
    ]

    for dataset_key, label, color in datasets_info:
        if dataset_key not in data:
            continue

        df = data[dataset_key]['yearly']
        periods = df['Period'].values

        # Normalize x-axis to 0-1 for comparison
        x_norm = np.linspace(0, 1, len(periods))

        ax.plot(x_norm, df['AUC'],
               marker='o', linewidth=3, markersize=10,
               color=color, label=label, alpha=0.9)

        # Annotate start and end
        ax.annotate(f'{df["AUC"].iloc[0]:.3f}\n({periods[0]})',
                   xy=(x_norm[0], df['AUC'].iloc[0]),
                   xytext=(-30, -20), textcoords='offset points',
                   fontsize=9, color=color)
        ax.annotate(f'{df["AUC"].iloc[-1]:.3f}\n({periods[-1]})',
                   xy=(x_norm[-1], df['AUC'].iloc[-1]),
                   xytext=(5, -20), textcoords='offset points',
                   fontsize=9, color=color)

    ax.set_xlabel('Normalized Time (Start to End of Study Period)', fontsize=12)
    ax.set_ylabel('AUC (Discrimination)', fontsize=12)
    ax.set_title('Overall SOFA Performance Drift Across Datasets', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=12)
    ax.set_ylim(0.55, 0.85)
    ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(0.5, 0.705, 'Acceptable threshold (0.7)', ha='center', fontsize=9, color='gray')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def create_race_drift_mimic(data, save_path):
    """Create race-stratified drift plot for MIMIC."""
    if 'mimic_mouthcare' not in data:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    df = data['mimic_mouthcare']['race']
    periods = df['Period'].unique()
    period_to_x = {p: i for i, p in enumerate(sorted(periods, key=str))}

    # Focus on main races with sufficient sample
    main_races = ['White', 'Black', 'Hispanic', 'Asian', 'Other']

    for race in main_races:
        subset = df[df['Race'] == race].copy()
        if len(subset) < 2:
            continue
        subset['x'] = subset['Period'].map(period_to_x)
        subset = subset.sort_values('x')

        color = RACE_COLORS.get(race, '#333')
        ax.plot(subset['x'], subset['AUC'],
               marker='o', linewidth=2.5, markersize=8,
               color=color, label=race, alpha=0.9)

        # Endpoint annotation
        first = subset.iloc[0]
        last = subset.iloc[-1]
        change = last['AUC'] - first['AUC']
        sign = '+' if change > 0 else ''
        ax.annotate(f'{sign}{change:.3f}',
                   xy=(last['x'] + 0.1, last['AUC']),
                   fontsize=9, fontweight='bold',
                   color=color)

    ax.set_xticks(list(period_to_x.values()))
    ax.set_xticklabels(list(period_to_x.keys()), rotation=45, ha='right')
    ax.set_xlabel('Time Period')
    ax.set_ylabel('AUC (Discrimination)')
    ax.set_title('Racial Disparities in SOFA Performance Drift\nMIMIC Mouthcare (2008-2019)',
                fontsize=14, fontweight='bold')
    ax.legend(title='Race', loc='best')
    ax.set_ylim(0.45, 0.75)

    # Highlight concerning trend
    ax.axhspan(0.45, 0.6, alpha=0.1, color='red', label='Concerning (<0.6)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def create_summary_figure(data, save_path):
    """Create a comprehensive 2x2 summary figure."""
    fig = plt.figure(figsize=(16, 12))

    # 1. Overall comparison (top left)
    ax1 = fig.add_subplot(2, 2, 1)
    for dataset_key, label, color in [('mimic_mouthcare', 'MIMIC Mouthcare', '#E63946'),
                                        ('amsterdam_icu', 'Amsterdam ICU', '#457B9D')]:
        if dataset_key not in data:
            continue
        df = data[dataset_key]['yearly']
        x_norm = np.linspace(0, 1, len(df))
        ax1.plot(x_norm, df['AUC'], marker='o', linewidth=2.5, markersize=8,
                color=color, label=label, alpha=0.9)
    ax1.set_xlabel('Normalized Time')
    ax1.set_ylabel('AUC')
    ax1.set_title('A. Overall SOFA Performance Drift', fontweight='bold')
    ax1.legend(loc='best')
    ax1.set_ylim(0.55, 0.85)
    ax1.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5)

    # 2. Age divergence - MIMIC (top right)
    ax2 = fig.add_subplot(2, 2, 2)
    if 'mimic_mouthcare' in data:
        df = data['mimic_mouthcare']['age']
        periods = sorted(df['Period'].unique(), key=str)
        period_to_x = {p: i for i, p in enumerate(periods)}
        for age_group in ['<50', '50-65', '65-80', '80+']:
            subset = df[df['Age_Group'] == age_group].copy()
            subset['x'] = subset['Period'].map(period_to_x)
            subset = subset.sort_values('x')
            ax2.plot(subset['x'], subset['AUC'], marker='o', linewidth=2,
                    color=AGE_COLORS.get(age_group), label=age_group)
        ax2.set_xticks(range(len(periods)))
        ax2.set_xticklabels(periods, rotation=45, ha='right')
    ax2.set_xlabel('Period')
    ax2.set_ylabel('AUC')
    ax2.set_title('B. Age Group Drift - MIMIC Mouthcare', fontweight='bold')
    ax2.legend(title='Age', loc='best')
    ax2.set_ylim(0.5, 0.8)

    # 3. Age divergence - Amsterdam (bottom left)
    ax3 = fig.add_subplot(2, 2, 3)
    if 'amsterdam_icu' in data:
        df = data['amsterdam_icu']['age']
        periods = sorted(df['Period'].unique())
        period_to_x = {p: i for i, p in enumerate(periods)}
        for age_group in ['<50', '50-65', '65-80', '80+']:
            subset = df[df['Age_Group'] == age_group].copy()
            subset['x'] = subset['Period'].map(period_to_x)
            subset = subset.sort_values('x')
            ax3.plot(subset['x'], subset['AUC'], marker='o', linewidth=2,
                    color=AGE_COLORS.get(age_group), label=age_group)
        ax3.set_xticks(range(0, len(periods), 2))
        ax3.set_xticklabels([periods[i] for i in range(0, len(periods), 2)])
    ax3.set_xlabel('Year')
    ax3.set_ylabel('AUC')
    ax3.set_title('C. Age Group Drift - Amsterdam ICU', fontweight='bold')
    ax3.legend(title='Age', loc='best')
    ax3.set_ylim(0.6, 0.95)

    # 4. Drift delta summary (bottom right)
    ax4 = fig.add_subplot(2, 2, 4)

    # Calculate deltas
    deltas = []
    labels = []
    colors = []

    for dataset_key, ds_label in [('mimic_mouthcare', 'MIMIC'), ('amsterdam_icu', 'Amsterdam')]:
        if dataset_key not in data:
            continue
        df = data[dataset_key]['age']
        for age_group in ['<50', '50-65', '65-80', '80+']:
            subset = df[df['Age_Group'] == age_group].sort_values('Period')
            if len(subset) >= 2:
                delta = subset.iloc[-1]['AUC'] - subset.iloc[0]['AUC']
                deltas.append(delta)
                labels.append(f'{ds_label}\n{age_group}')
                colors.append(AGE_COLORS.get(age_group))

    y_pos = range(len(deltas))
    bars = ax4.barh(y_pos, deltas, color=colors, alpha=0.8, edgecolor='black')
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(labels)
    ax4.set_xlabel('AUC Change (First to Last Period)')
    ax4.set_title('D. Drift Magnitude by Subgroup', fontweight='bold')
    ax4.axvline(x=0, color='black', linewidth=1)

    # Add value labels
    for i, (bar, delta) in enumerate(zip(bars, deltas)):
        ax4.text(delta + 0.01 if delta >= 0 else delta - 0.01,
                bar.get_y() + bar.get_height()/2,
                f'{delta:+.3f}', va='center', fontsize=9,
                ha='left' if delta >= 0 else 'right')

    plt.suptitle('Subgroup-Specific Drift in SOFA Score Performance',
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    """Generate all figures."""
    print("Loading data...")
    data = load_data()

    print(f"Found datasets: {list(data.keys())}")

    print("\nGenerating figures...")

    # 1. Diverging slopes - Age
    create_diverging_slopes_age(data, FIGURES_DIR / 'diverging_slopes_age.png')

    # 2. Diverging slopes - Gender
    create_diverging_slopes_gender(data, FIGURES_DIR / 'diverging_slopes_gender.png')

    # 3. Drift delta heatmap
    create_drift_delta_heatmap(data, FIGURES_DIR / 'drift_delta_heatmap.png')

    # 4. Overall comparison
    create_overall_comparison(data, FIGURES_DIR / 'overall_drift_comparison.png')

    # 5. Race drift (MIMIC only)
    create_race_drift_mimic(data, FIGURES_DIR / 'race_drift_mimic.png')

    # 6. Summary figure (4-panel)
    create_summary_figure(data, FIGURES_DIR / 'summary_figure.png')

    print("\nAll figures generated successfully!")
    print(f"Output directory: {FIGURES_DIR}")


if __name__ == '__main__':
    main()
