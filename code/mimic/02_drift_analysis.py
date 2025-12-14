"""
Step 2: Subgroup-Specific Drift Analysis
Generalizable across multiple ICU datasets
Analyzes whether model drift affects patient subgroups equally
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path to import config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_active_config, get_output_path, ACTIVE_DATASET, ANALYSIS_CONFIG

# ============================================================
# LOAD CONFIGURATION
# ============================================================
print("="*60)
print(f"DRIFT ANALYSIS: {ACTIVE_DATASET}")
print("="*60)

config = get_active_config()
output_path = get_output_path()

print(f"\nDataset: {config['name']}")
print(f"Output: {output_path}")

# ============================================================
# LOAD DATA
# ============================================================
print("\nLoading data...")
data_file = os.path.join(config['data_path'], config['file'])

if not os.path.exists(data_file):
    print(f"\n❌ ERROR: Data file not found: {data_file}")
    exit(1)

df = pd.read_csv(data_file)
print(f"✅ Loaded {len(df)} records")

# ============================================================
# PREPROCESSING
# ============================================================
print("\nPreprocessing data...")

# Convert outcome to binary
df['outcome_binary'] = (df[config['outcome_col']] == config['outcome_positive']).astype(int)

# Check if SOFA scores exist
if not config['has_precomputed_sofa']:
    print(f"\n❌ ERROR: This dataset requires SOFA score computation first!")
    print(f"   Please compute SOFA scores and update config.py")
    exit(1)

if config['score_col'] not in df.columns:
    print(f"\n❌ ERROR: Score column '{config['score_col']}' not found!")
    exit(1)

# Create age groups if age column exists
age_col = config['demographic_cols'].get('age')
if age_col and age_col in df.columns:
    df['age_group'] = pd.cut(df[age_col],
                             bins=ANALYSIS_CONFIG['age_bins'],
                             labels=ANALYSIS_CONFIG['age_labels'])
    print(f"✅ Created age groups: {ANALYSIS_CONFIG['age_labels']}")
else:
    print(f"⚠️  Age column not available, skipping age analysis")

# Create care quartiles if care frequency column exists
care_col = config['clinical_cols'].get('care_frequency')
if care_col and care_col in df.columns:
    df['care_quartile'] = pd.qcut(df[care_col],
                                   q=ANALYSIS_CONFIG['care_quartiles'],
                                   labels=['Q1_High', 'Q2', 'Q3', 'Q4_Low'],
                                   duplicates='drop')
    print(f"✅ Created care quartiles based on '{care_col}'")
else:
    print(f"⚠️  Care frequency column not available, skipping care analysis")

# Determine year periods
year_col = config['year_col']
if config['year_bins']:
    year_order = config['year_bins']
    print(f"✅ Using predefined time periods: {year_order}")
else:
    # Auto-generate year bins
    years = sorted(df[year_col].unique())
    year_order = years
    print(f"✅ Using individual years: {year_order}")

# ============================================================
# METRICS FUNCTION
# ============================================================
def calculate_metrics(data, score_col=config['score_col'], outcome_col='outcome_binary'):
    """Calculate prediction performance metrics."""
    if len(data) < ANALYSIS_CONFIG['min_sample_size']:
        return None

    y_true = data[outcome_col].values
    y_score = data[score_col].values

    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_score))
    y_true = y_true[mask]
    y_score = y_score[mask]

    # Skip if no variance in outcome
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return None

    # Binary prediction using median threshold
    threshold = np.median(y_score)
    y_pred = (y_score >= threshold).astype(int)

    try:
        auc = roc_auc_score(y_true, y_score)
    except:
        auc = None

    return {
        'AUC': auc,
        'Accuracy': accuracy_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'N': len(y_true),
        'Mortality_Rate': y_true.mean(),
        'Mean_Score': y_score.mean()
    }

# ============================================================
# ANALYSIS 1: OVERALL PERFORMANCE BY TIME PERIOD
# ============================================================
print("\n" + "="*60)
print("ANALYSIS 1: OVERALL PERFORMANCE BY TIME PERIOD")
print("="*60)

yearly_results = []
for period in year_order:
    subset = df[df[year_col] == period]
    metrics = calculate_metrics(subset)
    if metrics:
        metrics['Period'] = period
        yearly_results.append(metrics)
        print(f"{period}: AUC={metrics['AUC']:.3f}, N={metrics['N']}, Mortality={metrics['Mortality_Rate']:.1%}")

yearly_df = pd.DataFrame(yearly_results)

# ============================================================
# ANALYSIS 2: DRIFT BY RACE
# ============================================================
race_col = config['demographic_cols'].get('race')
race_df = pd.DataFrame()

if race_col and race_col in df.columns:
    print("\n" + "="*60)
    print("ANALYSIS 2: PERFORMANCE BY RACE OVER TIME")
    print("="*60)

    race_results = []
    races = df[race_col].dropna().unique()

    for period in year_order:
        for race in races:
            subset = df[(df[year_col] == period) & (df[race_col] == race)]
            metrics = calculate_metrics(subset)
            if metrics:
                metrics['Period'] = period
                metrics['Race'] = race
                race_results.append(metrics)

    race_df = pd.DataFrame(race_results)
    if len(race_df) > 0:
        pivot = race_df.pivot(index='Period', columns='Race', values='AUC')
        print(pivot.round(3))
else:
    print(f"\n⚠️  Skipping race analysis (column: {race_col})")

# ============================================================
# ANALYSIS 3: DRIFT BY GENDER
# ============================================================
gender_col = config['demographic_cols'].get('gender')
gender_df = pd.DataFrame()

if gender_col and gender_col in df.columns:
    print("\n" + "="*60)
    print("ANALYSIS 3: PERFORMANCE BY GENDER OVER TIME")
    print("="*60)

    gender_results = []
    for period in year_order:
        for gender in df[gender_col].unique():
            subset = df[(df[year_col] == period) & (df[gender_col] == gender)]
            metrics = calculate_metrics(subset)
            if metrics:
                metrics['Period'] = period
                metrics['Gender'] = gender
                gender_results.append(metrics)

    gender_df = pd.DataFrame(gender_results)
    if len(gender_df) > 0:
        print(gender_df.pivot(index='Period', columns='Gender', values='AUC').round(3))
else:
    print(f"\n⚠️  Skipping gender analysis (column: {gender_col})")

# ============================================================
# ANALYSIS 4: DRIFT BY CARE QUARTILE
# ============================================================
care_df = pd.DataFrame()

if 'care_quartile' in df.columns:
    print("\n" + "="*60)
    print("ANALYSIS 4: PERFORMANCE BY CARE FREQUENCY")
    print("="*60)

    care_results = []
    for period in year_order:
        for q in ['Q1_High', 'Q4_Low']:
            subset = df[(df[year_col] == period) & (df['care_quartile'] == q)]
            metrics = calculate_metrics(subset)
            if metrics:
                metrics['Period'] = period
                metrics['Care_Quartile'] = q
                care_results.append(metrics)

    care_df = pd.DataFrame(care_results)
    if len(care_df) > 0:
        print(care_df.pivot(index='Period', columns='Care_Quartile', values='AUC').round(3))
else:
    print(f"\n⚠️  Skipping care quartile analysis")

# ============================================================
# ANALYSIS 5: DRIFT BY AGE GROUP
# ============================================================
age_df = pd.DataFrame()

if 'age_group' in df.columns:
    print("\n" + "="*60)
    print("ANALYSIS 5: PERFORMANCE BY AGE GROUP")
    print("="*60)

    age_results = []
    for period in year_order:
        for age in ANALYSIS_CONFIG['age_labels']:
            subset = df[(df[year_col] == period) & (df['age_group'] == age)]
            metrics = calculate_metrics(subset)
            if metrics:
                metrics['Period'] = period
                metrics['Age_Group'] = age
                age_results.append(metrics)

    age_df = pd.DataFrame(age_results)
    if len(age_df) > 0:
        print(age_df.pivot(index='Period', columns='Age_Group', values='AUC').round(3))
else:
    print(f"\n⚠️  Skipping age group analysis")

# ============================================================
# CREATE VISUALIZATION
# ============================================================
print("\n" + "="*60)
print("CREATING FIGURES...")
print("="*60)

# Determine which plots to include
plot_configs = [
    ('overall', yearly_df, True),
    ('race', race_df, race_col and len(race_df) > 0),
    ('gender', gender_df, gender_col and len(gender_df) > 0),
    ('care', care_df, len(care_df) > 0),
    ('age', age_df, len(age_df) > 0)
]

active_plots = [p for p in plot_configs if p[2]]
n_plots = len(active_plots)

# Create figure with appropriate layout
if n_plots == 0:
    print("❌ No data available for plotting")
else:
    ncols = 3
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=ANALYSIS_CONFIG['figure_size'])
    fig.suptitle(f'Subgroup-Specific Drift Analysis: {config["name"]}\n'
                 f'Does Model Drift Affect All Subgroups Equally?',
                 fontsize=14, fontweight='bold')

    # Flatten axes for easier indexing
    if nrows == 1 and ncols == 1:
        axes = [axes]
    elif nrows == 1 or ncols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    # X-axis labels (shortened if using year ranges)
    if config['year_bins'] and len(year_order[0]) > 10:
        x_labels = [y.replace(' - ', '-')[-5:] if '-' in y else y for y in year_order]
    else:
        x_labels = [str(y)[-4:] if len(str(y)) > 4 else str(y) for y in year_order]

    plot_idx = 0

    # Plot 1: Overall
    if plot_idx < len(active_plots) and active_plots[plot_idx][0] == 'overall':
        ax = axes[plot_idx]
        ax.plot(range(len(yearly_df)), yearly_df['AUC'], 'o-', linewidth=2, markersize=10, color='#2c3e50')
        ax.fill_between(range(len(yearly_df)), yearly_df['AUC'], alpha=0.3, color='#2c3e50')
        ax.set_xticks(range(len(yearly_df)))
        ax.set_xticklabels(x_labels[:len(yearly_df)])
        ax.set_xlabel('Time Period')
        ax.set_ylabel('AUC')
        ax.set_title('A. Overall Performance')
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Plot 2: Race
    if plot_idx < len(active_plots) and active_plots[plot_idx][0] == 'race':
        ax = axes[plot_idx]
        colors_race = {'White': '#3498db', 'Black': '#e74c3c', 'Asian': '#2ecc71',
                      'Hispanic': '#9b59b6', 'Other': '#f39c12'}
        for race in race_df['Race'].unique():
            subset = race_df[race_df['Race'] == race].sort_values('Period')
            if len(subset) > 1:
                ax.plot(range(len(subset)), subset['AUC'], 'o-', label=race, linewidth=2,
                       markersize=8, color=colors_race.get(race, 'gray'))
        ax.set_xticks(range(min(4, len(subset))))
        ax.set_xticklabels(x_labels[:min(4, len(subset))])
        ax.set_xlabel('Time Period')
        ax.set_ylabel('AUC')
        ax.set_title('B. Performance by Race')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Plot 3: Gender
    if plot_idx < len(active_plots) and active_plots[plot_idx][0] == 'gender':
        ax = axes[plot_idx]
        colors_gender = {'Male': '#3498db', 'Female': '#e74c3c', 'M': '#3498db', 'F': '#e74c3c'}
        for gender in gender_df['Gender'].unique():
            subset = gender_df[gender_df['Gender'] == gender].sort_values('Period')
            ax.plot(range(len(subset)), subset['AUC'], 'o-', label=gender, linewidth=2,
                   markersize=10, color=colors_gender.get(gender, 'gray'))
        ax.set_xticks(range(len(subset)))
        ax.set_xticklabels(x_labels[:len(subset)])
        ax.set_xlabel('Time Period')
        ax.set_ylabel('AUC')
        ax.set_title('C. Performance by Gender')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Plot 4: Care
    if plot_idx < len(active_plots) and active_plots[plot_idx][0] == 'care':
        ax = axes[plot_idx]
        colors_care = {'Q1_High': '#27ae60', 'Q4_Low': '#e74c3c'}
        labels_care = {'Q1_High': 'Q1 (High-freq)', 'Q4_Low': 'Q4 (Low-freq)'}
        for q in ['Q1_High', 'Q4_Low']:
            subset = care_df[care_df['Care_Quartile'] == q].sort_values('Period')
            if len(subset) > 0:
                ax.plot(range(len(subset)), subset['AUC'], 'o-', label=labels_care[q], linewidth=2,
                       markersize=10, color=colors_care[q])
        ax.set_xticks(range(len(subset)))
        ax.set_xticklabels(x_labels[:len(subset)])
        ax.set_xlabel('Time Period')
        ax.set_ylabel('AUC')
        ax.set_title('D. Performance by Care Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Plot 5: Age
    if plot_idx < len(active_plots) and active_plots[plot_idx][0] == 'age':
        ax = axes[plot_idx]
        colors_age = {'<50': '#3498db', '50-65': '#2ecc71', '65-80': '#f39c12', '80+': '#e74c3c'}
        for age in ANALYSIS_CONFIG['age_labels']:
            subset = age_df[age_df['Age_Group'] == age].sort_values('Period')
            if len(subset) > 1:
                ax.plot(range(len(subset)), subset['AUC'], 'o-', label=age, linewidth=2,
                       markersize=8, color=colors_age.get(age, 'gray'))
        ax.set_xticks(range(min(4, len(subset))))
        ax.set_xticklabels(x_labels[:min(4, len(subset))])
        ax.set_xlabel('Time Period')
        ax.set_ylabel('AUC')
        ax.set_title('E. Performance by Age Group')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    output_file = os.path.join(output_path, f'{ACTIVE_DATASET}_drift_analysis.png')
    plt.savefig(output_file, dpi=ANALYSIS_CONFIG['figure_dpi'], bbox_inches='tight', facecolor='white')
    print(f"\n✅ Figure saved: {output_file}")

# ============================================================
# SAVE DATA TABLES
# ============================================================
yearly_df.to_csv(os.path.join(output_path, f'{ACTIVE_DATASET}_yearly_performance.csv'), index=False)
print(f"✅ Saved: {ACTIVE_DATASET}_yearly_performance.csv")

if len(race_df) > 0:
    race_df.to_csv(os.path.join(output_path, f'{ACTIVE_DATASET}_race_performance.csv'), index=False)
    print(f"✅ Saved: {ACTIVE_DATASET}_race_performance.csv")

if len(gender_df) > 0:
    gender_df.to_csv(os.path.join(output_path, f'{ACTIVE_DATASET}_gender_performance.csv'), index=False)
    print(f"✅ Saved: {ACTIVE_DATASET}_gender_performance.csv")

if len(care_df) > 0:
    care_df.to_csv(os.path.join(output_path, f'{ACTIVE_DATASET}_care_performance.csv'), index=False)
    print(f"✅ Saved: {ACTIVE_DATASET}_care_performance.csv")

if len(age_df) > 0:
    age_df.to_csv(os.path.join(output_path, f'{ACTIVE_DATASET}_age_performance.csv'), index=False)
    print(f"✅ Saved: {ACTIVE_DATASET}_age_performance.csv")

# ============================================================
# SUMMARY STATISTICS
# ============================================================
print("\n" + "="*60)
print("SUMMARY: KEY FINDINGS")
print("="*60)

if len(yearly_df) >= 2:
    auc_change = yearly_df['AUC'].iloc[-1] - yearly_df['AUC'].iloc[0]
    print(f"Overall AUC change (first to last period): {auc_change:+.3f}")

print("\nDrift by subgroup (AUC change from first to last period):")

for name, results_df, col in [('Race', race_df, 'Race'),
                               ('Gender', gender_df, 'Gender'),
                               ('Care Quartile', care_df, 'Care_Quartile'),
                               ('Age Group', age_df, 'Age_Group')]:
    if len(results_df) > 0 and col in results_df.columns:
        print(f"\n{name}:")
        for group in results_df[col].unique():
            subset = results_df[results_df[col] == group].sort_values('Period')
            if len(subset) >= 2:
                first_auc = subset['AUC'].iloc[0]
                last_auc = subset['AUC'].iloc[-1]
                change = last_auc - first_auc
                print(f"  {group}: {first_auc:.3f} → {last_auc:.3f} ({change:+.3f})")

print(f"\n{'='*60}")
print("✅ ANALYSIS COMPLETE!")
print(f"{'='*60}")
print(f"Dataset: {config['name']}")
print(f"Output location: {output_path}")
print(f"\nNext: Review {ACTIVE_DATASET}_drift_analysis.png")
