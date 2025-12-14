"""
Step 1: Explore dataset structure
Generalizable across multiple ICU datasets
"""
import pandas as pd
import os
import sys

# Add parent directory to path to import config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_active_config, ACTIVE_DATASET

# ============================================================
# LOAD CONFIGURATION
# ============================================================
print("="*60)
print(f"DATASET: {ACTIVE_DATASET}")
print("="*60)

config = get_active_config()
print(f"\nDataset: {config['name']}")
print(f"Description: {config['description']}")
print(f"Pre-computed SOFA: {config['has_precomputed_sofa']}")

# ============================================================
# LOAD DATA
# ============================================================
print("\nLoading data...")
data_file = os.path.join(config['data_path'], config['file'])

if not os.path.exists(data_file):
    print(f"\n❌ ERROR: Data file not found: {data_file}")
    print(f"\nPlease update the data_path in config.py for dataset '{ACTIVE_DATASET}'")
    exit(1)

df = pd.read_csv(data_file)
print(f"✅ Loaded {len(df)} records from {config['file']}")

# ============================================================
# CONVERT OUTCOME TO BINARY
# ============================================================
if config['outcome_col'] not in df.columns:
    print(f"\n❌ ERROR: Outcome column '{config['outcome_col']}' not found in dataset")
    print(f"Available columns: {df.columns.tolist()}")
    exit(1)

df['outcome_binary'] = (df[config['outcome_col']] == config['outcome_positive']).astype(int)

# ============================================================
# BASIC DATASET INFO
# ============================================================
print(f"\n{'='*60}")
print("DATASET STRUCTURE")
print(f"{'='*60}")
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst 5 rows:\n{df.head()}")

# ============================================================
# OUTCOME DISTRIBUTION
# ============================================================
print(f"\n{'='*60}")
print("OUTCOME DISTRIBUTION")
print(f"{'='*60}")
print(df[config['outcome_col']].value_counts())
print(f"\nMortality rate: {df['outcome_binary'].mean():.1%}")

# ============================================================
# TEMPORAL DISTRIBUTION
# ============================================================
if config['year_col'] and config['year_col'] in df.columns:
    print(f"\n{'='*60}")
    print("TEMPORAL DISTRIBUTION")
    print(f"{'='*60}")

    # Check if year_bins are predefined or need to be created
    if config['year_bins']:
        print(f"Time periods: {config['year_bins']}")
        print("\nPatients by time period:")
        print(df.groupby(config['year_col']).agg({
            df.columns[0]: 'count',  # Use first column for count
            'outcome_binary': 'mean'
        }).rename(columns={df.columns[0]: 'N_patients', 'outcome_binary': 'Mortality_rate'}))
    else:
        print(f"Years available: {sorted(df[config['year_col']].unique())}")
        print("\nPatients by year:")
        print(df.groupby(config['year_col']).agg({
            df.columns[0]: 'count',
            'outcome_binary': 'mean'
        }).rename(columns={df.columns[0]: 'N_patients', 'outcome_binary': 'Mortality_rate'}))
else:
    print(f"\n⚠️  No temporal column configured (year_col: {config['year_col']})")

# ============================================================
# DEMOGRAPHIC DISTRIBUTIONS
# ============================================================
print(f"\n{'='*60}")
print("DEMOGRAPHIC DISTRIBUTIONS")
print(f"{'='*60}")

for demo_type, demo_col in config['demographic_cols'].items():
    if demo_col and demo_col in df.columns:
        print(f"\n{demo_type.upper()} ({demo_col}):")
        print(df.groupby(demo_col).agg({
            df.columns[0]: 'count',
            'outcome_binary': 'mean'
        }).rename(columns={df.columns[0]: 'N_patients', 'outcome_binary': 'Mortality_rate'}))
    else:
        print(f"\n⚠️  {demo_type.upper()}: Column '{demo_col}' not found")

# ============================================================
# SCORE DISTRIBUTION
# ============================================================
if config['score_col'] and config['score_col'] in df.columns:
    print(f"\n{'='*60}")
    print(f"SCORE DISTRIBUTION ({config['score_col'].upper()})")
    print(f"{'='*60}")
    print(f"Range: {df[config['score_col']].min():.1f} to {df[config['score_col']].max():.1f}")
    print(f"Mean: {df[config['score_col']].mean():.2f}")
    print(f"Median: {df[config['score_col']].median():.2f}")
    print(f"\nDescriptive statistics:")
    print(df[config['score_col']].describe())
elif not config['has_precomputed_sofa']:
    print(f"\n{'='*60}")
    print(f"SCORE COMPUTATION REQUIRED")
    print(f"{'='*60}")
    print(f"⚠️  This dataset does not have pre-computed SOFA scores.")
    print(f"⚠️  You need to run the SOFA computation script first.")
    print(f"⚠️  See: Emma's task - use GitHub code to compute SOFA")
else:
    print(f"\n❌ ERROR: Score column '{config['score_col']}' not found in dataset")

# ============================================================
# CLINICAL VARIABLES
# ============================================================
clinical_vars_available = [col for col in config['clinical_cols'].values() if col and col in df.columns]

if clinical_vars_available:
    print(f"\n{'='*60}")
    print("CLINICAL VARIABLES SUMMARY")
    print(f"{'='*60}")
    print(df[clinical_vars_available].describe())
else:
    print(f"\n⚠️  No clinical variables configured for this dataset")

# ============================================================
# MISSING DATA SUMMARY
# ============================================================
print(f"\n{'='*60}")
print("MISSING DATA SUMMARY")
print(f"{'='*60}")
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_summary = missing_pct[missing_pct > 0].sort_values(ascending=False)

if len(missing_summary) > 0:
    print(missing_summary)
else:
    print("✅ No missing data!")

# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*60}")
print("EXPLORATION COMPLETE")
print(f"{'='*60}")
print(f"Dataset: {config['name']}")
print(f"Records: {len(df):,}")
print(f"Mortality rate: {df['outcome_binary'].mean():.1%}")
print(f"Ready for drift analysis: {config['has_precomputed_sofa']}")

if not config['has_precomputed_sofa']:
    print(f"\n⚠️  ACTION REQUIRED: Compute SOFA scores before running drift analysis")
else:
    print(f"\n✅ Ready to run: python 02_drift_analysis.py")
