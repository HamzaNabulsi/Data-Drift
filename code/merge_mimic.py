#!/usr/bin/env python3
"""
Merge MIMIC-III and MIMIC-IV datasets into a single combined dataset.

This script combines MIMIC-III (2001-2008) and MIMIC-IV (2008-2022) into a unified
dataset for continuous long-term drift analysis spanning 2001-2022.

Key operations:
1. Standardize column names (lowercase, consistent naming)
2. Harmonize ID columns (icustay_id vs stay_id)
3. Add source indicator column
4. Combine and save merged dataset

Output: data/mimic_combined/mimic_combined_ml-scores_bias.csv
"""

import pandas as pd
from pathlib import Path


def merge_mimic_datasets():
    """Merge MIMIC-III and MIMIC-IV into a combined dataset."""

    # Paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'

    mimic3_path = data_dir / 'mimic_iii' / 'mimiciii_ml-scores_bias.csv'
    mimic4_path = data_dir / 'mimic_iv_x' / 'mimiciv_ml-scores_bias.csv'
    output_dir = data_dir / 'mimic_combined'
    output_path = output_dir / 'mimic_combined_ml-scores_bias.csv'

    print("=" * 60)
    print("Merging MIMIC-III and MIMIC-IV datasets")
    print("=" * 60)

    # Load datasets
    print("\n1. Loading datasets...")
    df3 = pd.read_csv(mimic3_path)
    df4 = pd.read_csv(mimic4_path)

    print(f"   MIMIC-III: {len(df3):,} patients (2001-2008)")
    print(f"   MIMIC-IV:  {len(df4):,} patients (2008-2022)")

    # Standardize column names
    print("\n2. Standardizing column names...")

    # MIMIC-III: lowercase column names
    df3.columns = df3.columns.str.lower()

    # MIMIC-III has 'icustay_id', MIMIC-IV has 'stay_id'
    # Rename to common 'stay_id'
    if 'icustay_id' in df3.columns:
        df3 = df3.rename(columns={'icustay_id': 'stay_id'})

    # Add source indicator
    df3['source'] = 'mimic-iii'
    df4['source'] = 'mimic-iv'

    # Select common columns for merging
    # Core columns needed for analysis
    core_cols = [
        'subject_id', 'hadm_id', 'stay_id',
        'gender', 'age', 'ethnicity',
        'first_careunit', 'weight', 'height',
        'admittime', 'dischtime', 'dod',
        'intime', 'outtime',
        'los_hospital_day', 'los_icu_day', 'icu_admit_day',
        'death_hosp', 'deathtime_icu_hour',
        'admission_type',
        'anchor_year_group',
        # Score columns
        'apsiii', 'apsiii_prob',
        'oasis', 'oasis_prob',
        'sapsii', 'sapsii_prob',
        'sofa',
        # Source indicator
        'source'
    ]

    # Keep only columns that exist in both datasets
    cols3 = set(df3.columns)
    cols4 = set(df4.columns)
    common_cols = [c for c in core_cols if c in cols3 and c in cols4]

    print(f"   Common columns: {len(common_cols)}")

    # Subset to common columns
    df3_subset = df3[common_cols].copy()
    df4_subset = df4[common_cols].copy()

    # Combine datasets
    print("\n3. Combining datasets...")
    df_combined = pd.concat([df3_subset, df4_subset], ignore_index=True)

    print(f"   Combined: {len(df_combined):,} total patients")

    # Verify year groups
    print("\n4. Year group distribution:")
    year_counts = df_combined['anchor_year_group'].value_counts().sort_index()
    for year, count in year_counts.items():
        print(f"   {year}: {count:,}")

    # Summary statistics
    print("\n5. Summary statistics:")
    print(f"   Total patients: {len(df_combined):,}")
    print(f"   MIMIC-III: {(df_combined['source'] == 'mimic-iii').sum():,}")
    print(f"   MIMIC-IV:  {(df_combined['source'] == 'mimic-iv').sum():,}")
    print(f"   Overall mortality: {df_combined['death_hosp'].mean():.1%}")
    print(f"   MIMIC-III mortality: {df3_subset['death_hosp'].mean():.1%}")
    print(f"   MIMIC-IV mortality:  {df4_subset['death_hosp'].mean():.1%}")

    # Save combined dataset
    print(f"\n6. Saving to: {output_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(output_path, index=False)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   File size: {file_size_mb:.1f} MB")

    print("\n" + "=" * 60)
    print("Merge complete!")
    print("=" * 60)

    return df_combined


if __name__ == '__main__':
    merge_mimic_datasets()
