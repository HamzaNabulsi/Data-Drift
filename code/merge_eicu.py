#!/usr/bin/env python3
"""
Merge eICU-old and eICU-new datasets into a single combined dataset.

This script combines eICU (2014-2015) and eICU-new (2020-2021) into a unified
dataset for continuous temporal drift analysis (7 years).

Key operations:
1. Load both datasets (identical column structure)
2. Add source indicator column
3. Combine and save merged dataset

Output: data/eicu_combined/eicu_combined_ml-scores_bias.csv
"""

import pandas as pd
from pathlib import Path


def merge_eicu_datasets():
    """Merge eICU-old and eICU-new into a combined dataset."""

    # Paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'

    eicu_old_path = data_dir / 'eicu' / 'eicu_ml-scores_bias.csv'
    eicu_new_path = data_dir / 'eicu_new' / 'eicu_new_ml-scores_bias.csv'
    output_dir = data_dir / 'eicu_combined'
    output_path = output_dir / 'eicu_combined_ml-scores_bias.csv'

    print("=" * 60)
    print("Merging eICU-old and eICU-new datasets")
    print("=" * 60)

    # Load datasets
    print("\n1. Loading datasets...")
    df_old = pd.read_csv(eicu_old_path)
    df_new = pd.read_csv(eicu_new_path)

    print(f"   eICU-old: {len(df_old):,} patients (2014-2015)")
    print(f"   eICU-new: {len(df_new):,} patients (2020-2021)")

    # Add source indicator
    print("\n2. Adding source indicators...")
    df_old['source'] = 'eicu-old'
    df_new['source'] = 'eicu-new'

    # Combine datasets (columns are identical)
    print("\n3. Combining datasets...")
    df_combined = pd.concat([df_old, df_new], ignore_index=True)

    print(f"   Combined: {len(df_combined):,} total patients")

    # Verify year distribution
    print("\n4. Year distribution:")
    year_counts = df_combined['hospitaldischargeyear'].value_counts().sort_index()
    for year, count in year_counts.items():
        source = 'eicu-old' if year < 2020 else 'eicu-new'
        print(f"   {year}: {count:,} ({source})")

    # Summary statistics
    print("\n5. Summary statistics:")
    print(f"   Total patients: {len(df_combined):,}")
    print(f"   eICU-old: {(df_combined['source'] == 'eicu-old').sum():,}")
    print(f"   eICU-new: {(df_combined['source'] == 'eicu-new').sum():,}")
    print(f"   Overall mortality: {df_combined['hosp_mort'].mean():.1%}")
    print(f"   eICU-old mortality: {df_old['hosp_mort'].mean():.1%}")
    print(f"   eICU-new mortality: {df_new['hosp_mort'].mean():.1%}")

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
    merge_eicu_datasets()
