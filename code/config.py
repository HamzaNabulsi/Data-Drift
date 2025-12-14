"""
Configuration file for drift analysis across multiple datasets.
Edit this file to specify which dataset to analyze.
"""

# ============================================================
# DATASET CONFIGURATIONS
# ============================================================

DATASETS = {
    'mimic': {
        'name': 'MIMIC (Mechanical Ventilation)',
        'data_path': r'C:\Users\sebastian.cajasordon\Documents\Data-Drift\data\mimic',
        'file': 'turning_interval_frequency.csv',
        'outcome_col': 'outcome',
        'outcome_positive': 'Deceased',  # Value indicating positive outcome (mortality)
        'score_col': 'sofa',  # Pre-computed SOFA score
        'year_col': 'admission_year',
        'year_bins': ['2008 - 2010', '2011 - 2013', '2014 - 2016', '2017 - 2019'],
        'demographic_cols': {
            'race': 'race',
            'gender': 'gender',
            'age': 'admission_age'
        },
        'clinical_cols': {
            'care_frequency': 'turning_interval_frequency',
            'los': 'days',
            'comorbidity': 'cci'
        },
        'has_precomputed_sofa': True,
        'description': 'BWH ICU patients on mechanical ventilation (2008-2019)'
    },

    'eicu_v1': {
        'name': 'eICU v1 (Sepsis Cohort)',
        'data_path': r'C:\Users\sebastian.cajasordon\Documents\Data-Drift\data\eicu',
        'file': 'sepsis_adult_eicu_v1.csv',
        'outcome_col': 'hospitaldischargestatus',  # Adjust based on actual column
        'outcome_positive': 'Expired',  # Adjust based on actual value
        'score_col': 'sofa',  # To be computed
        'year_col': 'hospitaladmityear',  # Adjust based on actual column
        'year_bins': None,  # Will be computed automatically
        'demographic_cols': {
            'race': 'ethnicity',
            'gender': 'gender',
            'age': 'age'
        },
        'clinical_cols': {},
        'has_precomputed_sofa': False,  # Needs computation
        'description': 'eICU sepsis cohort from multi-center database'
    },

    'eicu_v2': {
        'name': 'eICU v2 (Sepsis Cohort)',
        'data_path': r'C:\Users\sebastian.cajasordon\Documents\Data-Drift\data\eicu',
        'file': 'sepsis_adult_eicu_v2.csv',
        'outcome_col': 'hospitaldischargestatus',
        'outcome_positive': 'Expired',
        'score_col': 'sofa',
        'year_col': 'hospitaladmityear',
        'year_bins': None,
        'demographic_cols': {
            'race': 'ethnicity',
            'gender': 'gender',
            'age': 'age'
        },
        'clinical_cols': {},
        'has_precomputed_sofa': False,
        'description': 'eICU sepsis cohort v2 from multi-center database'
    },

    'mimic_mouthcare': {
        'name': 'MIMIC (Mouthcare Cohort)',
        'data_path': r'C:\Users\sebastian.cajasordon\Documents\Data-Drift\data\mimic',
        'file': 'mouthcare_interval_frequency.csv',
        'outcome_col': 'outcome',
        'outcome_positive': 'Deceased',
        'score_col': 'sofa',
        'year_col': 'admission_year',
        'year_bins': ['2008 - 2010', '2011 - 2013', '2014 - 2016', '2017 - 2019'],
        'demographic_cols': {
            'race': 'race',
            'gender': 'gender',
            'age': 'admission_age'
        },
        'clinical_cols': {
            'care_frequency': 'mouthcare_interval_frequency',
            'los': 'days',
            'comorbidity': 'cci'
        },
        'has_precomputed_sofa': True,
        'description': 'BWH ICU patients with mouthcare data (2008-2019)'
    },

    # Placeholder for future datasets
    'chinese_icu': {
        'name': 'Chinese ICU Dataset',
        'data_path': None,  # To be specified
        'file': None,
        'outcome_col': None,
        'outcome_positive': None,
        'score_col': 'sofa',
        'year_col': None,
        'year_bins': None,
        'demographic_cols': {},
        'clinical_cols': {},
        'has_precomputed_sofa': False,
        'description': 'Chinese ICU dataset (pending - Ziyue)'
    },

    'amsterdam_icu': {
        'name': 'Amsterdam ICU Dataset',
        'data_path': r'C:\Users\sebastian.cajasordon\Documents\Data-Drift\data\amsterdam',
        'file': 'salz_ml-scores_bias.csv',
        'outcome_col': 'death_hosp',
        'outcome_positive': 1,  # 1 = death, 0 = survived
        'score_col': 'sofa',
        'year_col': 'anchor_year_group',
        'year_bins': None,  # Will use individual years (2013-2021)
        'demographic_cols': {
            'gender': 'gender',
            'age': 'age'
            # Note: No race/ethnicity data in this dataset
        },
        'clinical_cols': {
            'los_icu': 'los_icu_day',
            'los_hospital': 'los_hospital_day',
            'sapsii': 'sapsii',
            'apsiii': 'apsiii',
            'oasis': 'oasis',
            'bmi': 'bmi'
        },
        'has_precomputed_sofa': True,
        'description': 'Amsterdam UMC ICU dataset (2013-2021)'
    }
}

# ============================================================
# ACTIVE DATASET - CHANGE THIS TO SWITCH DATASETS
# ============================================================
ACTIVE_DATASET = 'amsterdam_icu'  # Options: 'mimic', 'eicu_v1', 'eicu_v2', 'mimic_mouthcare', 'amsterdam_icu', etc.

# ============================================================
# OUTPUT CONFIGURATION
# ============================================================
OUTPUT_PATH = r'C:\Users\sebastian.cajasordon\Documents\Data-Drift\output'

# Create dataset-specific output subdirectories
OUTPUT_SUBDIRS = True  # If True, creates output/mimic/, output/eicu_v1/, etc.

# ============================================================
# ANALYSIS PARAMETERS
# ============================================================
ANALYSIS_CONFIG = {
    'min_sample_size': 30,  # Minimum patients per subgroup to include in analysis
    'age_bins': [0, 50, 65, 80, 200],  # Age group boundaries
    'age_labels': ['<50', '50-65', '65-80', '80+'],
    'care_quartiles': 4,  # Number of quartiles for care frequency analysis
    'figure_dpi': 300,  # Resolution for saved figures
    'figure_size': (16, 10),  # Figure dimensions in inches
}

# ============================================================
# HELPER FUNCTION
# ============================================================
def get_active_config():
    """Returns the configuration for the currently active dataset."""
    if ACTIVE_DATASET not in DATASETS:
        raise ValueError(f"Invalid ACTIVE_DATASET: {ACTIVE_DATASET}. "
                        f"Available options: {list(DATASETS.keys())}")
    return DATASETS[ACTIVE_DATASET]

def get_output_path(dataset_name=None):
    """Returns the output path for the specified dataset."""
    import os
    if dataset_name is None:
        dataset_name = ACTIVE_DATASET

    if OUTPUT_SUBDIRS:
        path = os.path.join(OUTPUT_PATH, dataset_name)
    else:
        path = OUTPUT_PATH

    os.makedirs(path, exist_ok=True)
    return path
