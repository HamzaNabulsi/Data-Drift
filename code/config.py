"""
Configuration file for drift analysis across multiple datasets.
Edit this file to specify which dataset to analyze.
"""

import os
from pathlib import Path

# ============================================================
# BASE PATHS - Automatically detected based on script location
# ============================================================
_CONFIG_DIR = Path(__file__).parent  # code/
_BASE_DIR = _CONFIG_DIR.parent        # Data-Drift/
_DATA_DIR = _BASE_DIR / 'data'
_OUTPUT_DIR = _BASE_DIR / 'output'

# ============================================================
# DATASET CONFIGURATIONS
# ============================================================

DATASETS = {
    'mimic': {
        'name': 'MIMIC (Mechanical Ventilation)',
        'data_path': str(_DATA_DIR / 'mimic_iv_lc'),
        'file': 'turning_interval_frequency.csv',
        'outcome_col': 'outcome',
        'outcome_positive': 'Deceased',  # Value indicating positive outcome (mortality)
        'score_col': 'sofa',  # Pre-computed SOFA score
        'score_cols': ['sofa'],  # All available scores
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

    'mimic_mouthcare': {
        'name': 'MIMIC (Mouthcare Cohort)',
        'data_path': str(_DATA_DIR / 'mimic_iv_lc'),
        'file': 'mouthcare_interval_frequency.csv',
        'outcome_col': 'outcome',
        'outcome_positive': 'Deceased',
        'score_col': 'sofa',
        'score_cols': ['sofa'],
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

    # ============================================================
    # NEW SALZ DATASETS (ML-Scores Bias Study)
    # ============================================================

    'mimic_combined': {
        'name': 'MIMIC Combined (2001-2022)',
        'data_path': str(_DATA_DIR / 'mimic_combined'),
        'file': 'mimic_combined_ml-scores_bias.csv',
        'outcome_col': 'death_hosp',
        'outcome_positive': 1,
        'score_col': 'oasis',  # Primary score
        'score_cols': ['sofa', 'oasis', 'sapsii', 'apsiii'],  # All available scores
        'year_col': 'anchor_year_group',
        'year_bins': ['2001 - 2008', '2008 - 2010', '2011 - 2013', '2014 - 2016', '2017 - 2019', '2020 - 2022'],
        'demographic_cols': {
            'race': 'ethnicity',
            'gender': 'gender',
            'age': 'age'
        },
        'clinical_cols': {
            'los_icu': 'los_icu_day',
            'los_hospital': 'los_hospital_day',
            'source': 'source'  # mimic-iii or mimic-iv
        },
        'has_precomputed_sofa': True,
        'description': 'MIMIC-III + MIMIC-IV combined (2001-2022) - 21 years continuous drift analysis'
    },

    'mimiciii': {
        'name': 'MIMIC-III (2001-2008)',
        'data_path': str(_DATA_DIR / 'mimic_iii'),
        'file': 'mimiciii_ml-scores_bias.csv',
        'outcome_col': 'death_hosp',
        'outcome_positive': 1,
        'score_col': 'oasis',  # Primary score
        'score_cols': ['sofa', 'oasis', 'sapsii', 'apsiii'],  # All available scores including SOFA
        'year_col': 'anchor_year_group',
        'year_bins': None,  # Single period: 2001-2008
        'demographic_cols': {
            'race': 'ethnicity',
            'gender': 'gender',
            'age': 'age'
        },
        'clinical_cols': {
            'los_icu': 'los_icu_day',
            'los_hospital': 'los_hospital_day'
        },
        'has_precomputed_sofa': True,
        'description': 'MIMIC-III ICU dataset (2001-2008) - single time bin'
    },

    'mimiciv': {
        'name': 'MIMIC-IV (2008-2022)',
        'data_path': str(_DATA_DIR / 'mimic_iv_x'),
        'file': 'mimiciv_ml-scores_bias.csv',
        'outcome_col': 'death_hosp',
        'outcome_positive': 1,
        'score_col': 'oasis',  # Primary score
        'score_cols': ['sofa', 'oasis', 'sapsii', 'apsiii'],  # All available scores including SOFA
        'year_col': 'anchor_year_group',
        'year_bins': ['2008 - 2010', '2011 - 2013', '2014 - 2016', '2017 - 2019', '2020 - 2022'],
        'demographic_cols': {
            'race': 'ethnicity',
            'gender': 'gender',
            'age': 'age'
        },
        'clinical_cols': {
            'los_icu': 'los_icu_day',
            'los_hospital': 'los_hospital_day'
        },
        'has_precomputed_sofa': True,
        'description': 'MIMIC-IV ICU dataset (2008-2022) - 5 time bins'
    },

    'eicu_combined': {
        'name': 'eICU Combined (2014-2021)',
        'data_path': str(_DATA_DIR / 'eicu_combined'),
        'file': 'eicu_combined_ml-scores_bias.csv',
        'outcome_col': 'hosp_mort',
        'outcome_positive': 1,
        'score_col': 'oasis',  # Primary score
        'score_cols': ['sofa', 'oasis', 'sapsii', 'apsiii', 'apachescore'],  # All scores including SOFA
        'year_col': 'hospitaldischargeyear',
        'year_bins': [2014, 2015, 2020, 2021],  # 4 year bins for temporal analysis
        'demographic_cols': {
            'race': 'ethnicity',
            'gender': 'gender',
            'age': 'age'
        },
        'clinical_cols': {
            'los_icu': 'los_icu_day',
            'los_hospital': 'los_hosp_day',
            'region': 'region',
            'source': 'source'  # eicu-old or eicu-new
        },
        'has_precomputed_sofa': True,
        'description': 'eICU-old + eICU-new combined (2014-2021) - 7 years temporal drift analysis'
    },

    'eicu': {
        'name': 'eICU (2014-2015)',
        'data_path': str(_DATA_DIR / 'eicu'),
        'file': 'eicu_ml-scores_bias.csv',
        'outcome_col': 'hosp_mort',
        'outcome_positive': 1,
        'score_col': 'oasis',  # Primary score
        'score_cols': ['sofa', 'oasis', 'sapsii', 'apsiii', 'apachescore'],  # All scores including SOFA
        'year_col': 'hospitaldischargeyear',
        'year_bins': None,  # Will use 2014, 2015
        'demographic_cols': {
            'race': 'ethnicity',
            'gender': 'gender',
            'age': 'age'
        },
        'clinical_cols': {
            'los_icu': 'los_icu_day',
            'los_hospital': 'los_hosp_day',
            'region': 'region'
        },
        'has_precomputed_sofa': True,
        'description': 'eICU Collaborative Research Database (2014-2015)'
    },

    'eicu_new': {
        'name': 'eICU-New (2020-2021)',
        'data_path': str(_DATA_DIR / 'eicu_new'),
        'file': 'eicu_new_ml-scores_bias.csv',
        'outcome_col': 'hosp_mort',
        'outcome_positive': 1,
        'score_col': 'oasis',  # Primary score
        'score_cols': ['sofa', 'oasis', 'sapsii', 'apsiii', 'apachescore'],  # All scores including SOFA
        'year_col': 'hospitaldischargeyear',
        'year_bins': None,  # Will use 2020, 2021
        'demographic_cols': {
            'race': 'ethnicity',
            'gender': 'gender',
            'age': 'age'
        },
        'clinical_cols': {
            'los_icu': 'los_icu_day',
            'los_hospital': 'los_hosp_day',
            'region': 'region'
        },
        'has_precomputed_sofa': True,
        'description': 'eICU dataset (2020-2021)'
    },

    'zhejiang': {
        'name': 'Zhejiang ICU (2011-2022)',
        'data_path': str(_DATA_DIR / 'zhejiang'),
        'file': 'zhejiang_ml-scores_bias.csv',
        'outcome_col': 'death_hosp',
        'outcome_positive': 1,
        'score_col': 'sofa',  # Has SOFA!
        'score_cols': ['sofa', 'oasis', 'sapsii', 'apsiii'],  # All scores including SOFA
        'year_col': 'anchor_year_group',
        'year_bins': ['2011 - 2013', '2014 - 2016', '2017 - 2019', '2020 - 2022'],
        'demographic_cols': {
            'gender': 'gender',
            'age': 'age'
            # No race/ethnicity data
        },
        'clinical_cols': {
            'los_icu': 'los_icu_day',
            'los_hospital': 'los_hospital_day'
        },
        'has_precomputed_sofa': True,
        'description': 'Zhejiang Provincial Hospital ICU (2011-2022, China)'
    },

    'saltz': {
        'name': 'Saltz ICU (2013-2021)',
        'data_path': str(_DATA_DIR / 'saltz'),
        'file': 'salz_ml-scores_bias.csv',
        'outcome_col': 'death_hosp',
        'outcome_positive': 1,
        'score_col': 'sofa',  # Has SOFA!
        'score_cols': ['sofa', 'oasis', 'sapsii', 'apsiii'],  # All scores
        'year_col': 'anchor_year_group',
        'year_bins': None,  # Will use individual years (2013-2021)
        'demographic_cols': {
            'gender': 'gender',
            'age': 'age'
            # No race/ethnicity data
        },
        'clinical_cols': {
            'los_icu': 'los_icu_day',
            'los_hospital': 'los_hospital_day',
            'bmi': 'bmi'
        },
        'has_precomputed_sofa': True,
        'description': 'Saltz ICU dataset (2013-2021)'
    },

    # ============================================================
    # LEGACY CONFIGS (kept for backwards compatibility)
    # ============================================================

    'eicu_v1': {
        'name': 'eICU v1 (Sepsis Cohort - Legacy)',
        'data_path': str(_DATA_DIR / 'eicu'),
        'file': 'sepsis_adult_eicu_v1.csv',
        'outcome_col': 'hospitaldischargestatus',
        'outcome_positive': 'Expired',
        'score_col': 'sofa',
        'score_cols': ['sofa'],
        'year_col': 'hospitaladmityear',
        'year_bins': None,
        'demographic_cols': {
            'race': 'ethnicity',
            'gender': 'gender',
            'age': 'age'
        },
        'clinical_cols': {},
        'has_precomputed_sofa': False,
        'description': 'eICU sepsis cohort v1 (legacy - needs SOFA computation)'
    },

    'eicu_v2': {
        'name': 'eICU v2 (Sepsis Cohort - Legacy)',
        'data_path': str(_DATA_DIR / 'eicu'),
        'file': 'sepsis_adult_eicu_v2.csv',
        'outcome_col': 'hospitaldischargestatus',
        'outcome_positive': 'Expired',
        'score_col': 'sofa',
        'score_cols': ['sofa'],
        'year_col': 'hospitaladmityear',
        'year_bins': None,
        'demographic_cols': {
            'race': 'ethnicity',
            'gender': 'gender',
            'age': 'age'
        },
        'clinical_cols': {},
        'has_precomputed_sofa': False,
        'description': 'eICU sepsis cohort v2 (legacy - needs SOFA computation)'
    }
}

# ============================================================
# ACTIVE DATASET - CHANGE THIS TO SWITCH DATASETS
# ============================================================
ACTIVE_DATASET = 'mimic_mouthcare'  # Options: 'mimic', 'eicu_v1', 'eicu_v2', 'mimic_mouthcare', 'saltz', etc.

# ============================================================
# OUTPUT CONFIGURATION
# ============================================================
OUTPUT_PATH = str(_OUTPUT_DIR)

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
