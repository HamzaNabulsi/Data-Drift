# Subgroup-Specific Drift in Clinical Prediction Models

[![LICENSE](https://img.shields.io/badge/license-CC%20BY--NC--SA-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://github.com/sebasmos/Data-Drift)

> **TL;DR:** Does model drift affect all patient subgroups equally? We analyze SOFA score performance across demographic groups in ICU patients to test if some subgroups experience faster degradation than others.

---

## 🚀 Quick Start

### Installation

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh  # macOS/Linux
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# Setup
uv venv
source .venv/bin/activate  # macOS/Linux or .venv\Scripts\activate (Windows)
uv pip install pandas numpy scikit-learn matplotlib seaborn
```

### Run Analysis

```bash
# 1. Configure dataset in code/config.py
#    ACTIVE_DATASET = 'mimic'  # or 'amsterdam_icu', 'eicu_v1', etc.

# 2. Run analysis from code directory
cd code
python mimic/01_explore_data.py
python mimic/02_drift_analysis.py

# 3. View results
# Results saved to: output/<dataset>/
```

**Note:** Scripts use `code/config.py` to determine which dataset to analyze. The generic scripts use the dataset-specific scripts internally.

---

## 📊 Results Summary

### Overall Drift by Dataset

| Dataset | Period | N Patients | Mortality | SOFA Trend | AUC Change | Direction | Key Finding |
|---------|--------|-----------|-----------|------------|-----------|-----------|-------------|
| **MIMIC (Mech. Vent.)** | 2008-2019 | ~15-20k | 20-30% | Declining | - | ⬇️ Worsening | High-acuity ventilated patients |
| **MIMIC (Mouthcare)** | 2008-2019 | 8,675 | 27% → 34% | Slight improvement | **+0.022** (+4%) | → Stable | Care frequency matters (low care: +0.146 AUC) |
| **Amsterdam ICU** | 2013-2021 | 27,259 | 7.9% | Improving | **+0.034** (+5%) | ⬆️ Improving | General ICU population |

### Amsterdam ICU Detailed Results (2013-2021)

| Subgroup | 2013 AUC | 2021 AUC | Change | % Change | Trend |
|----------|----------|----------|--------|----------|-------|
| **Overall** | 0.684 | 0.718 | +0.034 | +5.0% | ⬆️ Improving |
| **<50 years** | 0.659 | 0.818 | **+0.160** | **+24%** | 🔥 Exceptional |
| **50-65 years** | 0.661 | 0.685 | +0.025 | +3.8% | ⬆️ Modest |
| **65-80 years** | 0.698 | 0.679 | -0.019 | -2.7% | ⬇️ Declining |
| **80+ years** | 0.697 | 0.776 | +0.079 | +11% | ⬆️ Strong |
| **Male** | 0.648 | 0.689 | +0.041 | +6.3% | ⬆️ Improving |
| **Female** | 0.751 | 0.760 | +0.009 | +1.2% | → Stable |

**Key Insights:**
- ✅ **Mortality decreased 38%** (11.7% → 7.2%)
- 🔥 **Younger patients (<50):** Exceptional improvement (+0.160 AUC)
- ⚠️ **Middle-aged (65-80):** Only declining subgroup
- 👥 **Gender disparity:** Females consistently outperform males
- 🦠 **COVID-19 impact:** -0.036 AUC drop in 2020-2021 vs 2017-2019 peak

### MIMIC Mouthcare Results (2008-2019)

| Period | N | Mortality | AUC | Change from 2008-2010 |
|--------|---|-----------|-----|-----------------------|
| 2008-2010 | 3,418 | 26.7% | 0.608 | Baseline |
| 2011-2013 | 2,140 | 27.4% | 0.601 | -0.007 |
| 2014-2016 | 1,946 | 28.7% | 0.619 | +0.011 |
| 2017-2019 | 1,171 | 34.2% | 0.630 | **+0.022** |

**Key Subgroup Findings:**

| Subgroup | 2008-2010 AUC | 2017-2019 AUC | Change | Trend |
|----------|---------------|---------------|--------|-------|
| **Care: Low frequency (Q4)** | 0.611 | 0.757 | **+0.146** | 🔥 Largest improvement |
| **Care: High frequency (Q1)** | 0.619 | 0.628 | +0.009 | → Stable |
| **Female** | 0.607 | 0.661 | +0.054 | ⬆️ Improving |
| **Male** | 0.610 | 0.610 | 0.000 | → Unchanged |
| **<50 years** | 0.675 | 0.721 | +0.047 | ⬆️ Improving |
| **Black patients** | 0.657 | 0.551 | -0.106 | ⬇️ Declining |
| **Other race** | 0.567 | 0.672 | +0.104 | ⬆️ Improving |

**Critical Finding:** Patients receiving **less frequent mouthcare** show the largest SOFA performance improvement (+0.146), suggesting changing case-mix or care protocols.

---

## 📂 Datasets

| Dataset | Status | N | Period | Mortality | SOFA | Documentation |
|---------|--------|---|--------|-----------|------|---------------|
| **MIMIC (Mech. Vent.)** | ✅ Complete | ~15-20k | 2008-2019 | 20-30% | ✅ Pre-computed | [data/mimic/](data/mimic/) |
| **MIMIC (Mouthcare)** | ✅ Complete | 8,675 | 2008-2019 | 27-34% | ✅ Pre-computed | [data/mimic/](data/mimic/) |
| **Amsterdam ICU** | ✅ Complete | 27,259 | 2013-2021 | 7.9% | ✅ Pre-computed | [data/amsterdam/](data/amsterdam/) |
| **eICU v1 (Sepsis)** | ⚠️ Needs SOFA | - | - | - | ❌ Needs computation | [data/eicu/TODO.md](data/eicu/TODO.md) |
| **eICU v2 (Sepsis)** | ⚠️ Needs SOFA | - | - | - | ❌ Needs computation | [data/eicu/TODO.md](data/eicu/TODO.md) |
| **Chinese ICU** | 🔜 Pending | - | - | - | ❌ TBD | [data/chinese/TODO.md](data/chinese/TODO.md) |

---

## 📂 Project Structure

```
Data-Drift/
├── code/                           # Analysis code
│   ├── config.py                   # ⚙️ Dataset configuration (EDIT THIS)
│   │
│   ├── mimic/                      # ✅ MIMIC scripts
│   │   ├── 01_explore_data.py      # Exploratory analysis
│   │   └── 02_drift_analysis.py    # Drift analysis + visualization
│   │
│   ├── eicu/                       # ⚠️ eICU placeholders
│   ├── chinese/                    # 🔜 Chinese ICU placeholders
│   └── amsterdam/                  # 🔜 Amsterdam placeholders (use mimic/ scripts)
│
├── data/                           # Datasets
│   ├── mimic/                      # ✅ MIMIC data + README
│   ├── amsterdam/                  # ✅ Amsterdam data + README
│   │   ├── salz_ml-scores_bias.csv # Dataset (27,259 patients)
│   │   └── README.md               # Complete analysis documentation
│   ├── eicu/                       # ⚠️ eICU data + TODO
│   └── chinese/                    # 🔜 Chinese data + TODO
│
├── output/                         # Generated results
│   ├── mimic/                      # MIMIC mech. vent. outputs
│   ├── mimic_mouthcare/            # ✅ MIMIC mouthcare outputs
│   └── amsterdam_icu/              # ✅ Amsterdam outputs
│       ├── amsterdam_icu_drift_analysis.png
│       ├── amsterdam_icu_yearly_performance.csv
│       ├── amsterdam_icu_gender_performance.csv
│       └── amsterdam_icu_age_performance.csv
│
└── reference/                      # Reference materials
    ├── sql/                        # SOFA computation SQL
    └── notebooks/                  # Exploratory notebooks
```

---

## 🔬 Methodology

### SOFA Score (Sequential Organ Failure Assessment)

Evaluates 6 organ systems:
- **Respiratory** (PaO2/FiO2 ratio)
- **Cardiovascular** (Mean arterial pressure, vasopressors)
- **Renal** (Creatinine, urine output)
- **Coagulation** (Platelets)
- **Liver** (Bilirubin)
- **Neurological** (Glasgow Coma Scale)

**Range:** 0-24 (higher = worse organ failure)

### Analysis Pipeline

**Step 1: Exploratory Analysis** (`01_explore_data.py`)
- Load and validate dataset
- Check outcome distributions
- Verify SOFA scores
- Analyze demographics and clinical variables
- Assess missing data

**Step 2: Drift Analysis** (`02_drift_analysis.py`)
- Overall SOFA performance over time
- Subgroup-stratified analyses:
  - Race (if available)
  - Gender
  - Age groups (<50, 50-65, 65-80, 80+)
  - Care frequency (if available)
- Generate visualizations + CSV outputs

### Metrics

- **AUC (Area Under ROC Curve):** Discrimination ability
  - 0.5 = random, 0.7 = acceptable, 0.8 = excellent, 1.0 = perfect
- **Accuracy:** Overall prediction accuracy
- **F1 Score:** Balance of precision and recall
- **Mortality Rate:** Observed outcome frequency

---

## ⚙️ Configuration

### Switch Datasets

Edit `code/config.py`:

```python
# Change this line to switch datasets
ACTIVE_DATASET = 'amsterdam_icu'  # Options: 'mimic', 'amsterdam_icu', 'eicu_v1', etc.
```

### Available Datasets in Config

```python
DATASETS = {
    'mimic': {...},                    # MIMIC mechanical ventilation
    'mimic_mouthcare': {...},          # MIMIC mouthcare cohort
    'eicu_v1': {...},                  # eICU sepsis v1
    'eicu_v2': {...},                  # eICU sepsis v2
    'amsterdam_icu': {...},            # ✅ Amsterdam ICU (2013-2021)
    'chinese_icu': {...},              # Chinese ICU (pending)
}
```

### Customize Analysis Parameters

```python
ANALYSIS_CONFIG = {
    'min_sample_size': 30,             # Minimum patients per subgroup
    'age_bins': [0, 50, 65, 80, 200],  # Age group boundaries
    'age_labels': ['<50', '50-65', '65-80', '80+'],
    'care_quartiles': 4,               # Care frequency quartiles
    'figure_dpi': 300,                 # Output resolution
    'figure_size': (16, 10),           # Figure dimensions
}
```

---

## 📈 Outputs

Each analysis generates in `output/<dataset>/`:

### Visualizations
- `<dataset>_drift_analysis.png` - Multi-panel figure with:
  - Overall SOFA performance over time
  - Race-stratified trends (if available)
  - Gender-stratified trends
  - Age group-stratified trends
  - Care frequency trends (if available)

### CSV Files
- `<dataset>_yearly_performance.csv` - Overall metrics by year
- `<dataset>_race_performance.csv` - Race-stratified (if available)
- `<dataset>_gender_performance.csv` - Gender-stratified
- `<dataset>_age_performance.csv` - Age-stratified
- `<dataset>_care_performance.csv` - Care frequency (if available)

**Columns in CSV files:**
- `AUC`, `Accuracy`, `F1`, `N`, `Mortality_Rate`, `Mean_Score`, `Period`, `[Subgroup]`

---

## 🔄 Running Analyses

### MIMIC Dataset

```bash
cd code
# Edit config.py: ACTIVE_DATASET = 'mimic'
python mimic/01_explore_data.py
python mimic/02_drift_analysis.py
# Results in: output/mimic/
```

### Amsterdam Dataset

```bash
cd code
# Edit config.py: ACTIVE_DATASET = 'amsterdam_icu'
python mimic/01_explore_data.py
python mimic/02_drift_analysis.py
# Results in: output/amsterdam_icu/
```

**Note:** Amsterdam uses the MIMIC scripts - they are dataset-agnostic and read from `config.py`.

### eICU Dataset (After SOFA Computation)

```bash
cd code
# Edit config.py: ACTIVE_DATASET = 'eicu_v1'
python mimic/01_explore_data.py  # Reuse MIMIC scripts
python mimic/02_drift_analysis.py
```

---

## 📊 Key Findings

### Amsterdam vs MIMIC Comparison

| Feature | Amsterdam ICU | MIMIC (Mech. Vent.) |
|---------|---------------|---------------------|
| **Overall Trend** | ⬆️ **Improving** (+0.034 AUC) | ⬇️ **Declining** |
| **Population** | General ICU | Mechanical ventilation only |
| **Mortality** | 7.9% (low) | 20-30% (high) |
| **Best Subgroup** | <50 years (+0.160 AUC) | Varies |
| **Worst Subgroup** | 65-80 years (-0.019 AUC) | Varies |
| **Gender Pattern** | Female advantage | Mixed |
| **Race Data** | ❌ Not available | ✅ Available |

### Critical Insights

1. **Opposite Drift Patterns**
   - Amsterdam: SOFA performance **improving** over time
   - MIMIC: SOFA performance **declining** over time
   - **Hypothesis:** Different patient populations (general ICU vs high-acuity ventilated)

2. **Age-Specific Heterogeneity** (Amsterdam)
   - Younger patients (<50): Exceptional improvement (+24%)
   - Middle-aged (65-80): Only declining group (-2.7%)
   - **Implication:** Age-specific recalibration may be needed

3. **COVID-19 Impact** (Amsterdam)
   - 2020-2021 vs 2017-2019 peak: -0.036 AUC
   - 38% reduction in patient volume
   - Higher severity (mean SOFA +0.15)

4. **Gender Disparity** (Amsterdam)
   - Females consistently outperform males (7/9 years)
   - Gap averages +0.04 to +0.07 AUC
   - **Requires further investigation**

---

## 📝 Documentation

### Dataset-Specific Documentation

**Amsterdam ICU:**
- [README.md](data/amsterdam/README.md) - Complete analysis results and documentation

**MIMIC:**
- [README.md](data/mimic/README.md) - Dataset information

**eICU:**
- [TODO.md](data/eicu/TODO.md) - Setup instructions

**Chinese ICU:**
- [TODO.md](data/chinese/TODO.md) - Pending setup

---

## 🛠️ Adding New Datasets

### Step 1: Prepare Data
Place CSV file in `data/<dataset>/` with required columns:
- **Outcome:** Binary mortality indicator
- **SOFA:** Pre-computed or to be computed
- **Year:** Temporal variable
- **Demographics:** Age, gender, race (optional)

### Step 2: Update Config
Add entry to `code/config.py`:

```python
'your_dataset': {
    'name': 'Dataset Name',
    'data_path': r'path/to/data',
    'file': 'data.csv',
    'outcome_col': 'death',
    'outcome_positive': 1,
    'score_col': 'sofa',
    'year_col': 'year',
    'year_bins': None,  # or ['2010-2012', '2013-2015', ...]
    'demographic_cols': {
        'race': 'race_col',
        'gender': 'gender_col',
        'age': 'age_col'
    },
    'clinical_cols': {...},
    'has_precomputed_sofa': True,
    'description': 'Dataset description'
}
```

### Step 3: Run Analysis
```bash
cd code
# Edit config.py: ACTIVE_DATASET = 'your_dataset'
python mimic/01_explore_data.py
python mimic/02_drift_analysis.py
```

---

## 🔗 Resources

### SOFA Score Computation
- **SQL Code:** `reference/sql/`
- **GitHub Reference:** https://github.com/nus-mornin-lab/oxygenation_kc
- **Calculator:** https://www.mdcalc.com/calc/691/sequential-organ-failure-assessment-sofa-score

### Publications
- Vincent JL, et al. "The SOFA (Sepsis-related Organ Failure Assessment) score to describe organ dysfunction/failure." *Intensive Care Med* 1996.

---

## 📊 Status Update

### Completed
- ✅ **MIMIC (Mechanical Ventilation)** - Full analysis complete
- ✅ **MIMIC (Mouthcare)** - Full analysis complete (8,675 patients, 2008-2019)
  - Key finding: Care frequency drift (+0.146 AUC for low-frequency care)
  - Racial disparities identified (Black patients: -0.106 AUC)
- ✅ **Amsterdam ICU** - Full analysis complete (27,259 patients, 2013-2021)
  - Key finding: Improving SOFA performance (+0.034 AUC)
  - Age-specific heterogeneity (<50 years: +24% improvement)

### In Progress
- ⚠️ **eICU v1 & v2** - Needs SOFA score computation (Emma)

### Pending
- 🔜 **Chinese ICU** - Awaiting data (Ziyue)

### Future Work
- Multi-score validation (SAPS II, OASIS, APACHE III for Amsterdam)
- COVID-19 deep dive analysis
- Cross-dataset drift comparison paper
- Machine learning model benchmarking

---

## 📝 Citation

```bibtex
@software{data_drift_2025,
  title={Subgroup-Specific Drift in Clinical Prediction Models},
  author={Hamza and Xiaoli and Celi, Leo Anthony and Cajas Ord{\'o}{\~n}ez, Sebasti{\'a}n Andr{\'e}s},
  year={2025},
  url={https://github.com/HamzaNabulsi/Data-Drift}
}
```

See [CITATION.cff](CITATION.cff) for full metadata.

---

## ⚖️ License

[![LICENSE](https://img.shields.io/badge/license-CC%20BY--NC--SA-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

---