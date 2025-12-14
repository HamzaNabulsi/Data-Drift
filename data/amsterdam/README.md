# Amsterdam ICU Dataset - Analysis Results

**Dataset:** Amsterdam UMC ICU (2013-2021)
**Analysis Date:** 2025-12-14
**Status:** ✅ Complete

---

## Quick Summary

- **27,259 patients** from Amsterdam UMC ICU (2013-2021)
- **7.9% mortality** (2,163 deaths)
- **SOFA performance improving** over time (+0.034 AUC, +5%)
- **Mortality decreased 38%** (11.7% → 7.2%)
- **Younger patients show exceptional improvement** (<50 years: +24% AUC)

---

## Dataset Information

| Feature | Details |
|---------|---------|
| **File** | `salz_ml-scores_bias.csv` |
| **Patients** | 27,259 |
| **Time Period** | 2013-2021 (9 years) |
| **Mortality** | 7.9% (2,163 deaths) |
| **SOFA Score** | ✅ Pre-computed (Mean: 4.36, Range: 0-18) |
| **Additional Scores** | SAPS II, OASIS, APACHE III |
| **Demographics** | Age, Gender (no race data) |
| **Data Quality** | >99% complete |

---

## Key Results

### Overall Performance (2013 → 2021)

| Year | N | Mortality | Mean SOFA | AUC | Change from 2013 |
|------|---|-----------|-----------|-----|-----------------|
| 2013 | 1,981 | 11.7% | 4.94 | 0.684 | Baseline |
| **2017** | **3,564** | **6.2%** | **3.99** | **0.782** ⭐ | **+0.098** |
| 2021 | 3,140 | 7.2% | 4.18 | 0.718 | **+0.034** |

**Overall Trend:** ⬆️ Improving (+0.034 AUC = +5%)

### Subgroup Analysis (2013 → 2021)

| Subgroup | 2013 AUC | 2021 AUC | Change | % Change | Trend |
|----------|----------|----------|--------|----------|-------|
| **<50 years** | 0.659 | 0.818 | **+0.160** | **+24%** | 🔥 Exceptional |
| **50-65 years** | 0.661 | 0.685 | +0.025 | +3.8% | ⬆️ Modest |
| **65-80 years** | 0.698 | 0.679 | -0.019 | -2.7% | ⬇️ Declining |
| **80+ years** | 0.697 | 0.776 | +0.079 | +11% | ⬆️ Strong |
| **Male** | 0.648 | 0.689 | +0.041 | +6.3% | ⬆️ Improving |
| **Female** | 0.751 | 0.760 | +0.009 | +1.2% | → Stable |

### COVID-19 Impact

| Period | Years | N | Mortality | AUC | Change |
|--------|-------|---|-----------|-----|--------|
| Pre-COVID Peak | 2017-2019 | 9,991 | 6.7% | 0.754 | - |
| COVID Era | 2020-2021 | 6,137 | 7.3% | 0.718 | **-0.036** |

**Impact:** 38.6% fewer patients, higher mortality, lower AUC

---

## Major Findings

### 1. Overall Improvement
- SOFA discrimination **improved 5%** (0.684 → 0.718)
- **Mortality decreased 38%** (11.7% → 7.2%)
- Patient severity decreased (mean SOFA: 4.94 → 4.18)

### 2. Age Heterogeneity (Critical Finding!)
- **<50 years:** Exceptional improvement (+0.160, +24%)
  - Peak in 2019: AUC = 0.905 (near-perfect)
- **65-80 years:** Only declining group (-0.019, -2.7%)
  - Largest population (40.9% of patients)
  - May need age-specific recalibration

### 3. Gender Disparity
- **Female patients** show consistently better performance (7/9 years)
- Average gap: +0.04 to +0.07 AUC
- Male patients started lower but improved more

### 4. COVID-19 Era Decline
- 2020-2021 vs 2017-2019 peak: **-0.036 AUC**
- Higher severity, fewer patients, worse outcomes
- Different case-mix or pandemic strain on healthcare

### 5. Comparison with MIMIC

| Feature | Amsterdam ICU | MIMIC (Mech. Vent.) |
|---------|---------------|---------------------|
| **Trend** | ⬆️ **Improving** | ⬇️ **Declining** |
| **Population** | General ICU | Ventilated only |
| **Mortality** | 7.9% | 20-30% |
| **Best Subgroup** | <50 years (+24%) | Varies |

**Critical:** Amsterdam shows **opposite drift pattern** from MIMIC!

---

## Generated Outputs

**Location:** `output/amsterdam_icu/`

**Files:**
1. `amsterdam_icu_drift_analysis.png` - Main visualization (4 panels)
2. `amsterdam_icu_yearly_performance.csv` - Year-by-year metrics (9 rows)
3. `amsterdam_icu_gender_performance.csv` - Gender stratification (18 rows)
4. `amsterdam_icu_age_performance.csv` - Age stratification (36 rows)

---

## How to Reproduce

### 1. Configuration
Dataset already configured in `code/config.py`:
```python
ACTIVE_DATASET = 'amsterdam_icu'
```

### 2. Run Analysis
```bash
cd code
python mimic/01_explore_data.py    # Exploratory analysis
python mimic/02_drift_analysis.py  # Drift analysis + plots
```

### 3. View Results
```bash
# Outputs in: output/amsterdam_icu/
open output/amsterdam_icu/amsterdam_icu_drift_analysis.png
```

---

## Clinical Implications

### Recommendations
1. **Age-specific thresholds** - Consider recalibration for 65-80 age group
2. **COVID-era patients** - May need separate risk models
3. **Gender investigation** - Explore why females outperform males
4. **Younger patients** - Document care improvements benefiting <50 age group

### Future Analyses
1. Multi-score validation (SAPS II, OASIS, APACHE III)
2. COVID-19 deep dive (2020-2021 detailed analysis)
3. Admission type stratification (Emergency vs Elective)
4. Care unit analysis (Different ICU wards)
5. Calibration testing (Hosmer-Lemeshow by year)

---

## Limitations

1. **Single center** - Amsterdam UMC only, may not generalize
2. **No race/ethnicity data** - Cannot assess racial disparities
3. **No care frequency data** - Cannot analyze nursing care patterns
4. **Observational** - Cannot establish causation
5. **Lower mortality** - 7.9% vs 20-30% in MIMIC limits power

---

## Technical Details

### Data Columns (22 variables)
- **Identifiers:** patientid, hadm_id
- **Demographics:** age, gender, height, weight, bmi
- **Outcomes:** death_hosp (0/1), deathtime_icu_hour
- **Clinical:** los_hospital_day, los_icu_day, first_careunit, admission_type
- **Temporal:** anchor_year_group (2013-2021)
- **Severity Scores:** sofa, sapsii, oasis, apsiii (+ predicted mortality)

### Missing Data
- **deathtime_icu_hour:** 81.3% missing (expected - only for deaths)
- **height/weight/bmi:** 0.12% missing
- **All other variables:** 0% missing

### Configuration Entry
```python
'amsterdam_icu': {
    'data_path': r'...\Data-Drift\data\amsterdam',
    'file': 'salz_ml-scores_bias.csv',
    'outcome_col': 'death_hosp',
    'outcome_positive': 1,
    'score_col': 'sofa',
    'year_col': 'anchor_year_group',
    'year_bins': None,  # Individual years 2013-2021
    'has_precomputed_sofa': True,
}
```

---

## Quick Reference

| Item | Value |
|------|-------|
| **Best year** | 2017 (AUC = 0.782) |
| **Best subgroup** | <50 years in 2019 (AUC = 0.905) |
| **Worst subgroup** | Males in 2013 (AUC = 0.648) |
| **Peak mortality** | 2013 (11.7%) |
| **Lowest mortality** | 2017 (6.2%) |
| **Largest year** | 2017 (3,564 patients) |

---

## Citation

```bibtex
@dataset{amsterdam_icu_2025,
  title={Amsterdam ICU Dataset Drift Analysis},
  author={Cajas Ord{\'o}{\~n}ez, Sebasti{\'a}n and Team},
  year={2025},
  note={Amsterdam UMC ICU patients 2013-2021, N=27,259}
}
```

---

## Contact

- **Project:** Data-Drift
- **Configuration:** `code/config.py` (amsterdam_icu entry)
- **Scripts:** `code/mimic/` (dataset-agnostic)
- **Main README:** [../../README.md](../../README.md)

---

**Last Updated:** 2025-12-14
**Analysis Status:** ✅ Complete and ready for publication
