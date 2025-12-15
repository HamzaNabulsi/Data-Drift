# Subgroup-Specific Drift in ICU Severity Scores

[![LICENSE](https://img.shields.io/badge/license-CC%20BY--NC--SA-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://github.com/sebasmos/Data-Drift)

> **TL;DR:** ICU severity scores (OASIS, SAPS-II, APS-III) drift differently across demographic subgroups. We analyze 809,991 ICU admissions across 6 datasets from the US, Europe, and Asia (2001-2022) to quantify these disparities.

---

## Quick Start

```bash
# Setup
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv pip install -r requirements.txt

# Run full analysis
./run_all.sh

# Or run individual steps
./run_all.sh --analysis   # Run drift analysis only
./run_all.sh --figures    # Generate figures only
```

---

## Results

### Datasets Analyzed

| Dataset | N | Period | Mortality | Scores | Race Data |
|---------|---|--------|-----------|--------|-----------|
| MIMIC-III | 27,226 | 2001-2008 | 12.9% | OASIS, SAPS-II, APS-III | Yes |
| MIMIC-IV | 85,242 | 2008-2022 | 10.9% | OASIS, SAPS-II, APS-III | Yes |
| eICU | 289,503 | 2014-2015 | 8.7% | OASIS, SAPS-II, APS-III, APACHE | Yes |
| eICU-New | 371,855 | 2020-2021 | 12.7% | OASIS, SAPS-II, APS-III, APACHE | Yes |
| Amsterdam | 27,259 | 2013-2021 | 7.9% | SOFA, OASIS, SAPS-II, APS-III | No |
| Zhejiang | 7,932 | 2011-2022 | 14.7% | SOFA, OASIS, SAPS-II, APS-III | No |

**Total: 809,017 ICU admissions**

### Overall Drift by Dataset

| Dataset | OASIS | SAPS-II | APS-III | Direction |
|---------|-------|---------|---------|-----------|
| MIMIC-IV | +0.011 | +0.008 | +0.022 | Improving |
| Amsterdam | +0.049 | +0.054 | +0.076 | Improving |
| Zhejiang | +0.049 | +0.057 | +0.111 | Improving |
| eICU | -0.006 | +0.010 | +0.008 | Stable |
| eICU-New | -0.014 | -0.008 | -0.023 | Declining |

### Subgroup-Specific OASIS Drift

| Subgroup | MIMIC-IV | Amsterdam | Zhejiang | eICU | eICU-New |
|----------|----------|-----------|----------|------|----------|
| **Age 18-44** | +0.056 | +0.059 | -0.034 | +0.001 | -0.023 |
| **Age 80+** | -0.035 | +0.030 | -0.004 | -0.016 | +0.004 |
| **Male** | +0.026 | +0.069 | +0.030 | -0.005 | -0.017 |
| **Female** | -0.007 | +0.006 | +0.082 | -0.007 | -0.010 |
| **White** | +0.010 | - | - | -0.005 | -0.015 |
| **Black** | +0.013 | - | - | -0.019 | -0.027 |
| **Hispanic** | - | - | - | -0.021 | **-0.078** |
| **Asian** | **+0.114** | - | - | +0.046 | -0.040 |

### Key Findings

1. **COVID-era decline**: eICU-New (2020-21) shows universal performance degradation vs pre-COVID eICU (2014-15)
2. **Hispanic patients most affected**: -0.078 AUC decline in COVID era (largest drop)
3. **Asian patients improved most**: +0.114 AUC in MIMIC-IV over 14 years
4. **Age divergence**: Young (18-44) generally improve while elderly (80+) decline
5. **Geographic variation**: European/Asian datasets show improvement; US COVID-era shows decline

### Figures

![Summary Figure](figures/fig7_money_figure.png)
*Multi-panel summary: (A) Age group divergence, (B) Race disparities, (C) COVID impact, (D) Cross-dataset heatmap*

<details>
<summary>Additional Figures</summary>

![Overall Drift](figures/fig1_overall_drift_comparison.png)
*Figure 1: Overall score performance trends*

![Age Stratified](figures/fig2_age_stratified_drift.png)
*Figure 2: Age-stratified drift by dataset*

![Race Disparities](figures/fig3_race_disparities.png)
*Figure 3: Race/ethnicity disparities (US datasets)*

![Delta Summary](figures/fig4_drift_delta_summary.png)
*Figure 4: AUC change summary by subgroup*

![Heatmap](figures/fig5_comprehensive_heatmap.png)
*Figure 5: Comprehensive drift heatmap*

![COVID Comparison](figures/fig6_covid_era_comparison.png)
*Figure 6: Pre-COVID vs COVID era comparison*

</details>

---

## Project Structure

```
Data-Drift/
├── code/
│   ├── config.py                 # Dataset configurations
│   ├── batch_analysis.py         # Multi-dataset drift analysis
│   └── generate_all_figures.py   # Figure generation
├── data/
│   ├── mimiciii/                 # MIMIC-III data
│   ├── mimiciv/                  # MIMIC-IV data
│   ├── eicu/                     # eICU + eICU-New data
│   ├── amsterdam/                # Amsterdam UMC data
│   └── zhejiang/                 # Zhejiang Hospital data
├── figures/                      # Generated figures (fig1-7)
├── output/                       # Analysis results (CSV)
├── run_all.sh                    # Reproducibility script
└── requirements.txt              # Python dependencies
```

---

## Methodology

### Severity Scores Analyzed

| Score | Components | Range |
|-------|------------|-------|
| **OASIS** | 10 variables (age, GCS, vitals, ventilation, etc.) | 0-47 |
| **SAPS-II** | 17 variables (age, vitals, labs, chronic conditions) | 0-163 |
| **APS-III** | 20 variables (similar to APACHE III) | 0-299 |
| **SOFA** | 6 organ systems (respiratory, cardiovascular, etc.) | 0-24 |
| **APACHE** | Acute physiology + chronic health evaluation | 0-299 |

### Analysis Pipeline

1. **Data loading**: Standardize demographics across datasets
2. **Subgroup stratification**: Age (18-44, 45-64, 65-79, 80+), Gender, Race/Ethnicity
3. **AUC computation**: Score discrimination for mortality prediction per time period
4. **Drift quantification**: Delta AUC between first and last time periods

---

## Outputs

| File | Description |
|------|-------------|
| `output/all_datasets_drift_results.csv` | Full results (690 rows) |
| `output/all_datasets_drift_deltas.csv` | Drift changes (174 comparisons) |
| `output/table1_overall_drift.csv` | Overall drift by dataset/score |
| `output/table2_oasis_subgroup_drift.csv` | OASIS drift by subgroup |
| `figures/fig1-7*.png` | Publication-quality figures |

---

## Next Steps

### Immediate Priorities

- [ ] **Statistical testing**: Add confidence intervals and significance tests for drift comparisons
- [ ] **Calibration analysis**: Assess calibration drift (Brier score, calibration curves) alongside discrimination
- [ ] **Intersectional analysis**: Examine combinations (e.g., elderly Black females) for compounded disparities

### Extended Analysis

- [ ] **Recalibration strategies**: Test age-specific or era-specific recalibration approaches
- [ ] **Feature importance**: Identify which score components drive subgroup-specific drift
- [ ] **External validation**: Apply findings to additional datasets (ANZICS, UK ICU)

### Clinical Translation

- [ ] **Decision threshold analysis**: How drift affects clinical decision-making at specific thresholds
- [ ] **Fairness metrics**: Compute equalized odds, demographic parity across subgroups
- [ ] **Intervention simulation**: Model impact of periodic recalibration on patient outcomes

### Publication

- [ ] **Manuscript preparation**: Draft methods and results sections
- [ ] **Supplementary materials**: Compile all tables and sensitivity analyses
- [ ] **Code release**: Clean and document code for reproducibility

---

## Citation

```bibtex
@software{data_drift_2025,
  title={Subgroup-Specific Drift in ICU Severity Scores},
  author={Hamza, Nabulsi and Liu, Xiaoli and Celi, Leo Anthony and Cajas, Sebastian},
  year={2025},
  url={https://github.com/sebasmos/Data-Drift}
}
```

---

## License

[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
