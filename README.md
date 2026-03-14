# Uniform Recalibration Is Unsafe: Subgroup-Specific Drift in ICU Severity Scores

[![LICENSE](https://img.shields.io/badge/license-CC%20BY--NC--SA-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://github.com/sebasmos/Data-Drift)

> **Hypothesis:** Applying a single recalibration factor to ICU severity scores is clinically unsafe. Aggregate performance improvement conceals targeted degradation in minority and intersectional subgroups, meaning uniform recalibration may actively harm the patients it is intended to help.

---

## Table of Contents

1. [Key Findings](#key-findings)
2. [Datasets](#datasets)
3. [Statistical Methods](#statistical-methods)
4. [SOFA Threshold Sensitivity](#sofa-threshold-sensitivity)
5. [eICU Regional Analysis](#eicu-regional-analysis)
6. [Volatility Indicators](#volatility-indicators)
7. [Figure Organization](#figure-organization)
8. [Output Files](#output-files)
9. [Reproducibility](#reproducibility)
10. [Requirements](#requirements)
11. [Citation](#citation)

---

## Key Findings

Across 809,017 ICU admissions from 4 datasets spanning the US, Europe, and Asia (2001--2022):

1. **Overall improvement masks clinically significant minority degradation.** In eICU, overall SOFA discrimination improves (+0.013, p < 0.001), yet 18--44 Male Black patients degrade by -0.157 (p < 0.001) -- a 0.482 AUROC spread between best and worst intersectional groups. A hospital recalibrating on the aggregate trend would worsen predictions for this subgroup.

2. **Between-group drift differences exceed minimum clinically significant thresholds.** Hispanic SOFA drift is significantly worse than White drift (delta = -0.051, FDR p < 0.001), exceeding the 0.05 AUROC minimum effect size. 85--91% of between-group comparisons are statistically and clinically significant across all datasets.

3. **Age groups drift in opposite directions** -- young patients degrade while elderly patients improve, a pattern replicated in US, European, and Asian healthcare systems.

4. **The same subgroup can improve in one system and decline in another.** No single recalibration factor generalizes across institutions or geographies.

### Summary Figure

![Summary Figure](figures/fig5_money_figure.png)

---

## Datasets

| Dataset | N | Period | Mortality | Scores | Race Data | Source |
|---------|---|--------|-----------|--------|-----------|--------|
| MIMIC Combined | 112,468 | 2001--2022 | 11.1% | SOFA, OASIS, SAPS-II, APS-III | Yes | US (Boston, single-center) |
| eICU Combined | 661,358 | 2014--2021 | 10.9% | SOFA, OASIS, SAPS-II, APS-III, APACHE | Yes | US (multi-center, 4 regions) |
| Saltz | 27,259 | 2013--2021 | 7.9% | SOFA, OASIS, SAPS-II, APS-III | No | Europe (Netherlands) |
| Zhejiang | 7,932 | 2011--2022 | 14.7% | SOFA, OASIS, SAPS-II, APS-III | No | Asia (China) |

**Total: 809,017 ICU admissions.** Each dataset is analyzed independently. SOFA is the primary metric. Intersectional groupings follow a clean hierarchy: gender-race, age-race, age-gender-race. Single-subgroup analyses (age-only, gender-only, race-only) appear in supplementary materials only.

---

## Statistical Methods

| Method | Purpose |
|--------|---------|
| **Bootstrap CIs** | Percentile-method 95% confidence intervals for AUROC, with stratified resampling (n = 100--1000) |
| **Bootstrap independence** | First half of replicates reserved for trend tests, second half for between-group comparisons -- eliminates reuse-driven significance inflation |
| **Page's L trend test** | Detects monotonic AUROC trends across all ordered time periods, not just endpoints |
| **Between-group comparison** | Mann-Whitney U on independent bootstrap delta distributions tests whether one subgroup's drift differs from another's |
| **Pooled FDR correction** | Benjamini-Hochberg applied once across all scores simultaneously, reflecting the unified claim that drift is non-uniform |
| **Minimum clinically significant effect size** | Between-group differences must exceed 0.05 AUROC to be labeled clinically significant, based on published minimally important differences for critical care prediction models |

A finding is reported as **clinically significant** only when it is both statistically significant (pooled FDR p < 0.05) and exceeds the minimum effect size threshold.

---

## SOFA Threshold Sensitivity

Multiple SOFA binarization thresholds (2, 6, 8, 10) are tested to confirm that drift patterns and fairness findings are robust to threshold choice. Per-threshold results are saved separately as `drift_deltas_sofa{T}.csv` in each dataset's output directory.

---

## eICU Regional Analysis

MIMIC Black patients improve over time while eICU Black patients degrade -- a divergence that demands explanation. Because MIMIC represents a single Boston hospital while eICU spans heterogeneous US practice patterns, the eICU regional breakdown (Midwest, Northeast, South, West) and teaching-status stratification test whether regional variation accounts for this discrepancy. Results are saved to `regional_breakdown.csv`.

---

## Volatility Indicators

Simple first-to-last AUROC deltas can obscure unstable trajectories. Three volatility metrics characterize drift dynamics:

- **Coefficient of variation (CV):** Normalized spread of AUROC across time periods
- **Max drawdown:** Largest peak-to-trough AUROC decline
- **Trend reversal count:** Number of direction changes in the AUROC trajectory

Results are saved to `volatility_indicators.csv`.

---

## Figure Organization

### Main Figures (6 maximum)

| Figure | Content |
|--------|---------|
| **1** | Study flow diagram and cohort characteristics across all 4 datasets |
| **2--3** | Cross-dataset SOFA drift trajectories and fairness metrics, with forest plots for between-group comparisons and intersectional breakdowns |
| **4--5** | Nursing care phenotypes (mouthcare and mechanical ventilation turning frequency) as proxies for unmeasured intersectional factors, with demographic cross-tabulation by care quartile |
| **6** | Multi-panel summary: age-group divergence, race disparities, comprehensive heatmap |

#### MIMIC Combined (2001--2022)

![MIMIC Combined — Subgroup Drift](figures/fig1_mimic_combined.png)
![MIMIC Combined — Intersectional](figures/fig1b_mimic_combined_intersectional.png)

#### eICU Combined (2014--2021)

![eICU Combined — Subgroup Drift](figures/fig2_eicu_combined.png)
![eICU Combined — Intersectional](figures/fig2b_eicu_combined_intersectional.png)

#### Saltz ICU (2013--2021, Europe)

![Saltz — Subgroup Drift](figures/fig3_saltz.png)

#### Zhejiang ICU (2011--2022, China)

![Zhejiang — Subgroup Drift](figures/fig4_zhejiang.png)

#### Calibration & Fairness

| MIMIC | eICU |
|-------|------|
| ![MIMIC Calibration](figures/fig7_mimic_combined_calibration.png) | ![eICU Calibration](figures/fig7_eicu_combined_calibration.png) |
| ![MIMIC Fairness](figures/fig8_mimic_combined_fairness.png) | ![eICU Fairness](figures/fig8_eicu_combined_fairness.png) |

#### VA CAN Style Drift

| MIMIC | eICU | Saltz | Zhejiang |
|-------|------|-------|----------|
| ![MIMIC VA CAN](figures/fig6b_mimic_combined_va_can_drift.png) | ![eICU VA CAN](figures/fig6b_eicu_combined_va_can_drift.png) | ![Saltz VA CAN](figures/fig6b_saltz_va_can_drift.png) | ![Zhejiang VA CAN](figures/fig6b_zhejiang_va_can_drift.png) |

### Supplementary Figures

<details>
<summary>Click to expand supplementary figures</summary>

#### Cross-Dataset Comparisons

![Overall Drift Comparison](figures/supplementary/figS3_overall_drift_comparison.png)
![Age Comparison](figures/supplementary/figS4_age_comparison.png)
![Race Comparison](figures/supplementary/figS5_race_comparison.png)
![Gender Comparison](figures/supplementary/figS7_gender_comparison.png)

#### Statistical Results

![Significance Forest Plot](figures/supplementary/figS6_significance_forest_plot.png)
![Between-Group Comparison](figures/supplementary/figS12_between_group_comparison.png)
![Drift Delta Summary](figures/supplementary/figS8_drift_delta_summary.png)

#### Detailed Analyses

![Score Comparison by Age](figures/supplementary/figS10_score_comparison_by_age.png)
![Temporal Trajectory](figures/supplementary/figS11_temporal_trajectory.png)
![Comprehensive Heatmap](figures/supplementary/figS9_comprehensive_heatmap.png)

#### Care Phenotypes (MIMIC only)

![Mouthcare](figures/supplementary/figS1_mimic_mouthcare.png)
![Mechanical Ventilation](figures/supplementary/figS2_mimic_mechvent.png)

</details>

---

## Output Files

Each dataset produces the following in `output/{dataset}/`:

| File | Description |
|------|-------------|
| `drift_results.csv` | Per-period AUROC with 95% CIs |
| `drift_deltas.csv` | Page's L trend test results (pooled FDR-corrected) |
| `between_group_comparisons.csv` | Between-group drift tests with effect sizes and CIs |
| `summary_by_score.csv` | Overall summary across all scores |
| `subgroup_drift.csv` | Subgroup-level drift results |
| `volatility_indicators.csv` | CV, max drawdown, trend reversal count |
| `care_demographics_correlation.csv` | Care quartile by demographic intersection cross-tabulation |
| `regional_breakdown.csv` | eICU regional and teaching-status stratification |
| `drift_deltas_sofa{T}.csv` | Per-threshold drift results (T = 2, 6, 8, 10) |

---

## Reproducibility

**Requirements:** Python 3.10+, [uv](https://github.com/astral-sh/uv) (`pip install uv`), dataset CSVs in `data/`

```bash
# Setup
uv venv && source .venv/bin/activate && uv pip install -r requirements.txt

# Run full pipeline
./run_all.sh --fast       # Fast testing (~1 min, 2 bootstrap iterations)
./run_all.sh              # Default (~15 min, 100 iterations)
./run_all.sh -b 1000      # Production (~2-4 hours, 1000 iterations)

# Individual steps
./run_all.sh --setup      # Only setup environment
./run_all.sh --analysis   # Only run analysis
./run_all.sh --figures    # Only generate figures
```

---

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`
- Dataset access: MIMIC (PhysioNet credentialed), eICU (PhysioNet credentialed), Saltz and Zhejiang (by arrangement with data owners)

---

## Citation

```bibtex
@software{data_drift_2025,
  title={Uniform Recalibration Is Unsafe: Subgroup-Specific Drift in ICU Severity Scores},
  author={Nabulsi, Hamza and Liu, Xiaoli and Celi, Leo Anthony and Cajas, Sebastian},
  year={2025}
}
```

## License

[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
