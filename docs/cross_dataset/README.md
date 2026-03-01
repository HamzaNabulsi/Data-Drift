# Cross-Dataset Comparisons

> **Note:** These analyses compare patterns across multiple datasets. The main analysis focuses on per-dataset findings. See the [main README](../../README.md) for per-dataset results.

---

## Figure 5: Summary (Key Findings)

![Summary Figure](../../figures/fig5_money_figure.png)
*Figure 5: Multi-panel summary showing (A) Age group divergence, (B) Race disparities, (C) Comprehensive heatmap*

---

## Figure 6: Metrics Summary (Classification, Calibration, Fairness)

![3-Panel Summary](../../figures/fig9_xiaoli_3panel_summary.png)
*Figure 6: Three-panel summary of SOFA ≥ 10 threshold metrics across all datasets: (A) AUC drift over time, (B) SMR calibration drift, (C) Fairness metrics heatmap*

**Key Classification & Calibration Findings:**
- SOFA ≥ 10 threshold corresponds to ~40% mortality (JAMA 2001)
- TPR (Sensitivity) and PPV vary significantly across age groups and time
- SMR (Standardized Mortality Ratio) drift indicates calibration changes over time
- Brier score captures both discrimination and calibration performance

**Key Fairness Findings:**
- Demographic parity difference measures prediction rate disparities across groups
- Equalized odds difference captures TPR/FPR disparities across protected groups
- Cross-dataset patterns reveal consistent fairness concerns in certain subgroups

---

## Supplementary Cross-Dataset Figures (S3-S11)

![Overall Drift](../../figures/supplementary/figS3_overall_drift_comparison.png)
*Figure S3: Overall score performance trends (cross-dataset comparison)*

![Age Comparison](../../figures/supplementary/figS4_age_comparison.png)
*Figure S4: Age-stratified drift comparison across all datasets*

![Race Comparison](../../figures/supplementary/figS5_race_comparison.png)
*Figure S5: Race/ethnicity disparities comparison across US datasets*

![Significance Forest Plot](../../figures/supplementary/figS6_significance_forest_plot.png)
*Figure S6: Forest plot of statistically significant drift findings (FDR-corrected p < 0.05, Page's trend test) with confidence intervals*

![Gender Comparison](../../figures/supplementary/figS7_gender_comparison.png)
*Figure S7: Gender-specific drift patterns across datasets*

![Drift Delta Summary](../../figures/supplementary/figS8_drift_delta_summary.png)
*Figure S8: Summary of drift deltas by subgroup type (cross-dataset)*

![Comprehensive Heatmap](../../figures/supplementary/figS9_comprehensive_heatmap.png)
*Figure S9: Comprehensive drift heatmap showing all datasets, subgroups, and scores*

![Score Comparison by Age](../../figures/supplementary/figS10_score_comparison_by_age.png)
*Figure S10: Drift patterns by age group across all severity scores*

![Temporal Trajectories](../../figures/supplementary/figS11_temporal_trajectory.png)
*Figure S11: Full temporal trajectories showing how subgroups diverge over multiple time periods*

---

## COVID-19 Era Analysis (eICU Combined 2020-2021)

> **Note:** This analysis is specific to eICU Combined which includes data from the COVID-19 pandemic period (2020-2021).

**Racial/Ethnic Disparities during COVID-19 Era:**

| Subgroup | OASIS Δ | SOFA Δ | APS-III Δ | APACHE Δ |
|----------|---------|--------|-----------|----------|
| Hispanic | **-0.117*** | **-0.039*** | **-0.104*** | **-0.092*** |
| Black | **-0.069*** | -0.002 | **-0.125*** | **-0.030*** |
| Asian | -0.022 | **+0.101*** | **-0.104*** | -0.021 |
| White | **-0.027*** | **+0.012*** | **-0.091*** | **-0.045*** |

*The COVID-era eICU data (2020-2021) shows pervasive score degradation (87.4% of comparisons significant, FDR-corrected), with Hispanic and Black patients experiencing the largest declines in several scores.*

**Gender Differences during COVID-19 Era:**

| Region | Dataset | Male (OASIS) | Female (OASIS) | Pattern |
|--------|---------|--------------|----------------|---------|
| Europe | Saltz | +0.069 | +0.006 | Males improve 10x more |
| Asia | Zhejiang | +0.030 | +0.082 | Females improve 3x more |
| US | MIMIC Combined | +0.017 | -0.006 | Males improve, females decline |
| US | eICU Combined | **-0.038*** | **-0.034*** | Both decline significantly |
