# Care Phenotypes: A Novel Proxy for Intersectional Demographics

## Overview

Care phenotypes represent a novel approach to understanding how **nursing care intensity patterns** can serve as a proxy for intersectional demographic characteristics that are difficult to measure directly. This methodology allows us to study disparities in ICU score performance across patient populations defined by their care delivery patterns rather than traditional demographic categories alone.

## Motivation

Traditional demographic analysis examines drift across categories like age, gender, and race independently. However, real-world healthcare disparities often arise from **intersectional factors** that combine multiple dimensions:

- Socioeconomic status affecting access to care
- Insurance coverage influencing treatment decisions
- Geographic factors (urban vs. rural)
- Language barriers and health literacy
- Cultural factors in care-seeking behavior

These factors are often unmeasured or unavailable in clinical datasets. **Care phenotypes provide an indirect but measurable proxy** for these complex intersectional characteristics.

## The Care Phenotype Hypothesis

Patients who receive different frequencies of nursing interventions (e.g., mouthcare, turning/repositioning) may represent distinct populations with different:

1. **Acuity levels** - Sicker patients may receive more frequent care
2. **Staffing patterns** - Care frequency reflects nurse-to-patient ratios
3. **Care quality** - Higher frequency may indicate better adherence to protocols
4. **Patient characteristics** - Certain patients may receive differential attention

By stratifying patients into **care frequency quartiles**, we can examine whether ICU severity scores (like SOFA) drift differently across these phenotypically-defined groups.

## Methodology

### Care Frequency Metrics

We analyze two care intervention types from the MIMIC-IV dataset:

| Metric | Description | Clinical Relevance |
|--------|-------------|-------------------|
| **Mouthcare Interval** | Time between oral care interventions | Ventilator-associated pneumonia prevention |
| **Turning Interval** | Time between repositioning events | Pressure ulcer prevention |

### Quartile Stratification

Patients are stratified into four groups based on care frequency:

- **Q1 (High Frequency)**: Most frequent care interventions
- **Q2**: Above-average frequency
- **Q3**: Below-average frequency
- **Q4 (Low Frequency)**: Least frequent care interventions

### Analysis Framework

For each care phenotype quartile, we compute:

1. **SOFA AUC** for mortality prediction at each time period
2. **Temporal drift** (change in AUC from first to last period)
3. **95% Bootstrap confidence intervals**
4. **Comparison across quartiles** to identify disparities

## Key Findings

### Care Frequency and Score Performance

Our analysis reveals that care frequency quartiles show **differential drift patterns**:

| Quartile | Typical Finding | Interpretation |
|----------|-----------------|----------------|
| Q1 (High) | Moderate drift | High-attention patients - scores may be better calibrated |
| Q2-Q3 | Variable | Intermediate groups show mixed patterns |
| Q4 (Low) | Often larger drift | Low-attention patients may be systematically different |

### Intersectional Insights

Care phenotypes capture information that traditional demographics miss:

- Patients in Q4 (low care frequency) may represent a **disadvantaged phenotype** that experiences worse score calibration
- This could reflect unmeasured confounders like socioeconomic factors, language barriers, or systemic biases in care delivery
- The care phenotype approach allows us to study these disparities without requiring explicit measurement of sensitive variables

## Clinical Implications

1. **Quality Improvement**: Care frequency metrics could flag patients at risk for score miscalibration
2. **Equity Monitoring**: Track care frequency disparities as a proxy for equitable care delivery
3. **Score Validation**: Consider care phenotypes when validating ICU scores for new populations
4. **Resource Allocation**: Understand how staffing patterns affect score reliability

## Relationship to Prior Work

This approach builds on foundational work in nursing care intensity and health disparities:

### Key References

1. **Ghassemi M, et al.** "State of the Art Review: The Data Revolution in Critical Care." *Critical Care* (2015).
   - Foundational work on using EHR data to understand ICU care patterns

2. **Sayed L, et al.** "Characterizing and Visualizing Nursing Care Variability." *AMIA Annual Symposium* (2019).
   - Methodology for quantifying nursing care patterns from EHR data

3. **Celi LA, et al.** "The MIMIC Code Repository: enabling reproducibility in critical care research." *JAMIA* (2018).
   - Infrastructure enabling care frequency analysis

### Novel Contribution

Our work extends this literature by:

- Using care frequency as a **stratification variable** for score drift analysis
- Demonstrating that care phenotypes reveal disparities **invisible to traditional demographic analysis**
- Proposing care phenotypes as a **practical proxy for intersectional demographics**

## Data Sources

### MIMIC-IV Care Cohorts

| Cohort | N | Period | Care Metric |
|--------|---|--------|-------------|
| Mouthcare | 8,675 | 2008-2019 | Oral care interval frequency |
| Mechanical Ventilation | 8,919 | 2008-2019 | Turning/repositioning interval |

### Variables Used

- **Outcome**: Hospital mortality
- **Score**: SOFA (Sequential Organ Failure Assessment)
- **Time**: Admission year (4 bins: 2008-2010, 2011-2013, 2014-2016, 2017-2019)
- **Care Metric**: Intervention interval frequency (continuous → quartiles)

## Reproducibility

Analysis code: `code/supplementary_analysis.py`

```bash
# Run care phenotype analysis
python code/supplementary_analysis.py

# Fast mode (for testing)
python code/supplementary_analysis.py --fast

# Production mode (1000 bootstrap iterations)
python code/supplementary_analysis.py --bootstrap 1000
```

Output files:
- `output/mimic_sofa_results.csv` - Full results by subgroup including care quartiles
- `output/mimic_sofa_deltas.csv` - Drift deltas for all subgroups
- `figures/supplementary/figS1_mimic_mouthcare.png` - Mouthcare cohort visualization
- `figures/supplementary/figS2_mimic_mechvent.png` - Mechanical ventilation cohort visualization

## Future Directions

1. **Expand care phenotypes**: Include additional nursing interventions (vital sign frequency, medication timing)
2. **Multi-site validation**: Test care phenotype approach across different healthcare systems
3. **Causal analysis**: Investigate whether care frequency *causes* or *reflects* score miscalibration
4. **Integration with demographics**: Combine care phenotypes with traditional demographics for richer intersectional analysis

## Citation

If you use the care phenotype methodology in your research, please cite:

```bibtex
@article{datadrift2025,
  title={Subgroup-Specific Drift in ICU Severity Scores: A Multi-Dataset Analysis},
  author={Nabulsi, Hamza and Liu, Xiaoli and Celi, Leo Anthony and Cajas, Sebastian},
  year={2025}
}
```
