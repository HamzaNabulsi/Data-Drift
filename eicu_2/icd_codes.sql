CREATE temp TABLE icd_codes AS
WITH pivoted AS(
    SELECT
        patientunitstayid,
        STRING_AGG(icd9code) as icd9code
    FROM `lcp-consortium.eicu_crd_ii_v0_1_0..diagnosis`
    GROUP BY patientunitstayid
)

SELECT
    sepsis.*,
    icd9code

FROM sepsis
LEFT JOIN pivoted
    ON sepsis.patientunitstayid = pivoted.patientunitstayid
;
