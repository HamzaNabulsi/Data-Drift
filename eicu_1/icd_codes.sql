CREATE temp TABLE icd_codes AS
WITH pivoted AS(
    SELECT
        patientunitstayid,
        STRING_AGG(icd9code) as icd9code
    FROM `physionet-data.eicu_crd.diagnosis`
    GROUP BY patientunitstayid
)

SELECT
    sepsis.*,
    icd9code

FROM sepsis
LEFT JOIN pivoted
    ON sepsis.patientunitstayid = pivoted.patientunitstayid
;
