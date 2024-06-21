create temp table patient as
WITH pivoted AS (
    SELECT
          patientunitstayid,
          MAX(hospitalid) as max_hid,
          STRING_AGG(ethnicity) as ethnicity,

    FROM `physionet-data.eicu_crd.patient` AS patient
    GROUP BY patient.patientunitstayid
)
SELECT
      creatinine_labs.*,
    -- from pivoted_score table
      max_hid,

FROM creatinine_labs
LEFT JOIN pivoted
    ON creatinine_labs.patientunitstayid = pivoted.patientunitstayid
;
