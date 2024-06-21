create temp table admissionlabs as
WITH pivoted AS (
    SELECT
          patientunitstayid,
          min(creatinine) AS min_creatinine_day1,
          max(creatinine) AS max_creatinine_day1,
          min(potassium) AS min_potassium,
          max(potassium) AS max_potassium,
          min(sodium) AS min_sodium,
          max(sodium) AS max_sodium,
          min(wbc) AS min_wbc,
          max(wbc) AS max_wbc,


    FROM `physionet-data.eicu_crd_derived.pivoted_lab` AS al
    WHERE al.chartoffset <= 360
    GROUP BY al.patientunitstayid
)
SELECT
      vital.*,
    -- from pivoted_score table
      min_creatinine_day1,
      max_creatinine_day1,
      min_potassium,
      max_potassium,
      min_sodium,
      max_sodium,
      min_wbc,
      max_wbc,

FROM vital
LEFT JOIN pivoted
    ON vital.patientunitstayid = pivoted.patientunitstayid
;
