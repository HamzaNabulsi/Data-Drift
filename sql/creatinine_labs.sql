create temp table creatinine_labs as
WITH creatinine_day3 AS (
    SELECT
          patientunitstayid
          ,MAX(labresult) as max_creatinine_day_3
          ,MIN(labresult) as min_creatinine_day_3
    FROM `physionet-data.eicu_crd.lab` AS creatinine_labs
    WHERE labname LIKE '%creatinine%'
    AND labresultoffset BETWEEN 4320 AND 5760
    GROUP BY creatinine_labs.patientunitstayid
)
,creatinine_day5 AS (
    SELECT
          patientunitstayid
          ,MAX(labresult) as max_creatinine_day_5
          ,MIN(labresult) as min_creatinine_day_5
    FROM `physionet-data.eicu_crd.lab` AS creatinine_labs
    WHERE labname LIKE '%creatinine%'
    AND labresultoffset BETWEEN 7200 AND 8640
    GROUP BY creatinine_labs.patientunitstayid
)
SELECT
      admissionlabs.*,
    -- from pivoted_score table
      max_creatinine_day_3,
      min_creatinine_day_3,
      max_creatinine_day_5,
      min_creatinine_day_5,

FROM admissionlabs
LEFT JOIN creatinine_day3
    ON admissionlabs.patientunitstayid = creatinine_day3.patientunitstayid
LEFT JOIN creatinine_day5
    ON admissionlabs.patientunitstayid = creatinine_day5.patientunitstayid
;



