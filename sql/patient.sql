create temp table patient as
WITH pivoted AS (
    SELECT
          patientunitstayid,
          MAX(hospitalid) as max_hid,
          STRING_AGG(ethnicity) as ethnicity_agg,
          MAX(hospitalDischargeYear) as discharge_year

    FROM `physionet-data.eicu_crd.patient` AS patient
    GROUP BY patient.patientunitstayid
)
SELECT
      admissionlabs.*
    -- from pivoted_score table
      ,max_hid
      ,CASE
        WHEN discharge_year BETWEEN 2010 AND 2014 THEN "2010-2014"
        WHEN discharge_year BETWEEN 2015 AND 2019 THEN "2015-2019"
        ELSE NULL
      END AS year_group
      ,CASE
         WHEN ethnicity_agg LIKE "Caucasian" THEN 1
         ELSE 0
      END AS Caucasian
      ,CASE
        WHEN ethnicity_agg LIKE "African American" THEN 1
        ELSE 0
      END AS African_American
      ,CASE
        WHEN ethnicity_agg LIKE "Hispanic" THEN 1
        ELSE 0
      END AS Hispanic
      ,CASE
        WHEN ethnicity_agg LIKE "Asian" THEN 1
        ELSE 0
      END AS Asian
      ,CASE
        WHEN ethnicity_agg LIKE "Native American" THEN 1
        ELSE 0
      END AS Native_American
      ,CASE
        WHEN ethnicity_agg LIKE "Other/Unknown" THEN 1
        WHEN ethnicity_agg LIKE "" THEN 1
        ELSE 0
      END AS Other_Unknown_Missing

FROM admissionlabs
LEFT JOIN pivoted
    ON admissionlabs.patientunitstayid = pivoted.patientunitstayid
;
