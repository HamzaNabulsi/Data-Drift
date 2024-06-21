create temp table apache_data as
WITH pivoted AS (
    SELECT
        patientunitstayid
       ,MAX (apachescore) as apachescore_int
       ,MAX (acutephysiologyscore) asacutephysiologyscore
       ,MAX (actualhospitallos) asactualhospitallos
       ,MAX (actualiculos) as actualiculos

    FROM `physionet-data.eicu_crd.apachepatientresult` AS apache_data
    GROUP BY apache_data.patientunitstayid
)
SELECT
      apache_v.*
    -- from apachepatientresult table
    ,apachescore_int
    ,asacutephysiologyscore
    ,asactualhospitallos
    ,actualiculos

FROM apache_v
LEFT JOIN pivoted
    ON apache_v.patientunitstayid = pivoted.patientunitstayid
;
