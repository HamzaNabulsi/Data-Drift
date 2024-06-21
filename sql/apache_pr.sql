create temp table apache_pr as
WITH pivoted AS (
    SELECT
         patientunitstayid
        ,MAX(predictedicumortality) as predictedicumortality
        ,MAX(actualicumortality) as actualicumortality

    FROM `physionet-data.eicu_crd.apachepatientresult` AS apache_pr
    GROUP BY apache_pr.patientunitstayid
)
SELECT
      pastcancer.*
    -- from apachepatientresult table
    ,predictedicumortality
    ,actualicumortality


FROM pastcancer
LEFT JOIN pivoted
    ON pastcancer.patientunitstayid = pivoted.patientunitstayid
;
