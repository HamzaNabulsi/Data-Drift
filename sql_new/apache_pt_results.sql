CREATE temp TABLE apache_pt_results AS
WITH pivoted AS (
    SELECT
        patientunitstayid
        ,MAX(actualiculos) AS icu_los
        ,MAX(actualhospitallos) AS hosp_los
        ,MAX(actualventdays) as daysonvent
        , MAX(CASE
            WHEN REGEXP_CONTAINS(actualhospitalmortality, r'(?i)(EXPIRED)') THEN 1
            ELSE 0
          END) AS hosp_mortality
        ,MAX(CASE
            WHEN REGEXP_CONTAINS(actualicumortality, r'(?i)(EXPIRED)') THEN 1
            ELSE 0
            END) AS icu_mortality

    FROM `lcp-consortium.eicu_crd_ii_v0_1_0.apachepatientresults`
    GROUP BY patientunitstayid
)

SELECT
    apache_vars.*
    ,icu_los
    ,hosp_los
    ,daysonvent
    ,hosp_mortality
    ,icu_mortality

FROM apache_vars
LEFT JOIN pivoted
    ON apache_vars.patientunitstayid = pivoted.patientunitstayid
WHERE hosp_mortality IS NOT NULL
-- OR icu_mortality IS NOT NULL
;