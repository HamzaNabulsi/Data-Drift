create temp table apache_v as
WITH pivoted AS (
    SELECT
          patientunitstayid
        ,MAX(intubated) AS intubated
        ,MAX(vent) AS vent
        ,MAX(dialysis) AS dialysis
        ,MAX(eyes) AS eyes
        ,MAX(motor) AS motor
        ,MAX(verbal) AS verbal
        ,MAX(meds) AS meds
        ,MAX(wbc) AS wbc
        ,MAX(temperature) AS temperature
        ,MAX(respiratoryrate) AS respiratoryrate
        ,MAX(sodium) AS sodium
        ,MAX(meanbp) AS meanbp
        ,MAX(ph) AS ph
        ,MAX(hematocrit) AS hematocrit
        ,MAX(creatinine) AS creatinine
        ,MAX(albumin) AS albumin
        ,MAX(pao2) AS pao2
        ,MAX(pco2) AS pco2
        ,MAX(glucose) AS glucose
        ,MAX(bilirubin) AS bilirubin

    FROM `physionet-data.eicu_crd.apacheapsvar` AS apache_v
    GROUP BY apache_v.patientunitstayid
)
, vent_time AS
(
    SELECT
         patientunitstayid
        ,MIN(ventstartoffset) AS vent_start
        ,MAX(ventendoffset) AS vent_end
--         ,priorventstartoffset
--         ,priorventendoffset
--         ,MAX(CASE
--             WHEN MAX(ventendoffset) != 0 THEN 1
--             ELSE null
--         END) as days_on_vent
/*(ABS((MAX(ventendoffset)-MIN(ventstartoffset))) + ABS((MAX(priorventendoffset)-MIN(priorventstartoffset))))/1440
*/
    FROM `physionet-data.eicu_crd.respiratorycare`
    GROUP BY patientunitstayid
)
SELECT
      apache_pr.*
    -- from pivoted_score table
    ,intubated
    ,vent
    ,dialysis
    ,eyes
    ,motor
    ,verbal
    ,meds
    ,wbc
    ,temperature
    ,respiratoryrate
    ,sodium
    ,meanbp
    ,ph
    ,hematocrit
    ,creatinine
    ,albumin
    ,pao2
    ,pco2
    ,glucose
    ,bilirubin
    ,CASE
        WHEN vent_time.vent_start <= 0 AND vent = 1 THEN 1
        ELSE 0
    END AS vent_upon_admission
    ,CASE
        WHEN vent_time.vent_start > 0 AND vent = 1
        THEN 1
        ELSE 0
    END AS vent_during_hosp

FROM apache_pr
LEFT JOIN pivoted
    ON apache_pr.patientunitstayid = pivoted.patientunitstayid
LEFT JOIN vent_time vent_time
    ON apache_pr.patientunitstayid = vent_time.patientunitstayid
-- LEFT JOIN vent_length vent_length
--     ON apache_pr.patientunitstayid = vent_length.patientunitstayid
;
