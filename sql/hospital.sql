create temp table hospital as
WITH pivoted AS (
    SELECT
         hospitalid
        ,MAX(teachingstatus) as teachingstatus
        ,STRING_AGG(numbedscategory) as numbedscategory

    FROM `physionet-data.eicu_crd.hospital` AS hospital
    GROUP BY hospital.hospitalid
)
SELECT
      patient.*
    -- from apachepatientresult table
    ,teachingstatus
    ,numbedscategory


FROM patient
RIGHT JOIN pivoted
    ON patient.max_hid = pivoted.hospitalid
;
