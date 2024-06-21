create temp table labs as
WITH labs AS
(
    SELECT
          d.*
    FROM demographics d
    LEFT JOIN `physionet-data.eicu_crd_derived.labsfirstday` l
        ON d.patientunitstayid = l.patientunitstayid
)
, vaso_first_day AS
(
    SELECT
          patientunitstayid
        , MAX(vasopressor) AS vasopressor
    FROM `physionet-data.eicu_crd_derived.pivoted_treatment_vasopressor`
    WHERE chartoffset <= 1440
    GROUP BY patientunitstayid
)
, operative AS
(
    SELECT
          patientunitstayid
        , MAX(CASE
            WHEN admitdxpath LIKE '%Operative%' THEN 1 ELSE 0
          END) AS surgical
    FROM `physionet-data.eicu_crd.admissiondx`
    GROUP BY patientunitstayid
)
SELECT
      l.*
    , CASE
        WHEN v.vasopressor > 0 THEN 1 ELSE 0
      END AS vasopressor
    , o.surgical
FROM labs l
LEFT JOIN vaso_first_day v
    ON l.patientunitstayid = v.patientunitstayid
LEFT JOIN operative o
    ON l.patientunitstayid = o.patientunitstayid
;