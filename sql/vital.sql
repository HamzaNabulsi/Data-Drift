create temp table vital as
WITH pivoted AS (
    SELECT
    patientunitstayid,
    min(heartrate) AS min_heartrate, max(heartrate) AS max_heartrate,
    min(respiratoryrate) AS min_respiratoryrate, max(respiratoryrate) AS max_respiratoryrate,
    min(temperature) AS min_temperature, max(temperature) AS max_temperature,
    min(spo2) AS min_spo2, max(spo2) AS max_spo2,

    FROM `physionet-data.eicu_crd_derived.pivoted_vital` AS vital
    WHERE vital.chartoffset <= 1440
    GROUP BY
    vital.patientunitstayid
)

SELECT
    apache_data.*,
-- from pivoted_vital table
    min_heartrate,
    max_heartrate,
    min_respiratoryrate,
    max_respiratoryrate,
    min_temperature,
    max_temperature,
    min_spo2,
    max_spo2

FROM apache_data
LEFT JOIN pivoted
    ON apache_data.patientunitstayid = pivoted.patientunitstayid;

