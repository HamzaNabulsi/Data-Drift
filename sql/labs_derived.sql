create temp table labs_derived as
WITH pivoted as (
    SELECT
        patientunitstayid
        , min(ANIONGAP_min) AS ANIONGAP_min
        , max(ANIONGAP_max) AS ANIONGAP_max
        , min(ALBUMIN_min) AS ALBUMIN_min
        , max(ALBUMIN_max) AS ALBUMIN_max
        , min(BANDS_min) AS BANDS_min
        , max(BANDS_max) AS BANDS_max
        , min(BICARBONATE_min) AS BICARBONATE_min
        , max(BICARBONATE_max) AS BICARBONATE_max
        , min(HCO3_min) AS HCO3_min
        , max(HCO3_max) AS HCO3_max
        , min(BILIRUBIN_min) AS BILIRUBIN_min
        , max(BILIRUBIN_max) AS BILIRUBIN_max
        , min(CREATININE_min) AS CREATININE_min
        , max(CREATININE_max) AS CREATININE_max
        , min(CHLORIDE_min) AS CHLORIDE_min
        , max(CHLORIDE_max) AS CHLORIDE_max
        , min(GLUCOSE_min) AS GLUCOSE_min
        , max(GLUCOSE_max) AS GLUCOSE_max
        , min(HEMATOCRIT_min) AS HEMATOCRIT_min
        , max(HEMATOCRIT_max) AS HEMATOCRIT_max
        , min(HEMOGLOBIN_min) AS HEMOGLOBIN_min
        , max(HEMOGLOBIN_max) AS HEMOGLOBIN_max
        , min(LACTATE_min) AS LACTATE_min
        , max(LACTATE_max) AS LACTATE_max
        , min(PLATELET_min) AS PLATELET_min
        , max(PLATELET_max) AS PLATELET_max
        , min(POTASSIUM_min) AS POTASSIUM_min
        , max(POTASSIUM_max) AS POTASSIUM_max
        , min(PTT_min) AS PTT_min
        , max(PTT_max) AS PTT_max
        , min(INR_min) AS INR_min
        , max(INR_max) AS INR_max
        , min(PT_min) AS PT_min
        , max(PT_max) AS PT_max
        , min(SODIUM_min) AS SODIUM_min
        , max(SODIUM_max) AS SODIUM_max
        , min(BUN_min) AS BUN_min
        , max(BUN_max) AS BUN_max
        , min(WBC_min) AS WBC_min
        , max(WBC_max) AS WBC_max

    FROM `physionet-data.eicu_crd_derived.labsfirstday` AS labsfirstday
    GROUP BY labsfirstday.patientunitstayid

)

SELECT
    creatinine_labs.*
    , ANIONGAP_min
    , ANIONGAP_max
    , ALBUMIN_min
    , ALBUMIN_max
    , BANDS_min
    , BANDS_max
    , BICARBONATE_min
    , BICARBONATE_max
    , HCO3_min
    , HCO3_max
    , BILIRUBIN_min
    , BILIRUBIN_max
    , CREATININE_min
    , CREATININE_max
    , CHLORIDE_min
    , CHLORIDE_max
    , GLUCOSE_min
    , GLUCOSE_max
    , HEMATOCRIT_min
    , HEMATOCRIT_max
    , HEMOGLOBIN_min
    , HEMOGLOBIN_max
    , LACTATE_min
    , LACTATE_max
    , PLATELET_min
    , PLATELET_max
    , POTASSIUM_min
    , POTASSIUM_max
    , PTT_min
    , PTT_max
    , INR_min
    , INR_max
    , PT_min
    , PT_max
    , SODIUM_min
    , SODIUM_max
    , BUN_min
    , BUN_max
    , WBC_min
    , WBC_max
    , CASE
        WHEN ANIONGAP_min IS NULL
        AND ANIONGAP_max IS NULL THEN 1 ELSE 0
     END AS missing_aniongap
    , CASE
        WHEN ALBUMIN_min IS NULL
        AND ALBUMIN_max IS NULL THEN 1 ELSE 0
     END AS missing_albumin
    , CASE
        WHEN BANDS_min IS NULL
        AND BANDS_max IS NULL THEN 1 ELSE 0
     END AS missing_bands
    , CASE
        WHEN BICARBONATE_min IS NULL
        AND BICARBONATE_max IS NULL THEN 1 ELSE 0
     END AS missing_bicarbonate
    , CASE
        WHEN HCO3_min IS NULL
        AND HCO3_max IS NULL THEN 1 ELSE 0
     END AS missing_hco3
    , CASE
        WHEN BILIRUBIN_min IS NULL
        AND BILIRUBIN_max IS NULL THEN 1 ELSE 0
     END AS missing_bilirubin
    , CASE
        WHEN CREATININE_min IS NULL
        AND CREATININE_max IS NULL THEN 1 ELSE 0
     END AS missing_creatinine
    , CASE
        WHEN CHLORIDE_min IS NULL
        AND CHLORIDE_max IS NULL THEN 1 ELSE 0
     END AS missing_chloride
    , CASE
        WHEN GLUCOSE_min IS NULL
        AND GLUCOSE_max IS NULL THEN 1 ELSE 0
     END AS missing_glucose
    , CASE
        WHEN HEMATOCRIT_min IS NULL
        AND HEMATOCRIT_max IS NULL THEN 1 ELSE 0
     END AS missing_hematocrit
    , CASE
        WHEN HEMOGLOBIN_min IS NULL
        AND HEMOGLOBIN_max IS NULL THEN 1 ELSE 0
     END AS missing_hemoglobin
    , CASE
        WHEN LACTATE_min IS NULL
        AND LACTATE_max IS NULL THEN 1 ELSE 0
     END AS missing_lactate
    , CASE
        WHEN PLATELET_min IS NULL
        AND PLATELET_max IS NULL THEN 1 ELSE 0
     END AS missing_platelet
    , CASE
        WHEN POTASSIUM_min IS NULL
        AND POTASSIUM_max IS NULL THEN 1 ELSE 0
     END AS missing_potassium
    , CASE
        WHEN PTT_min IS NULL
        AND PTT_max IS NULL THEN 1 ELSE 0
     END AS missing_ptt
    , CASE
        WHEN INR_min IS NULL
        AND INR_max IS NULL THEN 1 ELSE 0
     END AS missing_inr
    , CASE
        WHEN PT_min IS NULL
        AND PT_max IS NULL THEN 1 ELSE 0
     END AS missing_pt
    , CASE
        WHEN SODIUM_min IS NULL
        AND SODIUM_max IS NULL THEN 1 ELSE 0
     END AS missing_sodium
    , CASE
        WHEN BUN_min IS NULL
        AND BUN_max IS NULL THEN 1 ELSE 0
     END AS missing_bun
    , CASE
        WHEN WBC_min IS NULL
        AND WBC_max IS NULL THEN 1 ELSE 0
     END AS missing_wbc


FROM admission_labs
RIGHT JOIN pivoted
    ON admission_labs.patientunitstayid = pivoted.patientunitstayid
;