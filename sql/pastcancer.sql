CREATE temp TABLE pastcancer AS
WITH pivoted AS(
 SELECT
          patientunitstayid
        ,STRING_AGG(pasthistorypath) AS past_cancer

    FROM `physionet-data.eicu_crd.pasthistory` as pastcancer
    WHERE pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Cancer-Primary Site/bile duct%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Cancer-Primary Site/bladder%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Cancer-Primary Site/bone%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Cancer-Primary Site/brain%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Cancer-Primary Site/breast%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Cancer-Primary Site/colon%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Cancer-Primary Site/esophagus%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Cancer-Primary Site/head and neck%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Cancer-Primary Site/kidney%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Cancer-Primary Site/liver%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Cancer-Primary Site/lung%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Cancer-Primary Site/melanoma%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Cancer-Primary Site/other%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Cancer-Primary Site/ovary%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Cancer-Primary Site/pancreas - adenocarcinoma%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Cancer-Primary Site/pancreas - islet cell%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Cancer-Primary Site/prostate%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Cancer-Primary Site/sarcoma%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Cancer-Primary Site/stomach%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Cancer-Primary Site/testes%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Cancer-Primary Site/unknown%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Cancer-Primary Site/uterus%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Hematologic Malignancy/ALL%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Hematologic Malignancy/AML%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Hematologic Malignancy/CLL%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Hematologic Malignancy/CML%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Hematologic Malignancy/Hodgkins disease%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Hematologic Malignancy/leukemia - other%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Hematologic Malignancy/multiple myeloma%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Hematologic Malignancy/non-Hodgkins lymphoma%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Hematologic Malignancy/other hematologic malignancy%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Metastases/bone%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Metastases/brain%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Metastases/carcinomatosis%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Metastases/intra-abdominal%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Metastases/liver%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Metastases/lung%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Metastases/nodes%'
    OR pasthistorypath LIKE '%%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Metastases/other'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer Therapy/Chemotherapy/Alkylating agents (bleomycin, cytoxan, cyclophos.)%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer Therapy/Chemotherapy/Anthracyclines (adriamycin, daunorubicin)%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer Therapy/Chemotherapy/chemotherapy within past 6 mos.%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer Therapy/Chemotherapy/chemotherapy within past mo.%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer Therapy/Chemotherapy/Cis-platinum%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer Therapy/Chemotherapy/Vincristine%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer Therapy/Radiation Therapy within past 6 months/bone%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer Therapy/Radiation Therapy within past 6 months/brain%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer Therapy/Radiation Therapy within past 6 months/liver%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer Therapy/Radiation Therapy within past 6 months/lung%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer Therapy/Radiation Therapy within past 6 months/nodes%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer Therapy/Radiation Therapy within past 6 months/other%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer Therapy/Radiation Therapy within past 6 months/primary site%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Myeloproliferative Disease/myelofibrosis%'
    OR pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Myeloproliferative Disease/polycythemia vera%'

    GROUP BY pastcancer.patientunitstayid

)

SELECT
      cancer_diagnosis.*
    ,past_cancer

FROM cancer_diagnosis
LEFT JOIN pivoted
    ON cancer_diagnosis.patientunitstayid = pivoted.patientunitstayid;


