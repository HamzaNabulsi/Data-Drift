create temp table demographics as
WITH dx_cancer AS
(
    SELECT
          patientunitstayid
        , MAX(CASE
            WHEN REGEXP_CONTAINS(diagnosisstring, r'(?i)(hemato)|(leukemia)|(lymphoma)|(myeloma)') THEN 1
            ELSE 0
          END) AS hemato
        , MAX(CASE
            WHEN REGEXP_CONTAINS(diagnosisstring, r'(?i)(GU)|(renal)|(kidney)|(genetal)') THEN 1
            ELSE 0
          END) AS renal_GU
        , MAX(CASE
            WHEN REGEXP_CONTAINS(diagnosisstring, r'(?i)(thymoma)|(breast)|(mediastinal)') THEN 1
            ELSE 0
          END) AS chest_breast
        , MAX(CASE
            WHEN REGEXP_CONTAINS(diagnosisstring, r'(?i)(pulmonary)|(lung)') THEN 1
            ELSE 0
          END) AS lung_pulmonary
        , MAX(CASE
            WHEN REGEXP_CONTAINS(diagnosisstring, r'(?i)(skin)|(muscle)|(skeletal)') THEN 1
            ELSE 0
          END) AS mss_skin
        , MAX(CASE
            WHEN REGEXP_CONTAINS(diagnosisstring, r'(?i)(head)|(neck)') THEN 1
            ELSE 0
          END) AS head_neck
        , MAX(CASE
            WHEN REGEXP_CONTAINS(diagnosisstring, r'(?i)(GI)|(colon)|(intestinal)|(gastric)|(liver)|(pancreatic)|(gallbladder)|(esophageal)|(anal)') THEN 1
            ELSE 0
          END) AS GI
        , MAX(CASE
            WHEN REGEXP_CONTAINS(diagnosisstring, r'(?i)(brain)|(spine)|(spinal)|(CNS)') THEN 1
            ELSE 0
          END) AS CNS
        , MAX(CASE
            WHEN REGEXP_CONTAINS(diagnosisstring, r'(?i)(endocrine)') THEN 1
            ELSE 0
          END) AS endocrine
    FROM `physionet-data.eicu_crd.diagnosis`
    WHERE REGEXP_CONTAINS(diagnosisstring, r'(?i)((^|[|])onc)|(renal|electrolyte imbalance|hyponatremia|due to elevated ADH levels|from tumor other than lung)'||
                                            'renal|electrolyte imbalance|hypocalcemia|due to tumor lysis'||
                                             'renal|electrolyte imbalance|hyperkalemia|due to tumor lysis'||
                                            'renal|electrolyte imbalance|hypercalcemia|due to malignancy' ||
                                            'renal|abnormality of urine quantity or quality|diabetes insipidus|central|from primary or metastatic tumor' ||
                                            'pulmonary|disorders of vasculature|pulmonary hemorrhage|due to neoplasm' ||
                                            'pulmonary|disorders of lung parenchyma|malignancy|primary lung cancer|squamous cell CA' ||
                                            'pulmonary|disorders of lung parenchyma|malignancy|primary lung cancer|small cell CA' ||
                                            'pulmonary|disorders of lung parenchyma|malignancy|primary lung cancer|poorly-differentiated' ||
                                            'pulmonary|disorders of lung parenchyma|malignancy|primary lung cancer|mesothelioma' ||
                                            'pulmonary|disorders of lung parenchyma|malignancy|primary lung cancer|large cell CA' ||
                                            'pulmonary|disorders of lung parenchyma|malignancy|primary lung cancer|bronchoalveolar cell CA' ||
                                            'pulmonary|disorders of lung parenchyma|malignancy|primary lung cancer|biopsy pending' ||
                                            'pulmonary|disorders of lung parenchyma|malignancy|primary lung cancer|adenocarcinoma' ||
                                            'pulmonary|disorders of lung parenchyma|malignancy|primary lung cancer' ||
                                            'pulmonary|disorders of lung parenchyma|malignancy|lymphoma' ||
                                            'pulmonary|disorders of lung parenchyma|malignancy|cancer with metastases to the lung' ||
                                            'pulmonary|disorders of lung parenchyma|malignancy'||
                                            'pulmonary|disorders of lung parenchyma|interstitial lung disease|hematological malignancy' ||
                                            'oncology|skin, muscle and skeletal tumors|squamous cell CA' ||
                                            'oncology|skin, muscle and skeletal tumors|soft tissue sarcoma|Kaposi' ||
                                            'oncology|skin, muscle and skeletal tumors|soft tissue sarcoma' ||
                                            'oncology|skin, muscle and skeletal tumors|melanoma|superficial spreading' ||
                                            'oncology|skin, muscle and skeletal tumors|melanoma|nodular' ||
                                            'oncology|skin, muscle and skeletal tumors|melanoma|lentigo maligna melanoma' ||
                                            'oncology|skin, muscle and skeletal tumors|melanoma|acral lentiginous melanoma' ||
                                            'oncology|skin, muscle and skeletal tumors|leiomyosarcoma' ||
                                            'oncology|skin, muscle and skeletal tumors|bone tumors|osteosarcoma' ||
                                            'oncology|skin, muscle and skeletal tumors|bone tumors|ewings sarcoma' ||
                                            'oncology|skin, muscle and skeletal tumors|bone tumors|chondrosarcoma' ||
                                            'oncology|skin, muscle and skeletal tumors|bone tumors|bony metastasis' ||
                                            'oncology|skin, muscle and skeletal tumors|bone tumors' ||
                                            'oncology|hematologic malignancy|multiple myeloma' ||
                                            'oncology|hematologic malignancy|lymphoproliferative disease|non-Hodgkins lymphoma' ||
                                            'oncology|hematologic malignancy|lymphoproliferative disease|Hodgkins disease|nodular sclerosis' ||
                                            'oncology|hematologic malignancy|lymphoproliferative disease|Hodgkins disease|mixed cellularity' ||
                                            'oncology|hematologic malignancy|lymphoproliferative disease|Hodgkins disease|lymphocyte predominance' ||
                                            'oncology|hematologic malignancy|lymphoproliferative disease|Hodgkins disease|lymphocyte depleted' ||
                                            'oncology|hematologic malignancy|lymphoproliferative disease|Hodgkins disease' ||
                                            'oncology|hematologic malignancy|lymphoproliferative disease' ||
                                            'oncology|hematologic malignancy|leukemia|unspecified' ||
                                            'oncology|hematologic malignancy|leukemia|chronic myelogenous|without Philadelphia chromosome' ||
                                            'oncology|hematologic malignancy|leukemia|chronic myelogenous|with Philadelphia chromosome' ||
                                            'oncology|hematologic malignancy|leukemia|chronic myelogenous' ||
                                            'oncology|hematologic malignancy|leukemia|chronic lymphocytic' ||
                                            'oncology|hematologic malignancy|leukemia|acute myelogenous|M7' ||
                                            'oncology|hematologic malignancy|leukemia|acute myelogenous|M6' ||
                                            'oncology|hematologic malignancy|leukemia|acute myelogenous|M5' ||
                                            'oncology|hematologic malignancy|leukemia|acute myelogenous|M4' ||
                                            'oncology|hematologic malignancy|leukemia|acute myelogenous|M3' ||
                                            'oncology|hematologic malignancy|leukemia|acute myelogenous|M2' ||
                                            'oncology|hematologic malignancy|leukemia|acute myelogenous|M1' ||
                                            'oncology|hematologic malignancy|leukemia|acute myelogenous|M0' ||
                                            'oncology|hematologic malignancy|leukemia|acute myelogenous' ||
                                            'oncology|hematologic malignancy|leukemia|acute lymphocytic|L3' ||
                                            'oncology|hematologic malignancy|leukemia|acute lymphocytic|L2' ||
                                            'oncology|hematologic malignancy|leukemia|acute lymphocytic|L1' ||
                                            'oncology|hematologic malignancy|leukemia|acute lymphocytic' ||
                                            'oncology|hematologic malignancy|leukemia' ||
                                            'oncology|head and neck tumors|nose tumor|squamous cell CA' ||
                                            'oncology|head and neck tumors|nose tumor|nasopharyngeal CA' ||
                                            'oncology|head and neck tumors|nose tumor|craniopharyngioma' ||
                                            'oncology|head and neck tumors|neck tumor|thyroid tumor|papillary CA' ||
                                            'oncology|head and neck tumors|neck tumor|thyroid tumor|medullary CA' ||
                                            'oncology|head and neck tumors|neck tumor|thyroid tumor|follicular CA' ||
                                            'oncology|head and neck tumors|neck tumor|thyroid tumor|anaplastic CA' ||
                                            'oncology|head and neck tumors|neck tumor|thyroid tumor' ||
                                            'oncology|head and neck tumors|neck tumor|other ill-defined sites' ||
                                            'oncology|head and neck tumors|neck tumor|oro/hypopharyngeal CA' ||
                                            'oncology|head and neck tumors|neck tumor|laryngeal CA' ||
                                            'oncology|head and neck tumors|neck tumor' ||
                                            'oncology|head and neck tumors|mouth and jaw tumor|oral cavity squamous cell CA' ||
                                            'oncology|head and neck tumors|mouth and jaw tumor|lip squamous cell CA' ||
                                            'oncology|head and neck tumors|mouth and jaw tumor' ||
                                            'oncology|head and neck tumors|eye tumor|right' ||
                                            'oncology|head and neck tumors|eye tumor|retinoblastoma' ||
                                            'oncology|head and neck tumors|eye tumor|melanoma' ||
                                            'oncology|head and neck tumors|eye tumor|left' ||
                                            'oncology|head and neck tumors|eye tumor' ||
                                            'oncology|GU tumors|uterine CA|uterine body' ||
                                            'oncology|GU tumors|uterine CA|choriocarcinoma' ||
                                            'oncology|GU tumors|uterine CA|adnexa' ||
                                            'oncology|GU tumors|uterine CA|adenocarcinoma' ||
                                            'oncology|GU tumors|uterine CA' ||
                                            'oncology|GU tumors|testicular CA|seminoma' ||
                                            'oncology|GU tumors|testicular CA|nonseminoma' ||
                                            'oncology|GU tumors|testicular CA' ||
                                            'oncology|GU tumors|renal cell CA|right kidney' ||
                                            'oncology|GU tumors|renal cell CA|left kidney' ||
                                            'oncology|GU tumors|renal cell CA' ||
                                            'oncology|GU tumors|prostate CA' ||
                                            'oncology|GU tumors|ovarian CA|right ovary' ||
                                            'oncology|GU tumors|ovarian CA|left ovary' ||
                                            'oncology|GU tumors|ovarian CA' ||
                                            'oncology|GU tumors|male genetalia CA' ||
                                            'oncology|GU tumors|female genetalia CA' ||
                                            'oncology|GU tumors|cervical CA' ||
                                            'oncology|GU tumors|bladder CA' ||
                                            'oncology|GI tumors|retroperitoneum/peritoneum CA' ||
                                            'oncology|GI tumors|pancreatic tumor|pancreatic CA' ||
                                            'oncology|GI tumors|pancreatic tumor|islet cell tumor' ||
                                            'oncology|GI tumors|pancreatic tumor' ||
                                            'oncology|GI tumors|liver CA|hepatocellular CA' ||
                                            'oncology|GI tumors|liver CA|CA metastatic to liver' ||
                                            'oncology|GI tumors|liver CA' ||
                                            'oncology|GI tumors|intra-abdominal sarcoma' ||
                                            'oncology|GI tumors|intestinal CA|lymphoma' ||
                                            'oncology|GI tumors|intestinal CA|carcinoid tumor' ||
                                            'oncology|GI tumors|intestinal CA|adenocarcinoma' ||
                                            'oncology|GI tumors|intestinal CA' ||
                                            'oncology|GI tumors|GI/peritoneum CA' ||
                                            'oncology|GI tumors|gastric CA|gastric lymphoma' ||
                                            'oncology|GI tumors|gastric CA|gastric adenocarcinoma' ||
                                            'oncology|GI tumors|gastric CA' ||
                                            'oncology|GI tumors|gallbladder CA' ||
                                            'oncology|GI tumors|esophageal CA'||
                                            'oncology|GI tumors|colon CA|transverse colon' ||
                                            'oncology|GI tumors|colon CA|sigmoid' ||
                                            'oncology|GI tumors|colon CA|right hemicolon' ||
                                            'oncology|GI tumors|colon CA|left hemicolon' ||
                                            'oncology|GI tumors|colon CA|cecum' ||
                                            'oncology|GI tumors|colon CA' ||
                                            'oncology|GI tumors|cholangiocarcinoma' ||
                                            'oncology|GI tumors|anal CA' ||
                                            'oncology|GI tumors|abdominal carcinomatosis' ||
                                            'oncology|CNS tumors|spinal cord tumors' ||
                                            'oncology|CNS tumors|brain tumor|oligodendroglioma' ||
                                            'oncology|CNS tumors|brain tumor|metastatic brain tumor' ||
                                            'oncology|CNS tumors|brain tumor|metastatic brain tumor' ||
                                            'oncology|CNS tumors|brain tumor|glioblastoma multiforme' ||
                                            'oncology|CNS tumors|brain tumor|ependymoma' ||
                                            'oncology|CNS tumors|brain tumor|CNS lymphoma' ||
                                            'oncology|CNS tumors|brain tumor|brain tumor - type unknown' ||
                                            'oncology|CNS tumors|brain tumor|astrocytoma, low-grade' ||
                                            'oncology|CNS tumors|brain tumor|astrocytoma, high-grade' ||
                                            'oncology|CNS tumors|brain tumor' ||
                                            'oncology|chest tumors|primary lung cancer|squamous cell CA' ||
                                            'oncology|chest tumors|primary lung cancer|small cell CA' ||
                                            'oncology|chest tumors|primary lung cancer|poorly-differentiated' ||
                                            'oncology|chest tumors|primary lung cancer|mesothelioma' ||
                                            'oncology|chest tumors|primary lung cancer|carcinoid' ||
                                            'oncology|chest tumors|primary lung cancer|bronchoalveolar cell CA' ||
                                            'oncology|chest tumors|primary lung cancer|biopsy pending' ||
                                            'oncology|chest tumors|primary lung cancer|adenocarcinoma' ||
                                            'oncology|chest tumors|primary lung cancer' ||
                                            'oncology|chest tumors|pleura' ||
                                            'oncology|chest tumors|metastatic lung CA' ||
                                            'oncology|chest tumors|mediastinal tumor|thymoma|malignant' ||
                                            'oncology|chest tumors|mediastinal tumor|thymoma' ||
                                            'oncology|chest tumors|mediastinal tumor|teratoma' ||
                                            'oncology|chest tumors|mediastinal tumor|lymphoma' ||
                                            'oncology|chest tumors|mediastinal tumor' ||
                                            'oncology|chest tumors|breast CA|male' ||
                                            'oncology|chest tumors|breast CA|female' ||
                                            'oncology|chest tumors|breast CA' ||
                                            'neurologic|disorders of vasculature|stroke|hemorrhagic stroke|into pre-existent CNS mass' ||
                                            'neurologic|disorders of the spinal cord and peripheral nervous system|spinal cord compression|secondary to metastasis' ||
                                            'neurologic|CNS mass lesions|brain tumor|with carcinomatous meningitis' ||
                                            'neurologic|CNS mass lesions|brain tumor|oligodendroglioma'||
                                            'neurologic|CNS mass lesions|brain tumor|metastatic brain tumor' ||
                                            'neurologic|CNS mass lesions|brain tumor|glioblastoma multiforme' ||
                                            'neurologic|CNS mass lesions|brain tumor|CNS lymphoma'||
                                            'neurologic|CNS mass lesions|brain tumor|astrocytoma, low-grade' ||
                                            'neurologic|CNS mass lesions|brain tumor|astrocytoma, high-grade' ||
                                            'hematology|white blood cell disorders|neutropenia|from chemotherapy' ||
                                            'hematology|white blood cell disorders|eosinophilia|from cancer' ||
                                            'hematology|platelet disorders|thrombocytopenia|primary hematologic malignancy'||
                                            'hematology|platelet disorders|thrombocytopenia|due to non-hematologic malignancy marrow metastas' ||
                                            'hematology|oncology and leukemia|tumor lysis syndrome' ||
                                            'hematology|oncology and leukemia|plasma cell disorders|multiple myeloma' ||
                                            'hematology|oncology and leukemia|plasma cell disorders|Bence-Jones proteinuria'||
                                            'hematology|oncology and leukemia|plasma cell disorders' ||
                                            'hematology|oncology and leukemia|myeloproliferative disorder|myelofibrosis' ||
                                            'hematology|oncology and leukemia|myeloproliferative disorder|essential thrombocythemia (clinical)' ||
                                            'hematology|oncology and leukemia|myeloproliferative disorder|chronic myeloproliferative disorder (clinical)' ||
                                            'hematology|oncology and leukemia|myeloproliferative disorder' ||
                                            'hematology|oncology and leukemia|myelodysplastic syndrome|refractory anemia with ringed sideroblasts' ||
                                            'hematology|oncology and leukemia|myelodysplastic syndrome|refractory anemia with excess blasts in transformation' ||
                                            'hematology|oncology and leukemia|myelodysplastic syndrome|refractory anemia with excess blasts' ||
                                            'hematology|oncology and leukemia|myelodysplastic syndrome|refractory anemia' ||
                                            'hematology|oncology and leukemia|myelodysplastic syndrome|chronic myelomonocytic leukemia' ||
                                            'hematology|oncology and leukemia|myelodysplastic syndrome' ||
                                            'hematology|oncology and leukemia|lymphoproliferative disease|non-Hodgkins lymphoma' ||
                                            'hematology|oncology and leukemia|lymphoproliferative disease|Hodgkins disease|nodular sclerosis' ||
                                            'hematology|oncology and leukemia|lymphoproliferative disease|Hodgkins disease|mixed cellularity' ||
                                            'hematology|oncology and leukemia|lymphoproliferative disease|Hodgkins disease|lymphocyte predominance' ||
                                            'hematology|oncology and leukemia|lymphoproliferative disease|Hodgkins disease|lymphocyte depleted' ||
                                            'hematology|oncology and leukemia|lymphoproliferative disease|Hodgkins disease' ||
                                            'hematology|oncology and leukemia|lymphoproliferative disease'||
                                            'hematology|oncology and leukemia|leukemia|chronic myelogenous|without Philadelphia chromosome' ||
                                            'hematology|oncology and leukemia|leukemia|chronic myelogenous|with Philadelphia chromosome' ||
                                            'hematology|oncology and leukemia|leukemia|chronic myelogenous' ||
                                            'hematology|oncology and leukemia|leukemia|chronic lymphocytic' ||
                                            'hematology|oncology and leukemia|leukemia|acute myelogenous|M6' ||
                                            'hematology|oncology and leukemia|leukemia|acute myelogenous|M5' ||
                                            'hematology|oncology and leukemia|leukemia|acute myelogenous|M4' ||
                                            'hematology|oncology and leukemia|leukemia|acute myelogenous|M3' ||
                                            'hematology|oncology and leukemia|leukemia|acute myelogenous|M2' ||
                                            'hematology|oncology and leukemia|leukemia|acute myelogenous|M1' ||
                                            'hematology|oncology and leukemia|leukemia|acute myelogenous|M0'||
                                            'hematology|oncology and leukemia|leukemia|acute myelogenous' ||
                                            'hematology|oncology and leukemia|leukemia|acute lymphocytic|L3' ||
                                            'hematology|oncology and leukemia|leukemia|acute lymphocytic|L1' ||
                                            'hematology|oncology and leukemia|leukemia|acute lymphocytic' ||
                                            'hematology|oncology and leukemia|leukemia' ||
                                            'hematology|oncology and leukemia|carcinomatosis' ||
                                            'hematology|coagulation disorders|DIC syndrome|associated with M3 leukemia' ||
                                            'gastrointestinal|GI bleeding / PUD|lower GI bleeding|due to malignancy' ||
                                            'endocrine|pituitary and temperature regulation|central diabetes insipidus|from primary or metastatic tumor' ||
                                            'endocrine|fluids and electrolytes|hyponatremia|due to elevated ADH levels|from tumor other than lung' ||
                                            'endocrine|fluids and electrolytes|hypocalcemia|due to tumor lysis' ||
                                            'endocrine|fluids and electrolytes|hyperkalemia|due to tumor lysis' ||
                                            'endocrine|fluids and electrolytes|hypercalcemia|due to malignancy' ||
                                            'endocrine|fluids and electrolytes|diabetes insipidus|central|from primary or metastatic tumor' ||
                                            'endocrine|endocrine tumors|thyroid carcinoma' ||
                                            'endocrine|endocrine tumors|pheochromocytoma|malignant' ||
                                            'endocrine|endocrine tumors|carcinoid syndrome')
    AND NOT REGEXP_CONTAINS(diagnosisstring, r'(?i)(benign)')
    GROUP BY patientunitstayid
)
, past_cancer AS
(
    SELECT
          patientunitstayid
        , MAX(CASE
                WHEN REGEXP_CONTAINS(pasthistorypath, r'(?i)(hemato)|(leukemia)|(lymphoma)|(myeloma)') THEN 1
            ELSE 0
          END) AS hemato
        , MAX(CASE
            WHEN REGEXP_CONTAINS(pasthistorypath, r'(?i)(GU)|(renal)|(kidney)|(genetal)') THEN 1
            ELSE 0
          END) AS renal_GU
        , MAX(CASE
            WHEN REGEXP_CONTAINS(pasthistorypath, r'(?i)(thymoma)|(breast)|(mediastinal)') THEN 1
            ELSE 0
          END) AS chest_breast
        , MAX(CASE
            WHEN REGEXP_CONTAINS(pasthistorypath, r'(?i)(pulmonary)|(lung)') THEN 1
            ELSE 0
          END) AS lung_pulmonary
        , MAX(CASE
            WHEN REGEXP_CONTAINS(pasthistorypath, r'(?i)(skin)|(muscle)|(skeletal)') THEN 1
            ELSE 0
          END) AS mss_skin
        , MAX(CASE
            WHEN REGEXP_CONTAINS(pasthistorypath, r'(?i)(head)|(neck)') THEN 1
            ELSE 0
          END) AS head_neck
        , MAX(CASE
            WHEN REGEXP_CONTAINS(pasthistorypath, r'(?i)(GI)|(colon)|(intestinal)|(gastric)|(liver)|(pancreatic)|(gallbladder)|(esophageal)|(anal)') THEN 1
            ELSE 0
          END) AS GI
        , MAX(CASE
            WHEN REGEXP_CONTAINS(pasthistorypath, r'(?i)(brain)|(spine)|(spinal)|(CNS)') THEN 1
            ELSE 0
          END) AS CNS
        , MAX(CASE
            WHEN REGEXP_CONTAINS(pasthistorypath, r'(?i)(endocrine)') THEN 1
            ELSE 0
          END) AS endocrine
    FROM `physionet-data.eicu_crd.pasthistory`
    WHERE REGEXP_CONTAINS(pasthistorypath, r'(?i)' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Cancer-Primary Site/bile duct' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Cancer-Primary Site/bladder' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Cancer-Primary Site/bone' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Cancer-Primary Site/brain' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Cancer-Primary Site/breast' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Cancer-Primary Site/colon' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Cancer-Primary Site/esophagus' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Cancer-Primary Site/head and neck' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Cancer-Primary Site/kidney' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Cancer-Primary Site/liver' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Cancer-Primary Site/lung' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Cancer-Primary Site/melanoma' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Cancer-Primary Site/other' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Cancer-Primary Site/ovary' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Cancer-Primary Site/pancreas - adenocarcinoma' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Cancer-Primary Site/pancreas - islet cell' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Cancer-Primary Site/prostate' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Cancer-Primary Site/sarcoma' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Cancer-Primary Site/stomach' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Cancer-Primary Site/testes' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Cancer-Primary Site/unknown' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Cancer-Primary Site/uterus' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Hematologic Malignancy/ALL' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Hematologic Malignancy/AML' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Hematologic Malignancy/CLL' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Hematologic Malignancy/CML' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Hematologic Malignancy/Hodgkins disease' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Hematologic Malignancy/leukemia - other' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Hematologic Malignancy/multiple myeloma' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Hematologic Malignancy/non-Hodgkins lymphoma' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Hematologic Malignancy/other hematologic malignancy' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Metastases/bone' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Metastases/brain' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Metastases/carcinomatosis' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Metastases/intra-abdominal' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Metastases/liver' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Metastases/lung' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Metastases/nodes' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Metastases/other' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer Therapy/Chemotherapy/Alkylating agents (bleomycin, cytoxan, cyclophos.)' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer Therapy/Chemotherapy/Anthracyclines (adriamycin, daunorubicin)' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer Therapy/Chemotherapy/chemotherapy within past 6 mos.' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer Therapy/Chemotherapy/chemotherapy within past mo.' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer Therapy/Chemotherapy/Cis-platinum' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer Therapy/Chemotherapy/Vincristine' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer Therapy/Radiation Therapy within past 6 months/bone' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer Therapy/Radiation Therapy within past 6 months/brain' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer Therapy/Radiation Therapy within past 6 months/liver' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer Therapy/Radiation Therapy within past 6 months/lung' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer Therapy/Radiation Therapy within past 6 months/nodes' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer Therapy/Radiation Therapy within past 6 months/other' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer Therapy/Radiation Therapy within past 6 months/primary site' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Myeloproliferative Disease/myelofibrosis' ||
                                            'notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Myeloproliferative Disease/polycythemia vera')
    AND NOT REGEXP_CONTAINS(pasthistorypath, r'(?i)(benign)')
    GROUP BY patientunitstayid
)
, dx_sepsis AS
(
    SELECT
          patientunitstayid
        , MAX(CASE
            WHEN REGEXP_CONTAINS(diagnosisstring, r'(?i)(sepsis)|(septic)') THEN 1
            ELSE 0
          END) AS sepsis
    FROM `physionet-data.eicu_crd.diagnosis`
    WHERE REGEXP_CONTAINS(diagnosisstring, '(?i)((^|[|])onc)|' ||
                                            'cardiovascular|shock / hypotension|sepsis' ||
                                            'cardiovascular|shock / hypotension|sepsis|sepsis with multi-organ dysfunction' ||
                                            'cardiovascular|shock / hypotension|sepsis|sepsis with single organ dysfunction- acute renal failure' ||
                                            'cardiovascular|shock / hypotension|sepsis|sepsis with single organ dysfunction- acute respiratory failure' ||
                                            'cardiovascular|shock / hypotension|sepsis|sepsis with single organ dysfunction- circulatory system failure' ||
                                            'cardiovascular|shock / hypotension|sepsis|sepsis with single organ dysfunction- congestive heart failure' ||
                                            'cardiovascular|shock / hypotension|sepsis|sepsis with single organ dysfunction- critical care myopathy' ||
                                            'cardiovascular|shock / hypotension|sepsis|sepsis with single organ dysfunction- critical care neuropathy' ||
                                            'cardiovascular|shock / hypotension|sepsis|sepsis with single organ dysfunction- metabolic encephalopathy' ||
                                            'cardiovascular|shock / hypotension|sepsis|sepsis with single organ dysfunction-acute hepatic failure' ||
                                            'cardiovascular|shock / hypotension|sepsis|severe' ||
                                            'cardiovascular|shock / hypotension|sepsis|severe|with septic shock' ||
                                            'cardiovascular|shock / hypotension|sepsis|severe|without septic shock' ||
                                            'cardiovascular|shock / hypotension|septic shock' ||
                                            'cardiovascular|shock / hypotension|septic shock|culture negative' ||
                                            'cardiovascular|shock / hypotension|septic shock|cultures pending' ||
                                            'cardiovascular|shock / hypotension|septic shock|organism identified' ||
                                            'cardiovascular|shock / hypotension|septic shock|organism identified|fungal' ||
                                            'cardiovascular|shock / hypotension|septic shock|organism identified|gram negative organism' ||
                                            'cardiovascular|shock / hypotension|septic shock|organism identified|gram positive organism' ||
                                            'cardiovascular|shock / hypotension|septic shock|organism identified|parasitic' ||
                                            'cardiovascular|shock / hypotension|signs and symptoms of sepsis (SIRS)' ||
                                            'cardiovascular|shock / hypotension|signs and symptoms of sepsis (SIRS)|due to infectious process with organ dysfunction' ||
                                            'cardiovascular|shock / hypotension|signs and symptoms of sepsis (SIRS)|due to infectious process with organ dysfunction|non-infectious origin with acute organ dysfunction' ||
                                            'cardiovascular|shock / hypotension|signs and symptoms of sepsis (SIRS)|due to infectious process without organ dysfunction' ||
                                            'cardiovascular|shock / hypotension|signs and symptoms of sepsis (SIRS)|due to infectious process without organ dysfunction|systemic inflammatory response syndrome (SIRS) of non-infectious origin witho' ||
                                            'cardiovascular|shock / hypotension|signs and symptoms of sepsis (SIRS)|due to infectious process without organ dysfunction|unspecified organism' ||
                                            'cardiovascular|shock / hypotension|signs and symptoms of sepsis (SIRS)|due to non-infectious process with organ dysfunction' ||
                                            'cardiovascular|shock / hypotension|signs and symptoms of sepsis (SIRS)|due to non-infectious process without organ dysfunction' ||
                                            'cardiovascular|vascular disorders|arterial thromboembolism|due to sepsis' ||
                                            'cardiovascular|vascular disorders|peripheral vascular ischemia|due to sepsis' ||
                                            'endocrine|fluids and electrolytes|hypocalcemia|due to sepsis' ||
                                            'hematology|coagulation disorders|DIC syndrome|associated with sepsis/septic shock' ||
                                            'infectious diseases|systemic/other infections|sepsis' ||
                                            'infectious diseases|systemic/other infections|sepsis|sepsis with multi-organ dysfunction syndrome' ||
                                            'infectious diseases|systemic/other infections|sepsis|sepsis with single organ dysfunction- acute renal failure' ||
                                            'infectious diseases|systemic/other infections|sepsis|sepsis with single organ dysfunction- acute respiratory failure' ||
                                            'infectious diseases|systemic/other infections|sepsis|sepsis with single organ dysfunction- circulatory system failure' ||
                                            'infectious diseases|systemic/other infections|sepsis|sepsis with single organ dysfunction- congestive heart failure' ||
                                            'infectious diseases|systemic/other infections|sepsis|sepsis with single organ dysfunction- critical care myopathy' ||
                                            'infectious diseases|systemic/other infections|sepsis|sepsis with single organ dysfunction- critical care neuropathy' ||
                                            'infectious diseases|systemic/other infections|sepsis|sepsis with single organ dysfunction- metabolic encephalopathy' ||
                                            'infectious diseases|systemic/other infections|sepsis|sepsis with single organ dysfunction-acute hepatic failure' ||
                                            'infectious diseases|systemic/other infections|sepsis|severe' ||
                                            'infectious diseases|systemic/other infections|sepsis|severe|septic shock' ||
                                            'infectious diseases|systemic/other infections|sepsis|severe|without septic shock' ||
                                            'infectious diseases|systemic/other infections|septic shock' ||
                                            'infectious diseases|systemic/other infections|septic shock|culture negative' ||
                                            'infectious diseases|systemic/other infections|septic shock|cultures pending' ||
                                            'infectious diseases|systemic/other infections|septic shock|fungal' ||
                                            'infectious diseases|systemic/other infections|septic shock|gram negative organism' ||
                                            'infectious diseases|systemic/other infections|septic shock|gram positive organism' ||
                                            'infectious diseases|systemic/other infections|septic shock|organism identified' ||
                                            'infectious diseases|systemic/other infections|septic shock|parasitic' ||
                                            'infectious diseases|systemic/other infections|signs and symptoms of sepsis (SIRS)' ||
                                            'infectious diseases|systemic/other infections|signs and symptoms of sepsis (SIRS)|due to infectious process with organ dysfunction' ||
                                            'infectious diseases|systemic/other infections|signs and symptoms of sepsis (SIRS)|due to infectious process without organ dysfunction' ||
                                            'infectious diseases|systemic/other infections|signs and symptoms of sepsis (SIRS)|due to non-infectious process with organ dysfunction' ||
                                            'infectious diseases|systemic/other infections|signs and symptoms of sepsis (SIRS)|due to non-infectious process without organ dysfunction' ||
                                            'pulmonary|respiratory failure|acute lung injury|non-pulmonary etiology|sepsis' ||
                                            'pulmonary|respiratory failure|ARDS|non-pulmonary etiology|sepsis' ||
                                            'renal|disorder of kidney|acute renal failure|due to sepsis' ||
                                            'renal|electrolyte imbalance|hypocalcemia|due to sepsis')
    GROUP BY patientunitstayid
)
, admit_sepsis AS
(
    SELECT
          patientunitstayid
        , MAX(CASE
            WHEN REGEXP_CONTAINS(admitdxpath, r'(?i)(sepsis)|(septic)') THEN 1
            ELSE 0
          END) AS sepsis
    FROM `physionet-data.eicu_crd.admissiondx`
    WHERE REGEXP_CONTAINS(admitdxpath, r'(?i)((^|[|])onc)|' ||
                                            'cardiovascular|shock / hypotension|sepsis' ||
                                            'cardiovascular|shock / hypotension|sepsis|sepsis with multi-organ dysfunction' ||
                                            'cardiovascular|shock / hypotension|sepsis|sepsis with single organ dysfunction- acute renal failure' ||
                                            'cardiovascular|shock / hypotension|sepsis|sepsis with single organ dysfunction- acute respiratory failure' ||
                                            'cardiovascular|shock / hypotension|sepsis|sepsis with single organ dysfunction- circulatory system failure' ||
                                            'cardiovascular|shock / hypotension|sepsis|sepsis with single organ dysfunction- congestive heart failure' ||
                                            'cardiovascular|shock / hypotension|sepsis|sepsis with single organ dysfunction- critical care myopathy' ||
                                            'cardiovascular|shock / hypotension|sepsis|sepsis with single organ dysfunction- critical care neuropathy' ||
                                            'cardiovascular|shock / hypotension|sepsis|sepsis with single organ dysfunction- metabolic encephalopathy' ||
                                            'cardiovascular|shock / hypotension|sepsis|sepsis with single organ dysfunction-acute hepatic failure' ||
                                            'cardiovascular|shock / hypotension|sepsis|severe' ||
                                            'cardiovascular|shock / hypotension|sepsis|severe|with septic shock' ||
                                            'cardiovascular|shock / hypotension|sepsis|severe|without septic shock' ||
                                            'cardiovascular|shock / hypotension|septic shock' ||
                                            'cardiovascular|shock / hypotension|septic shock|culture negative' ||
                                            'cardiovascular|shock / hypotension|septic shock|cultures pending' ||
                                            'cardiovascular|shock / hypotension|septic shock|organism identified' ||
                                            'cardiovascular|shock / hypotension|septic shock|organism identified|fungal' ||
                                            'cardiovascular|shock / hypotension|septic shock|organism identified|gram negative organism' ||
                                            'cardiovascular|shock / hypotension|septic shock|organism identified|gram positive organism' ||
                                            'cardiovascular|shock / hypotension|septic shock|organism identified|parasitic' ||
                                            'cardiovascular|shock / hypotension|signs and symptoms of sepsis (SIRS)' ||
                                            'cardiovascular|shock / hypotension|signs and symptoms of sepsis (SIRS)|due to infectious process with organ dysfunction' ||
                                            'cardiovascular|shock / hypotension|signs and symptoms of sepsis (SIRS)|due to infectious process with organ dysfunction|non-infectious origin with acute organ dysfunction' ||
                                            'cardiovascular|shock / hypotension|signs and symptoms of sepsis (SIRS)|due to infectious process without organ dysfunction' ||
                                            'cardiovascular|shock / hypotension|signs and symptoms of sepsis (SIRS)|due to infectious process without organ dysfunction|systemic inflammatory response syndrome (SIRS) of non-infectious origin witho' ||
                                            'cardiovascular|shock / hypotension|signs and symptoms of sepsis (SIRS)|due to infectious process without organ dysfunction|unspecified organism' ||
                                            'cardiovascular|shock / hypotension|signs and symptoms of sepsis (SIRS)|due to non-infectious process with organ dysfunction' ||
                                            'cardiovascular|shock / hypotension|signs and symptoms of sepsis (SIRS)|due to non-infectious process without organ dysfunction' ||
                                            'cardiovascular|vascular disorders|arterial thromboembolism|due to sepsis' ||
                                            'cardiovascular|vascular disorders|peripheral vascular ischemia|due to sepsis' ||
                                            'endocrine|fluids and electrolytes|hypocalcemia|due to sepsis' ||
                                            'hematology|coagulation disorders|DIC syndrome|associated with sepsis/septic shock' ||
                                            'infectious diseases|systemic/other infections|sepsis' ||
                                            'infectious diseases|systemic/other infections|sepsis|sepsis with multi-organ dysfunction syndrome' ||
                                            'infectious diseases|systemic/other infections|sepsis|sepsis with single organ dysfunction- acute renal failure' ||
                                            'infectious diseases|systemic/other infections|sepsis|sepsis with single organ dysfunction- acute respiratory failure' ||
                                            'infectious diseases|systemic/other infections|sepsis|sepsis with single organ dysfunction- circulatory system failure' ||
                                            'infectious diseases|systemic/other infections|sepsis|sepsis with single organ dysfunction- congestive heart failure' ||
                                            'infectious diseases|systemic/other infections|sepsis|sepsis with single organ dysfunction- critical care myopathy' ||
                                            'infectious diseases|systemic/other infections|sepsis|sepsis with single organ dysfunction- critical care neuropathy' ||
                                            'infectious diseases|systemic/other infections|sepsis|sepsis with single organ dysfunction- metabolic encephalopathy' ||
                                            'infectious diseases|systemic/other infections|sepsis|sepsis with single organ dysfunction-acute hepatic failure' ||
                                            'infectious diseases|systemic/other infections|sepsis|severe' ||
                                            'infectious diseases|systemic/other infections|sepsis|severe|septic shock' ||
                                            'infectious diseases|systemic/other infections|sepsis|severe|without septic shock' ||
                                            'infectious diseases|systemic/other infections|septic shock' ||
                                            'infectious diseases|systemic/other infections|septic shock|culture negative' ||
                                            'infectious diseases|systemic/other infections|septic shock|cultures pending' ||
                                            'infectious diseases|systemic/other infections|septic shock|fungal' ||
                                            'infectious diseases|systemic/other infections|septic shock|gram negative organism' ||
                                            'infectious diseases|systemic/other infections|septic shock|gram positive organism' ||
                                            'infectious diseases|systemic/other infections|septic shock|organism identified' ||
                                            'infectious diseases|systemic/other infections|septic shock|parasitic' ||
                                            'infectious diseases|systemic/other infections|signs and symptoms of sepsis (SIRS)' ||
                                            'infectious diseases|systemic/other infections|signs and symptoms of sepsis (SIRS)|due to infectious process with organ dysfunction' ||
                                            'infectious diseases|systemic/other infections|signs and symptoms of sepsis (SIRS)|due to infectious process without organ dysfunction' ||
                                            'infectious diseases|systemic/other infections|signs and symptoms of sepsis (SIRS)|due to non-infectious process with organ dysfunction' ||
                                            'infectious diseases|systemic/other infections|signs and symptoms of sepsis (SIRS)|due to non-infectious process without organ dysfunction' ||
                                            'pulmonary|respiratory failure|acute lung injury|non-pulmonary etiology|sepsis' ||
                                            'pulmonary|respiratory failure|ARDS|non-pulmonary etiology|sepsis' ||
                                            'renal|disorder of kidney|acute renal failure|due to sepsis' ||
                                            'renal|electrolyte imbalance|hypocalcemia|due to sepsis')
    GROUP BY patientunitstayid
)
, icd_cancer AS
(
    SELECT
          patientunitstayid
        , MAX(CASE
            WHEN SUBSTR(icd9code,0,3) BETWEEN '200' AND '208' THEN 1
            ELSE 0
          END) AS hemato
    FROM `physionet-data.eicu_crd.diagnosis`
    WHERE SUBSTR(icd9code,0,3) BETWEEN '140' AND '149'
    OR SUBSTR(icd9code,0,3) BETWEEN '150' AND '159'
    OR SUBSTR(icd9code,0,3) BETWEEN '160' AND '165'
    OR SUBSTR(icd9code,0,3) BETWEEN '170' AND '176'
    OR SUBSTR(icd9code,0,3) BETWEEN '179' AND '189'
    OR SUBSTR(icd9code,0,3) BETWEEN '190' AND '199'
    OR SUBSTR(icd9code,0,3) LIKE '209'
    OR SUBSTR(icd9code,0,3) BETWEEN '235' AND '239'
    OR SUBSTR(icd9code,0,3) BETWEEN '200' AND '208'
    OR SUBSTR(icd9code,0,3) BETWEEN '200' AND '208' -- blood-cancer
    GROUP BY patientunitstayid
)
, icd_sepsis AS -- CODE SYNTAX IS NOT GOOD, fix
(
    SELECT
          patientunitstayid
        , MAX(CASE
            WHEN SUBSTR(icd9code,0,3) BETWEEN '785.51' AND '995.92' THEN 1
            ELSE 0
          END) AS sepsis
    FROM `physionet-data.eicu_crd.diagnosis`
    WHERE SUBSTR(icd9code,0,3) LIKE '995.91'
    OR SUBSTR(icd9code,0,3) LIKE '785.52'
    GROUP BY patientunitstayid
)
, icd_mestastasis AS -- CODE SYNTAX IS NOT GOOD, fix
(
    SELECT
          patientunitstayid
        , MAX(CASE
            WHEN SUBSTR(icd9code,0,3) LIKE '198' THEN 1
            ELSE 0
          END) AS metastasis
    FROM `physionet-data.eicu_crd.diagnosis`
    WHERE SUBSTR(icd9code,0,3) LIKE '198'
    GROUP BY patientunitstayid
)
, score_apache AS
(
    SELECT
          patientunitstayid
        , max(apachescore) as score_apache
        , MAX(CASE
            WHEN REGEXP_CONTAINS(actualicumortality, r'(?i)(ALIVE)') THEN 1
            ELSE 0
          END) AS mortality
    FROM `physionet-data.eicu_crd.apachepatientresult`
    GROUP BY patientunitstayid
)
, metastasis AS
(
    SELECT
          patientunitstayid
        , MAX(CASE
            WHEN REGEXP_CONTAINS(diagnosisstring, r'(?i)(metastasis)|(metastatic)|(stage 4)') THEN 1
            ELSE 0
          END) AS metastasis
    FROM `physionet-data.eicu_crd.diagnosis`
    WHERE REGEXP_CONTAINS(diagnosisstring, r'(?i)(metastasis)|(metastatic)|(stage 4)')
    GROUP BY patientunitstayid
)
, gram_positive AS
(
    SELECT
          patientunitstayid
        , MAX(CASE
            WHEN REGEXP_CONTAINS(organism, r'(?i)(positive)|(streptococcus)|(staphylococcus)|(aureus)|(enterococcus)|(faecalis)|(clostridium)|(corynebacterium)') THEN 1
            ELSE 0
          END) AS gram_positive
    FROM `physionet-data.eicu_crd.microlab`
    GROUP BY patientunitstayid
)
, gram_negative AS
(
    SELECT
          patientunitstayid
        , MAX(CASE
            WHEN REGEXP_CONTAINS(organism, r'(?i)(negative)|(pseudomonas)|(aeruginosa)|(klebsiella)|(haemophilus)|(stenotrophomonas)|(campylobacter)' ||
                                            '(enterobacter)|(bacteroides)|(escherichia)|(proteus)|(serratia)|(acinetobacter)|(legionella)|(neisseria)') THEN 1
            ELSE 0
          END) AS gram_negative
    FROM `physionet-data.eicu_crd.microlab`
    GROUP BY patientunitstayid
)
, yeast AS
(
    SELECT
          patientunitstayid
        , MAX(CASE
            WHEN REGEXP_CONTAINS(organism, r'(?i)(yeast)|(candida)|(Aspergillus)') THEN 1
            ELSE 0
          END) AS yeast
    FROM `physionet-data.eicu_crd.microlab`
    GROUP BY patientunitstayid
)
, admission_diagnosis AS
(
    SELECT
          patientunitstayid
        ,STRING_AGG(admitdxtext) as admission_diagnosis
    FROM `physionet-data.eicu_crd.admissiondx`
    GROUP BY patientunitstayid
)



SELECT
      icu.*
    , aps.acutePhysiologyScore
    , aps.apacheScore
    , aps.predictedhospitalmortality
    , apache.metastaticcancer
    , admission_diagnosis.admission_diagnosis
    , CASE
        WHEN icd.hemato = 1 THEN 1
        WHEN icd.hemato = 0 THEN 1
        ELSE 0
      END AS has_icd_code
    , CASE
        WHEN dx.hemato = 1 THEN 1
        WHEN icd.hemato = 1 THEN 1
        WHEN past.hemato = 1 THEN 1
        WHEN apache.leukemia = 1 THEN 1
        WHEN apache.lymphoma = 1 THEN 1
        ELSE 0
      END AS cancer_hemato
    , CASE
        WHEN dx.hemato = 0 THEN 1
        WHEN icd.hemato = 0 THEN 1
        WHEN past.hemato = 0 THEN 1
        WHEN apache.metastaticcancer = 1 THEN 1
        ELSE 0
      END AS cancer_non_hemato
    ,CASE
        WHEN dx_sepsis.sepsis = 1 THEN 1
        WHEN icd_sepsis.sepsis = 1 THEN 1
        WHEN admit_sepsis.sepsis = 1 THEN 1
        ELSE 0
      END AS has_sepsis
    ,CASE
        WHEN score_apache.mortality = 1 THEN 1
        ELSE 0
    END AS mortality
    ,CASE
        WHEN metastasis.metastasis = 1 THEN 1
        WHEN apache.metastaticcancer = 1 THEN 1
        WHEN icd_mestastasis.metastasis = 1 THEN 1
        ELSE 0
    END AS metastasis
    ,CASE
        WHEN gram_positive.gram_positive = 1 THEN 1
        ELSE 0
    END AS gram_positive
    ,CASE
        WHEN gram_negative.gram_negative = 1 THEN 1
        ELSE 0
    END AS gram_negative
    ,CASE
        WHEN yeast.yeast = 1 THEN 1
        ELSE 0
    END AS yeast
    ,CASE
        WHEN dx.renal_GU = 1 THEN 1
        WHEN past.renal_GU = 1 THEN 1
        ELSE 0
    END AS renal_GU
    ,CASE
        WHEN dx.chest_breast = 1 THEN 1
        WHEN past.chest_breast = 1 THEN 1
        ELSE 0
    END AS chest_breast
     ,CASE
        WHEN dx.lung_pulmonary = 1 THEN 1
        WHEN past.lung_pulmonary = 1 THEN 1
        ELSE 0
    END AS lung_pulmonary
    ,CASE
        WHEN dx.mss_skin = 1 THEN 1
        WHEN past.mss_skin = 1 THEN 1
        ELSE 0
    END AS mss_skin
    ,CASE
        WHEN dx.head_neck = 1 THEN 1
        WHEN past.head_neck = 1 THEN 1
        ELSE 0
    END AS head_neck
    ,CASE
        WHEN dx.GI = 1 THEN 1
        WHEN past.GI = 1 THEN 1
        ELSE 0
    END AS GI
    ,CASE
        WHEN dx.CNS = 1 THEN 1
        WHEN past.CNS = 1 THEN 1
        ELSE 0
    END AS CNS
    ,CASE
        WHEN dx.endocrine = 1 THEN 1
        WHEN past.endocrine = 1 THEN 1
        ELSE 0
    END AS endocrine

FROM icustays icu
JOIN `physionet-data.eicu_crd.apachepatientresult` aps
    ON icu.patientunitstayid = aps.patientunitstayid
    AND aps.apacheversion = 'IVa'
JOIN `physionet-data.eicu_crd.apachepredvar` apache
    ON icu.patientunitstayid = apache.patientunitstayid
LEFT JOIN dx_cancer dx
    ON icu.patientunitstayid = dx.patientunitstayid
LEFT JOIN icd_cancer icd
    ON icu.patientunitstayid = icd.patientunitstayid
LEFT JOIN past_cancer past
    ON icu.patientunitstayid = past.patientunitstayid
LEFT JOIN dx_sepsis dx_sepsis
    ON icu.patientunitstayid = dx_sepsis.patientunitstayid
LEFT JOIN icd_sepsis icd_sepsis
    ON icu.patientunitstayid = icd_sepsis.patientunitstayid
LEFT JOIN admit_sepsis admit_sepsis
    ON icu.patientunitstayid = admit_sepsis.patientunitstayid
LEFT JOIN score_apache score_apache
    ON icu.patientunitstayid = score_apache.patientunitstayid
LEFT JOIN metastasis metastasis
    ON icu.patientunitstayid = metastasis.patientunitstayid
LEFT JOIN metastasis icd_mestastasis
    ON icu.patientunitstayid = icd_mestastasis.patientunitstayid
LEFT JOIN gram_positive gram_positive
    ON icu.patientunitstayid = gram_positive.patientunitstayid
LEFT JOIN gram_negative gram_negative
    ON icu.patientunitstayid = gram_negative.patientunitstayid
LEFT JOIN yeast yeast
    ON icu.patientunitstayid = yeast.patientunitstayid
LEFT JOIN admission_diagnosis admission_diagnosis
    ON icu.patientunitstayid = admission_diagnosis.patientunitstayid

WHERE apachescore is not null
AND apachescore >= 0
AND aps.predictedhospitalmortality >= 0
AND age_num >= 18
-- AND HOSP_NUM = 1
-- AND ICUSTAY_NUM = 1
;




