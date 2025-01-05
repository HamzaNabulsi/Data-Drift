CREATE temp TABLE icustays AS
WITH pt AS
(
    SELECT
          pt.*
        , CASE
            WHEN pt.age = '' THEN NULL
            WHEN REGEXP_CONTAINS(age, '>') then cast(REPLACE(age, '>','') as int64) + 1
            else cast(age as INT64)
          END as age_num
        , CASE
            WHEN REGEXP_CONTAINS(pt.gender, r'(?i)(female)') THEN 1 ELSE 0
          END AS female
        , CASE
            WHEN REGEXP_CONTAINS(pt.gender, r'(?i)(unknown)') THEN 1 ELSE 0
          END AS unkown
        , ROW_NUMBER() over
         (
        PARTITION BY uniquepid
        ORDER BY
              hospitaldischargeyear
            , age
         ) as HOSP_NUM
    from `lcp-consortium.eicu_crd_ii_v0_1_0.patient` pt
)

select
      pt.*
FROM pt
-- WHERE HOSP_NUM = 1
;
