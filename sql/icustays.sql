create temp table icustays as
with pt as
(
    select
          pt.*
        , case
            when pt.age = '' then null
            when REGEXP_CONTAINS(age, '>') then cast(REPLACE(age, '>','') as int64) + 1
            else cast(age as INT64)
          END as age_num
        , CASE
            WHEN REGEXP_CONTAINS(pt.gender, r'(?i)(male)') THEN 1 ELSE 0
          END AS male
        , CASE
            WHEN REGEXP_CONTAINS(pt.gender, r'(?i)(female)') THEN 1 ELSE 0
          END AS female
    from `physionet-data.eicu_crd.patient` pt
    where lower(unittype) like '%icu%'
)
select
      pt.*
    , ROW_NUMBER() over
    (
        PARTITION BY uniquepid
        ORDER BY
              hospitaldischargeyear
            , age
    ) as HOSP_NUM

from pt
;
