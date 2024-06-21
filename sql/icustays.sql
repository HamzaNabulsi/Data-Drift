create temp table icustays as
with pt as
(
    select
          pt.*
        , case
            when pt.age = '' then null
            when REGEXP_CONTAINS(age, '>') then cast(REPLACE(age, '>','') as int64) + 1
            else cast(age as INT64)
          end as age_num
    from `physionet-data.eicu_crd.patient` pt
    where lower(unittype) like '%icu%' -- only include ICUs HAMZA:No need, all eicu patients are icu patientsw
)
select
      pt.*
    , ROW_NUMBER() over
    (
        PARTITION BY uniquepid
        ORDER BY
              hospitaldischargeyear
            , age
--             , patienthealthsystemstayid -- this is temporarilly random but deterministic
--             , hospitaladmitoffset
    ) as HOSP_NUM
--     , ROW_NUMBER() over
--     (
--         PARTITION BY patienthealthsystemstayid
--         ORDER BY hospitaladmitoffset
--     ) as ICUSTAY_NUM
from pt
;
