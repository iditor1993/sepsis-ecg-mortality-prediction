-- 01_cohort_sepsis3.sql — Sepsis-3 队列提取（SAP 3.2 / 3.3 节）
--
-- 数据源：MIMIC-IV v3.1 DuckDB（E:/clinical_research/MIMIC_IV_3.1/mimic_iv_3_1.duckdb）
-- 判定：mimiciv_derived.sepsis3 官方派生表（疑似感染 t0 + t0±24h 内 SOFA 急性升高 ≥2）
--
-- 排除标准（SAP 3.3 节，按以下顺序依次施加，供图 1 流程图分级计数）：
--   1) 年龄 <18 岁（icustay_detail.admission_age）
--   2) 非首次脓毒症发作（同一患者多次发作仅取首次；在年龄合格的发作中按 t0 排序，
--      即"首个合格发作"）
--   3) ICU 入住 <6 h（按 icu_outtime - icu_intime 实际小时数判定，
--      避免 los_icu 两位小数舍入误差）
--   4) t0 前已死亡（admissions.deathtime <= t0）或自动出院
--      （discharge_location = 'AGAINST ADVICE' 且 dischtime <= t0）
--
-- ECG 相关排除（t0±24h 无可用 ECG / 时长不足 / 导联缺失 / 信号质量不合格）
-- 由 src/ecg_link.py 在下一步执行（SAP 3.4 节），不在本查询内。
--
-- 输出：发作级（stay_id 唯一）全量表，含逐级排除标志与最终入组标志 in_cohort；
-- 分级计数由 src/extract_cohort.py 据此计算。

WITH base AS (
    SELECT
        s.subject_id,
        s.stay_id,
        s.suspected_infection_time AS t0,
        s.antibiotic_time,
        s.culture_time,
        s.sofa_time,
        s.sofa_score,
        s.respiration    AS sofa_respiration,
        s.coagulation    AS sofa_coagulation,
        s.liver          AS sofa_liver,
        s.cardiovascular AS sofa_cardiovascular,
        s.cns            AS sofa_cns,
        s.renal          AS sofa_renal
    FROM mimiciv_derived.sepsis3 AS s
    WHERE s.sepsis3
),
enriched AS (
    SELECT
        b.*,
        d.hadm_id,
        d.gender,
        d.admission_age,
        d.admittime,
        d.dischtime,
        d.los_hospital,
        d.hospital_expire_flag,
        d.dod,
        d.icu_intime,
        d.icu_outtime,
        d.los_icu,
        a.admission_type,
        a.discharge_location,
        a.deathtime,
        p.anchor_year_group
    FROM base b
    JOIN mimiciv_derived.icustay_detail d ON b.stay_id = d.stay_id
    JOIN main.admissions            a ON d.hadm_id = a.hadm_id
    JOIN main.patients              p ON b.subject_id = p.subject_id
),
-- 年龄合格发作中的发作序号（用于"首次合格发作"判定）
ranked AS (
    SELECT
        stay_id,
        ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY t0, stay_id) AS episode_seq
    FROM enriched
    WHERE admission_age >= 18
)
SELECT
    e.subject_id,
    e.hadm_id,
    e.stay_id,
    e.t0,
    e.antibiotic_time,
    e.culture_time,
    e.sofa_time,
    e.sofa_score,
    e.sofa_respiration,
    e.sofa_coagulation,
    e.sofa_liver,
    e.sofa_cardiovascular,
    e.sofa_cns,
    e.sofa_renal,
    e.gender,
    e.admission_age,
    e.admittime,
    e.dischtime,
    e.los_hospital,
    e.hospital_expire_flag,
    e.dod,
    e.icu_intime,
    e.icu_outtime,
    e.los_icu,
    e.admission_type,
    e.discharge_location,
    e.deathtime,
    e.anchor_year_group,
    CASE
        WHEN e.anchor_year_group IN ('2008 - 2010', '2011 - 2013', '2014 - 2016')
            THEN 'dev_2008_2016'
        WHEN e.anchor_year_group = '2017 - 2019' THEN 'temporal_2017_2019'
        WHEN e.anchor_year_group = '2020 - 2022' THEN 'covid_2020_2022'
    END AS cohort_period,
    -- 逐级排除标志（累计语义：后续标志成立蕴含前序标志成立）
    (e.admission_age >= 18) AS age_ok,
    (r.episode_seq = 1)     AS first_episode_ok,
    (date_diff('second', e.icu_intime, e.icu_outtime) / 3600.0 >= 6) AS icu6h_ok,
    NOT (
        (e.deathtime IS NOT NULL AND e.deathtime <= e.t0)
        OR (e.discharge_location = 'AGAINST ADVICE' AND e.dischtime <= e.t0)
    ) AS alive_ok,
    (
        e.admission_age >= 18
        AND r.episode_seq = 1
        AND date_diff('second', e.icu_intime, e.icu_outtime) / 3600.0 >= 6
        AND NOT (
            (e.deathtime IS NOT NULL AND e.deathtime <= e.t0)
            OR (e.discharge_location = 'AGAINST ADVICE' AND e.dischtime <= e.t0)
        )
    ) AS in_cohort
FROM enriched e
LEFT JOIN ranked r ON e.stay_id = r.stay_id
ORDER BY e.subject_id, e.t0;
