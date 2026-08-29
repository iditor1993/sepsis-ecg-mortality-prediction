-- 02_outcomes.sql — 结局判定（SAP 第四章）
--
-- 输入：Python 端通过 duckdb.register 注册的 cohort 视图（cohort_base.parquet，31,857 例；
--       含 ECG 不可链接者，以满足 A1a 对两组 28 天死亡率的比较需求）
--
-- 结局定义：
--   death_28d    主要结局：28 天全因死亡（patients.dod，自 t0 起算 0–28 天，含院外死亡记录）
--   death_inhosp 次要结局 1：院内全因死亡（hospital_expire_flag）
--   death_90d    次要结局 2：90 天全因死亡
--   death_icu    次要结局 3：ICU 内死亡（deathtime 落于 index ICU 留观内）
--   shock_28d    次要结局 4：28 天内脓毒性休克（Sepsis-3 休克标准操作化）：
--                t0 至 t0+28d 内使用血管加压药（去甲肾上腺素/肾上腺素/多巴胺/
--                去氧肾上腺素/血管加压素；dobutamine、milrinone 为强心药不计入），
--                且 t0−24h 至首次用药时点+24h 内最大乳酸 >2 mmol/L。
--                已在用药者以 t0 为锚点（vaso_start = GREATEST(starttime, t0)）。
-- 乳酸 itemid：50813 / 52442（血气）、53154（化学），单位 mmol/L。

WITH c AS (
    SELECT subject_id, hadm_id, stay_id, t0,
           icu_intime, icu_outtime, deathtime, dod, hospital_expire_flag
    FROM cohort
),
vaso AS (
    SELECT c.stay_id,
           MIN(GREATEST(v.starttime, c.t0)) AS vaso_start
    FROM c
    JOIN mimiciv_derived.vasoactive_agent v ON c.stay_id = v.stay_id
    WHERE v.starttime <= c.t0 + INTERVAL '28' DAY
      AND v.endtime   >= c.t0
      AND (v.norepinephrine IS NOT NULL OR v.epinephrine IS NOT NULL
           OR v.dopamine IS NOT NULL OR v.phenylephrine IS NOT NULL
           OR v.vasopressin IS NOT NULL)
    GROUP BY c.stay_id
),
lact AS (
    SELECT c.stay_id, MAX(l.valuenum) AS lactate_max
    FROM c
    JOIN vaso v ON c.stay_id = v.stay_id
    JOIN main.labevents l ON l.hadm_id = c.hadm_id
    WHERE l.itemid IN (50813, 52442, 53154)
      AND l.charttime >= c.t0 - INTERVAL '24' HOUR
      AND l.charttime <= v.vaso_start + INTERVAL '24' HOUR
    GROUP BY c.stay_id
)
SELECT
    c.subject_id,
    c.stay_id,
    date_diff('day', CAST(c.t0 AS DATE), c.dod) AS days_to_death,
    COALESCE(date_diff('day', CAST(c.t0 AS DATE), c.dod) BETWEEN 0 AND 28, FALSE) AS death_28d,
    (c.hospital_expire_flag = 1) AS death_inhosp,
    COALESCE(date_diff('day', CAST(c.t0 AS DATE), c.dod) BETWEEN 0 AND 90, FALSE) AS death_90d,
    (c.deathtime IS NOT NULL
        AND c.deathtime BETWEEN c.icu_intime AND c.icu_outtime) AS death_icu,
    (v.vaso_start IS NOT NULL AND COALESCE(la.lactate_max, 0) > 2.0) AS shock_28d,
    v.vaso_start,
    la.lactate_max AS shock_lactate_max
FROM c
LEFT JOIN vaso v  ON c.stay_id = v.stay_id
LEFT JOIN lact la ON la.stay_id = c.stay_id;
