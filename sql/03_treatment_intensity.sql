-- 03_treatment_intensity.sql — 治疗强度与 Charlson 合并症指数
-- （SAP 5.3 协变量之机械通气/血管活性药物，时间窗 t0±24 h；同时服务 9.7 A1a）
--
-- 输入：Python 端注册的 cohort 视图（cohort_base.parquet）
-- 口径：
--   mech_vent_24h  有创机械通气（ventilation_status ∈ InvasiveVent/Tracheostomy），
--                  通气时段与 t0±24h 窗口有交叠
--   vaso_24h       血管活性药物（vasoactive_agent 中任一药物非空，含强心药），
--                  用药时段与 t0±24h 窗口有交叠

WITH c AS (
    SELECT subject_id, hadm_id, stay_id, t0 FROM cohort
)
SELECT
    c.subject_id,
    c.stay_id,
    EXISTS (
        SELECT 1 FROM mimiciv_derived.ventilation v
        WHERE v.stay_id = c.stay_id
          AND v.ventilation_status IN ('InvasiveVent', 'Tracheostomy')
          AND v.starttime <= c.t0 + INTERVAL '24' HOUR
          AND v.endtime   >= c.t0 - INTERVAL '24' HOUR
    ) AS mech_vent_24h,
    EXISTS (
        SELECT 1 FROM mimiciv_derived.vasoactive_agent va
        WHERE va.stay_id = c.stay_id
          AND (va.norepinephrine IS NOT NULL OR va.epinephrine IS NOT NULL
               OR va.dopamine IS NOT NULL OR va.phenylephrine IS NOT NULL
               OR va.vasopressin IS NOT NULL OR va.dobutamine IS NOT NULL
               OR va.milrinone IS NOT NULL)
          AND va.starttime <= c.t0 + INTERVAL '24' HOUR
          AND va.endtime   >= c.t0 - INTERVAL '24' HOUR
    ) AS vaso_24h,
    ch.charlson_comorbidity_index
FROM c
LEFT JOIN mimiciv_derived.charlson ch ON c.hadm_id = ch.hadm_id;
