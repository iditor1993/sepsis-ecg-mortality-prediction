-- 06_m1plus_labs.sql — t0±24h 检验通道汇总（M1+ 强表格基线，SAP 8.2）
--
-- 输入：Python 端注册的 cohort 视图（cohort_ecg.parquet）
-- 10 个检验通道（itemid 与前序研究 [18] 对齐；乳酸单列为 M1 变量，不重复纳入）：
--   bilirubin 50885；platelets 51265；creatinine 50912；wbc 51301/51300；
--   hemoglobin 51222；glucose 50931；sodium 50983；potassium 50971；
--   bicarbonate 50882；inr 51237
-- 每通道 mean/min/max/std/last（末次值），共 50 列。

WITH c AS (
    SELECT subject_id, hadm_id, stay_id, t0 FROM cohort
),
le AS (
    SELECT c.stay_id, l.charttime,
        CASE
            WHEN l.itemid = 50885 THEN 'bili'
            WHEN l.itemid = 51265 THEN 'plt'
            WHEN l.itemid = 50912 THEN 'cr'
            WHEN l.itemid IN (51301, 51300) THEN 'wbc'
            WHEN l.itemid = 51222 THEN 'hgb'
            WHEN l.itemid = 50931 THEN 'glu'
            WHEN l.itemid = 50983 THEN 'na'
            WHEN l.itemid = 50971 THEN 'k'
            WHEN l.itemid = 50882 THEN 'bicarb'
            WHEN l.itemid = 51237 THEN 'inr'
        END AS channel,
        l.valuenum AS value
    FROM c
    JOIN main.labevents l ON l.hadm_id = c.hadm_id
    WHERE l.charttime BETWEEN c.t0 - INTERVAL '24' HOUR AND c.t0 + INTERVAL '24' HOUR
      AND l.itemid IN (50885, 51265, 50912, 51301, 51300, 51222, 50931,
                       50983, 50971, 50882, 51237)
      AND l.valuenum IS NOT NULL
),
agg AS (
    SELECT stay_id, channel,
           AVG(value) AS mean, MIN(value) AS min, MAX(value) AS max,
           STDDEV_SAMP(value) AS std,
           arg_max(value, charttime) AS last
    FROM le
    GROUP BY stay_id, channel
)
SELECT stay_id,
    MAX(CASE WHEN channel='bili'   THEN mean END) AS bili_mean,
    MAX(CASE WHEN channel='bili'   THEN min END)  AS bili_min,
    MAX(CASE WHEN channel='bili'   THEN max END)  AS bili_max,
    MAX(CASE WHEN channel='bili'   THEN std END)  AS bili_std,
    MAX(CASE WHEN channel='bili'   THEN last END) AS bili_last,
    MAX(CASE WHEN channel='plt'    THEN mean END) AS plt_mean,
    MAX(CASE WHEN channel='plt'    THEN min END)  AS plt_min,
    MAX(CASE WHEN channel='plt'    THEN max END)  AS plt_max,
    MAX(CASE WHEN channel='plt'    THEN std END)  AS plt_std,
    MAX(CASE WHEN channel='plt'    THEN last END) AS plt_last,
    MAX(CASE WHEN channel='cr'     THEN mean END) AS cr_mean,
    MAX(CASE WHEN channel='cr'     THEN min END)  AS cr_min,
    MAX(CASE WHEN channel='cr'     THEN max END)  AS cr_max,
    MAX(CASE WHEN channel='cr'     THEN std END)  AS cr_std,
    MAX(CASE WHEN channel='cr'     THEN last END) AS cr_last,
    MAX(CASE WHEN channel='wbc'    THEN mean END) AS wbc_mean,
    MAX(CASE WHEN channel='wbc'    THEN min END)  AS wbc_min,
    MAX(CASE WHEN channel='wbc'    THEN max END)  AS wbc_max,
    MAX(CASE WHEN channel='wbc'    THEN std END)  AS wbc_std,
    MAX(CASE WHEN channel='wbc'    THEN last END) AS wbc_last,
    MAX(CASE WHEN channel='hgb'    THEN mean END) AS hgb_mean,
    MAX(CASE WHEN channel='hgb'    THEN min END)  AS hgb_min,
    MAX(CASE WHEN channel='hgb'    THEN max END)  AS hgb_max,
    MAX(CASE WHEN channel='hgb'    THEN std END)  AS hgb_std,
    MAX(CASE WHEN channel='hgb'    THEN last END) AS hgb_last,
    MAX(CASE WHEN channel='glu'    THEN mean END) AS glu_mean,
    MAX(CASE WHEN channel='glu'    THEN min END)  AS glu_min,
    MAX(CASE WHEN channel='glu'    THEN max END)  AS glu_max,
    MAX(CASE WHEN channel='glu'    THEN std END)  AS glu_std,
    MAX(CASE WHEN channel='glu'    THEN last END) AS glu_last,
    MAX(CASE WHEN channel='na'     THEN mean END) AS na_mean,
    MAX(CASE WHEN channel='na'     THEN min END)  AS na_min,
    MAX(CASE WHEN channel='na'     THEN max END)  AS na_max,
    MAX(CASE WHEN channel='na'     THEN std END)  AS na_std,
    MAX(CASE WHEN channel='na'     THEN last END) AS na_last,
    MAX(CASE WHEN channel='k'      THEN mean END) AS k_mean,
    MAX(CASE WHEN channel='k'      THEN min END)  AS k_min,
    MAX(CASE WHEN channel='k'      THEN max END)  AS k_max,
    MAX(CASE WHEN channel='k'      THEN std END)  AS k_std,
    MAX(CASE WHEN channel='k'      THEN last END) AS k_last,
    MAX(CASE WHEN channel='bicarb' THEN mean END) AS bicarb_mean,
    MAX(CASE WHEN channel='bicarb' THEN min END)  AS bicarb_min,
    MAX(CASE WHEN channel='bicarb' THEN max END)  AS bicarb_max,
    MAX(CASE WHEN channel='bicarb' THEN std END)  AS bicarb_std,
    MAX(CASE WHEN channel='bicarb' THEN last END) AS bicarb_last,
    MAX(CASE WHEN channel='inr'    THEN mean END) AS inr_mean,
    MAX(CASE WHEN channel='inr'    THEN min END)  AS inr_min,
    MAX(CASE WHEN channel='inr'    THEN max END)  AS inr_max,
    MAX(CASE WHEN channel='inr'    THEN std END)  AS inr_std,
    MAX(CASE WHEN channel='inr'    THEN last END) AS inr_last
FROM agg
GROUP BY stay_id;
