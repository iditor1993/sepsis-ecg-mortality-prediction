-- 07_m1plus_pre.sql — 纯预测窗 [t0-24h, t0] 临床特征提取（M1+pre / M1pre，额外分析）
--
-- 目的：M1+ 的 t0±24h 窗口含 t0 后数据（决策时点信息泄漏质疑），
-- 本查询将全部窗口改为 [t0-24h, t0]（t0 前 24h 回看）。
-- 输入：Python 端注册的 cohort 视图（cohort_ecg.parquet）
-- 内容（每 stay 一行）：
--   7 生命体征通道 x 5 统计量（mean/min/max/std/last，itemid 同 sql/05）
--   10 检验通道 x 5 统计量（同 sql/06）
--   lactate_pre：t0 前 6h 内末次值，缺失回退 t0 前 24h 内末次值
--   gcs_min / fio2_max（pre 评分用，pre 窗口最差值）
--   mv_pre / vaso_pre：[t0-24h, t0] 内有创通气/血管活性药（EXISTS）

WITH c AS (
    SELECT subject_id, hadm_id, stay_id, t0 FROM cohort
),
ce AS (
    SELECT c.stay_id, e.charttime,
        CASE
            WHEN e.itemid IN (220045) THEN 'hr'
            WHEN e.itemid IN (220050, 220179, 225309) THEN 'sbp'
            WHEN e.itemid IN (220051, 220180, 225310) THEN 'dbp'
            WHEN e.itemid IN (220052, 220181, 225312) THEN 'mbp'
            WHEN e.itemid IN (220210, 224689, 224690) THEN 'rr'
            WHEN e.itemid IN (220277) THEN 'spo2'
            WHEN e.itemid IN (223762, 223761) THEN 'temp'
        END AS channel,
        CASE WHEN e.itemid = 223761 THEN (e.valuenum - 32) / 1.8
             ELSE e.valuenum END AS value
    FROM c
    JOIN main.chartevents e ON e.stay_id = c.stay_id
    WHERE e.charttime BETWEEN c.t0 - INTERVAL '24' HOUR AND c.t0
      AND e.itemid IN (220045, 220050, 220179, 225309, 220051, 220180, 225310,
                       220052, 220181, 225312, 220210, 224689, 224690, 220277,
                       223762, 223761)
      AND e.valuenum IS NOT NULL
),
v_agg AS (
    SELECT stay_id, channel,
           AVG(value) AS mean, MIN(value) AS min, MAX(value) AS max,
           STDDEV_SAMP(value) AS std, arg_max(value, charttime) AS last
    FROM ce GROUP BY stay_id, channel
),
v_piv AS (
    SELECT stay_id,
        MAX(CASE WHEN channel='hr'   THEN mean END) AS hr_mean,
        MAX(CASE WHEN channel='hr'   THEN min END)  AS hr_min,
        MAX(CASE WHEN channel='hr'   THEN max END)  AS hr_max,
        MAX(CASE WHEN channel='hr'   THEN std END)  AS hr_std,
        MAX(CASE WHEN channel='hr'   THEN last END) AS hr_last,
        MAX(CASE WHEN channel='sbp'  THEN mean END) AS sbp_mean,
        MAX(CASE WHEN channel='sbp'  THEN min END)  AS sbp_min,
        MAX(CASE WHEN channel='sbp'  THEN max END)  AS sbp_max,
        MAX(CASE WHEN channel='sbp'  THEN std END)  AS sbp_std,
        MAX(CASE WHEN channel='sbp'  THEN last END) AS sbp_last,
        MAX(CASE WHEN channel='dbp'  THEN mean END) AS dbp_mean,
        MAX(CASE WHEN channel='dbp'  THEN min END)  AS dbp_min,
        MAX(CASE WHEN channel='dbp'  THEN max END)  AS dbp_max,
        MAX(CASE WHEN channel='dbp'  THEN std END)  AS dbp_std,
        MAX(CASE WHEN channel='dbp'  THEN last END) AS dbp_last,
        MAX(CASE WHEN channel='mbp'  THEN mean END) AS mbp_mean,
        MAX(CASE WHEN channel='mbp'  THEN min END)  AS mbp_min,
        MAX(CASE WHEN channel='mbp'  THEN max END)  AS mbp_max,
        MAX(CASE WHEN channel='mbp'  THEN std END)  AS mbp_std,
        MAX(CASE WHEN channel='mbp'  THEN last END) AS mbp_last,
        MAX(CASE WHEN channel='rr'   THEN mean END) AS rr_mean,
        MAX(CASE WHEN channel='rr'   THEN min END)  AS rr_min,
        MAX(CASE WHEN channel='rr'   THEN max END)  AS rr_max,
        MAX(CASE WHEN channel='rr'   THEN std END)  AS rr_std,
        MAX(CASE WHEN channel='rr'   THEN last END) AS rr_last,
        MAX(CASE WHEN channel='spo2' THEN mean END) AS spo2_mean,
        MAX(CASE WHEN channel='spo2' THEN min END)  AS spo2_min,
        MAX(CASE WHEN channel='spo2' THEN max END)  AS spo2_max,
        MAX(CASE WHEN channel='spo2' THEN std END)  AS spo2_std,
        MAX(CASE WHEN channel='spo2' THEN last END) AS spo2_last,
        MAX(CASE WHEN channel='temp' THEN mean END) AS temp_mean,
        MAX(CASE WHEN channel='temp' THEN min END)  AS temp_min,
        MAX(CASE WHEN channel='temp' THEN max END)  AS temp_max,
        MAX(CASE WHEN channel='temp' THEN std END)  AS temp_std,
        MAX(CASE WHEN channel='temp' THEN last END) AS temp_last
    FROM v_agg GROUP BY stay_id
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
    WHERE l.charttime BETWEEN c.t0 - INTERVAL '24' HOUR AND c.t0
      AND l.itemid IN (50885, 51265, 50912, 51301, 51300, 51222, 50931,
                       50983, 50971, 50882, 51237)
      AND l.valuenum IS NOT NULL
),
l_agg AS (
    SELECT stay_id, channel,
           AVG(value) AS mean, MIN(value) AS min, MAX(value) AS max,
           STDDEV_SAMP(value) AS std, arg_max(value, charttime) AS last
    FROM le GROUP BY stay_id, channel
),
l_piv AS (
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
    FROM l_agg GROUP BY stay_id
),
lact_ranked AS (
    SELECT c.stay_id, l.valuenum,
           (l.charttime >= c.t0 - INTERVAL '6' HOUR) AS in6,
           ROW_NUMBER() OVER (
               PARTITION BY c.stay_id, (l.charttime >= c.t0 - INTERVAL '6' HOUR)
               ORDER BY l.charttime DESC) AS rn
    FROM c
    JOIN main.labevents l ON l.hadm_id = c.hadm_id
    WHERE l.itemid IN (50813, 52442, 53154)
      AND l.charttime BETWEEN c.t0 - INTERVAL '24' HOUR AND c.t0
      AND l.valuenum IS NOT NULL
),
lact AS (
    SELECT stay_id,
           COALESCE(MAX(CASE WHEN in6 AND rn = 1 THEN valuenum END),
                    MAX(CASE WHEN NOT in6 AND rn = 1 THEN valuenum END)) AS lactate_pre
    FROM lact_ranked GROUP BY stay_id
),
gcs AS (
    SELECT stay_id, MIN(gcs_total) AS gcs_min_pre
    FROM (
        SELECT c.stay_id, e.charttime, SUM(e.valuenum) AS gcs_total
        FROM c
        JOIN main.chartevents e ON e.stay_id = c.stay_id
        WHERE e.charttime BETWEEN c.t0 - INTERVAL '24' HOUR AND c.t0
          AND e.itemid IN (220739, 223900, 223901)
          AND e.valuenum IS NOT NULL
        GROUP BY c.stay_id, e.charttime
        HAVING COUNT(*) = 3
    ) g GROUP BY stay_id
),
fio2 AS (
    SELECT c.stay_id, MAX(e.valuenum) AS fio2_max_pre
    FROM c
    JOIN main.chartevents e ON e.stay_id = c.stay_id
    WHERE e.charttime BETWEEN c.t0 - INTERVAL '24' HOUR AND c.t0
      AND e.itemid = 223835 AND e.valuenum IS NOT NULL AND e.valuenum BETWEEN 21 AND 100
    GROUP BY c.stay_id
)
SELECT c.subject_id, c.stay_id,
       v_piv.* EXCLUDE (stay_id),
       l_piv.* EXCLUDE (stay_id),
       lact.lactate_pre,
       gcs.gcs_min_pre,
       fio2.fio2_max_pre,
       EXISTS (
           SELECT 1 FROM mimiciv_derived.ventilation v
           WHERE v.stay_id = c.stay_id
             AND v.ventilation_status IN ('InvasiveVent', 'Tracheostomy')
             AND v.starttime <= c.t0 AND v.endtime >= c.t0 - INTERVAL '24' HOUR
       ) AS mv_pre,
       EXISTS (
           SELECT 1 FROM mimiciv_derived.vasoactive_agent va
           WHERE va.stay_id = c.stay_id
             AND (va.norepinephrine IS NOT NULL OR va.epinephrine IS NOT NULL
                  OR va.dopamine IS NOT NULL OR va.phenylephrine IS NOT NULL
                  OR va.vasopressin IS NOT NULL OR va.dobutamine IS NOT NULL
                  OR va.milrinone IS NOT NULL)
             AND va.starttime <= c.t0 AND va.endtime >= c.t0 - INTERVAL '24' HOUR
       ) AS vaso_pre
FROM c
LEFT JOIN v_piv ON v_piv.stay_id = c.stay_id
LEFT JOIN l_piv ON l_piv.stay_id = c.stay_id
LEFT JOIN lact  ON lact.stay_id  = c.stay_id
LEFT JOIN gcs   ON gcs.stay_id   = c.stay_id
LEFT JOIN fio2  ON fio2.stay_id  = c.stay_id;
