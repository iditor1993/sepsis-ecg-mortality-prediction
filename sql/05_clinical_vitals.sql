-- 05_clinical_vitals.sql — t0±24h 生命体征汇总（M1 评分 + M1+ 特征共用，SAP 5.2/8.2）
--
-- 输入：Python 端注册的 cohort 视图（cohort_ecg.parquet）
-- itemid 与前序研究 [18]（sepsis-mm-dyn src/data/config.py §5.2）对齐：
--   hr 220045；sbp 220050/220179/225309；dbp 220051/220180/225310；
--   mbp 220052/220181/225312；rr 220210/224689/224690；spo2 220277；
--   temp 223762(℃)/223761(℉，转℃)；fio2 223835；gcs 220739/223900/223901
-- 输出：每通道 mean/min/max/std/last（末次值）+ gcs_min + fio2_max（供 qSOFA/NEWS/MEWS 评分）

WITH c AS (
    SELECT subject_id, stay_id, t0 FROM cohort
),
ce AS (
    SELECT c.stay_id, c.t0, e.charttime,
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
    WHERE e.charttime BETWEEN c.t0 - INTERVAL '24' HOUR AND c.t0 + INTERVAL '24' HOUR
      AND e.itemid IN (220045, 220050, 220179, 225309, 220051, 220180, 225310,
                       220052, 220181, 225312, 220210, 224689, 224690, 220277,
                       223762, 223761)
      AND e.valuenum IS NOT NULL
),
agg AS (
    SELECT stay_id, channel,
           AVG(value) AS mean, MIN(value) AS min, MAX(value) AS max,
           STDDEV_SAMP(value) AS std,
           arg_max(value, charttime) AS last
    FROM ce
    GROUP BY stay_id, channel
),
piv AS (
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
    FROM agg
    GROUP BY stay_id
),
gcs AS (
    SELECT stay_id, MIN(gcs_total) AS gcs_min
    FROM (
        SELECT c.stay_id, e.charttime, SUM(e.valuenum) AS gcs_total
        FROM c
        JOIN main.chartevents e ON e.stay_id = c.stay_id
        WHERE e.charttime BETWEEN c.t0 - INTERVAL '24' HOUR AND c.t0 + INTERVAL '24' HOUR
          AND e.itemid IN (220739, 223900, 223901)
          AND e.valuenum IS NOT NULL
        GROUP BY c.stay_id, e.charttime
        HAVING COUNT(*) = 3
    ) g
    GROUP BY stay_id
),
fio2 AS (
    SELECT c.stay_id, MAX(e.valuenum) AS fio2_max
    FROM c
    JOIN main.chartevents e ON e.stay_id = c.stay_id
    WHERE e.charttime BETWEEN c.t0 - INTERVAL '24' HOUR AND c.t0 + INTERVAL '24' HOUR
      AND e.itemid = 223835 AND e.valuenum IS NOT NULL AND e.valuenum BETWEEN 21 AND 100
    GROUP BY c.stay_id
)
SELECT c.subject_id, c.stay_id,
       piv.* EXCLUDE (stay_id), gcs.gcs_min, fio2.fio2_max
FROM c
LEFT JOIN piv  ON piv.stay_id = c.stay_id
LEFT JOIN gcs  ON gcs.stay_id = c.stay_id
LEFT JOIN fio2 ON fio2.stay_id = c.stay_id;
