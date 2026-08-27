-- 04_covariates.sql — 模型协变量提取（SAP 5.2 / 5.3 节，附录 A，W3）
--
-- 输入：Python 端注册的 cohort 视图（cohort_ecg.parquet，最终分析队列 16,499 例）
-- 内容：
--   lactate            乳酸：t0±6h 内首次值；缺失时取 t0±24h 内首次值（SAP 5.2）
--   lactate_window     乳酸取值窗口标记（'pm6h' / 'pm24h'）
--   infection_site     感染部位（t0±24h 内距 t0 最近的培养标本 spec_type_desc 映射：
--                      respiratory / urinary / bloodstream / abdominal / other；
--                      无窗内培养者为 other）
--   pre_icu_los_h      入 ICU 前住院时长（小时，icu_intime - admittime）
--   admission_emergency 入院类型二值：TRUE=急诊/紧急类，FALSE=择期类
--                      （ELECTIVE / SURGICAL SAME DAY ADMISSION 归择期，其余归急诊）

WITH c AS (
    SELECT subject_id, hadm_id, stay_id, t0,
           admittime, icu_intime, admission_type
    FROM cohort
),
lact_all AS (
    SELECT c.stay_id, l.valuenum, l.charttime,
           (l.charttime BETWEEN c.t0 - INTERVAL '6' HOUR
                            AND c.t0 + INTERVAL '6' HOUR) AS in_6h
    FROM c
    JOIN main.labevents l ON l.hadm_id = c.hadm_id
    WHERE l.itemid IN (50813, 52442, 53154)
      AND l.charttime BETWEEN c.t0 - INTERVAL '24' HOUR
                          AND c.t0 + INTERVAL '24' HOUR
      AND l.valuenum IS NOT NULL
),
lact_ranked AS (
    SELECT stay_id, valuenum, in_6h,
           ROW_NUMBER() OVER (PARTITION BY stay_id, in_6h
                              ORDER BY charttime) AS rn
    FROM lact_all
),
lact AS (
    SELECT stay_id,
           COALESCE(MAX(CASE WHEN in_6h AND rn = 1 THEN valuenum END),
                    MAX(CASE WHEN NOT in_6h AND rn = 1 THEN valuenum END)) AS lactate,
           CASE WHEN MAX(CASE WHEN in_6h AND rn = 1 THEN 1 ELSE 0 END) = 1
                THEN 'pm6h' ELSE 'pm24h' END AS lactate_window
    FROM lact_ranked
    GROUP BY stay_id
),
micro AS (
    -- 排除监测性筛查与血清学检测（不指示感染部位）：
    -- MRSA/金葡菌监测拭子、血清学、免疫学、EBV/CMV 血检
    SELECT c.stay_id, m.spec_type_desc,
           ROW_NUMBER() OVER (
               PARTITION BY c.stay_id
               ORDER BY abs(epoch(COALESCE(m.charttime, CAST(m.chartdate AS TIMESTAMP)) - c.t0))
           ) AS rn
    FROM c
    JOIN main.microbiologyevents m ON m.hadm_id = c.hadm_id
    WHERE COALESCE(m.charttime, CAST(m.chartdate AS TIMESTAMP))
              BETWEEN c.t0 - INTERVAL '24' HOUR AND c.t0 + INTERVAL '24' HOUR
      AND m.spec_type_desc IS NOT NULL
      AND NOT regexp_matches(lower(m.spec_type_desc),
           'mrsa|staph aureus swab|serology|immunology|blood \((ebv|cmv)')
)
SELECT
    c.subject_id,
    c.stay_id,
    la.lactate,
    la.lactate_window,
    CASE
        WHEN mi.spec_type_desc IS NULL THEN 'other'
        WHEN regexp_matches(lower(mi.spec_type_desc),
             'sputum|bronch|bal|lavage|tracheal|pleural|throat|nasophar|respir|lung|influenza|viral screen') THEN 'respiratory'
        WHEN regexp_matches(lower(mi.spec_type_desc), 'urine|urinary') THEN 'urinary'
        WHEN regexp_matches(lower(mi.spec_type_desc), 'blood') THEN 'bloodstream'
        WHEN regexp_matches(lower(mi.spec_type_desc),
             'peritoneal|ascites|abdom|bile|periton') THEN 'abdominal'
        ELSE 'other'
    END AS infection_site,
    date_diff('second', c.admittime, c.icu_intime) / 3600.0 AS pre_icu_los_h,
    (c.admission_type NOT IN ('ELECTIVE', 'SURGICAL SAME DAY ADMISSION'))
        AS admission_emergency
FROM c
LEFT JOIN lact la ON la.stay_id = c.stay_id
LEFT JOIN micro mi ON mi.stay_id = c.stay_id AND mi.rn = 1;
