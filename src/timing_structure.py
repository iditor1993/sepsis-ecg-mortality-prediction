"""timing_structure.py — 数据时间结构分析（回应 t0±24h 窗口泄漏质疑）。

核心事实：本队列 t0（疑似感染时刻）紧贴 ICU 入住，[t0-24h, t0] 对多数
患者落在 ICU 前时段，严格 t0 前临床模型不可构建。本脚本量化：
  1) t0 - icu_intime 分布（中位数、IQR、<0 与 <6h 比例）
  2) 代表通道在 [t0-24h,t0) 与 (t0, t0+24h] 两个半窗的可得率
     （vitals: hr/sbp/rr/spo2；labs: cr/wbc/bili；乳酸）
  3) 输出 results/timing_structure.csv 与双联图
     results/figures/timing_structure.png
供论文方法学"时间窗设计"段落与审稿质疑回应使用。
"""

from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = "E:/clinical_research/MIMIC_IV_3.1/mimic_iv_3_1.duckdb"


def main() -> None:
    datadir = REPO_ROOT / "data"
    results_dir = REPO_ROOT / "results"
    cohort = pd.read_parquet(datadir / "cohort_ecg.parquet")

    con = duckdb.connect(DEFAULT_DB, read_only=True)
    try:
        con.register("cohort", cohort)
        avail = con.sql("""
        WITH c AS (SELECT subject_id, hadm_id, stay_id, t0 FROM cohort)
        SELECT c.stay_id,
          MAX(CASE WHEN e.charttime <  c.t0 THEN 1 ELSE 0 END) AS vit_pre,
          MAX(CASE WHEN e.charttime >= c.t0 THEN 1 ELSE 0 END) AS vit_post
        FROM c JOIN main.chartevents e ON e.stay_id = c.stay_id
        WHERE e.itemid IN (220045, 220050, 220179, 220210, 220277)
          AND e.charttime BETWEEN c.t0 - INTERVAL '24' HOUR AND c.t0 + INTERVAL '24' HOUR
        GROUP BY c.stay_id
        """).df()
        labs = con.sql("""
        WITH c AS (SELECT subject_id, hadm_id, stay_id, t0 FROM cohort)
        SELECT c.stay_id,
          MAX(CASE WHEN l.charttime <  c.t0 THEN 1 ELSE 0 END) AS lab_pre,
          MAX(CASE WHEN l.charttime >= c.t0 THEN 1 ELSE 0 END) AS lab_post,
          MAX(CASE WHEN l.itemid IN (50813,52442,53154) AND l.charttime <  c.t0 THEN 1 ELSE 0 END) AS lac_pre,
          MAX(CASE WHEN l.itemid IN (50813,52442,53154) AND l.charttime >= c.t0 THEN 1 ELSE 0 END) AS lac_post
        FROM c JOIN main.labevents l ON l.hadm_id = c.hadm_id
        WHERE l.itemid IN (50912, 51301, 50885, 50813, 52442, 53154)
          AND l.charttime BETWEEN c.t0 - INTERVAL '24' HOUR AND c.t0 + INTERVAL '24' HOUR
        GROUP BY c.stay_id
        """).df()
    finally:
        con.close()

    dt_h = (cohort["t0"] - cohort["icu_intime"]).dt.total_seconds() / 3600
    av = cohort[["stay_id"]].merge(avail, on="stay_id", how="left").merge(
        labs, on="stay_id", how="left")
    for c_ in ["vit_pre", "vit_post", "lab_pre", "lab_post", "lac_pre", "lac_post"]:
        av[c_] = av[c_].fillna(0)

    stats_rows = [
        ("t0 - ICU admission time, median (h)", f"{dt_h.median():.2f}"),
        ("t0 - ICU admission time, IQR (h)", f"{dt_h.quantile(0.25):.2f} ~ {dt_h.quantile(0.75):.2f}"),
        ("t0 before ICU admission", f"{(dt_h < 0).mean():.1%}"),
        ("t0 within +-6 h of ICU admission", f"{(dt_h.abs() <= 6).mean():.1%}"),
        ("Vital signs [t0-24h,t0) availability", f"{av['vit_pre'].mean():.1%}"),
        ("Vital signs (t0,t0+24h] availability", f"{av['vit_post'].mean():.1%}"),
        ("Laboratory tests [t0-24h,t0) availability", f"{av['lab_pre'].mean():.1%}"),
        ("Laboratory tests (t0,t0+24h] availability", f"{av['lab_post'].mean():.1%}"),
        ("Lactate [t0-24h,t0) availability", f"{av['lac_pre'].mean():.1%}"),
        ("Lactate (t0,t0+24h] availability", f"{av['lac_post'].mean():.1%}"),
    ]
    stats_df = pd.DataFrame(stats_rows, columns=["Metric", "Value"])
    stats_df.to_csv(results_dir / "timing_structure.csv", index=False,
                    encoding="utf-8-sig")
    print(stats_df.to_string(index=False))

    # 双联图
    # Two-panel figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    ax.hist(dt_h.clip(-24, 72), bins=60, color='#4472a8', alpha=0.85)
    ax.axvline(0, color='darkred', ls='--', lw=1.5)
    ax.set_xlabel('t0 - ICU admission time (h; clipped -24 to 72)')
    ax.set_ylabel('Patients')
    ax.set_title(f't0 near ICU admission: median {dt_h.median():.1f} h; '

                 f'{100 * (dt_h < 0).mean():.1f}% before ICU')

    ax = axes[1]
    groups = ['Vital signs', 'Labs (cr/wbc/bili)', 'Lactate']
    pre = [av['vit_pre'].mean(), av['lab_pre'].mean(), av['lac_pre'].mean()]
    post = [av['vit_post'].mean(), av['lab_post'].mean(), av['lac_post'].mean()]
    x = np.arange(len(groups))
    ax.bar(x - 0.18, pre, 0.36, label='[t0-24h, t0)', color='#c0504d')
    ax.bar(x + 0.18, post, 0.36, label='(t0, t0+24h]', color='#4472a8')
    for i, (p1, p2) in enumerate(zip(pre, post)):
        ax.text(i - 0.18, p1 + 0.02, f'{p1:.0%}', ha='center', fontsize=9)
        ax.text(i + 0.18, p2 + 0.02, f'{p2:.0%}', ha='center', fontsize=9)
    ax.set_xticks(x, groups)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Availability')
    ax.set_title('Pre-t0 data are physically sparse; post-t0 data accumulate naturally')
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / 'figures' / 'timing_structure.png', dpi=200)
    print("\n输出 -> results/timing_structure.csv, results/figures/timing_structure.png")


if __name__ == "__main__":
    main()
