"""extract_clinical_features.py — 临床评分与 M1+ 汇总特征提取（SAP 5.2/8.2，W4）。

执行 sql/05_clinical_vitals.sql 与 sql/06_m1plus_labs.sql，并据 t0±24h
最差值计算急诊常规评分：
  qSOFA：SBP≤100 + RR≥22 + GCS<15
  NEWS2：RR/SpO2/氧疗/SBP/HR/体温/意识（氧疗以 FiO2>21% 代理；意识以 GCS<15 代理）
  MEWS ：SBP/HR/RR/体温/AVPU（AVPU 由 GCS 近似：15=A, 12-14=V, 9-11=P, ≤8=U）
规则：某组分无窗内测量时按 0 分计（EHR 常规约定，视作无异常记录），
      各评分组分完整度另列输出。

输出：
  data/clinical_scores.parquet   subject_id, stay_id, qsofa, news, mews 及组分值
  data/m1plus_features.parquet   subject_id, stay_id, 85 列汇总特征
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = "E:/clinical_research/MIMIC_IV_3.1/mimic_iv_3_1.duckdb"


def news2(df: pd.DataFrame) -> pd.Series:
    s = pd.Series(0, index=df.index, dtype=float)
    rr = df["rr_max"]
    s += np.select([rr <= 8, rr <= 11, rr <= 20, rr <= 24, rr >= 25], [3, 1, 0, 2, 3], default=0)
    sp = df["spo2_min"]
    s += np.select([sp <= 91, sp <= 93, sp <= 95, sp >= 96], [3, 2, 1, 0], default=0)
    s += (df["fio2_max"] > 21).astype(float) * 2
    sb = df["sbp_min"]
    s += np.select([sb <= 90, sb <= 100, sb <= 110, sb < 220, sb >= 220],
                   [3, 2, 1, 0, 3], default=0)
    hr = df["hr_max"]
    s += np.select([hr <= 40, hr <= 50, hr <= 90, hr <= 110, hr <= 130, hr >= 131],
                   [3, 1, 0, 1, 2, 3], default=0)
    # 体温取高低两端较差者
    t_lo = np.select([df["temp_min"] <= 35, df["temp_min"] <= 36], [3, 1], default=0)
    t_hi = np.select([df["temp_max"] >= 39.1, df["temp_max"] >= 38.1], [2, 1], default=0)
    s += np.maximum(t_lo, t_hi)
    s += (df["gcs_min"] < 15).astype(float) * 3
    return s


def mews(df: pd.DataFrame) -> pd.Series:
    s = pd.Series(0, index=df.index, dtype=float)
    sb = df["sbp_min"]
    s += np.select([sb <= 70, sb <= 80, sb <= 100, sb < 200, sb >= 200],
                   [3, 2, 1, 0, 2], default=0)
    hr = df["hr_max"]
    s += np.select([hr <= 40, hr <= 50, hr <= 100, hr <= 110, hr <= 129, hr >= 130],
                   [2, 1, 0, 1, 2, 3], default=0)
    rr = df["rr_max"]
    s += np.select([rr < 9, rr <= 14, rr <= 20, rr <= 29, rr >= 30],
                   [2, 0, 1, 2, 3], default=0)
    t_lo = (df["temp_min"] < 35).astype(float) * 2
    t_hi = (df["temp_max"] >= 38.5).astype(float) * 2
    s += np.maximum(t_lo, t_hi)
    g = df["gcs_min"]
    s += np.select([g == 15, g >= 12, g >= 9, g <= 8], [0, 1, 2, 3], default=0)
    return s


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="cohort_ecg.parquet")
    parser.add_argument("--suffix", default="", help="输出文件名后缀（如 _all）")
    args = parser.parse_args()
    datadir = REPO_ROOT / "data"
    cohort = pd.read_parquet(datadir / args.cohort)

    con = duckdb.connect(DEFAULT_DB, read_only=True)
    try:
        con.register("cohort", cohort)
        vit = con.sql((REPO_ROOT / "sql" / "05_clinical_vitals.sql").read_text(encoding="utf-8")).df()
        lab = con.sql((REPO_ROOT / "sql" / "06_m1plus_labs.sql").read_text(encoding="utf-8")).df()
    finally:
        con.close()

    # 守卫：sql 输出不得含重复的 id 列（历史 bug：piv.* 带入 stay_id_1）
    for bad in [c for c in vit.columns if c.endswith("_1")]:
        raise RuntimeError(f"sql/05 输出含重复列 {bad}，请检查 SQL")
    assert vit["stay_id"].is_unique and lab["stay_id"].is_unique

    # ---- 评分 ----
    scores = vit[["subject_id", "stay_id"]].copy()
    scores["qsofa"] = ((vit["sbp_min"] <= 100).astype(int)
                       + (vit["rr_max"] >= 22).astype(int)
                       + (vit["gcs_min"] < 15).astype(int))
    scores["news"] = news2(vit)
    scores["mews"] = mews(vit)
    comp_cols = ["sbp_min", "rr_max", "hr_max", "spo2_min", "temp_min", "temp_max",
                 "gcs_min", "fio2_max"]
    scores["n_missing_components"] = vit[comp_cols].isna().sum(axis=1)
    scores = scores.merge(vit[["stay_id"] + comp_cols], on="stay_id")
    scores.to_parquet(datadir / f"clinical_scores{args.suffix}.parquet", index=False)

    # ---- M1+ 85 列特征 ----
    vital_stats = [c for c in vit.columns
                   if c not in ("subject_id", "stay_id", "gcs_min", "fio2_max")]
    m1p = vit[["subject_id", "stay_id"] + vital_stats].merge(lab, on="stay_id", how="left")
    m1p.to_parquet(datadir / f"m1plus_features{args.suffix}.parquet", index=False)

    n = len(cohort)
    print(f"临床特征提取完成（N={n:,}）")
    print(f"\n[评分] qSOFA≥2: {(scores['qsofa'] >= 2).mean():.1%}；"
          f"NEWS 中位 {scores['news'].median():.0f} (IQR {scores['news'].quantile(0.25):.0f}-"
          f"{scores['news'].quantile(0.75):.0f})；MEWS 中位 {scores['mews'].median():.0f}")
    print(f"评分组分缺失分布: {scores['n_missing_components'].value_counts().sort_index().to_dict()}")
    print(f"\n[M1+ 特征] {m1p.shape[1] - 2} 列；各通道缺失率：")
    miss = m1p[[c for c in m1p.columns if c.endswith("_mean")]].isna().mean()
    print(miss.map("{:.1%}".format).to_string())
    print(f"\n输出: data/clinical_scores.parquet, data/m1plus_features.parquet")


if __name__ == "__main__":
    main()
