"""extract_pre_features.py — 纯预测窗 [t0-24h, t0] 特征提取（额外分析）。

执行 sql/07_m1plus_pre.sql，并由 pre 窗口最差值计算 qSOFA/NEWS/MEWS
（规则与 extract_clinical_features.py 一致）：
  data/pre_features.parquet：17 通道 x 5 统计量 + lactate_pre + pre 三评分
                              + mv_pre / vaso_pre
注意：pre 窗口 SOFA 总分沿用 data/sofa_t0.parquet（逐时表 t0 时刻
      sofa_24hours，天然为 t0 前信息），不在本脚本内。
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from extract_clinical_features import mews, news2

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = "E:/clinical_research/MIMIC_IV_3.1/mimic_iv_3_1.duckdb"


def main() -> None:
    datadir = REPO_ROOT / "data"
    cohort = pd.read_parquet(datadir / "cohort_ecg.parquet")

    con = duckdb.connect(DEFAULT_DB, read_only=True)
    try:
        con.register("cohort", cohort)
        df = con.sql((REPO_ROOT / "sql" / "07_m1plus_pre.sql").read_text(encoding="utf-8")).df()
    finally:
        con.close()

    assert "stay_id_1" not in df.columns and df["stay_id"].is_unique

    # pre 评分（把 pre 列名映射到评分函数期望的列名）
    m = df.rename(columns={"gcs_min_pre": "gcs_min", "fio2_max_pre": "fio2_max"})
    df["qsofa_pre"] = ((m["sbp_min"] <= 100).astype(int)
                       + (m["rr_max"] >= 22).astype(int)
                       + (m["gcs_min"] < 15).astype(int))
    df["news_pre"] = news2(m)
    df["mews_pre"] = mews(m)

    df.to_parquet(datadir / "pre_features.parquet", index=False)

    n = len(df)
    stat_cols = [c for c in df.columns if c.endswith("_mean")]
    print(f"pre 特征提取完成（N={n:,}，{df.shape[1]} 列）")
    print(f"乳酸 lactate_pre 可得 {df['lactate_pre'].notna().mean():.1%}"
          f"（主分析 t0±6h/±24h 口径 80.0%）")
    print("各通道缺失率（mean 列）:")
    print(df[stat_cols].isna().mean().map("{:.1%}".format).to_string())
    print(f"\nmv_pre {df['mv_pre'].mean():.1%}；vaso_pre {df['vaso_pre'].mean():.1%}")
    print(f"NEWS_pre 中位 {df['news_pre'].median():.0f}；MEWS_pre 中位 {df['mews_pre'].median():.0f}")
    print(f"\n输出: {datadir / 'pre_features.parquet'}")


if __name__ == "__main__":
    main()
