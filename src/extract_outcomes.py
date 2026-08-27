"""extract_outcomes.py — 结局判定与治疗强度提取（SAP 第四章 / 5.3，W2）。

对 cohort_base（31,857 例，含 ECG 不可链接者）执行：
  sql/02_outcomes.sql              -> data/outcomes.parquet
  sql/03_treatment_intensity.sql   -> data/treatment_intensity.parquet

数据库只读连接；cohort 经 duckdb.register 注册为视图供 SQL 引用。
注意：本脚本生成结局标签，仅用于 A1a 描述性对照与后续特征矩阵；
测试集标签在 ΔAUC 功效核算通过前不得用于任何模型评估（SAP 第六章）。
"""

from pathlib import Path

import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = "E:/clinical_research/MIMIC_IV_3.1/mimic_iv_3_1.duckdb"


def main() -> None:
    datadir = REPO_ROOT / "data"
    cohort = pd.read_parquet(datadir / "cohort_base.parquet")

    con = duckdb.connect(DEFAULT_DB, read_only=True)
    try:
        con.register("cohort", cohort)

        sql = (REPO_ROOT / "sql" / "02_outcomes.sql").read_text(encoding="utf-8")
        outcomes = con.sql(sql).df()
        assert outcomes["stay_id"].is_unique
        outcomes.to_parquet(datadir / "outcomes.parquet", index=False)

        sql = (REPO_ROOT / "sql" / "03_treatment_intensity.sql").read_text(encoding="utf-8")
        intensity = con.sql(sql).df()
        assert intensity["stay_id"].is_unique
        intensity.to_parquet(datadir / "treatment_intensity.parquet", index=False)
    finally:
        con.close()

    n = len(outcomes)
    print("=" * 60)
    print("结局事件率（cohort_base，N={:,}，含 ECG 不可链接者）".format(n))
    print("=" * 60)
    for col, label in [
        ("death_28d", "主要结局：28 天全因死亡"),
        ("death_inhosp", "次要 1：院内死亡"),
        ("death_90d", "次要 2：90 天死亡"),
        ("death_icu", "次要 3：ICU 内死亡"),
        ("shock_28d", "次要 4：28 天脓毒性休克"),
    ]:
        k = int(outcomes[col].sum())
        print(f"  {label}: {k:,} ({k / n:.1%})")
    print("\n治疗强度（t0±24h）:")
    for col in ["mech_vent_24h", "vaso_24h"]:
        k = int(intensity[col].sum())
        print(f"  {col}: {k:,} ({k / n:.1%})")
    print("\nCharlson 指数: 中位数 {:.0f} (IQR {:.0f}-{:.0f})，缺失 {:,}".format(
        intensity["charlson_comorbidity_index"].median(),
        intensity["charlson_comorbidity_index"].quantile(0.25),
        intensity["charlson_comorbidity_index"].quantile(0.75),
        int(intensity["charlson_comorbidity_index"].isna().sum()),
    ))
    print("\ndays_to_death 最小值: {}（应 >=0）".format(outcomes["days_to_death"].min()))
    print("输出: data/outcomes.parquet, data/treatment_intensity.parquet")


if __name__ == "__main__":
    main()
