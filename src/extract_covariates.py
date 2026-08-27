"""extract_covariates.py — 模型协变量提取（SAP 5.2/5.3，W3）。

执行 sql/04_covariates.sql（对 cohort_ecg 最终分析队列），
输出 data/covariates.parquet 并打印缺失情况与分布摘要。
"""

from pathlib import Path

import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = "E:/clinical_research/MIMIC_IV_3.1/mimic_iv_3_1.duckdb"


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="cohort_ecg.parquet")
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    datadir = REPO_ROOT / "data"
    cohort = pd.read_parquet(datadir / args.cohort)

    con = duckdb.connect(DEFAULT_DB, read_only=True)
    try:
        con.register("cohort", cohort)
        sql = (REPO_ROOT / "sql" / "04_covariates.sql").read_text(encoding="utf-8")
        cov = con.sql(sql).df()
    finally:
        con.close()

    assert cov["stay_id"].is_unique and len(cov) == len(cohort)
    cov.to_parquet(datadir / f"covariates{args.suffix}.parquet", index=False)

    n = len(cov)
    print(f"协变量提取完成（N={n:,}）:")
    print("\n[乳酸]")
    n_lac = int(cov["lactate"].notna().sum())
    print(f"  可得 {n_lac:,} ({n_lac / n:.1%})，缺失 {n - n_lac:,}")
    print(f"  窗口分布: {cov['lactate_window'].value_counts().to_dict()}")
    print("  中位数 {:.1f} mmol/L (IQR {:.1f}-{:.1f})".format(
        cov["lactate"].median(), cov["lactate"].quantile(0.25), cov["lactate"].quantile(0.75)))
    print("\n[感染部位]")
    print(cov["infection_site"].value_counts().to_string())
    print("\n[入 ICU 前住院时长] 中位数 {:.1f} h (IQR {:.1f}-{:.1f})".format(
        cov["pre_icu_los_h"].median(), cov["pre_icu_los_h"].quantile(0.25),
        cov["pre_icu_los_h"].quantile(0.75)))
    print(f"\n[急诊入院] {cov['admission_emergency'].mean():.1%}")
    print(f"\n输出: {datadir / 'covariates.parquet'}")


if __name__ == "__main__":
    main()
