"""extract_cohort.py — Sepsis-3 队列提取与分级计数（SAP 3.2 / 3.3 节，W1）。

执行 sql/01_cohort_sepsis3.sql，输出：
  data/cohort_episodes.parquet  全部 Sepsis-3 发作（stay_id 级），含逐级排除标志
  data/cohort_base.parquet      最终入组队列（in_cohort = True）
  data/cohort_flow.csv          流程图分级计数（回填图 1 / 表 3-1）

数据库以只读方式连接；本脚本不触碰结局标签相关分析（结局判定另行执行）。
"""

import argparse
from pathlib import Path

import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = "E:/clinical_research/MIMIC_IV_3.1/mimic_iv_3_1.duckdb"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sepsis-3 队列提取（SAP 3.2/3.3）")
    parser.add_argument("--db", default=DEFAULT_DB, help="DuckDB 数据库路径")
    parser.add_argument("--outdir", default=str(REPO_ROOT / "data"), help="输出目录")
    parser.add_argument(
        "--sql", default=str(REPO_ROOT / "sql" / "01_cohort_sepsis3.sql"), help="队列提取 SQL"
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sql = Path(args.sql).read_text(encoding="utf-8")

    con = duckdb.connect(args.db, read_only=True)
    try:
        episodes = con.sql(sql).df()
        n_patients_db = con.sql("SELECT COUNT(*) FROM main.patients").fetchone()[0]
        n_admissions_db = con.sql("SELECT COUNT(*) FROM main.admissions").fetchone()[0]
    finally:
        con.close()

    # ---- 完整性检查 ----
    assert episodes["stay_id"].is_unique, "stay_id 不唯一，sepsis3 表粒度异常"
    assert episodes["t0"].notna().all(), "存在 t0 缺失的发作"
    n_join_missing = episodes["hadm_id"].isna().sum()
    if n_join_missing:
        raise RuntimeError(f"{n_join_missing} 条发作未能关联 icustay_detail/admissions")

    episodes.to_parquet(outdir / "cohort_episodes.parquet", index=False)

    cohort = episodes[episodes["in_cohort"]].copy()
    cohort.to_parquet(outdir / "cohort_base.parquet", index=False)

    # ---- 流程图分级计数（逐级累计排除）----
    stages = []
    prev = None
    for label, mask in [
        ("sepsis3_episodes", pd.Series(True, index=episodes.index)),
        ("age_ge_18", episodes["age_ok"]),
        ("first_episode", episodes["first_episode_ok"]),
        ("icu_ge_6h", episodes["first_episode_ok"] & episodes["icu6h_ok"]),
        ("alive_at_t0_final", episodes["in_cohort"]),
    ]:
        sub = episodes[mask]
        row = {
            "stage": label,
            "n_stays": len(sub),
            "n_subjects": sub["subject_id"].nunique(),
            "excluded_at_step": (prev - len(sub)) if prev is not None else 0,
        }
        stages.append(row)
        prev = len(sub)
    flow = pd.DataFrame(stages)
    flow.to_csv(outdir / "cohort_flow.csv", index=False)

    # ---- 控制台摘要 ----
    print("=" * 60)
    print("MIMIC-IV v3.1 Sepsis-3 队列提取（SAP 3.2/3.3，ECG 排除前）")
    print("=" * 60)
    print(f"数据库全部住院患者: {n_patients_db:,} 例 / {n_admissions_db:,} 次住院")
    print("\n[流程图分级计数]")
    print(flow.to_string(index=False))

    print("\n[入组队列概况] cohort_base.parquet")
    print(f"发作数(=患者数，每人首次): {len(cohort):,}")
    print("\n时段分布（anchor_year_group -> cohort_period）:")
    print(cohort["cohort_period"].value_counts().to_string())
    print("\n年龄: 中位数 {:.1f} 岁 (IQR {:.1f}-{:.1f})".format(
        cohort["admission_age"].median(),
        cohort["admission_age"].quantile(0.25),
        cohort["admission_age"].quantile(0.75),
    ))
    print("性别: " + ", ".join(
        f"{k}={v:,}({v / len(cohort):.1%})" for k, v in cohort["gender"].value_counts().items()
    ))
    print("ICU 时长: 中位数 {:.2f} 天 (IQR {:.2f}-{:.2f})".format(
        cohort["los_icu"].astype(float).median(),
        cohort["los_icu"].astype(float).quantile(0.25),
        cohort["los_icu"].astype(float).quantile(0.75),
    ))
    print("SOFA(t0): 中位数 {:.0f} (IQR {:.0f}-{:.0f})".format(
        cohort["sofa_score"].median(),
        cohort["sofa_score"].quantile(0.25),
        cohort["sofa_score"].quantile(0.75),
    ))
    print(f"\nt0 范围(偏移后时间): {cohort['t0'].min()} ~ {cohort['t0'].max()}")
    print("\n输出文件:")
    for f in ["cohort_episodes.parquet", "cohort_base.parquet", "cohort_flow.csv"]:
        print(f"  {outdir / f}")


if __name__ == "__main__":
    main()
