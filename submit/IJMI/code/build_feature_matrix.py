"""build_feature_matrix.py — 特征矩阵组装与数据锁定（SAP 1.3 / 9.10，W3）。

合并：cohort_ecg（标识/t0/SOFA/人口学）+ Track A（z1-z16）+ Track B（tb1-tb32）
      + outcomes（结局标签）+ treatment_intensity + covariates + splits
输出：
  data/features_dev.parquet      开发队列特征矩阵（14,780 例，含 subset 标签）
  data/features_temporal.parquet 时间外推验证队列（1,719 例）
  data/DATA_LOCK.md              数据锁定记录（SHA-256 哈希）

锁定即 SAP 1.3 的数据锁定时点：此后 SAP 实质性修改须以偏离形式记录；
测试集结局标签在 ΔAUC 功效核算通过前不得解盲（SAP 第六章）。
"""

import hashlib
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

ID_COLS = ["subject_id", "hadm_id", "stay_id", "t0", "cohort_period", "subset"]
OUTCOME_COLS = ["death_28d", "death_inhosp", "death_90d", "death_icu", "shock_28d"]
CLIN_COLS = [
    "gender", "admission_age", "anchor_year_group",
    "sofa_score", "sofa_respiration", "sofa_coagulation", "sofa_liver",
    "sofa_cardiovascular", "sofa_cns", "sofa_renal",
    "admittime", "dischtime", "los_hospital", "icu_intime", "icu_outtime", "los_icu",
    "admission_type", "hospital_expire_flag", "dod",
    "ecg_time", "signed_t0_diff_h", "study_id",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    datadir = REPO_ROOT / "data"

    base = pd.read_parquet(datadir / "cohort_ecg.parquet")
    ta = pd.read_parquet(datadir / "features_trackA.parquet")
    tb = pd.read_parquet(datadir / "features_trackB.parquet")
    out = pd.read_parquet(datadir / "outcomes.parquet")
    inten = pd.read_parquet(datadir / "treatment_intensity.parquet")
    cov = pd.read_parquet(datadir / "covariates.parquet")
    splits = pd.read_csv(datadir / "splits.csv")

    df = (
        base[CLIN_COLS + ["subject_id", "stay_id", "cohort_period"]]
        .merge(splits, on=["subject_id", "stay_id"], how="left")
        .merge(out[["stay_id"] + OUTCOME_COLS], on="stay_id", how="left")
        .merge(inten[["stay_id", "mech_vent_24h", "vaso_24h",
                      "charlson_comorbidity_index"]], on="stay_id", how="left")
        .merge(cov[["stay_id", "lactate", "lactate_window", "infection_site",
                    "pre_icu_los_h", "admission_emergency"]], on="stay_id", how="left")
        .merge(ta.drop(columns=["load_ok"]), on=["subject_id", "stay_id"], how="left")
        .merge(tb.drop(columns=["load_ok"]), on=["subject_id", "stay_id"], how="left")
    )
    # 列去重（CLIN_COLS 与 ID_COLS 可能有 subject_id/stay_id 重复选择）
    df = df.loc[:, ~df.columns.duplicated()]

    n_expect = len(base)
    assert len(df) == n_expect, f"行数异常: {len(df)} != {n_expect}"
    assert df["stay_id"].is_unique

    z_cols = [f"z{i}" for i in range(1, 17)]
    tb_cols = [f"tb{i}" for i in range(1, 33)]
    ordered = ([c for c in ID_COLS if c in df.columns] + OUTCOME_COLS
               + [c for c in df.columns
                  if c not in ID_COLS + OUTCOME_COLS + z_cols + tb_cols]
               + z_cols + tb_cols)
    df = df[ordered]

    dev = df[df["cohort_period"] == "dev_2008_2016"]
    temporal = df[df["cohort_period"] == "temporal_2017_2019"]
    dev.to_parquet(datadir / "features_dev.parquet", index=False)
    temporal.to_parquet(datadir / "features_temporal.parquet", index=False)

    # ---- 数据锁定 ----
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    locks = []
    for f in ["features_dev.parquet", "features_temporal.parquet", "splits.csv"]:
        locks.append((f, sha256(datadir / f)))

    lock_md = ["# DATA_LOCK — 数据锁定记录（SAP 1.3 / 9.10 节）", ""]
    lock_md.append(f"**数据锁定时点：{now}**（特征矩阵生成并记录 SHA-256 哈希）")
    lock_md.append("")
    lock_md.append("锁定前任何模型不得接触测试集结局标签；锁定后 SAP 实质性修改")
    lock_md.append("仅允许以偏离形式记录（SAP 第十三章）。")
    lock_md.append("")
    lock_md.append("| 文件 | SHA-256 | 锁定时间 | 操作人 |")
    lock_md.append("|---|---|---|---|")
    for f, h in locks:
        lock_md.append(f"| {f} | `{h}` | {now} | iditor1993 |")
    lock_md.append("")
    lock_md.append("## 锁定前已完成的质量门槛")
    lock_md.append("")
    lock_md.append("- [x] Sepsis-3 队列判定与排除流程（cohort_flow.csv）")
    lock_md.append("- [x] ECG 链接与质控（qc_config.yaml，PROPOSED 阈值冻结为本版本）")
    lock_md.append("- [x] 数据集划分（种子 20260823，标签解盲前冻结）")
    lock_md.append("- [ ] ΔAUC 功效核算（power_delong_sim.py）——测试集解盲前执行")
    (datadir / "DATA_LOCK.md").write_text("\n".join(lock_md), encoding="utf-8")

    # ---- 摘要 ----
    print("=" * 60)
    print(f"特征矩阵已生成并锁定（{now}）")
    print("=" * 60)
    print(f"features_dev:      {dev.shape}  (train/tune/test = "
          f"{(dev['subset'] == 'train').sum()}/{(dev['subset'] == 'tune').sum()}"
          f"/{(dev['subset'] == 'test').sum()})")
    print(f"features_temporal: {temporal.shape}")
    print("\n[关键变量缺失率（dev）]")
    key = ["lactate", "charlson_comorbidity_index", "pre_icu_los_h"] + z_cols[:2] + tb_cols[:2]
    print(dev[key].isna().mean().map("{:.1%}".format).to_string())
    print("\n[结局事件率]")
    for name, d in [("dev", dev), ("temporal", temporal)]:
        print(f"  {name}: 28d={d['death_28d'].mean():.1%} 90d={d['death_90d'].mean():.1%}"
              f" 院内={d['death_inhosp'].mean():.1%} 休克={d['shock_28d'].mean():.1%}")
    print("\n[SHA-256]")
    for f, h in locks:
        print(f"  {f}: {h}")
    print(f"\n锁定记录 -> {datadir / 'DATA_LOCK.md'}")


if __name__ == "__main__":
    main()
