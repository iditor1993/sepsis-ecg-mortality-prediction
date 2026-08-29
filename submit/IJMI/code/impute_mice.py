"""impute_mice.py — 缺失数据多重插补（SAP 第十章，Python 实现，m=20）。

主要方案：链式方程多重插补（MICE），m=20。
  - 需插补变量：LR 模型变量中仅乳酸缺失（20.5%）；其余 LR 变量完整
  - 插补模型预测矩阵：全部分析变量（SOFA 总分与分项、qSOFA/NEWS/MEWS、
    全部协变量、ECG 潜向量 z1-z16）+ 结局 Nelson-Aalen 累积风险估计量（NAE）
  - M1+ 85 列汇总特征不纳入插补模型（仅供 XGBoost 使用，原生处理 NaN）；
    ECG 潜向量无缺失，仅作预测因子（SAP 第十章）
  - 插补器仅在 train 子集拟合，transform 应用于 train/tune/test/temporal
    （避免从验证/测试分布泄漏）

输出：data/mice_lactate.parquet（stay_id, subset, lactate_m01..lactate_m20）
"""

from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import NelsonAalenFitter
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED = 20260823
M = 20

SCORE_COLS = ["sofa_score", "sofa_respiration", "sofa_coagulation", "sofa_liver",
              "sofa_cardiovascular", "sofa_cns", "sofa_renal"]
COV_COLS = ["admission_age", "male", "charlson_comorbidity_index",
            "admission_emergency", "pre_icu_los_h", "mech_vent_24h", "vaso_24h"]
Z_COLS = [f"z{i}" for i in range(1, 17)]


def main() -> None:
    datadir = REPO_ROOT / "data"
    dev = pd.read_parquet(datadir / "features_dev.parquet")
    temporal = pd.read_parquet(datadir / "features_temporal.parquet")
    scores = pd.read_parquet(datadir / "clinical_scores.parquet")
    out = pd.read_parquet(datadir / "outcomes.parquet")

    df = pd.concat([dev, temporal], ignore_index=True)
    df = df.merge(scores[["stay_id", "qsofa", "news", "mews"]], on="stay_id", how="left")
    df = df.merge(out[["stay_id", "days_to_death"]], on="stay_id", how="left")
    df["male"] = (df["gender"] == "M").astype(int)
    df = pd.get_dummies(df, columns=["infection_site"], prefix="site", dtype=int)

    # ---- Nelson-Aalen 累积风险估计量（train 拟合）----
    time = np.where(df["death_28d"], df["days_to_death"], 28).astype(float)
    event = df["death_28d"].astype(int).to_numpy()
    tr_mask = (df["subset"] == "train").to_numpy()
    naf = NelsonAalenFitter()
    naf.fit(time[tr_mask], event[tr_mask])
    df["nae"] = np.asarray(naf.cumulative_hazard_at_times(time)).ravel()

    pred_cols = (SCORE_COLS + ["qsofa", "news", "mews"] + COV_COLS + Z_COLS
                 + [c for c in df.columns if c.startswith("site_")] + ["nae"])
    complete = df[pred_cols].notna().all(axis=1)
    print(f"预测矩阵完整行: {int(complete.sum()):,}/{len(df):,}"
          f"（不完整 {int((~complete).sum())} 行回退为中位数填充预测值）")

    results = {}
    df["log_lactate"] = np.log(df["lactate"])  # 对数尺度插补（乳酸右偏且须为正）
    for k in range(1, M + 1):
        imp = IterativeImputer(max_iter=10, sample_posterior=True,
                               random_state=SEED + k)
        imp.fit(df.loc[tr_mask & complete, pred_cols + ["log_lactate"]])
        cols = pred_cols + ["log_lactate"]
        x = df[cols].copy()
        # 极少数预测因子不完整行：先以 train 中位数填补预测矩阵（仅 transform 用）
        med = df.loc[tr_mask, pred_cols].median()
        x[pred_cols] = x[pred_cols].fillna(med)
        results[f"lactate_m{k:02d}"] = np.exp(imp.transform(x[cols])[:, -1])
        if k % 5 == 0 or k == 1:
            print(f"  插补 {k}/{M} 完成", flush=True)

    out_df = df[["subject_id", "stay_id", "subset"]].copy()
    for name, vals in results.items():
        out_df[name] = vals
    out_df.to_parquet(datadir / "mice_lactate.parquet", index=False)

    print(f"\n乳酸插补完成（m={M}）-> data/mice_lactate.parquet")
    orig = df.loc[df["lactate"].notna(), "lactate"]
    imp_vals = out_df.loc[df["lactate"].isna(),
                          [c for c in out_df.columns if c.startswith("lactate_m")]]
    print(f"原始乳酸: 中位 {orig.median():.2f} (IQR {orig.quantile(0.25):.2f}-"
          f"{orig.quantile(0.75):.2f})")
    print(f"插补乳酸: 中位 {imp_vals.median().median():.2f}，"
          f"20 套间标准差（插补不确定性）{imp_vals.std(axis=1).mean():.3f}")


if __name__ == "__main__":
    main()
