"""a1b_availability.py — A1b ECG 可得性指示变量对照（SAP 9.7，V1.2 新增；即 S11）。

问题：ECG 增量是否由开单行为（可得性）而非波形内容驱动？
设计：
  1) 全队列（cohort_base，31,857 例，含 ECG 不可链接/不合格者）。
     分析专用划分：dev 时段（2008-2016）按患者级 70/15/15（种子 20260823）；
     锁定测试集患者（2,217 例）从 A/B 两模型的训练与调优中剔除，
     以保证 M3 对比评估无泄漏。
  2) 模型 A：M1+ 特征（85 列汇总 + 乳酸 + 协变量）XGBoost（沿用主分析最佳配置）；
     模型 B：A + 二分类 ECG 可得性指示变量（ecg_available）。
  3) 测试集：ΔAUC(B-A) + bootstrap CI —— 可得性的独立判别贡献；
     锁定测试集（全部 ECG 可得）：ΔAUC(M3 - B) + bootstrap CI ——
     控制可得性后 M3 波形内容是否仍有增量。
输出 results/a1b_availability.csv。
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED = 20260823
N_BOOT = 2000
COV_COLS = ["admission_age", "male", "charlson_comorbidity_index",
            "admission_emergency", "pre_icu_los_h", "mech_vent_24h", "vaso_24h"]
BEST_CFG = dict(max_depth=4, eta=0.05, min_child_weight=1)


def boot_delta(y, pa, pb, rng):
    n, diffs = len(y), []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        if y[idx].sum() in (0, n):
            continue
        diffs.append(roc_auc_score(y[idx], pa[idx]) - roc_auc_score(y[idx], pb[idx]))
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main() -> None:
    import xgboost as xgb

    datadir = REPO_ROOT / "data"
    results_dir = REPO_ROOT / "results"
    rng = np.random.default_rng(SEED)

    base = pd.read_parquet(datadir / "cohort_base.parquet")
    link = pd.read_parquet(datadir / "ecg_linked.parquet",
                           columns=["stay_id", "ecg_available"])
    out = pd.read_parquet(datadir / "outcomes.parquet")
    inten = pd.read_parquet(datadir / "treatment_intensity.parquet")
    cov = pd.read_parquet(datadir / "covariates_all.parquet")
    m1p = pd.read_parquet(datadir / "m1plus_features_all.parquet")

    df = (base[["subject_id", "stay_id", "admission_age", "gender", "anchor_year_group"]]
          .merge(link, on="stay_id", how="left")
          .merge(out[["stay_id", "death_28d"]], on="stay_id", how="left")
          .merge(inten, on=["subject_id", "stay_id"], how="left")
          .merge(cov[["stay_id", "lactate", "infection_site", "pre_icu_los_h",
                      "admission_emergency"]], on="stay_id", how="left"))
    df["male"] = (df["gender"] == "M").astype(int)
    df["ecg_available"] = df["ecg_available"].astype(float).fillna(0.0)
    df = pd.get_dummies(df, columns=["infection_site"], prefix="site", dtype=int)
    m1p_feats = [c for c in m1p.columns if c not in ("subject_id", "stay_id")]
    df = df.merge(m1p[["stay_id"] + m1p_feats], on="stay_id", how="left")

    site_cols = sorted(c for c in df.columns if c.startswith("site_"))
    feats = m1p_feats + ["lactate"] + COV_COLS + site_cols

    # 分析专用划分（仅 dev 时段；种子与主划分一致）
    dev = df[df["anchor_year_group"].isin(
        ["2008 - 2010", "2011 - 2013", "2014 - 2016"])].copy()
    perm = np.random.RandomState(SEED).permutation(len(dev))
    n_tr, n_tu = int(round(len(dev) * 0.7)), int(round(len(dev) * 0.15))
    dev["a1b_subset"] = "test"
    dev.iloc[perm[:n_tr], dev.columns.get_loc("a1b_subset")] = "train"
    dev.iloc[perm[n_tr:n_tr + n_tu], dev.columns.get_loc("a1b_subset")] = "tune"

    locked_test = pd.read_csv(datadir / "splits.csv")
    locked_test = set(locked_test.loc[locked_test["subset"] == "test", "stay_id"])
    no_leak = ~dev["stay_id"].isin(locked_test)  # 锁定测试集不参与训练/调优

    tr = (dev["a1b_subset"] == "train") & no_leak
    tu = (dev["a1b_subset"] == "tune") & no_leak
    te = dev["a1b_subset"] == "test"
    ytr, ytu = (dev.loc[m, "death_28d"].astype(int).to_numpy() for m in (tr, tu))
    yte = dev.loc[te, "death_28d"].astype(int).to_numpy()
    print(f"A1b 队列: dev {len(dev):,}（ECG 可得 {dev['ecg_available'].mean():.1%}）；"
          f"train {tr.sum():,} / tune {tu.sum():,} / test {te.sum():,}")

    def fit(xtr_df, y, xtu_df, yt):
        m = xgb.XGBClassifier(n_estimators=5000, subsample=0.8, colsample_bytree=0.8,
                              tree_method="hist", eval_metric="auc",
                              random_state=SEED, early_stopping_rounds=50,
                              n_jobs=8, **BEST_CFG)
        m.fit(xtr_df, y, eval_set=[(xtu_df, yt)], verbose=False)
        return m

    m_a = fit(dev.loc[tr, feats], ytr, dev.loc[tu, feats], ytu)
    feats_b = feats + ["ecg_available"]
    m_b = fit(dev.loc[tr, feats_b], ytr, dev.loc[tu, feats_b], ytu)

    p_a = m_a.predict_proba(dev.loc[te, feats])[:, 1]
    p_b = m_b.predict_proba(dev.loc[te, feats_b])[:, 1]
    auc_a, auc_b = roc_auc_score(yte, p_a), roc_auc_score(yte, p_b)
    d_ind = boot_delta(yte, p_b, p_a, rng)
    print(f"[A1b-1] M1+ AUC {auc_a:.4f}；+可得性指示 AUC {auc_b:.4f}；"
          f"ΔAUC {d_ind[0]:+.4f} ({d_ind[1]:+.4f}~{d_ind[2]:+.4f})")

    # 锁定测试集（全部 ECG 可得）：M3 vs B
    pred_te = pd.read_parquet(results_dir / "test_predictions.parquet")
    lt = dev[dev["stay_id"].isin(locked_test)].copy()
    lt = lt.merge(pred_te[["stay_id", "M3_cal"]], on="stay_id", how="left")
    ylt = lt["death_28d"].astype(int).to_numpy()
    p_b_lt = m_b.predict_proba(lt[feats_b])[:, 1]
    d_m3 = boot_delta(ylt, lt["M3_cal"].to_numpy(), p_b_lt, rng)
    print(f"[A1b-2] 锁定测试集（n={len(lt):,}）：M3 AUC "
          f"{roc_auc_score(ylt, lt['M3_cal']):.4f}；M1++指示 AUC "
          f"{roc_auc_score(ylt, p_b_lt):.4f}；ΔAUC(M3-B) {d_m3[0]:+.4f} "
          f"({d_m3[1]:+.4f}~{d_m3[2]:+.4f})")

    res = pd.DataFrame([
        {"item": "可得性指示变量的独立贡献（全队列）", "auc_base": auc_a,
         "auc_with_ind": auc_b, "delta": d_ind[0], "ci_lo": d_ind[1], "ci_hi": d_ind[2]},
        {"item": "M3 vs M1++指示（锁定测试集）",
         "auc_base": roc_auc_score(ylt, p_b_lt),
         "auc_with_ind": roc_auc_score(ylt, lt["M3_cal"]),
         "delta": d_m3[0], "ci_lo": d_m3[1], "ci_hi": d_m3[2]},
    ])
    res.to_csv(results_dir / "a1b_availability.csv", index=False)
    print("\n输出 -> results/a1b_availability.csv")
    print("判读：若 ΔAUC(M3-B) ≤0 且不显著为正，则 M3 波形内容在控制可得性后无增量，"
          "与'A1a 可得性携带信息但内容无增量'的叙事一致。")


if __name__ == "__main__":
    main()
