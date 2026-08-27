"""sensitivity_analyses.py — 敏感性分析 S4/S5/S9/S10（SAP 9.9，W7）。

  S4  完整病例分析（乳酸非缺失子集训练与评估）与 MICE 主分析对比
  S5  Track B 特征（tb1-tb32）替换 Track A（z1-z16）重跑 M3
  S9  校准方法对比：isotonic（tune 拟合）vs Platt 的测试集 Brier
  S10 去除乳酸重跑 M3（评估乳酸在增量价值中的贡献占比）

评估均在测试集执行：ΔAUC（对照 M1）+ 2000 次 bootstrap 95% CI。
训练协议与主分析一致（LR 系按 20 套插补预测均值；S10 无乳酸故单套）。
输出 results/sensitivity_s4_s5_s9_s10.csv。
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED = 20260823
N_BOOT = 2000
COV_COLS = ["admission_age", "male", "charlson_comorbidity_index",
            "admission_emergency", "pre_icu_los_h", "mech_vent_24h", "vaso_24h"]
SCORE_COLS = ["sofa_score", "qsofa", "news", "mews"]
Z_COLS = [f"z{i}" for i in range(1, 17)]
TB_COLS = [f"tb{i}" for i in range(1, 33)]


def boot_delta(y, pa, pb, rng):
    n, diffs = len(y), []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        if y[idx].sum() in (0, n):
            continue
        diffs.append(roc_auc_score(y[idx], pa[idx]) - roc_auc_score(y[idx], pb[idx]))
    return (float(np.mean(diffs)), float(np.percentile(diffs, 2.5)),
            float(np.percentile(diffs, 97.5)))


def lasso_fit(xtr, ytr):
    cv = LogisticRegressionCV(Cs=np.logspace(-3, 1, 12), penalty="l1", solver="saga",
                              scoring="roc_auc", cv=10, max_iter=5000, n_jobs=8,
                              random_state=SEED)
    cv.fit(xtr, ytr)
    auc_c = cv.scores_[1].mean(axis=0)
    se_c = cv.scores_[1].std(axis=0, ddof=1) / np.sqrt(10)
    i_min = int(np.argmax(auc_c))
    elig = np.where(auc_c >= auc_c[i_min] - se_c[i_min])[0]
    c = float(cv.Cs_[elig[0]])
    m = LogisticRegression(C=c, penalty="l1", solver="saga", max_iter=5000,
                           random_state=SEED)
    m.fit(xtr, ytr)
    return m


def build(datadir):
    dev = pd.read_parquet(datadir / "features_dev.parquet")
    scores = pd.read_parquet(datadir / "clinical_scores.parquet")
    mice = pd.read_parquet(datadir / "mice_lactate.parquet")
    df = dev.merge(scores[["stay_id", "qsofa", "news", "mews"]], on="stay_id", how="left")
    df = df.merge(mice.drop(columns=["subject_id", "subset"]), on="stay_id", how="left")
    df["male"] = (df["gender"] == "M").astype(int)
    df = pd.get_dummies(df, columns=["infection_site"], prefix="site", dtype=int)
    return df


def train_lasso_mice(df, feats, use_mice=True):
    """按 MICE 协议训练 LASSO（20 套插补预测均值），返回 tune/test 预测。"""
    mice_cols = [f"lactate_m{k:02d}" for k in range(1, 21)]
    tr, tu, te = (df["subset"] == s for s in ("train", "tune", "test"))
    ytr = df.loc[tr, "death_28d"].astype(int).to_numpy()
    p_tu = np.zeros(tu.sum())
    p_te = np.zeros(te.sum())
    n_sets = 20 if use_mice else 1
    for k in range(n_sets):
        cols = [c if c != "lactate" else mice_cols[k] for c in feats]
        xtr = df.loc[tr, cols].copy(); xtr.columns = feats
        xtu = df.loc[tu, cols].copy(); xtu.columns = feats
        xte = df.loc[te, cols].copy(); xte.columns = feats
        sc = StandardScaler().fit(xtr)
        m = lasso_fit(sc.transform(xtr), ytr)
        p_tu += m.predict_proba(sc.transform(xtu))[:, 1]
        p_te += m.predict_proba(sc.transform(xte))[:, 1]
    return p_tu / n_sets, p_te / n_sets


def main() -> None:
    datadir = REPO_ROOT / "data"
    results_dir = REPO_ROOT / "results"
    rng = np.random.default_rng(SEED)
    df = build(datadir)
    site_cols = sorted(c for c in df.columns if c.startswith("site_"))
    cov = COV_COLS + site_cols
    tr, tu, te = (df["subset"] == s for s in ("train", "tune", "test"))
    ytr = df.loc[tr, "death_28d"].astype(int).to_numpy()
    ytu = df.loc[tu, "death_28d"].astype(int).to_numpy()
    yte = df.loc[te, "death_28d"].astype(int).to_numpy()

    # M1 参照（普通 LR 重训，与主分析协议一致：20 套插补预测均值）
    f_m1 = SCORE_COLS + ["lactate"] + cov
    from sklearn.linear_model import LogisticRegression as LR
    mice_cols = [f"lactate_m{k:02d}" for k in range(1, 21)]
    p_m1_tu = np.zeros(tu.sum()); p_m1_te = np.zeros(te.sum())
    for k in range(20):
        cols = [c if c != "lactate" else mice_cols[k] for c in f_m1]
        xtr = df.loc[tr, cols].copy(); xtr.columns = f_m1
        xte = df.loc[te, cols].copy(); xte.columns = f_m1
        xtu = df.loc[tu, cols].copy(); xtu.columns = f_m1
        sc = StandardScaler().fit(xtr)
        m = LR(max_iter=2000).fit(sc.transform(xtr), ytr)
        p_m1_tu += m.predict_proba(sc.transform(xtu))[:, 1]
        p_m1_te += m.predict_proba(sc.transform(xte))[:, 1]
    p_m1_tu /= 20; p_m1_te /= 20

    rows = []

    # ---- S5：Track B 替换 Track A ----
    f_s5 = SCORE_COLS + ["lactate"] + cov + TB_COLS
    p_s5_tu, p_s5_te = train_lasso_mice(df, f_s5)
    d = boot_delta(yte, p_s5_te, p_m1_te, rng)
    rows.append({"id": "S5", "内容": "Track B(tb32) 替换 Track A 的 M3",
                 "test_auc": roc_auc_score(yte, p_s5_te),
                 "delta_vs_m1": d[0], "ci_lo": d[1], "ci_hi": d[2]})
    print(f"[S5] AUC {rows[-1]['test_auc']:.4f}，ΔAUC vs M1 {d[0]:+.4f} "
          f"({d[1]:+.4f}~{d[2]:+.4f})", flush=True)

    # ---- S10：去乳酸 ----
    f_s10 = SCORE_COLS + cov + Z_COLS
    p_s10_tu, p_s10_te = train_lasso_mice(df, f_s10, use_mice=False)
    d = boot_delta(yte, p_s10_te, p_m1_te, rng)
    rows.append({"id": "S10", "内容": "M3 去除乳酸",
                 "test_auc": roc_auc_score(yte, p_s10_te),
                 "delta_vs_m1": d[0], "ci_lo": d[1], "ci_hi": d[2]})
    print(f"[S10] AUC {rows[-1]['test_auc']:.4f}，ΔAUC vs M1 {d[0]:+.4f} "
          f"({d[1]:+.4f}~{d[2]:+.4f})", flush=True)

    # ---- S4：完整病例 ----
    cc_tr = tr & df["lactate"].notna()
    cc_te = te & df["lactate"].notna()
    f_m3 = SCORE_COLS + ["lactate"] + cov + Z_COLS
    sc = StandardScaler().fit(df.loc[cc_tr, f_m3])
    m_cc = lasso_fit(sc.transform(df.loc[cc_tr, f_m3]),
                     df.loc[cc_tr, "death_28d"].astype(int).to_numpy())
    p_cc = m_cc.predict_proba(sc.transform(df.loc[cc_te, f_m3]))[:, 1]
    ycc = df.loc[cc_te, "death_28d"].astype(int).to_numpy()
    # 主分析 M3 预测（读 test_predictions）在同一子集对比
    pred_te = pd.read_parquet(results_dir / "test_predictions.parquet")
    pred_te = pred_te.merge(df[["stay_id", "lactate"]], on="stay_id")
    mask = pred_te["lactate"].notna().to_numpy()
    d = boot_delta(ycc, p_cc, pred_te.loc[mask, "M1_cal"].to_numpy(), rng)
    rows.append({"id": "S4", "内容": "完整病例 M3（对比同子集 MICE-M1）",
                 "test_auc": roc_auc_score(ycc, p_cc),
                 "delta_vs_m1": d[0], "ci_lo": d[1], "ci_hi": d[2],
                 "备注": f"完整病例 n={int(cc_te.sum())}"})
    print(f"[S4] 完整病例 AUC {rows[-1]['test_auc']:.4f}（n={cc_te.sum():,}），"
          f"ΔAUC vs M1 {d[0]:+.4f}", flush=True)

    # ---- S9：isotonic vs Platt ----
    platt = joblib.load(datadir / "models" / "platt.joblib")
    pred_tu = pd.read_parquet(results_dir / "tune_predictions.parquet")
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(pred_tu["M3_raw"], ytu)
    p_m3_te_raw = pred_te["M3_raw"].to_numpy()
    p_m3_te_platt = pred_te["M3_cal"].to_numpy()
    yte_full = pred_te["y"].astype(int).to_numpy()
    b_iso = brier_score_loss(yte_full, iso.predict(p_m3_te_raw))
    b_platt = brier_score_loss(yte_full, p_m3_te_platt)
    b_raw = brier_score_loss(yte_full, p_m3_te_raw)
    rows.append({"id": "S9", "内容": "M3 校准方法对比（测试集 Brier）",
                 "test_auc": np.nan, "delta_vs_m1": np.nan,
                 "ci_lo": np.nan, "ci_hi": np.nan,
                 "备注": f"raw {b_raw:.4f} / Platt {b_platt:.4f} / isotonic {b_iso:.4f}"})
    print(f"[S9] Brier: raw {b_raw:.4f} / Platt {b_platt:.4f} / isotonic {b_iso:.4f}",
          flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(results_dir / "sensitivity_s4_s5_s9_s10.csv", index=False)
    print("\n输出 -> results/sensitivity_s4_s5_s9_s10.csv")


if __name__ == "__main__":
    main()
