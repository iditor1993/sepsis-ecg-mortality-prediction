"""train_m3plus.py — M3+：LASSO 保留特征 + XGBoost（Track A/B 双轨）。

设计：
  1) 特征选择：LASSO（lambda.1se，20 套 MICE）各套非零系数，取 ≥50% 套数
     保留的特征（Track A 用既有 m3_models.joblib；Track B 对 M1+tb32 重跑
     同样的 LASSO 筛选）
  2) M3+：保留特征上训练 XGBoost（网格与早停协议同 M1+/M4，SAP 8.2）
  3) 评估：测试集 AUC、ΔAUC vs M1（配对 bootstrap 2000 次 95% CI）
输出：results/m3plus_results.csv；控制台展示筛选名单与结果。
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED = 20260823
M = 20
COV_COLS = ["admission_age", "male", "charlson_comorbidity_index",
            "admission_emergency", "pre_icu_los_h", "mech_vent_24h", "vaso_24h"]
SCORE_COLS = ["sofa_score", "qsofa", "news", "mews"]
Z_COLS = [f"z{i}" for i in range(1, 17)]
TB_COLS = [f"tb{i}" for i in range(1, 33)]


def boot_delta(y, pa, pb, rng, n_boot=2000):
    n, diffs = len(y), []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].sum() in (0, n):
            continue
        diffs.append(roc_auc_score(y[idx], pa[idx]) - roc_auc_score(y[idx], pb[idx]))
    return (float(np.mean(diffs)), float(np.percentile(diffs, 2.5)),
            float(np.percentile(diffs, 97.5)))


def lasso_select_tb(df, f_m3b, tr, ytr, mice_cols):
    """Track B：重跑 LASSO MICE 并返回各特征保留频率。"""
    from sensitivity_analyses import lasso_fit
    freq = pd.Series(0, index=f_m3b)
    for k in range(M):
        cols = [c if c != "lactate" else mice_cols[k] for c in f_m3b]
        xtr = df.loc[tr, cols].copy(); xtr.columns = f_m3b
        sc = StandardScaler().fit(xtr)
        m = lasso_fit(sc.transform(xtr), ytr)
        freq += (m.coef_.ravel() != 0).astype(int)
    return freq


def main() -> None:
    datadir = REPO_ROOT / "data"
    results_dir = REPO_ROOT / "results"
    rng = np.random.default_rng(SEED)
    from sensitivity_analyses import build
    from train_m0_m5 import fit_xgb_grid

    df = build(datadir)
    site_cols = sorted(c for c in df.columns if c.startswith("site_"))
    cov = COV_COLS + site_cols
    tr, tu, te = (df["subset"] == s for s in ("train", "tune", "test"))
    ytr = df.loc[tr, "death_28d"].astype(int).to_numpy()
    ytu = df.loc[tu, "death_28d"].astype(int).to_numpy()
    yte = df.loc[te, "death_28d"].astype(int).to_numpy()
    mice_cols = [f"lactate_m{k:02d}" for k in range(1, M + 1)]
    pred_te = pd.read_parquet(results_dir / "test_predictions.parquet")
    p_m1 = pred_te["M1_cal"].to_numpy()
    f_m1 = SCORE_COLS + ["lactate"] + cov

    rows = []

    # ---- Track A：读取既有 M3 LASSO 入选频率 ----
    pack = joblib.load(datadir / "models" / "m3_models.joblib")
    feats_all = pack["features"]
    freq_a = pd.Series({f: int(sum(mk["model"].coef_.ravel()[i] != 0
                                   for mk in pack["models"]))
                        for i, f in enumerate(feats_all)})
    sel_a = freq_a[freq_a >= M // 2].index.tolist()
    print(f"[M3+TA] LASSO 保留特征（≥{M // 2}/20 套，n={len(sel_a)}）: {sel_a}")
    best_a = fit_xgb_grid(df.loc[tr, sel_a], ytr, df.loc[tu, sel_a], ytu, "M3+TA")
    p_a = best_a["model"].predict_proba(df.loc[te, sel_a])[:, 1]
    auc_a = roc_auc_score(yte, p_a)
    joblib.dump({"features": sel_a, "cfg": best_a["cfg"], "model": best_a["model"]},
                datadir / "models" / "m3plus_ta_xgb.joblib")
    d, lo, hi = boot_delta(yte, p_a, p_m1, rng)
    rows.append({"model": "M3+(Track A)", "n_features": len(sel_a),
                 "features": ",".join(sel_a), "test_auc": auc_a,
                 "delta_vs_m1": d, "ci_lo": lo, "ci_hi": hi,
                 "best_cfg": str(best_a["cfg"])})
    print(f"[M3+TA] test AUC {auc_a:.4f}；ΔAUC vs M1 {d:+.4f} ({lo:+.4f}~{hi:+.4f})",
          flush=True)

    # ---- Track B：先 LASSO 筛选（M1+tb32），再 XGBoost ----
    f_m3b = f_m1 + TB_COLS
    freq_b = lasso_select_tb(df, f_m3b, tr, ytr, mice_cols)
    sel_b = freq_b[freq_b >= M // 2].index.tolist()
    print(f"[M3+TB] LASSO 保留特征（≥{M // 2}/20 套，n={len(sel_b)}）: {sel_b}",
          flush=True)
    best_b = fit_xgb_grid(df.loc[tr, sel_b], ytr, df.loc[tu, sel_b], ytu, "M3+TB")
    p_b = best_b["model"].predict_proba(df.loc[te, sel_b])[:, 1]
    auc_b = roc_auc_score(yte, p_b)
    joblib.dump({"features": sel_b, "cfg": best_b["cfg"], "model": best_b["model"]},
                datadir / "models" / "m3plus_tb_xgb.joblib")
    d, lo, hi = boot_delta(yte, p_b, p_m1, rng)
    rows.append({"model": "M3+(Track B)", "n_features": len(sel_b),
                 "features": ",".join(sel_b), "test_auc": auc_b,
                 "delta_vs_m1": d, "ci_lo": lo, "ci_hi": hi,
                 "best_cfg": str(best_b["cfg"])})
    print(f"[M3+TB] test AUC {auc_b:.4f}；ΔAUC vs M1 {d:+.4f} ({lo:+.4f}~{hi:+.4f})",
          flush=True)

    # 两轨 M3+ 之间配对对比
    d, lo, hi = boot_delta(yte, p_b, p_a, rng)
    rows.append({"model": "M3+(TB) vs M3+(TA)", "n_features": np.nan,
                 "features": "", "test_auc": np.nan,
                 "delta_vs_m1": d, "ci_lo": lo, "ci_hi": hi, "best_cfg": ""})
    print(f"\n[M3+ 双轨] ΔAUC(TB-TA) {d:+.4f} ({lo:+.4f}~{hi:+.4f})")

    res = pd.DataFrame(rows)
    res.to_csv(results_dir / "m3plus_results.csv", index=False, encoding="utf-8-sig")
    print("\n输出 -> results/m3plus_results.csv")


if __name__ == "__main__":
    main()
