"""train_m4plus.py — M4+：M1+ 强表格特征 + ECG 潜向量（补充分析）。

正面回答"ECG 加到最强表格基线上是否有增量"：
  特征 = M1+ 85 列汇总 + 乳酸 + 协变量 + z1-z16（约 101 列）
  算法 = XGBoost，网格与早停协议同 M1+/M4（SAP 8.2）
评估：测试集 AUC、ΔAUC vs M1+（配对 bootstrap 2000 次 95% CI）；
      TreeExplainer 检验 z 维度是否被吸收（SHAP 均值排序）。
输出：results/m4plus_results.csv、results/m4plus_shap.csv、
      data/models/m4plus_xgb.joblib。
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED = 20260823
COV_COLS = ["admission_age", "male", "charlson_comorbidity_index",
            "admission_emergency", "pre_icu_los_h", "mech_vent_24h", "vaso_24h"]
Z_COLS = [f"z{i}" for i in range(1, 17)]


def main() -> None:
    datadir = REPO_ROOT / "data"
    results_dir = REPO_ROOT / "results"
    rng = np.random.default_rng(SEED)
    from sensitivity_analyses import build
    from train_m0_m5 import fit_xgb_grid

    df = build(datadir)
    site_cols = sorted(c for c in df.columns if c.startswith("site_"))
    m1p = pd.read_parquet(datadir / "m1plus_features.parquet")
    m1p_feats = [c for c in m1p.columns if c not in ("subject_id", "stay_id")]
    df = df.merge(m1p[["stay_id"] + m1p_feats], on="stay_id", how="left")

    feats = m1p_feats + ["lactate"] + COV_COLS + site_cols + Z_COLS
    tr, tu, te = (df["subset"] == s for s in ("train", "tune", "test"))
    ytr = df.loc[tr, "death_28d"].astype(int).to_numpy()
    ytu = df.loc[tu, "death_28d"].astype(int).to_numpy()
    yte = df.loc[te, "death_28d"].astype(int).to_numpy()
    print(f"[M4+] 特征 {len(feats)} 列（M1+ {len(m1p_feats)} + 乳酸/协变量 + z16）")

    best = fit_xgb_grid(df.loc[tr, feats], ytr, df.loc[tu, feats], ytu, "M4+")
    p_te = best["model"].predict_proba(df.loc[te, feats])[:, 1]
    auc_te = roc_auc_score(yte, p_te)
    joblib.dump({"features": feats, "cfg": best["cfg"], "model": best["model"]},
                datadir / "models" / "m4plus_xgb.joblib")

    # 配对 ΔAUC vs M1+
    pred_te = pd.read_parquet(results_dir / "test_predictions.parquet")
    p_m1p = pred_te["M1+_cal"].to_numpy()
    n, diffs = len(yte), []
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        if yte[idx].sum() in (0, n):
            continue
        diffs.append(roc_auc_score(yte[idx], p_te[idx]) - roc_auc_score(yte[idx], p_m1p[idx]))
    d, lo, hi = float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

    res = pd.DataFrame([{"model": "M4+（M1+特征+z16）", "test_auc": auc_te,
                         "对比": "vs M1+", "delta_auc": d, "ci_lo": lo, "ci_hi": hi,
                         "best_cfg": str(best["cfg"]),
                         "best_iter": best["best_iter"]}])
    res.to_csv(results_dir / "m4plus_results.csv", index=False, encoding="utf-8-sig")
    print(f"\n[M4+] test AUC {auc_te:.4f}；ΔAUC vs M1+ {d:+.4f} ({lo:+.4f}~{hi:+.4f})")

    # SHAP：z 维度是否被吸收
    import shap
    explainer = shap.TreeExplainer(best["model"])
    sv = explainer.shap_values(df.loc[te, feats])
    imp = pd.DataFrame({"feature": feats,
                        "mean_abs_shap": np.abs(sv).mean(axis=0)})
    imp = imp.sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(results_dir / "m4plus_shap.csv", index=False)
    z_rank = imp.reset_index(drop=True)
    z_top = z_rank[z_rank["feature"].isin(Z_COLS)]
    print("\n[M4+ SHAP] 前 10 特征:")
    print(imp.head(10).round(4).to_string(index=False))
    print(f"\n[M4+ SHAP] z 维度最高排名: 第 {z_top.index.min() + 1} 位"
          f"（{z_top.iloc[z_top['mean_abs_shap'].argmax()]['feature']}，"
          f"|SHAP| {z_top['mean_abs_shap'].max():.4f}）")
    print("\n输出 -> results/m4plus_results.csv, results/m4plus_shap.csv")


if __name__ == "__main__":
    main()
