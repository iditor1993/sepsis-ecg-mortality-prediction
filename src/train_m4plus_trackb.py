"""train_m4plus_trackb.py — M4+ 的 Track B 版（M1+ 特征 + tb1-tb32）。

补全 Track A/B 对比表 M4+ 行：特征 = M1+ 86 列汇总 + 乳酸 + 协变量 +
tb1-tb32，XGBoost 同一网格协议；测试集 AUC、ΔAUC vs M1+ 与
ΔAUC vs M4+(Track A 版)（均配对 bootstrap 2000 次）。
结果追加至 results/m4plus_results.csv。
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

    feats = m1p_feats + ["lactate"] + COV_COLS + site_cols + TB_COLS
    tr, tu, te = (df["subset"] == s for s in ("train", "tune", "test"))
    ytr = df.loc[tr, "death_28d"].astype(int).to_numpy()
    ytu = df.loc[tu, "death_28d"].astype(int).to_numpy()
    yte = df.loc[te, "death_28d"].astype(int).to_numpy()
    print(f"[M4+-TB] 特征 {len(feats)} 列")

    best = fit_xgb_grid(df.loc[tr, feats], ytr, df.loc[tu, feats], ytu, "M4+-TB")
    p_te = best["model"].predict_proba(df.loc[te, feats])[:, 1]
    auc_te = roc_auc_score(yte, p_te)
    joblib.dump({"features": feats, "cfg": best["cfg"], "model": best["model"]},
                datadir / "models" / "m4plus_trackb_xgb.joblib")

    pred_te = pd.read_parquet(results_dir / "test_predictions.parquet")
    d1, lo1, hi1 = boot_delta(yte, p_te, pred_te["M1+_cal"].to_numpy(), rng)

    # vs M4+ Track A 版：重建其测试预测
    pack_a = joblib.load(datadir / "models" / "m4plus_xgb.joblib")
    p_a = pack_a["model"].predict_proba(df.loc[te, pack_a["features"]])[:, 1]
    d2, lo2, hi2 = boot_delta(yte, p_te, p_a, rng)

    res = pd.DataFrame([{
        "model": "M4+-TB（M1+特征+tb32）", "test_auc": auc_te,
        "对比": f"vs M1+：{d1:+.4f}({lo1:+.4f}~{hi1:+.4f})；"
                f"vs M4+(TA)：{d2:+.4f}({lo2:+.4f}~{hi2:+.4f})",
        "delta_auc": d1, "ci_lo": lo1, "ci_hi": hi1,
        "best_cfg": str(best["cfg"]), "best_iter": best["best_iter"]}])
    res.to_csv(results_dir / "m4plus_trackb_results.csv", index=False,
               encoding="utf-8-sig")
    print(f"\n[M4+-TB] test AUC {auc_te:.4f}")
    print(f"  vs M1+：{d1:+.4f} ({lo1:+.4f}~{hi1:+.4f})")
    print(f"  vs M4+(Track A)：{d2:+.4f} ({lo2:+.4f}~{hi2:+.4f})")
    print("\n输出 -> results/m4plus_trackb_results.csv")


if __name__ == "__main__":
    main()
