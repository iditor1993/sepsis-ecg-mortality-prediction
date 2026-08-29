"""train_m2plus.py — M2+：仅 ECG 潜向量 + XGBoost（Track A/B 双轨）。

对照 M2（潜向量 LR）检验非线性树模型对纯 ECG 表征的增益：
  M2+(TA)：z1-z16 -> XGBoost；M2+(TB)：tb1-tb32 -> XGBoost
  网格与早停协议同 M1+/M4（SAP 8.2）。
评估：测试集 AUC；ΔAUC vs M2-LR 同轨版（M2-TA 0.642 / M2-TB 0.681，
 配对 bootstrap 2000 次 95% CI）；双轨间 ΔAUC(TB-TA)。
输出：results/m2plus_results.csv；模型存 data/models/。
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED = 20260823
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


def main() -> None:
    datadir = REPO_ROOT / "data"
    results_dir = REPO_ROOT / "results"
    rng = np.random.default_rng(SEED)
    from sensitivity_analyses import build
    from train_m0_m5 import fit_xgb_grid

    df = build(datadir)
    tr, tu, te = (df["subset"] == s for s in ("train", "tune", "test"))
    ytr = df.loc[tr, "death_28d"].astype(int).to_numpy()
    ytu = df.loc[tu, "death_28d"].astype(int).to_numpy()
    yte = df.loc[te, "death_28d"].astype(int).to_numpy()

    # M2-LR 参考预测（TA 用既有入库预测；TB 用 trackB_predictions）
    pred_te = pd.read_parquet(results_dir / "test_predictions.parquet")
    p_m2ta_lr = pred_te["M2_cal"].to_numpy()
    tb_preds = pd.read_parquet(results_dir / "trackB_predictions.parquet")
    p_m2tb_lr = tb_preds["M2_TB"].to_numpy()

    rows, preds = [], {}
    for tag, cols in [("M2+(Track A)", Z_COLS), ("M2+(Track B)", TB_COLS)]:
        best = fit_xgb_grid(df.loc[tr, cols], ytr, df.loc[tu, cols], ytu, tag)
        p = best["model"].predict_proba(df.loc[te, cols])[:, 1]
        preds[tag] = p
        auc = roc_auc_score(yte, p)
        joblib.dump({"features": cols, "cfg": best["cfg"], "model": best["model"]},
                    datadir / "models" / f"m2plus_{'ta' if 'A' in tag else 'tb'}_xgb.joblib")
        p_lr = p_m2ta_lr if "A" in tag else p_m2tb_lr
        d, lo, hi = boot_delta(yte, p, p_lr, rng)
        rows.append({"model": tag, "n_features": len(cols), "test_auc": auc,
                     "comparison": f"{tag} vs M2-LR（同轨）",
                     "delta_auc": d, "ci_lo": lo, "ci_hi": hi,
                     "best_cfg": str(best["cfg"])})
        print(f"[{tag}] test AUC {auc:.4f}；vs M2-LR 同轨 {d:+.4f} "
              f"({lo:+.4f}~{hi:+.4f})", flush=True)

    d, lo, hi = boot_delta(yte, preds["M2+(Track B)"], preds["M2+(Track A)"], rng)
    rows.append({"model": "M2+(TB) vs M2+(TA)", "n_features": np.nan,
                 "test_auc": np.nan, "comparison": "AUC(M2+TB) - AUC(M2+TA)",
                 "delta_auc": d, "ci_lo": lo, "ci_hi": hi, "best_cfg": ""})
    print(f"[M2+ 双轨] ΔAUC(TB-TA) {d:+.4f} ({lo:+.4f}~{hi:+.4f})")

    pd.DataFrame(rows).to_csv(results_dir / "m2plus_results.csv",
                              index=False, encoding="utf-8-sig")
    print("\n输出 -> results/m2plus_results.csv")


if __name__ == "__main__":
    main()
