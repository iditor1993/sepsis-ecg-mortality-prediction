"""repair_m1plus.py — stay_id_1 混入事件的 M1+ 修复重训。

背景：sql/05 输出的 piv.* 带入重复列 stay_id_1（= stay_id），被
train_m0_m5.py 的 M1+ 特征选择误纳为建模特征。本脚本：
  1) 用修正后的 85 列特征重训 M1+（同一网格/种子/协议）
  2) 重算 tune 预测并为 M1+ 重新拟合 Platt（其余模型校准器不动）
  3) 更新 data/models/m1plus_xgb.joblib、platt.joblib、
     results/tune_predictions.parquet 与 tune_metrics.csv 的 M1+ 行
  4) 打印新旧模型 tune/test AUC 与预测相关性（影响评估）
后续须重跑 evaluate.py 以刷新全部测试集指标。
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED = 20260823
COV_COLS = ["admission_age", "male", "charlson_comorbidity_index",
            "admission_emergency", "pre_icu_los_h", "mech_vent_24h", "vaso_24h"]


def main() -> None:
    datadir = REPO_ROOT / "data"
    results_dir = REPO_ROOT / "results"
    from sensitivity_analyses import build
    from train_m0_m5 import fit_xgb_grid, platt_apply, platt_fit, youden_threshold

    df = build(datadir)
    site_cols = sorted(c for c in df.columns if c.startswith("site_"))
    m1p = pd.read_parquet(datadir / "m1plus_features.parquet")
    assert "stay_id_1" not in m1p.columns
    m1p_feats = [c for c in m1p.columns if c not in ("subject_id", "stay_id")]
    assert len(m1p_feats) == 85, f"特征列数异常: {len(m1p_feats)}"
    df = df.merge(m1p[["stay_id"] + m1p_feats], on="stay_id", how="left")

    feats = m1p_feats + ["lactate"] + COV_COLS + site_cols
    tr, tu, te = (df["subset"] == s for s in ("train", "tune", "test"))
    ytr = df.loc[tr, "death_28d"].astype(int).to_numpy()
    ytu = df.loc[tu, "death_28d"].astype(int).to_numpy()

    pred_tu = pd.read_parquet(results_dir / "tune_predictions.parquet")
    pred_te = pd.read_parquet(results_dir / "test_predictions.parquet")
    old_tu = pred_tu["M1+_raw"].to_numpy()
    old_te = pred_te["M1+_raw"].to_numpy()
    yte = pred_te["y"].astype(int).to_numpy()

    best = fit_xgb_grid(df.loc[tr, feats], ytr, df.loc[tu, feats], ytu, "M1+修复")
    p_tu = best["model"].predict_proba(df.loc[tu, feats])[:, 1]
    p_te = best["model"].predict_proba(df.loc[te, feats])[:, 1]

    print("\n[影响评估]")
    print(f"旧 M1+: tune AUC {roc_auc_score(ytu, old_tu):.4f} / "
          f"test AUC {roc_auc_score(yte, old_te):.4f}")
    print(f"新 M1+: tune AUC {roc_auc_score(ytu, p_tu):.4f} / "
          f"test AUC {roc_auc_score(yte, p_te):.4f}")
    print(f"预测相关性: tune r={np.corrcoef(old_tu, p_tu)[0,1]:.5f}；"
          f"test r={np.corrcoef(old_te, p_te)[0,1]:.5f}")
    print(f"test AUC 变化: {roc_auc_score(yte, p_te) - roc_auc_score(yte, old_te):+.5f}")

    # 落盘：模型、Platt、tune 预测与指标
    joblib.dump({"features": feats, "cfg": best["cfg"], "model": best["model"]},
                datadir / "models" / "m1plus_xgb.joblib")
    platt = joblib.load(datadir / "models" / "platt.joblib")
    platt["M1+"] = platt_fit(p_tu, ytu)
    joblib.dump(platt, datadir / "models" / "platt.joblib")

    pred_tu["M1+_raw"] = p_tu
    pred_tu["M1+_cal"] = platt_apply(p_tu, platt["M1+"])
    pred_tu.to_parquet(results_dir / "tune_predictions.parquet", index=False)

    tm = pd.read_csv(results_dir / "tune_metrics.csv")
    pc = pred_tu["M1+_cal"].to_numpy()
    tm.loc[tm["model"] == "M1+", ["tune_auc_raw", "tune_auc_cal",
                                  "tune_brier_raw", "tune_brier_cal",
                                  "youden_threshold"]] = [
        roc_auc_score(ytu, p_tu), roc_auc_score(ytu, pc),
        brier_score_loss(ytu, p_tu), brier_score_loss(ytu, pc),
        youden_threshold(ytu, pc)]
    tm.to_csv(results_dir / "tune_metrics.csv", index=False)
    print("\n模型/校准器/tune 预测与指标已更新；请重跑 evaluate.py 刷新测试集指标。")


if __name__ == "__main__":
    main()
