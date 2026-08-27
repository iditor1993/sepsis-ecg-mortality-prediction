"""train_m0_m5.py — 模型体系训练（SAP 8.1/8.2，W4-W5）。

模型（本脚本覆盖 M0-M4 与 M1+；M5 端到端微调为探索性，另行实现）：
  M0  sofa_score -> LR（参照基线）
  M1  评分(SOFA/qSOFA/NEWS/MEWS)+乳酸+协变量 -> LR（临床常规基线）
  M2  z1-z16 -> LR（ECG 单独贡献）
  M3  M1 特征+z1-z16 -> LASSO-LR（主模型；10 折 CV 选 lambda，报 lambda.min
      与 lambda.1se，以 lambda.1se 为最终模型）
  M1+ 85 列 t0±24h 汇总特征+乳酸+协变量 -> XGBoost（强表格基线）
  M4  同 M3 全部特征 -> XGBoost（非线性基准）

协议（SAP 8.2 / 第十章）：
  - LR 系（M0/M1/M2/M3）：m=20 套 MICE 插补数据分别训练，tune 集预测取
    20 套均值（Rubin 预测合并）；连续变量标准化（train 拟合）
  - XGBoost（M1+/M4）：原始特征原生处理 NaN；超参网格（SAP 8.2）
    max_depth{3,4,6} x eta{0.01,0.05,0.1} x min_child_weight{1,5}，
    subsample=colsample=0.8，tune 集早停 50 轮
  - Platt 校准（tune 集拟合）；约登阈值（tune 集确定并冻结）
  - 训练仅用 train/tune 子集；test/temporal 不在本脚本出现

输出：
  results/tune_predictions.parquet  tune 集预测（各模型 raw/calibrated）
  results/tune_metrics.csv          tune 集 AUC/Brier/阈值
  data/models/*.joblib              模型与校准器（本地，不入库）
  控制台打印 tune 集 AUC 与经验 ρ（M3 vs M1、M3 vs M1+，功效复核用）
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED = 20260823
M = 20

COV_COLS = ["admission_age", "male", "charlson_comorbidity_index",
            "admission_emergency", "pre_icu_los_h", "mech_vent_24h", "vaso_24h"]
SCORE_COLS = ["sofa_score", "qsofa", "news", "mews"]
Z_COLS = [f"z{i}" for i in range(1, 17)]
XGB_GRID = [{"max_depth": d, "eta": e, "min_child_weight": w}
            for d in (3, 4, 6) for e in (0.01, 0.05, 0.1) for w in (1, 5)]


def platt_fit(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Platt 校准：y ~ sigmoid(a*logit(p)+b)，tune 集拟合。"""
    x = logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(x, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def platt_apply(p: np.ndarray, ab: tuple[float, float]) -> np.ndarray:
    a, b = ab
    return expit(a * logit(np.clip(p, 1e-6, 1 - 1e-6)) + b)


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def fit_xgb_grid(xtr, ytr, xtu, ytu, label: str):
    import xgboost as xgb

    best = {"auc": -1.0}
    for cfg in XGB_GRID:
        m = xgb.XGBClassifier(
            n_estimators=5000, subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", eval_metric="auc", random_state=SEED,
            early_stopping_rounds=50, n_jobs=8, **cfg)
        m.fit(xtr, ytr, eval_set=[(xtu, ytu)], verbose=False)
        auc = roc_auc_score(ytu, m.predict_proba(xtu)[:, 1])
        if auc > best["auc"]:
            best = {"auc": auc, "cfg": cfg, "model": m,
                    "best_iter": int(m.best_iteration)}
        print(f"  [{label}] {cfg} -> tune AUC {auc:.4f} "
              f"(iter {int(m.best_iteration)})", flush=True)
    print(f"  [{label}] 最佳: {best['cfg']} AUC {best['auc']:.4f}")
    return best


def main() -> None:
    datadir = REPO_ROOT / "data"
    results_dir = REPO_ROOT / "results"
    models_dir = datadir / "models"
    models_dir.mkdir(exist_ok=True)

    dev = pd.read_parquet(datadir / "features_dev.parquet")
    scores = pd.read_parquet(datadir / "clinical_scores.parquet")
    m1p = pd.read_parquet(datadir / "m1plus_features.parquet")
    mice = pd.read_parquet(datadir / "mice_lactate.parquet")

    df = dev.merge(scores[["stay_id", "qsofa", "news", "mews"]], on="stay_id", how="left")
    df = df.merge(mice.drop(columns=["subject_id", "subset"]), on="stay_id", how="left")
    df["male"] = (df["gender"] == "M").astype(int)
    df = pd.get_dummies(df, columns=["infection_site"], prefix="site", dtype=int)
    site_cols = sorted(c for c in df.columns if c.startswith("site_"))
    cov_cols = COV_COLS + site_cols
    m1p_feats = [c for c in m1p.columns if c not in ("subject_id", "stay_id")]
    df = df.merge(m1p[["stay_id"] + m1p_feats], on="stay_id", how="left")

    tr = df["subset"] == "train"
    tu = df["subset"] == "tune"
    ytr = df.loc[tr, "death_28d"].astype(int).to_numpy()
    ytu = df.loc[tu, "death_28d"].astype(int).to_numpy()
    print(f"train {tr.sum():,}（事件 {ytr.sum()}）/ tune {tu.sum():,}（事件 {ytu.sum()}）")
    print(f"M1+ 特征列数: {len(m1p_feats)}")

    mice_cols = [f"lactate_m{k:02d}" for k in range(1, M + 1)]
    f_m0 = ["sofa_score"]
    f_m1 = SCORE_COLS + ["lactate"] + cov_cols
    f_m2 = Z_COLS
    f_m3 = f_m1 + Z_COLS

    # ---- LR 系：20 套插补分别训练，tune 预测均值合并 ----
    preds_raw = {}
    saved = {}
    for name, feats, lasso in [("M0", f_m0, False), ("M1", f_m1, False),
                               ("M2", f_m2, False), ("M3", f_m3, True)]:
        p_sum, models_k, c_choices = np.zeros(tu.sum()), [], []
        for k, mc in enumerate(mice_cols):
            xtr = df.loc[tr, [c if c != "lactate" else mc for c in feats]].copy()
            xtr.columns = feats
            xtu = df.loc[tu, [c if c != "lactate" else mc for c in feats]].copy()
            xtu.columns = feats
            sc = StandardScaler().fit(xtr)
            xtr_s, xtu_s = sc.transform(xtr), sc.transform(xtu)
            if lasso:
                cv = LogisticRegressionCV(
                    Cs=np.logspace(-3, 1, 12), penalty="l1", solver="saga",
                    scoring="roc_auc", cv=10, max_iter=5000, n_jobs=8,
                    random_state=SEED)
                cv.fit(xtr_s, ytr)
                auc_per_c = cv.scores_[1].mean(axis=0)
                se_per_c = cv.scores_[1].std(axis=0, ddof=1) / np.sqrt(10)
                i_min = int(np.argmax(auc_per_c))
                thr1se = auc_per_c[i_min] - se_per_c[i_min]
                eligible = np.where(auc_per_c >= thr1se)[0]
                i_1se = int(eligible[0])  # Cs 升序：满足 1se 的最小 C（最简模型）
                c_choices.append({"lambda_min_C": float(cv.Cs_[i_min]),
                                  "lambda_1se_C": float(cv.Cs_[i_1se]),
                                  "cv_auc_min": float(auc_per_c[i_min])})
                m = LogisticRegression(C=float(cv.Cs_[i_1se]), penalty="l1",
                                       solver="saga", max_iter=5000,
                                       random_state=SEED)
                m.fit(xtr_s, ytr)
            else:
                m = LogisticRegression(max_iter=2000)
                m.fit(xtr_s, ytr)
            p_sum += m.predict_proba(xtu_s)[:, 1]
            models_k.append({"scaler": sc, "model": m})
        preds_raw[name] = p_sum / M
        saved[name] = {"features": feats, "models": models_k,
                       "c_choices": c_choices if lasso else None}
        joblib.dump(saved[name], models_dir / f"{name.lower()}_models.joblib")
        extra = f"；lambda.1se C 中位 {np.median([c['lambda_1se_C'] for c in c_choices]):.4f}" if lasso else ""
        print(f"[{name}] tune AUC {roc_auc_score(ytu, preds_raw[name]):.4f}{extra}", flush=True)

    # ---- XGBoost：M1+ 与 M4 ----
    f_m1p = m1p_feats + ["lactate"] + cov_cols
    xtr_raw = df.loc[tr, f_m1p]; xtu_raw = df.loc[tu, f_m1p]
    best_m1p = fit_xgb_grid(xtr_raw, ytr, xtu_raw, ytu, "M1+")
    preds_raw["M1+"] = best_m1p["model"].predict_proba(xtu_raw)[:, 1]
    joblib.dump({"features": f_m1p, "cfg": best_m1p["cfg"],
                 "model": best_m1p["model"]}, models_dir / "m1plus_xgb.joblib")

    xtr4 = df.loc[tr, f_m3]; xtu4 = df.loc[tu, f_m3]
    best_m4 = fit_xgb_grid(xtr4, ytr, xtu4, ytu, "M4")
    preds_raw["M4"] = best_m4["model"].predict_proba(xtu4)[:, 1]
    joblib.dump({"features": f_m3, "cfg": best_m4["cfg"],
                 "model": best_m4["model"]}, models_dir / "m4_xgb.joblib")

    # ---- Platt 校准 + 约登阈值 + 指标 ----
    rows, platt, thresholds = [], {}, {}
    pred_df = df.loc[tu, ["subject_id", "stay_id"]].copy()
    pred_df["y"] = ytu
    for name, p in preds_raw.items():
        ab = platt_fit(p, ytu)
        platt[name] = ab
        pc = platt_apply(p, ab)
        thr = youden_threshold(ytu, pc)
        thresholds[name] = thr
        pred_df[f"{name}_raw"] = p
        pred_df[f"{name}_cal"] = pc
        rows.append({
            "model": name,
            "tune_auc_raw": roc_auc_score(ytu, p),
            "tune_auc_cal": roc_auc_score(ytu, pc),
            "tune_brier_raw": brier_score_loss(ytu, p),
            "tune_brier_cal": brier_score_loss(ytu, pc),
            "youden_threshold": thr,
        })
    joblib.dump(platt, models_dir / "platt.joblib")
    (models_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2))

    metrics = pd.DataFrame(rows)
    metrics.to_csv(results_dir / "tune_metrics.csv", index=False)
    pred_df.to_parquet(results_dir / "tune_predictions.parquet", index=False)

    print("\n[tune 集指标]")
    print(metrics.round(4).to_string(index=False))
    rho_m3_m1 = float(np.corrcoef(preds_raw["M3"], preds_raw["M1"])[0, 1])
    rho_m3_m1p = float(np.corrcoef(preds_raw["M3"], preds_raw["M1+"])[0, 1])
    print(f"\n[经验 ρ（tune 预测值 Pearson）] M3 vs M1: {rho_m3_m1:.4f}；"
          f"M3 vs M1+: {rho_m3_m1p:.4f}")
    print(f"M1 tune AUC {roc_auc_score(ytu, preds_raw['M1']):.4f}；"
          f"M1+ tune AUC {roc_auc_score(ytu, preds_raw['M1+']):.4f}")


if __name__ == "__main__":
    main()
