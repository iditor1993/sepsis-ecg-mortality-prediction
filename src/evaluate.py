"""evaluate.py — 测试集一次性评估与时间外推验证（SAP 第九章，W6）。

H1 双门槛（估计性分析，SAP 第六章功效门槛未过后的预设表述）：
  M3 vs M1、M3 vs M1+ 的 ΔAUC 点估计 + 2000 次 bootstrap 95% CI + DeLong p；
  交集-并集联合判定 p_joint = max(p1, p2)（SAP 9.8）
H2：连续 NRI、类别 NRI（切点 10%/20%）、IDI，均附 bootstrap 95% CI
H3：时间外推（2017-2019）AUC 下降幅度；校准斜率 <0.8 时执行预设
    重校准预案（仅更新截距，报告更新前后指标）
H4：校准截距/斜率（Platt 前后）、Brier、DCA（阈值 5%-50% 净获益）
H1-H2 正式 p 值按 Holm 法校正（SAP 9.8）

模型应用协议与训练一致：LR 系按 20 套插补预测取均值；XGBoost 原生 NaN；
Platt 校准与约登阈值来自 tune 集（冻结，不重新选择）。

输出（患者级预测不入库）：
  results/test_predictions.parquet / results/temporal_predictions.parquet
  results/test_metrics.csv / results/h1_delong.csv / results/h2_nri_idi.csv
  results/h3_temporal.csv / results/dca_curve.csv
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import brier_score_loss, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED = 20260823
N_BOOT = 2000
MODELS = ["M0", "M1", "M2", "M3", "M1+", "M4"]


# ---------------------------------------------------------------------------
# 指标函数
# ---------------------------------------------------------------------------

def boot_delta_auc(y, p_a, p_b, rng, n_boot=N_BOOT):
    """ΔAUC = AUC(p_a) - AUC(p_b) 的 bootstrap 95% CI。"""
    n = len(y)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].sum() in (0, n):
            diffs[b] = np.nan
            continue
        diffs[b] = roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx])
    return float(np.nanmean(diffs)), (float(np.nanpercentile(diffs, 2.5)),
                                      float(np.nanpercentile(diffs, 97.5)))


def nri_idi(y, p_new, p_old, cutoffs=(0.10, 0.20)):
    """连续 NRI、类别 NRI（切点 10%/20%）、IDI。"""
    up = p_new > p_old
    ev, ne = y == 1, y == 0
    nri_cont = ((up & ev).mean() - ((~up) & ev).mean()
                + ((~up) & ne).mean() - (up & ne).mean())
    cat_new = np.digitize(p_new, cutoffs)
    cat_old = np.digitize(p_old, cutoffs)
    upc, dnc = cat_new > cat_old, cat_new < cat_old
    nri_cat = ((upc & ev).mean() - (dnc & ev).mean()
               + (dnc & ne).mean() - (upc & ne).mean())
    idi = ((p_new[ev].mean() - p_new[ne].mean())
           - (p_old[ev].mean() - p_old[ne].mean()))
    return nri_cont, nri_cat, idi


def boot_nri_idi(y, p_new, p_old, rng, n_boot=N_BOOT):
    n = len(y)
    stats = np.empty((n_boot, 3))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].sum() in (0, n):
            stats[b] = np.nan
            continue
        stats[b] = nri_idi(y[idx], p_new[idx], p_old[idx])
    out = {}
    for j, name in enumerate(["nri_cont", "nri_cat", "idi"]):
        out[name] = (float(np.nanmean(stats[:, j])),
                     float(np.nanpercentile(stats[:, j], 2.5)),
                     float(np.nanpercentile(stats[:, j], 97.5)),
                     float(np.nanstd(stats[:, j], ddof=1)))
    return out


def calib_intercept_slope(y, p):
    """校准截距（CITL：y ~ a + offset(logit p)）与校准斜率（y ~ a + b·logit p）。

    返回 (citl_intercept, slope, slope_fit_intercept)。
    """
    import statsmodels.api as sm
    x = logit(np.clip(p, 1e-6, 1 - 1e-6))
    m_off = sm.GLM(y, np.ones((len(y), 1)),
                   family=sm.families.Binomial(), offset=x).fit()
    m_slope = sm.GLM(y, sm.add_constant(x),
                     family=sm.families.Binomial()).fit()
    return (float(m_off.params[0]), float(m_slope.params[1]),
            float(m_slope.params[0]))


def dca_net_benefit(y, p, thresholds):
    prev = y.mean()
    nb = []
    for t in thresholds:
        pred = p >= t
        tp = (pred & (y == 1)).mean()
        fp = (pred & (y == 0)).mean()
        nb.append(tp - fp * t / (1 - t))
    return np.array(nb), prev - (1 - prev) * thresholds / (1 - thresholds)


# ---------------------------------------------------------------------------
# 预测
# ---------------------------------------------------------------------------

def build_matrix(datadir, parquet):
    df = pd.read_parquet(datadir / parquet)
    scores = pd.read_parquet(datadir / "clinical_scores.parquet")
    m1p = pd.read_parquet(datadir / "m1plus_features.parquet")
    mice = pd.read_parquet(datadir / "mice_lactate.parquet")
    df = df.merge(scores[["stay_id", "qsofa", "news", "mews"]], on="stay_id", how="left")
    df = df.merge(mice.drop(columns=["subject_id", "subset"]), on="stay_id", how="left")
    df["male"] = (df["gender"] == "M").astype(int)
    df = pd.get_dummies(df, columns=["infection_site"], prefix="site", dtype=int)
    m1p_feats = [c for c in m1p.columns if c not in ("subject_id", "stay_id")]
    df = df.merge(m1p[["stay_id"] + m1p_feats], on="stay_id", how="left")
    return df, m1p_feats


def predict_all(df, datadir, mice_cols):
    models_dir = datadir / "models"
    platt = joblib.load(models_dir / "platt.joblib")
    out = {}
    for name in ["M0", "M1", "M2", "M3"]:
        pack = joblib.load(models_dir / f"{name.lower()}_models.joblib")
        feats = pack["features"]
        p_sum = np.zeros(len(df))
        for k, mk in enumerate(pack["models"]):
            cols = [c if c != "lactate" else mice_cols[k] for c in feats]
            x = df[cols].copy()
            x.columns = feats
            p_sum += mk["model"].predict_proba(mk["scaler"].transform(x))[:, 1]
        out[name] = p_sum / len(pack["models"])
    for name, fname in [("M1+", "m1plus_xgb.joblib"), ("M4", "m4_xgb.joblib")]:
        pack = joblib.load(models_dir / fname)
        out[name] = pack["model"].predict_proba(df[pack["features"]])[:, 1]
    raw = dict(out)
    cal = {name: expit(platt[name][0] * logit(np.clip(p, 1e-6, 1 - 1e-6)) + platt[name][1])
           for name, p in out.items()}
    return raw, cal


def delong_p(y, p_a, p_b):
    import sys
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from power_delong_sim import delong_test
    preds = np.column_stack([p_a, p_b])
    return delong_test(preds, y)


# ---------------------------------------------------------------------------


def main() -> None:
    datadir = REPO_ROOT / "data"
    results_dir = REPO_ROOT / "results"
    rng = np.random.default_rng(SEED)
    mice_cols = [f"lactate_m{k:02d}" for k in range(1, 21)]

    dev, _ = build_matrix(datadir, "features_dev.parquet")
    te = dev[dev["subset"] == "test"].reset_index(drop=True)
    temporal, _ = build_matrix(datadir, "features_temporal.parquet")
    temporal = temporal.reset_index(drop=True)
    yte = te["death_28d"].astype(int).to_numpy()
    ytp = temporal["death_28d"].astype(int).to_numpy()
    print(f"测试集 {len(te):,}（事件 {yte.sum()}）；时间外推 {len(temporal):,}（事件 {ytp.sum()}）")

    raw_te, cal_te = predict_all(te, datadir, mice_cols)
    raw_tp, cal_tp = predict_all(temporal, datadir, mice_cols)

    pred_te = te[["subject_id", "stay_id"]].copy(); pred_te["y"] = yte
    pred_tp = temporal[["subject_id", "stay_id"]].copy(); pred_tp["y"] = ytp
    for m in MODELS:
        pred_te[f"{m}_raw"], pred_te[f"{m}_cal"] = raw_te[m], cal_te[m]
        pred_tp[f"{m}_raw"], pred_tp[f"{m}_cal"] = raw_tp[m], cal_tp[m]
    pred_te.to_parquet(results_dir / "test_predictions.parquet", index=False)
    pred_tp.to_parquet(results_dir / "temporal_predictions.parquet", index=False)

    # ---- AUC（测试 + 时间外推，bootstrap CI）----
    auc_rows = []
    for m in MODELS:
        n = len(yte)
        stats = np.empty(N_BOOT)
        for b in range(N_BOOT):
            idx = rng.integers(0, n, n)
            if yte[idx].sum() in (0, n):
                stats[b] = np.nan
                continue
            stats[b] = roc_auc_score(yte[idx], cal_te[m][idx])
        auc_rows.append({"model": m,
                         "test_auc": roc_auc_score(yte, cal_te[m]),
                         "auc_lo": float(np.nanpercentile(stats, 2.5)),
                         "auc_hi": float(np.nanpercentile(stats, 97.5)),
                         "temporal_auc": roc_auc_score(ytp, cal_tp[m]),
                         "auc_drop": roc_auc_score(yte, cal_te[m]) - roc_auc_score(ytp, cal_tp[m])})
    auc_df = pd.DataFrame(auc_rows)
    auc_df.to_csv(results_dir / "test_metrics.csv", index=False)
    print("\n[AUC（测试 / 时间外推）]")
    print(auc_df.round(4).to_string(index=False))

    # ---- H1 双门槛 ----
    h1_rows = []
    for comp in ["M1", "M1+"]:
        p_d, delta_d = delong_p(yte, cal_te["M3"], cal_te[comp])
        mean_d, (lo, hi) = boot_delta_auc(yte, cal_te["M3"], cal_te[comp], rng)
        h1_rows.append({"comparison": f"M3 vs {comp}", "delta_auc": delta_d,
                        "ci_lo": lo, "ci_hi": hi, "delong_p": p_d})
    h1 = pd.DataFrame(h1_rows)
    p_joint = float(h1["delong_p"].max())
    h1["p_joint_iut"] = p_joint
    h1.to_csv(results_dir / "h1_delong.csv", index=False)
    print("\n[H1 双门槛（估计性分析）]")
    print(h1.round(4).to_string(index=False))

    # ---- H2 NRI/IDI ----
    h2_rows = []
    for comp in ["M1", "M1+"]:
        r = boot_nri_idi(yte, cal_te["M3"], cal_te[comp], rng)
        for key, label in [("nri_cont", "连续NRI"), ("nri_cat", "类别NRI"), ("idi", "IDI")]:
            est, lo, hi, se = r[key]
            z = est / se if se > 0 else 0.0
            from scipy.stats import norm
            h2_rows.append({"comparison": f"M3 vs {comp}", "metric": label,
                            "estimate": est, "ci_lo": lo, "ci_hi": hi,
                            "p": 2 * (1 - norm.cdf(abs(z)))})
    h2 = pd.DataFrame(h2_rows)
    h2.to_csv(results_dir / "h2_nri_idi.csv", index=False)
    print("\n[H2 重分类（M3 vs M1 / M1+）]")
    print(h2.round(4).to_string(index=False))

    # ---- Holm 校正（H1 joint + H2 连续NRI 两比较）----
    ps = sorted([("H1", p_joint)]
                + [("H2_" + r["comparison"], r["p"]) for r in h2_rows
                   if r["metric"] == "连续NRI"], key=lambda t: t[1])
    m = len(ps)
    holm = {name: min(1.0, p * (m - i)) for i, (name, p) in enumerate(ps)}
    print("\n[Holm 校正 p] " + json.dumps({k: round(v, 4) for k, v in holm.items()}))
    with open(results_dir / "holm_p.json", "w") as f:
        json.dump({"raw": dict(ps), "holm": holm}, f, indent=2)

    # ---- H4 校准与 DCA ----
    cal_rows, dca_frames = [], []
    thr_grid = np.arange(0.05, 0.5001, 0.01)
    for m in ["M1", "M3", "M1+"]:
        for tag, p in [("raw", raw_te[m]), ("cal", cal_te[m])]:
            inter, slope, inter2 = calib_intercept_slope(yte, p)
            cal_rows.append({"model": m, "stage": tag,
                             "calib_intercept": inter, "calib_slope": slope,
                             "brier": brier_score_loss(yte, p)})
        nb_m, nb_all = dca_net_benefit(yte, cal_te[m], thr_grid)
        dca_frames.append(pd.DataFrame({"threshold": thr_grid, "model": m,
                                        "net_benefit": nb_m, "treat_all": nb_all}))
    cal_df = pd.DataFrame(cal_rows)
    cal_df.to_csv(results_dir / "calibration.csv", index=False)
    dca_df = pd.concat(dca_frames)
    dca_df.to_csv(results_dir / "dca_curve.csv", index=False)
    print("\n[校准（测试集，Platt 前/后）]")
    print(cal_df.round(4).to_string(index=False))

    # ---- H3 时间外推 + 重校准预案 ----
    h3_rows = []
    for m in ["M1", "M3", "M1+"]:
        inter, slope, inter2 = calib_intercept_slope(ytp, cal_tp[m])
        row = {"model": m, "temporal_auc": roc_auc_score(ytp, cal_tp[m]),
               "calib_intercept": inter, "calib_slope": slope,
               "brier": brier_score_loss(ytp, cal_tp[m])}
        if slope < 0.8:
            # 预设重校准：仅更新截距（斜率固定 1）
            p_recal = expit(logit(np.clip(cal_tp[m], 1e-6, 1 - 1e-6))
                            + (logit(np.clip(ytp.mean(), 1e-6, 1 - 1e-6))
                               - np.mean(logit(np.clip(cal_tp[m], 1e-6, 1 - 1e-6)))))
            row["recal_brier"] = brier_score_loss(ytp, p_recal)
        h3_rows.append(row)
    h3 = pd.DataFrame(h3_rows)
    h3.to_csv(results_dir / "h3_temporal.csv", index=False)
    print("\n[H3 时间外推（校准 + 预案）]")
    print(h3.round(4).to_string(index=False))

    print("\n输出: results/{test_metrics,h1_delong,h2_nri_idi,calibration,dca_curve,h3_temporal,holm_p}.csv/json")


if __name__ == "__main__":
    main()
