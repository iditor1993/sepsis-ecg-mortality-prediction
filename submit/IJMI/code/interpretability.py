"""interpretability.py — 可解释性分析（SAP 第十一章，W7）。

1) SHAP：
   - M3（LASSO-LR）：20 套插补模型的线性 SHAP（系数 × 标准化特征离差）逐套
     计算后平均（MICE 一致）
   - M4（XGBoost）：TreeExplainer，测试集蜂群图
   - 输出特征重要性排序（含 z1-z16 各维度贡献）
2) Saliency：测试集随机 50 例 ECG（种子 20260823），对 V14 编码器中
   M3 权重绝对值最大的 z 维度计算输入梯度（梯度×输入），波形+saliency
   双联图存 results/figures/saliency/，供两名心电资质医师线下判读
   （判读表模板 results/saliency_review_template.csv，Kappa 线下计算）。

输出：results/shap_importance.csv、results/figures/shap_m3_bar.png、
      results/figures/shap_m4_beeswarm.png、results/figures/saliency/*.png、
      results/saliency_review_template.csv
"""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED = 20260823
COV_COLS = ["admission_age", "male", "charlson_comorbidity_index",
            "admission_emergency", "pre_icu_los_h", "mech_vent_24h", "vaso_24h"]
SCORE_COLS = ["sofa_score", "qsofa", "news", "mews"]
Z_COLS = [f"z{i}" for i in range(1, 17)]


def shap_m3(df, datadir, te, yte, results_dir):
    """M3 线性 SHAP（20 套插补平均）。"""
    pack = joblib.load(datadir / "models" / "m3_models.joblib")
    feats = pack["features"]
    mice_cols = [f"lactate_m{k:02d}" for k in range(1, 21)]
    sv_sum = np.zeros((te.sum(), len(feats)))
    for k, mk in enumerate(pack["models"]):
        cols = [c if c != "lactate" else mice_cols[k] for c in feats]
        xte = df.loc[te, cols].copy(); xte.columns = feats
        sc = mk["scaler"]
        xs = sc.transform(xte)
        coef = mk["model"].coef_.ravel()
        # 线性模型 SHAP = coef × (x_scaled - 背景均值)；标准化后背景均值为 0
        sv_sum += xs * coef
    sv = sv_sum / len(pack["models"])
    imp = pd.DataFrame({"feature": feats,
                        "mean_abs_shap": np.abs(sv).mean(axis=0)})
    imp = imp.sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(results_dir / "shap_m3_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    top = imp.head(20)
    ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="steelblue")
    ax.set_title("M3 特征重要性（|SHAP| 均值，20 套插补平均）")
    ax.set_xlabel("mean |SHAP|")
    fig.tight_layout()
    fig.savefig(results_dir / "figures" / "shap_m3_bar.png", dpi=200)
    plt.close(fig)
    return imp


def shap_m4(df, datadir, te, results_dir):
    import shap
    pack = joblib.load(datadir / "models" / "m4_xgb.joblib")
    feats = pack["features"]
    xte = df.loc[te, feats]
    explainer = shap.TreeExplainer(pack["model"])
    sv = explainer.shap_values(xte)
    imp = pd.DataFrame({"feature": feats,
                        "mean_abs_shap": np.abs(sv).mean(axis=0)})
    imp = imp.sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(results_dir / "shap_m4_importance.csv", index=False)
    plt.figure()
    shap.summary_plot(sv, xte, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(results_dir / "figures" / "shap_m4_beeswarm.png", dpi=200,
                bbox_inches="tight")
    plt.close()
    return imp


def saliency_50(df, datadir, te, results_dir):
    """50 例测试集 ECG 的梯度 saliency 图（V14 编码器 + M3 最大权重 z 维度）。"""
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    from features_trackA import V14_ENCODER, SIGNAL_LEN, _load_lead2

    # M3 各 z 维度的合并系数（20 套平均，标准化尺度）
    pack = joblib.load(datadir / "models" / "m3_models.joblib")
    feats = pack["features"]
    z_idx = [feats.index(c) for c in Z_COLS]
    coef = np.mean([mk["model"].coef_.ravel() for mk in pack["models"]], axis=0)
    z_coef = dict(zip(Z_COLS, coef[z_idx]))
    top_z = max(z_coef, key=lambda k: abs(z_coef[k]))
    z_dim = int(top_z[1:]) - 1
    print(f"[saliency] M3 权重最大的 z 维度: {top_z}（系数 {z_coef[top_z]:+.4f}）")

    te_df = df[te].merge(
        pd.read_parquet(datadir / "cohort_ecg.parquet",
                        columns=["stay_id", "ecg_path"]), on="stay_id", how="left")
    te_df = te_df[te_df["ecg_path"].notna()]
    sel = te_df.sample(50, random_state=SEED)
    encoder = tf.keras.models.load_model(V14_ENCODER)

    outdir = results_dir / "figures" / "saliency"
    outdir.mkdir(parents=True, exist_ok=True)
    review = []
    for _, row in sel.iterrows():
        sid, stid, rel = int(row["subject_id"]), int(row["stay_id"]), row["ecg_path"]
        res = _load_lead2((sid, stid, rel))
        sig = res[2]
        if sig is None:
            continue
        x = tf.convert_to_tensor(sig.reshape(1, SIGNAL_LEN, 1))
        with tf.GradientTape() as tape:
            tape.watch(x)
            z = encoder(x, training=False)[0, z_dim]
        grad = tape.gradient(z, x).numpy().ravel()
        sal = np.abs(grad * sig)
        sal = sal / (sal.max() + 1e-12)

        fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 1]})
        axes[0].plot(sig, lw=0.6, color="black")
        axes[0].set_title(f"study {Path(rel).name} | subject {sid} | Lead II "
                          f"| 28d死亡={int(row['death_28d'])}")
        sc = axes[1].scatter(np.arange(SIGNAL_LEN), sig, c=sal, cmap="hot",
                             s=1, vmin=0, vmax=1)
        axes[1].set_title(f"梯度×输入 saliency（{top_z}）")
        fig.colorbar(sc, ax=axes[1], label="相对 saliency")
        fig.tight_layout()
        fname = f"saliency_{sid}_{Path(rel).name}.png"
        fig.savefig(outdir / fname, dpi=150)
        plt.close(fig)
        review.append({"subject_id": sid, "stay_id": stid,
                       "ecg_file": fname, "death_28d": int(row["death_28d"]),
                       "目标维度": top_z,
                       "医师1_PQRST定位": "", "医师1_模式判读": "",
                       "医师2_PQRST定位": "", "医师2_模式判读": ""})
    pd.DataFrame(review).to_csv(results_dir / "saliency_review_template.csv",
                                index=False)
    print(f"[saliency] 生成 {len(review)} 例 -> {outdir}；判读模板已输出")


def main() -> None:
    datadir = REPO_ROOT / "data"
    results_dir = REPO_ROOT / "results"
    from sensitivity_analyses import build
    df = build(datadir)
    te = df["subset"] == "test"
    yte = df.loc[te, "death_28d"].astype(int).to_numpy()

    print("=== M3 SHAP ===")
    imp3 = shap_m3(df, datadir, te, yte, results_dir)
    print(imp3.head(10).round(4).to_string(index=False))
    print("\n=== M4 SHAP ===")
    imp4 = shap_m4(df, datadir, te, results_dir)
    print(imp4.head(10).round(4).to_string(index=False))
    print("\n=== Saliency 50 例 ===")
    saliency_50(df, datadir, te, results_dir)


if __name__ == "__main__":
    main()
