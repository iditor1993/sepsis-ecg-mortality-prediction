"""sensitivity_encoder_retrain.py — 敏感性分析：项目内从零重训编码器（额外分析）。

目的：验证主结论对 V14 编码器来源（前项目无监督预训练）的稳健性——
在本项目内用**同一架构、同一配方、仅 train 子集波形**从零训练自编码器，
提取 z'1-z16 并重跑 M2'/M3'，比较 ΔAUC 与主分析（M3 vs M1 = -0.0014）。

配方与 V14 scripts/v14_extract_ecg.py 完全一致：架构
Input(2500,1)->Conv1D(16,7,s2)->Conv1D(32,5,s2)->Conv1D(64,3,s5)->GAP->Dense(16)
+对称解码器；Adam(1e-3)，MSE，30 epoch，batch 64；种子 20260823。
波形复用 data/features_trackA_signals.npy 缓存（预处理与 Track A 相同）。
重训仅用 train 子集信号（10,346 条），无测试接触、无结局标签。

输出：results/sensitivity_encoder_retrain.csv；
      data/models/encoder_retrained.keras（不入库）。
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED = 20260823
M = 20
Z_COLS = [f"z{i}" for i in range(1, 17)]
COV_COLS = ["admission_age", "male", "charlson_comorbidity_index",
            "admission_emergency", "pre_icu_los_h", "mech_vent_24h", "vaso_24h"]
SCORE_COLS = ["sofa_score", "qsofa", "news", "mews"]
SIGNAL_LEN = 2500


def build_autoencoder():
    from tensorflow import keras
    inputs = keras.Input(shape=(SIGNAL_LEN, 1))
    x = keras.layers.Conv1D(16, 7, strides=2, padding="same", activation="relu")(inputs)
    x = keras.layers.Conv1D(32, 5, strides=2, padding="same", activation="relu")(x)
    x = keras.layers.Conv1D(64, 3, strides=5, padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    latent = keras.layers.Dense(16, name="latent")(x)
    d = keras.layers.Dense(125 * 32)(latent)
    d = keras.layers.Reshape((125, 32))(d)
    d = keras.layers.UpSampling1D(5)(d)
    d = keras.layers.Conv1D(32, 5, padding="same", activation="relu")(d)
    d = keras.layers.UpSampling1D(2)(d)
    d = keras.layers.Conv1D(16, 5, padding="same", activation="relu")(d)
    d = keras.layers.UpSampling1D(2)(d)
    outputs = keras.layers.Conv1D(1, 7, padding="same", activation="linear")(d)
    model = keras.Model(inputs, outputs, name="ecg_autoencoder_retrained")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


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
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    from tensorflow import keras
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    datadir = REPO_ROOT / "data"
    results_dir = REPO_ROOT / "results"
    rng = np.random.default_rng(SEED)
    from sensitivity_analyses import build
    from sensitivity_s6_s7 import lasso_mice, lr_mice

    signals = np.load(datadir / "features_trackA_signals.npy")
    index = pd.read_csv(datadir / "features_trackA_index.csv")
    df = build(datadir)
    df = df.merge(index.reset_index()[["stay_id", "index"]], on="stay_id", how="left")
    sig_idx = df["index"].astype(int).to_numpy()
    site_cols = sorted(c for c in df.columns if c.startswith("site_"))
    cov = COV_COLS + site_cols
    tr, te = df["subset"] == "train", df["subset"] == "test"
    ytr = df.loc[tr, "death_28d"].astype(int).to_numpy()
    yte = df.loc[te, "death_28d"].astype(int).to_numpy()
    mice_cols = [f"lactate_m{k:02d}" for k in range(1, M + 1)]

    # ---- 1) 从零重训自编码器（仅 train 子集信号）----
    x_train = signals[sig_idx[tr.to_numpy()]].reshape(-1, SIGNAL_LEN, 1)
    print(f"[retrain] 训练信号 {x_train.shape[0]:,} 条（仅 train 子集）")
    ae = build_autoencoder()
    ae.fit(x_train, x_train, epochs=30, batch_size=64, shuffle=True, verbose=0)
    encoder = keras.Model(ae.input, ae.get_layer("latent").output)
    encoder.save(datadir / "models" / "encoder_retrained.keras")

    # ---- 2) 全量提取 z' ----
    z = encoder.predict(signals.reshape(-1, SIGNAL_LEN, 1), batch_size=256, verbose=0)
    zdf = pd.DataFrame(z, columns=Z_COLS)
    zdf["stay_id"] = index["stay_id"].to_numpy()  # 信号缓存与 cohort_ecg 对齐
    df = df.drop(columns=Z_COLS).merge(zdf, on="stay_id", how="left")

    # ---- 3) 重跑 M2' / M3' ----
    rows = []
    f_m2 = Z_COLS
    sc = StandardScaler().fit(df.loc[tr, f_m2])
    m2 = LogisticRegression(max_iter=2000).fit(sc.transform(df.loc[tr, f_m2]), ytr)
    p_m2 = m2.predict_proba(sc.transform(df.loc[te, f_m2]))[:, 1]
    auc_m2 = roc_auc_score(yte, p_m2)
    print(f"[M2'] test AUC {auc_m2:.4f}（主分析 M2 0.6419）", flush=True)
    rows.append({"model": "M2'（内部重训编码器潜向量 LR）", "test_auc": auc_m2,
                 "comparison": "对照主分析 M2 0.6419", "delta_auc": np.nan,
                 "ci_lo": np.nan, "ci_hi": np.nan})

    f_m1 = SCORE_COLS + ["lactate"] + cov
    p_m1 = lr_mice(df, f_m1, tr, te, ytr, mice_cols)
    f_m3 = f_m1 + Z_COLS
    p_m3 = lasso_mice(df, f_m3, tr, te, ytr, mice_cols)
    auc_m1, auc_m3 = roc_auc_score(yte, p_m1), roc_auc_score(yte, p_m3)
    d, lo, hi = boot_delta(yte, p_m3, p_m1, rng)
    rows.append({"model": "M3'（内部重训编码器版 M3）", "test_auc": auc_m3,
                 "comparison": "M3' vs M1'（ΔAUC）", "delta_auc": d,
                 "ci_lo": lo, "ci_hi": hi})
    print(f"[M3'] test AUC {auc_m3:.4f} vs M1' {auc_m1:.4f}；"
          f"ΔAUC {d:+.4f} ({lo:+.4f}~{hi:+.4f})", flush=True)

    # 与主分析 M3（Track A）配对对比
    pred_te = pd.read_parquet(results_dir / "test_predictions.parquet")
    d2, lo2, hi2 = boot_delta(yte, p_m3, pred_te["M3_cal"].to_numpy(), rng)
    rows.append({"model": "M3' vs M3（主分析）", "test_auc": np.nan,
                 "comparison": "AUC(M3') - AUC(M3)", "delta_auc": d2,
                 "ci_lo": lo2, "ci_hi": hi2})
    print(f"[M3' vs M3] ΔAUC {d2:+.4f} ({lo2:+.4f}~{hi2:+.4f})", flush=True)

    pd.DataFrame(rows).to_csv(results_dir / "sensitivity_encoder_retrain.csv",
                              index=False, encoding="utf-8-sig")
    print("\n输出 -> results/sensitivity_encoder_retrain.csv")
    print("判读：若 M3' vs M1' 的 ΔAUC 仍围绕 0 且 CI 排除 0.02，"
          "则阴性结论对编码器来源稳健。")


if __name__ == "__main__":
    main()
