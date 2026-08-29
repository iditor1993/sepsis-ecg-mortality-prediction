"""train_m5.py — M5 端到端微调（SAP 8.2，探索性）。

V14 预训练编码器解冻末两层（末层 Conv1D + latent Dense），联合临床分支：
  信号(2500,1) -> V14 encoder -> z16 ┐
                                      concat -> Dense(32,relu) -> Dense(1,sigmoid)
  临床特征（M1 特征集，标准化）  -----/
优化：双优化器（编码器 lr=1e-4，分类头 lr=1e-3），batch 64，早停 patience 10
（tune AUC）。CPU 训练（Windows 原生 TF 无 GPU）。探索性模型，结论以 M3 为准。

输出：data/models/m5_model.keras、results/m5_results.csv（测试集 AUC 与
      ΔAUC vs M1 + bootstrap CI）。
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED = 20260823
BATCH, MAX_EPOCH, PATIENCE = 64, 50, 10
COV_COLS = ["admission_age", "male", "charlson_comorbidity_index",
            "admission_emergency", "pre_icu_los_h", "mech_vent_24h", "vaso_24h"]
SCORE_COLS = ["sofa_score", "qsofa", "news", "mews"]


def main() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    from tensorflow import keras
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    datadir = REPO_ROOT / "data"
    results_dir = REPO_ROOT / "results"
    from features_trackA import V14_ENCODER, SIGNAL_LEN
    from sensitivity_analyses import boot_delta, build

    # 信号（与 cohort_ecg 行序一致）
    signals = np.load(datadir / "features_trackA_signals.npy")
    index = pd.read_csv(datadir / "features_trackA_index.csv")

    df = build(datadir)  # features_dev
    df = df.merge(index.reset_index()[["stay_id", "index"]], on="stay_id", how="left")
    site_cols = sorted(c for c in df.columns if c.startswith("site_"))
    feats = SCORE_COLS + ["lactate"] + COV_COLS + site_cols
    mice = pd.read_parquet(datadir / "mice_lactate.parquet")
    lac_mean = mice[[c for c in mice.columns if c.startswith("lactate_m")]].mean(axis=1)
    lac_map = pd.Series(lac_mean.values, index=mice["stay_id"].values)
    df["lactate"] = df["lactate"].fillna(df["stay_id"].map(lac_map))

    tr, tu, te = (df["subset"] == s for s in ("train", "tune", "test"))
    sig_idx = df["index"].astype(int).to_numpy()
    x_sig = {"train": signals[sig_idx[tr.to_numpy()]],
             "tune": signals[sig_idx[tu.to_numpy()]],
             "test": signals[sig_idx[te.to_numpy()]]}
    sc = StandardScaler().fit(df.loc[tr, feats])
    x_cli = {s: sc.transform(df.loc[m, feats])
             for s, m in [("train", tr), ("tune", tu), ("test", te)]}
    y = {s: df.loc[m, "death_28d"].astype(int).to_numpy().astype(np.float32)
         for s, m in [("train", tr), ("tune", tu), ("test", te)]}
    n_cli = len(feats)
    print(f"[M5] train {len(y['train']):,} / tune {len(y['tune']):,} / "
          f"test {len(y['test']):,}；临床特征 {n_cli} 维")

    # ---- 模型 ----
    encoder = keras.models.load_model(V14_ENCODER)
    for layer in encoder.layers:
        layer.trainable = False
    # 解冻末两层：末层 Conv1D + latent Dense
    encoder.get_layer("latent").trainable = True
    conv_layers = [l for l in encoder.layers if isinstance(l, keras.layers.Conv1D)]
    conv_layers[-1].trainable = True

    sig_in = keras.Input(shape=(SIGNAL_LEN, 1), name="ecg")
    cli_in = keras.Input(shape=(n_cli,), name="clinical")
    z = encoder(sig_in)
    h = keras.layers.Concatenate()([z, cli_in])
    h = keras.layers.Dense(32, activation="relu")(h)
    out = keras.layers.Dense(1, activation="sigmoid")(h)
    model = keras.Model([sig_in, cli_in], out)

    enc_vars = [v for l in encoder.layers if l.trainable for v in l.trainable_variables]
    head_vars = [v for v in model.trainable_variables if all(v is not e for e in enc_vars)]
    opt_enc = keras.optimizers.Adam(1e-4)
    opt_head = keras.optimizers.Adam(1e-3)
    loss_fn = keras.losses.BinaryCrossentropy()

    def batch_iter(sig, cli, yb, shuffle=True):
        n = len(yb)
        idx = np.random.permutation(n) if shuffle else np.arange(n)
        for s in range(0, n, BATCH):
            b = idx[s:s + BATCH]
            yield sig[b], cli[b], yb[b]

    best_auc, best_weights, patience = -1.0, None, 0
    for ep in range(1, MAX_EPOCH + 1):
        tot, cnt = 0.0, 0
        for xs, xc, yb in batch_iter(x_sig["train"], x_cli["train"], y["train"]):
            with tf.GradientTape() as tape:
                p = model([xs, xc], training=True)
                loss = loss_fn(yb[:, None], p)
            grads = tape.gradient(loss, enc_vars + head_vars)
            opt_enc.apply_gradients([(g, v) for g, v in zip(grads[:len(enc_vars)], enc_vars)
                                     if g is not None])
            opt_head.apply_gradients([(g, v) for g, v in zip(grads[len(enc_vars):], head_vars)
                                      if g is not None])
            tot += float(loss) * len(yb); cnt += len(yb)
        p_tu = model.predict([x_sig["tune"], x_cli["tune"]], batch_size=256, verbose=0)
        auc = roc_auc_score(y["tune"], p_tu.ravel())
        print(f"[M5] epoch {ep}: loss={tot / cnt:.4f} tune_auc={auc:.4f}", flush=True)
        if auc > best_auc:
            best_auc, best_weights, patience = auc, model.get_weights(), 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"[M5] 早停，最佳 tune_auc={best_auc:.4f}")
                break
    if best_weights:
        model.set_weights(best_weights)
    model.save(datadir / "models" / "m5_model.keras")

    p_te = model.predict([x_sig["test"], x_cli["test"]], batch_size=256, verbose=0).ravel()
    pred_te = pd.read_parquet(results_dir / "test_predictions.parquet")
    rng = np.random.default_rng(SEED)
    d, lo, hi = boot_delta(y["test"], p_te, pred_te["M1_cal"].to_numpy(), rng)
    auc_te = roc_auc_score(y["test"], p_te)
    res = pd.DataFrame([{"model": "M5", "test_auc": auc_te,
                         "delta_vs_m1": d, "ci_lo": lo, "ci_hi": hi,
                         "best_tune_auc": best_auc}])
    res.to_csv(results_dir / "m5_results.csv", index=False)
    print(f"\n[M5] test AUC {auc_te:.4f}（tune 最佳 {best_auc:.4f}）；"
          f"ΔAUC vs M1 {d:+.4f} ({lo:+.4f}~{hi:+.4f})")
    print("输出 -> results/m5_results.csv, data/models/m5_model.keras")


if __name__ == "__main__":
    main()
