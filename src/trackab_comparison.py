"""trackab_comparison.py — Track A vs Track B 全模型对比（额外分析）。

模型体系在两轨上的定义：
  M0/M1/M1+：不含 ECG 特征，两轨完全相同（直接引用 Track A 既有结果）
  M2：ECG 潜向量（A: z1-z16 / B: tb1-tb32）LR
  M3：M1 特征 + 潜向量，LASSO（B 版即 S5，此处重训以获得配对预测）
  M4：M3 特征 XGBoost 网格（B 版新训）
  M5：端到端（A: V14 解冻微调+临床分支，已完成 0.8147；
      B: 2D-ResNet 从零训练权重初始化 + 临床分支端到端训练，新训）

评估：测试集 AUC；M2/M3/M4/M5 的 ΔAUC(B−A) 配对 bootstrap 95% CI（2000 次）。
输出：results/trackab_comparison.csv、results/trackB_predictions.parquet。
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED = 20260823
M = 20
TB_COLS = [f"tb{i}" for i in range(1, 33)]
COV_COLS = ["admission_age", "male", "charlson_comorbidity_index",
            "admission_emergency", "pre_icu_los_h", "mech_vent_24h", "vaso_24h"]
SCORE_COLS = ["sofa_score", "qsofa", "news", "mews"]


def boot_delta(y, pa, pb, rng, n_boot=2000):
    n, diffs = len(y), []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].sum() in (0, n):
            continue
        diffs.append(roc_auc_score(y[idx], pa[idx]) - roc_auc_score(y[idx], pb[idx]))
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def predict_m5_trackA(df, datadir, tr, te, cov):
    """重建 M5-TrackA 测试预测（模型已存；预处理与 train_m5.py 完全一致）。"""
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from tensorflow import keras

    signals = np.load(datadir / "features_trackA_signals.npy")
    index = pd.read_csv(datadir / "features_trackA_index.csv")
    df = df.merge(index.reset_index()[["stay_id", "index"]], on="stay_id", how="left")
    feats_cli = SCORE_COLS + ["lactate"] + cov
    mice = pd.read_parquet(datadir / "mice_lactate.parquet")
    lac_mean = mice[[c for c in mice.columns if c.startswith("lactate_m")]].mean(axis=1)
    lac_map = pd.Series(lac_mean.values, index=mice["stay_id"].values)
    df["lactate"] = df["lactate"].fillna(df["stay_id"].map(lac_map))
    sc = StandardScaler().fit(df.loc[tr, feats_cli])
    sig_idx = df["index"].astype(int).to_numpy()
    x_sig = signals[sig_idx[te.to_numpy()]]
    x_cli = sc.transform(df.loc[te, feats_cli])
    model = keras.models.load_model(datadir / "models" / "m5_model.keras")
    return model.predict([x_sig, x_cli], batch_size=256, verbose=0).ravel()


def main() -> None:
    datadir = REPO_ROOT / "data"
    results_dir = REPO_ROOT / "results"
    rng = np.random.default_rng(SEED)
    from sensitivity_s6_s7 import lasso_mice, lr_mice
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
    pred_a = pd.read_parquet(results_dir / "test_predictions.parquet")

    trackB_preds = pd.DataFrame({"stay_id": df.loc[te, "stay_id"], "y": yte})

    # ---- M2-TrackB：LR(tb1-32) ----
    sc = StandardScaler().fit(df.loc[tr, TB_COLS])
    m2b = LogisticRegression(max_iter=2000).fit(sc.transform(df.loc[tr, TB_COLS]), ytr)
    p_m2b = m2b.predict_proba(sc.transform(df.loc[te, TB_COLS]))[:, 1]
    trackB_preds["M2_TB"] = p_m2b
    auc_m2b = roc_auc_score(yte, p_m2b)
    print(f"[M2-TB] test AUC {auc_m2b:.4f}", flush=True)

    # ---- M3-TrackB：LASSO(M1 + tb32)，MICE 协议（重训以配对）----
    f_m1 = SCORE_COLS + ["lactate"] + cov
    f_m3b = f_m1 + TB_COLS
    p_m3b = lasso_mice(df, f_m3b, tr, te, ytr, mice_cols)
    trackB_preds["M3_TB"] = p_m3b
    auc_m3b = roc_auc_score(yte, p_m3b)
    print(f"[M3-TB] test AUC {auc_m3b:.4f}", flush=True)

    # ---- M4-TrackB：XGBoost(M1 + tb32) 网格 ----
    xtr, xtu, xte = df.loc[tr, f_m3b], df.loc[tu, f_m3b], df.loc[te, f_m3b]
    best = fit_xgb_grid(xtr, ytr, xtu, ytu, "M4-TB")
    p_m4b = best["model"].predict_proba(xte)[:, 1]
    trackB_preds["M4_TB"] = p_m4b
    auc_m4b = roc_auc_score(yte, p_m4b)
    joblib.dump({"features": f_m3b, "cfg": best["cfg"], "model": best["model"]},
                datadir / "models" / "m4_trackB_xgb.joblib")
    print(f"[M4-TB] test AUC {auc_m4b:.4f}（最佳 {best['cfg']}）", flush=True)

    # ---- M5-TrackB：2D-ResNet + 临床分支端到端 ----
    p_m5b, auc_m5b = train_m5_trackB(df, datadir, tr, tu, te, ytr, ytu, yte, cov)
    trackB_preds["M5_TB"] = p_m5b
    print(f"[M5-TB] test AUC {auc_m5b:.4f}", flush=True)

    # ---- M5-TrackA 测试预测重建（用于配对 CI；模型已存，推理确定）----
    p_m5a = predict_m5_trackA(df, datadir, tr, te, cov)
    auc_m5a = roc_auc_score(yte, p_m5a)
    print(f"[M5-TA] test AUC {auc_m5a:.4f}（复算）", flush=True)

    trackB_preds.to_parquet(results_dir / "trackB_predictions.parquet", index=False)

    # ---- 汇总表 ----
    rows = []
    ta = {"M0": 0.5938, "M1": 0.7968, "M1+": 0.8542, "M2": 0.6419,
          "M3": 0.7953, "M4": 0.8048, "M5": 0.8147}
    tb = {"M0": 0.5938, "M1": 0.7968, "M1+": 0.8542, "M2": auc_m2b,
          "M3": auc_m3b, "M4": auc_m4b, "M5": auc_m5b}
    paired = {"M2": "M2_TB", "M3": "M3_TB", "M4": "M4_TB", "M5": "M5_TB"}
    note = {"M0": "不含 ECG 特征，两轨相同", "M1": "不含 ECG 特征，两轨相同",
            "M1+": "不含 ECG 特征，两轨相同", "M2": "潜向量 LR",
            "M3": "M1+潜向量 LASSO", "M4": "M1+潜向量 XGBoost",
            "M5": "端到端+临床分支"}
    for m in ["M0", "M1", "M1+", "M2", "M3", "M4", "M5"]:
        row = {"模型": m, "Track A test AUC": round(ta[m], 4),
               "Track B test AUC": round(tb[m], 4), "说明": note[m]}
        if m in paired:
            p_tb = trackB_preds[paired[m]].to_numpy()
            p_ta = {"M2": pred_a["M2_cal"].to_numpy(),
                    "M3": pred_a["M3_cal"].to_numpy(),
                    "M4": pred_a["M4_cal"].to_numpy(),
                    "M5": p_m5a}[m]
            d, lo, hi = boot_delta(yte, p_tb, p_ta, rng)
            row.update({"ΔAUC(B-A)": round(d, 4), "95%CI": f"{lo:+.4f}~{hi:+.4f}"})
        else:
            row.update({"ΔAUC(B-A)": 0.0, "95%CI": "—"})
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(results_dir / "trackab_comparison.csv", index=False,
               encoding="utf-8-sig")
    print("\n=== Track A vs Track B 全模型对比（测试集）===")
    print(res.to_string(index=False))
    print("\n输出 -> results/trackab_comparison.csv")


def train_m5_trackB(df, datadir, tr, tu, te, ytr, ytu, yte, cov):
    """M5-TrackB：2D-ResNet 初始化 + 临床分支端到端（fp32 + 梯度裁剪）。"""
    import torch
    from torch.utils.data import DataLoader, Dataset

    from features_trackB import FEAT_DIM, N_LEADS, SIG_LEN, _build_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    signals = np.memmap(datadir / "features_trackB_signals.dat", dtype=np.float32,
                        mode="r", shape=(len(pd.read_csv(datadir / "features_trackB_index.csv")),
                                         N_LEADS, SIG_LEN))
    index = pd.read_csv(datadir / "features_trackB_index.csv")
    df = df.merge(index.reset_index()[["stay_id", "index"]], on="stay_id", how="left")
    feats_cli = SCORE_COLS + ["lactate"] + cov
    mice = pd.read_parquet(datadir / "mice_lactate.parquet")
    lac_mean = mice[[c for c in mice.columns if c.startswith("lactate_m")]].mean(axis=1)
    lac_map = pd.Series(lac_mean.values, index=mice["stay_id"].values)
    df["lactate"] = df["lactate"].fillna(df["stay_id"].map(lac_map))

    sc_cli = StandardScaler().fit(df.loc[tr, feats_cli])
    sig_idx = df["index"].astype(int).to_numpy()
    pos = {"train": np.where(tr.to_numpy())[0], "tune": np.where(tu.to_numpy())[0],
           "test": np.where(te.to_numpy())[0]}
    cli = {s: sc_cli.transform(df.loc[m, feats_cli]) for s, m in
           [("train", tr), ("tune", tu), ("test", te)]}
    y = {"train": ytr.astype(np.float32), "tune": ytu.astype(np.float32),
         "test": yte.astype(np.float32)}

    base = _build_model().to(device)
    base.load_state_dict(torch.load(datadir / "trackB_model.pt", map_location=device))

    class Fusion(torch.nn.Module):
        def __init__(self, base, n_cli):
            super().__init__()
            self.base = base
            self.head = torch.nn.Sequential(
                torch.nn.Linear(FEAT_DIM + n_cli, 32), torch.nn.ReLU(),
                torch.nn.Linear(32, 1))

        def forward(self, x_sig, x_cli):
            f = self.base.features(x_sig)
            return self.head(torch.cat([f, x_cli], dim=1)).squeeze(1)

    model = Fusion(base, len(feats_cli)).to(device)
    torch.manual_seed(SEED)
    enc_vars = list(base.parameters())
    head_vars = list(model.head.parameters())
    opt_e = torch.optim.Adam(enc_vars, lr=1e-4)
    opt_h = torch.optim.Adam(head_vars, lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    class DS(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            r = self.rows[i]
            return (torch.from_numpy(np.asarray(signals[sig_idx[r]])),
                    torch.from_numpy(cli["train"][i].copy()),
                    torch.tensor(y["train"][i]))

    loader = DataLoader(DS(pos["train"]), batch_size=64, shuffle=True,
                        num_workers=0, pin_memory=True)

    def predict(rows, cli_arr):
        model.eval()
        ps = []
        with torch.no_grad():
            for s in range(0, len(rows), 256):
                b = rows[s:s + 256]
                xs = torch.from_numpy(np.asarray(signals[sig_idx[b]])).unsqueeze(1).to(device)
                xc = torch.from_numpy(cli_arr[s:s + 256].copy()).float().to(device)
                ps.append(torch.sigmoid(model(xs, xc)).cpu().numpy())
        return np.concatenate(ps)

    best_auc, best_state, pat = -1.0, None, 0
    for ep in range(1, 51):
        model.train()
        tot, cnt = 0.0, 0
        for xs, xc, yb in loader:
            xs, xc, yb = (xs.unsqueeze(1).to(device),
                           xc.float().to(device), yb.to(device))
            loss = loss_fn(model(xs, xc), yb)
            opt_e.zero_grad(); opt_h.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_e.step(); opt_h.step()
            tot += loss.item() * len(yb); cnt += len(yb)
        auc_t = roc_auc_score(y["tune"], predict(pos["tune"], cli["tune"]))
        print(f"[M5-TB] epoch {ep}: loss={tot / cnt:.4f} tune_auc={auc_t:.4f}", flush=True)
        if auc_t > best_auc:
            best_auc, best_state, pat = auc_t, model.state_dict(), 0
        else:
            pat += 1
            if pat >= 10:
                print(f"[M5-TB] 早停，最佳 tune_auc={best_auc:.4f}")
                break
    if best_state:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), datadir / "models" / "m5_trackB_model.pt")
    p_te = predict(pos["test"], cli["test"])
    return p_te, roc_auc_score(yte, p_te)


if __name__ == "__main__":
    main()
