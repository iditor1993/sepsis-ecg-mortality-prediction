"""sensitivity_s1.py — S1：ECG 链接窗口改为 [t0-48h, t0)（纯预测窗，SAP 9.9）。

在锁定队列（16,499 例）与冻结划分不变的前提下，将 ECG 替换为 t0 前 48h 内
（严格早于 t0）距 t0 最近且质控合格的一份，重新提取 Track A 潜向量并重训
M1/M3，评估测试集 ΔAUC。MICE 乳酸插补沿用主分析（同一批患者）。
QC 阈值沿用冻结版 qc_config.yaml；新增波形读取仅针对此前未测的 study。

输出：data/s1_link.parquet、data/s1_features_trackA.parquet、
      results/sensitivity_s1.csv
"""

from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
ECG_ROOT = Path("E:/clinical_research/MIMIC_IV_3.1/ecg")
SEED = 20260823
M = 20


def phase_link(datadir: Path) -> pd.DataFrame:
    out = datadir / "s1_link.parquet"
    if out.exists():
        return pd.read_parquet(out)
    cohort = pd.read_parquet(datadir / "cohort_ecg.parquet")
    records = pd.read_csv(ECG_ROOT / "record_list.csv", parse_dates=["ecg_time"])
    cand = cohort[["subject_id", "stay_id", "t0"]].merge(records, on="subject_id", how="left")
    dt_h = (cand["ecg_time"] - cand["t0"]).dt.total_seconds() / 3600.0
    in_win = cand[(dt_h < 0) & (dt_h >= -48)].copy()  # [t0-48h, t0)
    in_win["abs_diff_h"] = -dt_h[(dt_h < 0) & (dt_h >= -48)]
    in_win = in_win.sort_values(["abs_diff_h", "ecg_time", "study_id"])
    best = in_win.drop_duplicates("subject_id", keep="first")[
        ["subject_id", "study_id", "path", "ecg_time", "abs_diff_h"]]
    best = best.rename(columns={"path": "ecg_path"})
    link = cohort[["subject_id", "stay_id"]].merge(best, on="subject_id", how="left")
    link.to_parquet(out, index=False)
    return link


def phase_qc(link: pd.DataFrame, datadir: Path, workers: int) -> pd.DataFrame:
    """合并缓存与新增 QC 指标，应用冻结阈值，返回 S1 合格标记。"""
    from ecg_link import _qc_one
    qc = yaml.safe_load((REPO_ROOT / "src" / "qc_config.yaml").read_text(encoding="utf-8"))
    cache = pd.read_parquet(datadir / "ecg_qc_metrics.parquet")
    have = set(zip(cache["subject_id"], cache["study_id"]))
    todo = link[link["study_id"].notna()][["subject_id", "study_id", "ecg_path"]]
    mask_new = ~todo.apply(lambda r: (r["subject_id"], r["study_id"]) in have, axis=1)
    new_rows = todo[mask_new]
    if len(new_rows):
        print(f"[S1] 新增波形 QC {len(new_rows):,} 条")
        tasks = list(new_rows.itertuples(index=False, name=None))
        rows = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, row in enumerate(pool.map(_qc_one, tasks, chunksize=32), 1):
                rows.append(row)
                if i % 2000 == 0 or i == len(tasks):
                    print(f"[S1-QC] {i:,}/{len(tasks):,}", flush=True)
        new_metrics = pd.DataFrame(rows)
        cache = pd.concat([cache, new_metrics], ignore_index=True)
        cache.to_parquet(datadir / "ecg_qc_metrics.parquet", index=False)
    m = link.merge(cache, on=["subject_id", "study_id"], how="left")
    ok = (m["study_id"].notna() & m["read_error"].isna()
          & (m["fs"] == qc["expected_fs"])
          & (m["duration_s"] >= qc["min_duration_s"])
          & (m["n_sig"] == 12)
          & (m["n_missing_leads"] <= qc["max_missing_leads"])
          & (m["clip_run_ms_max"] <= qc["clip_run_ms"])
          & (m["wander_mv_max"] <= qc["baseline_wander_mv"])
          & (m["hf_rms_mv_max"] <= qc["hf_noise_rms_mv"]))
    m["s1_ok"] = ok
    return m


def phase_features(qc_df: pd.DataFrame, datadir: Path, workers: int) -> pd.DataFrame:
    from features_trackA import _load_lead2, V14_ENCODER, SIGNAL_LEN, LATENT_DIM
    out_path = datadir / "s1_features_trackA.parquet"
    if out_path.exists():
        return pd.read_parquet(out_path)
    todo = qc_df[qc_df["s1_ok"]][["subject_id", "stay_id", "ecg_path"]]
    tasks = list(todo.itertuples(index=False, name=None))
    signals = np.zeros((len(todo), SIGNAL_LEN), dtype=np.float32)
    ok = np.zeros(len(todo), dtype=bool)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (sid, stid, sig) in enumerate(pool.map(_load_lead2, tasks, chunksize=32), 1):
            if sig is not None:
                signals[i - 1] = sig
                ok[i - 1] = True
            if i % 2000 == 0 or i == len(tasks):
                print(f"[S1-feat] 读取 {i:,}/{len(tasks):,}", flush=True)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf  # noqa: F401
    from tensorflow import keras
    encoder = keras.models.load_model(V14_ENCODER)
    z = np.full((len(todo), LATENT_DIM), np.nan, dtype=np.float32)
    z[ok] = encoder.predict(signals[ok].reshape(-1, SIGNAL_LEN, 1),
                            batch_size=256, verbose=0)
    feats = pd.DataFrame(z, columns=[f"z{i}" for i in range(1, LATENT_DIM + 1)])
    out = pd.concat([todo[["subject_id", "stay_id"]].reset_index(drop=True), feats], axis=1)
    out["load_ok"] = ok
    out.to_parquet(out_path, index=False)
    return out


def main() -> None:
    datadir = REPO_ROOT / "data"
    results_dir = REPO_ROOT / "results"
    workers = max(1, (os.cpu_count() or 4) - 2)
    rng = np.random.default_rng(SEED)

    link = phase_link(datadir)
    n_linked = int(link["study_id"].notna().sum())
    print(f"[S1] [t0-48h, t0) 窗内可链接 {n_linked:,}/{len(link):,}")

    qc_df = phase_qc(link, datadir, workers)
    n_ok = int(qc_df["s1_ok"].sum())
    print(f"[S1] 质控合格 {n_ok:,}（队列保留率 {n_ok / len(link):.1%}）")

    s1_feats = phase_features(qc_df, datadir, workers)

    # ---- 重训（S1 队列 = 锁定队列 ∩ s1_ok；划分沿用锁定版）----
    from sensitivity_s6_s7 import lasso_mice, lr_mice
    from sensitivity_analyses import COV_COLS, SCORE_COLS, build
    z_cols = [f"z{i}" for i in range(1, 17)]
    df = build(datadir)
    df = df.drop(columns=[c for c in z_cols if c in df.columns])
    df = df.merge(s1_feats[["stay_id"] + z_cols], on="stay_id", how="left")
    keep = df["stay_id"].isin(qc_df.loc[qc_df["s1_ok"], "stay_id"]) & df["z1"].notna()
    site_cols = sorted(c for c in df.columns if c.startswith("site_"))
    cov = COV_COLS + site_cols
    mice_cols = [f"lactate_m{k:02d}" for k in range(1, M + 1)]
    tr = (df["subset"] == "train") & keep
    te = (df["subset"] == "test") & keep
    ytr = df.loc[tr, "death_28d"].astype(int).to_numpy()
    yte = df.loc[te, "death_28d"].astype(int).to_numpy()
    print(f"[S1] train {tr.sum():,} / test {te.sum():,}（事件 {yte.sum()}）")

    f_m1 = SCORE_COLS + ["lactate"] + cov
    f_m3 = f_m1 + z_cols
    p1 = lr_mice(df, f_m1, tr, te, ytr, mice_cols)
    p3 = lasso_mice(df, f_m3, tr, te, ytr, mice_cols)
    d, lo, hi = __import__("sensitivity_analyses").boot_delta(yte, p3, p1, rng)
    res = pd.DataFrame([{
        "id": "S1", "内容": "ECG 窗口改 [t0-48h, t0) 纯预测窗",
        "n_test": int(te.sum()), "test_auc_m3": roc_auc_score(yte, p3),
        "test_auc_m1": roc_auc_score(yte, p1),
        "delta_auc": d, "ci_lo": lo, "ci_hi": hi}])
    res.to_csv(results_dir / "sensitivity_s1.csv", index=False)
    print(f"\n[S1] M3 {roc_auc_score(yte, p3):.4f} vs M1 {roc_auc_score(yte, p1):.4f}，"
          f"ΔAUC {d:+.4f} ({lo:+.4f}~{hi:+.4f})")
    print("输出 -> results/sensitivity_s1.csv")


if __name__ == "__main__":
    main()
