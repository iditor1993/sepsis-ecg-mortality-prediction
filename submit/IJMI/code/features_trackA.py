"""features_trackA.py — Track A 主特征提取（SAP 5.1 节，W3）。

V14 一维 CNN 自编码器的 encoder 部分**冻结迁移**（权重不在本队列重训）：
  编码器来源：models/v14_ecg_encoder.keras（仓库内置；溯源与 SHA-256 见
    models/v14_ecg_encoder_provenance.md；无监督自编码器，于 V14 项目
    MIMIC 队列约 4 万条 Lead II 信号上训练，不涉及任何结局标签）
  结构：Input(2500,1) -> Conv1D(16,7,s2) -> Conv1D(32,5,s2) -> Conv1D(64,3,s5)
        -> GlobalAveragePooling1D -> Dense(16) => z1-z16
  预处理与 V14 训练时完全一致：Lead II（缺失回退 Lead I/首通道）->
    重采样至 250 Hz -> 截断/零填充至 2500 点（10 s）-> 零均值单位方差 -> NaN 填 0

注意：SAP 5.1 表述为"Lead II 8 s 片段"，V14 编码器原生输入为 10 s@250 Hz；
      冻结迁移须匹配训练输入规格，故按 10 s 执行（本队列 ECG 均为 10 s），
      SAP 文本已由 V1.3 修订更正。

流程：多进程读取并预处理波形（缓存 .npy）-> 批量推理 -> 输出
      data/features_trackA.parquet（subject_id, stay_id, z1-z16, load_ok）。
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample

REPO_ROOT = Path(__file__).resolve().parent.parent
ECG_ROOT = Path("E:/clinical_research/MIMIC_IV_3.1/ecg")
# 仓库内置编码器（自 V14 项目逐字节复制；溯源见 models/v14_ecg_encoder_provenance.md）
V14_ENCODER = REPO_ROOT / "models" / "v14_ecg_encoder.keras"
SIGNAL_LEN = 2500  # 10 s @ 250 Hz（V14 原生输入规格）
LATENT_DIM = 16


def _load_lead2(task: tuple) -> tuple:
    """读取一条记录并预处理为 (SIGNAL_LEN,) float32；失败返回 None。

    与 V14 scripts/v14_extract_ecg.py 的 load_lead2 完全一致。
    """
    subject_id, stay_id, rel_path = task
    try:
        record = wfdb.rdrecord(str(ECG_ROOT / rel_path))
        if "II" in record.sig_name:
            sig = record.p_signal[:, record.sig_name.index("II")]
        elif "I" in record.sig_name:
            sig = record.p_signal[:, record.sig_name.index("I")]
        else:
            sig = record.p_signal[:, 0]
        sig = np.asarray(sig, dtype=float)
        if np.all(np.isnan(sig)):
            return subject_id, stay_id, None
        if record.fs == 500:
            sig = resample(sig, len(sig) // 2)
        elif record.fs != 250:
            sig = resample(sig, int(len(sig) * 250 / record.fs))
        if len(sig) < SIGNAL_LEN:
            sig = np.pad(sig, (0, SIGNAL_LEN - len(sig)), mode="constant")
        else:
            sig = sig[:SIGNAL_LEN]
        sig = (sig - np.nanmean(sig)) / (np.nanstd(sig) + 1e-8)
        sig = np.nan_to_num(sig, nan=0.0)
        return subject_id, stay_id, sig.astype(np.float32)
    except Exception:
        return subject_id, stay_id, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Track A 特征提取（V14 冻结迁移）")
    parser.add_argument("--datadir", default=str(REPO_ROOT / "data"))
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="仅处理前 N 条（冒烟测试）")
    parser.add_argument("--recompute", action="store_true", help="忽略信号缓存重算")
    args = parser.parse_args()

    datadir = Path(args.datadir)
    cohort = pd.read_parquet(datadir / "cohort_ecg.parquet")
    if args.limit:
        cohort = cohort.head(args.limit)
    n = len(cohort)
    print(f"[trackA] 待提取 {n:,} 条 ECG；编码器: {V14_ENCODER.name}")

    # ---- 阶段 1：波形读取与预处理（多进程 + npy 缓存）----
    sig_path = datadir / "features_trackA_signals.npy"
    idx_path = datadir / "features_trackA_index.csv"
    if sig_path.exists() and idx_path.exists() and not args.recompute and not args.limit:
        signals = np.load(sig_path)
        index = pd.read_csv(idx_path)
        print(f"[trackA] 复用信号缓存 {signals.shape}")
    else:
        tasks = list(
            cohort[["subject_id", "stay_id", "ecg_path"]].itertuples(index=False, name=None)
        )
        signals = np.full((n, SIGNAL_LEN), np.nan, dtype=np.float32)
        ok_flags = np.zeros(n, dtype=bool)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, (sid, stid, sig) in enumerate(
                pool.map(_load_lead2, tasks, chunksize=32), 1
            ):
                if sig is not None:
                    signals[i - 1] = sig
                    ok_flags[i - 1] = True
                if i % 2000 == 0 or i == n:
                    print(f"[trackA] 读取 {i:,}/{n:,}")
        index = cohort[["subject_id", "stay_id"]].copy()
        index["load_ok"] = ok_flags
        if not args.limit:
            np.save(sig_path, signals)
            index.to_csv(idx_path, index=False)
            print(f"[trackA] 信号缓存 -> {sig_path}")

    n_ok = int(np.isfinite(signals[:, 0]).sum())
    print(f"[trackA] 预处理成功 {n_ok:,}/{n:,}")

    # ---- 阶段 2：冻结编码器推理（GPU）----
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf  # 延迟导入：避免多进程 worker 加载 TF
    from tensorflow import keras

    encoder = keras.models.load_model(V14_ENCODER)
    valid = np.isfinite(signals[:, 0])
    z = np.full((n, LATENT_DIM), np.nan, dtype=np.float32)
    x = signals[valid].reshape(-1, SIGNAL_LEN, 1)
    z[valid] = encoder.predict(x, batch_size=args.batch_size, verbose=1).astype(np.float32)

    feats = pd.DataFrame(z, columns=[f"z{i}" for i in range(1, LATENT_DIM + 1)])
    out = pd.concat([cohort[["subject_id", "stay_id"]].reset_index(drop=True), feats], axis=1)
    out["load_ok"] = valid
    if not args.limit:
        out.to_parquet(datadir / "features_trackA.parquet", index=False)
        print(f"[trackA] 特征 -> {datadir / 'features_trackA.parquet'}")

    print("\n[trackA] z 向量摘要（有效行）:")
    print(out[[f"z{i}" for i in range(1, 5)]].describe().loc[
        ["mean", "std", "min", "max"]].to_string())
    print(f"NaN 特征行数: {int((~valid).sum())}")


if __name__ == "__main__":
    main()
