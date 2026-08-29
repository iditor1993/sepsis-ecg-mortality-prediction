"""features_trackB.py — Track B 基准特征提取（SAP 5.1 节，W3）。

Kwon 式二维残差 CNN（无公开预训练权重，按 SAP 5.1 于本队列从零训练，
对应 V1.2 所述"从零训练编码器"基准范式）：
  输入：12 导联 × 4000 点矩阵（8 s @ 500 Hz，取每份 ECG 前 8 s；
        导联按 I,II,III,aVR,aVL,aVF,V1-V6 顺序重排；逐导联 z 标准化，NaN 填 0）
  标签：28 天全因死亡（仅 train 子集参与训练，tune 子集早停选模；
        test/temporal 仅用于后续特征推理，不参与训练）
  结构：2D ResNet-18 式（stem Conv2d(12×7) 融合导联 + 4 阶段残差块），
        GAP 后 512 维笔层特征 → PCA 降维至 32 维（PCA 仅在 train 子集拟合）

流程：多进程读取波形 -> memmap 缓存 -> 训练 -> 全量推理 + PCA ->
      data/features_trackB.parquet（subject_id, stay_id, tb1-tb32, load_ok）。
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

REPO_ROOT = Path(__file__).resolve().parent.parent
ECG_ROOT = Path("E:/clinical_research/MIMIC_IV_3.1/ecg")
SEED = 20260823
LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]
N_LEADS, SIG_LEN = 12, 4000  # 8 s @ 500 Hz
FEAT_DIM, PCA_DIM = 512, 32


def _load_12lead(task: tuple) -> tuple:
    """读取一条记录 -> (12, 4000) float32；失败返回 None。"""
    subject_id, stay_id, rel_path = task
    try:
        rec = wfdb.rdrecord(str(ECG_ROOT / rel_path))
        x = np.asarray(rec.p_signal, dtype=float)
        cols = []
        for name in LEAD_ORDER:
            if name in rec.sig_name:
                cols.append(x[:, rec.sig_name.index(name)])
            else:
                cols.append(np.full(x.shape[0], np.nan))
        m = np.stack(cols, axis=0)[:, :SIG_LEN]
        if m.shape[1] < SIG_LEN:
            m = np.pad(m, ((0, 0), (0, SIG_LEN - m.shape[1])), mode="constant")
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(m, axis=1, keepdims=True)
            std = np.nanstd(m, axis=1, keepdims=True) + 1e-8
        m = np.nan_to_num((m - mean) / std, nan=0.0)
        return subject_id, stay_id, m.astype(np.float32)
    except Exception:
        return subject_id, stay_id, None


# ---------------------------------------------------------------------------
# 模型（torch 延迟导入，避免多进程 worker 加载）
# ---------------------------------------------------------------------------


def _build_model():
    import torch
    import torch.nn as nn

    class BasicBlock(nn.Module):
        def __init__(self, cin, cout, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(cin, cout, (1, 7), (1, stride), (0, 3), bias=False)
            self.bn1 = nn.BatchNorm2d(cout)
            self.conv2 = nn.Conv2d(cout, cout, (1, 5), (1, 1), (0, 2), bias=False)
            self.bn2 = nn.BatchNorm2d(cout)
            self.relu = nn.ReLU(inplace=True)
            self.down = None
            if stride != 1 or cin != cout:
                self.down = nn.Sequential(
                    nn.Conv2d(cin, cout, 1, (1, stride), bias=False),
                    nn.BatchNorm2d(cout),
                )

        def forward(self, x):
            idn = x if self.down is None else self.down(x)
            y = self.relu(self.bn1(self.conv1(x)))
            y = self.bn2(self.conv2(y))
            return self.relu(y + idn)

    class EcgResNet2D(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(1, 64, (N_LEADS, 7), (1, 2), (0, 3), bias=False),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.MaxPool2d((1, 2)),
            )
            self.stage1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
            self.stage2 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128))
            self.stage3 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256))
            self.stage4 = nn.Sequential(BasicBlock(256, 512, 2), BasicBlock(512, 512))
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.head = nn.Linear(FEAT_DIM, 1)

        def features(self, x):
            y = self.pool(self.stage4(self.stage3(self.stage2(self.stage1(self.stem(x))))))
            return torch.flatten(y, 1)  # (B, 512)

        def forward(self, x):
            return self.head(self.features(x)).squeeze(1)  # (B,)

    return EcgResNet2D()


def _phase_signals(cohort, datadir, workers, limit, recompute):
    """波形读取 -> memmap 缓存；返回 (signals_memmap或ndarray, load_ok)。"""
    sig_path = datadir / "features_trackB_signals.dat"
    idx_path = datadir / "features_trackB_index.csv"
    n = len(cohort)
    if sig_path.exists() and idx_path.exists() and not recompute and not limit:
        signals = np.memmap(sig_path, dtype=np.float32, mode="r",
                            shape=(n, N_LEADS, SIG_LEN))
        load_ok = pd.read_csv(idx_path)["load_ok"].to_numpy()
        print(f"[trackB] 复用信号缓存 {signals.shape}")
        return signals, load_ok

    tasks = list(cohort[["subject_id", "stay_id", "ecg_path"]].itertuples(index=False, name=None))
    signals = (np.zeros((n, N_LEADS, SIG_LEN), dtype=np.float32) if limit
               else np.memmap(sig_path, dtype=np.float32, mode="w+",
                              shape=(n, N_LEADS, SIG_LEN)))
    load_ok = np.zeros(n, dtype=bool)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (sid, stid, m) in enumerate(pool.map(_load_12lead, tasks, chunksize=16), 1):
            if m is not None:
                signals[i - 1] = m
                load_ok[i - 1] = True
            if i % 1000 == 0 or i == n:
                print(f"[trackB] 读取 {i:,}/{n:,}", flush=True)
    if isinstance(signals, np.memmap):
        signals.flush()
    if not limit:
        pd.DataFrame({"subject_id": cohort["subject_id"], "stay_id": cohort["stay_id"],
                      "load_ok": load_ok}).to_csv(idx_path, index=False)
    return signals, load_ok


def _phase_train(signals, load_ok, cohort, datadir, epochs, batch_size, limit):
    import torch
    from torch.utils.data import DataLoader, Dataset

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    meta = cohort[["subject_id", "stay_id"]].copy()
    meta = meta.merge(pd.read_csv(datadir / "splits.csv"), on=["subject_id", "stay_id"])
    out = pd.read_parquet(datadir / "outcomes.parquet")
    meta = meta.merge(out[["stay_id", "death_28d"]], on="stay_id")
    meta["row"] = np.arange(len(meta))

    class EcgDataset(Dataset):
        def __init__(self, frame):
            self.rows = frame["row"].to_numpy()
            self.labels = frame["death_28d"].astype(np.float32).to_numpy()

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            r = self.rows[i]
            return torch.from_numpy(np.asarray(signals[r])), torch.tensor(self.labels[i])

    tr = meta[(meta["subset"] == "train") & meta["death_28d"].notna()
              & load_ok[meta["row"].to_numpy()]]
    tu = meta[(meta["subset"] == "tune") & meta["death_28d"].notna()
              & load_ok[meta["row"].to_numpy()]]
    print(f"[trackB] 训练 {len(tr):,} / 调优 {len(tu):,}（阳性率 "
          f"{tr['death_28d'].mean():.1%}）")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    kw = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    tr_loader = DataLoader(EcgDataset(tr), shuffle=True, **kw)
    tu_loader = DataLoader(EcgDataset(tu), shuffle=False, **kw)

    from sklearn.metrics import roc_auc_score
    best_auc, best_state, patience = -1.0, None, 0
    log_rows = []
    for ep in range(1, epochs + 1):
        model.train()
        tot, cnt, n_bad = 0.0, 0, 0
        for x, y in tr_loader:
            x, y = x.unsqueeze(1).to(device), y.to(device)
            loss = loss_fn(model(x), y)
            if not torch.isfinite(loss):  # 数值安全：跳过异常批次
                n_bad += 1
                opt.zero_grad()
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            tot += loss.item() * len(y); cnt += len(y)
        if cnt == 0:
            raise RuntimeError(f"epoch {ep}: 全部批次损失非有限值，训练发散")
        model.eval(); ps, ys = [], []
        with torch.no_grad():
            for x, y in tu_loader:
                p = model(x.unsqueeze(1).to(device))
                ps.append(torch.sigmoid(p).float().cpu().numpy()); ys.append(y.numpy())
        ps, ys = np.concatenate(ps), np.concatenate(ys)
        if not np.isfinite(ps).all():
            raise RuntimeError(f"epoch {ep}: 调优集预测出现 NaN/Inf，训练发散")
        auc = roc_auc_score(ys, ps)
        log_rows.append({"epoch": ep, "train_loss": tot / cnt, "tune_auc": auc,
                         "skipped_batches": n_bad})
        print(f"[trackB] epoch {ep}: loss={tot / cnt:.4f} tune_auc={auc:.4f}"
              + (f" (skip {n_bad})" if n_bad else ""), flush=True)
        if auc > best_auc:
            best_auc, best_state, patience = auc, model.state_dict(), 0
        else:
            patience += 1
            if patience >= 10:
                print(f"[trackB] 早停（patience 10），最佳 tune_auc={best_auc:.4f}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    if not limit:
        torch.save(model.state_dict(), datadir / "trackB_model.pt")
        pd.DataFrame(log_rows).to_csv(datadir / "trackB_train_log.csv", index=False)
    return model, device


def _phase_features(model, device, signals, load_ok, cohort, datadir, batch_size, limit):
    import torch
    from sklearn.decomposition import PCA

    model.eval()
    n = len(cohort)
    feats = np.full((n, FEAT_DIM), np.nan, dtype=np.float32)
    idx = np.where(load_ok)[0]
    with torch.no_grad():
        for s in range(0, len(idx), batch_size):
            r = idx[s:s + batch_size]
            x = torch.from_numpy(np.asarray(signals[r])).unsqueeze(1).to(device)
            f = model.features(x)
            feats[r] = f.float().cpu().numpy()
            if (s // batch_size) % 20 == 0:
                print(f"[trackB] 推理 {r[-1] + 1:,}/{n:,}", flush=True)

    meta = cohort[["subject_id", "stay_id"]].copy()
    meta = meta.merge(pd.read_csv(datadir / "splits.csv"), on=["subject_id", "stay_id"])
    train_rows = np.where((meta["subset"] == "train") & load_ok)[0]
    pca = PCA(n_components=PCA_DIM, random_state=SEED)
    pca.fit(feats[train_rows])
    pcs = np.full((n, PCA_DIM), np.nan, dtype=np.float32)
    pcs[load_ok] = pca.transform(feats[load_ok]).astype(np.float32)
    ev = float(pca.explained_variance_ratio_.sum())
    print(f"[trackB] PCA32 累计解释方差: {ev:.1%}")

    out = pd.concat(
        [cohort[["subject_id", "stay_id"]].reset_index(drop=True),
         pd.DataFrame(pcs, columns=[f"tb{i}" for i in range(1, PCA_DIM + 1)])],
        axis=1,
    )
    out["load_ok"] = load_ok
    if not limit:
        out.to_parquet(datadir / "features_trackB.parquet", index=False)
        print(f"[trackB] 特征 -> {datadir / 'features_trackB.parquet'}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Track B 特征提取（2D-ResNet 从零训练）")
    parser.add_argument("--datadir", default=str(REPO_ROOT / "data"))
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0, help="仅前 N 条冒烟测试（不写缓存）")
    parser.add_argument("--recompute", action="store_true", help="重读波形并重训")
    args = parser.parse_args()

    datadir = Path(args.datadir)
    cohort = pd.read_parquet(datadir / "cohort_ecg.parquet")
    if args.limit:
        cohort = cohort.head(args.limit)
    print(f"[trackB] 队列 {len(cohort):,} 条")

    signals, load_ok = _phase_signals(cohort, datadir, args.workers, args.limit, args.recompute)
    print(f"[trackB] 波形可用 {int(load_ok.sum()):,}/{len(cohort):,}")

    model_path = datadir / "trackB_model.pt"
    if model_path.exists() and not args.recompute and not args.limit:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = _build_model().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[trackB] 复用已训模型 {model_path.name}")
    else:
        model, device = _phase_train(signals, load_ok, cohort, datadir,
                                     args.epochs, args.batch_size, args.limit)

    out = _phase_features(model, device, signals, load_ok, cohort, datadir,
                          args.batch_size, args.limit)
    print(out[[f"tb{i}" for i in range(1, 5)]].describe().loc[["mean", "std"]].to_string())


if __name__ == "__main__":
    main()
