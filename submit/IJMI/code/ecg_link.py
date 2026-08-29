"""ecg_link.py — ECG 链接与质控（SAP 3.4 节，W1-W2）。

流程：
  1) 链接：读 ecg/record_list.csv 索引，对 cohort_base 每位患者在
     t0±window_h 窗口内取距 t0 最近的一份 ECG（ tie 依次取较早 ecg_time、
     较小 study_id，保证确定性）；无窗内 ECG 者记为不可链接（A1a 分组）。
  2) 质控：读取 WFDB 波形，计算元数据（时长/采样率/导联数/缺失导联）与
     信号质量指标（电极饱和平台、基线漂移极差、高频噪声 RMS），
     按 src/qc_config.yaml 阈值判定。失败不回退、不挽救（SAP 3.4）。
  3) 输出：
     data/ecg_linked.parquet   每位入组患者一行，含链接信息、QC 指标、
                               排除原因与 A1a 分组标记 ecg_available
     data/cohort_ecg.parquet   ECG 排除后的最终分析队列
     data/ecg_qc_metrics.parquet  QC 指标缓存（改阈值后复用，免重读波形）
     data/cohort_flow.csv      追加 ECG 阶段分级计数（幂等）

波形读取较耗时，指标经多进程计算并缓存；重复运行默认复用缓存。
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
import yaml
from scipy import ndimage, signal

REPO_ROOT = Path(__file__).resolve().parent.parent
ECG_ROOT = Path("E:/clinical_research/MIMIC_IV_3.1/ecg")
RECORD_LIST = ECG_ROOT / "record_list.csv"

# ---------------------------------------------------------------------------
# 单条波形 QC 指标计算（多进程 worker，须为模块级函数）
# ---------------------------------------------------------------------------


def _clean_lead(x: np.ndarray) -> np.ndarray | None:
    """线性插值填补 NaN；全 NaN 返回 None。"""
    if not np.isnan(x).any():
        return x
    ok = ~np.isnan(x)
    if ok.sum() < 2:
        return None
    idx = np.arange(len(x))
    return np.interp(idx, idx[ok], x[ok])


def _longest_extreme_run(x: np.ndarray) -> int:
    """信号取值等于本导联最大/最小值的最长连续段长度（样本数）。"""
    best = 0
    for extreme in (x.max(), x.min()):
        at = x == extreme
        if not at.any():
            continue
        # 差分法求最长连续 True 段
        d = np.diff(np.concatenate(([0], at.view(np.int8), [0])))
        runs = np.where(d == -1)[0] - np.where(d == 1)[0]
        best = max(best, int(runs.max()))
    return best


def qc_metrics(rec_path: str) -> dict:
    """读取一条 WFDB 记录，返回元数据与信号质量指标。"""
    header = wfdb.rdheader(rec_path)
    fs = float(header.fs)
    out = {
        "fs": fs,
        "sig_len": int(header.sig_len),
        "n_sig": int(header.n_sig),
        "duration_s": header.sig_len / fs,
        "read_error": None,
    }
    rec = wfdb.rdrecord(rec_path, physical=True)
    x = np.asarray(rec.p_signal, dtype=float)  # (n_samples, n_leads)，单位 mV
    names = list(rec.sig_name)

    n_missing, missing = 0, []
    clip_run_ms, wander_mv, hf_rms_mv = 0.0, 0.0, 0.0
    kernel = int(round(0.6 * fs)) | 1  # 0.6 s 中值滤波窗（奇数）
    if fs > 80:  # 高通截止 40 Hz 需 fs > 80
        b, a = signal.butter(4, 40.0 / (fs / 2), btype="high")
    else:
        b = a = None

    for j, name in enumerate(names):
        lead = _clean_lead(x[:, j])
        if lead is None or np.ptp(lead) == 0 or np.std(lead) == 0:
            n_missing += 1
            missing.append(name)
            continue
        clip_run_ms = max(clip_run_ms, _longest_extreme_run(lead) / fs * 1000.0)
        baseline = ndimage.median_filter(lead, size=kernel, mode="nearest")
        wander_mv = max(
            wander_mv,
            float(np.percentile(baseline, 95) - np.percentile(baseline, 5)),
        )
        if b is not None:
            hf = signal.filtfilt(b, a, lead)
            hf_rms_mv = max(hf_rms_mv, float(np.sqrt(np.mean(hf**2))))

    out.update(
        n_missing_leads=n_missing,
        missing_leads=",".join(missing),
        clip_run_ms_max=round(clip_run_ms, 1),
        wander_mv_max=round(wander_mv, 4),
        hf_rms_mv_max=round(hf_rms_mv, 4),
    )
    return out


def _qc_one(task: tuple) -> dict:
    subject_id, study_id, rel_path = task
    row = {"subject_id": subject_id, "study_id": study_id}
    try:
        row.update(qc_metrics(str(ECG_ROOT / rel_path)))
    except Exception as exc:  # 文件缺失/损坏等：记 read_error，按不合格处理
        row.update(
            fs=np.nan, sig_len=np.nan, n_sig=np.nan, duration_s=np.nan,
            read_error=str(exc)[:200], n_missing_leads=np.nan,
            missing_leads="", clip_run_ms_max=np.nan,
            wander_mv_max=np.nan, hf_rms_mv_max=np.nan,
        )
    return row


# ---------------------------------------------------------------------------
# 阶段 1：链接
# ---------------------------------------------------------------------------


def link_nearest(cohort: pd.DataFrame, records: pd.DataFrame, window_h: float) -> pd.DataFrame:
    """每位患者取 t0±window_h 内距 t0 最近的 ECG；无窗内 ECG 者保留空行。"""
    cand = cohort[["subject_id", "stay_id", "t0"]].merge(records, on="subject_id", how="left")
    cand["abs_diff_h"] = (cand["ecg_time"] - cand["t0"]).abs().dt.total_seconds() / 3600.0
    in_win = cand[cand["abs_diff_h"] <= window_h].copy()
    n_in_win = in_win.groupby("subject_id").size().rename("n_ecg_in_window")
    in_win = in_win.sort_values(["abs_diff_h", "ecg_time", "study_id"])
    best = in_win.drop_duplicates("subject_id", keep="first")[
        ["subject_id", "study_id", "path", "ecg_time", "abs_diff_h"]
    ].rename(columns={"path": "ecg_path", "abs_diff_h": "abs_t0_diff_h"})

    out = cohort[["subject_id"]].merge(best, on="subject_id", how="left")
    out = out.merge(n_in_win, on="subject_id", how="left")
    out["n_ecg_in_window"] = out["n_ecg_in_window"].fillna(0).astype(int)
    t0_map = out["subject_id"].map(cohort.set_index("subject_id")["t0"])
    out["signed_t0_diff_h"] = (out["ecg_time"] - t0_map).dt.total_seconds() / 3600.0
    return out


# ---------------------------------------------------------------------------
# 阶段 3：阈值判定
# ---------------------------------------------------------------------------


def apply_qc(linked: pd.DataFrame, metrics: pd.DataFrame, qc: dict) -> pd.DataFrame:
    df = linked.merge(metrics, on=["subject_id", "study_id"], how="left")

    has_ecg = df["study_id"].notna()
    reasons = pd.Series("none", index=df.index, dtype=object)
    reasons[~has_ecg] = "no_ecg_in_window"
    reasons[has_ecg & df["read_error"].notna()] = "read_error"
    ok = has_ecg & df["read_error"].isna()
    reasons[ok & (df["fs"] != qc["expected_fs"])] = "fs_abnormal"
    reasons[(reasons == "none") & (df["duration_s"] < qc["min_duration_s"])] = "duration_lt_8s"
    reasons[(reasons == "none") & (df["n_sig"] != 12)] = "lead_count_ne_12"
    reasons[(reasons == "none") & (df["n_missing_leads"] > qc["max_missing_leads"])] = "missing_leads_gt_2"
    sig_fail = (
        (df["clip_run_ms_max"] > qc["clip_run_ms"])
        | (df["wander_mv_max"] > qc["baseline_wander_mv"])
        | (df["hf_rms_mv_max"] > qc["hf_noise_rms_mv"])
    )
    reasons[(reasons == "none") & sig_fail] = "signal_quality"

    df["exclusion_reason"] = reasons
    df["ecg_available"] = reasons == "none"  # A1a 分组标记：可链接且合格
    return df


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="ECG 链接与质控（SAP 3.4）")
    parser.add_argument("--datadir", default=str(REPO_ROOT / "data"))
    parser.add_argument("--qc-config", default=str(REPO_ROOT / "src" / "qc_config.yaml"))
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--limit", type=int, default=0, help="仅处理前 N 条链接（冒烟测试）")
    parser.add_argument("--recompute", action="store_true", help="忽略指标缓存重算")
    args = parser.parse_args()

    datadir = Path(args.datadir)
    qc = yaml.safe_load(Path(args.qc_config).read_text(encoding="utf-8"))
    print(f"[qc_config] {qc}")

    cohort = pd.read_parquet(datadir / "cohort_base.parquet")
    records = pd.read_csv(RECORD_LIST, parse_dates=["ecg_time"])
    print(f"[link] 入组患者 {len(cohort):,}；ECG 索引 {len(records):,} 条")

    linked = link_nearest(cohort, records, float(qc["window_h"]))
    n_linked = linked["study_id"].notna().sum()
    print(f"[link] 窗内可链接 {n_linked:,}；不可链接 {len(linked) - n_linked:,}"
          f"（A1a 不可链接组）")

    # ---- 阶段 2：波形 QC 指标（多进程 + 缓存）----
    cache = datadir / "ecg_qc_metrics.parquet"
    todo = linked[linked["study_id"].notna()][["subject_id", "study_id", "ecg_path"]]
    if args.limit:
        todo = todo.head(args.limit)
        linked = linked[linked["study_id"].isin(todo["study_id"]) | linked["study_id"].isna()]
    if cache.exists() and not args.recompute:
        metrics = pd.read_parquet(cache)
        print(f"[qc] 复用缓存指标 {len(metrics):,} 条")
    else:
        tasks = list(todo.itertuples(index=False, name=None))
        print(f"[qc] 读取波形 {len(tasks):,} 条，workers={args.workers} ...")
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, row in enumerate(pool.map(_qc_one, tasks, chunksize=32), 1):
                rows.append(row)
                if i % 2000 == 0 or i == len(tasks):
                    print(f"[qc]   {i:,}/{len(tasks):,}")
        metrics = pd.DataFrame(rows)
        if not args.limit:
            metrics.to_parquet(cache, index=False)
            print(f"[qc] 指标已缓存 -> {cache}")

    # ---- 阶段 3：阈值判定与输出 ----
    detail = apply_qc(linked, metrics, qc)
    out = cohort.merge(
        detail.drop(columns=["t0"], errors="ignore"), on="subject_id", how="left"
    )
    final = out[out["ecg_available"] == True].copy()  # noqa: E712（NaN 视为不可用）

    if args.limit:
        print("\n[冒烟测试 --limit] 不写出任何文件")
        flow = None
    else:
        out.to_parquet(datadir / "ecg_linked.parquet", index=False)
        final.to_parquet(datadir / "cohort_ecg.parquet", index=False)

        # 流程图计数（幂等追加）
        flow_path = datadir / "cohort_flow.csv"
        flow = pd.read_csv(flow_path)
        flow = flow[~flow["stage"].str.startswith("ecg_")]
        prev = int(flow.iloc[-1]["n_stays"])
        for stage, n in [
            ("ecg_in_window", int(detail["study_id"].notna().sum())),
            ("ecg_meta_ok", int((~detail["exclusion_reason"].isin(
                ["no_ecg_in_window", "read_error", "fs_abnormal",
                 "duration_lt_8s", "lead_count_ne_12", "missing_leads_gt_2"])).sum())),
            ("ecg_signal_ok_final", int(detail["ecg_available"].sum())),
        ]:
            flow.loc[len(flow)] = [stage, n, n, prev - n]
            prev = n
        flow.to_csv(flow_path, index=False)

    # ---- 摘要 ----
    print("\n" + "=" * 60)
    print("ECG 链接与质控结果（SAP 3.4）")
    print("=" * 60)
    print("\n[排除原因分布]")
    print(detail["exclusion_reason"].value_counts().to_string())
    if flow is not None:
        print("\n[流程图全阶段]")
        print(flow.to_string(index=False))
    print("\n[最终队列时段分布]")
    print(final["cohort_period"].value_counts().to_string())
    print("\n[A1a 分组] 可链接合格 vs 不可链接/不合格:")
    print(detail["ecg_available"].map({True: "ecg_available", False: "ecg_unavailable"})
          .value_counts().to_string())
    print("\n[QC 指标分位数（已读波形者）]")
    q = detail[["clip_run_ms_max", "wander_mv_max", "hf_rms_mv_max"]].quantile(
        [0.5, 0.75, 0.9, 0.95, 0.99])
    print(q.to_string())
    print("\n输出文件:")
    for f in ["ecg_linked.parquet", "cohort_ecg.parquet", "ecg_qc_metrics.parquet",
              "cohort_flow.csv"]:
        print(f"  {datadir / f}")


if __name__ == "__main__":
    main()
