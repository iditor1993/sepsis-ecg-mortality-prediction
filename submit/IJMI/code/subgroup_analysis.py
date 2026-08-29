"""subgroup_analysis.py — 预设亚组分析（SAP 9.7，探索性，W7）。

亚组（测试集，M3 vs M1 的 ΔAUC + bootstrap CI + 交互检验 + 森林图）：
  年龄（<65 / ≥65）、性别、脓毒性休克（t0±24h 血管活性药+乳酸>2，基线代理）、
  SOFA 三分位（dev 切点）、感染部位、房颤（ECG 报告文本）、心室率（<100 / ≥100，
  ECG 机器测量 rr_interval 换算）

交互检验：二分类亚组用 bootstrap 差值；SOFA 三分位用等级趋势（bootstrap 斜率）；
感染部位用中心化 omnibus bootstrap。探索性，不做显著性声明（SAP 9.8）。

输出：results/subgroup_analysis.csv、results/figures/subgroup_forest.png、
      data/ecg_rhythm_flags.parquet（房颤/起搏/心室率，供 S6 复用）。
"""

from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = "E:/clinical_research/MIMIC_IV_3.1/mimic_iv_3_1.duckdb"
SEED = 20260823
N_BOOT = 2000


def extract_rhythm_flags(datadir: Path) -> pd.DataFrame:
    """按研究队列链接的 study_id 提取房颤/起搏（报告文本）与心室率。"""
    out_path = datadir / "ecg_rhythm_flags.parquet"
    if out_path.exists():
        return pd.read_parquet(out_path)
    link = pd.read_parquet(datadir / "ecg_linked.parquet")
    ecg = link[link["study_id"].notna()][["subject_id", "stay_id", "study_id"]]
    con = duckdb.connect(DEFAULT_DB, read_only=True)
    try:
        con.register("ecg", ecg)
        flags = con.sql("""
            SELECT e.study_id,
                   MAX(CASE WHEN regexp_matches(lower(r.report_text),
                        'atrial fibrillation|a-fib|afib') THEN 1 ELSE 0 END) AS af,
                   MAX(CASE WHEN regexp_matches(lower(r.report_text),
                        'paced|pacemaker') THEN 1 ELSE 0 END) AS paced
            FROM ecg e
            LEFT JOIN main.ecg_reports r ON r.study_id = e.study_id
            GROUP BY e.study_id
        """).df()
        hr = con.sql("""
            SELECT e.study_id, MEDIAN(60000.0 / NULLIF(mm.rr_interval, 0)) AS vent_rate
            FROM ecg e
            LEFT JOIN main.ecg_measurements mm ON mm.study_id = e.study_id
            GROUP BY e.study_id
        """).df()
    finally:
        con.close()
    out = ecg.merge(flags, on="study_id").merge(hr, on="study_id")
    out[["af", "paced"]] = out[["af", "paced"]].fillna(0).astype(int)
    out.to_parquet(out_path, index=False)
    return out


def boot_auc_delta(y, pa, pb, rng, n_boot=N_BOOT):
    n = len(y)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].sum() in (0, n):
            diffs[b] = np.nan
        else:
            diffs[b] = roc_auc_score(y[idx], pa[idx]) - roc_auc_score(y[idx], pb[idx])
    return diffs


def main() -> None:
    datadir = REPO_ROOT / "data"
    results_dir = REPO_ROOT / "results"
    rng = np.random.default_rng(SEED)

    pred = pd.read_parquet(results_dir / "test_predictions.parquet")
    dev = pd.read_parquet(datadir / "features_dev.parquet")
    cov = pd.read_parquet(datadir / "covariates.parquet")
    mice = pd.read_parquet(datadir / "mice_lactate.parquet")
    rhythm = extract_rhythm_flags(datadir)

    df = pred.merge(dev[["stay_id", "admission_age", "gender", "sofa_score",
                         "infection_site", "vaso_24h"]], on="stay_id")
    df = df.merge(cov[["stay_id", "lactate"]].rename(columns={"lactate": "lac_obs"}),
                  on="stay_id", how="left")
    lac_imp = mice[[c for c in mice.columns if c.startswith("lactate_m")]].mean(axis=1)
    lac_map = pd.Series(lac_imp.values, index=mice["stay_id"].values)
    df["lactate_full"] = df["lac_obs"].fillna(df["stay_id"].map(lac_map))
    df = df.merge(rhythm[["stay_id", "af", "vent_rate"]], on="stay_id", how="left")
    y = df["y"].astype(int).to_numpy()
    p3, p1 = df["M3_cal"].to_numpy(), df["M1_cal"].to_numpy()

    # 亚组变量
    q1, q2 = dev["sofa_score"].quantile([1 / 3, 2 / 3]).tolist()
    subs = {
        "年龄": ("<65", df["admission_age"] < 65, "≥65", df["admission_age"] >= 65),
        "性别": ("女", df["gender"] == "F", "男", df["gender"] == "M"),
        "脓毒性休克": ("无", ~((df["vaso_24h"]) & (df["lactate_full"] > 2)),
                  "有", (df["vaso_24h"]) & (df["lactate_full"] > 2)),
        "心室率": ("<100", df["vent_rate"] < 100, "≥100", df["vent_rate"] >= 100),
        "房颤": ("无", df["af"] == 0, "有", df["af"] == 1),
    }

    rows, forest = [], []
    for name, (l1, m1_, l2, m2_) in subs.items():
        m1_, m2_ = m1_.to_numpy(), m2_.to_numpy()
        d1 = boot_auc_delta(y[m1_], p3[m1_], p1[m1_], rng)
        d2 = boot_auc_delta(y[m2_], p3[m2_], p1[m2_], rng)
        dd = np.nanmean(d1) - np.nanmean(d2)
        dd_boot = d1 - d2
        p_int = 2 * min(float(np.nanmean(dd_boot > 0)), float(np.nanmean(dd_boot < 0)))
        for label, mask, d in [(l1, m1_, d1), (l2, m2_, d2)]:
            rows.append({"亚组": name, "水平": label, "n": int(mask.sum()),
                         "事件率": float(y[mask].mean()),
                         "delta_auc": float(np.nanmean(d)),
                         "ci_lo": float(np.nanpercentile(d, 2.5)),
                         "ci_hi": float(np.nanpercentile(d, 97.5)),
                         "p_interaction": round(min(p_int, 1.0), 4)})
        forest.append((name, l1, d1))
        forest.append((name, l2, d2))
        print(f"[{name}] {l1} ΔAUC {np.nanmean(d1):+.4f} vs {l2} {np.nanmean(d2):+.4f}"
          f"；交互 p={p_int:.3f}", flush=True)

    # SOFA 三分位（等级趋势）
    tert = pd.cut(df["sofa_score"], [-np.inf, q1, q2, np.inf], labels=["T1", "T2", "T3"])
    deltas, trends = [], []
    for t in ["T1", "T2", "T3"]:
        m = (tert == t).to_numpy()
        deltas.append(boot_auc_delta(y[m], p3[m], p1[m], rng))
        rows.append({"亚组": "SOFA三分位", "水平": t, "n": int(m.sum()),
                     "事件率": float(y[m].mean()),
                     "delta_auc": float(np.nanmean(deltas[-1])),
                     "ci_lo": float(np.nanpercentile(deltas[-1], 2.5)),
                     "ci_hi": float(np.nanpercentile(deltas[-1], 97.5)),
                     "p_interaction": np.nan})
        forest.append(("SOFA三分位", t, deltas[-1]))
    trend_boot = (deltas[2] - deltas[0]) / 2
    p_trend = 2 * min(float(np.nanmean(trend_boot > 0)), float(np.nanmean(trend_boot < 0)))
    for r in rows[-3:]:
        r["p_interaction"] = round(min(p_trend, 1.0), 4)
    print(f"[SOFA三分位] ΔAUC: " + " / ".join(f"{np.nanmean(d):+.4f}" for d in deltas)
          + f"；趋势 p={p_trend:.3f}", flush=True)

    # 感染部位（omnibus）
    sites = sorted(df["infection_site"].unique())
    site_d = []
    pooled = np.nanmean(boot_auc_delta(y, p3, p1, rng))
    for s in sites:
        m = (df["infection_site"] == s).to_numpy()
        d = boot_auc_delta(y[m], p3[m], p1[m], rng)
        site_d.append(d)
        rows.append({"亚组": "感染部位", "水平": s, "n": int(m.sum()),
                     "事件率": float(y[m].mean()),
                     "delta_auc": float(np.nanmean(d)),
                     "ci_lo": float(np.nanpercentile(d, 2.5)),
                     "ci_hi": float(np.nanpercentile(d, 97.5)),
                     "p_interaction": np.nan})
        forest.append(("感染部位", s, d))
    obs = max(abs(np.nanmean(d) - pooled) for d in site_d)
    obs_means = [np.nanmean(d) for d in site_d]
    cnt = 0
    for b in range(N_BOOT):
        centered = [(d[b] if not np.isnan(d[b]) else np.nanmean(d)) - om + pooled
                    for d, om in zip(site_d, obs_means)]
        if max(abs(c - pooled) for c in centered) >= obs:
            cnt += 1
    p_omni = cnt / N_BOOT
    for r in rows[-len(sites):]:
        r["p_interaction"] = round(p_omni, 4)
    print(f"[感染部位] omnibus p={p_omni:.3f}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(results_dir / "subgroup_analysis.csv", index=False)

    # 森林图
    fig, ax = plt.subplots(figsize=(8, max(5, len(forest) * 0.45)))
    yticks, ylabels = [], []
    for i, (name, level, d) in enumerate(forest):
        m, lo, hi = np.nanmean(d), np.nanpercentile(d, 2.5), np.nanpercentile(d, 97.5)
        ax.plot([lo, hi], [i, i], "-", lw=2, color="steelblue")
        ax.plot(m, i, "s", color="darkred", ms=6)
        yticks.append(i)
        ylabels.append(f"{name} | {level}")
    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.axvline(-0.0014, color="darkgreen", ls=":", lw=1.5, label="总体 ΔAUC = -0.0014")
    ax.set_yticks(yticks, ylabels)
    ax.invert_yaxis()
    ax.set_xlabel("ΔAUC（M3 − M1，测试集）")
    ax.legend(loc="lower right")
    ax.set_title("预设亚组：M3 vs M1 的 ΔAUC（95% CI）")
    fig.tight_layout()
    fig.savefig(results_dir / "figures" / "subgroup_forest.png", dpi=200)
    print("\n输出 -> results/subgroup_analysis.csv, results/figures/subgroup_forest.png")


if __name__ == "__main__":
    main()
