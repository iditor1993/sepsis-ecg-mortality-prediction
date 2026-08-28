"""sensitivity_s8.py — S8：竞争风险框架复核（SAP 9.9）。

以"存活出院"为竞争事件、28 天死亡为目标事件，用 Fine-Gray 思路复核
主要关联（M3 特征与 28 天死亡的关联是否受竞争风险影响）。

实现说明：本队列 28 天内仅存在行政截尾（day 28），无随机截尾，故
Fine-Gray 权重 G(t)/G(t_i) 恒为 1——竞争事件者保留于风险集至 28 天，
等价于 start-stop 格式的 Cox 模型中竞争事件者不删失（Geskus 权重特例）。
以 lifelines CoxPHFitter（start-stop + robust SE）实现。乳酸取 20 套
插补均值（探索性敏感性分析的简化，文档化）。

报告：M3 特征各变量的 subHR 与 p；重点为 z1-z16 关联是否改变。
输出 results/sensitivity_s8.csv。
"""

from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED = 20260823
COV_COLS = ["admission_age", "male", "charlson_comorbidity_index",
            "admission_emergency", "pre_icu_los_h", "mech_vent_24h", "vaso_24h"]
SCORE_COLS = ["sofa_score", "qsofa", "news", "mews"]
Z_COLS = [f"z{i}" for i in range(1, 17)]


def main() -> None:
    datadir = REPO_ROOT / "data"
    results_dir = REPO_ROOT / "results"
    from sensitivity_analyses import build
    df = build(datadir)
    site_cols = sorted(c for c in df.columns if c.startswith("site_"))[1:]  # 去一档避免虚拟变量陷阱
    mice = pd.read_parquet(datadir / "mice_lactate.parquet")
    lac_mean = mice[[c for c in mice.columns if c.startswith("lactate_m")]].mean(axis=1)
    lac_map = pd.Series(lac_mean.values, index=mice["stay_id"].values)
    df["lactate_imp_mean"] = df["stay_id"].map(lac_map)

    # 结局时间：28 天内死亡/存活出院/行政截尾
    days_death = df["days_to_death"] if "days_to_death" in df.columns else None
    out = pd.read_parquet(datadir / "outcomes.parquet")
    t0_map = pd.read_parquet(datadir / "cohort_ecg.parquet",
                             columns=["stay_id", "t0"])
    df = df.drop(columns=["days_to_death"], errors="ignore").merge(
        out[["stay_id", "days_to_death"]], on="stay_id", how="left")
    df = df.merge(t0_map, on="stay_id", how="left")
    days_disch = (df["dischtime"] - df["t0"]).dt.total_seconds() / 86400.0
    death = df["days_to_death"].astype(float)
    alive_disch = (df["hospital_expire_flag"] == 0) & days_disch.notna()

    t_death = np.where(death.notna(), death, np.inf)
    t_disch = np.where(alive_disch, days_disch, np.inf)
    time = np.minimum(np.minimum(t_death, t_disch), 28.0)
    event = np.where((t_death <= t_disch) & (t_death <= 28), 1,
                     np.where((t_disch < 28) & (t_disch < t_death), 2, 0))
    df["fg_time"] = np.clip(time, 0.01, 28.0)
    df["fg_event"] = event
    print(f"[S8] 结局分布: 死亡 {int((event == 1).sum()):,}，"
          f"存活出院(竞争) {int((event == 2).sum()):,}，截尾 {int((event == 0).sum()):,}")

    feats = SCORE_COLS + ["lactate_imp_mean"] + COV_COLS + site_cols + Z_COLS
    work = df.loc[df["subset"] == "train",
                  feats + ["fg_time", "fg_event", "death_28d"]].dropna().copy()
    # Fine-Gray 风险集处理：竞争事件者（event=2）不删失，保留至 28 天
    work["stop"] = np.where(work["fg_event"] == 2, 28.0, work["fg_time"])
    work["event_fg"] = (work["fg_event"] == 1).astype(int)
    work["start"] = 0.0
    sc = StandardScaler().fit(work[feats])
    xs = pd.DataFrame(sc.transform(work[feats]), columns=feats, index=work.index)
    cph_data = pd.concat([xs, work[["start", "stop", "event_fg"]]], axis=1)
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(cph_data, duration_col="stop", event_col="event_fg",
            entry_col="start", robust=True)

    # 对照：普通 Cox（竞争事件者在出院时删失）
    cph_ref_data = pd.concat([xs, work[["fg_time"]].rename(columns={"fg_time": "stop"}),
                              work[["event_fg"]]], axis=1)
    cph_ref = CoxPHFitter(penalizer=0.01)
    cph_ref.fit(cph_ref_data.assign(start=0.0), entry_col="start",
                duration_col="stop", event_col="event_fg", robust=True)

    rows = []
    for v in feats:
        rows.append({
            "feature": v,
            "subHR_FG": float(np.exp(cph.params_[v])),
            "p_FG": float(cph.summary.loc[v, "p"]),
            "HR_cox": float(np.exp(cph_ref.params_[v])),
            "p_cox": float(cph_ref.summary.loc[v, "p"]),
        })
    res = pd.DataFrame(rows)
    res.to_csv(results_dir / "sensitivity_s8.csv", index=False)

    z_res = res[res["feature"].isin(Z_COLS)]
    print("\n[S8] z 维度 subHR(Fine-Gray) vs HR(Cox)：显著性对比")
    sig_fg = set(z_res.loc[z_res["p_FG"] < 0.05, "feature"])
    sig_cx = set(z_res.loc[z_res["p_cox"] < 0.05, "feature"])
    print(f"  z 显著（FG）: {sorted(sig_fg) if sig_fg else '无'}")
    print(f"  z 显著（Cox）: {sorted(sig_cx) if sig_cx else '无'}")
    print("\n[主要临床变量]")
    for v in ["sofa_score", "lactate_imp_mean", "news", "mews"]:
        r = res[res["feature"] == v].iloc[0]
        print(f"  {v}: subHR {r['subHR_FG']:.3f} (p={r['p_FG']:.3g}) | "
              f"HR {r['HR_cox']:.3f} (p={r['p_cox']:.3g})")
    print("\n输出 -> results/sensitivity_s8.csv")
    print("判读：z 维度关联在竞争风险框架下是否保持不显著（与主分析一致）。")


if __name__ == "__main__":
    main()
