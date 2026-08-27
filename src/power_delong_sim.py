"""power_delong_sim.py — ΔAUC 模拟功效核算（SAP 第六章，V1.2 新增）。

**解盲门槛**：在测试集结局标签解盲之前执行。仅使用测试集样本量与事件数
两个聚合数字（SAP 第六章明确允许），不接触个体标签。

方法（预设模拟框架，参照前序研究 [18]）：
  二元正态潜变量模型——对照组预测分数 ~ N(0, I)，病例组 ~ N(a, I)，
  两模型分数相关 ρ（病例/对照内相同）；a = √2·Φ⁻¹(AUC)。
  每个模拟数据集上用 DeLong 法计算 ΔAUC 方差并作双侧 z 检验（α=0.05）。
  功效 = 模拟中 p<0.05 的比例。

预设参数网格（SAP 第六章）：
  基线 AUC ∈ {0.75, 0.775, 0.80}；嵌套模型相关 ρ ∈ {0.85, 0.90, 0.95}；
  真实 ΔAUC 网格 {0.010, 0.015, 0.020, 0.025, 0.030}（用于功效曲线与
  80% 功效最小可检测 ΔAUC 的插值）。
判定规则：最差参数单元（基线 0.75、ρ=0.85）下 80% 功效最小可检测 ΔAUC ≤0.02
  方可解盲执行 H1；否则 H1 降级为估计性分析（SAP 第六章）。

输出：results/power_delong_sim.csv；参数与判定结果追加至 data/DATA_LOCK.md。
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED = 20260823
N_SIM = 2000
ALPHA = 0.05
AUC1_GRID = [0.75, 0.775, 0.80]
RHO_GRID = [0.85, 0.90, 0.95]
DELTA_GRID = [0.010, 0.015, 0.020, 0.025, 0.030]


def delong_test(preds: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """两模型 DeLong 检验：返回 (p 值, ΔAUC 点估计)。preds: (n, 2)，y: (n,) 0/1。"""
    x, z = preds[y == 1], preds[y == 0]
    n1, n0 = len(x), len(z)
    v10 = np.empty((n1, 2))
    v01 = np.empty((n0, 2))
    for k in range(2):
        psi = (x[:, k:k + 1] > z[:, k]) + 0.5 * (x[:, k:k + 1] == z[:, k])
        v10[:, k] = psi.mean(axis=1)
        v01[:, k] = psi.mean(axis=0)
    s10 = np.cov(v10, rowvar=False, ddof=1)
    s01 = np.cov(v01, rowvar=False, ddof=1)
    var = ((s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / n1
           + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n0)
    delta = v10[:, 0].mean() - v10[:, 1].mean()
    if var <= 0:
        return 1.0, float(delta)
    z_score = delta / np.sqrt(var)
    return float(2 * (1 - norm.cdf(abs(z_score)))), float(delta)


def sim_power(auc1: float, delta: float, rho: float, n1: int, n0: int,
              n_sim: int, seed: int) -> tuple[float, float]:
    """给定参数下的 DeLong 检验功效与平均 ΔAUC 估计（ sanity 检查）。"""
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, rho], [rho, 1.0]])
    a1 = np.sqrt(2) * norm.ppf(auc1)
    a2 = np.sqrt(2) * norm.ppf(auc1 + delta)
    hits, deltas = 0, []
    for _ in range(n_sim):
        z0 = rng.multivariate_normal([0.0, 0.0], cov, n0)
        z1 = rng.multivariate_normal([a1, a2], cov, n1)
        preds = np.vstack([z1, z0])
        y = np.array([1] * n1 + [0] * n0)
        p, d_hat = delong_test(preds, y)
        hits += p < ALPHA
        deltas.append(d_hat)
    return hits / n_sim, float(np.mean(deltas))


def main() -> None:
    datadir = REPO_ROOT / "data"
    dev = pd.read_parquet(datadir / "features_dev.parquet",
                          columns=["subset", "death_28d"])
    te = dev[dev["subset"] == "test"]
    n_test, n_events = len(te), int(te["death_28d"].sum())
    n0 = n_test - n_events
    print("=" * 66)
    print("ΔAUC 模拟功效核算（SAP 第六章，解盲前）")
    print("=" * 66)
    print(f"测试集（聚合）：N={n_test:,}，事件={n_events}（{n_events / n_test:.1%}），"
          f"非事件={n0:,}")
    print(f"模拟次数/单元: {N_SIM}；α={ALPHA} 双侧；种子 {SEED}")

    rows = []
    for auc1 in AUC1_GRID:
        for rho in RHO_GRID:
            for di, delta in enumerate(DELTA_GRID):
                seed = SEED + int(auc1 * 1000) * 100 + int(rho * 100) * 10 + di
                power, mean_d = sim_power(auc1, delta, rho, n_events, n0, N_SIM, seed)
                rows.append({"auc1": auc1, "rho": rho, "delta": delta,
                             "power": round(power, 4), "mean_delta_auc_hat": round(mean_d, 4)})
                print(f"  AUC1={auc1:.3f} ρ={rho:.2f} Δ={delta:.3f} "
                      f"-> 功效 {power:.1%}（均值估计 {mean_d:+.4f}）", flush=True)

    res = pd.DataFrame(rows)
    (REPO_ROOT / "results").mkdir(exist_ok=True)
    res.to_csv(REPO_ROOT / "results" / "power_delong_sim.csv", index=False)

    # 80% 功效最小可检测 ΔAUC（按功效曲线线性插值）
    def mdd(sub: pd.DataFrame) -> float:
        sub = sub.sort_values("delta")
        p, d = sub["power"].to_numpy(), sub["delta"].to_numpy()
        if p.max() < 0.80:
            return float("nan")  # 最大 Δ 处功效仍不足 80%
        order = np.argsort(p)
        return float(np.interp(0.80, p[order], d[order]))

    mdd_rows = []
    for auc1 in AUC1_GRID:
        for rho in RHO_GRID:
            sub = res[(res["auc1"] == auc1) & (res["rho"] == rho)]
            mdd_rows.append({"auc1": auc1, "rho": rho,
                             "mdd_80pct": round(mdd(sub), 4),
                             "power_at_0.02": float(sub.loc[sub["delta"] == 0.02,
                                                            "power"].iloc[0])})
    mdd_df = pd.DataFrame(mdd_rows)
    print("\n[80% 功效最小可检测 ΔAUC（MDD）与 Δ=0.02 处功效]")
    print(mdd_df.to_string(index=False))

    worst = mdd_df[(mdd_df["auc1"] == min(AUC1_GRID)) & (mdd_df["rho"] == min(RHO_GRID))].iloc[0]
    gate_ok = bool((mdd_df["mdd_80pct"] <= 0.02).all()) and worst["power_at_0.02"] >= 0.80
    verdict = ("通过：全部参数单元 80% 功效 MDD ≤0.02，可解盲执行 H1 确认性检验"
               if gate_ok else
               "未通过：H1 按 SAP 第六章降级为估计性分析（报告 ΔAUC 点估计与 95% CI）")
    print(f"\n[解盲门槛判定] {verdict}")

    # 追加数据锁定记录（幂等：先移除既有同标题小节再写入）
    lock_path = datadir / "DATA_LOCK.md"
    section_title = "## ΔAUC 功效核算（SAP 第六章，解盲门槛）"
    existing = lock_path.read_text(encoding="utf-8")
    if section_title in existing:
        existing = existing[: existing.index(section_title)].rstrip() + "\n"
    lines = [
        "",
        section_title,
        "",
        f"- 测试集聚合：N={n_test:,}，事件={n_events}，非事件={n0:,}",
        f"- 参数网格：AUC1∈{AUC1_GRID}，ρ∈{RHO_GRID}，Δ∈{DELTA_GRID}；"
        f"模拟 {N_SIM} 次/单元，α={ALPHA} 双侧，种子 {SEED}",
        "- MDD（80% 功效最小可检测 ΔAUC）与 Δ=0.02 处功效：",
        "",
        "| AUC1 | ρ | MDD(80%) | power@0.02 |",
        "|---|---|---|---|",
    ]
    for r in mdd_rows:
        lines.append(f"| {r['auc1']} | {r['rho']} | {r['mdd_80pct']} "
                     f"| {r['power_at_0.02']:.4f} |")
    lines.append(f"\n- **判定：{verdict}**")
    lock_path.write_text(existing + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n结果 -> results/power_delong_sim.csv；判定已写入 {lock_path}")


if __name__ == "__main__":
    main()
