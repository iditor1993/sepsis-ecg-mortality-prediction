"""rho_power_recheck.py — 基于 tune 集经验参数的 ΔAUC 功效复核。

预设模拟（power_delong_sim.py）中基线 AUC 与 ρ 为假设值（0.75-0.80 / ≥0.85），
保守角点未达 80% 功效门槛。本脚本以 tune 集上的经验值复核：
  AUC1 := M1 / M1+ 的 tune AUC；ρ := M3 与 M1 / M1+ 的 tune 预测值 Pearson 相关
两个比较（M3 vs M1、M3 vs M1+）分别核算 80% 功效最小可检测 ΔAUC（MDD）。

方法学说明：tune 属开发队列，使用其标签属开发阶段合法操作（测试集仍未解盲）；
以经验参数替代预设假设属参数精化，将记录于偏离日志（D-002）。

输出：results/power_recheck_empirical.csv；控制台打印判定。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from power_delong_sim import DELTA_GRID, N_SIM, SEED, sim_power  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def mdd80(powers: list[float]) -> float:
    p = np.asarray(powers)
    d = np.asarray(DELTA_GRID)
    if p.max() < 0.80:
        return float("nan")
    order = np.argsort(p)
    return float(np.interp(0.80, p[order], d[order]))


def main() -> None:
    pred = pd.read_parquet(REPO_ROOT / "results" / "tune_predictions.parquet")
    y = pred["y"].to_numpy()
    auc_m1 = roc_auc_score(y, pred["M1_raw"])
    auc_m1p = roc_auc_score(y, pred["M1+_raw"])
    rho_m1 = float(np.corrcoef(pred["M3_raw"], pred["M1_raw"])[0, 1])
    rho_m1p = float(np.corrcoef(pred["M3_raw"], pred["M1+_raw"])[0, 1])
    n1, n0 = int(y.sum()), int((1 - y).sum())
    # 功效针对测试集样本结构：按测试集聚合（N=2,217，事件 425）
    n1_te, n0_te = 425, 2217 - 425

    print(f"tune 经验值：AUC(M1)={auc_m1:.4f}，AUC(M1+)={auc_m1p:.4f}，"
          f"ρ(M3,M1)={rho_m1:.4f}，ρ(M3,M1+)={rho_m1p:.4f}")

    rows = []
    for name, auc1, rho in [("M3 vs M1", auc_m1, rho_m1),
                            ("M3 vs M1+", auc_m1p, rho_m1p)]:
        powers = []
        for di, delta in enumerate(DELTA_GRID):
            p, _ = sim_power(auc1, delta, rho, n1_te, n0_te, N_SIM,
                             SEED + 777 + di)
            powers.append(p)
            rows.append({"comparison": name, "auc1": round(auc1, 4),
                         "rho": round(rho, 4), "delta": delta,
                         "power": round(p, 4)})
        m = mdd80(powers)
        print(f"\n[{name}] AUC1={auc1:.3f} ρ={rho:.3f}")
        for d, p in zip(DELTA_GRID, powers):
            print(f"  Δ={d:.3f} -> 功效 {p:.1%}")
        print(f"  MDD(80%) = {m:.4f} -> {'达标（≤0.02）' if m <= 0.02 else '未达标'}")

    res = pd.DataFrame(rows)
    res.to_csv(REPO_ROOT / "results" / "power_recheck_empirical.csv", index=False)
    print("\n输出 -> results/power_recheck_empirical.csv")


if __name__ == "__main__":
    main()
