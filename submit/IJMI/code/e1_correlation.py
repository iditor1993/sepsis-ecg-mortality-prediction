"""e1_correlation.py — E1 潜向量器官维度谱刻画（SAP 9.7，V1.1 新增/V1.2 修订）。

内容：开发队列（dev，14,780 例）上 z1-z16 与 SOFA 六个器官分项及乳酸的
Spearman 相关矩阵热图；|rho|>=0.3 者行偏相关分析（校正 SOFA 其余分项，
秩转换残差法）。探索性分析，Benjamini-Hochberg 控制 FDR（q=0.05）。

解释框架（SAP 9.7）：刻画器官维度谱而非预设循环维度；与 MINERS [17] 的
亚型发现及前序研究 [18] 的阴性交互一并讨论。

输出：results/e1_correlation.csv、results/figures/e1_heatmap.png。
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
Z_COLS = [f"z{i}" for i in range(1, 17)]
SOFA_COLS = ["sofa_respiration", "sofa_coagulation", "sofa_liver",
             "sofa_cardiovascular", "sofa_cns", "sofa_renal"]
EXTRA = ["lactate"]
CN = {"sofa_respiration": "呼吸", "sofa_coagulation": "凝血", "sofa_liver": "肝脏",
      "sofa_cardiovascular": "心血管", "sofa_cns": "神经", "sofa_renal": "肾脏",
      "lactate": "乳酸"}


def bh_fdr(pvals: np.ndarray, q: float = 0.05) -> np.ndarray:
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresh = q * (np.arange(1, len(pvals) + 1) / len(pvals))
    passed = ranked <= thresh
    cutoff = ranked[passed].max() if passed.any() else 0.0
    return pvals <= cutoff


def partial_spearman(z: pd.Series, y: pd.Series, controls: pd.DataFrame) -> float:
    """秩转换 + 线性残差的偏 Spearman（校正 controls）。"""
    rz = stats.rankdata(z)
    ry = stats.rankdata(y)
    rc = controls.apply(stats.rankdata)
    X = np.column_stack([np.ones(len(rc)), rc.to_numpy()])
    bz = np.linalg.lstsq(X, rz, rcond=None)[0]
    by = np.linalg.lstsq(X, ry, rcond=None)[0]
    return float(np.corrcoef(rz - X @ bz, ry - X @ by)[0, 1])


def main() -> None:
    datadir = REPO_ROOT / "data"
    df = pd.read_parquet(datadir / "features_dev.parquet")
    df = df[Z_COLS + SOFA_COLS + EXTRA].dropna()
    print(f"E1 分析样本: {len(df):,}（dev 完整行）")

    targets = SOFA_COLS + EXTRA
    rows = []
    for z in Z_COLS:
        for t in targets:
            rho, p = stats.spearmanr(df[z], df[t])
            rows.append({"z": z, "target": t, "target_cn": CN[t],
                         "spearman_rho": rho, "p": p})
    res = pd.DataFrame(rows)
    res["fdr_sig"] = bh_fdr(res["p"].to_numpy(), 0.05)

    # 偏相关：|rho|>=0.3 且 FDR 显著者，校正 SOFA 其余分项
    partials = []
    for _, r in res.iterrows():
        if abs(r["spearman_rho"]) >= 0.3 and r["fdr_sig"]:
            others = [c for c in SOFA_COLS if c != r["target"]]
            pr = partial_spearman(df[r["z"]], df[r["target"]], df[others])
            partials.append({"z": r["z"], "target": r["target"],
                             "partial_rho": pr})
    pdf = pd.DataFrame(partials)
    if len(pdf):
        res = res.merge(pdf, on=["z", "target"], how="left")
    res.to_csv(REPO_ROOT / "results" / "e1_correlation.csv", index=False)

    # 热图
    mat = res.pivot(index="z", columns="target_cn", values="spearman_rho")
    mat = mat[[CN[t] for t in targets]]
    sig = res.pivot(index="z", columns="target_cn", values="fdr_sig")[mat.columns]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat.to_numpy(), cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    ax.set_xticks(range(len(mat.columns)), mat.columns)
    ax.set_yticks(range(len(mat.index)), mat.index)
    for i in range(len(mat.index)):
        for j in range(len(mat.columns)):
            v = mat.iloc[i, j]
            star = "*" if sig.iloc[i, j] else ""
            ax.text(j, i, f"{v:.2f}{star}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, label="Spearman ρ")
    ax.set_title("E1: ECG 潜向量 z1–z16 × SOFA 器官分项/乳酸（* BH-FDR q<0.05）")
    fig.tight_layout()
    out_fig = REPO_ROOT / "results" / "figures" / "e1_heatmap.png"
    fig.savefig(out_fig, dpi=200)
    print(f"热图 -> {out_fig}")

    print("\n[FDR 显著且 |ρ|≥0.25 的关联]")
    top = res[res["fdr_sig"] & (res["spearman_rho"].abs() >= 0.25)]
    print(top.sort_values("spearman_rho", key=abs, ascending=False)
          .round(3).to_string(index=False))
    if not len(top):
        print("（无 |ρ|≥0.25 的显著关联；全部 |ρ| 最大值 "
          f"{res['spearman_rho'].abs().max():.3f}）")


if __name__ == "__main__":
    main()
