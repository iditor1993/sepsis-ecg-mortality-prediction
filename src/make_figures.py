"""make_figures.py — 论文图制作（W8）：图 1 流程图回填、ROC、校准曲线、DCA。

输出（results/figures/）：
  fig1_flow_filled.png      队列流程图（全部计数回填，STROBE 式）
  fig3_roc_test.png         测试集 ROC（M0-M4、M1+）
  fig4_calibration.png      测试集校准曲线（M1/M3/M1+，10 分箱 + loess）
  fig5_dca.png              决策曲线（M1/M3/M1+/全干预/全不干预，5%-50%）
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from statsmodels.nonparametric.smoothers_lowess import lowess

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG = REPO_ROOT / "results" / "figures"


def fig1():
    main_boxes = [
        "MIMIC-IV v3.1 全部住院患者（2008-2022）\nN = 364,627（546,028 次住院）",
        "符合 Sepsis-3 判定（疑似感染 + SOFA 急性升高 ≥2）\n41,295 次发作 / 31,910 例患者",
        "入组脓毒症患者\nN = 31,857",
        "ECG 链接成功（每人取距 t0 最近一份）\nN = 18,620",
        "最终分析队列\nN = 16,499",
    ]
    excl = [
        None,
        "排除：年龄 <18 岁（0）；非首次发作（9,385 次）；\nICU 入住 <6 h（37）；t0 前死亡/自动出院（16）",
        "排除：t0±24 h 内无可用 12 导联 ECG（13,218）；\n时长/采样率/导联不合格（19）",
        "排除：信号质量不合格（基线漂移/饱和/噪声超限，2,121）",
        None,
    ]
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.axis("off")
    w, h, gap = 0.52, 0.085, 0.045
    xs, y = 0.06, 0.97
    for i, (txt, ex) in enumerate(zip(main_boxes, excl)):
        y0 = y - h
        ax.add_patch(mpatches.FancyBboxPatch((xs, y0), w, h,
                     boxstyle="round,pad=0.01", fc="#e8f0f8", ec="#336699", lw=1.5))
        ax.text(xs + w / 2, y0 + h / 2, txt, ha="center", va="center", fontsize=11)
        if ex:
            ax.add_patch(mpatches.FancyBboxPatch((0.66, y0), 0.31, h,
                         boxstyle="round,pad=0.01", fc="#fdecea", ec="#cc4444", lw=1.2))
            ax.text(0.815, y0 + h / 2, ex, ha="center", va="center", fontsize=9.5)
        if i < len(main_boxes) - 1:
            ax.annotate("", xy=(xs + w / 2, y0 - gap + 0.005), xytext=(xs + w / 2, y0),
                        arrowprops=dict(arrowstyle="-|>", color="#444444"))
        y = y0 - gap
    # 底部分支
    yb = y - h / 2
    branches = [
        ("开发队列 2008-2016\nN = 14,780\n（train 10,346 / tune 2,217 / test 2,217）", "#dcecf7"),
        ("时间外推验证队列 2017-2019\nN = 1,719（事件 380）", "#e2f0dc"),
        ("COVID 探索队列 2020-2022\n不成立（MIMIC-IV-ECG 不覆盖）", "#fdf3d7"),
    ]
    bw = 0.29
    for i, (txt, fc) in enumerate(branches):
        x0 = 0.04 + i * (bw + 0.04)
        ax.add_patch(mpatches.FancyBboxPatch((x0, yb - h), bw, h * 1.3,
                     boxstyle="round,pad=0.01", fc=fc, ec="#555555", lw=1.2))
        ax.text(x0 + bw / 2, yb - h + h * 0.65, txt, ha="center", va="center", fontsize=9.5)
        ax.annotate("", xy=(x0 + bw / 2, yb + 0.005), xytext=(xs + w / 2, y + gap),
                    arrowprops=dict(arrowstyle="-|>", color="#444444"))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.savefig(FIG / "fig1_flow_filled.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig3_roc():
    pred = pd.read_parquet(REPO_ROOT / "results" / "test_predictions.parquet")
    y = pred["y"].astype(int).to_numpy()
    fig, ax = plt.subplots(figsize=(7, 6.5))
    colors = {"M0": "#999999", "M1": "#1f77b4", "M2": "#9467bd",
              "M3": "#d62728", "M1+": "#2ca02c", "M4": "#ff7f0e"}
    for m in ["M0", "M1", "M2", "M3", "M1+", "M4"]:
        p = pred[f"{m}_cal"].to_numpy()
        auc = roc_auc_score(y, p)
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, lw=2 if m in ("M3", "M1+") else 1.2,
                color=colors[m], label=f"{m} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("1 - 特异度"); ax.set_ylabel("灵敏度")
    ax.set_title("测试集 ROC（N=2,217，事件 425）")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_roc_test.png", dpi=200)
    plt.close(fig)


def fig4_calibration():
    pred = pd.read_parquet(REPO_ROOT / "results" / "test_predictions.parquet")
    y = pred["y"].astype(int).to_numpy()
    fig, ax = plt.subplots(figsize=(7, 6.5))
    colors = {"M1": "#1f77b4", "M3": "#d62728", "M1+": "#2ca02c"}
    for m, c in colors.items():
        p = pred[f"{m}_cal"].to_numpy()
        bins = pd.qcut(p, 10, duplicates="drop")
        gb = pd.DataFrame({"p": p, "y": y}).groupby(bins, observed=True)
        ax.plot(gb["p"].mean(), gb["y"].mean(), "o", color=c, ms=5, alpha=0.7)
        lo = lowess(y, p, frac=0.75, it=0)
        ax.plot(lo[:, 0], lo[:, 1], "-", color=c, lw=2, label=m)
    ax.plot([0, 0.6], [0, 0.6], "--", color="gray", lw=1)
    ax.set_xlim(0, 0.6); ax.set_ylim(0, 0.6)
    ax.set_xlabel("预测概率"); ax.set_ylabel("观察频率")
    ax.set_title("测试集校准曲线（Platt 后；10 分箱 + loess）")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_calibration.png", dpi=200)
    plt.close(fig)


def fig5_dca():
    d = pd.read_csv(REPO_ROOT / "results" / "dca_curve.csv")
    fig, ax = plt.subplots(figsize=(7, 6.5))
    colors = {"M1": "#1f77b4", "M3": "#d62728", "M1+": "#2ca02c"}
    for m, c in colors.items():
        sub = d[d["model"] == m]
        ax.plot(sub["threshold"], sub["net_benefit"], "-", color=c, lw=2, label=m)
    ref = d[d["model"] == "M1"]
    ax.plot(ref["threshold"], ref["treat_all"], "--", color="gray", lw=1.2,
            label="全部干预")
    ax.axhline(0, color="black", lw=1, label="全不干预")
    ax.set_xlim(0.05, 0.50)
    ax.set_ylim(-0.05, 0.20)
    ax.set_xlabel("阈值概率"); ax.set_ylabel("净获益")
    ax.set_title("决策曲线分析（测试集，阈值 5%-50%）")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_dca.png", dpi=200)
    plt.close(fig)


def main() -> None:
    fig1(); print("fig1_flow_filled.png")
    fig3_roc(); print("fig3_roc_test.png")
    fig4_calibration(); print("fig4_calibration.png")
    fig5_dca(); print("fig5_dca.png")


if __name__ == "__main__":
    main()
