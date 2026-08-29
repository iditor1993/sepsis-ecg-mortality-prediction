"""Generate English-language publication figures for the IJMI submission.

All figures are generated from the locked analysis outputs in the repository so
that the numbers and curves match the manuscript exactly. No Chinese labels are
used in the final figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from statsmodels.nonparametric.smoothers_lowess import lowess


REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "figures"
RES = REPO / "results"
DATA = REPO / "data"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.unicode_minus": False,
})

OUT.mkdir(parents=True, exist_ok=True)


def fig1_cohort() -> None:
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.axis("off")
    main_boxes = [
        "MIMIC-IV v3.1 hospital admissions (2008\u20132022)\nN = 364,627 admissions (546,028 stays)",
        "Sepsis-3 criteria (suspected infection + acute SOFA increase \u22652)\n41,295 episodes / 31,910 patients",
        "Eligible sepsis patients after first-episode and ICU rules\nN = 31,857",
        "ECG link success (nearest ECG within t0\u00b124 h)\nN = 18,620",
        "Final analysis cohort (ECG signal quality passed)\nN = 16,499",
    ]
    exclusions = [
        "",
        "Excluded: age <18 (0); non-first episode (9,385);\nICU stay <6 h (37); death/auto-discharge before t0 (16)",
        "Excluded: no eligible 12-lead ECG within t0\u00b124 h (13,218);\ninvalid duration/sampling/leads (19)",
        "Excluded: poor signal quality (baseline drift, saturation,\nhigh-frequency noise) (2,121)",
        "",
    ]
    w, h, gap = 0.52, 0.075, 0.035
    x0, y = 0.06, 0.98
    for i, (text, excl) in enumerate(zip(main_boxes, exclusions)):
        y0 = y - h
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x0, y0), w, h,
                boxstyle="round,pad=0.01", fc="#e8f0f8", ec="#336699", lw=1.5,
            )
        )
        ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center", fontsize=10.5)
        if excl:
            ax.add_patch(
                mpatches.FancyBboxPatch(
                    (0.66, y0), 0.32, h,
                    boxstyle="round,pad=0.01", fc="#fdecea", ec="#cc4444", lw=1.2,
                )
            )
            ax.text(0.82, y0 + h / 2, excl, ha="center", va="center", fontsize=8.5)
        if i < len(main_boxes) - 1:
            ax.annotate(
                "", xy=(x0 + w / 2, y0 - gap + 0.005),
                xytext=(x0 + w / 2, y0),
                arrowprops=dict(arrowstyle="-|>", color="#444444"),
            )
        y = y0 - gap

    yb = y - h / 2
    branches = [
        ("Development 2008\u20132016\nN = 14,780\n(train 10,346 / tune 2,217 / test 2,217)", "#dcecf7"),
        ("Temporal validation 2017\u20132019\nN = 1,719 (events = 380)", "#e2f0dc"),
        ("COVID window 2020\u20132022\nNot feasible: no linked ECGs", "#fdf3d7"),
    ]
    bw = 0.29
    for i, (text, fc) in enumerate(branches):
        bx = 0.04 + i * (bw + 0.04)
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (bx, yb - h), bw, h * 1.35,
                boxstyle="round,pad=0.01", fc=fc, ec="#555555", lw=1.2,
            )
        )
        ax.text(bx + bw / 2, yb - h + h * 0.68, text, ha="center", va="center", fontsize=9)
        ax.annotate(
            "", xy=(bx + bw / 2, yb + 0.005), xytext=(x0 + w / 2, y + gap),
            arrowprops=dict(arrowstyle="-|>", color="#444444"),
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Figure 1. Study cohort flow and development/validation sets",
                 fontsize=13, pad=12)
    fig.savefig(OUT / "Fig1_cohort_flow.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig2_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.axis("off")

    def box(x, y, w, h, text, fc="#eef4fb", ec="#336699", fontsize=10):
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.01", fc=fc, ec=ec, lw=1.3,
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.2))

    box(0.02, 0.69, 0.20, 0.22,
        "MIMIC-IV clinical data\n(t0\u00b124 h vitals, labs,\nscores, covariates)")
    box(0.02, 0.32, 0.20, 0.22,
        "MIMIC-IV-ECG\n12-lead diagnostic ECG,\n10 s at 500 Hz")
    box(0.30, 0.78, 0.25, 0.13,
        "Eligibility and ECG QC\nwindow = t0\u00b124 h;\nnearest ECG per patient")
    box(0.30, 0.52, 0.25, 0.13,
        "Track A\nfrozen V14 1D CNN autoencoder\nLead II \u2192 z1\u2013z16")
    box(0.30, 0.24, 0.25, 0.13,
        "Track B\n2D ResNet-like network from zero\nPCA \u2192 tb1\u2013tb32")
    box(0.63, 0.58, 0.18, 0.30,
        "Clinical feature matrix\nscores, lactate, covariates\nMICE imputation, m = 20")
    box(0.63, 0.16, 0.18, 0.30,
        "ECG representation\n16 or 32 latent dimensions\nfrozen or task-specific")
    box(0.87, 0.58, 0.12, 0.30,
        "M0 SOFA\nM1 clinical\nM1+ strong table\nM2 ECG only\nM3 ECG + clinical\nM4 gradient boosting\nM5 end-to-end",
        fontsize=8.5)
    box(0.87, 0.16, 0.12, 0.30,
        "Locked evaluation\nTune for calibration\nTest once\nTemporal validation",
        fontsize=8.5)
    arrow(0.22, 0.80, 0.30, 0.85)
    arrow(0.22, 0.43, 0.30, 0.85)
    arrow(0.22, 0.43, 0.30, 0.58)
    arrow(0.22, 0.43, 0.30, 0.30)
    arrow(0.55, 0.58, 0.63, 0.68)
    arrow(0.55, 0.30, 0.63, 0.25)
    arrow(0.81, 0.68, 0.87, 0.73)
    arrow(0.81, 0.35, 0.87, 0.60)

    ax.text(0.50, 0.96,
            "Figure 2. Data pipeline, model hierarchy and validation design",
            ha="center", fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig(OUT / "Fig2_analysis_pipeline.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig3_roc() -> None:
    pred = pd.read_parquet(RES / "test_predictions.parquet")
    y = pred["y"].astype(int).to_numpy()
    fig, ax = plt.subplots(figsize=(7, 6.4))
    colors = {
        "M0": "#999999", "M1": "#1f77b4", "M2": "#9467bd",
        "M3": "#d62728", "M1+": "#2ca02c", "M4": "#ff7f0e",
    }
    for model in ["M0", "M1", "M2", "M3", "M1+", "M4"]:
        p = pred[f"{model}_cal"].to_numpy()
        auc = roc_auc_score(y, p)
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(
            fpr, tpr, lw=2.0 if model in ("M3", "M1+") else 1.2,
            color=colors[model],
            label=f"{model} (AUC = {auc:.3f})",
        )
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("1 \u2212 specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title("Test-set ROC curves (N = 2,217; 425 events)")
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / "Fig3_ROC.png", dpi=300)
    plt.close(fig)


def fig4_calibration() -> None:
    pred = pd.read_parquet(RES / "test_predictions.parquet")
    y = pred["y"].astype(int).to_numpy()
    fig, ax = plt.subplots(figsize=(7, 6.4))
    colors = {"M1": "#1f77b4", "M3": "#d62728", "M1+": "#2ca02c"}
    for model, color in colors.items():
        p = pred[f"{model}_cal"].to_numpy()
        bins = pd.qcut(p, 10, duplicates="drop")
        grouped = pd.DataFrame({"p": p, "y": y}).groupby(bins, observed=True)
        ax.plot(grouped["p"].mean(), grouped["y"].mean(), "o", color=color, ms=5, alpha=0.7)
        lo = lowess(y, p, frac=0.75, it=0)
        ax.plot(lo[:, 0], lo[:, 1], "-", color=color, lw=2, label=model)
    ax.plot([0, 0.6], [0, 0.6], "--", color="gray", lw=1)
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 0.6)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Test-set calibration (Platt calibration; 10 bins + loess)")
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / "Fig4_calibration.png", dpi=300)
    plt.close(fig)


def fig5_dca() -> None:
    d = pd.read_csv(RES / "dca_curve.csv")
    fig, ax = plt.subplots(figsize=(7, 6.4))
    colors = {"M1": "#1f77b4", "M3": "#d62728", "M1+": "#2ca02c"}
    for model, color in colors.items():
        sub = d[d["model"] == model]
        ax.plot(sub["threshold"], sub["net_benefit"], "-", color=color, lw=2, label=model)
    ref = d[d["model"] == "M1"]
    ax.plot(ref["threshold"], ref["treat_all"], "--", color="gray", lw=1.2, label="Treat all")
    ax.axhline(0, color="black", lw=1, label="Treat none")
    ax.set_xlim(0.05, 0.50)
    ax.set_ylim(-0.05, 0.20)
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision curve analysis (test set, 5%\u201350%)")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / "Fig5_DCA.png", dpi=300)
    plt.close(fig)


def fig6_subgroups() -> None:
    d = pd.read_csv(RES / "subgroup_analysis.csv")
    group_map = {
        "年龄": "Age",
        "性别": "Sex",
        "脓毒性休克": "Septic shock",
        "心室率": "Ventricular rate",
        "房颤": "Atrial fibrillation",
        "SOFA三分位": "SOFA tertile",
        "感染部位": "Infection site",
    }
    level_map = {
        "女": "Female",
        "男": "Male",
        "无": "No",
        "有": "Yes",
        "<65": "<65",
        "≥65": "\u226565",
        "<100": "<100",
        "≥100": "\u2265100",
    }
    labels = []
    fig, ax = plt.subplots(figsize=(8, max(5, len(d) * 0.45)))
    for i, row in enumerate(d.itertuples(index=False)):
        ax.plot(
            [row.ci_lo, row.ci_hi], [i, i], "-", lw=2,
            color="steelblue",
        )
        ax.plot(row.delta_auc, i, "s", color="darkred", ms=6)
        group = group_map.get(row.亚组, row.亚组)
        level = level_map.get(row.水平, row.水平)
        if group == "SOFA tertile":
            level = level.replace("T", "Tertile ")
        labels.append(f"{group} | {level}")
    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.set_yticks(range(len(d)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("\u0394AUC (M3 \u2212 M1), test set")
    ax.set_title("Prespecified subgroup \u0394AUC with 95% bootstrap CIs")
    fig.tight_layout()
    fig.savefig(OUT / "Fig6_subgroups.png", dpi=300)
    plt.close(fig)


def fig7_shap() -> None:
    m3 = pd.read_csv(RES / "shap_m3_importance.csv").head(15)
    m4 = pd.read_csv(RES / "shap_m4_importance.csv").head(15)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), sharey=True)
    for ax, imp, title in zip(
        axes, [m3, m4],
        ["M3 (LASSO logistic regression)", "M4 (gradient boosting)"],
    ):
        imp = imp.iloc[::-1]
        ax.barh(imp["feature"], imp["mean_abs_shap"], color="steelblue")
        ax.set_title(title)
        ax.set_xlabel("Mean |SHAP|")
    fig.tight_layout()
    fig.savefig(OUT / "Fig7_SHAP.png", dpi=300)
    plt.close(fig)


def fig8_e1() -> None:
    d = pd.read_csv(RES / "e1_correlation.csv")
    targets = [
        "sofa_respiration", "sofa_coagulation", "sofa_liver",
        "sofa_cardiovascular", "sofa_cns", "sofa_renal", "lactate",
    ]
    target_names = [
        "Respiration", "Coagulation", "Liver", "Cardiovascular",
        "Neurological", "Renal", "Lactate",
    ]
    z_order = [f"z{i}" for i in range(1, 17)]
    matrix = d.pivot(index="z", columns="target", values="spearman_rho").reindex(
        index=z_order, columns=targets
    )
    sig = d.pivot(index="z", columns="target", values="fdr_sig").reindex(
        index=z_order, columns=targets
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    ax.set_xticks(range(len(targets)), target_names, rotation=30, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    for i in range(len(matrix.index)):
        for j in range(len(targets)):
            value = matrix.iloc[i, j]
            marker = "*" if sig.iloc[i, j] else ""
            ax.text(j, i, f"{value:.2f}{marker}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Spearman \u03c1")
    ax.set_title("Latent ECG dimensions versus clinical scores and lactate\n(* BH-FDR q < 0.05)")
    fig.tight_layout()
    fig.savefig(OUT / "Fig8_latent_spectrum.png", dpi=300)
    plt.close(fig)


def graphical_abstract() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.75))
    ax.axis("off")
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.03, 0.35), 0.27, 0.28,
            boxstyle="round,pad=0.01", fc="#eef4fb", ec="#336699", lw=2,
        )
    )
    ax.text(
        0.165, 0.55,
        "16,499 sepsis patients\nMIMIC-IV + ECG",
        ha="center", va="center", fontsize=13,
    )
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.37, 0.35), 0.27, 0.28,
            boxstyle="round,pad=0.01", fc="#f2f7ef", ec="#4a8a4a", lw=2,
        )
    )
    ax.text(
        0.505, 0.55,
        "Clinical pathway: AUC 0.854\nECG pathway alone: AUC 0.642\nCombined: AUC 0.795",
        ha="center", va="center", fontsize=12,
    )
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.71, 0.35), 0.26, 0.28,
            boxstyle="round,pad=0.01", fc="#fdecea", ec="#cc4444", lw=2,
        )
    )
    ax.text(
        0.84, 0.55,
        "No confirmed increment\n\u0394AUC (M3 vs M1) = \u22120.0014\n\u0394AUC (M3 vs M1+) = \u22120.0587",
        ha="center", va="center", fontsize=12,
    )
    ax.annotate("", xy=(0.37, 0.49), xytext=(0.30, 0.49),
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=2))
    ax.annotate("", xy=(0.71, 0.49), xytext=(0.64, 0.49),
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=2))
    ax.text(0.50, 0.18,
            "Sensitivity analyses, temporal validation, availability adjustment and\n"
            "interpretability all supported the null incremental-value finding.",
            ha="center", fontsize=12, color="#333333")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig(OUT / "Graphical_abstract.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fig1_cohort()
    fig2_pipeline()
    fig3_roc()
    fig4_calibration()
    fig5_dca()
    fig6_subgroups()
    fig7_shap()
    fig8_e1()
    graphical_abstract()
    print(f"Figures written to {OUT}")


if __name__ == "__main__":
    main()
