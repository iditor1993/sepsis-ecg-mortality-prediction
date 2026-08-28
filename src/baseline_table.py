"""baseline_table.py — 描述性基线表（SAP 9.2，表 1）。

最终分析队列（16,499 例），按 28 天全因死亡分层：
  连续变量正态者 mean±SD（t 检验）、偏态者 median(IQR)（Mann-Whitney U）；
  分类变量 n(%)（卡方/Fisher）；全部报告标准化差异（SMD>0.1 判不均衡）。
输出 results/baseline_table.csv 并打印。
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent

CONT_NORMAL = [("admission_age", "年龄（岁）")]
CONT_SKEW = [
    ("sofa_score", "SOFA 总分"), ("sofa_respiration", "SOFA-呼吸"),
    ("sofa_coagulation", "SOFA-凝血"), ("sofa_liver", "SOFA-肝脏"),
    ("sofa_cardiovascular", "SOFA-心血管"), ("sofa_cns", "SOFA-神经"),
    ("sofa_renal", "SOFA-肾脏"), ("qsofa", "qSOFA"), ("news", "NEWS"),
    ("mews", "MEWS"), ("lactate", "乳酸（mmol/L）"),
    ("charlson_comorbidity_index", "Charlson 指数"),
    ("pre_icu_los_h", "入 ICU 前住院时长（h）"),
    ("signed_t0_diff_h", "ECG 距 t0 时间（h，带符号）"),
]
BIN_VARS = [("male", "男性"), ("admission_emergency", "急诊入院"),
            ("mech_vent_24h", "有创机械通气（t0±24h）"),
            ("vaso_24h", "血管活性药物（t0±24h）")]
SITE_CN = {"respiratory": "呼吸", "abdominal": "腹腔", "urinary": "泌尿",
           "bloodstream": "血流", "other": "其他"}


def smd_cont(a, b):
    sd = np.sqrt((a.var() + b.var()) / 2)
    return (a.mean() - b.mean()) / sd if sd > 0 else 0.0


def smd_bin(a, b):
    pa, pb = a.mean(), b.mean()
    sd = np.sqrt((pa * (1 - pa) + pb * (1 - pb)) / 2)
    return (pa - pb) / sd if sd > 0 else 0.0


def main() -> None:
    datadir = REPO_ROOT / "data"
    results_dir = REPO_ROOT / "results"
    dev = pd.read_parquet(datadir / "features_dev.parquet")
    temporal = pd.read_parquet(datadir / "features_temporal.parquet")
    scores = pd.read_parquet(datadir / "clinical_scores.parquet")
    df = pd.concat([dev, temporal], ignore_index=True)
    df = df.merge(scores[["stay_id", "qsofa", "news", "mews"]], on="stay_id", how="left")
    df["male"] = (df["gender"] == "M").astype(int)

    y = df["death_28d"].astype(bool)
    g1, g0 = df[y], df[~y]  # 死亡 / 存活
    n1, n0 = len(g1), len(g0)

    rows = []
    for col, label in CONT_NORMAL:
        a, b = g1[col].dropna(), g0[col].dropna()
        p = stats.ttest_ind(a, b, equal_var=False).pvalue
        rows.append({"变量": label, "分组统计": f"{a.mean():.1f}±{a.std():.1f} / "
                     f"{b.mean():.1f}±{b.std():.1f}",
                     "死亡组(n={:,})".format(n1): f"{a.mean():.1f}±{a.std():.1f}",
                     "存活组(n={:,})".format(n0): f"{b.mean():.1f}±{b.std():.1f}",
                     "缺失n(%)": f"{df[col].isna().sum()}({df[col].isna().mean():.1%})",
                     "SMD": round(float(smd_cont(a, b)), 3), "p": p})
    for col, label in CONT_SKEW:
        a, b = g1[col].dropna(), g0[col].dropna()
        p = stats.mannwhitneyu(a, b).pvalue
        fmt = lambda s: f"{s.median():.1f}({s.quantile(0.25):.1f}-{s.quantile(0.75):.1f})"
        rows.append({"变量": label,
                     "死亡组(n={:,})".format(n1): fmt(a),
                     "存活组(n={:,})".format(n0): fmt(b),
                     "缺失n(%)": f"{df[col].isna().sum()}({df[col].isna().mean():.1%})",
                     "SMD": round(float(smd_cont(a, b)), 3), "p": p})
    for col, label in BIN_VARS:
        a, b = g1[col].astype(float), g0[col].astype(float)
        ct = pd.crosstab(y, df[col])
        p = stats.chi2_contingency(ct).pvalue
        rows.append({"变量": label,
                     "死亡组(n={:,})".format(n1): f"{int(a.sum()):,}({a.mean():.1%})",
                     "存活组(n={:,})".format(n0): f"{int(b.sum()):,}({b.mean():.1%})",
                     "缺失n(%)": "0(0.0%)",
                     "SMD": round(float(smd_bin(a, b)), 3), "p": p})
    for site, cn in SITE_CN.items():
        a = (g1["infection_site"] == site).astype(float)
        b = (g0["infection_site"] == site).astype(float)
        ct = pd.crosstab(y, df["infection_site"] == site)
        p = stats.chi2_contingency(ct).pvalue
        rows.append({"变量": f"感染部位-{cn}",
                     "死亡组(n={:,})".format(n1): f"{int(a.sum()):,}({a.mean():.1%})",
                     "存活组(n={:,})".format(n0): f"{int(b.sum()):,}({b.mean():.1%})",
                     "缺失n(%)": "0(0.0%)",
                     "SMD": round(float(smd_bin(a, b)), 3), "p": p})

    tab = pd.DataFrame(rows)
    tab["p"] = tab["p"].map(lambda x: f"{x:.3g}" if x >= 1e-4 else "<0.0001")
    tab.to_csv(results_dir / "baseline_table.csv", index=False, encoding="utf-8-sig")

    print("=" * 78)
    print(f"表 1：基线特征按 28 天死亡分层（死亡 {n1:,} / 存活 {n0:,}，总计 {len(df):,}）")
    print("=" * 78)
    print(tab.to_string(index=False))
    n_imb = int((tab["SMD"].abs() > 0.1).sum())
    print(f"\n|SMD|>0.1 的变量数: {n_imb}/{len(tab)}")
    print("输出 -> results/baseline_table.csv")


if __name__ == "__main__":
    main()
