"""availability.py — A1 ECG 可得性对照分析（SAP 9.7 节，V1.2 新增）。

A1a 描述性对照（本脚本）：比较 ECG 可链接且合格者（ecg_available=True）与
不可链接/不合格者的基线特征、SOFA、治疗强度与 28 天死亡率；
连续变量报告 mean±SD 与 Mann-Whitney U 检验，分类变量报告 n(%) 与卡方检验；
以 |SMD|>0.1 判定系统性差异。输出 data/a1a_comparison.csv。

A1b 指示变量对照（ECG 可得性指示变量加入 M1+ 与 M3 比较）依赖 M1+ 模型，
在模型训练阶段（W4-W5）另行执行，不在本脚本内。
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent

CONT_VARS = [
    ("admission_age", "年龄（岁）"),
    ("sofa_score", "SOFA 总分"),
    ("sofa_respiration", "SOFA-呼吸"),
    ("sofa_coagulation", "SOFA-凝血"),
    ("sofa_liver", "SOFA-肝脏"),
    ("sofa_cardiovascular", "SOFA-心血管"),
    ("sofa_cns", "SOFA-神经"),
    ("sofa_renal", "SOFA-肾脏"),
    ("charlson_comorbidity_index", "Charlson 指数"),
]
BIN_VARS = [
    ("male", "男性"),
    ("mech_vent_24h", "有创机械通气（t0±24h）"),
    ("vaso_24h", "血管活性药物（t0±24h）"),
    ("death_28d", "28 天全因死亡"),
]


def smd_cont(a: pd.Series, b: pd.Series) -> float:
    sd = np.sqrt((a.var() + b.var()) / 2)
    return float((a.mean() - b.mean()) / sd) if sd > 0 else 0.0


def smd_bin(a: pd.Series, b: pd.Series) -> float:
    pa, pb = a.mean(), b.mean()
    sd = np.sqrt((pa * (1 - pa) + pb * (1 - pb)) / 2)
    return float((pa - pb) / sd) if sd > 0 else 0.0


def main() -> None:
    datadir = REPO_ROOT / "data"
    df = pd.read_parquet(datadir / "ecg_linked.parquet")
    out = pd.read_parquet(datadir / "outcomes.parquet")
    inten = pd.read_parquet(datadir / "treatment_intensity.parquet")
    df = df.merge(out[["stay_id", "death_28d"]], on="stay_id", how="left")
    df = df.merge(inten, on=["subject_id", "stay_id"], how="left")
    df["male"] = df["gender"] == "M"

    g1 = df[df["ecg_available"] == True]  # noqa: E712 可链接且合格
    g0 = df[df["ecg_available"] != True]  # noqa: E712 不可链接/不合格

    rows = []
    for col, label in CONT_VARS:
        a, b = g1[col].dropna(), g0[col].dropna()
        p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
        rows.append({
            "variable": label, "type": "连续",
            "available": f"{a.mean():.2f}±{a.std():.2f}",
            "unavailable": f"{b.mean():.2f}±{b.std():.2f}",
            "smd": round(smd_cont(a, b), 3), "p_value": p,
        })
    for col, label in BIN_VARS:
        a, b = g1[col].astype(float), g0[col].astype(float)
        ct = pd.crosstab(df["ecg_available"] == True, df[col])  # noqa: E712
        p = stats.chi2_contingency(ct).pvalue
        rows.append({
            "variable": label, "type": "分类",
            "available": f"{a.mean():.1%} (n={int(a.sum()):,})",
            "unavailable": f"{b.mean():.1%} (n={int(b.sum()):,})",
            "smd": round(smd_bin(a, b), 3), "p_value": p,
        })

    tab = pd.DataFrame(rows)
    tab["smd_gt_0.1"] = tab["smd"].abs() > 0.1
    tab.to_csv(datadir / "a1a_comparison.csv", index=False)

    print("=" * 72)
    print(f"A1a ECG 可得性描述性对照（SAP 9.7）：可链接合格 N={len(g1):,} "
          f"vs 不可链接/不合格 N={len(g0):,}")
    print("=" * 72)
    show = tab.copy()
    show["p_value"] = show["p_value"].map(lambda x: f"{x:.2e}")
    print(show.to_string(index=False))
    n_imb = int(tab["smd_gt_0.1"].sum())
    print(f"\n|SMD|>0.1 的变量数: {n_imb}/{len(tab)}")
    if n_imb:
        print("系统性差异变量: "
              + "、".join(tab.loc[tab["smd_gt_0.1"], "variable"].tolist()))
    print("\n输出: data/a1a_comparison.csv")
    print("提示: A1b（可得性指示变量对照模型）待 M1+ 训练后执行。")


if __name__ == "__main__":
    main()
