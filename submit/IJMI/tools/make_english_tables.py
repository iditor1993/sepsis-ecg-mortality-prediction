"""Create editable English-language table files for the IJMI package."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[3]
RES = REPO / "results"
DATA = REPO / "data"
OUT = Path(__file__).resolve().parents[1] / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def table1() -> None:
    df = pd.read_csv(RES / "baseline_table.csv")
    mapping = {
        "年龄（岁）": "Age (years)",
        "SOFA 总分": "SOFA total",
        "SOFA-呼吸": "SOFA respiratory",
        "SOFA-凝血": "SOFA coagulation",
        "SOFA-肝脏": "SOFA liver",
        "SOFA-心血管": "SOFA cardiovascular",
        "SOFA-神经": "SOFA neurological",
        "SOFA-肾脏": "SOFA renal",
        "qSOFA": "qSOFA",
        "NEWS": "NEWS",
        "MEWS": "MEWS",
        "乳酸（mmol/L）": "Lactate (mmol/L)",
        "Charlson 指数": "Charlson index",
        "入 ICU 前住院时长（h）": "Pre-ICU length of stay (h)",
        "ECG 距 t0 时间（h，带符号）": "ECG time from t0 (h, signed)",
        "男性": "Male",
        "急诊入院": "Emergency admission",
        "有创机械通气（t0±24h）": "Invasive mechanical ventilation (t0±24 h)",
        "血管活性药物（t0±24h）": "Vasoactive drugs (t0±24 h)",
        "感染部位-呼吸": "Infection site: respiratory",
        "感染部位-腹腔": "Infection site: abdominal",
        "感染部位-泌尿": "Infection site: urinary",
        "感染部位-血流": "Infection site: bloodstream",
        "感染部位-其他": "Infection site: other",
    }
    df["Variable"] = df["变量"].map(mapping)
    df = df.drop(columns=["变量"])
    df = df.rename(columns={
        "分组统计": "Summary statistic",
        "死亡组(n=3,002)": "Deceased (n=3,002)",
        "存活组(n=13,497)": "Survivors (n=13,497)",
        "缺失n(%)": "Missing n (%)",
        "SMD": "SMD",
        "p": "P",
    })
    df.to_csv(OUT / "Table1_baseline_characteristics.csv", index=False, encoding="utf-8-sig")


def table2() -> None:
    metrics = pd.read_csv(RES / "test_metrics.csv")
    ci = metrics.rename(columns={
        "model": "Model",
        "test_auc": "Test AUC",
        "auc_lo": "Test AUC 95% CI lower",
        "auc_hi": "Test AUC 95% CI upper",
        "temporal_auc": "Temporal AUC",
        "auc_drop": "Temporal AUC drop",
    })[["Model", "Test AUC", "Test AUC 95% CI lower", "Test AUC 95% CI upper", "Temporal AUC", "Temporal AUC drop"]]
    m5 = pd.read_csv(RES / "m5_results.csv")
    m5_row = {
        "Model": "M5",
        "Test AUC": 0.8147,
        "Test AUC 95% CI lower": "",
        "Test AUC 95% CI upper": "",
        "Temporal AUC": "",
        "Temporal AUC drop": "",
    }
    ci = pd.concat([ci, pd.DataFrame([m5_row])], ignore_index=True)
    m4plus = pd.DataFrame([
        {
            "Model": "M4+",
            "Test AUC": 0.8572,
            "Test AUC 95% CI lower": "",
            "Test AUC 95% CI upper": "",
            "Temporal AUC": "",
            "Temporal AUC drop": "",
        },
        {
            "Model": "M4+-TB",
            "Test AUC": 0.8394,
            "Test AUC 95% CI lower": "",
            "Test AUC 95% CI upper": "",
            "Temporal AUC": "",
            "Temporal AUC drop": "",
        },
    ])
    ci = pd.concat([ci, m4plus], ignore_index=True)
    ci.to_csv(OUT / "Table2_model_performance.csv", index=False, encoding="utf-8-sig")


def table3() -> None:
    h1 = pd.read_csv(RES / "h1_delong.csv")
    h2 = pd.read_csv(RES / "h2_nri_idi.csv")
    rows = []
    for _, row in h1.iterrows():
        rows.append({
            "Comparison": row["comparison"],
            "Metric": "ΔAUC",
            "Estimate": row["delta_auc"],
            "95% CI lower": row["ci_lo"],
            "95% CI upper": row["ci_hi"],
            "P": row["delong_p"],
        })
    for _, row in h2.iterrows():
        rows.append({
            "Comparison": row["comparison"],
            "Metric": row["metric"],
            "Estimate": row["estimate"],
            "95% CI lower": row["ci_lo"],
            "95% CI upper": row["ci_hi"],
            "P": row["p"],
        })
    pd.DataFrame(rows).to_csv(OUT / "Table3_incremental_value.csv", index=False, encoding="utf-8-sig")


def table4() -> None:
    rows = []
    s1 = pd.read_csv(RES / "sensitivity_s1.csv").iloc[0]
    rows.append(["S1", "ECG window [t0−48 h, t0)", f"ΔAUC {s1['delta_auc']:+.4f} (95% CI {s1['ci_lo']:+.4f} to {s1['ci_hi']:+.4f})"])
    s2 = pd.read_csv(RES / "sensitivity_s2.csv").iloc[0]
    rows.append(["S2", "Outcome from admission", f"ΔAUC {s2['delta_auc']:+.4f} (95% CI {s2['ci_lo']:+.4f} to {s2['ci_hi']:+.4f})"])
    rows.append(["S3", "Exclude ICU-only subcohort", "Not applicable; cohort is ICU-level"])
    mix = pd.read_csv(RES / "sensitivity_s4_s5_s9_s10.csv")
    for _, r in mix.iterrows():
        if r["id"] == "S9":
            rows.append(["S9", "Calibration method comparison", "Brier raw 0.1265; Platt 0.1266; isotonic 0.1269"])
        else:
            rows.append([r["id"], r["内容"], f"AUC M3 {r['test_auc']:.4f}; ΔAUC {r['delta_vs_m1']:+.4f}"])
    s67 = pd.read_csv(RES / "sensitivity_s6_s7.csv")
    for _, r in s67.iterrows():
        rows.append([r["id"], r["内容"], f"ΔAUC {r['delta_auc']:+.4f} (95% CI {r['ci_lo']:+.4f} to {r['ci_hi']:+.4f})"])
    s8 = pd.read_csv(RES / "sensitivity_s8.csv")
    lac = s8[s8["feature"] == "lactate_imp_mean"].iloc[0]
    rows.append(["S8", "Competing-risk (Fine-Gray) framework", f"Lactate subdistribution HR {lac['subHR_FG']:.3f}; P {lac['p_FG']:.2e}"])
    a1 = pd.read_csv(RES / "a1b_availability.csv")
    rows.append(["S11", "ECG-availability indicator", f"Independent ΔAUC +{a1.iloc[0]['delta']:.4f}; adjusted M3 ΔAUC {a1.iloc[1]['delta']:.4f}"])
    enc = pd.read_csv(RES / "sensitivity_encoder_retrain.csv")
    row = enc[enc["model"].str.startswith("M3'")].iloc[0]
    rows.append(["Re-ENC", "Re-training encoder internally", f"M3' AUC {row['test_auc']:.4f}; vs M1' ΔAUC {row['delta_auc']:+.4f}"])
    pd.DataFrame(rows, columns=["ID", "Analysis", "Main result"]).to_csv(
        OUT / "Table4_sensitivity_analyses.csv", index=False, encoding="utf-8-sig"
    )


def table5() -> None:
    m3 = pd.read_csv(RES / "shap_m3_importance.csv").reset_index()
    m4 = pd.read_csv(RES / "shap_m4_importance.csv").reset_index()
    m3["rank"] = m3.index + 1
    m4["rank"] = m4.index + 1
    rows = []
    for rank in range(1, 16):
        a = m3[m3["rank"] == rank]
        b = m4[m4["rank"] == rank]
        rows.append({
            "M3 rank": rank,
            "M3 feature": a["feature"].iloc[0],
            "M3 mean |SHAP|": a["mean_abs_shap"].iloc[0],
            "M4 rank": rank,
            "M4 feature": b["feature"].iloc[0],
            "M4 mean |SHAP|": b["mean_abs_shap"].iloc[0],
        })
    pd.DataFrame(rows).to_csv(OUT / "Table5_SHAP_top_features.csv", index=False, encoding="utf-8-sig")


def table6_availability() -> None:
    df = pd.read_csv(DATA / "a1a_comparison.csv")
    mapping = {
        "年龄（岁）": "Age (years)",
        "SOFA 总分": "SOFA total",
        "SOFA-呼吸": "SOFA respiratory",
        "SOFA-凝血": "SOFA coagulation",
        "SOFA-肝脏": "SOFA liver",
        "SOFA-心血管": "SOFA cardiovascular",
        "SOFA-神经": "SOFA neurological",
        "SOFA-肾脏": "SOFA renal",
        "Charlson 指数": "Charlson index",
        "男性": "Male",
        "有创机械通气（t0±24h）": "Invasive mechanical ventilation (t0±24 h)",
        "血管活性药物（t0±24h）": "Vasoactive drugs (t0±24 h)",
        "28 天全因死亡": "28-day all-cause mortality",
    }
    df["Variable"] = df["variable"].map(mapping)
    df = df.rename(columns={
        "type": "Type",
        "available": "ECG available",
        "unavailable": "ECG unavailable",
        "smd": "SMD",
        "p_value": "P",
        "smd_gt_0.1": "|SMD| > 0.1",
    }).drop(columns=["variable"])
    df["Type"] = df["Type"].map({"连续": "Continuous", "分类": "Categorical"})
    df.to_csv(OUT / "Table6_ECG_availability.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    table1()
    table2()
    table3()
    table4()
    table5()
    table6_availability()
    print(f"Tables written to {OUT}")


if __name__ == "__main__":
    main()
