"""Generate Supplementary Table S1: development versus temporal cohort."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parents[1] / "data"
OUT = ROOT / "supplementary" / "Table_S1_development_vs_temporal.csv"


def fmt_quantile(s: pd.Series) -> str:
    return f"{s.median():.1f} ({s.quantile(0.25):.1f}-{s.quantile(0.75):.1f})"


def main() -> None:
    splits = pd.read_csv(DATA / "splits.csv")
    base = pd.read_parquet(DATA / "cohort_base.parquet")[
        ["subject_id", "stay_id", "gender", "admission_age", "sofa_score"]
    ]
    scores = pd.read_parquet(DATA / "clinical_scores_all.parquet")[
        ["subject_id", "stay_id", "qsofa", "news", "mews"]
    ]
    cov = pd.read_parquet(DATA / "covariates_all.parquet")[
        ["subject_id", "stay_id", "lactate", "admission_emergency"]
    ]
    trt = pd.read_parquet(DATA / "treatment_intensity.parquet")[
        ["subject_id", "stay_id", "mech_vent_24h", "vaso_24h", "charlson_comorbidity_index"]
    ]
    out = pd.read_parquet(DATA / "outcomes.parquet")[
        ["subject_id", "stay_id", "death_28d"]
    ]

    merged = (
        splits.merge(base, on=["subject_id", "stay_id"], how="left")
        .merge(scores, on=["subject_id", "stay_id"], how="left")
        .merge(cov, on=["subject_id", "stay_id"], how="left")
        .merge(trt, on=["subject_id", "stay_id"], how="left")
        .merge(out, on=["subject_id", "stay_id"], how="left")
    )
    merged["cohort"] = merged["subset"].map(
        {"train": "Development", "tune": "Development", "test": "Development", "temporal": "Temporal"}
    )

    summaries = {}
    for name, g in merged.groupby("cohort", sort=False):
        summaries[name] = [
            ["N", f"{len(g)}"],
            ["28-day deaths", f"{g.death_28d.sum():,} ({g.death_28d.mean()*100:.1f}%)"],
            ["Age (years)", f"{g.admission_age.mean():.1f} ± {g.admission_age.std():.1f}"],
            ["Male", f"{(g.gender == 'M').mean()*100:.1f}%"],
            ["SOFA total", fmt_quantile(g.sofa_score)],
            ["qSOFA", fmt_quantile(g.qsofa)],
            ["NEWS", fmt_quantile(g.news)],
            ["MEWS", fmt_quantile(g.mews)],
            ["Lactate (mmol/L)", fmt_quantile(g.lactate)],
            ["Charlson index", fmt_quantile(g.charlson_comorbidity_index)],
            ["Mechanical ventilation (t0±24 h)", f"{g.mech_vent_24h.mean()*100:.1f}%"],
            ["Vasoactive drugs (t0±24 h)", f"{g.vaso_24h.mean()*100:.1f}%"],
            ["Emergency admission", f"{g.admission_emergency.mean()*100:.1f}%"],
        ]

    labels = [row[0] for row in summaries["Development"]]
    output = pd.DataFrame(
        {
            "Variable": labels,
            "Development": [dict(summaries["Development"]).get(label, "") for label in labels],
            "Temporal": [dict(summaries["Temporal"]).get(label, "") for label in labels],
        }
    )
    output.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(OUT)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
