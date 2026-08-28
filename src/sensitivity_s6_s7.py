"""sensitivity_s6_s7.py — 敏感性分析 S6/S7（SAP 9.9，W7）。

  S6  排除房颤/起搏心律 ECG（报告文本标记）后重训 M1/M3，测试集 ΔAUC
  S7  SOFA 总分改用 t0 时刻值（mimiciv_derived.sofa 逐时表 sofa_24hours，
      取覆盖 t0 的逐时行；无覆盖行时取距 t0 最近行）重训 M1/M3

训练协议与主分析一致（M1 普通 LR、M3 LASSO lambda.1se，均 20 套 MICE 插补
预测均值）。输出 results/sensitivity_s6_s7.csv 与 data/sofa_t0.parquet。
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from sensitivity_analyses import (COV_COLS, SCORE_COLS, Z_COLS, boot_delta,
                                  build, lasso_fit)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = "E:/clinical_research/MIMIC_IV_3.1/mimic_iv_3_1.duckdb"
SEED = 20260823
M = 20


def extract_sofa_t0(datadir: Path) -> pd.DataFrame:
    out_path = datadir / "sofa_t0.parquet"
    if out_path.exists():
        return pd.read_parquet(out_path)
    cohort = pd.read_parquet(datadir / "cohort_ecg.parquet")
    con = duckdb.connect(DEFAULT_DB, read_only=True)
    try:
        con.register("cohort", cohort)
        df = con.sql("""
            SELECT stay_id, sofa_total_t0 FROM (
                SELECT c.stay_id, s.sofa_24hours AS sofa_total_t0,
                       ROW_NUMBER() OVER (
                           PARTITION BY c.stay_id
                           ORDER BY CASE WHEN s.starttime <= c.t0 AND c.t0 < s.endtime
                                         THEN 0 ELSE 1 END,
                                    abs(epoch(s.starttime - c.t0))
                       ) AS rn
                FROM cohort c
                JOIN mimiciv_derived.sofa s ON s.stay_id = c.stay_id
                WHERE s.endtime >= c.t0 - INTERVAL '24' HOUR
                  AND s.starttime <= c.t0 + INTERVAL '24' HOUR
            ) t WHERE rn = 1
        """).df()
    finally:
        con.close()
    df.to_parquet(out_path, index=False)
    return df


def lr_mice(df, feats, tr, tu_te, ytr, mice_cols):
    """普通 LR，MICE 20 套均值预测。"""
    p = np.zeros(tu_te.sum())
    for k in range(M):
        cols = [c if c != "lactate" else mice_cols[k] for c in feats]
        xtr = df.loc[tr, cols].copy(); xtr.columns = feats
        xev = df.loc[tu_te, cols].copy(); xev.columns = feats
        sc = StandardScaler().fit(xtr)
        m = LogisticRegression(max_iter=2000).fit(sc.transform(xtr), ytr)
        p += m.predict_proba(sc.transform(xev))[:, 1]
    return p / M


def lasso_mice(df, feats, tr, ev, ytr, mice_cols):
    p = np.zeros(ev.sum())
    for k in range(M):
        cols = [c if c != "lactate" else mice_cols[k] for c in feats]
        xtr = df.loc[tr, cols].copy(); xtr.columns = feats
        xev = df.loc[ev, cols].copy(); xev.columns = feats
        sc = StandardScaler().fit(xtr)
        m = lasso_fit(sc.transform(xtr), ytr)
        p += m.predict_proba(sc.transform(xev))[:, 1]
    return p / M


def main() -> None:
    datadir = REPO_ROOT / "data"
    results_dir = REPO_ROOT / "results"
    rng = np.random.default_rng(SEED)
    df = build(datadir)
    site_cols = sorted(c for c in df.columns if c.startswith("site_"))
    cov = COV_COLS + site_cols
    mice_cols = [f"lactate_m{k:02d}" for k in range(1, M + 1)]
    tr_all, te_all = df["subset"] == "train", df["subset"] == "test"

    rows = []

    # ---- S6：排除房颤/起搏 ----
    rhythm = pd.read_parquet(datadir / "ecg_rhythm_flags.parquet")
    df = df.merge(rhythm[["stay_id", "af", "paced"]], on="stay_id", how="left")
    df[["af", "paced"]] = df[["af", "paced"]].fillna(0).astype(int)
    keep = (df["af"] == 0) & (df["paced"] == 0)
    tr, te = tr_all & keep, te_all & keep
    ytr, yte = (df.loc[m, "death_28d"].astype(int).to_numpy() for m in (tr, te))
    f_m1 = SCORE_COLS + ["lactate"] + cov
    f_m3 = f_m1 + Z_COLS
    p1 = lr_mice(df, f_m1, tr, te, ytr, mice_cols)
    p3 = lasso_mice(df, f_m3, tr, te, ytr, mice_cols)
    d = boot_delta(yte, p3, p1, rng)
    rows.append({"id": "S6", "内容": "排除房颤/起搏心律 ECG",
                 "n_test": int(te.sum()), "test_auc_m3": roc_auc_score(yte, p3),
                 "test_auc_m1": roc_auc_score(yte, p1),
                 "delta_auc": d[0], "ci_lo": d[1], "ci_hi": d[2]})
    print(f"[S6] 排除 AF/起搏（测试 n={te.sum():,}）：M3 {roc_auc_score(yte, p3):.4f} vs "
          f"M1 {roc_auc_score(yte, p1):.4f}，ΔAUC {d[0]:+.4f} ({d[1]:+.4f}~{d[2]:+.4f})",
          flush=True)

    # ---- S7：SOFA 改 t0 时刻值 ----
    sofa_t0 = extract_sofa_t0(datadir)
    df = df.merge(sofa_t0, on="stay_id", how="left")
    df["sofa_total_t0"] = df["sofa_total_t0"].astype(float)
    n_imp_t0 = int(df["sofa_total_t0"].isna().sum())
    if n_imp_t0:  # 无逐时行覆盖者回退为主分析 SOFA（sepsis3 表起始值）
        df.loc[df["sofa_total_t0"].isna(), "sofa_total_t0"] = df["sofa_score"]
    f_m1_s7 = ["sofa_total_t0", "qsofa", "news", "mews", "lactate"] + cov
    f_m3_s7 = f_m1_s7 + Z_COLS
    tr, te = tr_all, te_all
    ytr, yte = (df.loc[m, "death_28d"].astype(int).to_numpy() for m in (tr, te))
    p1 = lr_mice(df, f_m1_s7, tr, te, ytr, mice_cols)
    p3 = lasso_mice(df, f_m3_s7, tr, te, ytr, mice_cols)
    d = boot_delta(yte, p3, p1, rng)
    rows.append({"id": "S7", "内容": f"SOFA 改 t0 时刻值（{n_imp_t0} 例回退）",
                 "n_test": int(te.sum()), "test_auc_m3": roc_auc_score(yte, p3),
                 "test_auc_m1": roc_auc_score(yte, p1),
                 "delta_auc": d[0], "ci_lo": d[1], "ci_hi": d[2]})
    print(f"[S7] SOFA t0 值：M3 {roc_auc_score(yte, p3):.4f} vs "
          f"M1 {roc_auc_score(yte, p1):.4f}，ΔAUC {d[0]:+.4f} ({d[1]:+.4f}~{d[2]:+.4f})",
          flush=True)

    pd.DataFrame(rows).to_csv(results_dir / "sensitivity_s6_s7.csv", index=False)
    print("\n输出 -> results/sensitivity_s6_s7.csv")


if __name__ == "__main__":
    main()
