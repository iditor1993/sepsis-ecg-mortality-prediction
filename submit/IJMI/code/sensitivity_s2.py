"""sensitivity_s2.py — S2：结局改为入院时刻起 28 天死亡（SAP 9.9）。

主要结局自 t0 起算改为自 admittime 起算（其余不变：MICE、LASSO 协议、
测试集评估 ΔAUC M3 vs M1）。输出 results/sensitivity_s2.csv。
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED = 20260823
M = 20


def main() -> None:
    datadir = REPO_ROOT / "data"
    results_dir = REPO_ROOT / "results"
    rng = np.random.default_rng(SEED)
    from sensitivity_s6_s7 import lasso_mice, lr_mice
    from sensitivity_analyses import COV_COLS, SCORE_COLS, Z_COLS, boot_delta, build

    df = build(datadir)
    site_cols = sorted(c for c in df.columns if c.startswith("site_"))
    cov = COV_COLS + site_cols

    days = (df["dod"] - df["admittime"].dt.normalize()).dt.days
    df["death_28d_admit"] = ((days >= 0) & (days <= 28)).astype(int)
    n_ev = int(df["death_28d_admit"].sum())
    print(f"[S2] 入院起算 28 天死亡事件: {n_ev:,}（t0 起算口径 "
          f"{int(df['death_28d'].sum()):,}）")

    tr, te = df["subset"] == "train", df["subset"] == "test"
    ytr = df.loc[tr, "death_28d_admit"].to_numpy()
    yte = df.loc[te, "death_28d_admit"].to_numpy()
    mice_cols = [f"lactate_m{k:02d}" for k in range(1, M + 1)]

    # lr_mice/lasso_mice 内部固定读 death_28d 列，故此处直接传入显式标签版本
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def run(feats, lasso):
        p = np.zeros(te.sum())
        for k in range(M):
            cols = [c if c != "lactate" else mice_cols[k] for c in feats]
            xtr = df.loc[tr, cols].copy(); xtr.columns = feats
            xte = df.loc[te, cols].copy(); xte.columns = feats
            sc = StandardScaler().fit(xtr)
            if lasso:
                from sensitivity_analyses import lasso_fit
                m = lasso_fit(sc.transform(xtr), ytr)
            else:
                m = LogisticRegression(max_iter=2000).fit(sc.transform(xtr), ytr)
            p += m.predict_proba(sc.transform(xte))[:, 1]
        return p / M

    f_m1 = SCORE_COLS + ["lactate"] + cov
    f_m3 = f_m1 + Z_COLS
    p1 = run(f_m1, lasso=False)
    p3 = run(f_m3, lasso=True)
    d, lo, hi = boot_delta(yte, p3, p1, rng)
    res = pd.DataFrame([{
        "id": "S2", "内容": "结局改为入院起 28 天死亡",
        "n_test": int(te.sum()), "事件率_test": float(yte.mean()),
        "test_auc_m3": roc_auc_score(yte, p3), "test_auc_m1": roc_auc_score(yte, p1),
        "delta_auc": d, "ci_lo": lo, "ci_hi": hi}])
    res.to_csv(results_dir / "sensitivity_s2.csv", index=False)
    print(f"[S2] M3 {roc_auc_score(yte, p3):.4f} vs M1 {roc_auc_score(yte, p1):.4f}，"
          f"ΔAUC {d:+.4f} ({lo:+.4f}~{hi:+.4f})")
    print("输出 -> results/sensitivity_s2.csv")


if __name__ == "__main__":
    main()
