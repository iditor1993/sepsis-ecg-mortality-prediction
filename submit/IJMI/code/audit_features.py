"""audit_features.py — 全量特征与数据审计（只读）。

排查 stay_id_1 同类问题及一般性数据规格风险：
  1) 各 data/*.parquet：列名中的 id 样列（*_id / *_idN / *_1 后缀）、
     重复含义列（与已知 id 列全等）、全 NaN/常量列、stay_id 唯一性
  2) data/models/*.joblib：各模型实际特征清单是否含 id 样列
  3) 建模特征列与各 parquet 的交叉核对
输出 results/feature_audit.csv 并打印问题清单。
"""

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
ID_PAT = re.compile(r"(^|_)(subject_id|stay_id|hadm_id|study_id|microevent_id|"
                    r"labevent_id|transfer_id|cart_id)$|_id_\d+$|_\d+$")


def main() -> None:
    datadir = REPO_ROOT / "data"
    issues, rows = [], []

    # ---- 1) parquet 审计 ----
    for f in sorted(datadir.glob("*.parquet")):
        df = pd.read_parquet(f)
        cols = list(df.columns)
        idish = [c for c in cols if ID_PAT.search(c)]
        dupes = [c for c in set(cols) if cols.count(c) > 1]
        all_nan = [c for c in cols if df[c].isna().all()]
        const = [c for c in cols
                 if df[c].nunique(dropna=True) <= 1 and not df[c].isna().all()]
        uniq_stay = (int(df["stay_id"].is_unique) if "stay_id" in cols else None)
        # 与已知 id 列全等的列
        twins = []
        for ref in ("subject_id", "stay_id", "hadm_id", "study_id"):
            if ref in cols:
                for c in cols:
                    if c != ref and df[c].dtype == df[ref].dtype:
                        try:
                            if len(df) and bool((df[c] == df[ref]).all()):
                                twins.append(f"{c}=={ref}")
                        except Exception:
                            pass
        rows.append({"file": f.name, "shape": f"{df.shape[0]}x{df.shape[1]}",
                     "id_like_cols": ";".join(idish), "dup_cols": ";".join(dupes),
                     "all_nan": ";".join(all_nan), "const_cols": ";".join(const),
                     "stay_unique": uniq_stay, "id_twins": ";".join(twins)})
        # id 样列在“特征型”文件中是否可疑（subject/stay/hadm 为正常键）
        for c in idish:
            if c not in ("subject_id", "stay_id", "hadm_id", "study_id"):
                issues.append(f"{f.name}: 可疑 id 样列 {c}")
        for t in twins:
            issues.append(f"{f.name}: 与 id 全等的列 {t}")
        for c in dupes:
            issues.append(f"{f.name}: 重复列名 {c}")

    # ---- 2) 模型特征清单审计 ----
    model_rows = []
    for f in sorted((datadir / "models").glob("*.joblib")):
        pack = joblib.load(f)
        feats = pack.get("features") if isinstance(pack, dict) else None
        if feats is None and isinstance(pack, dict) and "models" in pack:
            feats = pack.get("features")
        if feats:
            bad = [c for c in feats if ID_PAT.search(c)]
            model_rows.append({"model_file": f.name, "n_features": len(feats),
                               "id_like_in_features": ";".join(bad)})
            for c in bad:
                issues.append(f"models/{f.name}: 特征含 id 样列 {c}")
    # platt.joblib / thresholds.json 非特征文件，跳过

    pd.DataFrame(rows).to_csv(REPO_ROOT / "results" / "feature_audit.csv",
                              index=False, encoding="utf-8-sig")
    pd.DataFrame(model_rows).to_csv(REPO_ROOT / "results" / "feature_audit_models.csv",
                                    index=False, encoding="utf-8-sig")

    print("=== 数据文件审计 ===")
    for r in rows:
        flag = ""
        if r["id_twins"] or r["dup_cols"]:
            flag = "  <== 问题"
        print(f"{r['file']:<38} {r['shape']:>12}  stay唯一={r['stay_unique']} "
              f"id样列=[{r['id_like_cols']}] 全NaN=[{r['all_nan']}] 常量=[{r['const_cols']}]{flag}")
    print("\n=== 模型特征清单 ===")
    for r in model_rows:
        print(f"{r['model_file']:<28} n_feat={r['n_features']:>4} "
              f"id样特征=[{r['id_like_in_features']}]")

    print("\n" + "=" * 60)
    if issues:
        print(f"发现 {len(issues)} 项问题:")
        for i in issues:
            print(" -", i)
    else:
        print("未发现 stay_id_1 同类问题或 id 样特征混入。")
    print("输出 -> results/feature_audit.csv, results/feature_audit_models.csv")


if __name__ == "__main__":
    main()
