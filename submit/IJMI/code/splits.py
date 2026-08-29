"""splits.py — 数据集划分（SAP 第七章）。

开发队列（dev_2008_2016）患者级随机划分：训练 70% / 调优 15% / 内部测试 15%，
随机种子 20260823；时间外推与 COVID 队列全量仅用于验证，不参与训练调参。
队列已为每人一行（首次发作），患者级划分等价于行级划分。
划分不使用结局标签，在标签解盲前完成并冻结。
输出 data/splits.csv（数据锁定时记录 SHA-256 哈希）。
"""

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED = 20260823  # SAP 统一随机种子，勿改


def main() -> None:
    datadir = REPO_ROOT / "data"
    cohort = pd.read_parquet(datadir / "cohort_ecg.parquet")

    splits = cohort[["subject_id", "stay_id", "cohort_period"]].copy()
    splits["subset"] = splits["cohort_period"].map(
        {"dev_2008_2016": "", "temporal_2017_2019": "temporal",
         "covid_2020_2022": "covid"}
    )

    dev_idx = splits.index[splits["cohort_period"] == "dev_2008_2016"].to_numpy()
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(dev_idx))
    n_train = int(round(len(dev_idx) * 0.70))
    n_tune = int(round(len(dev_idx) * 0.15))
    labels = np.array(["train"] * n_train + ["tune"] * n_tune
                      + ["test"] * (len(dev_idx) - n_train - n_tune))
    splits.loc[dev_idx[perm], "subset"] = labels

    assert not (splits["subset"] == "").any()
    splits[["subject_id", "stay_id", "subset"]].to_csv(datadir / "splits.csv", index=False)

    print(f"划分完成（种子 {SEED}）:")
    print(splits["subset"].value_counts().to_string())
    print(f"\n输出: {datadir / 'splits.csv'}")


if __name__ == "__main__":
    main()
