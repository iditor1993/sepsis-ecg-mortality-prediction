# DATA_LOCK — 数据锁定记录（SAP 1.3 / 9.10 节）

**数据锁定时点：2026-08-27 17:55:54**（特征矩阵生成并记录 SHA-256 哈希）

锁定前任何模型不得接触测试集结局标签；锁定后 SAP 实质性修改
仅允许以偏离形式记录（SAP 第十三章）。

| 文件 | SHA-256 | 锁定时间 | 操作人 |
|---|---|---|---|
| features_dev.parquet | `5eef468f7bdc6ddd3cb5d5b21fea8825d4febb4fb032747a1f4ba5e479a06da3` | 2026-08-27 17:55:54 | iditor1993 |
| features_temporal.parquet | `1ce1d9c889c1283c5ffb29106ceebd1331f1de3d6284fbca8c62d701eec24268` | 2026-08-27 17:55:54 | iditor1993 |
| splits.csv | `b5bf9cf90ad82694f9f851a67a8a93e4127dc93a4dbe9856678dd58935ff22cf` | 2026-08-27 17:55:54 | iditor1993 |

## 锁定前已完成的质量门槛

- [x] Sepsis-3 队列判定与排除流程（cohort_flow.csv）
- [x] ECG 链接与质控（qc_config.yaml，PROPOSED 阈值冻结为本版本）
- [x] 数据集划分（种子 20260823，标签解盲前冻结）
- [ ] ΔAUC 功效核算（power_delong_sim.py）——测试集解盲前执行