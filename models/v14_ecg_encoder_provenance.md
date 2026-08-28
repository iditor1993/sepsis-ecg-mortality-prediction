# V14 ECG 编码器溯源（provenance）

## 文件

`models/v14_ecg_encoder.keras`（67 KB）

- **SHA-256**：`1596f4e3e02ffd3d1e508ed745c3d2502fefc4d479c75b3f274adc12bd7c3ee9`
- **复制日期**：2026-08-28（与源文件逐字节一致，已校验哈希）
- **来源路径**（V14 项目）：`D:/BaiduSyncdisk/work/part-time job/sepsis_associated_research/ECG_sepsis/V14/data/ECG_feature/v14_ecg_encoder.keras`

## 模型说明

- **类型**：一维 CNN 自编码器（autoencoder）的编码器部分，**无监督预训练**
- **训练数据**：V14 项目队列的 MIMIC-IV-ECG Lead II 信号约 4.1 万条
  （V14 项目 `data/ECG_feature/v14_lead2_signals.npy`；训练脚本
  `V14/scripts/v14_extract_ecg.py`），不含任何结局标签
- **输入**：Lead II 10 s 重采样至 250 Hz 共 2500 点，z 标准化
- **结构**：Input(2500,1) → Conv1D(16,7,s2) → Conv1D(32,5,s2) →
  Conv1D(64,3,s5) → GlobalAveragePooling1D → Dense(16) → **z1–z16**
- **本研究用途**：SAP 5.1 Track A 主特征（冻结迁移，权重不在本队列重训）；
  M5 探索性端到端微调以其初始化（解冻末两层）

## 方法学注记

该编码器训练所用 ECG 与本研究队列同出 MIMIC-IV-ECG，可能存在样本重叠；
因其训练无监督、无标签，属自监督预训练范畴（论文方法学部分应注明
编码器来源与本条目）。
