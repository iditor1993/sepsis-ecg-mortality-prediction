"""train_m0_m5.py — M0-M5 模型训练（SAP 8.1 / 8.2 节）。

M0 SOFA 单独 LR；M1 评分+乳酸+协变量 LR；M2 ECG 潜向量 LR；
M3 ECG+评分+乳酸+协变量 LASSO-LR（主模型，lambda.1se）；
M4 XGBoost；M5 端到端微调（探索）。
全部模型经 Platt 校准（调优集拟合）；约登指数阈值在调优集确定并冻结。

TODO: 待实现。
"""
