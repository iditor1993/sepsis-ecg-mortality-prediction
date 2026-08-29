# IJMI 投稿包（International Journal of Medical Informatics）

**项目**：脓毒症 ECG 深度学习预测模型增量价值研究（MIMIC-IV v3.1 + MIMIC-IV-ECG v1.0）

**投稿目标期刊**：International Journal of Medical Informatics (Elsevier)

**投稿包生成日期**：2026-08-29

**核心结论**：冻结深度学习 ECG 潜向量在常规临床评分和强 85 列临床路径之上未提供已证实的 28 天死亡预测增量价值；主比较因预设功效门槛未通过而按估计性分析表述。

---

## 1. 目录结构

| 路径 | 内容 |
|---|---|
| `manuscript_IJMI.md` | 可编辑主稿（Markdown） |
| `manuscript_IJMI.docx` | 主稿 Word 版 |
| `README.md` | 投稿包说明 |
| `figures/` | 8 张图和 graphical abstract |
| `tables/` | 6 张可编辑表格 CSV |
| `references/` | 46 篇参考文献 JSON、核对 CSV/JSON |
| `supplementary/` | TRIPOD+AI、PROBAST、SAP、偏差日志、审计报告、可重复性清单等 |
| `documents/` | Cover letter、投稿检查表、AI 声明、数据可用性、ICMJE 模板 |
| `code/` | 可公开的源码、SQL、环境和模型来源说明 |
| `tools/` | 参考文献核验、稿件校验、表格/图生成、Word 生成脚本 |

## 2. 主稿

- 标题：Frozen deep-learning ECG representations do not add confirmed incremental value to 28-day mortality prediction in sepsis: a retrospective development and temporal validation study
- 主文约 3,284 词（不含摘要、图表和参考文献）
- 摘要 358 词
- 46 篇参考文献，按正文首次出现顺序编号，全部通过 Crossref/PhysioNet 核验
- 报告遵循 TRIPOD、TRIPOD+AI 和 PROBAST

**提交前必须由作者填写：**

- 作者列表、单位、通讯作者、ORCID
- 伦理机构批复/豁免号
- 基金信息和基金号
- ChiCTR 注册号
- 代码仓库 URL
- `Acknowledgments`
- CRediT 作者贡献

## 3. 文献核验

`tools/verify_references.py` 执行以下流程：

1. 对每篇 DOI 调用 Crossref API；
2. 比较 Crossref 标题与稿件标题；
3. 记录 Crossref 期刊、年份和 DOI 状态；
4. 对 MIMIC-IV-ECG 数据集记录（`10.13026/4nqg-sb35`）使用 PhysioNet 官方页面人工核验；
5. 输出 `references/reference_verification.csv` 和 `.json`。

当前结果：**46/46 篇验证通过**，0 孤儿引用、0 悬空引用。

## 4. 图表

- `Fig1_cohort_flow.png`：队列流程图
- `Fig2_analysis_pipeline.png`：分析流程
- `Fig3_ROC.png`：测试集 ROC
- `Fig4_calibration.png`：校准曲线
- `Fig5_DCA.png`：决策曲线
- `Fig6_subgroups.png`：亚组交互
- `Fig7_SHAP.png`：SHAP 重要性
- `Fig8_latent_spectrum.png`：ECG 潜向量与临床指标相关热图
- `Graphical_abstract.png`：图形摘要

表格 CSV 可直接用 Excel/WPS 打开；Word 稿中已嵌入主要表格。

## 5. 补充材料

| 文件 | 说明 |
|---|---|
| `TRIPOD+AI_checklist.md` | 27 项主清单 + 13 项摘要清单 |
| `PROBAST_assessment.md` | 偏倚和适用性评估 |
| `Statistical_Analysis_Plan_V1.3_original.md` | 项目 SAP V1.3 |
| `deviation_log.md` | SAP 偏差日志（D-001~D-003） |
| `data_lock.md` | 数据锁和特征矩阵 SHA-256 |
| `saliency_review_template.csv` | 医生 saliency 审阅模板（待完成） |
| `Table_S1_development_vs_temporal.md/.csv` | 开发队列与时间队列关键特征比较（TRIPOD+AI 项 20c） |
| `internal_review_report.md` | 投稿前内部学术评审 |
| `integrity_verification_report.md` | 参考文献/数字完整性核查报告 |
| `analysis_reproducibility_manifest.md` | 可重复性清单（见下方） |

## 6. 投稿文档

- `documents/IJMI_cover_letter.md/.docx`
- `documents/IJMI_submission_checklist.md/.docx`
- `documents/IJMI_AI_declaration.md/.docx`
- `documents/IJMI_data_availability.md/.docx`
- `documents/ICMJE_disclosure_placeholders.md`

所有文档中的占位符必须在正式投稿前替换。

## 7. 可重复性

`code/` 包含项目源码（`src/*.py`、`sql/*.sql`）、`environment.yml`、`requirements_lock.txt` 和 V14 编码器来源说明。原始 MIMIC-IV/ECG 数据不在投稿包内；数据锁哈希见 `supplementary/data_lock.md`。

**复现顺序**：

`extract_cohort.py` → `ecg_link.py` → `splits.py` → `features_trackA/B.py` → `extract_covariates.py` / `extract_clinical_features.py` → `build_feature_matrix.py` → `power_delong_sim.py` → `impute_mice.py` → `train_m0_m5.py` → `evaluate.py` → `sensitivity_*.py` / `a1b_availability.py` / `subgroup_analysis.py` / `e1_correlation.py` / `interpretability.py` → `baseline_table.py` / `make_figures.py` → `trackab_comparison.py`

## 8. 校验命令

在仓库根目录运行：

```powershell
python submit/IJMI/tools/verify_references.py
python submit/IJMI/tools/validate_manuscript.py
```

## 9. 已知说明

1. 当前 Word 文件已生成，但本环境无法稳定调用 LibreOffice/Poppler 或 Word COM 完成自动 PDF 渲染；正式投稿前请用 Word/WPS 打开并导出 PDF 检查版面。
2. 作者、伦理、基金、ChiCTR 和仓库 URL 是占位符，严禁在未填写的情况下上传。
3. 外部多中心验证和 cardiologist saliency 双人审阅尚未完成，已作为 Limitations/后续工作明确说明。
4. M1+ 是 24 小时窗口上的高信息临床基准，不是精确决策时点模型；这是刻意设计，请勿将 M1+ 与可部署性结论混淆。
5. 主比较不是确认性检验，而是估计性分析；引用时勿表述为“已证明无价值”。

---

*This package was prepared by the authors with Codex as a writing/organising assistant. All numerical results and references were checked against project outputs before packaging.*
