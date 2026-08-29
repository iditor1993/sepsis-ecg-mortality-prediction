# 内部学术评审报告（IJMI 投稿前审核）

**稿件**：Frozen deep-learning ECG representations do not add confirmed incremental value to 28-day mortality prediction in sepsis: a retrospective development and temporal validation study

**审核日期**：2026-08-29

**审核目的**：在提交 International Journal of Medical Informatics（IJMI）前，检查方法严谨性、证据支持、结果解释、图表一致性、参考文献真实性和报告规范。

**总体结论**：稿件核心结论准确，阴性结果表述克制，方法学透明度和敏感性分析较强；建议在提交前完成作者/伦理/基金等必填项，并提交完整的 TRIPOD+AI 与 PROBAST 检查表。审核结论：**Minor Revision**。

---

## 1. 总体评分

| 维度 | 权重 | 评分（1-10） | 加权 | 关键证据 |
|---|---:|---:|---:|---|
| 原创性 | 20% | 7.0 | 1.40 | 以“冻结 ECG 潜向量 + 强临床表格式路径 + 可得性偏倚 + 功效门槛”组合回答增量价值问题，框架具有清晰的信息学价值。 |
| 方法学严谨性 | 25% | 8.0 | 2.00 | SAP、数据锁、MICE、功效核算、内部/时间外推、11 项敏感性、亚组、DCA、校准和 SHAP 均已报告。 |
| 证据充分性 | 25% | 8.0 | 2.00 | 主要数字均可追溯到 `results/` 及项目 SAP；46/46 篇参考文献通过 Crossref/PhysioNet 核验。 |
| 论证连贯性 | 15% | 7.5 | 1.13 | 引言-方法-结果-讨论逻辑清楚，阴性结论未过度外推；M1+ 时间窗属性需要更显著地强调。 |
| 写作质量 | 15% | 7.8 | 1.17 | 英文表达总体清晰，结构符合国际期刊习惯；占位符和少量表述需在提交前补齐。 |
| **合计** | 100% |  | **7.70** | 对应阈值：Minor Revision。 |

---

## 2. 优点

1. **“强基线 + 增量价值”问题定义明确**。稿件不仅比较 ECG 单独预测能力，还引入 85 列临床表格式路径（M1+），并报告 M3 vs M1、M3 vs M1+、M4+ vs M1+ 三组比较，避免把“ECG 可预测”误读为“ECG 有增价值”。
2. **功效门槛处理诚实**。保守角点 MDD 0.0215 > 0.02，作者明确将主比较降级为估计性分析，并报告 95% CI，未伪称“确认性阴性证据”。
3. **敏感性分析和偏倚控制较完整**。包括窗口、结局定义、完整病例、Track B、房颤/起搏、t0 SOFA、竞争风险、校准方法、去除乳酸、ECG 可得性和编码器重训；结果均与主结论一致。
4. **报告规范意识强**。引用 TRIPOD/TRIPOD+AI/PROBAST、声明生成式 AI、数据可用性和伦理说明。
5. **参考文献真实且可追溯**。46 篇全部有 DOI，45 篇 Crossref 解析成功且标题匹配，1 篇数据集记录通过 PhysioNet 官方页面人工核验。

---

## 3. 主要问题

### Critical

未发现直接导致“不能发表”的科学性问题。

### Major

1. **作者、单位、通讯作者、伦理批复、基金、ChiCTR 编号、机构知识库地址均为占位符**。
   - 影响：这些是 Editorial Manager 和期刊政策要求，不能以占位符提交。
   - 建议：由作者逐项填写，并在 cover letter 和投稿系统同步。

2. **M1+ 的 24 小时窗口需要作为“强比较器”而非“决策时点模型”明确呈现**。
   - 稿件 Methods 和 Limitations 已补充说明，但在 Abstract 中尚未体现；“best-performing prespecified model”会被审稿人追问。
   - 建议：在 Abstract 中增加半句“M1+ was a high-information benchmark, not a point-of-care model”，并说明该比较用于评估信息上限而非部署建议。

3. **缺少完成版 TRIPOD+AI 与 PROBAST 检查表**。
   - 稿件正文声称遵循 TRIPOD/TRIPOD+AI/PROBAST，但投稿包中尚未包含清单；本次任务会补充到 `supplementary/`。
   - 建议：随稿上传完成版清单，并注明缺项（如外部验证、saliency 双人审阅）已在 Limitations 中说明。

4. **M4+/M4+-TB 是探索性模型，且 Table 2 新增行没有 AUC 95% CI**。
   - 现已处理：Table 2 添加注释，正文明确 M4+ 与 M4+-TB 为探索性直接比较，未预设为验证模型。
   - 仍建议：投稿时在 cover letter 或回复中再次说明这些模型用于信息上限评估，不代表临床部署推荐。

5. **没有外部多中心验证**。
   - 稿件已在 Limitations 中说明，但 TRIPOD+AI 检查表中仍会显示不完整。
   - 建议：不要修改结论，明确写“外部验证是后续工作”，避免把内部时间外推描述为外部验证。

### Minor

1. 参考文献 37、41、39 的 Crossref `issued` 年份比期刊卷期年份早 1 年；期刊引用通常按正式卷期年份，当前引用可保留，但建议在检查表中注明“online-first vs issue year”。
2. 亚组分析的七个交互检验未做多重比较校正；正因作者标注为探索性，这不构成阻断，但应在 Methods 中明确“未进行 FDR 校正”。
3. `Table 4` 中 S3 为“不适用”，建议保留该行以证明所有 S1-S11 均已处理；当前稿已保留。
4. Figure 7/8 映射此前有误，现已修正：Figure 7 为 SHAP，Figure 8 为潜向量-临床相关谱。
5. `Table 6`（ECG 可得性）此前在正文没有引用，现已补充。
6. 已补充 Supplementary Table S1，覆盖 TRIPOD+AI 项 20c（开发/时间队列特征比较）。
7. 稿件有少量中文/英文混杂的表格源文件；当前 `Table6` 已统一为英文，最终 Word 版应复查所有表格。

---

## 4. 结果一致性抽查

| 稿件声明 | 来源文件 | 状态 |
|---|---|---|
| 16,499 例；3,002 例（18.2%）死亡 | `results/baseline_table.csv`、`data/splits.csv` + `outcomes.parquet` | 通过 |
| test 425/2,217；temporal 380/1,719 | 上述数据计算 | 通过 |
| M1+ test AUC 0.854（0.835-0.873） | `results/test_metrics.csv` | 通过 |
| M3 vs M1 ΔAUC −0.0014（−0.0051至+0.0023） | `results/h1_delong.csv` | 通过 |
| M3 vs M1+ ΔAUC −0.0587（−0.0761至−0.0427） | `results/h1_delong.csv` | 通过 |
| 连续 NRI −0.137；类别 NRI +0.001；IDI −0.002 | `results/h2_nri_idi.csv` | 通过 |
| 时间外推 M3 0.777；M1+ 0.842 | `results/h3_temporal.csv` | 通过 |
| ECG 可得性 48.0% vs 33.8%；死亡率 18.2% vs 23.2% | `results/a1b_availability.csv` | 通过 |
| M4+ vs M1+ ΔAUC +0.0032（−0.0002至+0.0065） | `results/m4plus_results.csv` | 通过 |
| M4+-TB vs M1+ ΔAUC −0.0146（−0.0256至−0.0031） | `results/m4plus_trackb_results.csv` | 通过 |
| SHAP z2 第 10/11 位，mean |SHAP| 0.042/0.041 | `results/shap_m3_importance.csv`、`results/shap_m4_importance.csv` | 通过 |
| 最大 |Spearman ρ| ≈ 0.15 | `results/e1_correlation.csv` | 通过 |

抽查未发现稿件数字与源数据之间的实质性不一致。

---

## 5. 参考文献核验

- 清单共 46 篇。
- `references/final_references.json` 与正文引用：0 个孤儿引用，0 个悬空引用。
- `tools/verify_references.py`：46/46 通过；45 篇 Crossref DOI 解析成功且标题匹配，1 篇 PhysioNet 数据集记录人工核验。
- 未发现虚构 DOI、作者或期刊。
- 2026 年发表的 4 篇（[20]、[22]、[23]、[26]）已通过出版商/PubMed/期刊页面额外抽查，均为真实存在。

---

## 6. 提交前应完成事项

1. 填写作者、单位、通讯作者、ORCID、基金、伦理批复、ChiCTR、机构知识库链接。
2. 将 `supplementary/TRIPOD+AI_checklist.md`、`PROBAST_assessment.md`、SAP、deviation log、saliency template、reproducibility manifest 加入投稿系统。
3. 在 Abstract 中明确 M1+ 是强信息基准而非点对点部署模型。
4. 为 Table 2 的 M4+/M4+-TB 加探索性注释。
5. 生成最终 Word 版并通过 PDF 渲染检查版面。
6. 上传前再次运行 `validate_manuscript.py` 和 `verify_references.py`。

---

**审核人置信度**：方法学部分高；临床实施与外部验证部分中等。本报告为投稿前内部质量审查，不替代期刊正式审稿或伦理/统计专家复核。
