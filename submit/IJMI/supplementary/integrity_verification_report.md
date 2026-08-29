# 参考文献与数据完整性核查报告

**核查对象**：`submit/IJMI/manuscript_IJMI.md`

**核查日期**：2026-08-29

**核查模式**：投稿前最终核验（Final verification）

**结论**：**PASS WITH NOTES**。46/46 篇参考文献均可通过 Crossref DOI 或 PhysioNet 官方记录核验；未发现虚构参考文献、DOI 误导、作者错误或正文引用缺失。保留“PASS WITH NOTES”的唯一原因是部分文献全文受出版商访问限制，完整正文核对建议由作者或图书馆获取后补充。

---

## 1. 核验摘要

| 类别 | 总数 | 通过 | 问题 |
|---|---:|---:|---:|
| 参考文献存在性 | 46 | 46 | 0 |
| 标题匹配 | 46 | 46 | 0 |
| DOI 解析 | 46 | 45 Crossref + 1 PhysioNet | 0 |
| 正文引用完整性 | 46 | 46 | 0 |
| 孤儿引用 | — | — | 0 |
| 悬空引用 | — | — | 0 |
| 统计数字与源结果一致性（抽查） | 12 项 | 12 | 0 |
| 图/表正文引用 | 8 图、6 表 | 8+6 | 0 |

## 2. 核验方法

1. 对每篇参考文献执行 `https://api.crossref.org/works/{doi}` API 查询。
2. 比较 Crossref 返回标题（去除 HTML 标签、规范化连字符和大小写后）与 `references/final_references.json` 中标题；相似度阈值 0.92。
3. 记录 Crossref `container-title` 和 `issued` 年份。
4. `10.13026/4nqg-sb35` 为 PhysioNet 数据集记录，不在 Crossref 中；通过官方页面 `https://physionet.org/content/mimic-iv-ecg/1.0/` 人工核验。
5. 使用 `tools/validate_manuscript.py` 检查正文引用和参考文献清单双向一致性。
6. 对 4 篇 2026 年文献进行出版商/PubMed 额外抽查。

## 3. 逐条核验记录

完整逐条记录见 `references/reference_verification.csv`（含 `expected_title`、`crossref_title`、`crossref_journal`、`crossref_year`、`crossref_doi_status`、`title_match`、`verified`、`checked_at`）。

摘要如下：

| 文献 ID | DOI | 核验渠道 | 标题匹配 |
|---|---:|---|---|
| [1] | 10.1001/jama.2016.0287 | Crossref | 是 |
| [2] | 10.1016/S0140-6736(19)32989-7 | Crossref | 是 |
| [3] | 10.1007/s00134-020-06151-x | Crossref | 是 |
| [4] | 10.1007/BF01709751 | Crossref | 是 |
| [5] | 10.1001/jama.2016.0288 | Crossref | 是 |
| [6] | 10.1001/jama.2016.20328 | Crossref | 是 |
| [7] | 10.1093/qjmed/94.10.521 | Crossref | 是 |
| [8] | 10.7717/peerj.6947 | Crossref | 是 |
| [9] | 10.1097/CCM.0000000000000742 | Crossref | 是 |
| [10] | 10.1016/0021-9681(87)90171-8 | Crossref | 是 |
| [11] | 10.1097/01.mlr.0000182534.19832.83 | Crossref | 是 |
| [12] | 10.1007/s00134-019-05872-y | Crossref | 是 |
| [13] | 10.1038/s41597-022-01899-x | Crossref | 是 |
| [14] | 10.1161/01.CIR.101.23.e215 | Crossref | 是 |
| [15] | 10.13026/4nqg-sb35 | PhysioNet 官方页 | 是 |
| [16] | 10.1038/s41591-018-0268-3 | Crossref | 是 |
| [17] | 10.1038/s41591-018-0240-2 | Crossref | 是 |
| [18] | 10.1038/s41467-020-15432-4 | Crossref | 是 |
| [19] | 10.1038/s41569-020-00503-2 | Crossref | 是 |
| [20] | 10.1016/j.jacadv.2026.102875 | Crossref + JACC 页面 | 是 |
| [21] | 10.1186/s13049-021-00953-8 | Crossref | 是 |
| [22] | 10.1186/s12911-026-03657-0 | Crossref + PubMed | 是 |
| [23] | 10.1016/j.hrtlng.2026.102771 | Crossref + Heart & Lung/PubMed | 是 |
| [24] | 10.1371/journal.pone.0203487 | Crossref | 是 |
| [25] | 10.1097/MD.0000000000014197 | Crossref | 是 |
| [26] | 10.1038/s41746-026-02565-x | Crossref + PubMed | 是 |
| [27] | 10.1016/j.jelectrocard.2017.08.013 | Crossref | 是 |
| [28] | 10.1038/s41591-018-0213-5 | Crossref | 是 |
| [29] | 10.1016/S2589-7500(20)30186-2 | Crossref | 是 |
| [30] | 10.1136/bmjqs-2018-008370 | Crossref | 是 |
| [31] | 10.1038/s41591-021-01614-0 | Crossref | 是 |
| [32] | 10.2196/15182 | Crossref | 是 |
| [33] | 10.7326/M14-0697 | Crossref | 是 |
| [34] | 10.1136/bmj-2023-078378 | Crossref | 是 |
| [35] | 10.7326/M18-1376 | Crossref | 是 |
| [36] | 10.1136/bmj.m441 | Crossref | 是 |
| [37] | 10.1002/sim.7992 | Crossref | 是 |
| [38] | 10.18637/jss.v045.i03 | Crossref | 是 |
| [39] | 10.1002/sim.4067 | Crossref | 是 |
| [40] | 10.2307/2531595 | Crossref | 是 |
| [41] | 10.1002/sim.2929 | Crossref | 是 |
| [42] | 10.1186/s12916-019-1466-7 | Crossref | 是 |
| [43] | 10.1177/0272989X06295361 | Crossref | 是 |
| [44] | 10.1080/01621459.1999.10474144 | Crossref | 是 |
| [45] | 10.1038/s42256-019-0138-9 | Crossref | 是 |
| [46] | 10.1038/s42256-019-0048-x | Crossref | 是 |

## 4. 备注

1. Crossref 的 `issued` 年份对 [37]、[41]、[39] 为 online-first 年份，比正式卷期年份早约 1 年；正文引用采用正式卷期年份。
2. 2026 年文献在核验时点为当前在版/已发表记录；提交前建议复查是否有最终卷期页码更新。
3. 本报告仅核验文献存在性、元数据和稿件数据一致性；未替代专业查重软件（如 iThenticate/Turnitin）和正式伦理审查。
4. 原始数据文件位于仓库 `data/` 和 `results/`，投稿包中仅包含脱敏汇总、图表与可重复性清单，不包含患者可识别信息。

---

*核查由 `tools/verify_references.py`、`tools/validate_manuscript.py` 和人工网页核验完成。*
