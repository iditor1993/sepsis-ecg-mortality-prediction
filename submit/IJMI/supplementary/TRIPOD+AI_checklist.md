# TRIPOD+AI Checklist

**Study**：Frozen deep-learning ECG representations do not add confirmed incremental value to 28-day mortality prediction in sepsis: a retrospective development and temporal validation study

**Checklist version**：TRIPOD+AI statement, 27-item checklist (Collins et al., BMJ 2024; doi:10.1136/bmj-2023-078378)

**Development/Evaluation**：Development + temporal evaluation (type 2b)

**Note**：This is a completed reporting checklist for submission. Page/line numbers refer to the Word manuscript version; the current editable markdown uses section headings as location anchors.

---

## Main checklist

| Section/Topic | Item | D/E | Checklist item | Reported | Location |
|---|---|---|---|---|---|
| Title | 1 | D;E | Identify study as developing/evaluating a multivariable prediction model, target population, outcome | Yes | Title |
| Abstract | 2 | D;E | See TRIPOD+AI for Abstracts | Yes | Abstract |
| Introduction: Background | 3a | D;E | Healthcare context, rationale, existing models | Yes | Introduction ¶1-3 |
| | 3b | D;E | Target population and intended purpose | Yes | Introduction ¶4; Methods 2.1-2.2 |
| | 3c | D;E | Known health inequalities between sociodemographic groups | No | Not reported; no formal equity analysis |
| Introduction: Objectives | 4 | D;E | Objectives, development/validation | Yes | Introduction ¶4 |
| Methods: Data | 5a | D;E | Sources, rationale, representativeness | Yes | Methods 2.1, 2.2 |
| | 5b | D;E | Dates of data collection and follow-up | Yes | Methods 2.2, 2.6 |
| Methods: Participants | 6a | D;E | Setting, centres | Yes | Methods 2.1-2.2; single-centre |
| | 6b | D;E | Eligibility criteria | Yes | Methods 2.2, 2.4 |
| | 6c | D;E | Treatments received and handling | Yes | Methods 2.1-2.3; treatment intensity covariates |
| Methods: Data preparation | 7 | D;E | Pre-processing, quality checking | Yes | Methods 2.2, 2.4, 2.5 |
| Methods: Outcome | 8a | D;E | Outcome, time horizon, rationale | Yes | Methods 2.2 |
| | 8b | D;E | Subjective outcome assessor qualifications | Not applicable | Mortality from administrative/death records |
| | 8c | D;E | Actions to blind outcome assessment | Yes | Retrospective outcome extraction; split locked before unblinding |
| Methods: Predictors | 9a | D | Choice and pre-selection of predictors | Yes | Methods 2.3, 2.4, 2.6 |
| | 9b | D;E | Definition and timing of predictors | Yes | Methods 2.3, 2.4 |
| | 9c | D;E | Subjective predictor assessor qualifications | Not applicable | Structured clinical/ECG features |
| Methods: Sample size | 10 | D;E | Sample size calculation and sufficiency | Yes | Methods 2.7; power gate and MDD |
| Methods: Missing data | 11 | D;E | Missing-data handling and reasons | Yes | Methods 2.1, 2.5; Table 1 |
| Methods: Analytical methods | 12a | D | Data partitioning and use | Yes | Methods 2.6, 2.7 |
| | 12b | D | Predictor handling (transformations/standardisation) | Yes | Methods 2.3, 2.5 |
| | 12c | D | Model type, building steps, tuning, internal validation | Yes | Methods 2.6, 2.7 |
| | 12d | D;E | Heterogeneity across clusters | Not applicable | Single-centre; no cluster heterogeneity analysis |
| | 12e | D;E | Performance measures and rationale | Yes | Methods 2.7; Results 3.2-3.4 |
| | 12f | E | Model updating/recalibration from evaluation | Yes | Methods 2.7; Results 3.3; no recalibration triggered |
| | 12g | E | How predictions were calculated | Yes | Methods 2.6; model specifications |
| Methods: Class imbalance | 13 | D;E | Class imbalance methods | Not used | Event rate 18.2%; no class imbalance adjustment |
| Methods: Fairness | 14 | D;E | Approaches to fairness | Partial | Age/sex/ECG availability subgroups; no formal fairness metric |
| Methods: Model output | 15 | D | Model output and thresholds | Yes | Predicted probabilities; no classification threshold used for primary comparison |
| Methods: Training vs evaluation | 16 | D;E | Differences between development and evaluation data | Yes | Methods 2.6; Results 3.1-3.3 |
| Methods: Ethical approval | 17 | D;E | IRB/ethics approval | Partial | Ethics declaration placeholder; institutional determination pending |
| Open Science: Funding | 18a | D;E | Funding and funder role | Partial | Funding placeholder |
| Open Science: Conflicts of interest | 18b | D;E | Conflicts of interest | Yes | Declaration of competing interests |
| Open Science: Protocol | 18c | D;E | Where protocol can be accessed | Yes | Supplementary; SAP V1.3 |
| Open Science: Registration | 18d | D;E | Registration number or state not registered | Partial | ChiCTR placeholder |
| Open Science: Data sharing | 18e | D;E | Data availability | Yes | Data availability statement; repository URL placeholder |
| Open Science: Code sharing | 18f | D;E | Code availability | Partial | Code documented and uploaded in repository; URL placeholder |
| Patient and public involvement | 19 | D;E | Details or state no involvement | Yes | No PPI involvement |
| Results: Participants | 20a | D;E | Flow of participants | Yes | Figure 1; Methods 2.2; Results 3.1 |
| | 20b | D;E | Characteristics, missing data, events | Yes | Table 1; Results 3.1 |
| | 20c | E | Comparison with development data | Yes | Supplementary Table S1; temporal cohort characteristics |
| Results: Model development | 21 | D;E | Participants and events per analysis | Yes | Results 3.1; Table 2 |
| | 22 | D | Full prediction model details | Partial | Model names/specifications; full coefficient lists and code placeholder |
| Results: Model performance | 23a | D;E | Performance with CIs and subgroups | Yes | Tables 2-3; Figure 3-8 |
| | 23b | D;E | Heterogeneity across clusters | Not applicable | Single-centre |
| Results: Model updating | 24 | E | Results from model updating | Not applicable | No model update performed |
| Discussion: Interpretation | 25 | D;E | Interpretation and fairness context | Yes | Discussion 4, 4.1-4.2 |
| Discussion: Limitations | 26 | D;E | Limitations and effects | Yes | Discussion 4.1 |
| Discussion: Usability | 27a | D | Handling poor/unavailable inputs | Partial | ECG QC and availability analysis; implementation guidance limited |
| | 27b | D | User expertise and interaction | Partial | Not fully described |
| | 27c | D;E | Next steps and generalisability | Yes | Discussion 4.2 |

---

## TRIPOD+AI for Abstracts

| Abstract item | Item No | Reported | Location |
|---|---:|---|---|
| Title identifies development/evaluation, population, outcome | 1 | Yes | Title |
| Healthcare context/rationale | 2 | Yes | Abstract Background |
| Objectives | 3 | Yes | Abstract Methods |
| Source of data | 4 | Yes | Abstract Methods |
| Eligibility/setting | 5 | Yes | Abstract Methods |
| Outcome and time horizon | 6 | Yes | Abstract Methods |
| Model type, building, internal validation | 7 | Yes | Abstract Methods |
| Performance measures | 8 | Yes | Abstract Methods |
| Number of participants and events | 9 | Yes | Abstract Results |
| Predictors in final model | 10 | Yes | Abstract Methods/Results |
| Performance estimates with CIs | 11 | Yes | Abstract Results |
| Overall interpretation | 12 | Yes | Abstract Conclusions |
| Registration number | 13 | Partial | ChiCTR placeholder |

---

*Checklist generated for the IJMI submission package. Items marked "Partial" or "No" should be completed by the authors before final submission or explicitly addressed in the cover letter.*
