# PROBAST Assessment

**Study**：Frozen deep-learning ECG representations do not add confirmed incremental value to 28-day mortality prediction in sepsis: a retrospective development and temporal validation study

**Tool**：PROBAST (Wolff et al., Ann Intern Med 2019; doi:10.7326/M18-1376)

**Assessment date**：2026-08-29

**Overall risk-of-bias conclusion**：**Unclear-to-high**, mainly because the strong M1+ comparator intentionally uses a 24-hour window around t0, which is not necessarily available at the exact clinical decision time, and because no external multi-centre validation was performed. The primary negative conclusion is nonetheless consistent across sensitivity analyses.

**Overall applicability conclusion**：**High concern for patient/setting generalisability** (single tertiary ICU database); **high concern for predictor timing** (M1+ is a benchmark, not a point-of-care model); **low concern for outcome definition**.

---

## Signalling questions

### Domain 1: Participant selection (Risk of bias / applicability)

| Item | Question | Answer | Comments |
|---|---|---|---|
| 1.1 | Were appropriate data sources used? | Yes | Retrospective MIMIC-IV v3.1 + MIMIC-IV-ECG v1.0 cohort; clinically relevant ICU population. |
| 1.2 | Were all inclusions and exclusions of participants appropriate? | Probably yes | Explicit Sepsis-3, first-episode, age, ICU, survival and ECG eligibility rules; ECG availability creates selection, which is analysed separately. |
| Applicability | Do included participants/setting match the review question? | High concern | Single tertiary ICU; MIMIC-IV data may not represent other health systems. |

### Domain 2: Predictors (Risk of bias / applicability)

| Item | Question | Answer | Comments |
|---|---|---|---|
| 2.1 | Were predictors defined and assessed similarly for all participants? | Yes | Structured clinical scores, laboratory values and ECG quality rules were applied consistently. |
| 2.2 | Were predictor assessments made without knowledge of outcome data? | Yes | Model development and feature extraction were completed before test-label unblinding; split seed fixed before outcome access. |
| 2.3 | Are all predictors available at the time the model is intended to be used? | No/Probably no for M1+ | M1+ uses 24-h summary statistics around t0 and is explicitly framed as a high-information benchmark, not a decision-time model. Primary M3/M1 also include treatment-intensity variables within the same window. |
| Applicability | Does predictor definition/timing match the review question? | High concern for M1+ | The incremental-value comparison against M1+ should be read as an upper bound on information already available in routine data, not as a deployment recommendation. |

### Domain 3: Outcome (Risk of bias / applicability)

| Item | Question | Answer | Comments |
|---|---|---|---|
| 3.1 | Was the outcome determined appropriately? | Yes | 28-day all-cause mortality from t0; hospital follow-up and mortality tables linked. |
| 3.2 | Was a pre-specified or standard outcome definition used? | Yes | SAP V1.3; binary 28-day death. |
| 3.3 | Were predictors excluded from the outcome definition? | Yes | ECG and clinical features were not used to define death. |
| 3.4 | Was outcome defined/determined similarly for all participants? | Yes | Uniform 28-day horizon from sepsis onset. |
| 3.5 | Was outcome determined without knowledge of predictor information? | Yes | Outcome extraction and model evaluation were separated; test-set unblinding was delayed until analysis plan and power gate were finalised. |
| 3.6 | Was the time interval between predictors and outcome appropriate? | Probably yes | Outcome measured from t0; ECG window [t0−24 h, t0] in primary analysis and restricted-window sensitivity [t0−48 h, t0). |
| Applicability | Does outcome definition/timing match the review question? | Low concern | Clear, clinically meaningful 28-day mortality outcome. |

### Domain 4: Analysis (Risk of bias)

| Item | Question | Answer | Comments |
|---|---|---|---|
| 4.1 | Was there a reasonable number of participants with the outcome? | Yes | 3,002 deaths / 16,499 patients; test set 425 events; temporal set 380 events. |
| 4.2 | Were continuous and categorical predictors handled appropriately? | Probably yes | Ages/SOFA/lactate as continuous; lactate log-transformed before imputation; categorical infection sites retained. |
| 4.3 | Were all enrolled participants included in the analysis? | Yes | No per-model participant exclusions after ECG availability; complete-case sensitivity retained. |
| 4.4 | Were participants with missing data handled appropriately? | Yes | Chained-equation multiple imputation (m=20) fitted on training subset; complete-case analysis as sensitivity. |
| 4.5 | Was selection of predictors based on univariable analysis avoided? | Yes | Prespecified clinical variables and ECG representations; no univariate screening before fitting. |
| 4.6 | Were complexities in the data accounted for? | Yes | Binary outcome primary; competing-risk Fine-Gray sensitivity; ECG availability indicator. |
| 4.7 | Were relevant model performance measures evaluated appropriately? | Yes | AUC, CIs, paired DeLong comparison, NRI/IDI, calibration, Brier, DCA, temporal validation, subgroups and SHAP. |
| 4.8 | Were model overfitting, underfitting and optimism accounted for? | Probably yes | Separate training/tuning/internal-test/temporal sets; LASSO; no recalibration triggered; M5 exploratory. |
| 4.9 | Do final model weights correspond to the reported multivariable analysis? | Partial | Model specifications and repository code provided; full coefficient table and reproducible prediction object placeholders remain for publication. |
| Risk of bias | Could the statistical analysis have introduced bias? | Unclear-to-high | The deliberate ±24 h M1+ benchmark, single-centre design and absence of external validation are the main concerns. |

---

## Summary

| Domain | Risk of bias | Applicability concern |
|---|---|---|
| Participant selection | Low-to-unclear | High |
| Predictors | High (M1+ timing) | High |
| Outcome | Low | Low |
| Analysis | Unclear-to-high | Not applicable |

*This is an author-prepared assessment for reporting transparency. It is not a substitute for independent methodological appraisal.*
