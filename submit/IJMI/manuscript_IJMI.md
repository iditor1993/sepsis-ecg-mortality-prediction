# Frozen deep-learning ECG representations do not add confirmed incremental value to 28-day mortality prediction in sepsis: a retrospective development and temporal validation study

**Authors**
[Author 1], [Author 2], [Author 3], [Author 4], [Author 5]

**Affiliations**
[Department, Institution, City, Country]

**Corresponding author**
[Name], [Email], [ORCID]

**Word counts**
Main text: 3,284 words (excluding abstract, figures, tables, references and declarations)

Abstract: 358 words

References: 46

**Running title**
ECG representations and sepsis mortality

## Highlights

- Frozen ECG features did not improve 28-day sepsis mortality prediction.
- A strong 85-column clinical pathway outperformed ECG-augmented models.
- Power gate fell below target; findings were reported as estimates.
- ECG availability was associated with illness severity and mortality.
- Strong clinical comparators and availability checks are needed in AI-ECG studies.

---

## Abstract

**Background.** Sepsis remains a leading cause of death, and existing clinical scores provide moderate discrimination for 28-day mortality. Deep learning models can extract patterns from routine electrocardiograms (ECGs), but evidence is limited on whether ECG-derived features add value to routine clinical information.

**Methods.** We performed a retrospective development and temporal validation study using MIMIC-IV and MIMIC-IV-ECG. Adults meeting Sepsis-3 criteria were included if an eligible 12-lead ECG was available within 24 h of the sepsis onset time. The primary outcome was 28-day all-cause mortality. A frozen, pretrained one-dimensional convolutional autoencoder produced 16 latent ECG dimensions (Track A). A 12-lead residual network trained from zero produced 32 task-specific dimensions (Track B). We compared an ECG-augmented model (M3; latent dimensions plus scores, lactate and covariates) with a conventional clinical model (M1) and a strong 85-column clinical tabular pathway (M1+), used as a high-information benchmark rather than a point-of-care model. Model discrimination, reclassification, calibration, decision-curve benefit, temporal performance, sensitivity analyses, subgroup interactions, availability adjustment and interpretability were assessed. Because the prespecified power gate was not met, the primary comparison was reported as estimation rather than confirmation.

**Results.** Of 16,499 patients, 3,002 (18.2%) died within 28 days. M1+ achieved test-set AUC 0.854 (95% CI 0.835–0.873), M1 0.797 (0.774–0.818), and M3 0.795 (0.773–0.817). M3 versus M1 gave ΔAUC −0.0014 (95% CI −0.0051 to +0.0023; P = 0.456); M3 versus M1+ gave ΔAUC −0.0587 (95% CI −0.0761 to −0.0427; P < 0.001). Temporal AUCs were 0.777, 0.777 and 0.842 for M1, M3 and M1+, respectively. Eleven sensitivity analyses, seven subgroup interaction tests, the ECG-availability adjustment and a replacement-representation test were consistent with the primary result. End-to-end fine tuning recovered a small signal (ΔAUC +0.018, 95% CI +0.008 to +0.028) but remained below M1+.

**Conclusions.** Frozen ECG representations did not provide a confirmed incremental benefit over strong clinical information for 28-day sepsis mortality. Routine clinical pathways may already capture the mortality signal that ECG models can extract, and apparent ECG gains may reflect care intensity and clinical ordering rather than waveform content.

**Keywords**
Sepsis; Electrocardiography; Machine learning; Mortality prediction; Clinical decision support; Incremental value

---

## Summary table

**What is already known**

- Deep learning models can identify sepsis and predict adverse outcomes from ECG waveforms.
- Standard clinical scores and longitudinal tabular data have limited but useful discrimination for sepsis mortality.
- Small or inadequately powered incremental-value studies may report optimistic or uncertain additions.

**What this study adds**

- In a large MIMIC-IV cohort, frozen ECG representations did not add confirmed incremental value over a conventional clinical model or a strong 85-column clinical pathway.
- Eleven sensitivity analyses, availability adjustment and all prespecified subgroup interactions were consistent with the negative primary finding.
- End-to-end adaptation improved ECG-only performance but did not exceed the strong clinical pathway, suggesting that ECG-derived mortality information substantially overlaps routine clinical data.

---

## 1. Introduction

Sepsis is a syndrome of host response dysregulation in which infection leads to life-threatening organ dysfunction [1]. It contributes to a substantial proportion of global deaths and hospital admissions [2,3]. The Sequential Organ Failure Assessment (SOFA) score, quick SOFA (qSOFA), National Early Warning Score (NEWS), Modified Early Warning Score (MEWS) and lactate are routinely used to assess severity [4–9]. These measures are inexpensive, immediately available and clinically interpretable, but their discriminative performance for intermediate-term mortality remains moderate [5,6,8].

Machine learning has been proposed as a means to improve sepsis risk prediction [10–12]. Electronic health record data provide a rich source of demographic, comorbidity, treatment, vital-sign and laboratory information. Among these, the Charlson comorbidity index and administrative coding algorithms are commonly used [10,11]. Systematic evidence demonstrates that machine learning models can outperform traditional screening instruments, although performance differs by setting, prediction window and outcome definition [12].

Large public data resources facilitate reproducible model development [13–15]. In particular, MIMIC-IV provides linked clinical events, and MIMIC-IV-ECG provides approximately 800,000 12-lead diagnostic ECGs across nearly 160,000 patients [13,15]. Deep neural networks trained on ECGs now achieve high performance for arrhythmia classification, ventricular dysfunction and other cardiovascular conditions [16–19]. AI-enabled ECG models have also been developed for serial mortality prediction in secondary care [20].

Several studies have directly examined ECG signals in sepsis. A deep learning screening model demonstrated high discrimination for sepsis and septic shock using ECG only [21]. Subsequent work has used 12-lead ECGs to predict bloodstream infection [22], derived an electrical risk score from standard ECGs to predict emergency-department mortality in sepsis [23], and combined heart-rate variability with machine learning for risk prediction [24,25]. Trajectory models and multiscale physiological dynamics have also shown potential for early deterioration and sepsis onset [26,27].

The relevant question for clinical practice, however, is not whether an ECG model can predict mortality when used alone. It is whether ECG-derived information adds incremental value after routine clinical information is already available. Prior work on reinforcement-learning treatment policies and machine learning delivery in sepsis has shown that high-performing algorithms do not automatically improve decisions or outcomes [28–32]. Generalizability, bias, clinical safety and implementation context are central concerns [29–32].

We therefore performed a prespecified, retrospective study in MIMIC-IV to test whether frozen deep-learning ECG representations improve 28-day all-cause mortality prediction beyond a conventional clinical model (M1) and a strong 85-column clinical pathway (M1+). We also evaluated temporal validation, calibration, decision-curve benefit, sensitivity to ECG representation choice, subgroup heterogeneity and availability bias.

## 2. Methods

### 2.1 Study design, data sources and reporting

This was a retrospective prediction-model development and temporal validation study (TRIPOD+AI type 2b) conducted with MIMIC-IV v3.1 and MIMIC-IV-ECG v1.0 [13,15]. The analysis was reported in accordance with TRIPOD, TRIPOD+AI and PROBAST [33–35]. Sample-size methods followed Riley et al. [36,37]. The statistical analysis protocol (V1.3) was finalized and signed before test-label unblinding; because the data lock occurred approximately one hour before the final protocol signature, this timing deviation is documented in the deviation log (D-001). The signed protocol, version history and deviation log are provided as supplementary material. The ChiCTR registration number will be entered before submission.

### 2.2 Participants and outcome

We identified patients from the sepsis3 derived table in MIMIC-IV. Sepsis-3 was defined as suspected infection plus an acute increase in SOFA score of at least two points [1]. Inclusion required age at least 18 years, a first qualifying sepsis episode, an ICU stay of at least six hours, survival and no automatic discharge before the sepsis onset time (t0), and an eligible diagnostic ECG within t0±24 h. One ECG per patient was selected as the closest ECG to t0. Automatic quality control excluded recordings with electrode saturation, baseline drift or excessive high-frequency noise [15].

The primary outcome was 28-day all-cause mortality from t0. Deaths after hospital discharge were determined from the MIMIC-IV mortality and follow-up tables. Because the primary analysis was binary, patients alive at 28 days were classified as survivors, and censoring was handled as described in the statistical analysis plan.

### 2.3 Clinical features

Clinical features included age, sex, Charlson comorbidity index [10,11], infection site, emergency admission status, pre-ICU hospital length of stay, invasive mechanical ventilation within t0±24 h, vasoactive drug use within t0±24 h, SOFA total score and six SOFA organ components, qSOFA, NEWS, MEWS, and lactate. Lactate was the first measurement within t0±6 h, with extension to t0±24 h when needed [9].

The conventional clinical model (M1) used scores, lactate and covariates in logistic regression. The strong pathway (M1+) used five summary statistics (mean, minimum, maximum, standard deviation and last value) for 17 core vital-sign and laboratory channels within t0±24 h, plus lactate and covariates, modeled with gradient boosting. This produced 85 clinical columns and was introduced to compete with the possibility that ECG gains merely re-encode existing clinical information. M1+ is an intentionally strong comparator rather than a point-of-care decision-time model; it uses the same 24-h window around t0 to provide a demanding tabular benchmark.

### 2.4 ECG representations

Track A used a frozen one-dimensional convolutional autoencoder that was pretrained without outcome labels on approximately 40,000 Lead II ECGs in a separate MIMIC-IV-ECG cohort. The 10-s Lead II segment was resampled to 250 Hz and 2,500 samples, producing 16 latent dimensions z1–z16. Weights were transferred without retraining and were not updated during clinical model fitting.

Track B used a residual convolutional network operating on 12 leads × 4,000 samples, trained on the development training subset. Its 512-dimensional embedding was reduced by principal component analysis to 32 dimensions (tb1–tb32). Track B was treated as a task-specific comparator.

### 2.5 Missing data

Lactate and a small number of clinical features had missing values. We used chained-equation multiple imputation with 20 imputations [38,39]. The imputation model included all analysis variables and the Nelson-Aalen cumulative hazard estimator. Fitting was restricted to the training subset. Lactate was imputed on the logarithmic scale after an initial raw-scale imputation produced implausible negative values. Complete-case analysis was retained as a sensitivity check.

### 2.6 Model development

M0 used SOFA alone. M1 used clinical scores, lactate and covariates in logistic regression. M2 used Track A latent dimensions alone. M3 was the prespecified primary model combining M1 features with z1–z16 in least absolute shrinkage and selection operator (LASSO) logistic regression. M4 used the same feature set with gradient boosting. M5 was an exploratory end-to-end model in which the last two encoder layers were unfrozen and a clinical branch was added. M1+ was the strong tabular pathway.

Patients were split at the patient level into development (2008–2016) and temporal validation (2017–2019) sets. The development set was divided into training (70%), tuning (15%) and internal test (15%) subsets using random seed 20260823. The split was fixed before outcome labels were accessed for model selection. All model fitting and hyperparameter tuning used training and tuning data only. The internal test set was evaluated once, followed by temporal validation.

### 2.7 Statistical analysis

Discrimination was summarized by AUC with 95% confidence intervals. Paired AUC comparisons used DeLong tests and 2,000 bootstrap resamples [40,41]. Reclassification was assessed with continuous net reclassification improvement, categorical net reclassification improvement and integrated discrimination improvement [41]. Calibration was summarized by intercept, slope and Brier score, with Platt calibration fitted on the tuning set [42]. Clinical utility was assessed with decision-curve analysis over threshold probabilities of 5%–50% [43].

Temporal performance was evaluated separately. Competing-risk analysis used the Fine-Gray subdistribution hazard model, with hospital discharge treated as a competing event [44]. Prespecified subgroup interactions were exploratory; age, sex, septic shock, SOFA tertile, infection site, atrial fibrillation and ventricular rate were examined, with no formal multiplicity correction applied. ECG availability was examined by comparing available and unavailable patients and by adding a binary availability indicator to M1+.

Before unblinding the test labels, we calculated the power to detect a ΔAUC of 0.02 under baseline AUCs of 0.75–0.80 and correlations of at least 0.85 for the nested comparisons. The minimum detectable ΔAUC at the conservative corner was 0.0215, exceeding the 0.02 target. Instead of a confirmatory claim, the primary comparison was therefore reported as an estimate with confidence intervals. Empirical correlation estimates were also reported after model fitting.

Feature importance was examined with SHAP [45]. Exploratory correlation analysis examined relationships between z1–z16 and SOFA organ components and lactate [45]. Explanatory methods were interpreted cautiously, in line with the argument that interpretability is not a substitute for clinical validation [46].

### 2.8 Reproducibility

The random seed, database versions, feature-matrix hashes and software versions were recorded. The overall analysis pipeline is shown in Figure 2. Analysis code is provided in the repository. The strong clinical feature set was audited to exclude identifier columns. Any deviations from the prespecified plan are documented in the deviation log.

## 3. Results

### 3.1 Cohort

After Sepsis-3 identification, first-episode restriction, ICU and survival rules, ECG linkage and signal-quality filtering, 16,499 patients remained (Figure 1). The development set comprised 14,780 patients (10,346 training, 2,217 tuning, 2,217 test). The temporal set comprised 1,719 patients. A total of 3,002 patients (18.2%) died within 28 days, including 425 of 2,217 test patients and 380 of 1,719 temporal patients. The COVID-period cohort was not feasible because MIMIC-IV-ECG v1.0 did not provide linked ECGs in that window.

Baseline characteristics are summarized in Table 1, and development versus temporal cohort characteristics are compared in Supplementary Table S1. Patients who died were older, had higher SOFA, NEWS, MEWS and lactate values, and were more likely to be admitted through the emergency department. Mechanical ventilation and vasoactive drug use were also more common among non-survivors.

### 3.2 Primary performance

Test-set AUCs were 0.594 for M0, 0.797 for M1, 0.642 for M2, 0.795 for M3, 0.854 for M1+ and 0.805 for M4 (Table 2, Figure 3). Among the prespecified models, M1+ had the highest discrimination. The exploratory direct comparison M4+ reached 0.857, but its gain over M1+ was +0.0032 and was not clinically meaningful (Section 3.6). M3 was not superior to M1 (ΔAUC −0.0014, 95% CI −0.0051 to +0.0023, P = 0.456) and was substantially worse than M1+ (ΔAUC −0.0587, 95% CI −0.0761 to −0.0427, P < 0.001). Continuous net reclassification improvement for M3 versus M1 was −0.137 (P < 0.001), category net reclassification improvement +0.001 (ns) and integrated discrimination improvement −0.002 (ns). All metrics worsened relative to M1+ (Table 3).

Calibration was acceptable for all three clinical models (Figure 4). On the test set, the Brier score for M3 was 0.1265 before calibration and 0.1266 after Platt calibration; for M1+ it was 0.1074 and 0.1069. Decision-curve analysis showed positive net benefit for M1, M3 and M1+ across the full 5%–50% range, with M1+ providing the greatest benefit at every threshold (Figure 5).

### 3.3 Temporal validation

Temporal AUCs were 0.612 for M0, 0.777 for M1, 0.601 for M2, 0.777 for M3, 0.842 for M1+ and 0.789 for M4. The M3 drop from test to temporal was 0.019, and the M1+ drop was 0.012, both below the prespecified 0.05 threshold. Temporal calibration slopes were 0.909 for M3 and 0.944 for M1+, with Brier scores of 0.140 and 0.120. No recalibration plan was triggered.

### 3.4 Sensitivity analyses

The main result was stable across all 11 prespecified sensitivity checks (Table 4). The window restricted to [t0–48 h, t0) retained 8,941 patients and gave ΔAUC +0.0006 (ns). Outcome defined from admission gave ΔAUC −0.0016 (ns). Complete-case analysis (n = 1,797) gave ΔAUC −0.0029 (ns). Replacing Track A with Track B gave ΔAUC −0.0139 (ns). Excluding atrial fibrillation or paced rhythm gave ΔAUC −0.0018 (ns), and using t0 SOFA gave ΔAUC −0.0004 (ns). In the competing-risk framework, lactate remained strongly associated with mortality (subdistribution HR 1.293, P ≈ 4 × 10−43). Platt, isotonic and uncalibrated Brier scores differed by less than 0.0004. Removing lactate reduced M3 AUC from 0.795 to 0.773, indicating that lactate was a substantial clinical carrier. A re-trained encoder with an identical architecture produced results consistent with the main analysis (M3′ versus M1′ ΔAUC −0.0025).

### 3.5 Subgroup and availability analyses

No prespecified subgroup interaction was significant (interaction P 0.089–0.792; Figure 6). Infection-site counts were small in some categories, so these estimates should be read as exploratory. Patients with linked ECGs differed from those without (Table 6): vasoactive drug use was 48.0% versus 33.8% (SMD 0.293), and 28-day mortality was 18.2% versus 23.2%. Four of 13 variables had |SMD| > 0.1, indicating that ECG ordering carried prognostic information. When a binary ECG-availability indicator was added to M1+, its independent contribution was +0.0016 (ns). After controlling for availability, M3 remained worse than the clinical pathway by ΔAUC −0.0626.

### 3.6 Representation choice and end-to-end adaptation

Track B performed better than Track A when ECG was used alone (M2: 0.681 vs. 0.642; ΔAUC +0.039). After adding clinical features, Track A was similar or better in the combined models (M3: 0.795 vs. 0.783; M4: 0.805 vs. 0.784). The strongest direct comparison added ECG latent dimensions directly to M1+. Track A yielded ΔAUC +0.0032 (95% CI −0.0002 to +0.0065), and Track B yielded ΔAUC −0.0146 (95% CI −0.0256 to −0.0031). Neither represented a clinically meaningful increment. The exploratory M5 model improved over M1 by ΔAUC +0.018 (95% CI +0.008 to +0.028) but reached 0.815, below M1+ at 0.854.

### 3.7 Interpretability and latent spectrum

SHAP analysis ranked clinical variables first (Table 5, Figure 7). For M3, the top features were emergency admission (mean |SHAP| 0.444), Charlson index (0.389), MEWS (0.241), lactate (0.197), NEWS (0.172) and age (0.154). The strongest latent dimension, z2, ranked tenth with mean |SHAP| 0.042. The pattern was similar in M4.

The exploratory correlation spectrum was diffuse (Figure 8). The largest absolute Spearman correlation between any z dimension and an organ score or lactate was 0.15. Correlations with cardiovascular SOFA were close to zero. Thus, the latent representation did not provide an obvious organ-specific physiological readout.

## 4. Discussion

In this large retrospective cohort, adding frozen deep-learning ECG representations to standard clinical scores did not improve 28-day mortality prediction. The primary ECG-augmented model was not better than a conventional clinical model and was clearly inferior to a strong 85-column clinical pathway. These results were robust across temporal validation, 11 sensitivity analyses, replacement by a supervised representation, availability adjustment and all prespecified subgroup tests.

Our findings do not imply that ECG signals contain no prognostic information. AI-ECG models can predict arrhythmias, ventricular dysfunction, infection and mortality [16–23], and a task-adapted end-to-end model in our study recovered a modest improvement over M1. Rather, our data indicate that the mortality-related signal available in a single resting ECG largely overlaps with information already contained in routine clinical features, illness scores, treatment intensity and chronological deterioration [24–27]. The null result is therefore consistent with the hypothesis that the practical question is incremental value, not whether ECG can predict risk when used alone.

This interpretation is reinforced by the ECG-availability analysis. Patients with an available ECG had higher vasoactive drug use and lower 28-day mortality than patients without one, suggesting that ECG ordering is related to clinical context. Adding a binary availability indicator to M1+ produced no meaningful gain, and the ECG-augmented model remained inferior after adjustment. Previous deployment studies similarly show that apparent model gains can be confounded by where and when tests are requested [29–32].

The strongest clinical pathway in our study achieved AUC 0.854 on the test set, 0.842 on temporal validation, acceptable calibration and positive net benefit across clinically plausible thresholds. These results are relevant to medical informatics because they show that well-designed tabular pathways may already be a strong comparator. Future work claiming a benefit from waveform models should therefore use strong clinical baselines, prespecified power analysis, availability adjustment and external validation from the outset.

### 4.1 Limitations

The study was single-centre and retrospective. External validation in another hospital system was not performed. MIMIC-IV-ECG contains one diagnostic ECG per encounter within a wide time window, not continuous monitoring. M1+ used summary statistics from a 24-h window around t0 and should therefore be read as a high-information benchmark rather than a point-of-care decision-time model; the information available to clinicians at the exact decision time may differ. Our sample size was large, but the paired ΔAUC comparison against M1+ lacked sufficient power to detect very small improvements. The V14 encoder was a team-specific pretrained model; although we re-trained an identical architecture internally, external reproducibility would benefit from public weights. Saliency review by cardiologists was planned but not completed before this manuscript was prepared. Finally, the reported 2020–2022 cohort could not be evaluated because MIMIC-IV-ECG v1.0 did not cover that period.

### 4.2 Clinical and research implications

For practice, the immediate implication is that adding a frozen ECG representation to an already strong clinical pathway is unlikely to change risk stratification for 28-day sepsis mortality. For research, our results argue for the adoption of strong clinical comparators and explicit tests of incremental value. They also suggest that promising ECG-only performance should not be interpreted as proof of clinical usefulness.

## 5. Conclusions

Frozen deep-learning ECG representations did not add a confirmed incremental benefit to 28-day mortality prediction in sepsis beyond standard or strong clinical information. A routine clinical tabular pathway was the best-performing prespecified model and remained well calibrated in temporal validation. ECG-derived information may overlap substantially with clinical context, and apparent predictive gains should be evaluated against strong baselines, availability bias and prespecified incremental-value criteria before deployment.

## 6. Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this manuscript, the authors used Codex, an AI-assisted writing tool, to draft text, organize tables and verify references. The authors reviewed all content, revised and edited the manuscript, and take full responsibility for the final submission. No generative AI tool was used to generate or alter the research data or results.

## 7. Declarations

### 7.1 Ethics approval

MIMIC-IV and MIMIC-IV-ECG are de-identified public datasets. Access was obtained under the PhysioNet data use agreement and standard credentialing requirements. The study did not involve direct patient contact or identifiable data. [Institutional ethics determination to be completed by the authors before submission.]

### 7.2 Data availability

MIMIC-IV and MIMIC-IV-ECG are available from PhysioNet under credentialed access. The processed feature matrices and model outputs are documented by SHA-256 hashes in the repository. The repository code is available at [repository URL] upon publication; no patient-identifiable data are included. The signed statistical analysis plan, version history and deviation log are provided as supplementary material.

### 7.3 Funding

[Funding information to be completed by the authors.]

### 7.4 Competing interests

The authors declare no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

### 7.5 Author contributions

CRediT statement to be completed by the authors: [Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing – original draft, Writing – review and editing, Visualization, Supervision, Project administration].

## 8. Acknowledgments

[Acknowledgments to be completed by the authors.]

---

## References

[1] M. Singer, C.S. Deutschman, C.W. Seymour, et al. The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). JAMA 315 (8) (2016) 801–810. https://doi.org/10.1001/jama.2016.0287.
[2] K.E. Rudd, S.C. Johnson, K.M. Agesa, et al. Global, regional, and national sepsis incidence and mortality, 1990–2017: analysis for the Global Burden of Disease Study. Lancet 395 (10219) (2020) 200–211. https://doi.org/10.1016/S0140-6736(19)32989-7.
[3] C. Fleischmann-Struzek, L. Mellhammar, N. Rose, et al. Incidence and mortality of hospital- and ICU-treated sepsis: results from an updated and expanded systematic review and meta-analysis. Intensive Care Med 46 (2020) 1552–1562. https://doi.org/10.1007/s00134-020-06151-x.
[4] J.-L. Vincent, R. Moreno, J. Takala, et al. The SOFA (Sepsis-related Organ Failure Assessment) score to describe organ dysfunction/failure. Intensive Care Med 22 (7) (1996) 707–710. https://doi.org/10.1007/BF01709751.
[5] C.W. Seymour, V.X. Liu, T.J. Iwashyna, et al. Assessment of Clinical Criteria for Sepsis. JAMA 315 (8) (2016) 762–774. https://doi.org/10.1001/jama.2016.0288.
[6] E.P. Raith, A.A. Udy, M. Bailey, et al. Prognostic accuracy of the SOFA score, SIRS criteria, and qSOFA score for in-hospital mortality among adults with suspected infection admitted to the intensive care unit. JAMA 317 (3) (2017) 290–300. https://doi.org/10.1001/jama.2016.20328.
[7] C.P. Subbe, M. Kruger, P. Rutherford, L. Gemmel. Validation of a modified Early Warning Score in medical admissions. QJM 94 (10) (2001) 521–526. https://doi.org/10.1093/qjmed/94.10.521.
[8] T. Mitsunaga, I. Hasegawa, M. Uzura, et al. Comparison of the National Early Warning Score (NEWS) and the Modified Early Warning Score (MEWS) for predicting admission and in-hospital mortality in elderly patients in the pre-hospital setting and in the emergency department. PeerJ 7 (2019) e6947. https://doi.org/10.7717/peerj.6947.
[9] B. Casserly, G.S. Phillips, C. Schorr, et al. Lactate measurements in sepsis-induced tissue hypoperfusion: results from the Surviving Sepsis Campaign database. Crit Care Med 43 (2015) 567–573. https://doi.org/10.1097/CCM.0000000000000742.
[10] M.E. Charlson, P. Pompei, K.L. Ales, C.R. MacKenzie. A new method of classifying prognostic comorbidity in longitudinal studies: development and validation. J Chronic Dis 40 (5) (1987) 373–383. https://doi.org/10.1016/0021-9681(87)90171-8.
[11] H. Quan, V. Sundararajan, P. Halfon, et al. Coding algorithms for defining comorbidities in ICD-9-CM and ICD-10 administrative data. Med Care 43 (11) (2005) 1130–1139. https://doi.org/10.1097/01.mlr.0000182534.19832.83.
[12] L.M. Fleuren, T.L.T. Klausch, C.L. Zwager, et al. Machine learning for the prediction of sepsis: a systematic review and meta-analysis of diagnostic test accuracy. Intensive Care Med 46 (3) (2020) 383–400. https://doi.org/10.1007/s00134-019-05872-y.
[13] A.E.W. Johnson, L. Bulgarelli, L. Shen, et al. MIMIC-IV, a freely accessible electronic health record dataset. Sci Data 10 (2023) 1. https://doi.org/10.1038/s41597-022-01899-x.
[14] A.L. Goldberger, L.A.N. Amaral, L. Glass, et al. PhysioBank, PhysioToolkit, and PhysioNet: components of a new research resource for complex physiologic signals. Circulation 101 (23) (2000) e215–e220. https://doi.org/10.1161/01.CIR.101.23.e215.
[15] B. Gow, T. Pollard, L.A. Nathanson, et al. MIMIC-IV-ECG: diagnostic electrocardiogram matched subset. PhysioNet (version 1.0, 2023). https://doi.org/10.13026/4nqg-sb35.
[16] A.Y. Hannun, P. Rajpurkar, M. Haghpanahi, et al. Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network. Nat Med 25 (2019) 65–69. https://doi.org/10.1038/s41591-018-0268-3.
[17] Z.I. Attia, S. Kapa, F. Lopez-Jimenez, et al. Screening for cardiac contractile dysfunction using an artificial intelligence-enabled electrocardiogram. Nat Med 25 (2019) 70–74. https://doi.org/10.1038/s41591-018-0240-2.
[18] A.H. Ribeiro, M.H. Ribeiro, G.M.M. Paixão, et al. Automatic diagnosis of the 12-lead ECG using a deep neural network. Nat Commun 11 (2020) 1760. https://doi.org/10.1038/s41467-020-15432-4.
[19] K.C. Siontis, P.A. Noseworthy, Z.I. Attia, P.A. Friedman. Artificial intelligence-enhanced electrocardiography in cardiovascular disease management. Nat Rev Cardiol 18 (7) (2021) 465–478. https://doi.org/10.1038/s41569-020-00503-2.
[20] G. Tsaban, A. Harari, A. Shiloh, et al. Artificial intelligence-enabled serial electrocardiograms for prediction of all-cause mortality in secondary care settings. JACC Adv 5 (7) (2026) 102875. https://doi.org/10.1016/j.jacadv.2026.102875.
[21] J.-M. Kwon, Y.R. Lee, M.-S. Jung, et al. Deep-learning model for screening sepsis using electrocardiography. Scand J Trauma Resusc Emerg Med 29 (2021) 145. https://doi.org/10.1186/s13049-021-00953-8.
[22] P.-H. Chen, S.-Y. Li, D.-J. Tsai, et al. Deep learning analysis of 12-lead electrocardiograms for bloodstream infection prediction: a multi-center validation study. BMC Med Inform Decis Mak (2026). https://doi.org/10.1186/s12911-026-03657-0.
[23] F.A. Ayyıldız, A. Ayyıldız, G. Yıldız, et al. Electrical Risk Score derived from standard ECG predicts mortality in sepsis patients presenting to the emergency department. Heart Lung 77 (2026) 102771. https://doi.org/10.1016/j.hrtlng.2026.102771.
[24] F.M. de Castilho, A.L.P. Ribeiro, V. Nobre, G. Barros, M.R. de Sousa. Heart rate variability as predictor of mortality in sepsis: a systematic review. PLoS One 13 (9) (2018) e0203487. https://doi.org/10.1371/journal.pone.0203487.
[25] C.J. Chiew, N. Liu, T. Tagami, et al. Heart rate variability based machine learning models for risk prediction of suspected sepsis patients in the emergency department. Medicine (Baltimore) 98 (6) (2019) e14197. https://doi.org/10.1097/MD.0000000000014197.
[26] R. Zhang, F. Long, Z. Zhao, et al. Machine learning predicts sepsis deterioration trajectories. NPJ Digit Med 9 (1) (2026). https://doi.org/10.1038/s41746-026-02565-x.
[27] S.P. Shashikumar, M.D. Stanley, I. Sadiq, et al. Early sepsis detection in critical care patients using multiscale blood pressure and heart rate dynamics. J Electrocardiol 50 (6) (2017) 739–743. https://doi.org/10.1016/j.jelectrocard.2017.08.013.
[28] M. Komorowski, L.A. Celi, O. Badawi, A.C. Gordon, A.A. Faisal. The Artificial Intelligence Clinician learns optimal treatment strategies for sepsis in intensive care. Nat Med 24 (11) (2018) 1716–1720. https://doi.org/10.1038/s41591-018-0213-5.
[29] J. Futoma, M. Simons, T. Panch, F. Doshi-Velez, L.A. Celi. The myth of generalisability in clinical research and machine learning in health care. Lancet Digit Health 2 (9) (2020) e489–e492. https://doi.org/10.1016/S2589-7500(20)30186-2.
[30] R. Challen, J. Denny, M. Pitt, et al. Artificial intelligence, bias and clinical safety. BMJ Qual Saf 28 (3) (2019) 231–237. https://doi.org/10.1136/bmjqs-2018-008370.
[31] P. Rajpurkar, E. Chen, O. Banerjee, E.J. Topol. AI in health and medicine. Nat Med 28 (1) (2022) 31–38. https://doi.org/10.1038/s41591-021-01614-0.
[32] M.P. Sendak, W. Ratliff, D. Sarro, et al. Real-world integration of a sepsis deep learning technology into routine clinical care: implementation study. JMIR Med Inform 8 (7) (2020) e15182. https://doi.org/10.2196/15182.
[33] G.S. Collins, J.B. Reitsma, D.G. Altman, K.G.M. Moons. Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis (TRIPOD): the TRIPOD statement. Ann Intern Med 162 (1) (2015) 55–63. https://doi.org/10.7326/M14-0697.
[34] G.S. Collins, K.G.M. Moons, P. Dhiman, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ 385 (2024) e078378. https://doi.org/10.1136/bmj-2023-078378.
[35] R.F. Wolff, K.G.M. Moons, R.D. Riley, et al. PROBAST: a tool to assess the risk of bias and applicability of prediction model studies. Ann Intern Med 170 (1) (2019) 51–58. https://doi.org/10.7326/M18-1376.
[36] R.D. Riley, J. Ensor, K.I.E. Snell, et al. Calculating the sample size required for developing a clinical prediction model. BMJ 368 (2020) m441. https://doi.org/10.1136/bmj.m441.
[37] R.D. Riley, K.I.E. Snell, J. Ensor, et al. Minimum sample size for developing a multivariable prediction model: part II – binary and time-to-event outcomes. Stat Med 38 (7) (2019) 1276–1296. https://doi.org/10.1002/sim.7992.
[38] S. van Buuren, K. Groothuis-Oudshoorn. mice: multivariate imputation by chained equations in R. J Stat Softw 45 (3) (2011) 1–67. https://doi.org/10.18637/jss.v045.i03.
[39] I.R. White, P. Royston, A.M. Wood. Multiple imputation using chained equations: issues and guidance for practice. Stat Med 30 (4) (2011) 377–399. https://doi.org/10.1002/sim.4067.
[40] E.R. DeLong, D.M. DeLong, D.L. Clarke-Pearson. Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. Biometrics 44 (3) (1988) 837–845. https://doi.org/10.2307/2531595.
[41] M.J. Pencina, R.B. D'Agostino Sr, R.B. D'Agostino Jr, R.S. Vasan. Evaluating the added predictive ability of a new marker: from area under the ROC curve to reclassification and beyond. Stat Med 27 (2) (2008) 157–172. https://doi.org/10.1002/sim.2929.
[42] B. Van Calster, D.J. McLernon, M. van Smeden, L. Wynants, E.W. Steyerberg. Calibration: the Achilles heel of predictive analytics. BMC Med 17 (2019) 230. https://doi.org/10.1186/s12916-019-1466-7.
[43] A.J. Vickers, E.B. Elkin. Decision curve analysis: a novel method for evaluating prediction models. Med Decis Mak 26 (6) (2006) 565–574. https://doi.org/10.1177/0272989X06295361.
[44] J.P. Fine, R.J. Gray. A proportional hazards model for the subdistribution of a competing risk. J Am Stat Assoc 94 (446) (1999) 496–509. https://doi.org/10.1080/01621459.1999.10474144.
[45] S.M. Lundberg, G. Erion, H. Chen, et al. From local explanations to global understanding with explainable AI for trees. Nat Mach Intell 2 (2020) 56–67. https://doi.org/10.1038/s42256-019-0138-9.
[46] C. Rudin. Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. Nat Mach Intell 1 (5) (2019) 206–215. https://doi.org/10.1038/s42256-019-0048-x.

---

## Tables

**Table 1. Baseline characteristics by 28-day mortality**
[Table 1 in editable form: `tables/Table1_baseline_characteristics.csv`]

**Table 2. Test-set and temporal discrimination**

| Model | Components | Test AUC (95% CI) | Temporal AUC |
|---|---|---:|---:|
| M0 | SOFA alone | 0.594 (0.565–0.623) | 0.612 |
| M1 | Scores + lactate + covariates (LR) | 0.797 (0.774–0.818) | 0.777 |
| M2 | Track A latent dimensions (LR) | 0.642 (0.613–0.669) | 0.601 |
| M3 | M1 + Track A latent dimensions (LASSO-LR) | 0.795 (0.773–0.817) | 0.777 |
| M1+ | 85-column clinical pathway (XGBoost) | 0.854 (0.835–0.873) | 0.842 |
| M4 | M3 features (XGBoost) | 0.805 (0.783–0.827) | 0.789 |
| M4+ | M1+ features + Track A latent dimensions (XGBoost) | 0.857 (exploratory) | Not evaluated |
| M4+-TB | M1+ features + Track B latent dimensions (XGBoost) | 0.839 (exploratory) | Not evaluated |
| M5 | End-to-end ECG + clinical branch | 0.815 | Not evaluated |

Note. M4+ and M4+-TB are exploratory direct comparisons and were not prespecified as validation models; 95% confidence intervals are not reported for these two rows.

**Table 3. Primary incremental-value and reclassification analyses**

| Comparison | Metric | Estimate (95% CI) | P |
|---|---|---:|---:|
| M3 vs M1 | ΔAUC | −0.0014 (−0.0051 to +0.0023) | 0.456 |
| M3 vs M1+ | ΔAUC | −0.0587 (−0.0761 to −0.0427) | <0.001 |
| M3 vs M1 | Continuous NRI | −0.137 (−0.177 to −0.096) | <0.001 |
| M3 vs M1 | Category NRI | +0.001 (−0.009 to +0.012) | 0.794 |
| M3 vs M1 | IDI | −0.002 (−0.005 to +0.001) | 0.178 |
| M3 vs M1+ | Continuous NRI | −0.390 (−0.428 to −0.352) | <0.001 |
| M3 vs M1+ | Category NRI | −0.115 (−0.138 to −0.093) | <0.001 |
| M3 vs M1+ | IDI | −0.115 (−0.136 to −0.094) | <0.001 |

**Table 4. Sensitivity analyses for the primary comparison**

| ID | Analysis | Main result |
|---|---|---|
| S1 | ECG window [t0–48 h, t0) | ΔAUC +0.0006 (ns) |
| S2 | Outcome from admission | ΔAUC −0.0016 (ns) |
| S3 | Exclude ICU-only subcohort | Not applicable; cohort is ICU-level |
| S4 | Complete case (n = 1,797) | ΔAUC −0.0029 (ns) |
| S5 | Track B replaces Track A | ΔAUC −0.0139 (ns) |
| S6 | Exclude atrial fibrillation/pacing | ΔAUC −0.0018 (ns) |
| S7 | SOFA at t0 | ΔAUC −0.0004 (ns) |
| S8 | Competing-risk framework | Lactate subHR 1.293, P ≈ 4 × 10−43 |
| S9 | Calibration method | Brier difference <0.0004 |
| S10 | Remove lactate | M3 AUC 0.773; ΔAUC −0.024 |
| S11 | ECG-availability indicator | +0.0016 (ns); adjusted M3 −0.063 |
| Re-ENC | Internal re-training of encoder | M3′ vs M1′ ΔAUC −0.0025 |

**Table 5. Top SHAP features**

| Rank | M3 feature | Mean \|SHAP\| | M4 feature | Mean \|SHAP\| |
|---|---:|---:|---:|---:|
| 1 | Admission emergency | 0.444 | Charlson index | 0.436 |
| 2 | Charlson index | 0.389 | Admission emergency | 0.431 |
| 3 | MEWS | 0.241 | Lactate | 0.259 |
| 4 | Lactate | 0.197 | MEWS | 0.202 |
| 5 | NEWS | 0.172 | NEWS | 0.192 |
| 6 | Age | 0.154 | Age | 0.163 |
| 10 | z2 | 0.042 | — | — |
| 11 | — | — | z2 | 0.041 |

**Table 6. ECG availability and outcome characteristics**
[Table 6 in editable form: `tables/Table6_ECG_availability.csv`]

---

## Figure legends

**Figure 1.** Cohort flow from MIMIC-IV hospital admissions to the locked analysis cohort, development, temporal and infeasible COVID-period sets.

**Figure 2.** Analysis pipeline, representation choices, model hierarchy and locked evaluation strategy.

**Figure 3.** Test-set receiver operating characteristic curves for M0, M1, M2, M3, M1+ and M4.

**Figure 4.** Test-set calibration for M1, M3 and M1+ after Platt calibration.

**Figure 5.** Decision-curve analysis from 5% to 50% threshold probability.

**Figure 6.** Prespecified subgroup ΔAUC (M3 minus M1) with 95% bootstrap confidence intervals.

**Figure 7.** SHAP mean absolute importance for M3 and M4; the strongest latent dimension (z2) ranked tenth in M3 and eleventh in M4.

**Figure 8.** Spearman correlation heatmap between ECG latent dimensions z1–z16 and SOFA organ components plus lactate.

**Graphical abstract.** The study flow, key model results and negative incremental-value conclusion.
