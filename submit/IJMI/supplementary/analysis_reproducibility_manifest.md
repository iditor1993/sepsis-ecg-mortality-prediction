# Analysis Reproducibility Manifest

**Project**：Sepsis ECG deep-learning incremental-value study

**Data versions**：MIMIC-IV v3.1 + MIMIC-IV-ECG v1.0

**Data lock**：2026-08-27 17:55:54（见 `supplementary/data_lock.md`）

**Random seed**：20260823

**Manifest date**：2026-08-29

---

## 1. Analysis code

`code/` contains 36 Python scripts, 6 SQL scripts, `environment.yml`, `requirements_lock.txt` and `v14_ecg_encoder_provenance.md`.

Key analysis scripts:

- Cohort extraction and ECG linkage: `extract_cohort.py`, `ecg_link.py`, `splits.py`
- Feature engineering: `features_trackA.py`, `features_trackB.py`, `extract_covariates.py`, `extract_clinical_features.py`, `build_feature_matrix.py`, `audit_features.py`
- Power and missing data: `power_delong_sim.py`, `rho_power_recheck.py`, `impute_mice.py`
- Model training and evaluation: `train_m0_m5.py`, `train_m1plus.py`, `train_m2plus.py`, `train_m3plus.py`, `train_m4plus.py`, `train_m4plus_trackb.py`, `train_m5.py`, `evaluate.py`
- Sensitivity and robustness: `sensitivity_analyses.py`, `sensitivity_encoder_retrain.py`, `sensitivity_s1.py`, `sensitivity_s2.py`, `sensitivity_s6_s7.py`, `sensitivity_s8.py`, `a1b_availability.py`, `subgroup_analysis.py`, `e1_correlation.py`, `interpretability.py`
- Reporting: `baseline_table.py`, `make_figures.py`, `trackab_comparison.py`

Reproduction order is listed in `README.md`.

## 2. Locked data hashes (from `data/DATA_LOCK.md`)

| File | SHA-256 |
|---|---|
| `features_dev.parquet` | `5eef468f7bdc6ddd3cb5d5b21fea8825d4febb4fb032747a1f4ba5e479a06da3` |
| `features_temporal.parquet` | `1ce1d9c889c1283c5ffb29106ceebd1331f1de3d6284fbca8c62d701eec24268` |
| `splits.csv` | `b5bf9cf90ad82694f9f851a67a8a93e4127dc93a4dbe9856678dd58935ff22cf` |

## 3. Submission package SHA-256 at manifest time

| File | SHA-256 |
|---|---|
| `manuscript_IJMI.md` | `72acd7070c280c28a758d5b764d86335897ce603600c20b09397dcf2f7d712ca` |
| `documents/IJMI_manuscript.docx` | `21d0c85a937e13bc789e567d47fe764d5c7e80b60a72b5811c76e7bceae63bd0` |
| `documents/IJMI_cover_letter.docx` | `6fedb4653d791f7f9d44b3316a748c200a4333ea8415e5f0b3e37c500c02cec7` |
| `references/final_references.json` | `d2235b68c6a788c3b433b1a75fe6ad7e3720c6f7d26f4494c92033098de736a5` |
| `README.md` | `640a2bb624fa332b702e96d667a9b184bfb8f252706634b0e3b1a6915d963fff` |
| `supplementary/TRIPOD+AI_checklist.md` | `5c1a3566c9612ede66e4ae4a0015c04d4a79147974d72be3a2b4101b6b88eb4e` |
| `supplementary/PROBAST_assessment.md` | `7730cfca0b1b0c1011110b6520c1dd67988e5c973331b07992d9bd66247ab2bf` |
| `supplementary/internal_review_report.md` | `0177c43a1a57f9e2356f2621a1cb059bf63644272636109d113f7e1136fbb837` |
| `supplementary/integrity_verification_report.md` | `dac72f21422a3e178b25d1828ff651d9455f8efef4cb3ef559f20a974238b1eb` |
| `supplementary/Statistical_Analysis_Plan_V1.3_original.md` | `fe0412b8c8b773dfc16a2e66de9a978e858193b38f008b822c592700f6d89aff` |
| `supplementary/deviation_log.md` | `bd5867e84e13fae1d4db1ec40ce49c54de4f2d9c48f4757c3c1aa8b898c70a93` |
| `supplementary/data_lock.md` | `1da7f16407949b81d953d86fcc716ef897da5ae70d099006e2dc433972d2d095` |
| `supplementary/Table_S1_development_vs_temporal.csv` | `495a7e5137361b5993eef2d8f76e053b4f7176f3c5404083d3d1595d58b5e952` |

## 4. Key software versions

See `code/requirements_lock.txt` for the full locked environment. Core versions:

- Python 3.11.7
- pandas 2.3.3
- numpy 2.4.6
- scipy 1.17.1
- scikit-learn 1.9.0
- xgboost 3.2.0
- torch 2.7.1
- tensorflow 2.21.0
- lifelines 0.30.3
- duckdb 1.5.3
- shap 0.51.0
- statsmodels 0.14.6
- wfdb 4.3.1

## 5. Known constraints

- The full raw MIMIC-IV/ECG waveforms are not included in the submission package because they are large and governed by PhysioNet terms.
- The pretrained V14 encoder weights are large; the model provenance and the internal re-training robustness check are documented, and public weights are recommended for external reproduction.
- MIMIC-IV-ECG v1.0 does not cover the 2020-2022 period; this was documented as a deviation and is not a data-processing omission.

---

*Hashes were computed after the manuscript and supplementary files were finalised for the initial submission package. Re-run `Get-FileHash -Algorithm SHA256` after any revision.*
