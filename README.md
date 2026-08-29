# Sepsis ECG Mortality Prediction

This repository contains the analysis code and models used in a retrospective
prediction-model study of 28-day mortality in sepsis using MIMIC-IV v3.1 and
MIMIC-IV-ECG v1.0.

## Contents

- `src/`: cohort extraction, ECG linkage, feature engineering, model training,
  evaluation, sensitivity analyses and interpretability scripts.
- `sql/`: SQL definitions used to build cohorts, outcomes, covariates, clinical
  features and MIMIC-IV-ECG links.
- `models/`: the V14 ECG encoder and its provenance/licence notes.
- `results/`: analysis outputs, figures and summary tables.
- `sap/`: signed statistical analysis plan and pipeline figures.

## Data

MIMIC-IV v3.1: https://physionet.org/content/mimiciv/3.1/

MIMIC-IV-ECG v1.0: https://physionet.org/content/mimic-iv-ecg/1.0/

The raw data are not stored in this repository. Access is governed by
PhysioNet terms.

## Model licence

The V14 ECG encoder in `models/` was trained from scratch by the project author
and is released under the MIT License. See `LICENSE`.

## Software environment

See `environment.yml` and `requirements_lock.txt`.
