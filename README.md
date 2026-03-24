# Bachelor Degree Research: ADMET Property Prediction

This repository contains the code and thesis PDF for a bachelor graduation project focused on **ADMET property prediction** using molecular fingerprints and multilayer perceptron (MLP) models.

- Thesis file: `20195067_박요한 최종논문.pdf`
- Main project directory: `ADMET_property_prediction/`

## Introduction

Drug discovery frequently fails at late stages because candidate molecules show poor
ADMET behavior (Absorption, Distribution, Metabolism, Excretion, and Toxicity), even
when they have good target activity. This project explores a machine learning workflow
that can estimate ADMET-related endpoints earlier in the screening process using
structure-derived molecular fingerprints and a lightweight neural network model.

The codebase is organized as practical Jupyter notebooks and utility modules, so the
research can be reproduced and extended for additional endpoints with the same
training/evaluation pipeline.

## Research Objective

The main objective of this research is to build and validate a reproducible ADMET
prediction framework that:

1. Learns endpoint-specific patterns from Morgan fingerprint representations.
2. Supports both classification and regression endpoints in a unified MLP backbone.
3. Compares conventional random split validation with scaffold-based splitting to better
   assess generalization to novel chemotypes.
4. Provides a baseline that can be improved later through architecture, feature, and
   data-centric enhancements.

## Project Scope

The implementation targets molecular property prediction workflows for both:

- **Classification** tasks (e.g., `BBB_logbb(cls)`)
- **Regression** tasks (e.g., `Lipophilicity`)

The experiments are organized around:

1. Baseline MLP training and evaluation (`MLP/Backbone.ipynb`)
2. Hyperparameter search (`MLP/Hyperparameter_tuning.ipynb`)
3. Scaffold split evaluation for better chemical generalization checks (`Scaffod_split/Classification.ipynb`, `Scaffod_split/Regression.ipynb`)

## Repository Structure

```text
Bachelor_degree/
├── 20195067_박요한 최종논문.pdf
├── README.md
└── ADMET_property_prediction/
    ├── Ablation/
    │   └── Classification.ipynb
    ├── MLP/
    │   ├── Backbone.ipynb
    │   ├── Hyperparameter_tuning.ipynb
    │   └── utils.py
    └── Scaffod_split/
        ├── Classification.ipynb
        ├── Regression.ipynb
        └── utils.py
```

> Note: `Scaffod_split` is the directory name used in this repository.

## Method Overview

### 1) Molecular Representation

- Uses precomputed Morgan fingerprint vectors loaded via `load_fingerprints(...)` from a Joblib file.
- Fingerprint files are expected to follow this naming pattern:
  - `fingerprints_<nbits>_<radius>.joblib`

### 2) Model Architecture

`MLP` (defined in `utils.py`) consists of:

- Input: `nBits` (typically `1024` in notebooks)
- Hidden layers:
  - Linear(`nBits` → 128) + BatchNorm + ReLU + Dropout
  - Linear(128 → 16) + BatchNorm + ReLU + Dropout
- Output:
  - Linear(16 → 1)
  - Sigmoid activation (used in the current implementation)

Other notable training utilities:

- `EarlyStopping` for validation-loss-based stopping
- Deterministic random seed setup for reproducibility
- `CustomDataset` wrapper for PyTorch DataLoader

### 3) Training / Validation Design

- Typical configuration in notebooks:
  - `nBits = 1024`
  - `num_epochs = 300`
  - `k_folds = 5`
  - `batch_size = 64`
- Baseline workflow uses `train_test_split` followed by K-fold CV.
- Scaffold workflow uses DeepChem `ScaffoldSplitter` to construct chemically distinct train/validation splits.

### 4) Evaluation Metrics

- Classification: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
- Regression: R², PCC, MSE, RMSE, MAE (plus MAPE in parts of the code)

## Environment

The notebooks and utility modules import the following core libraries:

- Python 3.x
- `numpy`, `pandas`, `matplotlib`
- `torch` (PyTorch)
- `scikit-learn`
- `scipy`
- `rdkit`
- `joblib`
- `deepchem` (for scaffold split notebooks)

If your environment does not yet have these packages, install them in your preferred way (conda/pip).

## How to Run

### 1) Prepare data and fingerprint files

In each notebook, set the dataset and fingerprint locations, for example:

- `file_path = '/path/to/your/data.csv'`
- `file_fingerprint = '/path/to/fingerprint/folder'`

### 2) Run baseline MLP experiments

- Open: `ADMET_property_prediction/MLP/Backbone.ipynb`
- Set the target endpoint (e.g., classification/regression)
- Run all cells

### 3) Run hyperparameter tuning

- Open: `ADMET_property_prediction/MLP/Hyperparameter_tuning.ipynb`
- Configure search space and data paths
- Run all cells

### 4) Run scaffold split experiments

- Open:
  - `ADMET_property_prediction/Scaffod_split/Classification.ipynb`
  - `ADMET_property_prediction/Scaffod_split/Regression.ipynb`
- Ensure `deepchem` is installed and SMILES column is configured
- Run all cells

## Notes on Reproducibility

The utilities set fixed random seeds (`777`) for NumPy and PyTorch and enable deterministic CUDA behavior when available. This improves run-to-run consistency.

## Thesis PDF

The bachelor thesis document is included as:

- `20195067_박요한 최종논문.pdf`

If you need a README section aligned to specific thesis chapters (e.g., Introduction/Method/Results), add a text-extractable version of the thesis (or chapter summaries), and this README can be refined further with chapter-level details.
