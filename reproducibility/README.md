# Reproducibility

This directory contains scripts to reproduce the experimental results and figures.

## Requirements

> [!NOTE]
> These scripts require Python 3.11 and the results data from the [`results`](../results/) directory.

> [!WARNING]
> Take into account that some paths in the scripts are hardcoded.

## Installation

Install all dependencies:

```bash
pip install -r ..\requirements.txt
```

## Scripts

### `create_datasets.py`

Creates train and test datasets from raw datasets. Merges raw data files, filters out TCP and ICMP packets, drops useless and constant columns, converts categorical columns to numeric, imputes missing values, and restores categorical columns.

**Usage:**

```bash
python create_datasets.py
```

---

### `create_heatmap.py`

Generates a heatmap visualization showing the class-wise performance of trained models across different attack categories and normal traffic.

**Usage:**

```bash
python create_heatmap.py
```

---

### `create_roc_plt.py`

Generates ROC curve plots comparing the performance of trained models. Loads trained models and test dataset, computes ROC curves and AUC scores for each model.

**Usage:**

```bash
python create_roc_plt.py
```

---

### `evaluate_models.py`

Evaluates trained models on test data by attack category. Computes performance metrics (accuracy, precision, recall, F1-score, ROC-AUC) for each model across different attack types.

**Usage:**

```bash
python evaluate_models.py
```

---

### `train_ensemble.py`

Trains and evaluates an ensemble anomaly detection model using LogisticRegression and SVM meta-learners. Combines predictions from multiple base detectors to improve detection performance.

**Usage:**

```bash
python train_ensemble.py
```

---

### `train_models.py`

Trains and evaluates multiple anomaly detection models using Grid Search for hyperparameter optimization. Tests various PyOD algorithms (ABOD, COPOD, ECOD, IForest, KNN, LOF, PCA, etc.) with different parameter configurations.

**Usage:**

```bash
python train_models.py
```
