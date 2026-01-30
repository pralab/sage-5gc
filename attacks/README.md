# Attacks

SAGE-5GC attacks implementations (black-box with genetic and random).

## Requirements

> [!NOTE]
> These scripts have been tested with Python 3.11.

## Installation

Install all dependencies:

```bash
pip install -r ..\requirements.txt
```

## Usage

### Black-box attack with genetic algorithm

#### Basic Comand

```bash
python blackbox_genetic_attack.py --model-name <MODEL_NAME> --ds-path <DATASET_PATH> --optimizer <OPTIMIZER_NAME>
```

#### Parameters

- `--model-name`: Name of the trained model to attack
    - `ABOD` - Angle-Based Outlier Detector
    - `CBLOF` - Cluster-Based Local Outlier Factor
    - `COPOD` - Copula-Based Outlier Detector
    - `ECOD` - Empirical Cumulative Distribution Functions for Outlier Detection
    - `Ensemble_SVC_C10_G10_HBOS_KNN_ABOD_INNE_PCA` - Ensemble of SVC, HBOS, KNN, ABOD, INNE, and PCA
    - `Ensemble_SVC_C10_G10_HBOS_KNN_GMM_INNE_PCA` - Ensemble of SVC, HBOS, KNN, GMM, INNE, and PCA
    - `Ensemble_SVC_C10_G10_HBOS_KNN_LOF_INNE_PCA` - Ensemble of SVC, HBOS, KNN, LOF, INNE, and PCA
    - `Ensemble_SVC_C100_G100_HBOS_KNN_LOF_INNE_FeatureBagging` - Ensemble of SVC, HBOS, KNN, LOF, INNE, and FeatureBagging
    - `FeatureBagging` - Feature Bagging
    - `GMM` - Gaussian Mixture Model
    - `HBOS` - Histogram-Based Outlier Detector
    - `IForest` - Isolation Forest
    - `INNE` - Influence Nearest Neighbor Ensemble
    - `KNN` - K-Nearest Neighbors
    - `LODA` - Lightweight On-line Detector of Anomalies
    - `LOF` - Local Outlier Factor
    - `PCA` - Principal Component Analysis

- `--ds-path`: Path to the attack dataset file (CSV format)

- `--optimizer`: Optimizer to use for the attack
    - `ES` - Evolution Strategies
    - `DE` - Differential Evolution
