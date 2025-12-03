"""Script to train and evaluate multiple anomaly detection models using Grid Search."""

import json
import os
from pathlib import Path
import re
import sys

import joblib
import numpy as np
import pandas as pd
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.gmm import GMM
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.inne import INNE
from pyod.models.kde import KDE
from pyod.models.knn import KNN
from pyod.models.lmdd import LMDD
from pyod.models.lof import LOF
from pyod.models.lscp import LSCP
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split

sys.path.append(str(Path(__file__).parent.parent))
import logging

from ml_models import Detector
from preprocessing import Preprocessor

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def make_json_serializable(d: dict) -> dict:
    """
    Recursively convert non-serializable values to serializable ones.

    Parameters
    ----------
    d : dict
        Input dictionary.

    Returns
    -------
    dict
        JSON-serializable dictionary.
    """
    out = {}
    for k, v in d.items():
        # For sklearn objects (like KMeans) convert to repr string
        if hasattr(v, "get_params"):
            out[k] = {"class": v.__class__.__name__, "params": v.get_params()}
        # For lists, recursively
        elif isinstance(v, list):
            out[k] = [
                make_json_serializable(x) if isinstance(x, dict) else str(x) for x in v
            ]
        # For numpy types, convert to Python
        elif hasattr(v, "item"):
            out[k] = v.item()
        # For all else, try str
        else:
            try:
                json.dumps(v)
                out[k] = v
            except Exception:
                out[k] = str(v)
    return out


def compute_and_save_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, test_result_file: str
) -> None:
    """
    Compute and log metrics, save to results.

    Parameters
    ----------
    y_true : np.ndarray
      True binary labels.
    y_pred : np.ndarray
        Predicted labels.
    test_result_file : str
        Path to save results JSON.
    """
    acc = (y_true == y_pred).mean()
    f1 = f1_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    logger.info(f"Accuracy   : {acc:.3f}")
    logger.info(f"F1         : {f1:.3f}")
    logger.info(f"ROC AUC    : {roc:.3f}")
    logger.info(f"Precision  : {prec:.3f}")
    logger.info(f"Recall     : {rec:.3f}")
    results.append(
        {
            "model": model_name,
            "best_params": best_params,
            "accuracy": acc,
            "f1": f1,
            "roc_auc": roc,
            "precision": prec,
            "recall": rec,
        }
    )

    with open(test_result_file, "w") as f:
        json.dump(make_json_serializable(results[-1]), f, indent=2)

    logger.debug(f"Test results saved to {test_result_file}")


def build_target(labels: pd.Series) -> np.ndarray:
    """
    Build binary target from labels column.

    Parameters
    ----------
    labels : pd.Series
        Series with original labels.

    Returns
    -------
    np.ndarray
        Binary target array (0 benign, 1 attack).
    """
    # NaN -> benign (0), qualsiasi valore numerico -> attack (1)
    if labels is not None:
        return (~pd.isna(labels)).astype(int).values
    else:
        return np.zeros(len(labels), dtype=int)


def restore_estimator(e: dict | list | str) -> object:
    """
    Restore estimator from serialized format.

    Parameters
    ----------
    e : dict | list | str
        Serialized estimator (dict, list, or repr string).

    Returns
    -------
    object
        Restored estimator object.
    """
    # Restore from dict (old format)
    if isinstance(e, dict) and "class" in e and "params" in e:
        if e["class"] == "KMeans":
            from sklearn.cluster import KMeans

            return KMeans(**e["params"])
        if e["class"] == "LOF":
            return LOF(**e["params"])
        # Add more models if necessary
    # Restore from list (recursive)
    elif isinstance(e, list):
        return [restore_estimator(i) for i in e]
    # Restore from repr string (new format)
    elif isinstance(e, str) and e.startswith("LOF("):
        match = re.search(r"n_neighbors=(\d+)", e)
        n_neighbors = int(match.group(1)) if match else 20
        return LOF(n_neighbors=n_neighbors)
    return e


def validation_scorer(estimator: object, X_unused: pd.DataFrame) -> float:
    """
    Score always on the fixed validation set.

    Parameters
    ----------
    estimator : object
        The estimator to evaluate.
    X_unused : pd.DataFrame
        Unused, required by sklearn interface.

    Returns
    -------
    float
        ROC AUC score on validation set.
    """
    pred = estimator.predict(X_val)
    return roc_auc_score(y_val, pred)


# --- LIST OF CONFIGURATIONS ---
PARAM_GRID_MODELS = {
    # "ABOD": {
    #     "n_neighbors": [1, 3, 5, 10, 20, 35, 50, 100],
    #     "contamination": [0.05, 0.1, 0.2],
    # },
    # "HBOS": {"n_bins": [5, 10, 25, 50, 100], "contamination": [0.05, 0.1, 0.2]},
    "IForest": {
        "n_estimators": [25, 50, 100, 200],
        "max_samples": [0.1, 0.5, 0.7, 1.0],
        "max_features": [0.5, 0.75, 1.0],
        "random_state": [42],
        "contamination": [0.05, 0.1, 0.2],
    },
    # "KNN": {
    #     "n_neighbors": [3, 5, 11, 20, 35, 60],
    #     "method": ["largest", "mean", "median"],
    #     "contamination": [0.05, 0.1, 0.2],
    # },
    # "LOF": {"n_neighbors": [3, 5, 11, 20, 35, 50], "contamination": [0.05, 0.1, 0.2]},
    # "CBLOF": {
    #     "check_estimator": [False],
    #     "random_state": [42],
    #     "alpha": [0.1, 0.5, 0.9],
    #     "beta": [2, 4, 7, 10, 20],
    #     "clustering_estimator": [KMeans(n_clusters=2), KMeans(n_clusters=5)],
    #     "contamination": [0.05, 0.1, 0.2],
    # },
    # "FeatureBagging": {
    #     "base_estimator": [
    #         LOF(n_neighbors=5),
    #         LOF(n_neighbors=15),
    #         LOF(n_neighbors=35),
    #         LOF(n_neighbors=50),
    #     ],
    #     "random_state": [42],
    #     "contamination": [0.05, 0.1, 0.2],
    # },
    # "MCD": {"random_state": [42], "contamination": [0.05, 0.1, 0.2]},
    # "OCSVM": {
    #     "kernel": ["rbf", "linear", "sigmoid", "poly"],
    #     "gamma": [0.001, 0.01, 0.1, 1, "auto"],
    #     "nu": [0.05, 0.1, 0.2, 0.35],
    #     "contamination": [0.05, 0.1, 0.2],
    # },
    # "PCA": {
    #     "n_components": [1, 5, 10, 20, 35],
    #     "random_state": [42],
    #     "contamination": [0.05, 0.1, 0.2],
    # },
    # "LSCP": {
    #     "detector_list": [
    #         LOF(n_neighbors=5),
    #         LOF(n_neighbors=10),
    #         LOF(n_neighbors=15),
    #         LOF(n_neighbors=20),
    #         LOF(n_neighbors=25),
    #         LOF(n_neighbors=30),
    #         LOF(n_neighbors=35),
    #         LOF(n_neighbors=40),
    #         LOF(n_neighbors=45),
    #         LOF(n_neighbors=50),
    #     ],
    #     "random_state": [42],
    #     "contamination": [0.05, 0.1, 0.2],
    # },
    # "INNE": {
    #     "max_samples": [2, 10, 50],
    #     "contamination": [0.05, 0.1, 0.2],
    #     "random_state": [42],
    # },
    # "GMM": {
    #     "n_components": [1, 2, 5, 10, 20],
    #     "covariance_type": ["full", "tied", "diag", "spherical"],
    #     "random_state": [42],
    #     "contamination": [0.05, 0.1, 0.2],
    # },
    # "KDE": {
    #     "kernel": ["gaussian", "tophat", "epanechnikov", "exponential"],
    #     "bandwidth": [0.1, 0.5, 1, 2, 5],
    #     "contamination": [0.05, 0.1, 0.2],
    # },
    # "LMDD": {
    #     "random_state": [42],
    #     "contamination": [0.05, 0.1, 0.2],
    #     "sub_estimator": ["auto", 1, 2],
    # },
}

# ---------------------------
# CONFIG PATHS AND CONSTANTS
# ---------------------------
TRAIN_PATH = Path(__file__).parent.parent / "data/datasets/train_dataset.csv"
TEST_PATH = Path(__file__).parent.parent / "data/datasets/test_dataset.csv"
LABEL_COL = "ip.opt.time_stamp"
VAL_SIZE = 0.05  # 5% of the test set for validation

# Setup to save best params
BEST_PARAMS_PATH = (
    Path(__file__).parent.parent / "results/training_results/detector_best_params.json"
)
# Directory to save trained model and results
MODEL_DIR = Path(__file__).parent.parent / "data/trained_models"
RESULTS_DIR = Path(__file__).parent.parent / "results/training_results"


# --------------------------------------
# [Step 1] Load data and preprocess
# --------------------------------------
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

df_train = pd.read_csv(TRAIN_PATH, sep=";", low_memory=False)
df_test = pd.read_csv(TEST_PATH, sep=";", low_memory=False)

processor = Preprocessor()
df_train = processor.train(df_train, output_dir="tmp", skip_preprocess=False)
df_test = processor.test(df_test, output_dir="tmp", skip_preprocess=False)

# --------------------------------------------
# [Step 2] Build targets and split validation
# --------------------------------------------
X_tr = df_train.drop(columns=[LABEL_COL], errors="ignore")
X_ts = df_test.drop(columns=[LABEL_COL], errors="ignore")

y_ts_bin = build_target(df_test[LABEL_COL] if LABEL_COL in df_test else None)
logger.debug("Test set - (0 benign, 1 attack):", np.bincount(y_ts_bin))

# Split test into validation + final test
X_val, X_ts, y_val, y_ts = train_test_split(
    X_ts, y_ts_bin, test_size=1 - VAL_SIZE, stratify=y_ts_bin, random_state=42
)
logger.debug(
    f"Validation set size: {len(X_val)} - Test size finale: {len(X_ts)}"
)

# ---------------------------------------------
# [Step 2.1] Load best params cache if present
# ---------------------------------------------
if os.path.exists(BEST_PARAMS_PATH):
    with open(BEST_PARAMS_PATH, "r") as f:
        all_best_params = json.load(f)
else:
    all_best_params = {}

# --------------------------------------
# [Step 3] Train models with Grid Search
# --------------------------------------
results = []
for model_name, param_grid in PARAM_GRID_MODELS.items():
    # Mapping string name -> True class
    model_class = eval(model_name)
    logger.info(f"Model: {model_name}")
    # Custom paths per-model
    model_file = f"{MODEL_DIR}/{model_name}.pkl"
    test_result_file = f"{RESULTS_DIR}/{model_name}.json"

    # Skip if test results already exist
    if os.path.exists(test_result_file):
        logger.info(f"Test results detected for {model_name}, skipping to next model.")
        continue

    # If model already trained, load
    if os.path.exists(model_file):
        logger.info(f"Model for {model_name} already trained and saved. Loading...")
        try:
            base_detector = joblib.load(model_file)
            logger.debug(f"Loaded {model_name} from {model_file}")
            best_params = all_best_params[model_name]
            logger.debug(f"Best params loaded from cache: {best_params}")
        except Exception as e:
            raise ValueError(f"Model load failed for {model_name}: {e}")

        # Always perform the test & save results on the final test split
        y_pred = base_detector.predict(X_ts)
        compute_and_save_metrics(y_ts, y_pred, test_result_file)

    else:
        # Check best params cache, fast train
        if model_name in all_best_params:
            best_params = all_best_params[model_name]
            for k, v in best_params.items():
                if k in ["clustering_estimator", "base_estimator", "detector_list"]:
                    best_params[k] = restore_estimator(v)
                    logger.debug(f"Best params loaded from cache: {best_params}")
            base_detector = Detector(detector_class=model_class, **best_params)
            logger.info(f"Training {model_name} with cached best params...")
            base_detector.fit(X_tr, output_dir="tmp", skip_preprocess=True)
            y_pred = base_detector.predict(
                X_ts, output_dir="tmp", skip_preprocess=True
            )
            joblib.dump(base_detector, model_file)
            logger.debug(f"Model saved to: {model_file}")
            compute_and_save_metrics(y_ts, y_pred, test_result_file)

        else:
            base_detector = Detector(detector_class=model_class)
            grid = GridSearchCV(
                base_detector,
                param_grid,
                scoring=validation_scorer,
                cv=3,
                refit=True,
                n_jobs=4
            )
            logger.info(f"Performing Grid Search for {model_name}...")

            try:
                grid.fit(X_tr, output_dir="tmp", skip_preprocess=True)

                logger.info(f"Best params computed: {grid.best_params_}")
                all_best_params[model_name] = grid.best_params_
                serializable_params = {
                    m: make_json_serializable(p) for m, p in all_best_params.items()
                }
                with open(BEST_PARAMS_PATH, "w") as f:
                    json.dump(serializable_params, f, indent=2)

                y_pred = grid.best_estimator_.predict(
                    X_ts, output_dir="tmp", skip_preprocess=True
                )

                joblib.dump(grid.best_estimator_, model_file)
                logger.debug(f"Model saved to: {model_file}")
                compute_and_save_metrics(y_ts, y_pred, test_result_file)
            except Exception as e:
                logger.info(f"Grid Search failed for {model_name}: {e}")
                continue


