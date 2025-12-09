"""Script to train and evaluate multiple anomaly detection models using Grid Search."""

import json
import logging
import os
from pathlib import Path
import re
import sys

import joblib
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pyod.models.abod import ABOD  # noqa: F401
from pyod.models.cblof import CBLOF  # noqa: F401
from pyod.models.copod import COPOD  # noqa: F401
from pyod.models.ecod import ECOD  # noqa: F401
from pyod.models.feature_bagging import FeatureBagging  # noqa: F401
from pyod.models.gmm import GMM  # noqa: F401
from pyod.models.hbos import HBOS  # noqa: F401
from pyod.models.iforest import IForest  # noqa: F401
from pyod.models.inne import INNE  # noqa: F401
from pyod.models.kde import KDE  # noqa: F401
from pyod.models.knn import KNN  # noqa: F401
from pyod.models.lmdd import LMDD  # noqa: F401
from pyod.models.loda import LODA  # noqa: F401
from pyod.models.lof import LOF
from pyod.models.lscp import LSCP  # noqa: F401
from pyod.models.mcd import MCD  # noqa: F401
from pyod.models.ocsvm import OCSVM  # noqa: F401
from pyod.models.pca import PCA  # noqa: F401
from pyod.utils.example import visualize
import seaborn as sns
from sklearn.cluster import KMeans  # noqa: F401
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid, train_test_split

sys.path.append(str(Path(__file__).parent.parent))

from ml_models import Detector
from preprocessing import Preprocessor

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# region Configurations ---

# --- PATHS AND CONSTANTS ---
TRAIN_PATH = Path(__file__).parent.parent / "data/datasets/train_dataset.csv"
TEST_PATH = Path(__file__).parent.parent / "data/datasets/test_dataset.csv"
PTRAIN_PATH = Path(__file__).parent.parent / "data/datasets/train_processed.csv"
PTEST_PATH = Path(__file__).parent.parent / "data/datasets/test_processed.csv"
PVAL_PATH = Path(__file__).parent.parent / "data/datasets/val_processed.csv"

LABEL_COL = "ip.opt.time_stamp"
VAL_SIZE = 0.50

BEST_PARAMS_PATH = (
    Path(__file__).parent.parent / "results/training_results/detector_best_params.json"
)
# Directory to save trained model and results
MODEL_DIR = Path(__file__).parent.parent / "data/trained_models"
RESULTS_DIR = Path(__file__).parent.parent / "results/training_results"

# --- LIST OF CONFIGURATIONS ---
PARAM_GRID_MODELS = {
    "ABOD": {
        "n_neighbors": [1, 3, 5, 10, 20, 30, 50, 100],
        "contamination": [0.001, 0.01],
    },
    "HBOS": {
        "n_bins": [5, 10, 25, 50, 75],
        "tol": [0.1, 0.5],
        "contamination": [0.001, 0.01],
    },
    "GMM": {
        "n_components": [1, 2, 3, 4, 5, 10],
        "covariance_type": ["full", "diag", "tied"],
        "contamination": [0.001, 0.01],
        "random_state": [42],
    },
    "KNN": {
        "n_neighbors": [3, 5, 10, 20, 50],
        "method": ["largest", "mean", "median"],
        "contamination": [0.001, 0.01],
    },
    "LOF": {"n_neighbors": [3, 5, 11, 20, 35, 50], "contamination": [0.01, 0.001]},
    "IForest": {
        "n_estimators": [25, 50, 100, 200],
        "max_samples": [0.1, 0.5, 0.7, 1.0],
        "max_features": [0.5, 0.75, 1.0],
        "random_state": [42],
        "contamination": [0.01, 0.001],
    },
    "INNE": {
        "max_samples": [2, 10, 50],
        "random_state": [42],
        "contamination": [0.001, 0.01],
    },
    "FeatureBagging": {
        "n_estimators": [10, 20],
        "base_estimator": [
            LOF(n_neighbors=5),
            LOF(n_neighbors=15),
            LOF(n_neighbors=35),
            LOF(n_neighbors=50),
        ],
        "contamination": [0.001, 0.01],
    },
    "PCA": {
        "n_components": [1, 5, 10, 20, 35],
        "contamination": [0.001, 0.01],
        "random_state": [42],
    },
    # "OCSVM": {
    #     "kernel": ["rbf", "linear", "sigmoid"],
    #     "gamma": [0.001, 0.01, 0.1, 1, "auto"],
    #     "nu": [0.05, 0.1, 0.2, 0.35],
    #     "contamination": [0.01, 0.001],
    # },
    "COPOD": {"contamination": [0.001, 0.01, 0.05]},
    "ECOD": {"contamination": [0.001, 0.01]},
    "LODA": {
        "n_bins": [10, 20, 50],
        "n_random_cuts": [50, 100],
        "contamination": [0.001, 0.01],
    },
}

# endregion Configurations ---


def _make_json_serializable(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        # For sklearn objects (like KMeans) convert to repr string
        if hasattr(v, "get_params"):
            out[k] = {"class": v.__class__.__name__, "params": v.get_params()}
        elif isinstance(v, list):
            out[k] = [
                _make_json_serializable(x) if isinstance(x, dict) else str(x) for x in v
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


def _convert_labels_to_binary(labels: pd.Series) -> np.ndarray:
    # NaN -> benign (0), qualsiasi valore numerico -> attack (1)
    if labels is not None:
        return (~pd.isna(labels)).astype(int).values
    else:
        return np.zeros(len(labels), dtype=int)


def _restore_estimator(e: dict | list | str) -> object:
    # Restore from dict (old format)
    if isinstance(e, dict) and "class" in e and "params" in e:
        if e["class"] == "KMeans":
            from sklearn.cluster import KMeans

            return KMeans(**e["params"])
        if e["class"] == "LOF":
            return LOF(**e["params"])
    # Restore from list (recursive)
    elif isinstance(e, list):
        return [_restore_estimator(i) for i in e]
    # Restore from repr string (new format)
    elif isinstance(e, str) and e.startswith("LOF("):
        match = re.search(r"n_neighbors=(\d+)", e)
        n_neighbors = int(match.group(1)) if match else 20
        return LOF(n_neighbors=n_neighbors)
    return e


def _tune_threshold(y_scores: np.ndarray, y_true: np.ndarray, n_grid=100) -> float:
    # Generate candidate thresholds between min and max score
    taus = np.linspace(y_scores.min(), y_scores.max(), n_grid)

    best_tau = None
    best_f1 = -1.0

    for t in taus:
        y_pred = (y_scores >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = t

    return best_tau, best_f1


def _check_model_sanity(y_true, y_scores, model_name):
    plt.figure(figsize=(10, 6))

    sns.histplot(
        x=y_scores,
        hue=y_true,
        common_norm=False,
        stat="density",
        element="step",
        bins=50,
    )

    plt.title(f"Distribution Score {model_name}")
    plt.xlabel("Anomaly Score (MCD output)")
    plt.legend(title="Label", labels=["Outline (1)", "Normal (0)"])
    plt.savefig(f"{model_name}_score_distribution.png")


def _evaluate_single_config(
    model_class,
    params: dict,
    X_tr: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> tuple[float, dict]:
    try:
        detector = Detector(detector_class=model_class, **params)
        detector.fit(X_tr, PTRAIN_PATH, True)
        scores = detector.predict(X_val)
        auc = roc_auc_score(y_val, scores)
        return (auc, params)
    except Exception as e:
        logging.error(f"Error evaluating params {params}: {e}")
        return (-1, params)


if __name__ == "__main__":
    # --------------------------------------
    # [Step 1] Load data and preprocess
    # --------------------------------------
    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    df_train = pd.read_csv(TRAIN_PATH, sep=";")
    df_test = pd.read_csv(TEST_PATH, sep=";")

    # --------------------------------------------
    # [Step 2] Build targets and split validation
    # --------------------------------------------
    X_tr = df_train.drop(columns=[LABEL_COL], errors="ignore")
    X_ts = df_test.drop(columns=[LABEL_COL], errors="ignore")

    y_ts = _convert_labels_to_binary(
        df_test[LABEL_COL] if LABEL_COL in df_test else None
    )

    # Split test into validation + final test
    X_val, X_ts, y_val, y_ts = train_test_split(
        X_ts, y_ts, test_size=1 - VAL_SIZE, stratify=y_ts, random_state=42
    )

    # ---------------------------------------------
    # [Step 2.1] Preprocess train and test datasets
    # ---------------------------------------------
    logger.info("Preprocessing datasets...")
    processor = Preprocessor()
    X_tr = processor.train(X_tr, PTRAIN_PATH)
    X_ts = processor.test(X_ts, PTEST_PATH)

    # ---------------------------------------------
    # [Step 2.2] Load best params cache if present
    # ---------------------------------------------
    if os.path.exists(BEST_PARAMS_PATH):
        with open(BEST_PARAMS_PATH, "r") as f:
            all_best_params = json.load(f)
    else:
        all_best_params = {}

    # ----------------------
    # [Step 3] Train models
    # ----------------------
    results = []
    for model_name, param_grid in PARAM_GRID_MODELS.items():
        logger.info(f"Processing Model: {model_name}")

        model_file = MODEL_DIR / f"{model_name}.pkl"
        test_result_file = RESULTS_DIR / f"{model_name}.json"

        if test_result_file.exists():
            logger.info("Results already exist. Skipping.")
            continue

        best_params = None
        best_model_state = None

        # Check Cache
        if model_name in all_best_params:
            best_params = all_best_params[model_name]
            for k, v in best_params.items():
                if k in ["clustering_estimator", "base_estimator"]:
                    best_params[k] = _restore_estimator(v)
            logger.info("Best params loaded from cache.")

        # Manual Tuning if no cached params
        if best_params is None:
            logger.info(f"Starting Manual Tuning for {model_name}...")

            param_list = list(ParameterGrid(param_grid))
            results_parallel = Parallel(n_jobs=6, verbose=2)(
                delayed(_evaluate_single_config)(
                    eval(model_name), p, X_tr, X_val, y_val
                )
                for p in param_list
            )

            best_auc_score = -1
            best_params = None
            for score, params in results_parallel:
                if score > best_auc_score:
                    best_auc_score = score
                    best_params = params

            all_best_params[model_name] = best_params
            with BEST_PARAMS_PATH.open("w") as f:
                serializable = {
                    m: _make_json_serializable(p) for m, p in all_best_params.items()
                }
                json.dump(serializable, f, indent=4)

        # Final Training on Train Set with best params
        logger.info(f"Training final {model_name} model...")

        final_detector = Detector(detector_class=eval(model_name), **best_params)
        final_detector.fit(X_tr, PTRAIN_PATH, True)

        logger.info("Tuning threshold on validation set...")

        y_scores = final_detector.decision_function(X_val)
        best_thresh, _ = _tune_threshold(y_scores, y_val)

        # Final Evaluation on Test Set
        y_scores = final_detector.decision_function(X_ts, PTEST_PATH, True)
        #y_pred = final_detector.predict(X_ts, PTEST_PATH, True)
        y_pred = (y_scores > best_thresh).astype(int)

        prec = precision_score(y_ts, y_pred)
        rec = recall_score(y_ts, y_pred)
        f1 = f1_score(y_ts, y_pred)

        # logger.info(f"Best Threshold: {best_thresh:.5f}")
        logger.info(f"Precision: {prec:.3f}")
        logger.info(f"Recall: {rec:.3f}")
        logger.info(f"F1 Score: {f1:.3f}")

        results_entry = {
            "model": model_name,
            "best_params": best_params,
            "best_threshold": best_thresh,
            "roc_auc": roc_auc_score(y_ts, y_scores),
            "f1": f1,
            "precision": prec,
            "recall": rec,
        }

        with test_result_file.open("w") as f:
            json.dump(_make_json_serializable(results_entry), f, indent=4)

        _check_model_sanity(y_ts, y_scores, model_name)
        joblib.dump(final_detector, model_file)
