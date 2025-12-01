import json
import os
from pathlib import Path
import re
import sys

import joblib
import numpy as np
import pandas as pd
from pyod.models.hbos import HBOS
from pyod.models.lof import LOF
from sklearn.cluster import KMeans
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split

sys.path.append(str(Path(__file__).parent.parent))
from ml_models import Detector
from preprocessing import Preprocessor


def make_json_serializable(d):
    """Recursively convert non-serializable values to serializable ones."""
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


def build_target(labels):
    # NaN -> benign (0), qualsiasi valore numerico -> attack (1)
    if labels is not None:
        return (~pd.isna(labels)).astype(int).values
    else:
        return np.zeros(len(labels), dtype=int)


def restore_estimator(e):
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


# === DATA LOADING ===
TRAIN_PATH = (
    Path(__file__).parent.parent / "data/cleaned_datasets/dataset_training_benign.csv"
)
TEST_PATH = Path(__file__).parent.parent / "data/cleaned_datasets/dataset_3_cleaned.csv"
LABEL_COL = "ip.opt.time_stamp"

df_train = pd.read_csv(TRAIN_PATH, sep=";", low_memory=False)
df_test = pd.read_csv(TEST_PATH, sep=";", low_memory=False)
processor = Preprocessor()
df_train = processor.train(df_train, output_dir="tmp", skip_preprocess=False)
df_test = processor.test(df_test, output_dir="tmp", skip_preprocess=False)

y_test_bin = build_target(df_test[LABEL_COL] if LABEL_COL in df_test else None)
X_train = df_train.drop(columns=[LABEL_COL], errors="ignore")
X_test = df_test.drop(columns=[LABEL_COL], errors="ignore")
print("Test set - label binaria (0 benign, 1 attack):", np.bincount(y_test_bin))

VAL_SIZE = 0.05
X_val, X_final_test, y_val, y_final_test = train_test_split(
    X_test, y_test_bin, test_size=1 - VAL_SIZE, stratify=y_test_bin, random_state=42
)
print(f"Validation set size: {len(X_val)} - Test size finale: {len(X_final_test)}")

# === LIST OF CONFIGURATIONS ===
param_grid_models = {
    # "ABOD": {
    #     "n_neighbors": [1, 3, 5, 10, 20, 35, 50, 100],
    #     "contamination": [0.05, 0.1, 0.2],
    # },
    "HBOS": {"n_bins": [5, 10, 25, 50, 100], "contamination": [0.05, 0.1, 0.2]},
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
    "CBLOF": {
        "check_estimator": [False],
        "random_state": [42],
        "alpha": [0.1, 0.5, 0.9],
        "beta": [2, 4, 7, 10, 20],
        "clustering_estimator": [KMeans(n_clusters=2), KMeans(n_clusters=5)],
        "contamination": [0.05, 0.1, 0.2],
    },
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
    "OCSVM": {
        "kernel": ["rbf", "linear", "sigmoid", "poly"],
        "gamma": [0.001, 0.01, 0.1, 1, "auto"],
        "nu": [0.05, 0.1, 0.2, 0.35],
        "contamination": [0.05, 0.1, 0.2],
    },
    "PCA": {
        "n_components": [1, 5, 10, 20, 35],
        "random_state": [42],
        "contamination": [0.05, 0.1, 0.2],
    },
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
    "INNE": {
        "max_samples": [2, 10, 50],
        "contamination": [0.05, 0.1, 0.2],
        "random_state": [42],
    },
    "GMM": {
        "n_components": [1, 2, 5, 10, 20],
        "covariance_type": ["full", "tied", "diag", "spherical"],
        "random_state": [42],
        "contamination": [0.05, 0.1, 0.2],
    },
    "KDE": {
        "kernel": ["gaussian", "tophat", "epanechnikov", "exponential"],
        "bandwidth": [0.1, 0.5, 1, 2, 5],
        "contamination": [0.05, 0.1, 0.2],
    },
    # "LMDD": {
    #     "random_state": [42],
    #     "contamination": [0.05, 0.1, 0.2],
    #     "sub_estimator": ["auto", 1, 2],
    # },
}


def validation_scorer(estimator, X_unused):
    """Score always on the fixed validation set."""
    pred = estimator.predict(X_val)
    return roc_auc_score(y_val, pred)


# ---- setup per salvataggio best params ----
BEST_PARAMS_PATH = Path(__file__).parent.parent / "results/detector_best_params.json"
# ---- directory to save trained model & results ----
MODEL_DIR = Path(__file__).parent.parent / "trained_models"
RESULTS_DIR = Path(__file__).parent.parent / "results/training_results"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
# ---- load existing best params if available ----
if os.path.exists(BEST_PARAMS_PATH):
    with open(BEST_PARAMS_PATH, "r") as f:
        all_best_params = json.load(f)
else:
    all_best_params = {}

results = []
for model_name, param_grid in param_grid_models.items():
    # Mapping string nome -> classe vera
    model_class = eval(model_name)
    print(f"\n=== Model: {model_name} ===")

    # Custom paths per-model
    model_file = f"{MODEL_DIR}/{model_name}.pkl"
    test_result_file = f"{RESULTS_DIR}/{model_name}.json"
    # SKIP IF TEST RESULTS ARE ALREADY SAVED
    if os.path.exists(test_result_file):
        print(f"🟢 Test results detected for {model_name}, skipping to next model.")
        continue

    # If model already trained, load
    if os.path.exists(model_file):
        print(f"🔵 Model for {model_name} already trained and saved. Loading...")
        try:
            base_detector = joblib.load(model_file)
            print(f"Loaded {model_name} from {model_file}")
            best_params = all_best_params[model_name]
            print(f"✅ Best params loaded from cache: {best_params}")
        except Exception as e:
            print(f"Model load failed for {model_name}: {e}")
            continue  # Skip on error
        # Always perform the test & save results
        try:
            pred_test = base_detector.predict(X_test)
            acc = (y_test_bin == pred_test).mean()
            f1 = f1_score(y_test_bin, pred_test)
            roc = roc_auc_score(y_test_bin, pred_test)
            prec = precision_score(y_test_bin, pred_test)
            rec = recall_score(y_test_bin, pred_test)
            print(f"Accuracy   : {acc:.3f}")
            print(f"F1         : {f1:.3f}")
            print(f"ROC AUC    : {roc:.3f}")
            print(f"Precision  : {prec:.3f}")
            print(f"Recall     : {rec:.3f}")
            print(classification_report(y_test_bin, pred_test))
            res_row = {
                "model": model_name,
                "best_params": best_params,
                "accuracy": acc,
                "f1": f1,
                "roc_auc": roc,
                "precision": prec,
                "recall": rec,
            }
            results.append(res_row)
            with open(test_result_file, "w") as f:
                json.dump(make_json_serializable(res_row), f, indent=2)
            print(f"🟢 Test results saved to {test_result_file}")
        except Exception as e:
            print(f"⚠️ Failed to run test for {model_name}: {e}")
        continue  # Move to next model
    else:
        # ---- check best params cache ----
        if model_name in all_best_params:
            best_params = all_best_params[model_name]
            print(f"✅ Best params loaded from cache: {best_params}")
            for k, v in best_params.items():
                if k in ["clustering_estimator", "base_estimator", "detector_list"]:
                    best_params[k] = restore_estimator(v)
                    print(f"✅ Best params loaded from cache: {best_params}")
            base_detector = Detector(detector_class=model_class, **best_params)
            base_detector.fit(X_train, output_dir="tmp", skip_preprocess=True)
            pred_test = base_detector.predict(
                X_test, output_dir="tmp", skip_preprocess=True
            )
            cv_score = None
        else:
            base_detector = Detector(detector_class=model_class)
            grid = GridSearchCV(
                base_detector,
                param_grid,
                scoring=validation_scorer,
                cv=3,
                refit=True,
            )
            # y_train_dummy = np.zeros(len(X_train), dtype=int)
            grid.fit(X_train, output_dir="tmp", skip_preprocess=True)
            best_params = grid.best_params_
            print(f"✅ Best params computed: {best_params}")
            all_best_params[model_name] = best_params
            serializable_params = {
                m: make_json_serializable(p) for m, p in all_best_params.items()
            }
            with open(BEST_PARAMS_PATH, "w") as f:
                json.dump(serializable_params, f, indent=2)
            best_detector = grid.best_estimator_
            pred_test = best_detector.predict(
                X_test, output_dir="tmp", skip_preprocess=True
            )

        # SAVE THE TRAINED MODEL
        try:
            joblib.dump(base_detector, model_file)
            print(f"✅ Model saved to: {model_file}")
        except Exception as e:
            print(f"⚠️ Failed to save model {model_name}: {e}")

        acc = (y_test_bin == pred_test).mean()
        f1 = f1_score(y_test_bin, pred_test)
        roc = roc_auc_score(y_test_bin, pred_test)
        prec = precision_score(y_test_bin, pred_test)
        rec = recall_score(y_test_bin, pred_test)
        print(f"Accuracy   : {acc:.3f}")
        print(f"F1         : {f1:.3f}")
        print(f"ROC AUC    : {roc:.3f}")
        print(f"Precision  : {prec:.3f}")
        print(f"Recall     : {rec:.3f}")
        print(classification_report(y_test_bin, pred_test))
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
        # SAVE TEST RESULTS
        with open(test_result_file, "w") as f:
            json.dump(make_json_serializable(results[-1]), f, indent=2)
        print(f"🟢 Test results saved to {test_result_file}")
