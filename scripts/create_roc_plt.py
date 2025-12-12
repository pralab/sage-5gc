"""
This script generates ROC curve plots.
- It loads the trained models and test dataset.
- It computes the ROC curve and AUC score comparison for each model.
"""

from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc, roc_curve

from ml_models import Detector
from preprocessing.preprocessor import Preprocessor

sys.path.append(str(Path(__file__).parent.parent))


def create_roc_curve_plot(model_path: str, test_data_path: str, output_dir: str):
    # Load the trained model
    with open(model_path, "rb") as f:
        detector: Detector = joblib.load(f)

    # Load and preprocess the test dataset
    df_test = pd.read_csv(test_data_path, sep=";", low_memory=False)
    preprocessor = Preprocessor()
    df_test_processed = preprocessor.test(df_test, output_dir=None)

    # Prepare features and labels
    X_test = df_test_processed.drop(columns=["ip.opt.time_stamp"], errors="ignore")
    y_true = df_test_processed["ip.opt.time_stamp"].notna().astype(int).values
    X_test = X_test.copy()
    X_test = X_test[sorted(X_test.columns)]

    # Get decision scores
    y_scores = detector._detector.decision_function(X_test.values)

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    roc_output_dir = Path(output_dir) / "roc_curve"
    roc_output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(roc_output_dir / "roc_curve.png")


def create_roc_curve_comparison_plot(
    model_paths: list, model_names: list, test_data_path: str, output_dir: str
):
    plt.figure(figsize=(8, 8))

    for model_path, model_name in zip(model_paths, model_names):
        # Load the trained model
        with open(model_path, "rb") as f:
            detector: Detector = joblib.load(f)

        # Load and preprocess the test dataset
        df_test = pd.read_csv(test_data_path, sep=";", low_memory=False)
        preprocessor = Preprocessor()
        df_test_processed = preprocessor.test(df_test, output_dir=None)

        # Prepare features and labels
        X_test = df_test_processed.drop(columns=["ip.opt.time_stamp"], errors="ignore")
        y_true = df_test_processed["ip.opt.time_stamp"].notna().astype(int).values
        X_test = X_test.copy()
        X_test = X_test[sorted(X_test.columns)]

        # Get decision scores
        y_scores = detector._detector.decision_function(X_test.values)

        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve logarithmic scale
        plt.plot(fpr, tpr, lw=2, label=f"{model_name} (area = {roc_auc:.2f})")

    plt.xlim([1e-4, 1.0])
    plt.ylim([0.4, 1.05])
    plt.xscale("log")
    plt.xlabel("False Positive Rate (log scale)")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic Comparison")
    plt.legend(loc="upper left")
    roc_output_dir = Path(output_dir)
    roc_output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(roc_output_dir / "roc_curve_comparison_log.png")


if __name__ == "__main__":
    model_paths = [
        Path(__file__).parent.parent / "data/trained_models/IForest.pkl",
        Path(__file__).parent.parent / "data/trained_models/HBOS.pkl",
        Path(__file__).parent.parent / "data/trained_models/GMM.pkl",
        Path(__file__).parent.parent / "data/trained_models/LOF.pkl",
        Path(__file__).parent.parent / "data/trained_models/OCSVM.pkl",
        Path(__file__).parent.parent / "data/trained_models/PCA.pkl",
        Path(__file__).parent.parent / "data/trained_models/CBLOF.pkl",
        Path(__file__).parent.parent / "data/trained_models/KNN.pkl",
        Path(__file__).parent.parent / "data/trained_models/ABOD.pkl",
        Path(__file__).parent.parent / "data/trained_models/KDE.pkl",
        Path(__file__).parent.parent / "data/trained_models/INNE.pkl",
        Path(__file__).parent.parent / "data/trained_models/MCD.pkl",
    ]
    model_names = [
        "IForest",
        "HBOS",
        "GMM",
        "LOF",
        "OCSVM",
        "PCA",
        "CBLOF",
        "KNN",
        "ABOD",
        "KDE",
        "INNE",
        "MCD",
    ]
    test_data_path = (
        Path(__file__).parent.parent / "data/datasets/test_dataset_filled.csv"
    )
    output_dir = Path(__file__).parent.parent / "results/figures"
    output_dir.mkdir(exist_ok=True)

    create_roc_curve_comparison_plot(
        model_paths, model_names, test_data_path, output_dir
    )
