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
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))


def _convert_labels_to_binary(labels: pd.Series) -> np.ndarray:
    if labels is not None:
        return (~pd.isna(labels)).astype(int).values
    else:
        return np.zeros(len(labels), dtype=int)


def create_roc_curve_comparison_plot(
        model_paths: list, model_names: list, test_data_path: str, output_dir: str
):
    plt.figure(figsize=(10, 8))

    colors = [
         '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
         '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
         '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
         '#98df8a', '#ff9896', '#c5b0d5', '#c49c94'
    ]

    # Load and preprocess the test dataset
    df_test = pd.read_csv(test_data_path, sep=";", low_memory=False)
    X_test = df_test.drop(columns=["ip.opt.time_stamp"], errors="ignore")
    y_true = _convert_labels_to_binary(
        df_test["ip.opt.time_stamp"] if "ip.opt.time_stamp" in df_test else None
    )
    preprocessor = Preprocessor()
    X_test = preprocessor.test(X_test)

    for idx, (model_path, model_name) in enumerate(zip(model_paths, model_names)):
        # Load the trained model
        with open(model_path, "rb") as f:
            detector: Detector = joblib.load(f)

        # Prepare features and labels
        X_test = X_test.copy()
        X_test = X_test[sorted(X_test.columns)]

        # Get decision scores
        y_scores = detector.decision_function(X_test, skip_preprocess=True)

        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve logarithmic scale with unique color
        plt.plot(
            fpr,
            tpr,
            lw=2,
            color=colors[idx],
            label=f"{model_name} (auc={roc_auc:.3f})"
        )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    #plt.xscale("log")
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title("ROC Curve Comparison", fontsize=18, pad=16)
    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.08),
        fontsize=11.5,
        ncol=4,
        frameon=False
    )
    plt.grid(True, alpha=0.3)

    roc_output_dir = Path(output_dir)
    roc_output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(roc_output_dir / "roc_curve_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to {roc_output_dir / 'roc_curve_comparison.png'}")


if __name__ == "__main__":
    model_paths = [
        Path(__file__).parent.parent / "data/trained_models/ABOD.pkl",
        Path(__file__).parent.parent / "data/trained_models/COPOD.pkl",
        Path(__file__).parent.parent / "data/trained_models/ECOD.pkl",
        Path(__file__).parent.parent / "data/trained_models/FeatureBagging.pkl",
        Path(__file__).parent.parent / "data/trained_models/GMM.pkl",
        Path(__file__).parent.parent / "data/trained_models/HBOS.pkl",
        Path(__file__).parent.parent / "data/trained_models/IForest.pkl",
        Path(__file__).parent.parent / "data/trained_models/INNE.pkl",
        Path(__file__).parent.parent / "data/trained_models/KNN.pkl",
        Path(__file__).parent.parent / "data/trained_models/LODA.pkl",
        Path(__file__).parent.parent / "data/trained_models/LOF.pkl",
        Path(__file__).parent.parent / "data/trained_models/PCA.pkl",
        # Ensemble
        Path(__file__).parent.parent / "data/trained_models/Ensemble_SVC_C10_G10_HBOS_KNN_ABOD_INNE_PCA.pkl",
        Path(__file__).parent.parent / "data/trained_models/Ensemble_SVC_C10_G10_HBOS_KNN_GMM_INNE_PCA.pkl",
        Path(__file__).parent.parent / "data/trained_models/Ensemble_SVC_C10_G10_HBOS_KNN_LOF_INNE_PCA.pkl",
        Path(__file__).parent.parent / "data/trained_models/Ensemble_SVC_C100_G100_HBOS_KNN_LOF_INNE_FeatureBagging.pkl",

    ]
    model_names = [
        "ABOD",
        "COPOD",
        "ECOD",
        "FeatureBagging",
        "GMM",
        "HBOS",
        "IForest",
        "INNE",
        "KNN",
        "LODA",
        "LOF",
        "PCA",
        # Ensemble
        "Ens-HKAIP",
        "Ens-HKGIP",
        "Ens-HKLIP",
        "Ens-HKLIF",
    ]
    test_data_path = (
        Path(__file__).parent.parent / "data/datasets/test_dataset.csv"
    )
    output_dir = Path(__file__).parent.parent / "results/figures"
    output_dir.mkdir(exist_ok=True)

    create_roc_curve_comparison_plot(
        model_paths, model_names, test_data_path, output_dir
    )
