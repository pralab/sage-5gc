import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from attack import add_noise
from ml_models import DetectionIsolationForest, DetectionKnn, DetectionRandomForest


def evaluate_robustness(
    detection,
    model,
    df_test: pd.DataFrame,
    noise_levels: list[float],
    distribution="normal",
    metric=accuracy_score,
) -> dict:
    """Return dict mapping noise_level → performance metric."""
    results = {}
    for nl in noise_levels:
        df_noisy = add_noise(
            df_test,
            noise_level=nl,
            cols = [
                "Chksum",
                "IP_Chksum",
                "IP_Flags",
                "IP_ID",
                "IP_IHL",
                "IP_TOS",
                "IP_TTL",
                "IP_Version",
                "TCP_Ack",
                "TCP_Dataofs",
                "TCP_Flags",
                "TCP_Seq",
                "TCP_Urgent",
                "TCP_Window",
                "dst_ip",
                "dst_port",
                "length",
                "protocol",
                "src_ip",
                "src_port",
                "z_score",
            ],
            distribution=distribution
        )
        y_test, y_pred = detection.run_predict(df_noisy, model)
        results[nl] = metric(y_test, y_pred)

    return results


if __name__ == "__main__":
    df_train_csv = "data/train_set_all.csv"
    df_test_csv = "data/test_set_all.csv"
    np.random.seed(42)

    # -----------------------------------------
    # [Step 1] Load training and test datasets
    # -----------------------------------------
    df_train = pd.read_csv(df_train_csv)
    sorted_columns = sorted(df_train.columns)
    df_train = df_train[sorted_columns]

    df_test = pd.read_csv(df_test_csv)
    sorted_columns = sorted(df_test.columns)
    df_test = df_test[sorted_columns]

    # -----------------------------------------
    # [Step 2] Load and prepare the model
    # -----------------------------------------
    detections = {
        "isolation_forest": DetectionIsolationForest(),
        "knn": DetectionKnn(),
        "random_forest": DetectionRandomForest()
    }

    models = {
        "isolation_forest": detections["isolation_forest"].run_train(df_train),
        "knn": detections["knn"].run_train(df_train),
        "random_forest": detections["random_forest"].run_train(df_train)
    }

    # -----------------------------------------
    # [Step 3] Robustness evaluation
    # -----------------------------------------
    results = {}

    for name, detection in detections.items():
        model = models[name]
        print(f"Evaluating robustness for {name}...")
        results[name] = evaluate_robustness(
            detection,
            model,
            df_test,
            noise_levels=[0.00, 0.01, 0.05, 0.10, 0.20, 0.50],
            distribution="normal",
            metric=accuracy_score,
        )

    for name in results:
        levels = sorted(results[name])
        scores = [results[name][n] for n in levels]
        plt.plot(levels, scores, marker='o', linestyle='-', label=name)

    plt.xlabel('Noise Level (fraction of σ)')
    plt.ylabel('Model Accuracy')
    plt.title('Robustness Curve')
    plt.xticks(levels, rotation=45)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.show()
