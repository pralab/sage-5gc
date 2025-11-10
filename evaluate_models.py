import logging

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from attack import add_noise, perform_fingerprinting3
from ml_models import DetectionIsolationForest, DetectionKnn, DetectionRandomForest

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

TRAIN = False


def evaluate_robustness(
    detection,
    model,
    df_test: pd.DataFrame,
    noise_levels: list[float],
    distribution="normal",
    metric=accuracy_score,
) -> dict:
    """
    Perform robustness evaluation of a model by adding noise to the test set.

    Parameters
    ----------
    detection: DetectionIsolationForest | DetectionKnn | DetectionRandomForest
        Detection wrapper that implements `run_predict(df, model)` and returns
        (y_test, y_pred).
    model: IsolationForest | KNeighborsClassifier | RandomForestClassifier
        Sklearn model to evaluate.
    df_test: pd.DataFrame
        DataFrame with test data (will not be modified in-place).
    noise_levels: list[float]
        List of noise levels to evaluate.
    distribution: dict
        Type of noise distribution, either 'normal' or 'uniform'.
    metric: callable
        Function to compute the performance metric, must take (y_true, y_pred).

    Returns
    -------
    dict
        Dictionary mapping each noise level to the corresponding metric score.
    """
    results = {}
    for nl in noise_levels:
        df_noisy = add_noise(
            df_test,
            noise_level=nl,
            cols=[
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
            distribution=distribution,
        )
        y_test, y_pred = detection.run_predict(df_noisy, model)
        results[nl] = metric(y_test, y_pred)

    return results


def main() -> None:
    df_train_csv = "data/train_set_all.csv"
    df_test_csv = "data/test_set_all.csv"
    np.random.seed(42)

    # -----------------------------------------
    # [Step 1] Load training and test datasets
    # -----------------------------------------
    logging.info("Loading training and test datasets...")

    if TRAIN:
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
        "random_forest": DetectionRandomForest(),
    }

    if TRAIN:
        logging.info("Training models...")

        models = {
            "isolation_forest": detections["isolation_forest"].run_train(df_train),
            "knn": detections["knn"].run_train(df_train),
            "random_forest": detections["random_forest"].run_train(df_train),
        }

        logging.info("Saving trained models to 'trained_models/'...")

        for name in models:
            joblib.dump(models[name], f"trained_models/{name}.joblib")
    else:
        logging.info("Loading trained models from 'trained_models/'...")

        models = {
            "isolation_forest": joblib.load("trained_models/isolation_forest.joblib"),
            "knn": joblib.load("trained_models/knn.joblib"),
            "random_forest": joblib.load("trained_models/random_forest.joblib"),
        }

    # ------------------------
    # [Step 3] Fingerprinting
    # ------------------------
    logging.info("Performing fingerprinting evaluation...")

    _, axs = plt.subplots(1, 3, figsize=(14, 6))

    for name in detections.keys():
        column_names, changed = perform_fingerprinting3(
            detections[name],
            models[name],
            df_test,
            noise_level=0.2,
        )

        logging.info(f"Fingerprinting results ({name.replace('_', ' ').capitalize()}):")
        for col, frac in zip(column_names, changed):
            logging.info(f"  Column: {col:30s}  Changed fraction: {frac:.2f} %")

        axs[list(detections.keys()).index(name)].barh(column_names, changed)
        axs[list(detections.keys()).index(name)].set_title(
            name.replace("_", " ").capitalize()
        )
        axs[list(detections.keys()).index(name)].set_xlabel("Changed fraction (%)")
        axs[list(detections.keys()).index(name)].set_xlim(0, 100)
        axs[list(detections.keys()).index(name)].grid(
            axis="x", linestyle="--", alpha=0.6
        )

    plt.tight_layout()
    plt.savefig("fingerprinting_results.pdf", dpi=300)

    # -----------------------------------------
    # [Step 3] Robustness evaluation
    # -----------------------------------------
    # results = {}

    # for name, detection in detections.items():
    #     model = models[name]
    #     print(f"Evaluating robustness for {name}...")
    #     results[name] = evaluate_robustness(
    #         detection,
    #         model,
    #         df_test,
    #         noise_levels=[0.00, 0.01, 0.05, 0.10, 0.20, 0.50],
    #         distribution="normal",
    #         metric=accuracy_score,
    #     )

    # for name in results:
    #     levels = sorted(results[name])
    #     scores = [results[name][n] for n in levels]
    #     plt.plot(levels, scores, marker="o", linestyle="-", label=name)

    # plt.xlabel("Noise Level (fraction of σ)")
    # plt.ylabel("Model Accuracy")
    # plt.title("Robustness Curve")
    # plt.xticks(levels, rotation=45)
    # plt.grid(True, linestyle="--", alpha=0.6)
    # plt.legend(title="Model")
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()
