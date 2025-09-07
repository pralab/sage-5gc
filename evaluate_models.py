import logging

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from attack import add_noise
from ml_models import DetectionIsolationForest, DetectionKnn, DetectionRandomForest

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

TRAIN = False


def perform_fingerprinting(
    detection, model, df: pd.DataFrame, noise_level: float = 0.01
) -> tuple[list[str], np.ndarray]:
    """
    Create a simple fingerprint by perturbing one column at a time and
    recording how often the model's prediction changes compared to the
    unmodified dataframe.

    Parameters
    ----------
    detection: DetectionIsolationForest | DetectionKnn | DetectionRandomForest
        Detection wrapper that implements `run_predict(df, model)` and returns
        (y_test, y_pred).
    model: IsolationForest | KNeighborsClassifier | RandomForestClassifier
        Sklearn model to evaluate.
    df: pd.DataFrame
        DataFrame with data (will not be modified in-place).
    noise_level: float
        Noise level to use for numeric columns.

    Returns
    -------
    tuple[list[str], np.ndarray]
        Tuple containing a list of perturbed column names and a numpy array with
        the fraction of samples whose prediction changed after perturbing that column.

    Notes
    -----
    - Numeric features are perturbed using `add_noise` from `attack.py`.
    - Categorical features are perturbed by rotating category values (deterministic).
    """
    categorical_cols = [
        "Chksum",
        "IP_Chksum",
        "IP_Flags",
        "IP_ID",
        "IP_IHL",
        "IP_TOS",
        "IP_TTL",
        "IP_Version",
        "TCP_Dataofs",
        "TCP_Flags",
        "protocol",
        "src_port",
    ]

    numeric_cols = ["TCP_Ack", "TCP_Seq", "TCP_Urgent", "TCP_Window", "length"]

    cols_to_test = [c for c in categorical_cols + numeric_cols if c in df.columns]

    # Baseline predictions
    _, y_pred_baseline = detection.run_predict(df, model)
    y_pred_baseline = np.asarray(y_pred_baseline)

    sensitivities: list[float] = []
    for col in cols_to_test:
        df_mod = df.copy()

        # Numeric features
        if col in numeric_cols and col in df_mod.columns:
            df_mod = add_noise(
                df_mod, noise_level=noise_level, cols=[col], distribution="normal"
            )
        else:
            # Categorical features
            uniques = pd.unique(df_mod[col]).tolist()
            if len(uniques) <= 1:
                sensitivities.append(0.0)
                continue

            # For each row, pick a random value from the pool of uniques.
            choices = np.random.choice(uniques, size=len(df_mod))
            df_mod[col] = choices

        try:
            _, y_pred_mod = detection.run_predict(df_mod, model)
        except Exception as e:
            logging.warning(f"Prediction failed for column {col}: {e}")
            sensitivities.append(0.0)
            continue

        y_pred_mod = np.asarray(y_pred_mod)

        # Compute fraction of changed predictions
        if y_pred_mod.shape != y_pred_baseline.shape:
            sensitivities.append(0.0)
        else:
            changed_frac = float(np.mean(y_pred_mod != y_pred_baseline)) * 100.0
            sensitivities.append(changed_frac)

    return cols_to_test, np.asarray(sensitivities)


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

    # --------------------
    # [Step]
    # --------------------
    fig, axs = plt.subplots(1, 3, figsize=(10, 6))

    for name in detections.keys():
        column_names, changed = perform_fingerprinting(
            detections[name],
            models[name],
            df_test,
            noise_level=0.2,
        )

        logging.info(f"Fingerprinting results ({name.replace('_', ' ').capitalize()}):")
        for col, frac in zip(column_names, changed):
            logging.info(f"  Column: {col:15s}  Changed fraction: {frac:.2f} %")

        axs[list(detections.keys()).index(name)].barh(column_names, changed)
        axs[list(detections.keys()).index(name)].set_title(
            name.replace("_", " ").capitalize()
        )
        axs[list(detections.keys()).index(name)].set_xlabel("Changed fraction (%)")
        axs[list(detections.keys()).index(name)].set_ylabel("Column")
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
