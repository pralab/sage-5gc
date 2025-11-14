import os
import logging

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from attack import add_noise, perform_fingerprinting, perform_fingerprinting2, perform_fingerprinting3

# Import unified models API
from ml_models import DetectionIsolationForest, run_isolation_forest
from ml_models import DetectionRandomForest, run_random_forest
from ml_models import DetectionKnn, run_knn
from ml_models import DetectionAutoEncoder, run_autoencoder

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

TRAIN = True


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


if __name__ == "__main__":
    os.makedirs("trained_models", exist_ok=True)

    # Datasets
    train_unsup = 'final_datasets/dataset_1_final.csv'  # IF + AE
    train_sup = 'final_datasets/dataset_2_final.csv'  # RF + KNN
    test_csvs = ['final_datasets/dataset_3_final.csv']  # add more if needed

    # ========================================================================
    # ISOLATION FOREST
    # ========================================================================
    print("\n" + "🌲" * 40)
    print("ISOLATION FOREST")
    print("🌲" * 40)

    if_det = DetectionIsolationForest()

    if TRAIN:
        print("⚙️  Training Isolation Forest...")
        if_det.load_train_data(train_unsup)
        if_det.train()
        print("⚙️  Optimizing threshold...")
        ### ALLERT: Loro di base usano il test set per scegliere la soglia
        run_isolation_forest(if_det, test_csvs[0], save_plot=True, show_plot=False)
        if_det.save_model('trained_models/isolation_forest.pkl')
    else:
        print("📂 Loading pre-trained Isolation Forest...")
        if_det.load_model('trained_models/isolation_forest.pkl')
        if_det.load_train_data(train_unsup)
        for t in test_csvs:
            run_isolation_forest(if_det, t, save_plot=True, show_plot=False)

    # ========================================================================
    # RANDOM FOREST
    # ========================================================================
    print("\n" + "🌳" * 40)
    print("RANDOM FOREST")
    print("🌳" * 40)

    rf_det = DetectionRandomForest()

    if TRAIN:
        print("⚙️  Training Random Forest...")
        rf_det.load_train_data(train_sup)
        rf_det.train()
        rf_det.save_model('trained_models/random_forest.pkl')
    else:
        print("📂 Loading pre-trained Random Forest...")
        rf_det.load_model('trained_models/random_forest.pkl')
        # Carica train per avere X_train necessario per load_test_data
        rf_det.load_train_data(train_sup)

    for t in test_csvs:
        run_random_forest(rf_det, t)

    # ========================================================================
    # K-NEAREST NEIGHBORS
    # ========================================================================
    print("\n" + "🔍" * 40)
    print("K-NEAREST NEIGHBORS")
    print("🔍" * 40)

    knn_det = DetectionKnn()

    if TRAIN:
        print("⚙️  Training KNN...")
        knn_det.load_train_data(train_sup)
        knn_det.train()
        knn_det.save_model('trained_models/knn.pkl')
    else:
        print("📂 Loading pre-trained KNN...")
        knn_det.load_model('trained_models/knn.pkl')
        # Carica train per X_train
        knn_det.load_train_data(train_sup)

    for t in test_csvs:
        run_knn(knn_det, t)

    # ========================================================================
    # MLP AUTOENCODER
    # ========================================================================
    print("\n" + "🧠" * 40)
    print("MLP AUTOENCODER")
    print("🧠" * 40)

    ae_det = DetectionAutoEncoder()

    if TRAIN:
        print("⚙️  Training Autoencoder...")
        ae_det.load_train_data(train_unsup)
        ae_det.train_autoencoder(latent_dim=2, num_epochs=200, patience=10)
        print("⚙️  Optimizing threshold...")
        # ALLERT: Loro di base usano il test set per scegliere la soglia
        run_autoencoder(ae_det, test_csvs[0], enforce_percentile=None)
        ae_det.save_model('trained_models/autoencoder.pth')
    else:
        print("📂 Loading pre-trained Autoencoder...")
        ae_det.load_model('trained_models/autoencoder.pth')
        for t in test_csvs:
            run_autoencoder(ae_det, t, enforce_percentile=None)

    print("\n" + "✅" * 40)
    print("EVALUATION COMPLETE!")
    print("✅" * 40)