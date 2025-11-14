"""
Robustness Evaluation Script

Testa la robustezza dei modelli aggiungendo noise ai dati di test.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score

from ml_models import (
    DetectionIsolationForest,
    DetectionRandomForest,
    DetectionKnn,
    DetectionAutoEncoder
)

from attack import add_noise

# ============================================================================
# CONFIGURAZIONE
# ============================================================================
MODELS_DIR = "trained_models"
TEST_CSV = "final_datasets/dataset_3_final.csv"
OUTPUT_DIR = "robustness_results"

# Livelli di noise da testare
NOISE_LEVELS = [0.00, 0.01, 0.05, 0.10, 0.20, 0.50]

# Tipo di distribuzione: 'normal' o 'uniform'
NOISE_DISTRIBUTION = "normal"

# Metrica principale da usare per i plot
METRIC = accuracy_score
METRIC_NAME = "Accuracy"

# ============================================================================
# FUNZIONE EVALUATE_ROBUSTNESS
# ============================================================================
def evaluate_robustness(
        detection,
        model,
        df_test: pd.DataFrame,
        noise_levels: list[float],
        distribution: str = "normal",
        metric=accuracy_score,
) -> dict:
    """
    Perform robustness evaluation of a model by adding noise to the test set.

    Parameters
    ----------
    detection : DetectionIsolationForest | DetectionKnn | DetectionRandomForest | DetectionAutoEncoder
        Detection wrapper that implements `run_predict(df, model)` and returns
        (y_test, y_pred).
    model : IsolationForest | KNeighborsClassifier | RandomForestClassifier | MLPAutoencoder
        Sklearn/PyTorch model to evaluate.
    df_test : pd.DataFrame
        DataFrame with test data (will not be modified in-place).
    noise_levels : list[float]
        List of noise levels to evaluate.
    distribution : str
        Type of noise distribution, either 'normal' or 'uniform'.
    metric : callable
        Function to compute the performance metric, must take (y_true, y_pred).

    Returns
    -------
    dict
        Dictionary mapping each noise level to the corresponding metric score.
    """
    results = {}
    for nl in noise_levels:
        print(f"  Noise level: {nl:.2f}", end=" ... ")
        # Aggiungi noise
        df_noisy = add_noise(df_test, noise_level=nl, distribution=distribution)
        # Predici
        y_test, y_pred = detection.run_predict(df_noisy)
        # Converti labels in formato binario uniforme: 0=normal, 1=anomaly/attack
        if hasattr(detection, '__class__') and 'IsolationForest' in detection.__class__.__name__:
            # IF: -1=anomaly, 1=normal → converti in 1=anomaly, 0=normal
            y_test_bin = np.where(y_test == -1, 1, 0)
            y_pred_bin = np.where(y_pred == -1, 1, 0)
        elif hasattr(detection, '__class__') and 'AutoEncoder' in detection.__class__.__name__:
            # AE: già {0, 1} dove 0=normal, 1=anomaly
            y_test_bin = y_test
            y_pred_bin = y_pred
        else:
            # RF/KNN: -1=normal, altri=attack → converti in 0=normal, 1=attack
            y_test_bin = np.where(y_test == -1, 0, 1)
            y_pred_bin = np.where(y_pred == -1, 0, 1)

        try:
            score = metric(y_test_bin, y_pred_bin)
            results[nl] = score
            print(f"Score: {score:.4f}")
        except Exception as e:
            print(f"ERROR: {e}")
            results[nl] = np.nan

    return results


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # ========================================================================
    # Carica dataset di test
    # ========================================================================
    print(f"📂 Loading test dataset: {TEST_CSV}")
    df_test = pd.read_csv(TEST_CSV, sep=';')
    print(f"   Loaded {len(df_test)} rows, {len(df_test.columns)} columns")
    # ========================================================================
    # Carica i modelli allenati
    # ========================================================================
    print("\n" + "=" * 80)
    print("LOADING MODELS")
    print("=" * 80)

    detections = {}
    models = {}

    # --- Isolation Forest ---
    print("\n🌲 Loading Isolation Forest...")
    if_det = DetectionIsolationForest()
    if_det.load_model(f"{MODELS_DIR}/isolation_forest.pkl")
    if_det.load_train_data('final_datasets/dataset_1_final.csv')
    detections['Isolation Forest'] = if_det
    models['Isolation Forest'] = if_det.model
    print("   ✓ Loaded")

    # --- Random Forest ---
    print("\n🌳 Loading Random Forest...")
    rf_det = DetectionRandomForest()
    rf_det.load_model(f"{MODELS_DIR}/random_forest.pkl")
    rf_det.load_train_data('final_datasets/dataset_2_final.csv')
    detections['Random Forest'] = rf_det
    models['Random Forest'] = rf_det.model
    print("   ✓ Loaded")

    # --- KNN ---
    print("\n🔍 Loading KNN...")
    knn_det = DetectionKnn()
    knn_det.load_model(f"{MODELS_DIR}/knn.pkl")
    knn_det.load_train_data('final_datasets/dataset_2_final.csv')
    detections['KNN'] = knn_det
    models['KNN'] = knn_det.model
    print("   ✓ Loaded")

    # --- Autoencoder ---
    print("\n🧠 Loading Autoencoder...")
    ae_det = DetectionAutoEncoder()
    ae_det.load_model(f"{MODELS_DIR}/autoencoder.pth")
    detections['Autoencoder'] = ae_det
    models['Autoencoder'] = ae_det.model
    print("   ✓ Loaded")

    # ========================================================================
    # Robustness Evaluation
    # ========================================================================
    print("\n" + "=" * 80)
    print("ROBUSTNESS EVALUATION")
    print("=" * 80)
    print(f"Noise levels: {NOISE_LEVELS}")
    print(f"Distribution: {NOISE_DISTRIBUTION}")
    print(f"Metric: {METRIC_NAME}")
    print("=" * 80 + "\n")

    results = {}

    for name, detection in detections.items():
        print(f"\n{'─' * 80}")
        print(f"Evaluating robustness for {name}...")
        print('─' * 80)
        model = models[name]
        results[name] = evaluate_robustness(
            detection,
            model,
            df_test,
            noise_levels=NOISE_LEVELS,
            distribution=NOISE_DISTRIBUTION,
            metric=METRIC,
        )
        print(f"\nResults for {name}:")
        for nl, score in results[name].items():
            print(f"  Noise {nl:.2f}: {score:.4f}")

    # ========================================================================
    # Plot Results
    # ========================================================================
    print("\n" + "=" * 80)
    print("PLOTTING RESULTS")
    print("=" * 80)

    plt.figure(figsize=(12, 7))

    for name in results:
        levels = sorted(results[name].keys())
        scores = [results[name][n] for n in levels]
        plt.plot(levels, scores, marker="o", linestyle="-", linewidth=2.5,
                 markersize=8, label=name)

    plt.xlabel("Noise Level (fraction of σ)", fontsize=14)
    plt.ylabel(f"Model {METRIC_NAME}", fontsize=14)
    plt.title(f"Robustness Evaluation - {METRIC_NAME} vs Noise Level", fontsize=16)
    plt.xticks(NOISE_LEVELS, rotation=0)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(title="Model", fontsize=12, loc='best')
    plt.tight_layout()

    # Salva plot
    plot_path = os.path.join(OUTPUT_DIR, f'robustness_{METRIC_NAME.lower()}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"📊 Saved plot: {plot_path}")
    plt.show()

    # ========================================================================
    # Save Results to CSV
    # ========================================================================
    df_results = pd.DataFrame(results, index=NOISE_LEVELS)
    df_results.index.name = 'Noise Level'
    csv_path = os.path.join(OUTPUT_DIR, f'robustness_{METRIC_NAME.lower()}_results.csv')
    df_results.to_csv(csv_path)
    print(f"💾 Saved results: {csv_path}")

    print("\n" + "✅" * 40)
    print("ROBUSTNESS EVALUATION COMPLETE!")
    print("✅" * 40)