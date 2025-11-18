import os
import logging

# Import unified models API
from ml_models import DetectionIsolationForest, run_isolation_forest
from ml_models import DetectionRandomForest, run_random_forest
from ml_models import DetectionKnn, run_knn
from ml_models import DetectionAutoEncoder, run_autoencoder

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

TRAIN = True


if __name__ == "__main__":
    os.makedirs("trained_models", exist_ok=True)

    # Datasets
    train_unsup = 'final_datasets/dataset_1_final.csv'  # train set IF + AE
    train_sup = 'final_datasets/dataset_2_final.csv'  # train set RF + KNN
    test_csvs = ['final_datasets/dataset_3_final.csv']  # test sets

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