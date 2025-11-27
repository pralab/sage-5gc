import logging
from pathlib import Path

from ml_models import (
    DetectionAutoEncoder,
    DetectionIsolationForest,
    DetectionKnn,
    DetectionRandomForest,
    run_autoencoder,
    run_isolation_forest,
    run_knn,
    run_random_forest,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

TRAIN = True
SAVE_DIR = Path(__file__).parent / "trained_models/"

if __name__ == "__main__":
    # Datasets
    train_unsup = (
        Path(__file__).parent / "data/final_datasets/dataset_1_final.csv"
    )  # train set IF + AE
    train_sup = (
        Path(__file__).parent / "data/final_datasets/dataset_2_final.csv"
    )  # train set RF + KNN
    test_csvs = [
        Path(__file__).parent / "data/final_datasets/dataset_3_final.csv"
    ]  # test sets

    # ========================================================================
    # ISOLATION FOREST
    # ========================================================================
    logger.info("ISOLATION FOREST")

    if_det = DetectionIsolationForest()

    if TRAIN:
        logger.info("   Training Isolation Forest...")
        if_det.load_train_data(train_unsup)
        if_det.train()
        logger.info("   Optimizing threshold...")
        run_isolation_forest(if_det, test_csvs[0], save_plot=False, show_plot=False)
        if_det.save_model(SAVE_DIR / "isolation_forest.pkl")
    else:
        logger.info("  Loading pre-trained Isolation Forest...")
        if_det.load_model(SAVE_DIR / "isolation_forest.pkl")
        if_det.load_train_data(train_unsup)
        for t in test_csvs:
            run_isolation_forest(if_det, t, save_plot=False, show_plot=False)

    # ========================================================================
    # RANDOM FOREST
    # ========================================================================
    logger.info("RANDOM FOREST")
    rf_det = DetectionRandomForest()

    if TRAIN:
        logger.info("    Training Random Forest...")
        rf_det.load_train_data(train_sup)
        rf_det.train()
        rf_det.save_model(SAVE_DIR / "random_forest.pkl")
    else:
        logger.info("   Loading pre-trained Random Forest...")
        rf_det.load_model(SAVE_DIR / "random_forest.pkl")
        rf_det.load_train_data(train_sup)

    for t in test_csvs:
        run_random_forest(rf_det, t)

    # ========================================================================
    # K-NEAREST NEIGHBORS
    # ========================================================================
    logger.info("K-NEAREST NEIGHBORS")
    knn_det = DetectionKnn()

    if TRAIN:
        logger.info("   Training KNN...")
        knn_det.load_train_data(train_sup)
        knn_det.train()
        knn_det.save_model(SAVE_DIR / "knn.pkl")
    else:
        logger.info("   Loading pre-trained KNN...")
        knn_det.load_model(SAVE_DIR / "knn.pkl")
        knn_det.load_train_data(train_sup)

    for t in test_csvs:
        run_knn(knn_det, t)

    # ========================================================================
    # MLP AUTOENCODER
    # ========================================================================
    logger.info("MLP AUTOENCODER")
    ae_det = DetectionAutoEncoder()

    if TRAIN:
        logger.info("   Training Autoencoder...")
        ae_det.load_train_data(train_unsup)
        ae_det.train_autoencoder(latent_dim=2, num_epochs=200, patience=10)
        logger.info("   Optimizing threshold...")
        run_autoencoder(ae_det, test_csvs[0], enforce_percentile=None)
        ae_det.save_model(SAVE_DIR / "autoencoder.pth")
    else:
        logger.info("  Loading pre-trained Autoencoder...")
        ae_det.load_model(SAVE_DIR / "autoencoder.pth")
        for t in test_csvs:
            run_autoencoder(ae_det, t, enforce_percentile=None)

    logger.info("EVALUATION COMPLETE!")
