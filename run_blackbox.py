import json
import logging
import os
from pathlib import Path
from typing import Dict

import pandas as pd
from proto_features_mapping import get_protocol_feature_mapping
from thales_project.blackbox_algo import blackbox_attack

from ml_models import (
    DetectionAutoEncoder,
    DetectionIsolationForest,
    DetectionKnn,
    DetectionRandomForest,
)
from modifiable_features_fingerprinting import MODIFIABLE_FEATURES

TEST_CSV = "data/cleaned_datasets/dataset_3_cleaned.csv"
MODELS_DIR = "trained_models"
OUTPUT_DIR = "results"
BUDGET = 100

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def read_csv(path: str) -> pd.DataFrame:
    """
    Read a CSV file trying different encodings.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame.
    """
    logger.info(f"Loading dataset: {path}")

    try:
        ds=  pd.read_csv(path, sep=";", low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        ds = pd.read_csv(path, sep=";", low_memory=False, encoding="latin-1")

    logger.info(f"Loaded {len(ds)} rows and {len(ds.columns)} columns")

    return ds


def load_models() -> Dict[str, object]:
    """"""
    models = {}

    # Isolation Forest
    try:
        if_det = DetectionIsolationForest()
        if_det.load_model(str(Path(MODELS_DIR) / "isolation_forest.pkl"))
        models["IsolationForest"] = if_det
        logger.info("Loaded IsolationForest")
    except Exception as e:
        logger.warning(f"Failed to load IsolationForest: {e}")

    # Random Forest
    try:
        rf_det = DetectionRandomForest()
        rf_det.load_model(str(Path(MODELS_DIR) / "random_forest.pkl"))
        models["RandomForest"] = rf_det
        logger.info("Loaded RandomForest")
    except Exception as e:
        logger.warning(f"Failed to load RandomForest: {e}")

    # k‑Nearest Neighbours
    try:
        knn_det = DetectionKnn()
        knn_det.load_model(str(Path(MODELS_DIR) / "knn.pkl"))
        models["KNN"] = knn_det
        logger.info("Loaded KNN")
    except Exception as e:
        logger.warning(f"Failed to load KNN: {e}")

    # Autoencoder
    try:
        ae_det = DetectionAutoEncoder()
        ae_det.load_model(str(Path(MODELS_DIR) / "autoencoder.pth"))
        models["Autoencoder"] = ae_det
        logger.info("Loaded Autoencoder")
    except Exception as e:
        logger.warning(f"Failed to load Autoencoder: {e}")

    return models


def main() -> None:

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    raw_ds = read_csv(TEST_CSV)

    modifiable_features = list(MODIFIABLE_FEATURES)
    # Compute protocol-specific feature constraints: for each ip.proto value,
    # determine which modifiable features have at least one non-null value.
    try:
        protocol_constraints = get_protocol_feature_mapping(
            df,
            modifiable_features,
            protocol_column="ip.proto",
        )
    except Exception as e:
        logger.warning(f"Could not compute protocol constraints: {e}")
        protocol_constraints = None

    models = load_models()
    if not models:
        logger.error("No models loaded; aborting attack.")
        return

    optimizer_type = "es"  # Evolution Strategy
    os.makedirs(f"{OUTPUT_DIR}/{optimizer_type}", exist_ok=True)

    results = {}
    for model_name, detector in models.items():
        logger.info(f"\n=== Attacking {model_name} ===")
        try:
            best_params, improvement, perturbed_df = blackbox_attack(

            )
        except Exception as e:
            logger.error(f"Black‑box attack failed for {model_name}: {e}")
            continue

































    #     logger.info(
    #         f"Best improvement for {model_name}: {improvement:.2f} percentage points"
    #     )
    #     results[model_name] = {
    #         "best_params": best_params,
    #         "improvement": improvement,
    #     }

    #     perturbed_path = (
    #         Path(OUTPUT_DIR) / f"{optimizer_type}/perturbed_{model_name}.csv"
    #     )
    #     try:
    #         perturbed_df.to_csv(
    #             perturbed_path, sep=";", index=False, encoding="latin-1"
    #         )
    #         results[model_name]["perturbed_csv"] = str(perturbed_path)
    #         logger.info(f"Perturbed dataset saved to {perturbed_path}")
    #     except Exception as e:
    #         logger.warning(f"Could not save perturbed dataset for {model_name}: {e}")

    #     result_json_path = (
    #         Path(OUTPUT_DIR) / f"{optimizer_type}/result_{model_name}.json"
    #     )
    #     try:
    #         with open(result_json_path, "w") as f:
    #             json.dump(results[model_name], f, indent=2)
    #         logger.info(f"Result JSON saved to {result_json_path}")
    #     except Exception as e:
    #         logger.warning(f"Could not save JSON results for {model_name}: {e}")

    # consolidated_path = Path(OUTPUT_DIR) / f"{optimizer_type}/results_all_models.json"
    # try:
    #     with open(consolidated_path, "w") as f:
    #         json.dump(results, f, indent=2)
    #     logger.info(f"Consolidated results saved to {consolidated_path}")
    # except Exception as e:
    #     logger.warning(f"Could not save consolidated results: {e}")


if __name__ == "__main__":
    main()
