import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from pyod.utils.utility import standardizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Import Preprocessor and Detector
sys.path.append(str(Path(__file__).parent.parent))
import logging

from ml_models import Detector
from preprocessing import Preprocessor

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Same constants as in train_models.py
VAL_SIZE = 0.50
RANDOM_STATE = 42
MODEL_DIR = Path(__file__).parent.parent / "data/trained_models"


def _convert_labels_to_binary(labels: pd.Series) -> np.ndarray:
    """
    Converts labels to binary:  NaN -> benign (0), any numeric value -> attack (1)
    """
    if labels is not None:
        return (~pd.isna(labels)).astype(int).values
    else:
        return np.zeros(len(labels), dtype=int)

# ---------------------------------------------
# Attack map
# ---------------------------------------------
ATTACK_TYPE_MAP = {
    0: "flooding",
    1: "session_deletion",
    2: "session_modification",
    3: "nmap_scan",
    4: "reverse_shell",
    5: "upf_pdn0_fault",
    6: "restoration_teid",
}

# ---------------------------------------------
# EVALUATION FUNCTION
# ---------------------------------------------
def evaluate_models_by_category(
    detector: Detector, X_ts: pd.DataFrame, y_ts: pd.Series
) -> dict:
    """
    Evaluate detector performance for each attack category separately.

    Overall metrics:  Normal vs All Attacks (binary classification)
    Per-category metrics: Detection rate for that specific attack type

    Returns a dictionary with overall metrics and per-category metrics.
    """

    logger.debug("Computing anomaly scores...")
    # Get the threshold from the detector
    # if detector._threshold is not None:
    #     threshold = float(detector._threshold)
    # elif hasattr(detector._detector, "threshold_"):
    #     threshold = float(detector._detector.threshold_)
    # else:
    #     logger.warning("No threshold found in detector!  Using default one.")
    #     threshold = None

    results = {
        # "threshold": threshold,
        "overall_performance": {},
        "normal_condition": {},
        "attacks_by_category": {},
    }

    # --------------------------------------------
    # OVERALL PERFORMANCE:  Normal vs All Attacks
    # --------------------------------------------
    logger.info("OVERALL PERFORMANCE: Normal vs All Attacks")

    if isinstance(detector, Detector):
        y_scores = detector.decision_function(X_ts, skip_preprocess=True)
        y_pred = detector.predict(X_ts, skip_preprocess=True)
    else:
        detector1 = joblib.load(MODEL_DIR / "HBOS.pkl")
        detector2 = joblib.load(MODEL_DIR / "ABOD.pkl")
        scores1_ts = detector1.decision_function(X_ts)
        scores2_ts = detector2.decision_function(X_ts)

        s1_ts_n = standardizer(scores1_ts.reshape(-1, 1)).ravel()
        s2_ts_n = standardizer(scores2_ts.reshape(-1, 1)).ravel()
        X_meta_ts = np.vstack([s1_ts_n, s2_ts_n]).T

        y_scores = detector.decision_function(X_meta_ts)
        y_pred = detector.predict(X_meta_ts)

    y_ts_bin = _convert_labels_to_binary(y_ts)

    try:
        auc_overall = roc_auc_score(y_ts, y_scores)
    except Exception as e:
        logger.warning(f"Could not compute overall AUC: {e}")
        auc_overall = np.nan

    f1_overall = f1_score(y_ts_bin, y_pred, zero_division=0)
    precision_overall = precision_score(y_ts_bin, y_pred, zero_division=0)
    recall_overall = recall_score(y_ts_bin, y_pred, zero_division=0)
    accuracy_overall = accuracy_score(y_ts_bin, y_pred)

    # Confusion matrix elements
    tp_overall = int(np.sum((y_ts == 1) & (y_pred == 1)))
    fn_overall = int(np.sum((y_ts == 1) & (y_pred == 0)))
    fp_overall = int(np.sum((y_ts == 0) & (y_pred == 1)))
    tn_overall = int(np.sum((y_ts == 0) & (y_pred == 0)))

    n_attacks_total = int(y_ts.sum())
    n_normal_total = int(len(y_ts) - n_attacks_total)

    detection_rate_overall = tp_overall / n_attacks_total if n_attacks_total > 0 else 0
    fp_rate_overall = fp_overall / n_normal_total if n_normal_total > 0 else 0

    logger.info(
        f"Total samples: {len(y_ts)} (Normal: {n_normal_total}, Attacks: {n_attacks_total})"
    )
    logger.info(
        f"AUC          : {auc_overall:.4f}"
        if not np.isnan(auc_overall)
        else "AUC          : N/A"
    )
    logger.info(f"F1 Score     : {f1_overall:.4f}")
    logger.info(f"Precision    : {precision_overall:.4f}")
    logger.info(f"Recall       :  {recall_overall:.4f}")
    logger.info(f"Accuracy     : {accuracy_overall:.4f}")
    logger.info(f"Detection Rate: {detection_rate_overall:.4f}")
    logger.info(f"FP Rate      : {fp_rate_overall:.4f}")
    logger.info(f"TP/FN/FP/TN  : {tp_overall}/{fn_overall}/{fp_overall}/{tn_overall}")

    results["overall_performance"] = {
        "n_samples_total": len(y_ts),
        "n_normal": n_normal_total,
        "n_attacks": n_attacks_total,
        "auc": float(auc_overall) if not np.isnan(auc_overall) else None,
        "f1": float(f1_overall),
        "precision": float(precision_overall),
        "recall": float(recall_overall),
        "accuracy": float(accuracy_overall),
        "detection_rate": float(detection_rate_overall),
        "fp_rate": float(fp_rate_overall),
        "tp": tp_overall,
        "fn": fn_overall,
        "fp": fp_overall,
        "tn": tn_overall,
    }

    # ------------------------------------------------------
    # NORMAL CONDITION METRICS (only normal traffic)
    # ------------------------------------------------------
    mask_normal = y_ts.isna()
    n_normal = mask_normal.sum()

    if n_normal > 0:
        logger.info(f"NORMAL CONDITION ({n_normal} samples)")

        y_pred_normal = y_ts_bin[mask_normal]
        y_scores_normal = y_scores[mask_normal]

        correctly_classified_normal = int(np.sum(y_pred_normal == 0))
        misclassified_as_attack = int(np.sum(y_pred_normal == 1))

        normal_accuracy = correctly_classified_normal / n_normal

        mean_score_normal = float(np.mean(y_scores_normal))
        min_score_normal = float(np.min(y_scores_normal))
        max_score_normal = float(np.max(y_scores_normal))

        logger.info(
            f"Correctly classified as normal: {correctly_classified_normal}/{n_normal} ({normal_accuracy:.4f})"
        )
        logger.info(
            f"Misclassified as attack (FP): {misclassified_as_attack}/{n_normal}"
        )
        logger.info(
            f"Score range: [{min_score_normal:.4f}, {max_score_normal:.4f}], mean={mean_score_normal:.4f}"
        )

        results["normal_condition"] = {
            "n_samples": int(n_normal),
            "correctly_classified": correctly_classified_normal,
            "misclassified_as_attack": misclassified_as_attack,
            "accuracy": float(normal_accuracy),
            "score_mean": mean_score_normal,
            "score_min": min_score_normal,
            "score_max": max_score_normal,
        }
    else:
        logger.warning("[WARN] No normal samples found in test data")

    # ------------------------------------------------------
    # EVALUATE EACH ATTACK TYPE:  detection metrics
    # ------------------------------------------------------

    for attack_code, attack_name in ATTACK_TYPE_MAP.items():
        mask_attack = y_ts == attack_code
        n_attack = mask_attack.sum()

        if n_attack == 0:
            continue

        logger.info(f"{attack_name.upper()} (code {attack_code})")
        logger.info(f"Attack samples: {n_attack}")

        # Get scores and predictions for this attack
        y_scores_attack = y_scores[mask_attack]
        y_pred_attack = y_ts_bin[mask_attack]

        # Detection metrics
        correctly_detected = int(np.sum(y_pred_attack == 1))
        missed = int(np.sum(y_pred_attack == 0))

        detection_rate = correctly_detected / n_attack if n_attack > 0 else 0

        # Score statistics
        mean_score = float(np.mean(y_scores_attack))
        min_score = float(np.min(y_scores_attack))
        max_score = float(np.max(y_scores_attack))

        logger.info(
            f"Detection Rate: {correctly_detected}/{n_attack} ({detection_rate:.4f})"
        )
        logger.info(f"Missed: {missed}")
        logger.info(
            f"Score range: [{min_score:.4f}, {max_score:.4f}], mean={mean_score:.4f}"
        )

        # Compute AUC for this attack vs normal
        mask_normal = y_ts.isna()
        mask_combined = mask_attack | mask_normal

        if mask_normal.sum() > 0:
            y_true_binary = mask_attack[mask_combined].astype(int).values
            y_scores_binary = y_scores[mask_combined]

            try:
                auc_score = roc_auc_score(y_true_binary, y_scores_binary)
                logger.info(f"AUC vs normal: {auc_score:.4f}")
            except Exception as e:
                auc_score = None
        else:
            auc_score = None

        results["attacks_by_category"][attack_name] = {
            "attack_code": attack_code,
            "n_samples": int(n_attack),
            "correctly_detected": correctly_detected,
            "missed": missed,
            "detection_rate": float(detection_rate),
            "auc_vs_normal": float(auc_score) if auc_score is not None else None,
            "score_mean": mean_score,
            "score_min": min_score,
            "score_max": max_score,
        }

    return results


# ---------------------------------------------
# MAIN LOOP THROUGH ALL TRAINED MODELS
# ---------------------------------------------
if __name__ == "__main__":
    models_dir = Path(__file__).parent.parent / "data/trained_models"
    output_dir = Path(__file__).parent.parent / "results/category_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    test_data_path = Path(__file__).parent.parent / "data/datasets/test_dataset.csv"

    if not test_data_path.exists():
        logger.error(f"Test dataset not found at:  {test_data_path}")
        sys.exit(1)

    logger.debug(f"Loading test dataset from: {test_data_path}")
    df_test_full = pd.read_csv(test_data_path, sep=";", low_memory=False)
    logger.debug(
        f"Test dataset loaded:  {df_test_full.shape[0]} rows, {df_test_full.shape[1]} columns"
    )

    logger.debug(f"\nSplitting test set (VAL_SIZE={VAL_SIZE})...")
    X_test_full = df_test_full.drop(columns=["ip.opt.time_stamp"], errors="ignore")
    y_test_full = _convert_labels_to_binary(
        df_test_full["ip.opt.time_stamp"]
        if "ip.opt.time_stamp" in df_test_full
        else None
    )

    X_val, X_ts, y_val, y_ts = train_test_split(
        X_test_full,
        y_test_full,
        test_size=1 - VAL_SIZE,
        stratify=y_test_full,
        random_state=RANDOM_STATE,
    )

    logger.debug(f"Test set: {len(X_ts)} samples")

    processor = Preprocessor()
    X_ts = processor.test(X_ts)
    y_ts = df_test_full.loc[X_ts.index, "ip.opt.time_stamp"]

    if "ip.opt.time_stamp" in df_test_full.columns:
        logger.info("\nLabel distribution in FINAL TEST set:")
        logger.info(f"Normal (NaN): {y_ts.isna().sum()}")
        label_counts = y_ts.value_counts(dropna=True)
        for code, name in ATTACK_TYPE_MAP.items():
            count = label_counts.get(code, 0)
            if count > 0:
                logger.info(f"{name} ({code}): {count}")

    if not models_dir.exists():
        logger.error(f"Models directory not found at: {models_dir}")
        sys.exit(1)

    model_files = list(models_dir.glob("Ensemble_LogisticRegression.pkl"))
    if not model_files:
        logger.error(f"No model files (*.pkl) found in: {models_dir}")
        sys.exit(1)

    for model_path in model_files:
        model_name = model_path.stem
        logger.info(f"EVALUATING MODEL: {model_name}")

        try:
            detector: Detector = joblib.load(model_path)
            results = evaluate_models_by_category(detector, X_ts.copy(), y_ts)
            if not results:
                logger.warning(f"No results generated for {model_name}")
                continue

            results["model_name"] = model_name
            out_file = output_dir / f"{model_name}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            logger.info(f"\n[SAVED] Results written to: {out_file}")

        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    logger.info(f"Results saved in: {output_dir}")
