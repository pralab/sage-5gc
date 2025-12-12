from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

sys.path.append(str(Path(__file__).parent.parent))

from ml_models import Detector, EnsembleDetector
from preprocessing import Preprocessor

TRAIN_PATH = Path(__file__).parent.parent / "data/datasets/train_dataset.csv"
TEST_PATH = Path(__file__).parent.parent / "data/datasets/test_dataset.csv"
MODEL_DIR = Path(__file__).parent.parent / "data/trained_models"
LABEL_COL = "ip.opt.time_stamp"
VAL_SIZE = 0.50


def _convert_labels_to_binary(labels: pd.Series) -> np.ndarray:
    if labels is not None:
        return (~pd.isna(labels)).astype(int).values
    else:
        return np.zeros(len(labels), dtype=int)


# ----------------------------------
# [Step 1] Load and preprocess data
# ----------------------------------
df_test = pd.read_csv(TEST_PATH, sep=";")

X_ts = df_test.drop(columns=[LABEL_COL], errors="ignore")
y_ts = _convert_labels_to_binary(df_test[LABEL_COL] if LABEL_COL in df_test else None)

X_val, X_ts, y_val, y_ts = train_test_split(
    X_ts, y_ts, test_size=1 - VAL_SIZE, stratify=y_ts, random_state=42
)

processor = Preprocessor()
X_ts = processor.test(X_ts)
X_val = processor.test(X_val)

# -----------------------------
# [Step 2] Load base detectors
# -----------------------------
detector1: Detector = joblib.load(MODEL_DIR / "HBOS.pkl")
detector2: Detector = joblib.load(MODEL_DIR / "LOF.pkl")
detector3: Detector = joblib.load(MODEL_DIR / "GMM.pkl")
detector4: Detector = joblib.load(MODEL_DIR / "PCA.pkl")
#detector5: Detector = joblib.load(MODEL_DIR / "FeatureBagging.pkl")

# -------------------------------
# [Step 3] Train meta-classifier
# -------------------------------
print("Training meta-classifier...")
#meta_clf = LogisticRegression(random_state=42, class_weight="balanced")
meta_clf = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)

detector = EnsembleDetector(
    detectors=[detector1, detector2, detector3, detector4],
    meta_clf=meta_clf,
)
detector.fit(X_val, y_val, skip_preprocess=True)

# -------------------------------------
# [Step 4] Evaluate on the test set
# -------------------------------------
y_pred = detector.predict(X_ts, skip_preprocess=True)
y_scores = detector.decision_function(X_ts, skip_preprocess=True)

# Compute auc, f1, precision and recall
auc = roc_auc_score(y_ts, y_scores)
f1 = f1_score(y_ts, y_pred)
precision = precision_score(y_ts, y_pred)
recall = recall_score(y_ts, y_pred)

print("Ensemble Model Performance on Test Set:")
print(f"AUC: {auc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

detectors = [
    detector1,
    detector2,
    detector3,
    detector4,
    #detector5,
]
out_path = f"Ensemble_{meta_clf.__class__.__name__}_" + "_".join(
    [type(detector._detector).__name__ for detector in detectors]
)
print(f"Saving ensemble model as {out_path}.pkl")
joblib.dump(detector, MODEL_DIR / f"{out_path}.pkl")
