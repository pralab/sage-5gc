from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from pyod.utils.utility import standardizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).parent.parent))

from ml_models import Detector
from preprocessing import Preprocessor

TRAIN_PATH = Path(__file__).parent.parent / "data/datasets/train_dataset.csv"
TEST_PATH = Path(__file__).parent.parent / "data/datasets/test_dataset.csv"
MODEL_DIR = Path(__file__).parent.parent / "data/trained_models"
LABEL_COL = "ip.opt.time_stamp"
VAL_SIZE = 0.50


def _convert_labels_to_binary(labels: pd.Series) -> np.ndarray:
    # NaN -> benign (0), qualsiasi valore numerico -> attack (1)
    if labels is not None:
        return (~pd.isna(labels)).astype(int).values
    else:
        return np.zeros(len(labels), dtype=int)


# ----------------------------------
# [Step 1] Load and preprocess data
# ----------------------------------
df_test = pd.read_csv(TEST_PATH, sep=";")

X_ts = df_test.drop(columns=[LABEL_COL], errors="ignore")
y_ts = _convert_labels_to_binary(
    df_test[LABEL_COL] if LABEL_COL in df_test else None
)

X_val, X_ts, y_val, y_ts = train_test_split(
    X_ts, y_ts, test_size=1 - VAL_SIZE, stratify=y_ts, random_state=42
)

processor = Preprocessor()
X_ts = processor.test(X_ts)
X_val = processor.test(X_val)

# -----------------------------
# [Step 2] Load base detectors
# -----------------------------
detector1: Detector= joblib.load(MODEL_DIR / "IForest.pkl")
detector2: Detector = joblib.load(MODEL_DIR / "KNN.pkl")
detector3: Detector = joblib.load(MODEL_DIR / "LOF.pkl")
detector4: Detector = joblib.load(MODEL_DIR / "INNE.pkl")
detector5: Detector = joblib.load(MODEL_DIR / "PCA.pkl")

detector1.set_threshold(None)
detector2.set_threshold(None)
detector3.set_threshold(None)
detector4.set_threshold(None)
detector5.set_threshold(None)

# --------------------------------
# [Step 3] Generate meta-features
# --------------------------------
print("Generating meta-features...")
scores1_tr = detector1.decision_function(X_val)
scores2_tr = detector2.decision_function(X_val)
scores3_tr = detector3.decision_function(X_val)
scores4_tr = detector4.decision_function(X_val)
scores5_tr = detector5.decision_function(X_val)

s1_tr_n = standardizer(scores1_tr.reshape(-1, 1)).ravel()
s2_tr_n = standardizer(scores2_tr.reshape(-1, 1)).ravel()
s3_tr_n = standardizer(scores3_tr.reshape(-1, 1)).ravel()
s4_tr_n = standardizer(scores4_tr.reshape(-1, 1)).ravel()
s5_tr_n = standardizer(scores5_tr.reshape(-1, 1)).ravel()

# shape (n_samples, 2)
X_meta_tr = np.vstack([s1_tr_n, s2_tr_n, s3_tr_n, s4_tr_n, s5_tr_n]).T

# -------------------------------
# [Step 4] Train meta-classifier
# -------------------------------
print("Training meta-classifier...")
meta_clf = LogisticRegression(random_state=42, class_weight="balanced")
meta_clf.fit(X_meta_tr, y_val)

# -------------------------------------
# [Step 5] Evaluate on the test set
# -------------------------------------
scores1_ts = detector1.decision_function(X_ts)
scores2_ts = detector2.decision_function(X_ts)
scores3_ts = detector3.decision_function(X_ts)
scores4_ts = detector4.decision_function(X_ts)
scores5_ts = detector5.decision_function(X_ts)

s1_ts_n = standardizer(scores1_ts.reshape(-1, 1)).ravel()
s2_ts_n = standardizer(scores2_ts.reshape(-1, 1)).ravel()
s3_ts_n = standardizer(scores3_ts.reshape(-1, 1)).ravel()
s4_ts_n = standardizer(scores4_ts.reshape(-1, 1)).ravel()
s5_ts_n = standardizer(scores5_ts.reshape(-1, 1)).ravel()

X_meta_ts = np.vstack([s1_ts_n, s2_ts_n, s3_ts_n, s4_ts_n, s5_ts_n]).T
y_pred = meta_clf.predict(X_meta_ts)
y_scores = meta_clf.predict_proba(X_meta_ts)[:, 1]

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

joblib.dump(meta_clf, MODEL_DIR / "Ensemble_LogisticRegression.pkl")
