import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DetectionRandomForest:
    """RandomForest-based anomaly detector"""

    def __init__(self):
        self.model = None
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None

    def load_train_data(self, df_train_csv):
        """
        Load and preprocess the training dataset.

        Parameters
        ----------
        df_train_csv : str
            Path to training CSV file.
        """
        dtypes = {}
        sample = pd.read_csv(df_train_csv, sep=";", nrows=5)
        for c in sample.columns:
            if sample[c].dtype == "float64":
                dtypes[c] = "float32"
            elif sample[c].dtype == "int64":
                dtypes[c] = "int8"
            else:
                dtypes[c] = sample[c].dtype
        df = pd.read_csv(df_train_csv, sep=";", dtype=dtypes)
        df["ip.opt.time_stamp"] = df["ip.opt.time_stamp"].fillna(-1)
        logger.info(f"Df train len: {len(df)}")
        cols = sorted(df.columns)
        df = df[cols]
        self.X_train = df.drop("ip.opt.time_stamp", axis=1)
        self.Y_train = df["ip.opt.time_stamp"]
        logger.info(len(self.X_train.columns))
        logger.debug(self.X_train.columns)

    def load_test_data(self, df_test_csv):
        """
        Load and preprocess the test dataset.

        Parameters
        ----------
        df_test_csv : str
            Path to test CSV file.
        """
        dtypes = {}
        sample = pd.read_csv(df_test_csv, sep=";", nrows=5)
        for c in sample.columns:
            if sample[c].dtype == "float64":
                dtypes[c] = "float32"
            elif sample[c].dtype == "int64":
                dtypes[c] = "int8"
            else:
                dtypes[c] = sample[c].dtype
        df = pd.read_csv(df_test_csv, sep=";", dtype=dtypes)
        df["ip.opt.time_stamp"] = df["ip.opt.time_stamp"].fillna(-1)
        logger.info(f"Df test len: {len(df)}")
        cols = sorted(df.columns)
        df = df[cols]
        self.X_test = df.drop("ip.opt.time_stamp", axis=1)
        self.Y_test = df["ip.opt.time_stamp"]
        logger.info(len(self.X_test.columns))
        logger.debug(self.X_test.columns)

    def train(self):
        """
        Train RandomForest classifier.
        """
        if self.X_train is None:
            raise ValueError("Load the training data first")
        self.model = RandomForestClassifier(
            class_weight="balanced",
            max_depth=3,
            max_features="sqrt",
            min_samples_split=2,
            n_estimators=200,
            random_state=42,
        )
        self.model.fit(self.X_train, self.Y_train)
        logger.info("✓ RandomForest trained")

    def predict(self):
        """
        Predict using the loaded test dataset.

        Returns
        -------
        np.ndarray
            Predicted labels.
        """
        if self.model is None:
            raise ValueError("Train or load the model first")
        if self.X_test is None:
            raise ValueError("Load the test data first")
        return self.model.predict(self.X_test)

    def save_model(self, path="random_forest.pkl"):
        """
        Save KNN model.

        Parameters
        ----------
        path : str
            Output path.
        """
        joblib.dump(self.model, path)
        logger.info(f"✓ Saved to {path}")

    def load_model(self, path="random_forest.pkl"):
        """
        Load KNN model.

        Parameters
        ----------
        path : str
            Model checkpoint.
        """
        self.model = joblib.load(path)
        logger.info(f"✓ Loaded from {path}")

    def run_predict(self, df_test: pd.DataFrame) -> tuple:
        """
        Predict.
        Parameters
        ----------
        df_test : pd.DataFrame
            DataFrame including 'ip.opt.time_stamp'

        Returns
        -------
        tuple
        """
        # Sort columns alphabetically
        sorted_columns = sorted(df_test.columns)
        df_sorted = df_test[sorted_columns].copy()
        # Fill NaN with -1
        df_sorted["ip.opt.time_stamp"] = df_sorted["ip.opt.time_stamp"].fillna(-1)
        # Features
        X_test = df_sorted.drop("ip.opt.time_stamp", axis=1)
        # Labels
        y_test = df_sorted["ip.opt.time_stamp"].values
        # Predict
        y_pred = self.model.predict(X_test)
        return y_test, y_pred

    def get_score(self, df_pp: pd.DataFrame, sample_idx: int) -> float:
        """
        Continuous score for black-box attacks.
        Using probability of normal class (-1).

        Parameters
        ----------
        df_pp : pd.DataFrame
            Batch of preprocessed samples.
        sample_idx : int
            Sample index.

        Returns
        -------
        float
            Probability of class -1 (higher = more normal).
        """
        # Sort columns
        sorted_columns = sorted(df_pp.columns)
        df_sorted = df_pp[sorted_columns].copy()
        if "ip.opt.time_stamp" in df_sorted.columns:
            df_sorted["ip.opt.time_stamp"] = df_sorted["ip.opt.time_stamp"].fillna(-1)
            X = df_sorted.drop("ip.opt.time_stamp", axis=1)
        else:
            X = df_sorted

        proba = self.model.predict_proba(X)[sample_idx]
        classes = self.model.classes_

        try:
            idx_norm = np.where(classes == -1)[0][0]
        except IndexError:
            raise RuntimeError("Class -1 not present in model.classes_. ")

        return float(proba[idx_norm])


def evaluate_predictions_rf(y_true, y_pred):
    """
    Evaluate RandomForest predictions.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like

    Returns
    -------
    np.ndarray
        Confusion matrix.
    """
    labels = sorted(np.unique(list(y_true) + list(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    logger.info("Confusion Matrix (multi-class):")
    logger.info(f"Labels: {labels}")
    logger.info(f"\n{cm}")
    logger.info("\nClassification Report:")
    logger.info(
        classification_report(y_true, y_pred, labels=labels, digits=4, zero_division=0)
    )
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    logger.info(f"Balanced accuracy: {bal_acc:.6f}")
    # Binary view (-1 vs others)
    y_true_bin = np.array([-1 if y == -1 else 1 for y in y_true])
    y_pred_bin = np.array([-1 if y == -1 else 1 for y in y_pred])
    cm2 = confusion_matrix(y_true_bin, y_pred_bin, labels=[1, -1])
    tp, fn, fp, tn = cm2[0, 0], cm2[0, 1], cm2[1, 0], cm2[1, 1]
    logger.info("Binary Confusion Matrix (1=normal, -1=attack):")
    logger.info(f"\n{cm2}")
    logger.info(f"TP={tp} | FN={fn} | FP={fp} | TN={tn}")
    return cm


def run_random_forest(detector: DetectionRandomForest, test_csv: str):
    """
    Full RF evaluation pipeline.

    Parameters
    ----------
    detector : DetectionRandomForest
        Loaded RandomForest detector.
    test_csv : str
        Test CSV path.
    """
    detector.load_test_data(test_csv)
    y_pred = detector.predict()
    logger.info(f"Unique Y_test: {np.unique(detector.Y_test)}")
    logger.info(pd.Series(detector.Y_test).value_counts())
    logger.info(f"Unique Y_pred: {np.unique(y_pred)}")
    logger.info(pd.Series(y_pred).value_counts())
    evaluate_predictions_rf(detector.Y_test, y_pred)
