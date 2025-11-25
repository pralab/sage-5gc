from imblearn.over_sampling import SMOTE
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
)
from sklearn.neighbors import KNeighborsClassifier


class DetectionKnn:
    def __init__(self):
        self.model = None
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None

    def load_train_data(self, df_train_csv):
        dtypes = {}
        sample = pd.read_csv(df_train_csv, sep=";", nrows=5)
        for c in sample.columns:
            if sample[c].dtype == "float64":
                dtypes[c] = "float32"
            elif sample[c].dtype == "int64":
                dtypes[c] = "int8"
            else:
                dtypes[c] = sample[c].dtype
        df = pd.read_csv(df_train_csv, sep=";")
        df["ip.opt.time_stamp"] = df["ip.opt.time_stamp"].fillna(-1)
        print("Df train len:", len(df))
        cols = sorted(df.columns)
        df = df[cols]
        self.X_train = df.drop("ip.opt.time_stamp", axis=1)
        self.Y_train = df["ip.opt.time_stamp"]
        print(df.describe())
        print(len(self.X_train.columns))
        print(self.X_train.columns)

    def load_test_data(self, df_test_csv):
        df = pd.read_csv(df_test_csv, sep=";")
        df["ip.opt.time_stamp"] = df["ip.opt.time_stamp"].fillna(-1)
        print("Df test len:", len(df))
        print(df.describe())
        cols = sorted(df.columns)
        df = df[cols]
        self.X_test = df.drop("ip.opt.time_stamp", axis=1)
        self.Y_test = df["ip.opt.time_stamp"]
        print(len(self.X_test.columns))
        print(self.X_test.columns)

    def train(self):
        if self.X_train is None:
            raise ValueError("Load training data first")
        self.model = KNeighborsClassifier(
            algorithm="auto", metric="manhattan", n_neighbors=3, p=1, weights="uniform"
        )
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(self.X_train, self.Y_train)
        self.model.fit(X_res, y_res)
        print("✓ KNN trained (SMOTE)")

    def predict(self):
        if self.model is None:
            raise ValueError("Train or load the model")
        if self.X_test is None:
            raise ValueError("Load test data first")
        return self.model.predict(self.X_test)

    def save_model(self, path="knn.pkl"):
        joblib.dump(self.model, path)
        print(f"✓ Saved to {path}")

    def load_model(self, path="knn.pkl"):
        self.model = joblib.load(path)
        print(f"✓ Loaded from {path}")

    def mse_train_test(self):
        if self.model is None:
            raise ValueError("Trained model is needed!")
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)
        return (
            mean_squared_error(self.Y_train, y_pred_train),
            mean_squared_error(self.Y_test, y_pred_test),
        )

    def run_predict(self, df_test: pd.DataFrame) -> tuple:
        """
        Predict.

        Parameters
        ----------
        df_test : pd.DataFrame
            DataFrame with column 'ip.opt.time_stamp' (label)

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


def evaluate_predictions_knn(y_true, y_pred):
    labels = sorted(np.unique(list(y_true) + list(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Confusion Matrix (multi-classes):")
    print("Labels:", labels)
    print(cm)
    print("\nClassification Report:")
    print(
        classification_report(y_true, y_pred, labels=labels, digits=4, zero_division=0)
    )
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    print("Balanced accuracy:", bal_acc)
    return cm


def run_knn(detector: DetectionKnn, test_csv: str):
    print("\n" + "=" * 80)
    print(f"TEST DATASET (KNN): {test_csv}")
    print("=" * 80)
    detector.load_test_data(test_csv)
    y_pred = detector.predict()
    print("\nUnique value in Y_test :", np.unique(detector.Y_test))
    print(pd.Series(detector.Y_test).value_counts())
    print("\nUnique value in  Y_predit :", np.unique(y_pred))
    print(pd.Series(y_pred).value_counts())
    evaluate_predictions_knn(detector.Y_test, y_pred)
    try:
        y_pred_train = detector.model.predict(detector.X_train)
        from sklearn.metrics import mean_squared_error

        train_mse = mean_squared_error(detector.Y_train, y_pred_train)
        test_mse = mean_squared_error(detector.Y_test, y_pred)
        print(f"MSE Train: {train_mse:.6f} | MSE Test: {test_mse:.6f}")
    except Exception:
        pass
