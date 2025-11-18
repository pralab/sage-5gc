import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, balanced_accuracy_score
)
import joblib


class DetectionRandomForest:
    def __init__(self):
        self.model = None
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None

    def load_train_data(self, df_train_csv):
        dtypes = {}
        sample = pd.read_csv(df_train_csv, sep=';', nrows=5)
        for c in sample.columns:
            if sample[c].dtype == 'float64':
                dtypes[c] = 'float32'
            elif sample[c].dtype == 'int64':
                dtypes[c] = 'int8'
            else:
                dtypes[c] = sample[c].dtype
        df = pd.read_csv(df_train_csv, sep=';', dtype=dtypes)
        df['ip.opt.time_stamp'] = df['ip.opt.time_stamp'].fillna(-1)
        print("Df train len:", len(df))
        cols = sorted(df.columns)
        df = df[cols]
        self.X_train = df.drop('ip.opt.time_stamp', axis=1)
        self.Y_train = df['ip.opt.time_stamp']
        print(len(self.X_train.columns))
        print(self.X_train.columns)

    def load_test_data(self, df_test_csv):
        dtypes = {}
        sample = pd.read_csv(df_test_csv, sep=';', nrows=5)
        for c in sample.columns:
            if sample[c].dtype == 'float64':
                dtypes[c] = 'float32'
            elif sample[c].dtype == 'int64':
                dtypes[c] = 'int8'
            else:
                dtypes[c] = sample[c].dtype
        df = pd.read_csv(df_test_csv, sep=';', dtype=dtypes)
        df['ip.opt.time_stamp'] = df['ip.opt.time_stamp'].fillna(-1)
        print("Df test len:", len(df))
        cols = sorted(df.columns)
        df = df[cols]
        self.X_test = df.drop('ip.opt.time_stamp', axis=1)
        self.Y_test = df['ip.opt.time_stamp']
        print(len(self.X_test.columns))
        print(self.X_test.columns)

    def train(self):
        if self.X_train is None:
            raise ValueError("Load the training data first")
        self.model = RandomForestClassifier(
            class_weight='balanced', max_depth=3, max_features='sqrt',
            min_samples_split=2, n_estimators=200, random_state=42
        )
        self.model.fit(self.X_train, self.Y_train)
        print("✓ RandomForest trained")

    def predict(self):
        if self.model is None:
            raise ValueError("Train or load the model first")
        if self.X_test is None:
            raise ValueError("Load the test data first")
        return self.model.predict(self.X_test)

    def save_model(self, path='random_forest.pkl'):
        joblib.dump(self.model, path)
        print(f"✓ Saved to {path}")

    def load_model(self, path='random_forest.pkl'):
        self.model = joblib.load(path)
        print(f"✓ Loaded from {path}")

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
        df_sorted['ip.opt.time_stamp'] = df_sorted['ip.opt.time_stamp'].fillna(-1)
        # Features
        X_test = df_sorted.drop('ip.opt.time_stamp', axis=1)
        # Labels
        y_test = df_sorted['ip.opt.time_stamp'].values
        # Predict
        y_pred = self.model.predict(X_test)
        return y_test, y_pred


def evaluate_predictions_rf(y_true, y_pred):
    labels = sorted(np.unique(list(y_true) + list(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Confusion Matrix (multi-classes):")
    print("Labels:", labels)
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=labels, digits=4, zero_division=0))
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    print("Balanced accuracy:", bal_acc)
    # Binary view (-1 vs others)
    y_true_bin = np.array([-1 if y == -1 else 1 for y in y_true])
    y_pred_bin = np.array([-1 if y == -1 else 1 for y in y_pred])
    cm2 = confusion_matrix(y_true_bin, y_pred_bin, labels=[1, -1])
    tp, fn, fp, tn = cm2[0, 0], cm2[0, 1], cm2[1, 0], cm2[1, 1]
    print("\nBinary Confusion Matrix (1=normal, -1=attack class -1):")
    print(cm2)
    print(f"TP: {tp} | FN: {fn} | FP: {fp} | TN: {tn}")
    return cm


def run_random_forest(detector: DetectionRandomForest, test_csv: str):
    print("\n" + "=" * 80)
    print(f"TEST DATASET (RF): {test_csv}")
    print("=" * 80)
    detector.load_test_data(test_csv)
    y_pred = detector.predict()
    print("\nUnique values in dans Y_test :", np.unique(detector.Y_test))
    print(pd.Series(detector.Y_test).value_counts())
    print("\nUnique values in Y_predit :", np.unique(y_pred))
    print(pd.Series(y_pred).value_counts())
    evaluate_predictions_rf(detector.Y_test, y_pred)