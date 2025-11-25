import os

import matplotlib
from matplotlib import pyplot as plt

matplotlib.use("TkAgg")
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


class DetectionIsolationForest:
    def __init__(self):
        """Initialize base attributes"""
        self.model = None
        self.X_train = None
        self.X_test = None
        self.Y_test = None
        self.df_test = None
        self.best_threshold = None

    def load_train_data(self, df_train_csv):
        """Load training data"""
        dtypes = {}
        sample = pd.read_csv(df_train_csv, sep=";", nrows=5)
        for col in sample.columns:
            if sample[col].dtype == "float64":
                dtypes[col] = "float32"
            elif sample[col].dtype == "int64":
                dtypes[col] = "int8"
            else:
                dtypes[col] = sample[col].dtype
        del sample

        df_train = pd.read_csv(df_train_csv, sep=";", dtype=dtypes)
        print("Df train len : ", len(df_train))
        sorted_columns = sorted(df_train.columns)
        self.X_train = df_train[sorted_columns].copy()
        del df_train
        if "ip.opt.time_stamp" in self.X_train.columns:
            self.X_train = self.X_train.drop("ip.opt.time_stamp", axis=1)
        print(len(self.X_train.columns))
        print(self.X_train.columns)

    def load_test_data(self, df_test_csv):
        """Load test data"""
        dtypes = {}
        sample = pd.read_csv(df_test_csv, sep=";", nrows=5)
        for col in sample.columns:
            if sample[col].dtype == "float64":
                dtypes[col] = "float32"
            elif sample[col].dtype == "int64":
                dtypes[col] = "int8"
            else:
                dtypes[col] = sample[col].dtype
        del sample

        df_test = pd.read_csv(df_test_csv, sep=";", dtype=dtypes)
        print("Df test len : ", len(df_test))
        sorted_columns = sorted(df_test.columns)
        self.df_test = df_test[sorted_columns].copy()
        self.X_test = self.df_test.drop("ip.opt.time_stamp", axis=1)
        self.df_test["anomaly"] = df_test.apply(self.tag_anomalies, axis=1)
        self.Y_test = self.df_test["anomaly"]
        print(len(self.X_test.columns))
        print(self.X_test.columns)

    def train(self):
        """Train the model"""
        if self.X_train is None:
            raise ValueError("Load training data first")

        self.model = IsolationForest(
            bootstrap=False,
            max_features=0.5,
            max_samples=1500,
            n_estimators=50,
            n_jobs=-1,
            random_state=42,
            verbose=0,
            warm_start=False,
        )
        self.model.fit(self.X_train)
        print("✓ Model trained")

    def predict(self):
        """Predict using raw model"""
        if self.model is None:
            raise ValueError("Train or load model first")
        if self.X_test is None:
            raise ValueError("Load test data first")

        return self.model.predict(self.X_test)

    def tag_anomalies(self, row):
        """Tag anomalies: -1=anomaly, 1=normal"""
        if row["ip.opt.time_stamp"] in range(0, 7):
            return -1  # Anomaly
        else:
            return 1  # Normal

    def classify_anomalies(self, row, prediction):
        """Classify anomaly type"""
        if prediction:
            if row["anomaly"] == 2 and row["predictions"] == -1:
                return 2
            elif row["anomaly"] == 3 and row["predictions"] == -1:
                return 3
            elif row["anomaly"] == 4 and row["predictions"] == -1:
                return 4
            elif row["anomaly"] == 5 and row["predictions"] == -1:
                return 5
            elif row["anomaly"] == 6 and row["predictions"] == -1:
                return 6
            elif row["anomaly"] == 7 and row["predictions"] == -1:
                return 7
            elif row["anomaly"] == 8 and row["predictions"] == -1:
                return 8
            elif row["anomaly"] == 9 and row["predictions"] == -1:
                return 9
            elif row["anomaly"] == 1 and row["predictions"] == 1:
                return 1
            else:
                return 0  # Misclassified
        else:
            if row["ip.opt.time_stamp"] == 0:
                return 2  # DoS
            elif row["ip.opt.time_stamp"] == 1:
                return 3  # deletion
            elif row["ip.opt.time_stamp"] == 2:
                return 4  # modification
            elif row["ip.opt.time_stamp"] == 3:
                return 5  # Nmap
            elif row["ip.opt.time_stamp"] == 4:
                return 6  # Reverse shell
            elif row["ip.opt.time_stamp"] == 5:
                return 7  # pdn type
            elif row["ip.opt.time_stamp"] == 6:
                return 8  # cve
            else:
                return 1  # Normal

    def save_model(self, path="isolation_forest.pkl"):
        """Save model and threshold"""
        joblib.dump({"model": self.model, "best_threshold": self.best_threshold}, path)
        print(f"✓ Saved to {path}")
        if self.best_threshold is not None:
            print(f"  Best threshold: {self.best_threshold:.6f}")

    def load_model(self, path="isolation_forest.pkl"):
        """Load model and threshold"""
        data = joblib.load(path)
        if isinstance(data, dict):
            self.model = data.get("model")
            self.best_threshold = data.get("best_threshold", None)
        else:
            self.model = data
            self.best_threshold = None
        print(f"✓ Loaded from {path}")
        if self.best_threshold is not None:
            print(f"  Best threshold: {self.best_threshold:.6f}")

    def run_predict(self, df_test: pd.DataFrame) -> tuple:
        """
        Predict using decision_function and best_threshold (if available).

        Parameters
        ----------
        df_test : pd.DataFrame
            DataFrame with column 'ip.opt.time_stamp' (label)

        Returns
        -------
        tuple
        """
        # Ordina colonne
        sorted_columns = sorted(df_test.columns)
        df_sorted = df_test[sorted_columns].copy()

        # Features
        X_test = df_sorted.drop("ip.opt.time_stamp", axis=1, errors="ignore")
        # Labels
        y_test = df_sorted.apply(self.tag_anomalies, axis=1).values
        if self.best_threshold is not None:
            scores = self.model.decision_function(X_test)
            y_pred_bin = (-scores >= self.best_threshold).astype(int)
            y_pred = np.where(y_pred_bin == 1, -1, 1)
        else:
            print("  ⚠️  WARNING: Using raw predict() without threshold optimization!")
            y_pred = self.model.predict(X_test)

        return y_test, y_pred


def evaluate_predictions_if(y_true, y_pred):
    """Evaluate predictions"""
    cm = confusion_matrix(y_true, y_pred, labels=[-1, 1])
    tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )
    print(f"Confusion Matrix:\n{cm}")
    print(f"TP: {tp} | FN: {fn} | FP: {fp} | TN: {tn}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall   : {recall:.6f}")
    print(f"F1 Score : {f1:.6f}")
    print(
        classification_report(y_true, y_pred, digits=4, labels=[-1, 1], zero_division=0)
    )
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    print("Balanced accuracy: ", bal_acc)
    return cm, precision, recall, f1


def run_isolation_forest(
    detector: DetectionIsolationForest,
    test_csv: str,
    save_plot: bool = True,
    show_plot: bool = False,
):
    """Complete evaluation using optimized threshold"""
    print("\n" + "=" * 80)
    print(f"TEST DATASET (IF): {test_csv}")
    print("=" * 80)
    # Load test data
    detector.load_test_data(test_csv)
    # Initial predictions (raw IF)
    Y_pred = detector.predict()
    print("\nUnique value in Y_test :", np.unique(detector.Y_test))
    print(pd.Series(detector.Y_test).value_counts())
    print("\nUnique value in Y_pred :", np.unique(Y_pred))
    print(pd.Series(Y_pred).value_counts())
    evaluate_predictions_if(detector.Y_test, Y_pred)
    # PR curve and threshold optimization
    scores = detector.model.decision_function(detector.X_test)
    y_true_bin = np.where(detector.Y_test == -1, 1, 0)  # 1 = anomalie, 0 = normal
    precision_raw, recall_raw, thresholds = precision_recall_curve(y_true_bin, -scores)
    ap = average_precision_score(y_true_bin, -scores)
    macro_precision_list, macro_recall_list, macro_f1_list = [], [], []
    for t in thresholds:
        y_pred = (-scores >= t).astype(int)
        macro_precision = precision_score(
            y_true_bin, y_pred, average="macro", zero_division=0
        )
        macro_recall = recall_score(
            y_true_bin, y_pred, average="macro", zero_division=0
        )
        macro_f1 = f1_score(y_true_bin, y_pred, average="macro", zero_division=0)
        macro_precision_list.append(macro_precision)
        macro_recall_list.append(macro_recall)
        macro_f1_list.append(macro_f1)

    best_index = int(np.argmax(macro_f1_list))
    best_threshold = float(thresholds[best_index])

    detector.best_threshold = best_threshold

    # Plot macro-averaged Precision/Recall
    plt.figure(figsize=(10, 6))
    plt.plot(
        thresholds,
        macro_precision_list,
        label="Macro Precision",
        linestyle="--",
        color="blue",
        linewidth=4,
    )
    plt.plot(
        thresholds, macro_recall_list, label="Macro Recall", color="orange", linewidth=4
    )
    plt.axvline(
        x=best_threshold,
        color="red",
        linestyle="--",
        linewidth=3,
        label=f"Optimal threshold = {best_threshold:.4f}",
    )
    plt.xlabel("Threshold", fontsize=40)
    plt.ylabel("Macro-averaged Score", fontsize=40)
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    plt.legend(fontsize=28, loc="upper right")
    plt.grid(True)

    base = os.path.splitext(os.path.basename(test_csv))[0]
    fig_out = f"Figure_iso_macro_{base}.png"
    if save_plot:
        plt.savefig(fig_out, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {fig_out}")
    if show_plot:
        plt.show()
    else:
        plt.close()

    print(f"Optimal threshold (macro F1 max): {best_threshold:.4f}")
    print(
        f"Macro Precision: {macro_precision_list[best_index]:.6f}, "
        f"Macro Recall: {macro_recall_list[best_index]:.6f}, "
        f"Macro F1: {macro_f1_list[best_index]:.6f}"
    )
    print(f"Average Precision (AUC-PR): {ap:.6f}")

    # Consistency check
    f1_scores_raw = (
        2
        * (precision_raw[:-1] * recall_raw[:-1])
        / (precision_raw[:-1] + recall_raw[:-1])
    )
    f1_scores_raw = np.nan_to_num(f1_scores_raw)
    print(f"Best threshold (F1 max) : {best_threshold:.4f}")
    print(
        f"Precision: {precision_raw[best_index]:.6f}, Recall: {recall_raw[best_index]:.6f}, F1: {f1_scores_raw[best_index]:.6f}"
    )

    # Apply best threshold
    y_pred_bin = (-scores >= best_threshold).astype(int)
    y_pred = np.where(y_pred_bin == 1, -1, 1)
    f1_check = f1_score(y_true_bin, y_pred_bin)
    print(
        f"\nF1 recalculated after threshold application (should match PR curve): {f1_check:.6f}"
    )
    evaluate_predictions_if(detector.Y_test, y_pred)

    # Remap 'anomaly' to multi-class BEFORE using prediction=True
    detector.df_test["anomaly"] = detector.df_test.apply(
        detector.classify_anomalies, axis=1, prediction=False
    )

    # Multi-class evaluation
    detector.df_test["predictions"] = y_pred
    detector.df_test["predictions"] = detector.df_test.apply(
        detector.classify_anomalies, axis=1, prediction=True
    )

    Y_test_classified = detector.df_test.apply(
        detector.classify_anomalies, axis=1, prediction=False
    )
    Y_pred_classified = detector.df_test["predictions"]

    best_cm = confusion_matrix(Y_test_classified, Y_pred_classified)
    print("\nMulti-classes Matrix :\n", best_cm)

    class_names = [
        "Miss",
        "Normal",
        "PFCP DoS",
        "PFCP Session Deletion",
        "PFCP Session Modifications",
        "Nmap",
        "Reverse shell",
        "Pdn type",
        "cve 2025 29646",
    ]

    precision_total = recall_total = f1_total = 0
    for i, class_name in enumerate(class_names):
        if class_name != "Miss":
            tp = best_cm[i, i]
            fp = best_cm[:, i].sum() - tp
            fn = best_cm[i, :].sum() - tp
            tn = best_cm.sum() - (tp + fp + fn)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            precision_total += prec
            recall_total += rec
            f1_total += f1
            print(f"\nClasse '{class_name}':")
            print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")
            print(
                f"Precision: {prec * 100:.2f}% | Recall: {rec * 100:.2f}% | F1: {f1 * 100:.2f}%"
            )

    n_classes = len(class_names) - 1
    print(f"\nAverage Precision: {(precision_total / n_classes) * 100:.2f}%")
    print(f"Average Recall   : {(recall_total / n_classes) * 100:.2f}%")
    print(f"Average F1-Score : {(f1_total / n_classes) * 100:.2f}%")

    return best_threshold
