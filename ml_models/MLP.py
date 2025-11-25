import os
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision.ops import MLP
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, f1_score,
    precision_score, recall_score, accuracy_score, balanced_accuracy_score,
)
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class MLPAutoencoder(nn.Module):
    """
    Simple MLP-based Autoencoder with symmetrical encoder/decoder.
    """
    def __init__(self, input_dim: int, latent_dim: int = 8, p_drop: float = 0.1):
        super().__init__()
        logger.debug(f"Initializing Autoencoder with latent_dim={latent_dim}")
        self.encoder = MLP(
            in_channels=input_dim,
            hidden_channels=[64, 16, latent_dim],
            activation_layer=nn.LeakyReLU,
            dropout=p_drop
        )
        self.enc_bn = nn.BatchNorm1d(latent_dim)
        self.decoder = MLP(
            in_channels=latent_dim,
            hidden_channels=[16, 64, input_dim],
            activation_layer=nn.LeakyReLU,
            dropout=p_drop
        )

    def forward(self, x):
        """
        Forward pass of the autoencoder.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Reconstructed batch with the same shape as the input.
        """
        # Encode (latent space)
        z = self.encoder(x)
        z = self.enc_bn(z)
        # Decode (reconstruction)
        x_hat = self.decoder(z)
        return x_hat


class DetectionAutoEncoder:
    """Wrapper for training/evaluating an MLP Autoencoder for anomaly detection."""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # State
        self.model = None
        self.input_dim = None
        self.threshold = None
        self.errors = None
        # Cache
        self.X_train = None
        self.X_test = None
        self.y_true_bin = None

    def load_train_data(self, df_train_csv: str):
        """
        Load training dataset.

        Parameters
        ----------
        df_train_csv : str
            Path to training CSV.
        """
        df_train = pd.read_csv(df_train_csv, sep=';')
        cols_to_drop = ["ip.opt.time_stamp", "frame.number"]
        X_train_df = df_train.drop(columns=cols_to_drop, errors="ignore").select_dtypes(include="number")
        self.X_train = X_train_df.values
        self.input_dim = X_train_df.shape[1]
        logger.info(len(X_train_df.columns))

    def load_test_data(self, df_test_csv: str):
        """
        Load test dataset and build binary labels.

        Parameters
        ----------
        df_test_csv : str
            Path to test CSV.
        """
        df_test = pd.read_csv(df_test_csv, sep=';')
        cols_to_drop = ["ip.opt.time_stamp", "frame.number"]
        X_test_df = df_test.drop(columns=cols_to_drop, errors="ignore").select_dtypes(include="number")
        self.X_test = X_test_df.values
        y_test_raw = df_test["ip.opt.time_stamp"]
        y_test = y_test_raw.fillna(-1).astype(int).values
        self.y_true_bin = (y_test != -1).astype(int)
        logger.info(len(X_test_df.columns))

    def train_autoencoder(self, lr=3e-4, num_epochs=200, patience=10, sparsity_coef=1e-4, latent_dim=2):
        """
        Train the MLP autoencoder with early stopping and sparsity regularization.

        Parameters
        ----------
        lr : float, optional
            Learning rate for the Adam optimizer. Default is 3e-4.
        num_epochs : int, optional
            Maximum number of training epochs. Default is 200.
        patience : int, optional
            Early stopping patience on validation loss. Default is 10.
        sparsity_coef : float, optional
            Weight for L1 sparsity penalty on the latent representation. Default is 1e-4.
        latent_dim : int, optional
            Dimensionality of the latent space. Default is 2.

        Returns
        -------
        None
            The trained model is stored internally in `self.model`.
        """
        if self.X_train is None or self.input_dim is None:
            raise ValueError("Call load_train_data() first")
        # Tensors
        X_train_tensor = torch.tensor(self.X_train, dtype=torch.float32)
        # DataLoaders
        full_train_ds = TensorDataset(X_train_tensor)
        val_ratio = 0.1
        val_size = max(1, int(len(full_train_ds) * val_ratio))
        train_size = len(full_train_ds) - val_size
        train_ds, val_ds = random_split(full_train_ds, [train_size, val_size])
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
        # Model
        self.model = MLPAutoencoder(input_dim=self.input_dim, latent_dim=latent_dim, p_drop=0.1).to(self.device)
        criterion = nn.L1Loss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        best_val = float("inf")
        bad = 0
        best_state = None
        for epoch in range(1, num_epochs + 1):
            # --- Training ---
            self.model.train()
            train_loss_sum, n_train = 0.0, 0
            for (x_batch,) in train_loader:
                x_batch = x_batch.to(self.device)
                optimizer.zero_grad()
                x_hat = self.model(x_batch)
                # Sparsity
                z = self.model.enc_bn(self.model.encoder(x_batch))
                loss = criterion(x_hat, x_batch) + sparsity_coef * z.abs().mean()
                loss.backward()
                optimizer.step()
                bs = x_batch.size(0)
                train_loss_sum += loss.item() * bs
                n_train += bs
            train_loss = train_loss_sum / max(1, n_train)

            self.model.eval()
            val_loss_sum, n_val = 0.0, 0
            with torch.no_grad():
                for (x_val,) in val_loader:
                    x_val = x_val.to(self.device)
                    x_hat = self.model(x_val)
                    z = self.model.enc_bn(self.model.encoder(x_val))
                    vloss = criterion(x_hat, x_val) + sparsity_coef * z.abs().mean()
                    bs = x_val.size(0)
                    val_loss_sum += vloss.item() * bs
                    n_val += bs
            val_loss = val_loss_sum / max(1, n_val)
            logger.info(f"Epoch [{epoch}/{num_epochs}] - train_loss: {train_loss:.6f} - val_loss: {val_loss:.6f}")
            # Early stopping
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                bad = 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    logger.info("Early stopping.")
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        logger.info("✓ Autoencoder trained")

    def compute_errors(self):
        """
        Compute reconstruction errors for all test samples.

        Computes the mean absolute reconstruction error for each row in
        `self.X_test` and stores the resulting array in `self.errors`.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Results stored in `self.errors` (np.ndarray of shape [n_samples]).
        """
        if self.X_test is None:
            raise ValueError("Call load_test_data() first")
        if self.model is None:
            raise ValueError("Call train_autoencoder() first")

        self.model.eval()
        X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32)
        test_loader = DataLoader(TensorDataset(X_test_tensor), batch_size=64, shuffle=False)
        reconstruction_errors = []
        with torch.no_grad():
            for (x_batch,) in test_loader:
                x_batch = x_batch.to(self.device)
                x_hat = self.model(x_batch)
                e = (x_hat - x_batch).abs().mean(dim=1)
                reconstruction_errors.extend(e.detach().cpu().numpy())

        errors = np.asarray(reconstruction_errors)
        self.errors = np.nan_to_num(errors, posinf=np.finfo(np.float32).max)

    def optimize_threshold(self, enforce_percentile=None):
        """
        Select an optimal anomaly threshold based on reconstruction errors.

        Parameters
        ----------
        enforce_percentile : float or None, optional
            If provided, forces the threshold to be the percentile of the
            reconstruction errors for normal samples. If None, selects the
            percentile that maximizes the F1-score.

        Returns
        -------
        float
            The chosen anomaly threshold.
        """
        if self.errors is None:
            self.compute_errors()
        if self.y_true_bin is None:
            raise ValueError("Call load_test_data() first")
        normal_errors = self.errors[self.y_true_bin == 0]
        if normal_errors.size == 0:
            normal_errors = self.errors

        candidates = [90, 92, 94, 95, 96, 97, 98, 98.5, 99, 99.5, 99.7, 99.9]
        if enforce_percentile is not None:
            th = np.percentile(normal_errors, enforce_percentile)
            th = np.nextafter(th, -np.inf)
            self.threshold = th
            y_pred = (self.errors > th).astype(int)
            f1 = f1_score(self.y_true_bin, y_pred)
            logger.info(f"Forced threshold (percentile {enforce_percentile}): {self.threshold:.6f} (F1={f1:.4f})")
        else:
            best = {"q": None, "th": None, "f1": -1, "y_pred": None}
            for q in candidates:
                th = np.percentile(normal_errors, q)
                th = np.nextafter(th, -np.inf)
                y_pred = (self.errors > th).astype(int)
                f1 = f1_score(self.y_true_bin, y_pred)
                if f1 > best["f1"]:
                    best.update(q=q, th=th, f1=f1, y_pred=y_pred)
            self.threshold = best["th"]
            logger.info(f"Chosen threshold (percentile {best['q']}): {self.threshold}")

        return self.threshold

    def predict(self):
        """
        Predict anomaly labels using the optimized threshold.
        Requires `self.threshold` and `self.errors` to be computed.

        Returns
        -------
        np.ndarray
            Binary predictions: 0 = normal, 1 = anomaly.
        """
        if self.threshold is None:
            raise ValueError("Call optimize_threshold() first")
        if self.errors is None:
            self.compute_errors()
        return (self.errors > self.threshold).astype(int)

    def save_model(self, path: str = "trained_models/autoencoder.pth"):
        """
        Save autoencoder weights and threshold to disk.

        Parameters
        ----------
        path : str
            Output checkpoint file (.pth).
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "threshold": self.threshold,
                "input_dim": self.input_dim,
            },
            path,
        )
        logger.info(f"✓ Autoencoder saved to {path}")

    def load_model(self, path: str = "trained_models/autoencoder.pth"):
        """
        Load autoencoder weights and threshold from checkpoint.

        Parameters
        ----------
        path : str
            Path to the .pth checkpoint file.

        Returns
        -------
        None
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.input_dim = ckpt.get("input_dim")
        self.threshold = ckpt.get("threshold", None)
        if self.input_dim is None:
            raise ValueError("Checkpoint without 'input_dim'.")
        self.model = MLPAutoencoder(input_dim=self.input_dim, latent_dim=2, p_drop=0.1).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        logger.info(f"✓ Autoencoder loaded from {path} (threshold={self.threshold})")

    def run_predict(self, df_test: pd.DataFrame) -> tuple:
        """
        Predict.
        Parameters
        ----------
        df_test : pd.DataFrame
            DataFrame including 'ip.opt.time_stamp'
        model : MLPAutoencoder (nn.Module)

        Returns
        -------
        tuple
        """
        cols_to_drop = ["ip.opt.time_stamp", "frame.number"]
        X_test_df = df_test.drop(columns=cols_to_drop, errors="ignore").select_dtypes(include="number")
        # Labels
        y_test_raw = df_test["ip.opt.time_stamp"]
        y_test_filled = y_test_raw.fillna(-1).astype(int)
        y_test = (y_test_filled != -1).astype(int)  # 0=normal, 1=attack
        # Convert to tensor
        X_test_tensor = torch.tensor(X_test_df.values, dtype=torch.float32)
        test_loader = DataLoader(TensorDataset(X_test_tensor), batch_size=64, shuffle=False)
        # Compute reconstruction errors
        self.model.eval()
        reconstruction_errors = []
        with torch.no_grad():
            for (x_batch,) in test_loader:
                x_batch = x_batch.to(self.device)
                x_hat = self.model(x_batch)
                e = (x_hat - x_batch).abs().mean(dim=1)
                reconstruction_errors.extend(e.detach().cpu().numpy())
        errors = np.asarray(reconstruction_errors)
        errors = np.nan_to_num(errors, posinf=np.finfo(np.float32).max)
        if self.threshold is None:
            raise ValueError(
                "Threshold not set! "
                "Execute run_autoencoder() with optimize_threshold() before using run_predict()"
            )
        y_pred = (errors > self.threshold).astype(int)

        return y_test, y_pred

    def get_score(self, df_pp: pd.DataFrame, sample_idx: int) -> float:
        """
        Compute the reconstruction error for a single sample.

        Parameters
        ----------
        df_pp : pd.DataFrame
            Preprocessed batch.
        sample_idx : int
            Index of the sample to evaluate.

        Returns
        -------
        float
            Mean absolute reconstruction error for the selected sample.
        """
        try:
            # Sort columns like training
            sorted_columns = sorted(df_pp.columns)
            df_sorted = df_pp[sorted_columns].copy()
            # Drop label
            X = df_sorted.drop(columns=["ip.opt.time_stamp"], errors="ignore")
            # Extract 1 sample
            x = X.iloc[sample_idx].values.astype(np.float32)
            # Convert to tensor AND ADD BATCH DIMENSION
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device) # shape becomes (1, num_features)
            self.model.eval()
            with torch.no_grad():
                x_hat = self.model(x_tensor)
                err = (x_hat - x_tensor).abs().mean().item()

            return float(err)

        except Exception as e:
            logger.error(f"get_score for sample {sample_idx}: {e}")
            return np.nan


def evaluate_predictions_ae(y_true_bin: np.ndarray, y_pred_bin: np.ndarray, errors: np.ndarray | None = None):
    """
    Evaluate autoencoder predictions using standard metrics.

    Parameters
    ----------
    y_true_bin : np.ndarray
       Ground-truth binary labels (0=normal, 1=anomaly).
    y_pred_bin : np.ndarray
       Predicted binary labels.
    errors : np.ndarray, optional
        Reconstruction errors used for ROC-AUC (if available).

    Returns
    -------
    dict
        Dictionary containing confusion matrix and metric values.
    """
    labels = [0, 1]
    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=labels)
    logger.info("Confusion Matrix (AE) [0=normal, 1=anomaly]:")
    logger.info(cm)
    logger.info("\nClassification Report (AE):")
    logger.info(classification_report(y_true_bin, y_pred_bin, labels=labels, digits=4, zero_division=0))
    bal_acc = balanced_accuracy_score(y_true_bin, y_pred_bin)
    logger.info(f"Balanced accuracy: {bal_acc}")
    if errors is not None:
        try:
            auc = roc_auc_score(y_true_bin, errors)
            logger.info("AUC:", auc)
        except Exception:
            pass

    acc = accuracy_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    logger.info(f"Accuracy: {acc:.6f}")
    logger.info(f"F1: {f1:.6f}")
    logger.info(f"Precision: {prec:.6f}")
    logger.info(f"Recall: {rec:.6f}")
    return {"cm": cm, "balanced_accuracy": bal_acc, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def run_autoencoder(detector: DetectionAutoEncoder, test_csv: str, enforce_percentile=None):
    """
    Full evaluation pipeline for the AE model.

    Parameters
    ----------
    detector : DetectionAutoEncoder
        Autoencoder detector instance.
    test_csv : str
        Path to the test CSV file.
    enforce_percentile : float or None
        Optional percentile to force threshold selection.
    """
    detector.load_test_data(test_csv)
    detector.compute_errors()
    detector.optimize_threshold(enforce_percentile=enforce_percentile)
    y_pred = detector.predict()  # 0=normal, 1=anomaly
    evaluate_predictions_ae(detector.y_true_bin, y_pred, errors=detector.errors)



