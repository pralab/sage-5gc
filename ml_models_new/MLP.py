import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision.ops import MLP
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, f1_score,
    precision_score, recall_score, accuracy_score, balanced_accuracy_score,
)
import torch
from torch.utils.data import TensorDataset, DataLoader

# ---------------------------
# Déterminisme - GLOBALE come l'originale!
# ---------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ---------------------------
# Modèle
# ---------------------------
class MLPAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 8, p_drop: float = 0.1):
        super().__init__()
        print(latent_dim)
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
        z = self.encoder(x)
        z = self.enc_bn(z)
        x_hat = self.decoder(z)
        return x_hat


class DetectionAutoEncoder:
    """
    Wrapper class per compatibilità con ml_models API
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Stato
        self.model = None
        self.input_dim = None
        self.threshold = None
        self.errors = None

        # Cache dati
        self.X_train = None
        self.X_test = None
        self.y_true_bin = None

    # ---------------------------
    # Data Loading
    # ---------------------------
    def load_train_data(self, df_train_csv: str):
        df_train = pd.read_csv(df_train_csv, sep=';')

        cols_to_drop = ["ip.opt.time_stamp", "frame.number"]
        X_train_df = df_train.drop(columns=cols_to_drop, errors="ignore").select_dtypes(include="number")

        self.X_train = X_train_df.values
        self.input_dim = X_train_df.shape[1]

        print(len(X_train_df.columns))

    def load_test_data(self, df_test_csv: str):
        df_test = pd.read_csv(df_test_csv, sep=';')

        cols_to_drop = ["ip.opt.time_stamp", "frame.number"]
        X_test_df = df_test.drop(columns=cols_to_drop, errors="ignore").select_dtypes(include="number")

        self.X_test = X_test_df.values

        y_test_raw = df_test["ip.opt.time_stamp"]
        y_test = y_test_raw.fillna(-1).astype(int).values
        self.y_true_bin = (y_test != -1).astype(int)

        print(len(X_test_df.columns))

    # ---------------------------
    # Training
    # ---------------------------
    def train_autoencoder(self, lr=3e-4, num_epochs=200, patience=10, sparsity_coef=1e-4, latent_dim=2):
        if self.X_train is None or self.input_dim is None:
            raise ValueError("Chiama prima load_train_data()")

        # Tensors
        X_train_tensor = torch.tensor(self.X_train, dtype=torch.float32)

        # DataLoaders (avec val split sur le sain)
        full_train_ds = TensorDataset(X_train_tensor)
        val_ratio = 0.1
        val_size = max(1, int(len(full_train_ds) * val_ratio))
        train_size = len(full_train_ds) - val_size
        train_ds, val_ds = random_split(full_train_ds, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

        # Crea il modello
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

                # Sparsity: L1 sul latente
                z = self.model.enc_bn(self.model.encoder(x_batch))
                loss = criterion(x_hat, x_batch) + sparsity_coef * z.abs().mean()

                loss.backward()
                optimizer.step()

                bs = x_batch.size(0)
                train_loss_sum += loss.item() * bs
                n_train += bs
            train_loss = train_loss_sum / max(1, n_train)

            # --- Validation ---
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

            print(f"Epoch [{epoch}/{num_epochs}] - train_loss: {train_loss:.6f} - val_loss: {val_loss:.6f}")

            # Early stopping
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                bad = 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    print("Early stopping.")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        print("✓ Autoencoder trained")

    # ---------------------------
    # Error Computation & Threshold
    # ---------------------------
    def compute_errors(self):
        if self.X_test is None:
            raise ValueError("Chiama prima load_test_data()")
        if self.model is None:
            raise ValueError("Chiama prima train_autoencoder()")

        self.model.eval()
        X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32)
        test_loader = DataLoader(TensorDataset(X_test_tensor), batch_size=64, shuffle=False)

        reconstruction_errors = []
        with torch.no_grad():
            for (x_batch,) in test_loader:
                x_batch = x_batch.to(self.device)
                x_hat = self.model(x_batch)
                # CRITICO: stesso ordine dell'originale!
                e = (x_hat - x_batch).abs().mean(dim=1)
                reconstruction_errors.extend(e.detach().cpu().numpy())

        errors = np.asarray(reconstruction_errors)
        self.errors = np.nan_to_num(errors, posinf=np.finfo(np.float32).max)

    def optimize_threshold(self, enforce_percentile=None):
        if self.errors is None:
            self.compute_errors()
        if self.y_true_bin is None:
            raise ValueError("Chiama prima load_test_data()")

        # Usa y_true_bin (0=normal, 1=attack)
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
            print(f"Threshold forzato (percentile {enforce_percentile}): {self.threshold:.6f} (F1={f1:.4f})")
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
            print(f"Seuil choisi (percentile {best['q']}): {self.threshold}")

        return self.threshold

    def predict(self):
        if self.threshold is None:
            raise ValueError("Chiama prima optimize_threshold()")
        if self.errors is None:
            self.compute_errors()
        return (self.errors > self.threshold).astype(int)

    # ---------------------------
    # Save / Load
    # ---------------------------
    def save_model(self, path: str = "trained_models/autoencoder.pth"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "threshold": self.threshold,
                "input_dim": self.input_dim,
            },
            path,
        )
        print(f"✓ Autoencoder saved to {path}")

    def load_model(self, path: str = "trained_models/autoencoder.pth"):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.input_dim = ckpt.get("input_dim")
        self.threshold = ckpt.get("threshold", None)
        if self.input_dim is None:
            raise ValueError("Checkpoint senza 'input_dim'.")

        self.model = MLPAutoencoder(input_dim=self.input_dim, latent_dim=2, p_drop=0.1).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        print(f"✓ Autoencoder loaded from {path} (threshold={self.threshold})")

    def run_predict(self, df_test: pd.DataFrame) -> tuple:
        """
        Predice su un DataFrame di test arbitrario usando un modello specifico.
        Parameters
        ----------
        df_test : pd.DataFrame
            DataFrame con colonna 'ip.opt.time_stamp' inclusa
        model : MLPAutoencoder (nn.Module)
            Modello PyTorch già allenato

        Returns
        -------
        tuple
            (y_test, y_pred) con labels {0, 1}
            0 = normal, 1 = anomaly
        """
        cols_to_drop = ["ip.opt.time_stamp", "frame.number"]
        X_test_df = df_test.drop(columns=cols_to_drop, errors="ignore").select_dtypes(include="number")
        # Estrai label (MLP_autoencoder.py riga 75-77)
        y_test_raw = df_test["ip.opt.time_stamp"]
        y_test_filled = y_test_raw.fillna(-1).astype(int)
        y_test = (y_test_filled != -1).astype(int)  # 0=normal, 1=attack
        # Converti in tensori
        X_test_tensor = torch.tensor(X_test_df.values, dtype=torch.float32)
        test_loader = DataLoader(TensorDataset(X_test_tensor), batch_size=64, shuffle=False)
        # Calcola errori di ricostruzione (MLP_autoencoder.py riga 169-176)
        self.model.eval()
        reconstruction_errors = []
        with torch.no_grad():
            for (x_batch,) in test_loader:
                x_batch = x_batch.to(self.device)
                x_hat = self.model(x_batch)
                e = (x_hat - x_batch).abs().mean(dim=1)  # MAE per sample
                reconstruction_errors.extend(e.detach().cpu().numpy())
        errors = np.asarray(reconstruction_errors)
        errors = np.nan_to_num(errors, posinf=np.finfo(np.float32).max)
        if self.threshold is None:
            raise ValueError(
                "Threshold non impostato! "
                "Devi eseguire run_autoencoder() con optimize_threshold() "
                "PRIMA di usare run_predict() per robustness evaluation."
            )
        y_pred = (errors > self.threshold).astype(int)

        return y_test, y_pred


def evaluate_predictions_ae(y_true_bin: np.ndarray, y_pred_bin: np.ndarray, errors: np.ndarray | None = None):
    labels = [0, 1]
    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=labels)
    print("Confusion Matrix (AE) [0=normal, 1=anomaly]:")
    print(cm)
    print("\nClassification Report (AE):")
    print(classification_report(y_true_bin, y_pred_bin, labels=labels, digits=4, zero_division=0))
    bal_acc = balanced_accuracy_score(y_true_bin, y_pred_bin)
    print("Balanced accuracy:", bal_acc)

    if errors is not None:
        try:
            auc = roc_auc_score(y_true_bin, errors)
            print("AUC:", auc)
        except Exception:
            pass

    acc = accuracy_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    print(f"Accuracy: {acc:.6f}")
    print(f"F1: {f1:.6f}")
    print(f"Précision: {prec:.6f}")
    print(f"Rappel: {rec:.6f}")
    return {"cm": cm, "balanced_accuracy": bal_acc, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def run_autoencoder(detector: DetectionAutoEncoder, test_csv: str, enforce_percentile=None):
    print("\n" + "=" * 80)
    print(f"TEST DATASET (AE): {test_csv}")
    print("=" * 80)
    detector.load_test_data(test_csv)
    detector.compute_errors()
    detector.optimize_threshold(enforce_percentile=enforce_percentile)
    y_pred = detector.predict()  # 0=normal, 1=anomaly
    evaluate_predictions_ae(detector.y_true_bin, y_pred, errors=detector.errors)

