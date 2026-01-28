import numpy as np
import pandas as pd

from .base_detector import BaseDetector
from .detector import Detector


class EnsembleDetector(BaseDetector):
    """Ensemble meta-classifier for anomaly detection."""

    def __init__(self, meta_clf: object, detectors: list[Detector]) -> None:
        self._meta_clf = meta_clf
        self._detectors = detectors
        self._meta_mean = None
        self._meta_std = None

    def _meta_features_raw(
        self, X: pd.DataFrame, skip_preprocess: bool
    ) -> np.ndarray:
        scores = []
        for det in self._detectors:
            s = det.decision_function(X, skip_preprocess=skip_preprocess)
            scores.append(s)
        return np.vstack(scores).T  # (n_samples, n_detectors)

    def _meta_features(
        self, X: pd.DataFrame, skip_preprocess: bool
    ) -> np.ndarray:
        S = self._meta_features_raw(X, skip_preprocess)
        return (S - self._meta_mean) / self._meta_std

    def fit(
        self, X: pd.DataFrame, y: np.ndarray, skip_preprocess: bool = False
    ) -> None:
        S = self._meta_features_raw(X, skip_preprocess)

        self._meta_mean = S.mean(axis=0)
        self._meta_std = S.std(axis=0) + 1e-8  # evita divisioni per 0

        X_meta = (S - self._meta_mean) / self._meta_std
        self._meta_clf.fit(X_meta, y)

    def predict(
        self, X: pd.DataFrame, skip_preprocess: bool = False
    ) -> np.ndarray:
        X_meta = self._meta_features(X, skip_preprocess)
        return self._meta_clf.predict(X_meta)

    def decision_function(
        self, X: pd.DataFrame, skip_preprocess: bool = False
    ) -> np.ndarray:
        X_meta = self._meta_features(X, skip_preprocess)
        return self._meta_clf.decision_function(X_meta)
