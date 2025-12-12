import numpy as np
import pandas as pd
from pyod.utils.utility import standardizer

from .base_detector import BaseDetector
from .detector import Detector


class EnsembleDetector(BaseDetector):
    def __init__(
        self, meta_clf: object, detectors: list[Detector]
    ) -> None:
        """
        Create and initialize the EnsembleDetector.

        Parameters
        ----------
        meta_clf : object
            The meta-classifier to combine base detector outputs.
        detectors : list[Detector]
            List of base Detector instances.
        """
        self._meta_clf = meta_clf
        self._detectors = detectors

    def fit(
        self, X: pd.DataFrame, y: np.ndarray, skip_preprocess: bool = False
    ) -> None:
        scores = []
        for detector in self._detectors:
            score = detector.decision_function(X, skip_preprocess=skip_preprocess)
            scores.append(standardizer(score.reshape(-1, 1)).ravel())
        X_meta = np.vstack(scores).T
        self._meta_clf.fit(X_meta, y)

    def predict(self, X: pd.DataFrame, skip_preprocess: bool = False) -> np.ndarray:
        scores = []
        for detector in self._detectors:
            score = detector.decision_function(X, skip_preprocess=skip_preprocess)
            scores.append(standardizer(score.reshape(-1, 1)).ravel())
        X_meta = np.vstack(scores).T
        return self._meta_clf.predict(X_meta)

    def decision_function(
        self, X: pd.DataFrame, skip_preprocess: bool = False
    ) -> np.ndarray:
        scores = []
        for detector in self._detectors:
            score = detector.decision_function(X, skip_preprocess=skip_preprocess)
            scores.append(standardizer(score.reshape(-1, 1)).ravel())
        X_meta = np.vstack(scores).T
        return self._meta_clf.decision_function(X_meta)
