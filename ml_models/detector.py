"""Detector wrapper class for anomaly detection models."""

from pathlib import Path

import numpy as np
import pandas as pd
from pyod.models.base import BaseDetector

from preprocessing import Preprocessor

from . import base_detector


class Detector(base_detector.BaseDetector):
    """Detector wrapper class for anomaly detection models."""

    def __init__(self, detector_class: BaseDetector, **detector_params) -> None:
        """
        Create and initialize the Detector.

        Parameters
        ----------
        detector_class : BaseDetector
            The class of the detector to be used (must inherit from BaseDetector).
        detector_params : dict
            Parameters to initialize the detector.
        """

        self.detector_class = detector_class
        self.detector_params = detector_params

        self._preprocessor = Preprocessor()
        self._trained = False
        self._detector: BaseDetector = self.detector_class(
            **self.detector_params
        )

    def fit(
        self,
        X: pd.DataFrame,
        data_path: Path | str | None = None,
        skip_preprocess: bool = False,
    ) -> "Detector":
        """
        Train the detector model.

        Parameters
        ----------
        X : pd.DataFrame
            The training data.
        data_path : Path | str | None
            Path to load/save the preprocessed data. If specified, the
            preprocessed data will be saved to this path after preprocessing.
            If a file exists at this path, it will be loaded instead of
            preprocessing the data again.
        skip_preprocess : bool, optional
            Whether to skip preprocessing, by default False.

        Returns
        -------
        Self
            The trained Detector instance.
        """
        X = X.copy()
        if not skip_preprocess:
            X = self._preprocessor.train(X, data_path)
        X = X[sorted(X.columns)]

        if X.isnull().any().any():
            raise ValueError(
                "Input data contains NaN values after preprocessing."
            )

        self._detector.fit(X.values)
        self._trained = True

        return self

    def predict(
        self,
        X: pd.DataFrame,
        data_path: Path | str | None = None,
        skip_preprocess: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : pd.DataFrame
            The data to predict on.
        data_path : Path | str | None
            Path to load/save the preprocessed data. If specified, the
            preprocessed data will be saved to this path after preprocessing.
            If a file exists at this path, it will be loaded instead of
            preprocessing the data again.
        skip_preprocess : bool, optional
            Whether to skip preprocessing, by default False.

        Returns
        -------
        tuple[np.ndarray, np.ndarray] | np.ndarray
            A tuple containing the predicted labels and anomaly scores
            or just the predicted labels depending on the detector.
        """
        if not self._trained:
            raise ValueError("Model not trained, call fit().")

        X = X.copy()
        if not skip_preprocess:
            X = self._preprocessor.test(X, data_path)
        X = X[sorted(X.columns)]

        if X.isnull().any().any():
            raise ValueError(
                "Input data contains NaN values after preprocessing."
            )

        return self._detector.predict(X.values)

    def decision_function(
        self,
        X: pd.DataFrame,
        data_path: Path | str | None = None,
        skip_preprocess: bool = False,
    ) -> np.ndarray:
        """
        Predict confidence scores for samples.

        Parameters
        ----------
        df : pd.DataFrame
            The data to compute scores for.
        data_path : Path | str | None
            Path to load/save the preprocessed data. If specified, the
            preprocessed data will be saved to this path after preprocessing.
            If a file exists at this path, it will be loaded instead of
            preprocessing the data again.
        skip_preprocess : bool, optional
            Whether to skip preprocessing, by default False.

        Returns
        -------
        np.ndarray
            The decision function scores.
        """
        if not self._trained:
            raise ValueError("Model not trained, call fit().")

        X = X.copy()
        if not skip_preprocess:
            X = self._preprocessor.test(X, data_path)
        X = X[sorted(X.columns)]

        if X.isnull().any().any():
            raise ValueError(
                "Input data contains NaN values after preprocessing."
            )

        return self._detector.decision_function(X.values)
