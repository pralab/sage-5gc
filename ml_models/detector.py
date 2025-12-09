from pathlib import Path

import numpy as np
import pandas as pd
from pyod.models.base import BaseDetector

from preprocessing import Preprocessor


class Detector:
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
        self._detector: BaseDetector = self.detector_class(**self.detector_params)

    def fit(
        self,
        df: pd.DataFrame,
        output_path: Path | str | None = None,
        skip_preprocess: bool = False,
    ) -> "Detector":
        """
        Train the detector model.

        Parameters
        ----------
        df : pd.DataFrame
            The training data.
        output_path : Path | str | None
            Path to save the preprocessed data.
        skip_preprocess : bool, optional
            Whether to skip preprocessing, by default False.

        Returns
        -------
        Self
            The trained Detector instance.
        """
        X = df.copy()
        X = self._preprocessor.train(X, output_path, skip_preprocess)
        X = X[sorted(X.columns)]

        self._detector.fit(X.values)
        self._trained = True

        return self

    def predict(
        self,
        df: pd.DataFrame,
        output_path: Path | str | None = None,
        skip_preprocess: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """
        Predict anomalies in the data.

        Parameters
        ----------
        df : pd.DataFrame
            The data to predict on.
        output_path : str | None
            Path to save the preprocessed data.
        skip_preprocess : bool, optional
            Whether to skip preprocessing, by default False.

        Returns
        -------
        tuple[np.ndarray, np.ndarray] | np.ndarray
            A tuple containing the predicted labels and anomaly scores
            or just the predicted labels depending on the detector.
        """
        if not self._trained:
            raise ValueError("Model not trained, call fit() before predict().")

        X = df.copy()
        X = self._preprocessor.test(X, output_path, skip_preprocess)
        X = X[sorted(X.columns)]

        if X.isnull().any().any():
            raise ValueError("Input data contains NaN values after preprocessing.")

        return self._detector.predict(X.values)

    def decision_function(
        self,
        df: pd.DataFrame,
        output_path: Path | str | None = None,
        skip_preprocess: bool = False,
    ) -> np.ndarray:
        """
        Compute the decision function scores for the data.

        Parameters
        ----------
        df : pd.DataFrame
            The data to compute scores for.
        output_path : Path | str | None
            Path to save the preprocessed data.
        skip_preprocess : bool, optional
            Whether to skip preprocessing, by default False.

        Returns
        -------
        np.ndarray
            The decision function scores.
        """
        X = df.copy()
        X = self._preprocessor.test(X, output_path, skip_preprocess)
        X = X[sorted(X.columns)]

        if X.isnull().any().any():
            raise ValueError("Input data contains NaN values after preprocessing.")

        return self._detector.decision_function(X.values)
