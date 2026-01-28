"""Abstract base class for anomaly detectors."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseDetector(ABC):
    """Abstract base class for anomaly detectors."""

    @abstractmethod
    def fit(
        self, X: pd.DataFrame, y: np.ndarray, skip_preprocess: bool = False
    ) -> None:
        """
        Train the classifier.

        Parameters
        ----------
        X : pd.DataFrame
            The training data.
        y : np.ndarray
            The true labels for the training data.
        skip_preprocess : bool, optional
            Whether to skip preprocessing, by default False.
        """

    @abstractmethod
    def predict(
        self, X: pd.DataFrame, skip_preprocess: bool = False
    ) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : pd.DataFrame
            The input data.
        skip_preprocess : bool, optional
            Whether to skip preprocessing, by default False.

        Returns
        -------
        np.ndarray
            The predicted class labels.
        """

    @abstractmethod
    def decision_function(
        self, X: pd.DataFrame, skip_preprocess: bool = False
    ) -> np.ndarray:
        """
        Predict confidence scores for samples.

        Parameters
        ----------
        X : pd.DataFrame
            The input data.
        skip_preprocess : bool, optional
            Whether to skip preprocessing, by default False.

        Returns
        -------
        np.ndarray
            The confidence scores.
        """
