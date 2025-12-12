from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseDetector(ABC):

    @abstractmethod
    def fit(
        self, X: pd.DataFrame, y: np.ndarray, skip_preprocess: bool = False
    ) -> None:
        """
        Train the ensemble meta-classifier.

        Parameters
        ----------
        X : pd.DataFrame
            The training data.
        y : np.ndarray
            The true labels for the training data.
        skip_preprocess : bool, optional
            Whether to skip preprocessing, by default False.

        Returns
        -------
        Self
            The trained EnsembleDetector instance.
        """

    @abstractmethod
    def predict(self, X: pd.DataFrame, skip_preprocess: bool = False) -> np.ndarray:
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
