import numpy as np
import pandas as pd
from pyod.models.base import BaseDetector
from sklearn.base import BaseEstimator

from preprocessing import Preprocessor


class Detector(BaseEstimator):
    """"""

    def __init__(self, detector_class: BaseDetector, **detector_params) -> None:
        self.detector_class = detector_class
        self.detector_params = detector_params

        self._preprocessor = Preprocessor()
        self._trained = False
        self._detector: BaseDetector = self.detector_class(**self.detector_params)

    def fit(
        self,
        df: pd.DataFrame,
        output_dir: str | None,
        skip_preprocess: bool = False,
    ) -> "Detector":
        X = df.copy()
        X = self._preprocessor.train(X, output_dir, skip_preprocess)
        X = X[sorted(X.columns)]

        self._detector.fit(X.values)
        self._trained = True

        return self

    def predict(
        self,
        df: pd.DataFrame,
        output_dir: str | None,
        skip_preprocess: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self._trained:
            raise ValueError("Model not trained, call fit() before predict().")

        X = df.copy()
        X = self._preprocessor.test(X, output_dir, skip_preprocess)
        X = X[sorted(X.columns)]

        return self._detector.predict(X.values)

    def decision_function(self, df: pd.DataFrame) -> np.ndarray:
        X = df.copy()
        X = self._preprocessor.test(X, None, False)
        X = X[sorted(X.columns)]
        return self._detector.decision_function(X.values)

    def get_params(self, deep: bool = True) -> dict:
        params = {"detector_class": self.detector_class}
        params.update(self.detector_params)
        return params

    def set_params(self, **params) -> "Detector":
        if "detector_class" in params:
            self.detector_class = params.pop("detector_class")

        self.detector_params.update(params)

        self._detector = self.detector_class(**self.detector_params)
        self._trained = False
        return self
