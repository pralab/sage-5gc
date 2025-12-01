import inspect

import pandas as pd
from pyod.models.base import BaseDetector
from sklearn.base import BaseEstimator, ClassifierMixin

from preprocessing import Preprocessor


class Detector(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        detector_class=BaseDetector,
        **detector_params,
    ) -> None:
        self.detector_class = detector_class
        # Salva i params come attributi per GridSearchCV!
        for k, v in detector_params.items():
            setattr(self, k, v)
        self.detector_params = detector_params
        self.detector = None
        self.preprocessor = Preprocessor()
        self._trained = False
        self._feature_columns = None

    def fit(
        self,
        df_train: pd.DataFrame,
        output_dir: str = "tmp",
        skip_preprocess: bool = False,
    ):
        df = self.preprocessor.train(df_train, output_dir, skip_preprocess)
        X_ = df.drop(columns=["ip.opt.time_stamp"], errors="ignore")
        X_ = X_.copy()
        X_ = X_[sorted(X_.columns)]
        self._feature_columns = X_.columns.tolist()

        # Costruisco i kwargs leggendo tutti gli attributi corrispondenti alla detector_class
        detector_kwargs = {}
        sig = inspect.signature(self.detector_class.__init__)
        for param in sig.parameters.values():
            if param.name != "self" and hasattr(self, param.name):
                detector_kwargs[param.name] = getattr(self, param.name)

        self.detector = self.detector_class(**detector_kwargs)
        self.detector.fit(X_.values)
        self._trained = True
        return self

    def predict(
        self,
        df_test: pd.DataFrame,
        output_dir: str = "tmp",
        skip_preprocess: bool = False,
    ):
        if not self._trained:
            raise ValueError("Model not trained, call fit() before predict().")

        df = self.preprocessor.test(df_test, output_dir, skip_preprocess)
        X_ = df.drop(columns=["ip.opt.time_stamp"], errors="ignore")
        X_ = X_.copy()
        if self._feature_columns is not None:
            X_ = X_[self._feature_columns]
        return self.detector.predict(X_.values)

    def decision_function(
        self, df_test: pd.DataFrame, sample_idx: int, skip_preprocess: bool = False
    ):
        df = self.preprocessor.test(df_test, "tmp", skip_preprocess)
        X = df.drop("ip.opt.time_stamp", axis=1, errors="ignore")
        X = X.copy()
        if self._feature_columns is not None:
            X = X[self._feature_columns]
        X_sample = X.iloc[[sample_idx]]
        return -self.detector.decision_function(X_sample.values)[0]

    def get_params(self, deep=True):
        out = {"detector_class": self.detector_class}
        # Aggiungi tutti i parametri detector come attributi
        out.update(self.detector_params)
        # Espone anche quelli settati come attributi individuali
        sig = inspect.signature(self.detector_class.__init__)
        for param in sig.parameters.values():
            if param.name != "self" and hasattr(self, param.name):
                out[param.name] = getattr(self, param.name)
        return out

    def set_params(self, **params):
        # aggiorna detector_class e attributi
        if "detector_class" in params:
            self.detector_class = params.pop("detector_class")
        for k, v in params.items():
            setattr(self, k, v)
            self.detector_params[k] = v
        return self
