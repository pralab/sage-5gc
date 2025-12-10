from pathlib import Path

import pandas as pd

from .utils import (
    preprocessing_test as _test,
)
from .utils import (
    preprocessing_train as _train,
)


class Preprocessor:
    """Class that exposes utility preprocessing functions as methods."""

    def train(
        self,
        df_train: pd.DataFrame,
        data_path: Path | str | None = None,
    ) -> pd.DataFrame:
        """
        Call the train preprocessing pipeline.

        Parameters
        ----------
        df_train : pd.DataFrame
            Training clean DataFrame.
        data_path : Path | str | None
            Path to load/save the preprocessed data. If specified, the preprocessed data
            will be saved to this path after preprocessing. If a file exists at this path,
            it will be loaded instead of preprocessing the data again.

        Returns
        -------
        pd.DataFrame
            The preprocessed DataFrame.
        """
        if data_path is not None and Path(data_path).exists():
            with Path(data_path).open("r") as f:
                return pd.read_csv(f, sep=";")

        return _train(df_train, data_path)

    def test(
        self,
        df_test: pd.DataFrame,
        data_path: Path | str | None = None,
    ) -> pd.DataFrame:
        """
        Call the test preprocessing pipeline.

        Parameters
        ----------
        df_test : pd.DataFrame
            Test clean DataFrame.
        data_path : Path | str | None
            Path to load/save the preprocessed data. If specified, the preprocessed data
            will be saved to this path after preprocessing. If a file exists at this path,
            it will be loaded instead of preprocessing the data again.

        Returns
        -------
        pd.DataFrame
            The preprocessed DataFrame.
        """
        if data_path is not None and Path(data_path).exists():
            with Path(data_path).open("r") as f:
                return pd.read_csv(f, sep=";")

        return _test(df_test, data_path)
