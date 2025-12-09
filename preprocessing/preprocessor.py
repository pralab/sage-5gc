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
        output_path: Path | str | None = None,
        skip_preprocess: bool = False,
    ) -> pd.DataFrame:
        """
        Call the train preprocessing pipeline.

        Parameters
        ----------
        df_train : pd.DataFrame
            Training clean DataFrame.
        output_path : Path | str | None
            Path to save the processed DataFrame.
        skip_preprocess : bool
            Whether to skip preprocessing if processed data already exists.

        Returns
        -------
        pd.DataFrame
            The preprocessed DataFrame.
        """
        if skip_preprocess and output_path is not None:
            if not Path(output_path).exists():
                raise FileNotFoundError(f"Dataset not found in {output_path}.")

            with Path(output_path).open("r") as f:
                df = pd.read_csv(f, sep=";")

        else:
            df = _train(df_train, output_path)

        return df

    def test(
        self,
        df_test: pd.DataFrame,
        output_path: Path | str | None = None,
        skip_preprocess: bool = False,
    ) -> pd.DataFrame:
        """
        Call the test preprocessing pipeline.

        Parameters
        ----------
        df_test : pd.DataFrame
            Test clean DataFrame.
        output_path : Path | str | None
            Path to save the processed DataFrame.
        skip_preprocess : bool
            Whether to skip preprocessing if processed data already exists.

        Returns
        -------
        pd.DataFrame
            The preprocessed DataFrame.
        """
        if skip_preprocess and output_path is not None:
            if not Path(output_path).exists():
                raise FileNotFoundError(f"Dataset not found in {output_path}.")

            with Path(output_path).open("r") as f:
                df = pd.read_csv(f, sep=";")
        else:
            df = _test(df_test, output_path)

        return df
