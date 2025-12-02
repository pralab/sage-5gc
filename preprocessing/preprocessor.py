from pathlib import Path
import pandas as pd

from .preprocessor_utils_new import (
    preprocessing_pipeline_train as _train,
    preprocessing_pipeline_test as _test,
)


class Preprocessor:
    """Wrapper class that exposes utility preprocessing functions as methods."""

    def train(
        self, df_train: pd.DataFrame, output_dir: str, skip_preprocess: bool = False
    ) -> pd.DataFrame:
        """
        Call the train preprocessing pipeline. Run only if not already fitted.

        Parameters
        ----------
        df_train : pd.DataFrame
            Training clean DataFrame.
        output_dir : str
            Output directory for results.
        skip_preprocess : bool, optional
            If True, skip preprocessing and load existing final dataset from output_dir,
            by default False.

        Returns
        -------
        pd.DataFrame
            The preprocessed DataFrame.
        """
        if skip_preprocess:
            if not (Path(output_dir) / "train_dataset_processed.csv").exists:
                raise FileNotFoundError(f"Missing final dataset in {output_dir}.")

            with (Path(output_dir) / "train_dataset_processed.csv").open("r") as f:
                df = pd.read_csv(f, low_memory=False, sep=";")
        else:
            # Run the actual training pipeline
            df = _train(df_train, output_dir)
        return df

    def test(
        self, df_test: pd.DataFrame, output_dir: str, skip_preprocess: bool = False
    ) -> pd.DataFrame:
        """
        Call the test preprocessing pipeline.

        Parameters
        ----------
        df_test : pd.DataFrame
            Test clean DataFrame.
        output_dir : str
            Output directory for results.

        Returns
        -------
        pd.DataFrame
            The preprocessed DataFrame.
        """
        if skip_preprocess:
            if not (Path(output_dir) / "test_dataset_processed.csv").exists:
                raise FileNotFoundError(f"Missing final dataset in {output_dir}.")

            with (Path(output_dir) / "test_dataset_processed.csv").open("r") as f:
                df = pd.read_csv(f, low_memory=False, sep=";")
        else:
            df = _test(df_test, output_dir)

        return df
