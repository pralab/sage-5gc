from .preprocessor_utils import (
    preprocessing_pipeline_train as _train,
    preprocessing_pipeline_test as _test,
)


class Preprocessor:
    """Wrapper class that exposes utility preprocessing functions as methods."""
    def __init__(self):
        self._is_fitted = False
        self._df_final = None

    def train(self, output_dir, input_file):
        """Call the train preprocessing pipeline. Run only if not already fitted."""
        if not self._is_fitted:
            # Run the actual training pipeline
            df = _train(output_dir, input_file)
            self._df_final = df
            self._is_fitted = True
            return df
        else:
            # Just load the already fitted data
            self._is_fitted = True
            return self._df_final

    @staticmethod
    def test(output_dir, input_file):
        """Call the test preprocessing pipeline."""
        return _test(output_dir, input_file)

    @property
    def is_fitted(self):
        return self._is_fitted