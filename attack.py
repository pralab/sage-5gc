import logging

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from ml_models import DetectionIsolationForest, DetectionKnn, DetectionRandomForest

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

NUMERIC_COLS = ["TCP_Ack", "TCP_Seq", "TCP_Urgent", "TCP_Window", "length"]
CATEGORICAL_COLS = [
    "Chksum",
    "IP_Chksum",
    "IP_Flags",
    "IP_ID",
    "IP_IHL",
    "IP_TOS",
    "IP_TTL",
    "IP_Version",
    "TCP_Dataofs",
    "TCP_Flags",
    "protocol",
    "src_port",
]


def add_noise(
    df: pd.DataFrame,
    noise_level: float = 0.01,
    cols: list[str] = None,
    distribution: str = "normal",
) -> pd.DataFrame:
    """
    Add noise to numeric columns of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    noise_level : float
        Standard deviation of noise as a fraction of each column’s std.
    cols : list[str], optional
        Subset of columns to perturb. Defaults to all numeric columns.
    distribution : {'normal','uniform'}
        Type of noise distribution.

    Returns
    -------
    pd.DataFrame
        New DataFrame with noise applied.
    """
    df = df.copy()

    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in cols:
        if not is_numeric_dtype(df[col]):
            print(f"Skipping non-numeric column {col!r}")
            continue

        std = df[col].std(ddof=0)
        scale = std * noise_level

        if distribution == "normal":
            noise = np.random.normal(loc=0, scale=scale, size=len(df))
        elif distribution == "uniform":
            noise = np.random.uniform(low=-scale, high=+scale, size=len(df))
        else:
            raise ValueError("distribution must be 'normal' or 'uniform'")

        df[col] = df[col] + noise

    return df


def _perturb_col(col: str, df_mod: pd.DataFrame, noise_level: float) -> pd.DataFrame:
    # Numeric features
    if col in NUMERIC_COLS and col in df_mod.columns:
        df_mod = add_noise(df_mod, noise_level=noise_level, cols=[col])
    else:
        # Categorical features
        uniques = pd.unique(df_mod[col]).tolist()
        # Pick a random value from the pool of uniques.
        choices = np.random.choice(uniques, size=len(df_mod))
        df_mod[col] = choices

    return df_mod


def perform_fingerprinting(
    detection, model, df: pd.DataFrame, noise_level: float = 0.01
) -> tuple[list[str], list[float]]:
    """
    Create a simple fingerprint by perturbing one column at a time and
    recording how often the model's prediction changes compared to the
    unmodified dataframe.

    Parameters
    ----------
    detection: DetectionIsolationForest | DetectionKnn | DetectionRandomForest
        Detection wrapper that implements `run_predict(df, model)` and returns
        (y_test, y_pred).
    model: IsolationForest | KNeighborsClassifier | RandomForestClassifier
        Sklearn model to evaluate.
    df: pd.DataFrame
        DataFrame with data (will not be modified in-place).
    noise_level: float
        Noise level to use for numeric columns.

    Returns
    -------
    tuple[list[str], list[float]]
        Tuple containing a list of perturbed column names and a list of the
        corresponding fraction of changed predictions (in percent).

    Notes
    -----
    - Numeric features are perturbed using `add_noise`.
    - Categorical features are perturbed by rotating category values.
    """
    cols_to_test = [c for c in CATEGORICAL_COLS + NUMERIC_COLS if c in df.columns]

    # Baseline predictions
    _, y_pred_baseline = detection.run_predict(df, model)
    y_pred_baseline = np.asarray(y_pred_baseline)

    sensitivities: list[float] = []
    for col in cols_to_test:
        df_mod = df.copy()

        df_mod[col] = _perturb_col(col, df_mod, noise_level)

        try:
            _, y_pred_mod = detection.run_predict(df_mod, model)
        except Exception as e:
            logging.warning(f"Prediction failed for column {col}: {e}")
            sensitivities.append(0.0)
            continue

        y_pred_mod = np.asarray(y_pred_mod)

        # Compute fraction of changed predictions
        if y_pred_mod.shape != y_pred_baseline.shape:
            sensitivities.append(0.0)
        else:
            changed_frac = float(np.mean(y_pred_mod != y_pred_baseline)) * 100.0
            sensitivities.append(changed_frac)

    return cols_to_test, sensitivities


def perform_fingerprinting2(
    detection: DetectionIsolationForest | DetectionKnn | DetectionRandomForest,
    model,
    df: pd.DataFrame,
    noise_level: float = 0.01,
) -> tuple[list[str], list[float]]:
    cols_to_test = [c for c in CATEGORICAL_COLS + NUMERIC_COLS if c in df.columns]
    tuples_to_test = [("IP_Flags", col) for col in cols_to_test if col != "IP_Flags"]

    # Baseline predictions
    _, y_pred_baseline = detection.run_predict(df, model)
    y_pred_baseline = np.asarray(y_pred_baseline)

    sensitivities: list[float] = []
    for col, col1 in tuples_to_test:
        df_mod = df.copy()

        df_mod = _perturb_col(col, df_mod, noise_level)
        df_mod = _perturb_col(col1, df_mod, noise_level)

        try:
            _, y_pred_mod = detection.run_predict(df_mod, model)
        except Exception as e:
            logging.warning(f"Prediction failed for column {col}: {e}")
            sensitivities.append(0.0)
            continue

        y_pred_mod = np.asarray(y_pred_mod)

        # Compute fraction of changed predictions
        if y_pred_mod.shape != y_pred_baseline.shape:
            sensitivities.append(0.0)
        else:
            changed_frac = float(np.mean(y_pred_mod != y_pred_baseline)) * 100.0
            sensitivities.append(changed_frac)

    return [f"{col} - {col1}" for (col, col1) in tuples_to_test], sensitivities


def perform_fingerprinting3(
    detection: DetectionIsolationForest | DetectionKnn | DetectionRandomForest,
    model,
    df: pd.DataFrame,
    noise_level: float = 0.01,
) -> tuple[list[str], list[float]]:
    cols_to_test = [c for c in CATEGORICAL_COLS + NUMERIC_COLS if c in df.columns]
    tuples_to_test = [
        ("IP_Flags", "TCP_Flags", col)
        for col in cols_to_test
        if col != "IP_Flags" or col != "TCP_Flags"
    ]

    # Baseline predictions
    _, y_pred_baseline = detection.run_predict(df, model)
    y_pred_baseline = np.asarray(y_pred_baseline)

    sensitivities: list[float] = []
    for col, col1, col2 in tuples_to_test:
        df_mod = df.copy()

        df_mod = _perturb_col(col, df_mod, noise_level)
        df_mod = _perturb_col(col1, df_mod, noise_level)
        df_mod = _perturb_col(col2, df_mod, noise_level)

        try:
            _, y_pred_mod = detection.run_predict(df_mod, model)
        except Exception as e:
            logging.warning(f"Prediction failed for column {col}: {e}")
            sensitivities.append(0.0)
            continue

        y_pred_mod = np.asarray(y_pred_mod)

        # Compute fraction of changed predictions
        if y_pred_mod.shape != y_pred_baseline.shape:
            sensitivities.append(0.0)
        else:
            changed_frac = float(np.mean(y_pred_mod != y_pred_baseline)) * 100.0
            sensitivities.append(changed_frac)

    return [
        f"{col} - {col1} - {col2}" for col, col1, col2 in tuples_to_test
    ], sensitivities
