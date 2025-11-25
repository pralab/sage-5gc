import logging
import os

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from ml_models import DetectionIsolationForest, DetectionKnn, DetectionRandomForest
from modifiable_features_fingerprinting import MODIFIABLE_FEATURES
from preprocessing.preprocessor import preprocessing_pipeline_partial

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


def perform_fingerprinting_modifiable_categorical_clean(
    detection,
    model,
    df: pd.DataFrame,
    threshold: float = 1.0,
    in_memory: bool = False,
) -> tuple[list[str], list[float]]:
     """
    Perform feature fingerprinting using a purely categorical swap strategy.
    Each modifiable feature is perturbed by randomly swapping its values
    with other valid values observed in the dataset (no noise, no scaling).

    The sensitivity score for each feature is defined as:
        impact = % of predictions that change after perturbation.

    Parameters
    ----------
    detection : Detection*
        Detection object providing run_predict(df_pp) → (scores, predictions).
    df : DataFrame
        Raw attack dataset before preprocessing.
    threshold : float
        Minimum impact (%) to consider a feature "effective".
    in_memory : bool
        If True, preprocessing_pipeline_partial uses a temporary directory.

    Returns
    -------
    tuple
        - tested_features: list[str] - feature names testate
        - sensitivities: list[float] - impact scores (%)
    """
    OUTPUT_DIR = "fingerprinted_datasets"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.info(
        "🔍 Starting model fingerprinting (CATEGORICAL treatment for ALL features)..."
    )

    # Filter modifiable features
    available_features = [f for f in MODIFIABLE_FEATURES if f in df.columns]
    logger.info(
        f"   Available modifiable features: {len(available_features)}/{len(MODIFIABLE_FEATURES)}"
    )
    if len(available_features) == 0:
        logger.warning("⚠️  No modifiable features found!")
        return [], []
    missing_features = [f for f in MODIFIABLE_FEATURES if f not in df.columns]
    # Log the missing features
    if missing_features:
        logger.warning(f"Missing features in the dataset: {missing_features}")

    # ========== Step 2: Baseline predictions ==========
    ###### ADD PREPROCESSING STEP ######
    df_pp = preprocessing_pipeline_partial(
        output_dir=f"{OUTPUT_DIR}/final_datasets_original", df=df, in_memory=in_memory
    )

    _, y_pred_baseline = detection.run_predict(df_pp)
    y_pred_baseline = np.asarray(y_pred_baseline)
    # ========== Step 3: Test each feature (CATEGORICAL swap) ==========
    sensitivities: list[float] = []
    tested_features: list[str] = []
    for col in available_features:
        tested_features.append(col)
        df_mod = df.copy()
        # Categorical swap
        try:
            uniques = df[col].unique().tolist()
            uniques = [v for v in uniques if pd.notna(v)]    # drop NaN
            if len(uniques) <= 1:
                # Cannot perturb constant features → zero sensitivity
                sensitivities.append(0.0)
                logger.info(f"   Feature '{col}': 0.00% ❌ SKIPPED (constant feature)")
                continue
            # Randomly sample valid categorical values to perturb column
            df_mod[col] = np.random.choice(uniques, size=len(df))

        except Exception as e:
            logger.warning(f"Failed to perturb {col}: {e}")
            sensitivities.append(0.0)
            continue

        # Predict after perturbation
        try:
            ###### ADD PREPROCESSING STEP ######
            clean_path = os.path.join(
                OUTPUT_DIR,
                f"cleaned_dataset_mod/dataset_3_cleaned_fingerprinted_{col}.csv",
            )
            df_mod.to_csv(clean_path, sep=";", index=False)
            df_mod_pp = preprocessing_pipeline_partial(
                output_dir=f"{OUTPUT_DIR}/final_datasets_mod",
                dataset_name=f"dataset_3_final_{col}",
                df=df_mod,
                in_memory=in_memory,
            )

            _, y_pred_mod = detection.run_predict(df_mod_pp)
            y_pred_mod = np.asarray(y_pred_mod)
            # Compute impact (sensitivity)
            if y_pred_mod.shape != y_pred_baseline.shape:
                impact = 0.0
            else:
                impact = float(np.mean(y_pred_mod != y_pred_baseline)) * 100.0
            sensitivities.append(impact)
            # Log
            status = "✅ EFFECTIVE" if impact > threshold else "❌ INEFFECTIVE"
            logger.info(f"   Feature '{col}' (CATEGORICAL): {impact:.2f}% {status}")

        except Exception as e:
            logger.warning(f"Prediction failed for {col}: {e}")
            sensitivities.append(0.0)

    logger.info(f"✅ Fingerprinting complete!")
    logger.info(
        f"   Effective features: {sum(1 for imp in sensitivities if imp > threshold)}/{len(sensitivities)}"
    )

    return tested_features, sensitivities
