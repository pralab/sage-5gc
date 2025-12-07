import ipaddress
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger.setLevel(logging.INFO)


def _drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_to_drop = [
        # Wireshark
        "frame.number",
        # IP-related columns
        "ip.src",
        "ip.dst",
        "ip.dst_host",
        "ip.host",
        "ip.addr",
        "ip.src_host",
        # UDP-related columns
        "udp.port",
        "udp.srcport",
        "udp.dstport",
        "udp.stream",
        "udp.time_delta",
        "udp.time_relative",
        "udp.payload",
        # PFCP-related columns
        "pfcp.flow_desc",
        "pfcp.network_instance",
        "pfcp.time_of_first_packet",
        "pfcp.time_of_last_packet",
        "pfcp.end_time",
        "pfcp.recovery_time_stamp",
    ]

    df_dropped = df.drop(columns=col_to_drop, errors="ignore")

    return df_dropped


def _drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    constant_col_to_drop = []

    # IP-related columns starting with 'ip.'
    ip_columns = [col for col in df.columns if col.startswith("ip.")]
    for col in ip_columns:
        if df[col].nunique() == 1:
            constant_col_to_drop.append(col)

    # UDP-related columns starting with 'udp.'
    udp_columns = [col for col in df.columns if col.startswith("udp.")]
    for col in udp_columns:
        if df[col].nunique() == 1:
            constant_col_to_drop.append(col)

    # PFCP-related columns starting with 'pfcp.'
    pfcp_columns = [col for col in df.columns if col.startswith("pfcp.")]
    for col in pfcp_columns:
        if df[col].nunique() == 1:
            constant_col_to_drop.append(col)

    if constant_col_to_drop:
        with (
            Path(__file__).parent / "models_preprocessing/constant_columns_dropped.json"
        ).open("w") as f:
            json.dump(constant_col_to_drop, f, indent=4)

    return df.drop(columns=constant_col_to_drop, errors="ignore")


def _detect_type(series: pd.Series) -> str:
    sample_values = series.dropna().unique()
    if len(sample_values) == 0:
        return "empty"

    samples_values = sample_values[:5]

    if series.dtype == bool or set(samples_values).issubset(
        {True, False, "True", "False", 0, 1}
    ):
        return "bool"

    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    first_val = str(samples_values[0])
    if isinstance(first_val, str) and first_val.startswith("0x"):
        try:
            int(first_val, 16)
            return "hex"
        except:  # noqa: E722
            pass

    try:
        ipaddress.ip_address(first_val)
        return "ip_address"
    except:  # noqa: E722
        pass

    try:
        pd.to_datetime(first_val)
        if len(first_val) > 10:
            return "datetime"
    except:  # noqa: E722
        pass

    return "undefined"


def _convert_to_numeric(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    cat_cols = []
    for col in df.columns:
        dtype = _detect_type(df[col])

        if dtype == "bool":
            cat_cols.append(col)
            df[col] = df[col].map({True: 1, False: 0, "True": 1, "False": 0})

        elif dtype == "hex":
            cat_cols.append(col)
            df[col] = df[col].apply(
                lambda x: int(str(x), 16)
                if pd.notnull(x) and str(x).startswith("0x")
                else np.nan
            )

        elif dtype == "ip_address":
            cat_cols.append(col)
            df[col] = df[col].apply(
                lambda x: int(ipaddress.ip_address(x)) if pd.notnull(x) else np.nan
            )

        elif dtype == "datetime":
            cat_cols.append(col)
            df[col] = pd.to_datetime(df[col], errors="coerce").astype("int64") // 10**9

        elif dtype in ["empty", "undefined"]:
            logger.debug(f"Column '{col}' missing type detected as '{dtype}'.")

    return df, cat_cols


def preprocessing_train(
    df_train: pd.DataFrame,
    output_path: Path | str | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Preprocessing pipeline for training data.

    Parameters
    ----------
    df_train: pd.DataFrame
        Training DataFrame.
    output_path: Path | str | None
        Path to save processed data.
    random_state: int
        Random state for reproducibility.

    Returns
    -------
    pd.DataFrame
        Processed training DataFrame.
    """
    (Path(__file__).parent / "models_preprocessing").mkdir(exist_ok=True)

    df = df_train.copy()

    # ------------------------------
    # [Step 1] Drop useless columns
    # ------------------------------
    df_processed = _drop_useless_columns(df)
    df_processed = _drop_constant_columns(df_processed)

    # ----------------------------------------
    # [Step 2] Convert non-numeric to numeric
    # ----------------------------------------
    df_processed, cat_cols = _convert_to_numeric(df_processed)

    # --------------------
    # [Step 3] Imputation
    # --------------------
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_jobs=-1, max_depth=10, n_estimators=50),
        initial_strategy="most_frequent",
        max_iter=10,
        random_state=random_state,
    )

    df_imputed_raw = imputer.fit_transform(df_processed)
    df_imputed = pd.DataFrame(
        df_imputed_raw, columns=df_processed.columns, index=df_processed.index
    )

    # Adding int type to categorical columns
    cat_cols.extend(["pfcp.duration_measurement"])

    for col in cat_cols:
        df_imputed[col] = df_imputed[col].round()

    # -----------------------------
    # [Step 4] Save processed data
    # -----------------------------
    joblib.dump(imputer, Path(__file__).parent / "models_preprocessing/imputer.pkl")

    if output_path is not None:
        Path(output_path.parent).mkdir(exist_ok=True)
        df_imputed.to_csv(output_path, sep=";")

    return df_imputed


def preprocessing_test(
    df_test: pd.DataFrame, output_path: Path | str | None = None
) -> pd.DataFrame:
    """
    Preprocessing pipeline for testing data / single sample.

    Parameters
    ----------
    df_test : pd.DataFrame
        Test DataFrame (also a single sample).
    output_path : Path | str | None
        Path to save processed data.

    Returns
    -------
    pd.DataFrame
        Processed training DataFrame/single sample.
    """
    df = df_test.copy()

    # ------------------------------
    # [Step 1] Drop useless columns
    # ------------------------------
    df_processed = _drop_useless_columns(df)

    constant_cols_path = (
        Path(__file__).parent / "models_preprocessing/constant_columns_dropped.json"
    )
    if not constant_cols_path.exists():
        raise SystemError("You have to preprocess training data before test data!")

    with constant_cols_path.open("r") as f:
        constant_ip_col_to_drop = json.load(f)

    df_processed = df_processed.drop(columns=constant_ip_col_to_drop, errors="ignore")

    # ----------------------------------------
    # [Step 2] Convert non-numeric to numeric
    # ----------------------------------------
    df_processed, cat_cols = _convert_to_numeric(df_processed)

    # --------------------
    # [Step 3] Imputation
    # --------------------
    imputer_path = Path(__file__).parent / "models_preprocessing/imputer.pkl"

    if not imputer_path.exists():
        raise SystemError("You have to preprocess training data before test data!")

    imputer: IterativeImputer = joblib.load(imputer_path)
    df_imputed_raw = imputer.transform(df_processed)
    df_imputed = pd.DataFrame(
        df_imputed_raw, columns=df_processed.columns, index=df_processed.index
    )

    # Adding int type to categorical columns
    cat_cols.extend(["pfcp.duration_measurement"])

    for col in cat_cols:
        df_imputed[col] = df_imputed[col].round()

    # -----------------------------
    # [Step 3] Save processed data
    # -----------------------------
    if output_path is not None:
        df_processed.to_csv(output_path, sep=";")

    return df_processed
