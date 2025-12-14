import ipaddress
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger.setLevel(logging.INFO)


def drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Useless columns or logic redundant columns
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
        "udp.length",
        "udp.srcport",
        "udp.dstport",
        "udp.stream",
        "udp.time_delta",
        "udp.time_relative",
        "udp.payload",
        # PFCP-related columns
        "pfcp.flow_desc",
        "pfcp.ie_len",
        "pfcp.length",
        "pfcp.network_instance",
        "pfcp.time_of_first_packet",
        "pfcp.time_of_last_packet",
        "pfcp.end_time",
        "pfcp.recovery_time_stamp",
    ]

    df_dropped = df.drop(columns=col_to_drop, errors="ignore")

    return df_dropped


def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    const_columns_path = (
        Path(__file__).parent / "models_preprocessing/constant_columns_dropped.json"
    )

    if const_columns_path.exists():
        with const_columns_path.open("r") as f:
            const_col_to_drop = json.load(f)
        return df.drop(columns=const_col_to_drop, errors="ignore")

    const_col_to_drop = []

    # IP-related columns starting with 'ip.'
    ip_columns = [col for col in df.columns if col.startswith("ip.")]
    for col in ip_columns:
        if df[col].nunique() == 1:
            const_col_to_drop.append(col)

    # UDP-related columns starting with 'udp.'
    udp_columns = [col for col in df.columns if col.startswith("udp.")]
    for col in udp_columns:
        if df[col].nunique() == 1:
            const_col_to_drop.append(col)

    # PFCP-related columns starting with 'pfcp.'
    pfcp_columns = [col for col in df.columns if col.startswith("pfcp.")]
    for col in pfcp_columns:
        if df[col].nunique() == 1:
            const_col_to_drop.append(col)

    if const_col_to_drop:
        with const_columns_path.open("w") as f:
            json.dump(const_col_to_drop, f, indent=4)

    return df.drop(columns=const_col_to_drop, errors="ignore")


def _detect_type(series: pd.Series) -> str:
    sample_values = series.dropna().unique()
    if len(sample_values) == 0:
        return "empty"

    samples_values = sample_values[:5]

    if series.dtype == bool or set(samples_values).issubset(
        {True, False, "True", "False", 0, 1}
    ):
        return "bool"

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

    return "skip"


def convert_to_numeric(df: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    cat_cols = []
    for col in df.columns:
        dtype = _detect_type(df[col])

        if dtype == "empty":
            logger.debug(f"Column '{col}' missing type detected as '{dtype}'.")
            continue

        if dtype == "skip":
            continue

        cat_cols.append((col, dtype))

        if dtype == "bool":
            df[col] = (
                df[col]
                .map({True: 1, False: 0, "True": 1, "False": 0})
                .astype("category")
            )

        elif dtype == "hex":
            df[col] = (
                df[col]
                .apply(
                    lambda x: int(str(x), 16)
                    if pd.notnull(x) and str(x).startswith("0x")
                    else np.nan
                )
                .astype("category")
            )

        elif dtype == "ip_address":
            df[col] = (
                df[col]
                .apply(
                    lambda x: int(ipaddress.ip_address(x)) if pd.notnull(x) else np.nan
                )
                .astype("category")
            )

        elif dtype == "datetime":
            df[col] = pd.to_datetime(df[col], errors="coerce").astype("int64") // 10**9
            df[col] = df[col].astype("category")

    return df, cat_cols


def restore_categoric_columns(
    df: pd.DataFrame, cat_cols: list[tuple[str, str]]
) -> pd.DataFrame:
    for col, dtype in cat_cols:
        if dtype == "bool":
            df[col] = df[col].map({1: True, 0: False}).astype("category")

        elif dtype == "hex":
            df[col] = df[col].apply(lambda x: hex(int(x)) if pd.notnull(x) else np.nan)
            df[col] = df[col].astype("category")

        elif dtype == "ip_address":
            df[col] = df[col].apply(
                lambda x: str(ipaddress.ip_address(int(x))) if pd.notnull(x) else np.nan
            )
            df[col] = df[col].astype("category")

        elif dtype == "datetime":
            df[col] = pd.to_datetime(df[col].astype("float"), unit="s", errors="coerce")
            df[col] = df[col].astype("category")

    return df


def load_imputers(random_state: int = 42) -> tuple[SimpleImputer, IterativeImputer]:
    simple_imputer_path = (
        Path(__file__).parent / "models_preprocessing/simple_imputer.pkl"
    )
    iter_imputer_path = Path(__file__).parent / "models_preprocessing/iter_imputer.pkl"

    if not simple_imputer_path.exists():
        simple_imputer = SimpleImputer(strategy="most_frequent")
        iter_imputer = IterativeImputer(
            estimator=RandomForestRegressor(
                n_jobs=-1, max_depth=20, n_estimators=50, random_state=random_state
            ),
            initial_strategy="median",
            max_iter=10,
            random_state=random_state,
            skip_complete=True,
        )
    else:
        simple_imputer: SimpleImputer = joblib.load(simple_imputer_path)
        iter_imputer: IterativeImputer = joblib.load(iter_imputer_path)

    return simple_imputer, iter_imputer
