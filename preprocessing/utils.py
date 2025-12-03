import json
import logging
from pathlib import Path

from joblib import dump, load
import pandas as pd
from sklearn.impute import SimpleImputer

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def drop_columns(df):
    """
    Drops IP/UDP/PFCP-related columns from the given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing network packet data.

    Returns
    -------
    pd.DataFrame
        DataFrame with IP-related columns removed.
    """
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
    ]

    df_dropped = df.drop(columns=col_to_drop, errors="ignore")
    logger.debug(f"Remaining DataFrame shape: {df_dropped.shape}")

    return df_dropped


def drop_constant_columns(df):
    """
    Drops constant IP-related columns from the given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing network packet data.

    Returns
    -------
    pd.DataFrame
        DataFrame with constant IP-related columns removed.
    """
    # Check for constant IP-related columns not useful for intrusion detection
    constant_col_to_drop = []

    # IP-related columns starting with 'ip.'
    ip_columns = [col for col in df.columns if col.startswith("ip.")]
    for col in ip_columns:
        if df[col].nunique() == 1:
            constant_col_to_drop.append(col)

    # PFCP-related columns starting with 'pfcp.'
    pfcp_columns = [col for col in df.columns if col.startswith("pfcp.")]
    for col in pfcp_columns:
        if df[col].nunique() == 1:
            constant_col_to_drop.append(col)

    if constant_col_to_drop:
        logger.debug(f"Dropping constant columns: {constant_col_to_drop}")

        with (
            Path(__file__).parent
            / "models_preprocessing/constant_columns_dropped.json"
        ).open("w") as f:
            json.dump(constant_col_to_drop, f, indent=4)

    return df.drop(columns=constant_col_to_drop, errors="ignore")


def boolean_to_numeric(df: pd.DataFrame):
    """
    Convert boolean columns to integer.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    df : pd.DataFrame
        Transformed dataframe.
    """
    df = df.copy()

    bool_cols = [
        "ip.flags.df",
        "pfcp.ue_ip_address_flag.sd",
        "pfcp.f_teid_flags.ch",
        "pfcp.f_teid_flags.ch_id",
        "pfcp.f_teid_flags.v6",
        "pfcp.apply_action.buff",
        "pfcp.apply_action.drop",
        "pfcp.apply_action.forw",
        "pfcp.apply_action.nocp",
        "pfcp.s",
    ]
    existing_bool_cols = [col for col in bool_cols if col in df.columns]

    for col in existing_bool_cols:
        df[col] = df[col].map({True: 1, False: 0})
        df[col] = df[col].fillna(-1).astype(int)

    return df


def timestamp_to_numeric(df):
    """
    Converts timestamp columns in the DataFrame to unix-like integer.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing network packet data.

    Returns
    -------
    pd.DataFrame
        DataFrame with timestamp columns converted to integer.
    """
    df = df.copy()
    timestamp_cols = [
        "pfcp.time_of_first_packet",
        "pfcp.time_of_last_packet",
        "pfcp.end_time",
        "pfcp.recovery_time_stamp",
    ]
    for col in timestamp_cols:
        try:
            df[col] = pd.to_datetime(df[col], errors="coerce").astype("int64") // 10**9
            logger.debug(f"Converted timestamp column to integer: {col}")
        except Exception as e:
            logger.warning(f"Could not convert column {col} to integer timestamp: {e}")

    return df


def hex_to_integer(df):
    """
    Converts hexadecimal columns in the DataFrame to integer.
    Parameters:
        df (pd.DataFrame): DataFrame containing network packet data.
    Returns:
        pd.DataFrame: DataFrame with hexadecimal columns converted to integer.
    """
    df = df.copy()
    hex_cols = [
        "ip.dsfield",
        "ip.flags",
        "ip.id",
        "ip.checksum",
        "udp.checksum",
        "pfcp.f_teid.teid",
        "pfcp.flags",
        "pfcp.outer_hdr_creation.teid",
        "pfcp.seid",
    ]
    for col in hex_cols:
        # Check if column exists in dataframe, it may not be dropped before if it was constant
        if col not in df.columns:
            logger.debug(f"Skipping {col} (not in dataframe)")
            continue
        try:
            df[col] = df[col].apply(
                lambda x: int(x, 16) if isinstance(x, str) and x.startswith("0x") else x
            )
            logger.debug(f"Converted hexadecimal column to integer: {col}")
        except Exception as e:
            logger.warning(f"Could not convert column {col} from hex to integer: {e}")

    return df


def ip_to_numeric(df):
    """
    Converts IP address columns in the DataFrame to integer.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing network packet data.

    Returns
    -------
    pd.DataFrame
        DataFrame with IP address columns converted to integer.
    """
    df = df.copy()
    ip_cols = [
        "pfcp.f_seid.ipv4",
        "pfcp.f_teid.ipv4_addr",
        "pfcp.node_id_ipv4",
        "pfcp.outer_hdr_creation.ipv4",
        "pfcp.ue_ip_addr_ipv4",
    ]
    for col in ip_cols:
        try:
            df[col] = df[col].apply(
                lambda x: int.from_bytes(map(int, x.split(".")), "big")
                if isinstance(x, str)
                else x
            )
            logger.debug(f"Converted IP address column to integer: {col}")
        except Exception as e:
            logger.warning(f"Could not convert column {col} from IP to integer: {e}")

    return df


def impute_pfcp_fields(df, fit_mode=False, imputer_counters=None, imputer_flags=None):
    """
    Impute PFCP-specific columns with semantically appropriate strategies.

    Strategy:
    - Counters/measurements (volume, duration) → median (robust)
    - Flags/types (precedence, pdn_type, source_interface) → most_frequent
    - IDs (IMEI, TEID, SEID) → most_frequent
    - Timestamps → median (or 0 if all NaN)
    - IP addresses → median (already converted to int)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    fit_mode : bool
        If True, fit imputers (training mode).
    imputer_counters : SimpleImputer or None
        Pre-fitted imputer for counter fields.
    imputer_flags : SimpleImputer or None
        Pre-fitted imputer for flag/categorical fields.

    Returns
    -------
    df : pd.DataFrame
        Dataframe with PFCP fields imputed.
    imputer_counters : SimpleImputer (only if fit_mode=True)
        Fitted imputer for counters.
    imputer_flags : SimpleImputer (only if fit_mode=True)
        Fitted imputer for flags.
    """
    df = df.copy()

    # Counters/measurements -> median strategy
    counter_cols = [
        "pfcp.duration_measurement",
        "pfcp.volume_measurement.tovol",
        "pfcp.volume_measurement.tonop",
        "pfcp.volume_measurement.dlvol",
        "pfcp.volume_measurement.dlnop",
        "pfcp.response_time",
        "pfcp.f_teid.teid",
        "pfcp.outer_hdr_creation.teid",
        "pfcp.outer_hdr_creation.ipv4",
        "pfcp.seid",
        "pfcp.f_teid.ipv4_addr",
        "pfcp.f_seid.ipv4",
        "pfcp.node_id_ipv4",
        "pfcp.ue_ip_addr_ipv4",
    ]

    # Flags/categorical -> most_frequent strategy
    flag_cols = [
        "pfcp.user_id.imei",
        "pfcp.precedence",
        "pfcp.pdn_type",
        "pfcp.flow_desc_len",
        "pfcp.source_interface",
        "pfcp.dst_interface",
        "pfcp.node_id_type",
        "pfcp.pdr_id",
        "pfcp.cause",
        "pfcp.response_to",
        "pfcp.ie_type",
        "pfcp.ie_len",
    ]

    # Filter to existing columns
    counter_cols = [c for c in counter_cols if c in df.columns]
    flag_cols = [c for c in flag_cols if c in df.columns]

    if fit_mode:
        # TRAINING: Fit imputers
        if counter_cols:
            imputer_counters = SimpleImputer(strategy="median")
            df[counter_cols] = imputer_counters.fit_transform(df[counter_cols])
            logger.debug(f"Fitted PFCP counter imputer on {len(counter_cols)} columns")

        if flag_cols:
            imputer_flags = SimpleImputer(strategy="most_frequent")
            df[flag_cols] = imputer_flags.fit_transform(df[flag_cols])
            logger.debug(f"Fitted PFCP flag imputer on {len(flag_cols)} columns")

        return df, imputer_counters, imputer_flags

    else:
        # TEST: Apply pre-fitted imputers
        if counter_cols and imputer_counters is not None:
            df[counter_cols] = imputer_counters.transform(df[counter_cols])
            logger.debug(f"Applied PFCP counter imputer to {len(counter_cols)} columns")

        if flag_cols and imputer_flags is not None:
            df[flag_cols] = imputer_flags.transform(df[flag_cols])
            logger.debug(f"Applied PFCP flag imputer to {len(flag_cols)} columns")

        return df


def preprocessing_pipeline_train(
    df_train: pd.DataFrame, output_dir: str | None
) -> pd.DataFrame:
    """
    Preprocessing pipeline for training data.

    Parameters
    ----------
    df_train: pd.DataFrame
        Training DataFrame.
    output_dir: str | None
        Directory to save processed data.

    Returns
    -------
    pd.DataFrame
        Processed training DataFrame.
    """
    # Ensure output directory exists
    if output_dir is not None:
        Path(output_dir).mkdir(exist_ok=True)
    (Path(__file__).parent / "models_preprocessing").mkdir(exist_ok=True)

    df_train = df_train.copy()

    # ------------------------------
    # [Step 1] Drop useless columns
    # ------------------------------
    df_train_processed = drop_columns(df_train)
    df_train_processed = drop_constant_columns(df_train_processed)

    # ----------------------------------------
    # [Step 2] Convert non-numeric to numeric
    # ----------------------------------------
    df_train_processed = boolean_to_numeric(df_train_processed)
    df_train_processed = timestamp_to_numeric(df_train_processed)
    df_train_processed = hex_to_integer(df_train_processed)
    df_train_processed = ip_to_numeric(df_train_processed)
    df_train_processed, pfcp_counter_imputer, pfcp_flag_imputer = impute_pfcp_fields(
        df_train_processed, fit_mode=True
    )
    dump(
        pfcp_counter_imputer,
        Path(__file__).parent / "models_preprocessing/pfcp_counter_imputer.pkl",
    )
    dump(
        pfcp_flag_imputer,
        Path(__file__).parent / "models_preprocessing/pfcp_flag_imputer.pkl",
    )

    # -----------------------------
    # [Step 3] Save processed data
    # -----------------------------
    if output_dir is not None:
        df_train_processed.to_csv(
            Path(output_dir) / "train_dataset_processed.csv", sep=";", index=False
        )

    return df_train_processed


def preprocessing_pipeline_test(
    df_test: pd.DataFrame, output_dir: str | None
) -> pd.DataFrame:
    """
    Preprocessing pipeline for training data.

    Parameters
    ----------
    df_test : pd.DataFrame
        Test DataFrame (also a single sample).
    output_dir : str
        Directory to save processed data.

    Returns
    -------
    pd.DataFrame
        Processed training DataFrame/single sample.
    """
    # Ensure output directory exists
    if output_dir is not None:
        Path(output_dir).mkdir(exist_ok=True)

    df_test = df_test.copy()

    # ------------------------------
    # [Step 1] Drop useless columns
    # ------------------------------
    df_test_processed = drop_columns(df_test)

    path_constant_cols = (
        Path(__file__).parent / "models_preprocessing/constant_columns_dropped.json"
    )
    if path_constant_cols.exists():
        with path_constant_cols.open("r") as f:
            constant_ip_col_to_drop = json.load(f)

        df_test_processed = df_test_processed.drop(
            columns=constant_ip_col_to_drop, errors="ignore"
        )

    # ----------------------------------------
    # [Step 2] Convert non-numeric to numeric
    # ----------------------------------------
    df_test_processed = boolean_to_numeric(df_test_processed)
    df_test_processed = timestamp_to_numeric(df_test_processed)
    df_test_processed = hex_to_integer(df_test_processed)
    df_test_processed = ip_to_numeric(df_test_processed)

    # Impute PFCP-specific fields
    path_pfcp_counter_imputer = (
        Path(__file__).parent / "models_preprocessing/pfcp_counter_imputer.pkl"
    )
    path_pfcp_flag_imputer = (
        Path(__file__).parent / "models_preprocessing/pfcp_flag_imputer.pkl"
    )

    if not path_pfcp_counter_imputer.exists():
        raise FileNotFoundError("pfcp_counter_imputer.pkl not found!")

    if not path_pfcp_flag_imputer.exists():
        raise FileNotFoundError("pfcp_flag_imputer.pkl not found!")

    pfcp_counter_imputer = load(path_pfcp_counter_imputer)
    pfcp_flag_imputer = load(path_pfcp_flag_imputer)

    df_test_processed = impute_pfcp_fields(
        df_test_processed,
        fit_mode=False,
        imputer_counters=pfcp_counter_imputer,
        imputer_flags=pfcp_flag_imputer,
    )

    # -----------------------------
    # [Step 3] Save processed data
    # -----------------------------
    if output_dir is not None:
        df_test_processed.to_csv(
            Path(output_dir) / "test_dataset_processed.csv", sep=";", index=False
        )

    return df_test_processed
