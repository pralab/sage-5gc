import os
import json
import pandas as pd
import logging
from joblib import dump, load
from sklearn.impute import SimpleImputer

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Function to drop IP columns from a DataFrame unusefull for intrusion detection
def drop_ip_columns(df):
    """
    Drops IP-related columns from the given DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing network packet data.

    Returns:
    pd.DataFrame: DataFrame with IP-related columns removed.
    """
    # IP-related columns not useful for intrusion detection
    ip_col_to_drop = ['ip.src', 'ip.dst', 'ip.dst_host', 'ip.host', 'ip.addr', 'ip.src_host']
    df_dropped = df.drop(columns=ip_col_to_drop, errors='ignore')
    logger.debug(f"Remaining DataFrame shape: {df_dropped.shape}")

    return df_dropped


# Function to drop constant columns from of IP from dataframe
def drop_constant_ip_columns(df):
    """
    Drops constant IP-related columns from the given DataFrame.
    Parameters:
        df (pd.DataFrame): DataFrame containing network packet data.
    Returns:
        pd.DataFrame: DataFrame with constant IP-related columns removed.
    """
    # Check for constant IP-related columns not useful for intrusion detection
    constant_ip_col_to_drop = []
    # IP-related columns starting with 'ip.'
    ip_columns = [col for col in df.columns if col.startswith('ip.')]
    for col in ip_columns:
        if df[col].nunique() == 1:
            constant_ip_col_to_drop.append(col)
    if constant_ip_col_to_drop:
        logger.debug(f"Dropping constant IP-related columns: {constant_ip_col_to_drop}")
        # Save the dropped columns names to a file json
        with open("preprocessing/models_preprocessing_new/constant_ip_columns_dropped.json", "w") as f:
            json.dump(constant_ip_col_to_drop, f, indent=2)
    df_dropped = df.drop(columns=constant_ip_col_to_drop, errors='ignore')
    logger.debug(f"Remaining DataFrame shape: {df_dropped.shape}")

    return df_dropped


# Function to drop UDP columns from a DataFrame unusefull for intrusion detection
def drop_udp_columns(df):
    """
    Drops UDP-related columns from the given DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing network packet data.

    Returns:
    pd.DataFrame: DataFrame with UDP-related columns removed.
    """
    # UDP-related columns not useful for intrusion detection
    udp_col_to_drop = ['udp.port', 'udp.srcport', 'udp.dstport',
                       'udp.stream', 'udp.time_delta',
                       'udp.time_relative', 'udp.payload']
    df_dropped = df.drop(columns=udp_col_to_drop, errors='ignore')
    logger.debug(f"Remaining DataFrame shape: {df_dropped.shape}")

    return df_dropped


# Function to drop PFCP columns from a DataFrame unusefull for intrusion detection
def drop_pfcp_columns(df):
    """
    Drops PFCP-related columns from the given DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing network packet data.

    Returns:
    pd.DataFrame: DataFrame with PFCP-related columns removed.
    """
    # PFCP-related columns not useful for intrusion detection
    pfcp_col_to_drop = ['pfcp.flow_desc', 'pfcp.network_instance']
    df_dropped = df.drop(columns=pfcp_col_to_drop, errors='ignore')
    logger.debug(f"Remaining DataFrame shape: {df_dropped.shape}")

    return df_dropped


# Function to convert boolean columns to integer (0/1)
def boolean_to_integer(df, imputer=None, fit_mode=False):
    """
    Convert boolean columns to integer WITH imputer for NaN handling.

    Strategy: Use SimpleImputer with 'most_frequent' strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    imputer : SimpleImputer or None
        Pre-fitted imputer (for test mode).
    fit_mode : bool
        If True, fit the imputer (training mode).

    Returns
    -------
    df : pd.DataFrame
        Transformed dataframe.
    imputer : SimpleImputer (only if fit_mode=True)
        Fitted imputer.
    """
    df = df.copy()

    bool_cols = [
        'ip.flags.df',
        'pfcp.ue_ip_address_flag.sd',
        'pfcp.f_teid_flags.ch',
        'pfcp.f_teid_flags.ch_id',
        'pfcp.f_teid_flags.v6',
        'pfcp.apply_action.buff',
        'pfcp.apply_action.drop',
        'pfcp.apply_action.forw',
        'pfcp.apply_action.nocp',
        'pfcp.s'
    ]
    existing_bool_cols = [col for col in bool_cols if col in df.columns]
    if not existing_bool_cols:
        return (df, imputer) if fit_mode else df
    # Convert True/False to 1/0 (NaN stays NaN)
    for col in existing_bool_cols:
        df[col] = df[col].map({True: 1, False: 0})  # NaN rimane NaN

    # Imputation
    if fit_mode:
        # TRAINING: Fit imputer
        imputer = SimpleImputer(strategy='most_frequent')
        df[existing_bool_cols] = imputer.fit_transform(df[existing_bool_cols])
        logger.debug(f"✅ Fitted boolean imputer on {len(existing_bool_cols)} columns")
        logger.debug(f"   Learned values: {dict(zip(existing_bool_cols, imputer.statistics_))}")

        return df, imputer
    else:
        # TEST: Apply pre-fitted imputer
        if imputer is None:
            raise ValueError("❌ Imputer must be provided in test mode (fit_mode=False)")
        df[existing_bool_cols] = imputer.transform(df[existing_bool_cols])
        logger.debug(f"✅ Applied boolean imputer to {len(existing_bool_cols)} columns")

        return df


# Function to convert timestamp columns to unix like integer
def timestamp_to_integer(df):
    """
    Converts timestamp columns in the DataFrame to unix-like integer.
    Parameters:
        df (pd.DataFrame): DataFrame containing network packet data.
    Returns:
        pd.DataFrame: DataFrame with timestamp columns converted to integer.
    """
    df = df.copy()
    timestamp_cols = ['pfcp.time_of_first_packet',
                      'pfcp.time_of_last_packet',
                      'pfcp.end_time',
                      'pfcp.recovery_time_stamp',]
    for col in timestamp_cols:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce').astype('int64') // 10**9
            logger.debug(f"Converted timestamp column to integer: {col}")
        except Exception as e:
            logger.warning(f"Could not convert column {col} to integer timestamp: {e}")

    return df


# Function to convert hex values to integer
def hex_to_integer(df):
    """
    Converts hexadecimal columns in the DataFrame to integer.
    Parameters:
        df (pd.DataFrame): DataFrame containing network packet data.
    Returns:
        pd.DataFrame: DataFrame with hexadecimal columns converted to integer.
    """
    df = df.copy()
    hex_cols = ['ip.dsfield',
                'ip.flags',
                'ip.id',
                'ip.checksum',
                'udp.checksum',
                'pfcp.f_teid.teid',
                'pfcp.flags',
                'pfcp.outer_hdr_creation.teid',
                'pfcp.seid']
    for col in hex_cols:
        # Check if column exists in dataframe, it may not be dropped before if it was constant
        if col not in df.columns:
            logger.debug(f"⏭️ Skipping {col} (not in dataframe)")
            continue
        try:
            df[col] = df[col].apply(lambda x: int(x, 16) if isinstance(x, str) and x.startswith('0x') else x)
            logger.debug(f"Converted hexadecimal column to integer: {col}")
        except Exception as e:
            logger.warning(f"Could not convert column {col} from hex to integer: {e}")

    return df


# Function to convert IP addresses to integer
def ip_to_integer(df):
    """
    Converts IP address columns in the DataFrame to integer.
    Parameters:
        df (pd.DataFrame): DataFrame containing network packet data.
    Returns:
        pd.DataFrame: DataFrame with IP address columns converted to integer.
    """
    df = df.copy()
    ip_cols = ['pfcp.f_seid.ipv4',
               'pfcp.f_teid.ipv4_addr',
               'pfcp.node_id_ipv4',
               'pfcp.outer_hdr_creation.ipv4',
               'pfcp.ue_ip_addr_ipv4']
    for col in ip_cols:
        try:
            df[col] = df[col].apply(lambda x: int.from_bytes(map(int, x.split('.')), 'big') if isinstance(x, str) else x)
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

    # ============================================================
    # PFCP FIELD CATEGORIZATION
    # ============================================================

    # Counters/measurements → median strategy
    counter_cols = [
        'pfcp.duration_measurement',
        'pfcp.volume_measurement.tovol',
        'pfcp.volume_measurement.tonop',
        'pfcp.volume_measurement.dlvol',
        'pfcp.volume_measurement.dlnop',
        'pfcp.response_time',
        'pfcp.f_teid.teid',
        'pfcp.outer_hdr_creation.teid',
        'pfcp.outer_hdr_creation.ipv4',
        'pfcp.seid',
        'pfcp.f_teid.ipv4_addr',
        'pfcp.f_seid.ipv4',
        'pfcp.node_id_ipv4',
        'pfcp.ue_ip_addr_ipv4',
    ]

    # Flags/categorical → most_frequent strategy
    flag_cols = [
        'pfcp.user_id.imei',
        'pfcp.precedence',
        'pfcp.pdn_type',
        'pfcp.flow_desc_len',
        'pfcp.source_interface',
        'pfcp.dst_interface',
        'pfcp.node_id_type',
        'pfcp.pdr_id',
        'pfcp.cause',
        'pfcp.response_to',
        'pfcp.ie_type',
        'pfcp.ie_len',
    ]

    # Filter to existing columns
    counter_cols = [c for c in counter_cols if c in df.columns]
    flag_cols = [c for c in flag_cols if c in df.columns]
    # IMPUTATION
    if fit_mode:
        # TRAINING: Fit imputers
        if counter_cols:
            imputer_counters = SimpleImputer(strategy='median')
            df[counter_cols] = imputer_counters.fit_transform(df[counter_cols])
            logger.info(f"Fitted PFCP counter imputer on {len(counter_cols)} columns")
        if flag_cols:
            imputer_flags = SimpleImputer(strategy='most_frequent')
            df[flag_cols] = imputer_flags.fit_transform(df[flag_cols])
            logger.info(f"Fitted PFCP flag imputer on {len(flag_cols)} columns")

        return df, imputer_counters, imputer_flags

    else:
        # TEST: Apply pre-fitted imputers
        if counter_cols and imputer_counters is not None:
            df[counter_cols] = imputer_counters.transform(df[counter_cols])
            logger.info(f"Applied PFCP counter imputer to {len(counter_cols)} columns")
        if flag_cols and imputer_flags is not None:
            df[flag_cols] = imputer_flags.transform(df[flag_cols])
            logger.info(f"Applied PFCP flag imputer to {len(flag_cols)} columns")

        return df


# Preprocessing pipeline for training data
def preprocessing_pipeline_train(
    df_train: pd.DataFrame,
    output_dir: str,
) -> pd.DataFrame:
    """
    Preprocessing pipeline for training data.
    Parameters:
        df_train (pd.DataFrame): Training DataFrame.
        output_dir (str): Directory to save processed data.
    Returns:
        pd.DataFrame: Processed training DataFrame.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Drop IP columns
    df_train_processed = drop_ip_columns(df_train)
    # Drop UDP columns
    df_train_processed = drop_udp_columns(df_train_processed)
    # Drop PFCP columns
    df_train_processed = drop_pfcp_columns(df_train_processed)
    # Drop constant IP columns
    df_train_processed = drop_constant_ip_columns(df_train_processed)
    # Convert boolean columns to integer
    df_train_processed, boolean_imputer = boolean_to_integer(df_train_processed, fit_mode=True)
    os.makedirs("preprocessing/models_preprocessing_new", exist_ok=True)
    dump(boolean_imputer, "preprocessing/models_preprocessing_new/boolean_imputer.pkl")
    # Convert timestamp columns to integer
    df_train_processed = timestamp_to_integer(df_train_processed)
    # Convert hex columns to integer
    df_train_processed = hex_to_integer(df_train_processed)
    # Convert IP address columns to integer
    df_train_processed = ip_to_integer(df_train_processed)

    df_train_processed, pfcp_counter_imputer, pfcp_flag_imputer = impute_pfcp_fields(
        df_train_processed,
        fit_mode=True
    )
    # SAVE PFCP imputers
    dump(pfcp_counter_imputer, "preprocessing/models_preprocessing_new/pfcp_counter_imputer.pkl")
    dump(pfcp_flag_imputer, "preprocessing/models_preprocessing_new/pfcp_flag_imputer.pkl")
    logger.info("💾 Saved PFCP imputers")

    # Saving final processed training data
    output_path = os.path.join(output_dir, "train_dataset_processed.csv")
    df_train_processed.to_csv(output_path, sep=";", index=False)

    return df_train_processed


# Preprocessing pipeline for test data
def preprocessing_pipeline_test(
    df_test: pd.DataFrame,
    output_dir: str,
) -> pd.DataFrame:
    """
    Preprocessing pipeline for training data.
    Parameters:
        df_test (pd.DataFrame): Test DataFrame (also a single sample).
        output_dir (str): Directory to save processed data.
    Returns:
        pd.DataFrame: Processed training DataFrame/single sample.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Drop IP columns
    df_test_processed = drop_ip_columns(df_test)
    # Drop UDP columns
    df_test_processed = drop_udp_columns(df_test_processed)
    # Drop PFCP columns
    df_test_processed = drop_pfcp_columns(df_test_processed)
    # Drop constant IP columns -> No sense if we have a single sample
    # df_test_processed = drop_constant_ip_columns(df_test_processed)
    # In test case it's better to recover the dropped constant IP columns from training if any
    if os.path.exists("preprocessing/models_preprocessing_new/constant_ip_columns_dropped.json"):
        with open("preprocessing/models_preprocessing_new/constant_ip_columns_dropped.json", "r") as f:
            constant_ip_col_to_drop = json.load(f)
        df_test_processed = df_test_processed.drop(columns=constant_ip_col_to_drop, errors='ignore')
        logger.debug(f"Dropped constant IP-related columns from test data: {constant_ip_col_to_drop}")
    # Convert boolean columns to integer
    if not os.path.exists("preprocessing/models_preprocessing_new/boolean_imputer.pkl"):
        raise FileNotFoundError("❌ boolean_imputer.pkl not found!  Run training first.")
    boolean_imputer = load("preprocessing/models_preprocessing_new/boolean_imputer.pkl")
    df_test_processed = boolean_to_integer(
        df_test_processed,
        imputer=boolean_imputer,
        fit_mode=False
    )
    # Convert timestamp columns to integer
    df_test_processed = timestamp_to_integer(df_test_processed)
    # Convert hex columns to integer
    df_test_processed = hex_to_integer(df_test_processed)
    # Convert IP address columns to integer
    df_test_processed = ip_to_integer(df_test_processed)

    if not os.path.exists("preprocessing/models_preprocessing_new/pfcp_counter_imputer.pkl"):
        raise FileNotFoundError("❌ pfcp_counter_imputer.pkl not found!")
    if not os.path.exists("preprocessing/models_preprocessing_new/pfcp_flag_imputer.pkl"):
        raise FileNotFoundError("❌ pfcp_flag_imputer.pkl not found!")

    pfcp_counter_imputer = load("preprocessing/models_preprocessing_new/pfcp_counter_imputer.pkl")
    pfcp_flag_imputer = load("preprocessing/models_preprocessing_new/pfcp_flag_imputer.pkl")

    df_test_processed = impute_pfcp_fields(
        df_test_processed,
        fit_mode=False,
        imputer_counters=pfcp_counter_imputer,
        imputer_flags=pfcp_flag_imputer
    )

    # Saving final processed training data
    output_path = os.path.join(output_dir, "test_dataset_processed.csv")
    df_test_processed.to_csv(output_path, sep=";", index=False)

    return df_test_processed


'''if __name__ == "__main__":
    # Example usage
    df_train = pd.read_csv("new_datasets/no_tcp_icmp/train_dataset.csv", sep=";", low_memory=False)
    df_test = pd.read_csv("new_datasets/no_tcp_icmp/test_dataset.csv", sep=";", low_memory=False)

    df_train_processed = preprocessing_pipeline_train(df_train, output_dir="processed_data")
    df_test_processed = preprocessing_pipeline_test(df_test, output_dir="processed_data")

    df_train_processed.to_csv("processed_data/train_dataset_processed.csv", sep=";", index=False)
    df_test_processed.to_csv("processed_data/test_dataset_processed.csv", sep=";", index=False)'''
