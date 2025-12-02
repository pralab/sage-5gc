import os
import json
import pandas as pd
import logging

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
    logger.info(f"Remaining DataFrame shape: {df_dropped.shape}")

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
        logger.info(f"Dropping constant IP-related columns: {constant_ip_col_to_drop}")
        # Save the dropped columns names to a file json
        with open("constant_ip_columns_dropped.json", "w") as f:
            json.dump(constant_ip_col_to_drop, f, indent=2)
    df_dropped = df.drop(columns=constant_ip_col_to_drop, errors='ignore')
    logger.info(f"Remaining DataFrame shape: {df_dropped.shape}")

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
    udp_col_to_drop = ['udp.port', 'udp.srcport', 'udp.dstport', 'udp.stream', 'udp.time_delta', 'udp.time_relative']
    df_dropped = df.drop(columns=udp_col_to_drop, errors='ignore')
    logger.info(f"Remaining DataFrame shape: {df_dropped.shape}")

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
    logger.info(f"Remaining DataFrame shape: {df_dropped.shape}")

    return df_dropped


# Function to convert boolean columns to integer (0/1)
def boolean_to_integer(df):
    """
    Converts boolean columns in the DataFrame to integer (0/1).
    Parameters:
        df (pd.DataFrame): DataFrame containing network packet data.
    Returns:
        pd.DataFrame: DataFrame with boolean columns converted to integer.
    """
    df = df.copy()
    bool_cols = ['ip.flags.df',
                 'pfcp.ue_ip_address_flag.sd',
                 'pfcp.f_teid_flags.ch',
                 'pfcp.f_teid_flags.ch_id',
                 'pfcp.f_teid_flags.v6',
                 'pfcp.apply_action.buff',
                 'pfcp.apply_action.drop',
                 'pfcp.apply_action.forw',
                 'pfcp.apply_action.nocp',
                 'pfcp.s']
    existing_bool_cols = [col for col in bool_cols if col in df.columns]
    if existing_bool_cols:
        # fill Riempi NaN with False (0)
        df[existing_bool_cols] = df[existing_bool_cols].fillna(False)
        df[existing_bool_cols] = df[existing_bool_cols].astype(int)
        logger.info(f"Converted {len(existing_bool_cols)} boolean columns to integer")

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
            logger.info(f"Converted timestamp column to integer: {col}")
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
    hex_cols = ['ip.dsfiled',
                'ip.flags',
                'ip.checksum',
                'udp.checksum',
                'pfcp.f_teid.teid',
                'pfcp.flags',
                'pfcp.outer_hdr_creation.teid',
                'pfcp.seid']
    for col in hex_cols:
        try:
            df[col] = df[col].apply(lambda x: int(x, 16) if isinstance(x, str) and x.startswith('0x') else x)
            logger.info(f"Converted hexadecimal column to integer: {col}")
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
               'pfcp.node_id.ipv4',
               'pfcp.outer_hdr_creation.ipv4_addr',
               'pfcp.ue_ip_addr_ipv4']
    for col in ip_cols:
        try:
            df[col] = df[col].apply(lambda x: int.from_bytes(map(int, x.split('.')), 'big') if isinstance(x, str) else x)
            logger.info(f"Converted IP address column to integer: {col}")
        except Exception as e:
            logger.warning(f"Could not convert column {col} from IP to integer: {e}")

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
    # Convert boolean columns to integer
    df_train_processed = boolean_to_integer(df_train_processed)
    # Convert timestamp columns to integer
    df_train_processed = timestamp_to_integer(df_train_processed)
    # Convert hex columns to integer
    df_train_processed = hex_to_integer(df_train_processed)
    # Convert IP address columns to integer
    df_train_processed = ip_to_integer(df_train_processed)
    # Drop constant IP columns
    df_train_processed = drop_constant_ip_columns(df_train_processed)

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
    # Convert boolean columns to integer
    df_test_processed = boolean_to_integer(df_test_processed)
    # Convert timestamp columns to integer
    df_test_processed = timestamp_to_integer(df_test_processed)
    # Convert hex columns to integer
    df_test_processed = hex_to_integer(df_test_processed)
    # Convert IP address columns to integer
    df_test_processed = ip_to_integer(df_test_processed)
    # Drop constant IP columns -> No sense if we have a single sample
    # df_test_processed = drop_constant_ip_columns(df_test_processed)
    # In test case it's better to recover the dropped constant IP columns from training if any
    if os.path.exists("constant_ip_columns_dropped.json"):
        with open("constant_ip_columns_dropped.json", "r") as f:
            constant_ip_col_to_drop = json.load(f)
        df_test_processed = df_test_processed.drop(columns=constant_ip_col_to_drop, errors='ignore')
        logger.info(f"Dropped constant IP-related columns from test data: {constant_ip_col_to_drop}")

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
