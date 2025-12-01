import json
import os
from pathlib import Path
import joblib
from joblib import dump
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# region --- PREPROCESSING STEPS ---


# ============================================================================
# STEP 2: TCP OPTIONS PARSING
# ============================================================================
'''def enrich_tcp_columns(input_file, output_file, chunksize=100_000):
    """
    Expand 'tcp.options' into binary indicator columns (mss, timestamp, sack, wscale).

    Parameters
    ----------
    input_file : str
        Path to cleaned CSV.
    output_file : str
        Output CSV containing expanded option flags.
    chunksize : int, optional
        Number of rows per chunk for streaming.

    Returns
    -------
    None
    """
    write_header = True
    for chunk in pd.read_csv(
        input_file, sep=";", chunksize=chunksize, low_memory=False
    ):
        if "tcp.options" in chunk.columns:
            chunk["tcp.options"] = chunk["tcp.options"].fillna("")
            # Binary features for TCP options
            chunk["tcp_opt_mss"] = (
                chunk["tcp.options"].str.contains("mss", case=False).astype(int)
            )
            chunk["tcp_opt_ts"] = (
                chunk["tcp.options"].str.contains("timestamp", case=False).astype(int)
            )
            chunk["tcp_opt_sack"] = (
                chunk["tcp.options"].str.contains("sack", case=False).astype(int)
            )
            chunk["tcp_opt_wscale"] = (
                chunk["tcp.options"].str.contains("wscale", case=False).astype(int)
            )
            chunk = chunk.drop(columns=["tcp.options"], errors="ignore")
        chunk.to_csv(
            output_file,
            sep=";",
            index=False,
            mode="w" if write_header else "a",
            header=write_header,
        )
        write_header = False'''
def enrich_tcp_columns(input_file, output_file, chunksize=100_000):
    """
    Expand 'tcp.options' into binary indicator columns (mss, timestamp, sack, wscale).

    Parameters
    ----------
    input_file : pd.DataFrame
        Cleaned CSV.
    output_file : str
        Output CSV containing expanded option flags.
    chunksize : int, optional
        Number of rows per chunk for streaming.

    Returns
    -------
    None
    """
    write_header = True
    chunk = input_file.copy()
    if "tcp.options" in chunk.columns:
        chunk["tcp.options"] = chunk["tcp.options"].fillna("")
        # Binary features for TCP options
        chunk["tcp_opt_mss"] = (
            chunk["tcp.options"].str.contains("mss", case=False).astype(int)
        )
        chunk["tcp_opt_ts"] = (
            chunk["tcp.options"].str.contains("timestamp", case=False).astype(int)
        )
        chunk["tcp_opt_sack"] = (
            chunk["tcp.options"].str.contains("sack", case=False).astype(int)
        )
        chunk["tcp_opt_wscale"] = (
            chunk["tcp.options"].str.contains("wscale", case=False).astype(int)
        )
        chunk = chunk.drop(columns=["tcp.options"], errors="ignore")
    chunk.to_csv(
        output_file,
        sep=";",
        index=False,
        mode="w" if write_header else "a",
        header=write_header,
    )
    write_header = False


# ============================================================================
# STEP 3: ADVANCED CLEANING
# ============================================================================
def drop_columns_chunked(input_file, output_file, is_attack=False, chunksize=100000):
    """
    Remove low-value packet fields and normalize PFCP hex fields into integers.

    Parameters
    ----------
    input_file : str
        Input CSV after TCP option parsing.
    output_file : str
        Cleaned CSV.
    is_attack : bool
        Flag kept for compatibility (not used internally).
    chunksize : int
        Chunk size for streaming I/O.

    Returns
    -------
    None
    """
    columns_to_delete = [
        "ip.hdr_len",
        "ip.len",
        "tcp.payload",
        "tcp.segment_data",
        "tcp.reassembled.data",
        "ip.id",
        "ip.checksum",
        "udp.payload",
        "source_file",
        "frame.number",
    ]
    write_header = True
    for chunk in pd.read_csv(
        input_file, sep=";", chunksize=chunksize, low_memory=False
    ):
        # Drop raw packet-level fields
        cols_to_drop = [col for col in columns_to_delete if col in chunk.columns]
        chunk.drop(columns=cols_to_drop, inplace=True)
        # Normalize PFCP hex-encoded fields
        if "pfcp.seid" in chunk.columns:
            chunk["pfcp.seid"] = chunk["pfcp.seid"].apply(
                lambda x: int(str(x), 16)
                if pd.notnull(x) and str(x).startswith("0x")
                else pd.to_numeric(x, errors="coerce")
            )
        if "pfcp.f_teid.teid" in chunk.columns:
            chunk["pfcp.f_teid.teid"] = chunk["pfcp.f_teid.teid"].apply(
                lambda x: int(str(x), 16)
                if pd.notnull(x) and str(x).startswith("0x")
                else pd.to_numeric(x, errors="coerce")
            )
        chunk.to_csv(
            output_file,
            sep=";",
            index=False,
            mode="w" if write_header else "a",
            header=write_header,
        )
        write_header = False


# ============================================================================
# STEP 4: IMPUTE NUMERICAL
# ============================================================================
def compute_numerical_medians(file_path, chunksize=100000):
    """
    Compute median values for all numeric columns (excluding timestamp + metadata).
    Used to fit a SimpleImputer.

    Parameters
    ----------
    file_path : str
        Input CSV path.
    chunksize : int
        Chunk size for streaming.

    Returns
    -------
    valid_cols : list[str]
        Columns included in the imputation.
    imputer : SimpleImputer
        Fitted median-imputer.
    """
    exclude_cols = ["ip.opt.time_stamp", "frame.number", "source_file"]
    numerics = None
    collected_chunks = []
    # Stream and accumulate numeric columns
    for chunk in pd.read_csv(file_path, sep=";", chunksize=chunksize, low_memory=False):
        numeric_cols = chunk.select_dtypes(include="number").columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        if numerics is None:
            numerics = numeric_cols
        collected_chunks.append(chunk[numeric_cols])
    # Combine data to compute robust medians
    full_df = pd.concat(collected_chunks, axis=0)
    # Keep only columns containing at least one non-null
    valid_cols = full_df.columns[full_df.notna().any()].tolist()
    full_df = full_df[valid_cols]
    # Fit median imputer
    imputer = SimpleImputer(strategy="median")
    imputer.fit(full_df)

    return valid_cols, imputer


def impute_file(input_file, output_file, valid_cols, imputer, chunksize=100000):
    """
    Apply median imputation to a CSV using chunk-based processing.

    Parameters
    ----------
    input_file : str
        Path to non-imputed CSV.
    output_file : str
        File that will contain imputed values.
    valid_cols : list[str]
        Columns the imputer was trained on.
    imputer : SimpleImputer
        Pretrained SimpleImputer.
    chunksize : int
        Chunk size.

    Returns
    -------
    None
    """
    exclude_cols = ["ip.opt.time_stamp", "frame.number", "source_file"]
    write_header = True

    for chunk in pd.read_csv(
        input_file, sep=";", chunksize=chunksize, low_memory=False
    ):
        # Preserve excluded columns
        excluded_data = {
            col: chunk[col] for col in exclude_cols if col in chunk.columns
        }
        chunk_numeric = chunk[valid_cols]
        chunk_imputed = pd.DataFrame(
            imputer.transform(chunk_numeric), columns=valid_cols, index=chunk.index
        )
        # Restore excluded fields (timestamps, file metadata)
        for col, data in excluded_data.items():
            chunk_imputed[col] = data
        # Reattach non-numeric/non-imputed fields
        other_cols = [
            c for c in chunk.columns if c not in valid_cols and c not in exclude_cols
        ]
        final_chunk = pd.concat(
            [
                chunk_imputed.reset_index(drop=True),
                chunk[other_cols].reset_index(drop=True),
            ],
            axis=1,
        )

        final_chunk.to_csv(
            output_file,
            sep=";",
            index=False,
            mode="w" if write_header else "a",
            header=write_header,
        )
        write_header = False


# ============================================================================
# STEP 5: ENCODING
# ============================================================================
def frequency_encode(df, col):
    """
    Apply frequency encoding: ranks categories by frequency (dense ranking).

    Parameters
    ----------
    df : DataFrame
        Input dataframe.
    col : str
        Column to encode.

    Returns
    -------
    Series
        Encoded values based on category ranks.
    """
    freq = df[col].value_counts()
    encoding = freq.rank(method="dense", ascending=False).astype(int)
    return df[col].map(encoding)


def time_conversion(df, col):
    """
    Convert timestamp-like strings to UNIX epoch seconds.

    Parameters
    ----------
    df : DataFrame
        Input dataframe.
    col : str
        Timestamp column as string.

    Returns
    -------
    Series
        Converted timestamps in seconds.
    """
    df[col] = pd.to_datetime(
        df[col], format="%b %d, %Y %H:%M:%S.%f %Z", errors="coerce"
    )
    df[col] = df[col].astype("int64") // 10**9  # convert ns→sec
    return df[col]


# ============================================================================
# STEP 6: CORRELATION FILTERING
# ============================================================================
def compute_pairwise_correlations(
    file_path, ref_col="ip.opt.time_stamp", special_cols=None
):
    """
    Compute Pearson correlations among numeric columns and with a reference label.

    Parameters
    ----------
    file_path : str
        Input CSV path for sampling.
    ref_col : str
        Label column used for correlation selection.
    special_cols : list[str], optional
        Columns to exclude from numeric selection.

    Returns
    -------
    numeric_cols : list[str]
        Numeric columns included in correlation matrix.
    corr_matrix : DataFrame
        Pairwise Pearson correlations.
    """
    if special_cols is None:
        special_cols = ["ip.opt.time_stamp", "frame.number"]
    logger.info(f"Loading sample {file_path}...")
    df = pd.read_csv(file_path, sep=";", nrows=5000, low_memory=False, encoding="latin-1")
    # Ensure reference column is numeric
    # Select numeric columns except known special ones and ports
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [
        c
        for c in numeric_cols
        if c
        not in special_cols
        + ["tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport"]
    ]
    # Correlation pairwise Matrix
    corr_matrix = df[numeric_cols].corr(method="pearson")

    return numeric_cols, corr_matrix


def find_correlated_pairs(
    corr, cols, input_path, threshold=0.90
):
    """
    Identify redundant feature pairs that are highly correlated in the datasets
    and select which feature to drop based on label correlation.

    Parameters
    ----------
    corr : DataFrame
        Correlation matrix from dataset.
    cols : list[str]
        Columns to evaluate.
    input_path : str
        Input dataset CSV path for sampling.
    threshold : float
        Absolute correlation threshold.

    Returns
    -------
    set[str]
        Columns to remove globally.
    """
    # Precompute variance for stability-based feature removal
    df_sample = pd.read_csv(input_path, sep=';', low_memory=False, encoding="latin-1")
    correlated_pairs = []
    cols_to_remove = set()
    # Check every pair of features
    for i, col1 in enumerate(cols):
        for j in range(i + 1, len(cols)):
            col2 = cols[j]
            # Fetch correlations
            c = corr.loc[col1, col2] if col1 in corr and col2 in corr else None
            # Keep only if both datasets show correlation ≥ threshold
            if (
                c is not None
                and abs(c) >= threshold
            ):
                correlated_pairs.append((col1, col2, c))
                # Variance-based elimination
                var1 = df_sample[col1].var()
                var2 = df_sample[col2].var()
                # Drop the least label-correlated feature
                if var1 < var2:
                    cols_to_remove.add(col1)
                    logger.info(f"Delete {col1} (corr label {var1:.3f}) vs {col2} (corr label {var2:.3f})")
                else:
                    cols_to_remove.add(col2)
                    logger.info(f"Delete {col2} (corr label {var2:.3f}) vs {col1} (corr label {var1:.3f})")

    logger.info(f"{len(correlated_pairs)} correlated pairs detected (|corr| >= {threshold})")
    logger.info(f"Columns to remove: {sorted(cols_to_remove)}")

    return cols_to_remove


def apply_filter_and_save(input_file, output_file, cols_to_drop, chunksize=50000):
    """
    Drop selected columns from dataset using chunked processing.

    Parameters
    ----------
    input_file : str
        Input CSV path.
    output_file : str
        Output CSV path.
    cols_to_drop : list[str] or set[str]
        Columns to remove.
    chunksize : int
        Chunk size for I/O.

    Returns
    -------
    None
    """
    write_header = True
    for chunk in pd.read_csv(
        input_file, sep=";", chunksize=chunksize, low_memory=False, encoding="latin-1"
    ):
        # Drop correlated features
        chunk_filtered = chunk.drop(columns=cols_to_drop, errors="ignore")
        chunk_filtered.to_csv(
            output_file,
            sep=";",
            index=False,
            header=write_header,
            mode="w" if write_header else "a",
        )
        write_header = False


def compute_pearson_filter_multi(input_path, output_path):
    """
    Compute redundant-correlated feature removal across datasets 2 and 3,
    then apply the resulting filter to all three datasets.

    Parameters
    ----------
    input_path : str
        Path to the encoded datasets.
    output_path : str
        Path to the filtered datasets.

    Returns
    -------
    None
    """
    # Compute correlations separately for dataset_2 and dataset_3
    cols, corr = compute_pairwise_correlations(input_path)   # Input -> DS Encoded
    # Only keep columns present in both datasets
    logger.info(f"Common numeric cols across datasets: {len(cols)}")
    # Identify globally removable columns
    cols_to_remove = find_correlated_pairs(corr, cols, input_path, threshold=0.90)
    logger.info("Columns removed:")
    for col in sorted(cols_to_remove):
        logger.info(f" - {col}")
    # Save removal list
    os.makedirs("models_preprocessing", exist_ok=True)
    with open("models_preprocessing/cols_to_remove.json", "w") as f:
        json.dump(list(cols_to_remove), f, indent=2)
    # Apply filtering to datasets 1–3
    apply_filter_and_save(
        input_path,
        output_path,
        cols_to_remove,
    )


# ============================================================================
# STEP 7: Z-SCORE NORMALIZATION
# ============================================================================
def fit_scaler_on_file(file_in, exclude_cols=None, chunksize=50000, sep=";"):
    """
    Fit a StandardScaler using streaming (partial_fit) over a full CSV.

    Parameters
    ----------
    file_in : str
        Input CSV.
    exclude_cols : list[str], optional
        Columns excluded from scaling.
    chunksize : int
        Chunk size.
    sep : str
        CSV separator.

    Returns
    -------
    scaler : StandardScaler
        Fitted scaler.
    columns_to_scale : list[str]
        Columns actually scaled.
    """
    scaler = StandardScaler()
    columns_to_scale = None
    for chunk in pd.read_csv(file_in, chunksize=chunksize, sep=sep):
        # Determine scalable columns on first chunk
        if columns_to_scale is None:
            exclude_cols = exclude_cols or []
            columns_to_scale = [col for col in chunk.columns if col not in exclude_cols]
        # Fill NaN to 0 before scaling
        chunk_to_scale = chunk[columns_to_scale].fillna(0).astype(float)
        scaler.partial_fit(chunk_to_scale)
    return scaler, columns_to_scale


def transform_file_with_scaler(
    file_in,
    file_out,
    scaler,
    columns_to_scale,
    exclude_cols=None,
    chunksize=50000,
    sep=";",
):
    """
    Apply Z-score normalization to a CSV using chunked transformation.

    Parameters
    ----------
    file_in : str
        Input data file.
    file_out : str
        Output normalized file.
    scaler : StandardScaler
        Fitted scaler.
    columns_to_scale : list[str]
        Columns to normalize.
    exclude_cols : list[str], optional
        Columns copied unchanged.
    chunksize : int
        Processing chunk size.
    sep : str
        CSV separator.

    Returns
    -------
    None
    """
    write_header = True
    for chunk in pd.read_csv(file_in, chunksize=chunksize, sep=sep):
        # Scale numerical subset
        chunk_to_scale = chunk[columns_to_scale].apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0).astype(float)
        scaled = pd.DataFrame(
            scaler.transform(chunk_to_scale),
            columns=columns_to_scale,
            index=chunk.index,
        )
        # Restore excluded fields unchanged
        for col in exclude_cols or []:
            if col in chunk.columns:
                scaled[col] = chunk[col].values
        scaled.to_csv(
            file_out,
            index=False,
            header=write_header,
            sep=sep,
            mode="w" if write_header else "a",
        )
        write_header = False


# endregion --- PREPROCESSING STEPS ---


def read_csv(path: str | Path) -> pd.DataFrame:
    """
    Safely load a CSV file by attempting UTF-8 first, then Latin-1.
    Supports both full DataFrame loading and chunked (TextFileReader) reading.

    Parameters
    ----------
    path : str | Path
        Path to the CSV file.

    Returns
    -------
    DataFrame or TextFileReader
        Loaded dataset or chunk iterator.
    """

    def load_and_preview(encoding):
        df = pd.read_csv(path, sep=";", low_memory=False, encoding=encoding)
        return df

    try:
        return load_and_preview("utf-8")
    except UnicodeDecodeError:
        return load_and_preview("latin-1")


def preprocessing_pipeline_up_to_step5(
    output_dir: str,
    input_file: str,
    dataset_type: str,
) -> pd.DataFrame:
    """
        Apply the same preprocessing pipeline as the full version but only to
        a single dataset. Can run from a DataFrame or from a CSV path.

        Parameters
        ----------
        input_file : str or None
            Input CSV path
        output_dir : str
            Output directory for results.
        dataset_type : str
            Type of dataset: "train" or "test"

        Returns
        -------
        DataFrame
            Fully preprocessed dataset.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 2: TCP OPTIONS PARSING
    tcp_path = f"{output_dir}/dataset_{dataset_type}_tcp.csv"
    enrich_tcp_columns(input_file, tcp_path)

    # === STEP 3: ADVANCED CLEANING ===
    drop_path = f"{output_dir}/dataset_{dataset_type}_drop.csv"
    drop_columns_chunked(tcp_path, drop_path, is_attack=True)

    # === STEP 4: IMPUTE NUMERICAL ===
    saine_cols, saine_imputer = compute_numerical_medians(drop_path)
    imputed_path = f"{output_dir}/dataset_{dataset_type}_imputed.csv"
    impute_file(drop_path, imputed_path, saine_cols, saine_imputer)
    os.makedirs("models_preprocessing", exist_ok=True)
    dump(saine_imputer, "models_preprocessing/imputer_saine.pkl")
    with open("models_preprocessing/saine_cols.json", "w") as f:
        json.dump(saine_cols, f, indent=2)

    # === STEP 5: ENCODING ===
    freq_cols = [
        "ip.src_host",
        "ip.dst_host",
        "ip.host",
        "ip.addr",
        "ip.src",
        "ip.dst",
        "tcp.srcport",
        "tcp.dstport",
        "udp.srcport",
        "udp.dstport",
        "pfcp.node_id_ipv4",
        "pfcp.outer_hdr_creation.ipv4",
        "pfcp.f_teid.ipv4_addr",
        "pfcp.f_seid.ipv4",
        "pfcp.outer_hdr_creation.teid",
        "pfcp.ue_ip_addr_ipv4",
        "tcp.checksum",
        "udp.checksum",
    ]
    time_columns = [
        "pfcp.time_of_first_packet",
        "pfcp.time_of_last_packet",
        "pfcp.end_time",
        "pfcp.recovery_time_stamp",
    ]
    special_columns = ["ip.opt.time_stamp"]

    df_saine = pd.read_csv(imputed_path, sep=";", low_memory=False)
    non_num_saine = df_saine.select_dtypes(include=["object"]).columns.tolist()
    non_num_cols = sorted((set(non_num_saine)))
    non_num_cols = [
        col
        for col in non_num_cols
        if col not in freq_cols
           and col not in special_columns
           and col not in time_columns
    ]
    # Convert "fake numeric" object columns into numeric types
    fake_num_cols = []
    for col in list(non_num_cols):
        try:
            converted = pd.to_numeric(df_saine[col], errors="coerce")
            # keep only if numeric in practice
            if converted.notna().sum() > 0 and converted.nunique() > 1:
                fake_num_cols.append(col)
                df_saine[col] = converted
                non_num_cols.remove(col)
        except Exception:
            pass
    if fake_num_cols:
        logger.info(f"Recast object -> numeric: {fake_num_cols}")

    # Fit one-hot encoder across all datasets
    df_saine_cat = df_saine[non_num_cols]
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(df_saine_cat)
    # Save categorical model
    dump(encoder, "models_preprocessing/encoder.pkl")
    with open("models_preprocessing/non_num_cols.json", "w") as f:
        json.dump(non_num_cols, f, indent=2)
    # Cleanup
    del (non_num_saine, df_saine_cat)

    # ENCODING
    df = df_saine.copy()
    # Frequency encoding
    for col in freq_cols:
        if col in df.columns:
            logger.info(f"   DF: {col}")
            df[col] = frequency_encode(df, col)
    df_freq_cols = df[[c for c in freq_cols if c in df.columns]]
    # Time conversion
    for col in time_columns:
        if col in df.columns:
            logger.info(f"    DF: {col}")
            df[col] = time_conversion(df, col)
    df_time_columns = df[[c for c in time_columns if c in df.columns]]
    df = df[
        [
            col
            for col in df.columns
            if col not in freq_cols and col not in time_columns
        ]
    ]

    df[non_num_cols] = df[non_num_cols].fillna("NaN").astype(str)
    df_encoded = pd.DataFrame(
        encoder.transform(df[non_num_cols]),
        columns=encoder.get_feature_names_out(non_num_cols),
    )
    # Drop categorical original columns
    df = df.drop(columns=non_num_cols).reset_index(drop=True)
    # TEMP SAVE PARTS (merged later in chunks)
    df.to_csv("df_main.csv", sep=";", index=False)
    df_freq_cols.to_csv("df_freq.csv", sep=";", index=False)
    df_encoded.to_csv("df_encoded.csv", sep=";", index=False)
    df_time_columns.to_csv("df_time_columns.csv", sep=";", index=False)
    del df, df_freq_cols, df_encoded, df_time_columns

    # MERGE IN CHUNKS
    chunk_size = 100000
    header_written = False
    encoded_path = f"{output_dir}/dataset_{dataset_type}_encoded.csv"
    with open(encoded_path, "w") as f_out:
        for parts in zip(
                pd.read_csv("df_main.csv", sep=";", chunksize=chunk_size),
                pd.read_csv("df_freq.csv", sep=";", chunksize=chunk_size),
                pd.read_csv("df_encoded.csv", sep=";", chunksize=chunk_size),
                pd.read_csv("df_time_columns.csv", sep=";", chunksize=chunk_size),
        ):
            merged = pd.concat([p for p in parts if not p.empty], axis=1)
            merged.to_csv(f_out, sep=";", index=False, header=not header_written)
            header_written = True
    # Clean temporary parts
    for temp_file in [
        "df_main.csv",
        "df_freq.csv",
        "df_encoded.csv",
        "df_time_columns.csv",
    ]:
        try:
            os.remove(temp_file)
        except FileNotFoundError:
            pass

    encoded_df = read_csv(encoded_path)
    logger.info(f"✅ Dataset preprocessed and saved in: {encoded_path}")

    return encoded_df


def preprocessing_pipeline_train(
    output_dir: str,
    input_file: pd.DataFrame,
) -> pd.DataFrame:
    """
        Apply the same preprocessing pipeline as the full version but only to
        a single dataset. Can run from a DataFrame or from a CSV path.

        Parameters
        ----------
        input_file : str or None
            Input CSV path
        output_dir : str
            Output directory for results.

        Returns
        -------
        DataFrame
            Fully preprocessed dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(f"{output_dir}/dataset_train_final.csv"):
        logger.info(f"✅ Dataset already preprocessed and saved in: {output_dir}/dataset_train_final.csv")
        return read_csv(f"{output_dir}/dataset_train_final.csv")

    # Step 2: TCP OPTIONS PARSING
    tcp_path = f"{output_dir}/dataset_train_tcp.csv"
    enrich_tcp_columns(input_file, tcp_path)

    # === STEP 3: ADVANCED CLEANING ===
    drop_path = f"{output_dir}/dataset_train_drop.csv"
    drop_columns_chunked(tcp_path, drop_path, is_attack=True)

    # === STEP 4: IMPUTE NUMERICAL ===
    saine_cols, saine_imputer = compute_numerical_medians(drop_path)
    imputed_path = f"{output_dir}/dataset_train_imputed.csv"
    impute_file(drop_path, imputed_path, saine_cols, saine_imputer)
    os.makedirs("models_preprocessing", exist_ok=True)
    dump(saine_imputer, "models_preprocessing/imputer_saine.pkl")
    with open("models_preprocessing/saine_cols.json", "w") as f:
        json.dump(saine_cols, f, indent=2)

    # === STEP 5: ENCODING ===
    freq_cols = [
        "ip.src_host",
        "ip.dst_host",
        "ip.host",
        "ip.addr",
        "ip.src",
        "ip.dst",
        "tcp.srcport",
        "tcp.dstport",
        "udp.srcport",
        "udp.dstport",
        "pfcp.node_id_ipv4",
        "pfcp.outer_hdr_creation.ipv4",
        "pfcp.f_teid.ipv4_addr",
        "pfcp.f_seid.ipv4",
        "pfcp.outer_hdr_creation.teid",
        "pfcp.ue_ip_addr_ipv4",
        "tcp.checksum",
        "udp.checksum",
    ]
    time_columns = [
        "pfcp.time_of_first_packet",
        "pfcp.time_of_last_packet",
        "pfcp.end_time",
        "pfcp.recovery_time_stamp",
    ]
    special_columns = ["ip.opt.time_stamp"]

    df_saine = pd.read_csv(imputed_path, sep=";", low_memory=False)
    non_num_saine = df_saine.select_dtypes(include=["object"]).columns.tolist()
    non_num_cols = sorted((set(non_num_saine)))
    non_num_cols = [
        col
        for col in non_num_cols
        if col not in freq_cols
           and col not in special_columns
           and col not in time_columns
    ]
    # Convert "fake numeric" object columns into numeric types
    fake_num_cols = []
    for col in list(non_num_cols):
        try:
            converted = pd.to_numeric(df_saine[col], errors="coerce")
            # keep only if numeric in practice
            if converted.notna().sum() > 0 and converted.nunique() > 1:
                fake_num_cols.append(col)
                df_saine[col] = converted
                non_num_cols.remove(col)
        except Exception:
            pass
    if fake_num_cols:
        logger.info(f"Recast object -> numeric: {fake_num_cols}")
        with open("models_preprocessing/fake_num_cols.json", "w") as f:
            json.dump(fake_num_cols, f, indent=2)

    # Fit one-hot encoder across all datasets
    df_saine_cat = df_saine[non_num_cols]
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(df_saine_cat)
    # Save categorical model
    dump(encoder, "models_preprocessing/encoder.pkl")
    with open("models_preprocessing/non_num_cols.json", "w") as f:
        json.dump(non_num_cols, f, indent=2)
    # Cleanup
    del (non_num_saine, df_saine_cat)

    # ENCODING
    df = df_saine.copy()
    # Frequency encoding
    for col in freq_cols:
        if col in df.columns:
            logger.info(f"   DF: {col}")
            df[col] = frequency_encode(df, col)
    df_freq_cols = df[[c for c in freq_cols if c in df.columns]]
    # Time conversion
    for col in time_columns:
        if col in df.columns:
            logger.info(f"    DF: {col}")
            df[col] = time_conversion(df, col)
    df_time_columns = df[[c for c in time_columns if c in df.columns]]
    df = df[
        [
            col
            for col in df.columns
            if col not in freq_cols and col not in time_columns
        ]
    ]

    df[non_num_cols] = df[non_num_cols].fillna("NaN").astype(str)
    df_encoded = pd.DataFrame(
        encoder.transform(df[non_num_cols]),
        columns=encoder.get_feature_names_out(non_num_cols),
    )
    # Drop categorical original columns
    df = df.drop(columns=non_num_cols).reset_index(drop=True)
    # TEMP SAVE PARTS (merged later in chunks)
    df.to_csv("df_main.csv", sep=";", index=False)
    df_freq_cols.to_csv("df_freq.csv", sep=";", index=False)
    df_encoded.to_csv("df_encoded.csv", sep=";", index=False)
    df_time_columns.to_csv("df_time_columns.csv", sep=";", index=False)
    del df, df_freq_cols, df_encoded, df_time_columns

    # MERGE IN CHUNKS
    chunk_size = 100000
    header_written = False
    encoded_path = f"{output_dir}/dataset_train_encoded.csv"
    with open(encoded_path, "w") as f_out:
        for parts in zip(
                pd.read_csv("df_main.csv", sep=";", chunksize=chunk_size),
                pd.read_csv("df_freq.csv", sep=";", chunksize=chunk_size),
                pd.read_csv("df_encoded.csv", sep=";", chunksize=chunk_size),
                pd.read_csv("df_time_columns.csv", sep=";", chunksize=chunk_size),
        ):
            merged = pd.concat([p for p in parts if not p.empty], axis=1)
            merged.to_csv(f_out, sep=";", index=False, header=not header_written)
            header_written = True
    # Clean temporary parts
    for temp_file in [
        "df_main.csv",
        "df_freq.csv",
        "df_encoded.csv",
        "df_time_columns.csv",
    ]:
        try:
            os.remove(temp_file)
        except FileNotFoundError:
            pass

    # === STEP 6: CORRELATION FILTERING ===
    # Save input to temp CSV (ensures compatibility with chunked functions)
    filtered_path = os.path.join(output_dir, f"dataset_train_filtered.csv")
    compute_pearson_filter_multi(encoded_path, filtered_path)
    # Save common columns for model consistency
    df = read_csv(filtered_path)
    common_filtered_cols = list(set(df.columns))
    with open("models_preprocessing/common_filtered_cols.json", "w") as f:
        json.dump(common_filtered_cols, f, indent=2)

    # === STEP 7: Z-SCORE NORMALIZATION ===
    sep = ";"
    chunksize = 50000
    exclude_att = ["ip.opt.time_stamp", "frame.number", "source_file"]
    scaler, columns_to_scale = fit_scaler_on_file(
        filtered_path,
        exclude_cols=exclude_att,
        chunksize=chunksize,
        sep=sep,
    )
    dump(scaler, "models_preprocessing/scaler.pkl")
    with open("models_preprocessing/columns_to_scale.json", "w") as f:
        json.dump(columns_to_scale, f, indent=2)

    final_path = os.path.join(output_dir, f"dataset_train_final.csv")
    transform_file_with_scaler(
        filtered_path,
        final_path,
        scaler,
        columns_to_scale,
        exclude_cols=exclude_att,
        chunksize=chunksize,
        sep=sep,
    )
    final_df = read_csv(final_path)
    logger.info(f"✅ Dataset preprocessed and saved in: {final_path}")

    return final_df


def preprocessing_pipeline_test(
    output_dir: str,
    input_file: pd.DataFrame,
) -> pd.DataFrame:
    """
        Apply the same preprocessing pipeline as the full version but only to
        a single dataset. Can run from a DataFrame or from a CSV path.

        Parameters
        ----------
        input_file : pd.DataFrame
            Input CSV
        output_dir : str
            Output directory for results.

        Returns
        -------
        DataFrame
            Fully preprocessed dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    # === STEP 2: TCP OPTIONS PARSING ===
    tcp_path = os.path.join(output_dir, f"dataset_test_tcp.csv")
    enrich_tcp_columns(input_file, tcp_path)

    # === STEP 3: ADVANCED CLEANING ===
    drop_path = os.path.join(output_dir, f"dataset_test_drop.csv")
    drop_columns_chunked(tcp_path, drop_path, is_attack=True)

    # === STEP 4: IMPUTE NUMERICAL ===
    imputer = joblib.load(Path(__file__).parent / "models_preprocessing/imputer_saine.pkl")
    with open(Path(__file__).parent / "models_preprocessing/saine_cols.json") as f:
        valid_cols = json.load(f)
    imputed_path = os.path.join(output_dir, f"dataset_test_imputed.csv")
    impute_file(drop_path, imputed_path, valid_cols, imputer)

    # === STEP 5: ENCODING ===
    encoder = joblib.load(Path(__file__).parent / "models_preprocessing/encoder.pkl")
    with open(Path(__file__).parent / "models_preprocessing/non_num_cols.json") as f:
        non_num_cols = json.load(f)

    df = read_csv(imputed_path)
    freq_cols = [
        "ip.src_host",
        "ip.dst_host",
        "ip.host",
        "ip.addr",
        "ip.src",
        "ip.dst",
        "tcp.srcport",
        "tcp.dstport",
        "udp.srcport",
        "udp.dstport",
        "pfcp.node_id_ipv4",
        "pfcp.outer_hdr_creation.ipv4",
        "pfcp.f_teid.ipv4_addr",
        "pfcp.f_seid.ipv4",
        "pfcp.outer_hdr_creation.teid",
        "pfcp.ue_ip_addr_ipv4",
        "tcp.checksum",
        "udp.checksum",
    ]
    time_columns = [
        "pfcp.time_of_first_packet",
        "pfcp.time_of_last_packet",
        "pfcp.end_time",
        "pfcp.recovery_time_stamp",
    ]
    special_columns = ["ip.opt.time_stamp"]

    timestamp_col = df[special_columns[0]] if special_columns[0] in df.columns else None
    if special_columns[0] in df.columns:
        df = df.drop(columns=[special_columns[0]])

    with open("models_preprocessing/fake_num_cols.json") as f:
        fake_num_cols = json.load(f)

    for col in fake_num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Frequency encoding
    for col in freq_cols:
        if col in df.columns:
            df[col] = frequency_encode(df, col)
    # Time conversion
    for col in time_columns:
        if col in df.columns:
            df[col] = time_conversion(df, col)

    # One-hot encoding (only known categorical columns)
    cat_cols = [col for col in non_num_cols if col in df.columns]
    if cat_cols:
        df[cat_cols] = df[cat_cols].fillna("NaN").astype(str)
        df_encoded = pd.DataFrame(
            encoder.transform(df[cat_cols]),
            columns=encoder.get_feature_names_out(cat_cols),
            index=df.index,
        )
        df = pd.concat([df.drop(columns=cat_cols), df_encoded], axis=1)

    encoded_path = os.path.join(output_dir, f"dataset_test_encoded.csv")
    # Insert special timestamp column in correct position
    if timestamp_col is not None:
        df[special_columns[0]] = timestamp_col

    df.to_csv(encoded_path, sep=";", index=False, encoding="latin-1")

    # === STEP 6: CORRELATION FILTERING ===
    with open(Path(__file__).parent / "models_preprocessing/cols_to_remove.json") as f:
        cols_to_remove = json.load(f)
    filtered_path = os.path.join(output_dir, f"dataset_test_filtered.csv")
    apply_filter_and_save(encoded_path, filtered_path, cols_to_remove)

    # === STEP 7: Z-SCORE NORMALIZATION ===
    scaler = joblib.load(Path(__file__).parent / "models_preprocessing/scaler.pkl")
    with open(
        Path(__file__).parent / "models_preprocessing/columns_to_scale.json"
    ) as f:
        columns_to_scale = json.load(f)

    final_path = os.path.join(output_dir, f"dataset_test_final.csv")
    transform_file_with_scaler(
        filtered_path,
        final_path,
        scaler,
        columns_to_scale,
        exclude_cols=["ip.opt.time_stamp", "frame.number", "source_file"],
        chunksize=50000,
        sep=";",
    )

    final_df = read_csv(final_path)
    logger.info(f"✅ Dataset preprocessed and saved in: {final_path}")

    return final_df

