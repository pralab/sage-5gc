import shutil
import tempfile
import joblib
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os
from joblib import dump
import json


def safe_read_csv(path, **kwargs):
    """
    Safely load a CSV file by attempting UTF-8 first, then Latin-1.
    Supports both full DataFrame loading and chunked (TextFileReader) reading.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    **kwargs :
        Additional arguments passed to pandas.read_csv.

    Returns
    -------
    DataFrame or TextFileReader
        Loaded dataset or chunk iterator.
    """
    def load_and_preview(encoding):
        df = pd.read_csv(path, sep=';', low_memory=False, encoding=encoding, **kwargs)
        # If read_csv returns an iterator (when chunksize is used), consume first chunk to validate
        if isinstance(df, pd.DataFrame):
            return df
        else:
            first_chunk = next(df)  # preview first chunk
            return df  
    try:
        return load_and_preview("utf-8")
    except UnicodeDecodeError:
        return load_and_preview("latin-1")

# ============================================================================
# STEP 2: TCP OPTIONS PARSING
# ============================================================================
def enrich_tcp_columns(input_file, output_file, chunksize=100_000):
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
    for chunk in pd.read_csv(input_file, sep=';', chunksize=chunksize, low_memory=False):
        if 'tcp.options' in chunk.columns:
            chunk['tcp.options'] = chunk['tcp.options'].fillna('')
            # Binary features for TCP options
            chunk['tcp_opt_mss'] = chunk['tcp.options'].str.contains('mss', case=False).astype(int)
            chunk['tcp_opt_ts'] = chunk['tcp.options'].str.contains('timestamp', case=False).astype(int)
            chunk['tcp_opt_sack'] = chunk['tcp.options'].str.contains('sack', case=False).astype(int)
            chunk['tcp_opt_wscale'] = chunk['tcp.options'].str.contains('wscale', case=False).astype(int)
            chunk = chunk.drop(columns=['tcp.options'], errors='ignore')
        chunk.to_csv(output_file, sep=';', index=False, mode='w' if write_header else 'a', header=write_header)
        write_header = False


# ============================================================================
# STEP 3: ADVANCED CLEANING
# ============================================================================
def drop_columns_chunked(input_file, output_file, is_attack=False, chunksize = 100000):
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
        'ip.hdr_len', 'ip.len', 'tcp.payload', 'tcp.segment_data',
        'tcp.reassembled.data', 'ip.id', 'ip.checksum', 'udp.payload', 'source_file', 'frame.number'
    ]
    write_header = True
    for chunk in pd.read_csv(input_file, sep=';', chunksize=chunksize, low_memory=False):
        # Drop raw packet-level fields
        cols_to_drop = [col for col in columns_to_delete if col in chunk.columns]
        chunk.drop(columns=cols_to_drop, inplace=True)
        # Normalize PFCP hex-encoded fields
        if 'pfcp.seid' in chunk.columns:
            chunk['pfcp.seid'] = chunk['pfcp.seid'].apply(
                lambda x: int(str(x), 16) if pd.notnull(x) and str(x).startswith("0x") else pd.to_numeric(x, errors='coerce')
            )
        if 'pfcp.f_teid.teid' in chunk.columns:
            chunk['pfcp.f_teid.teid'] = chunk['pfcp.f_teid.teid'].apply(
                lambda x: int(str(x), 16) if pd.notnull(x) and str(x).startswith("0x") else pd.to_numeric(x, errors='coerce')
            )
        chunk.to_csv(output_file, sep=';', index=False, mode='w' if write_header else 'a', header=write_header)
        write_header = False


# ============================================================================
# STEP 4: IMPUTE NUMERICAL
# ============================================================================
def compute_numerical_medians(file_path, chunksize = 100000):
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
    for chunk in pd.read_csv(file_path, sep=';', chunksize=chunksize, low_memory=False):
        numeric_cols = chunk.select_dtypes(include='number').columns.tolist()
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
    imputer = SimpleImputer(strategy='median')
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
    
    for chunk in pd.read_csv(input_file, sep=';', chunksize=chunksize, low_memory=False):
        # Preserve excluded columns
        excluded_data = {col: chunk[col] for col in exclude_cols if col in chunk.columns}
        chunk_numeric = chunk[valid_cols]
        chunk_imputed = pd.DataFrame(imputer.transform(chunk_numeric), columns=valid_cols, index=chunk.index)
        # Restore excluded fields (timestamps, file metadata)
        for col, data in excluded_data.items():
            chunk_imputed[col] = data
        # Reattach non-numeric/non-imputed fields
        other_cols = [c for c in chunk.columns if c not in valid_cols and c not in exclude_cols]
        final_chunk = pd.concat([
            chunk_imputed.reset_index(drop=True),
            chunk[other_cols].reset_index(drop=True)
        ], axis=1)

        final_chunk.to_csv(output_file, sep=';', index=False, mode='w' if write_header else 'a', header=write_header)
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
    encoding = freq.rank(method='dense', ascending=False).astype(int)
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
        df[col],
        format='%b %d, %Y %H:%M:%S.%f %Z',
        errors='coerce'
    )
    df[col] = df[col].astype('int64') // 10**9    # convert ns→sec
    return df[col]


# ============================================================================
# STEP 6: CORRELATION FILTERING
# ============================================================================
def compute_pairwise_correlations(file_path, ref_col="ip.opt.time_stamp",
                                  special_cols=None):
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
    special_corr : Series
        Correlation between each feature and the reference column.
    """
    if special_cols is None:
        special_cols = ["ip.opt.time_stamp", "frame.number"]
    print(f"Loading sample {file_path}...")
    df = pd.read_csv(file_path, sep=';', nrows=5000, low_memory=False, encoding='latin-1')
    # Ensure reference column is numeric
    if ref_col in df.columns:
        df[ref_col] = pd.to_numeric(df[ref_col], errors="coerce").fillna(-1).astype(int)
    # Select numeric columns except known special ones and ports
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in special_cols +
                    ['tcp.srcport', 'tcp.dstport', 'udp.srcport', 'udp.dstport']]
    # Correlation pairwise Matrix
    corr_matrix = df[numeric_cols].corr(method="pearson")
    # Compute correlation with ref
    special_corr = df[numeric_cols + [ref_col]].corr(method="pearson")[ref_col].drop(ref_col)

    return numeric_cols, corr_matrix, special_corr


def find_common_correlated_pairs(corr2, corr3, cols, special_corr2, special_corr3, threshold=0.90):
    """
    Identify redundant feature pairs that are highly correlated in 2 datasets
    and select which feature to drop based on label correlation.

    Parameters
    ----------
    corr2 : DataFrame
        Correlation matrix from dataset 2.
    corr3 : DataFrame
        Correlation matrix from dataset 3.
    cols : list[str]
        Columns to evaluate.
    special_corr2 : Series
        Correlation with label for dataset 2.
    special_corr3 : Series
        Correlation with label for dataset 3.
    threshold : float
        Absolute correlation threshold.

    Returns
    -------
    set[str]
        Columns to remove globally.
    """
    correlated_pairs = []
    cols_to_remove = set()
    # Check every pair of features
    for i, col1 in enumerate(cols):
        for j in range(i + 1, len(cols)):
            col2 = cols[j]
            # Fetch correlations
            c2 = corr2.loc[col1, col2] if col1 in corr2 and col2 in corr2 else None
            c3 = corr3.loc[col1, col2] if col1 in corr3 and col2 in corr3 else None
            # Keep only if both datasets show correlation ≥ threshold
            if c2 is not None and c3 is not None and abs(c2) >= threshold and abs(c3) >= threshold:
                correlated_pairs.append((col1, col2, c2, c3))
                # Compute importance via sum of absolute label correlations
                corr1 = abs(special_corr2.get(col1, 0)) + abs(special_corr3.get(col1, 0))
                corr2_val = abs(special_corr2.get(col2, 0)) + abs(special_corr3.get(col2, 0))
                # Drop the least label-correlated featur
                if corr1 < corr2_val:
                    cols_to_remove.add(col1)
                    print(f"Delete {col1} (corr label {corr1:.3f}) vs {col2} (corr label {corr2_val:.3f})")
                else:
                    cols_to_remove.add(col2)
                    print(f"Delete {col2} (corr label {corr2_val:.3f}) vs {col1} (corr label {corr1:.3f})")

    print(f"{len(correlated_pairs)} correlated pairs detected in BOTH games(|corr| >= {threshold})")
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
    for chunk in pd.read_csv(input_file, sep=';', chunksize=chunksize, low_memory=False, encoding='latin-1'):
        # Drop correlated features
        chunk_filtered = chunk.drop(columns=cols_to_drop, errors='ignore')
        chunk_filtered.to_csv(output_file, sep=';', index=False,
                              header=write_header, mode='w' if write_header else 'a', encoding='latin-1')
        write_header = False


def compute_pearson_filter_multi(input_dir, output_dir):
    """
    Compute redundant-correlated feature removal across datasets 2 and 3,
    then apply the resulting filter to all three datasets.

    Parameters
    ----------
    input_dir : str
        Directory containing input cleaned datasets.
    output_dir : str
        Directory containing encoded datasets.

    Returns
    -------
    None
    """
    # Compute correlations separately for dataset_2 and dataset_3
    cols2, corr2, special_corr2 = compute_pairwise_correlations(f"{output_dir}/dataset_2_encoded.csv")
    cols3, corr3, special_corr3 = compute_pairwise_correlations(f"{output_dir}/dataset_3_encoded.csv")
    # Only keep columns present in both datasets
    common_cols = sorted(set(cols2).intersection(set(cols3)))
    print(f"Common numeric cols across datasets: {len(common_cols)}")
    # Identify globally removable columns
    cols_to_remove = find_common_correlated_pairs(corr2, corr3, common_cols,
                                                  special_corr2, special_corr3,
                                                  threshold=0.90)
    print("Columns removed globally:")
    for col in sorted(cols_to_remove):
        print(f" - {col}")
    # Save removal list
    os.makedirs("models_preprocessing", exist_ok=True)
    with open("models_preprocessing/cols_to_remove.json", "w") as f:
        json.dump(list(cols_to_remove), f, indent=2)
    # Apply filtering to datasets 1–3
    apply_filter_and_save(f"{output_dir}/dataset_1_encoded.csv", f"{output_dir}/dataset_1_filtered.csv", cols_to_remove)
    apply_filter_and_save(f"{output_dir}/dataset_2_encoded.csv", f"{output_dir}/dataset_2_filtered.csv", cols_to_remove)
    apply_filter_and_save(f"{output_dir}/dataset_3_encoded.csv", f"{output_dir}/dataset_3_filtered.csv", cols_to_remove)


# ============================================================================
# STEP 7: Z-SCORE NORMALIZATION
# ============================================================================
def fit_scaler_on_file(file_in, exclude_cols=None, chunksize=50000, sep=';'):
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


def transform_file_with_scaler(file_in, file_out, scaler, columns_to_scale, exclude_cols=None, chunksize=50000, sep=';'):
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
        chunk_to_scale = chunk[columns_to_scale].fillna(0).astype(float)
        scaled = pd.DataFrame(scaler.transform(chunk_to_scale), columns=columns_to_scale, index=chunk.index)
        # Restore excluded fields unchanged
        for col in (exclude_cols or []):
            if col in chunk.columns:
                scaled[col] = chunk[col].values
        scaled.to_csv(file_out, index=False, header=write_header, sep=sep, mode='w' if write_header else 'a')
        write_header = False


# ============================================================================
# FULL PREPROCESSING PIPELINE
# ============================================================================
def preprocessing_pipeline(input_dir="cleaned_dataset", output_dir="final_datasets_from_preprocessing"):
    """
    Run the complete preprocessing pipeline on datasets 1, 2, and 3.
    Includes TCP parsing, column dropping, imputation, encoding,
    correlation filtering, and Z-score normalization.

    Parameters
    ----------
    input_dir : str
        Directory containing cleaned dataset CSVs.
    output_dir : str
        Directory where preprocessed output will be written.

    Returns
    -------
    None
    """
    # Step 2: TCP OPTIONS PARSING
    enrich_tcp_columns(f"{input_dir}/dataset_1_cleaned.csv", f"{output_dir}/dataset_1_tcp.csv")
    enrich_tcp_columns(f"{input_dir}/dataset_2_cleaned.csv", f"{output_dir}/dataset_2_tcp.csv")
    enrich_tcp_columns(f"{input_dir}/dataset_3_cleaned.csv", f"{output_dir}/dataset_3_tcp.csv")

    # Step 3: ADVANCED CLEANING
    drop_columns_chunked(f"{output_dir}/dataset_1_tcp.csv", f"{output_dir}/dataset_1_drop.csv", is_attack=False)
    drop_columns_chunked(f"{output_dir}/dataset_2_tcp.csv", f"{output_dir}/dataset_2_drop.csv", is_attack=True)
    drop_columns_chunked(f"{output_dir}/dataset_3_tcp.csv", f"{output_dir}/dataset_3_drop.csv", is_attack=True)

    # Step 4: IMPUTE NUMERICAL
    saine_cols, saine_imputer = compute_numerical_medians(f"{output_dir}/dataset_1_drop.csv")
    impute_file(f"{output_dir}/dataset_1_drop.csv", f"{output_dir}/dataset_1_imputed.csv", saine_cols, saine_imputer)
    attack_cols, attack_imputer = compute_numerical_medians(f"{output_dir}/dataset_2_drop.csv")
    impute_file(f"{output_dir}/dataset_2_drop.csv", f"{output_dir}/dataset_2_imputed.csv", attack_cols, attack_imputer)
    impute_file(f"{output_dir}/dataset_3_drop.csv", f"{output_dir}/dataset_3_imputed.csv", attack_cols, attack_imputer)
    # Save imputation models
    os.makedirs("models_preprocessing", exist_ok=True)
    dump(saine_imputer, "models_preprocessing/imputer_saine.pkl")
    dump(attack_imputer, "models_preprocessing/imputer_attack.pkl")
    with open("models_preprocessing/saine_cols.json", "w") as f:
        json.dump(saine_cols, f, indent=2)
    with open("models_preprocessing/attack_cols.json", "w") as f:
        json.dump(attack_cols, f, indent=2)

    # Step 5: ENCODING
    freq_cols = [
        'ip.src_host', 'ip.dst_host', 'ip.host', 'ip.addr', 'ip.src', 'ip.dst',
        'tcp.srcport', 'tcp.dstport', 'udp.srcport', 'udp.dstport',
        'pfcp.node_id_ipv4', 'pfcp.outer_hdr_creation.ipv4',
        'pfcp.f_teid.ipv4_addr', 'pfcp.f_seid.ipv4',
        'pfcp.outer_hdr_creation.teid', 'pfcp.ue_ip_addr_ipv4', 'tcp.checksum', 'udp.checksum'
    ]

    time_columns = [
        'pfcp.time_of_first_packet', 'pfcp.time_of_last_packet',
        'pfcp.end_time', 'pfcp.recovery_time_stamp'
    ]

    special_columns = ['ip.opt.time_stamp']
    # Load imputed datasets into memory (only once)
    df_attack = pd.read_csv(f"{output_dir}/dataset_2_imputed.csv", sep=';', low_memory=False)
    df_saine = pd.read_csv(f"{output_dir}/dataset_1_imputed.csv", sep=';', low_memory=False)
    df_attack2 = pd.read_csv(f"{output_dir}/dataset_3_imputed.csv", sep=';', low_memory=False)
    # Collect object columns
    non_num_attack = df_attack.select_dtypes(include=['object']).columns.tolist()
    non_num_saine = df_saine.select_dtypes(include=['object']).columns.tolist()
    non_num_attack2 = df_attack2.select_dtypes(include=['object']).columns.tolist()

    non_num_cols = sorted(set(non_num_attack).union(set(non_num_saine)).union(set(non_num_attack2)))
   # Remove frequency-encoded, special, and time columns
    non_num_cols = [col for col in non_num_cols if
                    col not in freq_cols and col not in special_columns and col not in time_columns]
    # Convert "fake numeric" object columns into numeric types
    fake_num_cols = []
    for col in list(non_num_cols):
        try:
            converted = pd.to_numeric(df_attack[col], errors='coerce')
            # keep only if numeric in practice
            if converted.notna().sum() > 0 and converted.nunique() > 1:
                fake_num_cols.append(col)
                for df_tmp in [df_attack, df_saine, df_attack2]:
                    if col in df_tmp.columns:
                        df_tmp[col] = pd.to_numeric(df_tmp[col], errors='coerce')
                non_num_cols.remove(col)
        except Exception:
            pass

    if fake_num_cols:
        print("Recast object→numeric:", fake_num_cols)
    # Fit one-hot encoder across all datasets
    df_attack_cat = df_attack[non_num_cols]
    df_saine_cat = df_saine[non_num_cols]
    df_attack2_cat = df_attack2[non_num_cols]

    df_cat_all = pd.concat([df_attack_cat, df_saine_cat, df_attack2_cat], axis=0)
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(df_cat_all[non_num_cols])
    # Save categorical model
    dump(encoder, "models_preprocessing/encoder.pkl")
    with open("models_preprocessing/non_num_cols.json", "w") as f:
        json.dump(non_num_cols, f, indent=2)
    # Cleanup
    del non_num_attack, non_num_saine, non_num_attack2, df_attack_cat, df_saine_cat, df_attack2_cat, df_cat_all

    # ENCODING LOOP FOR (attack, saine, attack2)
    data_frames = ['df_attack', 'df_saine', 'df_attack2']
    for data_frame in data_frames:
        if data_frame == 'df_attack':
            df = df_attack
            timestamp_col = df[special_columns[0]] if special_columns[0] in df else None
            if special_columns[0] in df:
                df = df.drop(columns=[special_columns[0]])
        elif data_frame == 'df_saine':
            df = df_saine.copy()
            timestamp_col = None
        else:  # df_attack2
            df = df_attack2
            timestamp_col = df[special_columns[0]] if special_columns[0] in df else None
            if special_columns[0] in df:
                df = df.drop(columns=[special_columns[0]])

        # Frequency encoding 
        for col in freq_cols:
            if col in df.columns:
                print(f"   {data_frame}: {col}")
                df[col] = frequency_encode(df, col)
        df_freq_cols = df[[c for c in freq_cols if c in df.columns]]

        # Time conversion 
        for col in time_columns:
            if col in df.columns:
                print(f"    {data_frame}: {col}")
                df[col] = time_conversion(df, col)
        df_time_columns = df[[c for c in time_columns if c in df.columns]]
        df = df[[col for col in df.columns if col not in freq_cols and col not in time_columns]]

        df[non_num_cols] = df[non_num_cols].fillna("NaN").astype(str)
        df_encoded = pd.DataFrame(
            encoder.transform(df[non_num_cols]),
            columns=encoder.get_feature_names_out(non_num_cols)
        )
        # Drop categorical original columns
        df = df.drop(columns=non_num_cols).reset_index(drop=True)
        # TEMP SAVE PARTS (merged later in chunks)
        df.to_csv('df_main.csv', sep=';', index=False)
        df_freq_cols.to_csv('df_freq.csv', sep=';', index=False)
        df_encoded.to_csv('df_encoded.csv', sep=';', index=False)
        df_time_columns.to_csv('df_time_columns.csv', sep=';', index=False)

        del df, df_freq_cols, df_encoded, df_time_columns
        # MERGE IN CHUNKS
        chunk_size = 100000
        if data_frame == 'df_attack':
            if timestamp_col is not None:
                timestamp_col.to_csv('df_timestamp.csv', sep=';', index=False)
            header_written = False
            # Merge df_main + freq + onehot + timestamp + time
            with open(f"{output_dir}/dataset_2_encoded.csv", 'w') as f_out:
                for parts in zip(
                        pd.read_csv('df_main.csv', sep=';', chunksize=chunk_size),
                        pd.read_csv('df_freq.csv', sep=';', chunksize=chunk_size),
                        pd.read_csv('df_encoded.csv', sep=';', chunksize=chunk_size),
                        (pd.read_csv('df_timestamp.csv', sep=';',
                                     chunksize=chunk_size) if timestamp_col is not None else [pd.DataFrame()]),
                        pd.read_csv('df_time_columns.csv', sep=';', chunksize=chunk_size),
                ):
                    merged = pd.concat([p for p in parts if not p.empty], axis=1)
                    merged.to_csv(f_out, sep=';', index=False, header=not header_written)
                    header_written = True
        elif data_frame == 'df_saine':
            header_written = False
            with open(f"{output_dir}/dataset_1_encoded.csv", 'w') as f_out:
                for parts in zip(
                        pd.read_csv('df_main.csv', sep=';', chunksize=chunk_size),
                        pd.read_csv('df_freq.csv', sep=';', chunksize=chunk_size),
                        pd.read_csv('df_encoded.csv', sep=';', chunksize=chunk_size),
                        pd.read_csv('df_time_columns.csv', sep=';', chunksize=chunk_size),
                ):
                    merged = pd.concat([p for p in parts if not p.empty], axis=1)
                    merged.to_csv(f_out, sep=';', index=False, header=not header_written)
                    header_written = True
        else:  # df_attack2
            if timestamp_col is not None:
                timestamp_col.to_csv('df_timestamp.csv', sep=';', index=False)
            header_written = False
            with open(f"{output_dir}/dataset_3_encoded.csv", 'w') as f_out:
                for parts in zip(
                        pd.read_csv('df_main.csv', sep=';', chunksize=chunk_size),
                        pd.read_csv('df_freq.csv', sep=';', chunksize=chunk_size),
                        pd.read_csv('df_encoded.csv', sep=';', chunksize=chunk_size),
                        (pd.read_csv('df_timestamp.csv', sep=';',
                                     chunksize=chunk_size) if timestamp_col is not None else [pd.DataFrame()]),
                        pd.read_csv('df_time_columns.csv', sep=';', chunksize=chunk_size),
                ):
                    merged = pd.concat([p for p in parts if not p.empty], axis=1)
                    merged.to_csv(f_out, sep=';', index=False, header=not header_written)
                    header_written = True
        # Clean temporary parts
        for temp_file in ['df_main.csv', 'df_freq.csv', 'df_encoded.csv',
                          'df_timestamp.csv', 'df_time_columns.csv']:
            try:
                os.remove(temp_file)
            except FileNotFoundError:
                pass

    # Step 6: CORRELATION FILTERING
    compute_pearson_filter_multi(input_dir, output_dir)
    # Save common columns for model consistency
    df1 = pd.read_csv(f"{output_dir}/dataset_1_filtered.csv", sep=';', nrows=1)
    df2 = pd.read_csv(f"{output_dir}/dataset_2_filtered.csv", sep=';', nrows=1)
    df3 = pd.read_csv(f"{output_dir}/dataset_3_filtered.csv", sep=';', nrows=1)
    common_filtered_cols = list(set(df1.columns) & set(df2.columns) & set(df3.columns))
    with open("models_preprocessing/common_filtered_cols.json", "w") as f:
        json.dump(common_filtered_cols, f, indent=2)

    # Step 7: Z-SCORE NORMALIZATION
    sep = ';'
    chunksize = 50000
    exclude_att = ["ip.opt.time_stamp", "frame.number", "source_file"]
    scaler, columns_to_scale = fit_scaler_on_file(f"{output_dir}/dataset_1_filtered.csv", exclude_cols=exclude_att,
                                                  chunksize=chunksize, sep=sep)
    dump(scaler, "models_preprocessing/scaler.pkl")
    with open("models_preprocessing/columns_to_scale.json", "w") as f:
        json.dump(columns_to_scale, f, indent=2)

    transform_file_with_scaler(f"{output_dir}/dataset_1_filtered.csv", f"{output_dir}/dataset_1_final.csv", scaler, columns_to_scale,
                               exclude_cols=exclude_att, chunksize=chunksize, sep=sep)
    transform_file_with_scaler(f"{output_dir}/dataset_2_filtered.csv", f"{output_dir}/dataset_2_final.csv", scaler, columns_to_scale,
                               exclude_cols=exclude_att, chunksize=chunksize, sep=sep)
    transform_file_with_scaler(f"{output_dir}/dataset_3_filtered.csv", f"{output_dir}/dataset_3_final.csv", scaler, columns_to_scale,
                               exclude_cols=exclude_att, chunksize=chunksize, sep=sep)


# ============================================================================
# PARTIAL PIPELINE FOR SINGLE-DATASET PROCESSING
# ============================================================================
def preprocessing_pipeline_partial(
    input_file: str | None = None,
    output_dir: str = "final_datasets_from_preprocessing_partial",
    dataset_name: str = "dataset_3",
    df: pd.DataFrame | None = None,
    in_memory: bool = False,
) -> pd.DataFrame:
     """
    Apply the same preprocessing pipeline as the full version but only to
    a single dataset. Can run from a DataFrame or from a CSV path.

    Parameters
    ----------
    input_file : str or None
        Input CSV path (ignored when df is provided).
    output_dir : str
        Output directory for results.
    dataset_name : str
        Prefix used for naming intermediate files.
    df : DataFrame or None
        Input dataframe (optional).
    in_memory : bool
        If True, pipeline writes into a temporary directory.

    Returns
    -------
    DataFrame
        Fully preprocessed dataset.
    """
    temp_dir = None
    if in_memory:
        # Use a temporary folder during preprocessing
        temp_dir = tempfile.mkdtemp(prefix="preproc_partial_")
        work_dir = temp_dir
    else:
        os.makedirs(output_dir, exist_ok=True)
        work_dir = output_dir

    # === STEP 1: LOAD DATA ===
    if df is None:
        if input_file is None:
            raise ValueError("Devi specificare input_file o df.")
        df = safe_read_csv(input_file)

    # Save input to temp CSV (ensures compatibility with chunked functions)
    temp_input = os.path.join(work_dir, f"{dataset_name}_input_temp.csv")
    df.to_csv(temp_input, sep=';', index=False)
    source_path = temp_input if input_file is None else input_file

    # === STEP 2: TCP OPTIONS PARSING ===
    tcp_path = os.path.join(work_dir, f"{dataset_name}_tcp.csv")
    enrich_tcp_columns(source_path, tcp_path)

    # === STEP 3: ADVANCED CLEANING ===
    drop_path = os.path.join(work_dir, f"{dataset_name}_drop.csv")
    drop_columns_chunked(tcp_path, drop_path, is_attack=True)

    # === STEP 4: IMPUTE NUMERICAL ===
    imputer = joblib.load("preprocessing/models_preprocessing/imputer_attack.pkl")
    with open("preprocessing/models_preprocessing/attack_cols.json") as f:
        valid_cols = json.load(f)
    imputed_path = os.path.join(work_dir, f"{dataset_name}_imputed.csv")
    impute_file(drop_path, imputed_path, valid_cols, imputer)
    # === STEP 5: ENCODING ===
    encoder = joblib.load("preprocessing/models_preprocessing/encoder.pkl")
    with open("preprocessing/models_preprocessing/non_num_cols.json") as f:
        non_num_cols = json.load(f)

    df = safe_read_csv(imputed_path)
    freq_cols = [
        'ip.src_host', 'ip.dst_host', 'ip.host', 'ip.addr', 'ip.src', 'ip.dst',
        'tcp.srcport', 'tcp.dstport', 'udp.srcport', 'udp.dstport',
        'pfcp.node_id_ipv4', 'pfcp.outer_hdr_creation.ipv4',
        'pfcp.f_teid.ipv4_addr', 'pfcp.f_seid.ipv4',
        'pfcp.outer_hdr_creation.teid', 'pfcp.ue_ip_addr_ipv4',
        'tcp.checksum', 'udp.checksum'
    ]
    time_columns = [
        'pfcp.time_of_first_packet', 'pfcp.time_of_last_packet',
        'pfcp.end_time', 'pfcp.recovery_time_stamp'
    ]
    special_columns = ['ip.opt.time_stamp']

    timestamp_col = df[special_columns[0]] if special_columns[0] in df.columns else None
    if special_columns[0] in df.columns:
        df = df.drop(columns=[special_columns[0]])

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
            index=df.index
        )
        df = pd.concat([df.drop(columns=cat_cols), df_encoded], axis=1)

    encoded_path = os.path.join(work_dir, f"{dataset_name}_encoded.csv")
    # Insert special timestamp column in correct position
    if timestamp_col is not None:
        if special_columns[0] not in df.columns:
            df_ref = safe_read_csv("preprocessing/final_datasets_from_preprocessing/dataset_3_encoded.csv", nrows=1)
            col_order = list(df_ref.columns)
            insert_pos = col_order.index(special_columns[0]) if special_columns[0] in col_order else len(df.columns)
            df.insert(insert_pos, special_columns[0], timestamp_col)
    df.to_csv(encoded_path, sep=';', index=False)

    # === STEP 6: CORRELATION FILTERING ===
    with open("preprocessing/models_preprocessing/cols_to_remove.json") as f:
        cols_to_remove = json.load(f)
    filtered_path = os.path.join(work_dir, f"{dataset_name}_filtered.csv")
    apply_filter_and_save(encoded_path, filtered_path, cols_to_remove)

    # === STEP 7: Z-SCORE NORMALIZATION ===
    scaler = joblib.load("preprocessing/models_preprocessing/scaler.pkl")
    with open("preprocessing/models_preprocessing/columns_to_scale.json") as f:
        columns_to_scale = json.load(f)
    final_path = os.path.join(work_dir, f"{dataset_name}_final.csv")
    transform_file_with_scaler(filtered_path, final_path, scaler, columns_to_scale,
                               exclude_cols=["ip.opt.time_stamp", "frame.number", "source_file"], chunksize=50000, sep=';')

    if temp_input and os.path.exists(temp_input):
        os.remove(temp_input)
    final_df = safe_read_csv(final_path)
    print(f"✅ Dataset preprocessed and saved in: {final_path}")

    if in_memory and temp_dir is not None:
        # Remove temporary workspace
        try:
            shutil.rmtree(temp_dir)
        except OSError:
            pass

    return final_df




