import json
import os
from pathlib import Path
import shutil
import tempfile

import joblib
from joblib import dump
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def safe_read_csv(path, **kwargs):
    """Prova UTF-8, poi Latin-1. Stampa la prima riga anche se si usa chunksize."""

    def load_and_preview(encoding):
        df = pd.read_csv(path, sep=";", low_memory=False, encoding=encoding, **kwargs)
        # Caso 1 → DataFrame
        if isinstance(df, pd.DataFrame):
            return df
        # Caso 2 → TextFileReader (chunksize > 0)
        else:
            first_chunk = next(df)  # prende il primo chunk
            return df

    try:
        return load_and_preview("utf-8")
    except UnicodeDecodeError:
        return load_and_preview("latin-1")


# ============================================================================
# STEP 2: TCP OPTIONS PARSING
# ============================================================================
def enrich_tcp_columns(input_file, output_file, chunksize=100_000):
    write_header = True
    for chunk in pd.read_csv(
        input_file, sep=";", chunksize=chunksize, low_memory=False
    ):
        if "tcp.options" in chunk.columns:
            chunk["tcp.options"] = chunk["tcp.options"].fillna("")
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
        cols_to_drop = [col for col in columns_to_delete if col in chunk.columns]
        chunk.drop(columns=cols_to_drop, inplace=True)

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
    exclude_cols = ["ip.opt.time_stamp", "frame.number", "source_file"]
    numerics = None
    collected_chunks = []

    for chunk in pd.read_csv(file_path, sep=";", chunksize=chunksize, low_memory=False):
        numeric_cols = chunk.select_dtypes(include="number").columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

        if numerics is None:
            numerics = numeric_cols
        collected_chunks.append(chunk[numeric_cols])

    full_df = pd.concat(collected_chunks, axis=0)
    valid_cols = full_df.columns[full_df.notna().any()].tolist()
    full_df = full_df[valid_cols]

    imputer = SimpleImputer(strategy="median")
    imputer.fit(full_df)

    return valid_cols, imputer


def impute_file(input_file, output_file, valid_cols, imputer, chunksize=100000):
    exclude_cols = ["ip.opt.time_stamp", "frame.number", "source_file"]
    write_header = True

    for chunk in pd.read_csv(
        input_file, sep=";", chunksize=chunksize, low_memory=False
    ):
        excluded_data = {
            col: chunk[col] for col in exclude_cols if col in chunk.columns
        }
        chunk_numeric = chunk[valid_cols]
        chunk_imputed = pd.DataFrame(
            imputer.transform(chunk_numeric), columns=valid_cols, index=chunk.index
        )

        for col, data in excluded_data.items():
            chunk_imputed[col] = data

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
    freq = df[col].value_counts()
    encoding = freq.rank(method="dense", ascending=False).astype(int)
    return df[col].map(encoding)


def time_conversion(df, col):
    df[col] = pd.to_datetime(
        df[col], format="%b %d, %Y %H:%M:%S.%f %Z", errors="coerce"
    )
    df[col] = df[col].astype("int64") // 10**9
    return df[col]


# ============================================================================
# STEP 6: CORRELATION FILTERING
# ============================================================================
def compute_pairwise_correlations(
    file_path, ref_col="ip.opt.time_stamp", special_cols=None
):
    """
    Calcule la matrice de corrélation d'un dataset,
    retourne matrice et corrélation avec le label.
    """
    if special_cols is None:
        special_cols = ["ip.opt.time_stamp", "frame.number"]

    print(f"Chargement échantillon {file_path}...")
    df = pd.read_csv(
        file_path, sep=";", nrows=5000, low_memory=False, encoding="latin-1"
    )

    # Forcer la colonne ref_col en numérique
    if ref_col in df.columns:
        df[ref_col] = pd.to_numeric(df[ref_col], errors="coerce").fillna(-1).astype(int)

    # Colonnes numériques sauf spéciales
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [
        c
        for c in numeric_cols
        if c
        not in special_cols
        + ["tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport"]
    ]

    # Matrice de corrélation
    corr_matrix = df[numeric_cols].corr(method="pearson")

    # Corrélation avec le label
    special_corr = (
        df[numeric_cols + [ref_col]].corr(method="pearson")[ref_col].drop(ref_col)
    )

    return numeric_cols, corr_matrix, special_corr


def find_common_correlated_pairs(
    corr2, corr3, cols, special_corr2, special_corr3, threshold=0.90
):
    """
    Trouve les paires très corrélées dans corr2 ET corr3,
    choisit la colonne à supprimer selon la corrélation avec le label.
    """
    correlated_pairs = []
    cols_to_remove = set()

    for i, col1 in enumerate(cols):
        for j in range(i + 1, len(cols)):
            col2 = cols[j]

            # Vérifier si la paire est très corrélée dans les deux matrices
            c2 = corr2.loc[col1, col2] if col1 in corr2 and col2 in corr2 else None
            c3 = corr3.loc[col1, col2] if col1 in corr3 and col2 in corr3 else None

            if (
                c2 is not None
                and c3 is not None
                and abs(c2) >= threshold
                and abs(c3) >= threshold
            ):
                correlated_pairs.append((col1, col2, c2, c3))

                # Choisir colonne à supprimer selon corrélation au label
                corr1 = abs(special_corr2.get(col1, 0)) + abs(
                    special_corr3.get(col1, 0)
                )
                corr2_val = abs(special_corr2.get(col2, 0)) + abs(
                    special_corr3.get(col2, 0)
                )

                if corr1 < corr2_val:
                    cols_to_remove.add(col1)
                    print(
                        f"Suppression {col1} (corr label {corr1:.3f}) vs {col2} (corr label {corr2_val:.3f})"
                    )
                else:
                    cols_to_remove.add(col2)
                    print(
                        f"Suppression {col2} (corr label {corr2_val:.3f}) vs {col1} (corr label {corr1:.3f})"
                    )

    print(
        f"{len(correlated_pairs)} paires corrélées détectées dans les DEUX jeux (|corr| >= {threshold})"
    )
    return cols_to_remove


def apply_filter_and_save(input_file, output_file, cols_to_drop, chunksize=50000):
    """Filtre les colonnes à supprimer et sauvegarde en chunké."""
    write_header = True
    for chunk in pd.read_csv(
        input_file, sep=";", chunksize=chunksize, low_memory=False, encoding="latin-1"
    ):
        chunk_filtered = chunk.drop(columns=cols_to_drop, errors="ignore")
        chunk_filtered.to_csv(
            output_file,
            sep=";",
            index=False,
            header=write_header,
            mode="w" if write_header else "a",
            encoding="latin-1",
        )
        write_header = False


def compute_pearson_filter_multi(input_dir, output_dir):
    # Corrélation séparée sur jeu_2 et jeu_3
    cols2, corr2, special_corr2 = compute_pairwise_correlations(
        f"{output_dir}/dataset_2_encoded.csv"
    )
    cols3, corr3, special_corr3 = compute_pairwise_correlations(
        f"{output_dir}/dataset_3_encoded.csv"
    )

    # Colonnes communes (sinon pas comparable)
    common_cols = sorted(set(cols2).intersection(set(cols3)))
    print(f"Colonnes numériques communes aux deux attaques : {len(common_cols)}")

    # Trouver colonnes à supprimer
    cols_to_remove = find_common_correlated_pairs(
        corr2, corr3, common_cols, special_corr2, special_corr3, threshold=0.90
    )

    print("Colonnes finales à retirer globalement :")
    for col in sorted(cols_to_remove):
        print(f" - {col}")
    os.makedirs("models_preprocessing", exist_ok=True)
    with open("models_preprocessing/cols_to_remove.json", "w") as f:
        json.dump(list(cols_to_remove), f, indent=2)

    # Appliquer suppression sur les 3 jeux
    apply_filter_and_save(
        f"{output_dir}/dataset_1_encoded.csv",
        f"{output_dir}/dataset_1_filtered.csv",
        cols_to_remove,
    )
    apply_filter_and_save(
        f"{output_dir}/dataset_2_encoded.csv",
        f"{output_dir}/dataset_2_filtered.csv",
        cols_to_remove,
    )
    apply_filter_and_save(
        f"{output_dir}/dataset_3_encoded.csv",
        f"{output_dir}/dataset_3_filtered.csv",
        cols_to_remove,
    )


# ============================================================================
# STEP 7: Z-SCORE NORMALIZATION
# ============================================================================
def fit_scaler_on_file(file_in, exclude_cols=None, chunksize=50000, sep=";"):
    print(f"[FIT] Fichier : {file_in}")
    scaler = StandardScaler()
    columns_to_scale = None

    for chunk in pd.read_csv(file_in, chunksize=chunksize, sep=sep):
        if columns_to_scale is None:
            exclude_cols = exclude_cols or []
            columns_to_scale = [col for col in chunk.columns if col not in exclude_cols]
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
    write_header = True
    for chunk in pd.read_csv(file_in, chunksize=chunksize, sep=sep):
        chunk_to_scale = chunk[columns_to_scale].fillna(0).astype(float)
        scaled = pd.DataFrame(
            scaler.transform(chunk_to_scale),
            columns=columns_to_scale,
            index=chunk.index,
        )

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


# Preprocessing pipeline execution
def preprocessing_pipeline(
    input_dir="cleaned_dataset", output_dir="final_datasets_from_preprocessing"
):
    # Step 2: TCP OPTIONS PARSING
    enrich_tcp_columns(
        f"{input_dir}/dataset_1_cleaned.csv", f"{output_dir}/dataset_1_tcp.csv"
    )
    enrich_tcp_columns(
        f"{input_dir}/dataset_2_cleaned.csv", f"{output_dir}/dataset_2_tcp.csv"
    )
    enrich_tcp_columns(
        f"{input_dir}/dataset_3_cleaned.csv", f"{output_dir}/dataset_3_tcp.csv"
    )

    # Step 3: ADVANCED CLEANING
    drop_columns_chunked(
        f"{output_dir}/dataset_1_tcp.csv",
        f"{output_dir}/dataset_1_drop.csv",
        is_attack=False,
    )
    drop_columns_chunked(
        f"{output_dir}/dataset_2_tcp.csv",
        f"{output_dir}/dataset_2_drop.csv",
        is_attack=True,
    )
    drop_columns_chunked(
        f"{output_dir}/dataset_3_tcp.csv",
        f"{output_dir}/dataset_3_drop.csv",
        is_attack=True,
    )

    # Step 4: IMPUTE NUMERICAL
    saine_cols, saine_imputer = compute_numerical_medians(
        f"{output_dir}/dataset_1_drop.csv"
    )
    impute_file(
        f"{output_dir}/dataset_1_drop.csv",
        f"{output_dir}/dataset_1_imputed.csv",
        saine_cols,
        saine_imputer,
    )
    attack_cols, attack_imputer = compute_numerical_medians(
        f"{output_dir}/dataset_2_drop.csv"
    )
    impute_file(
        f"{output_dir}/dataset_2_drop.csv",
        f"{output_dir}/dataset_2_imputed.csv",
        attack_cols,
        attack_imputer,
    )
    impute_file(
        f"{output_dir}/dataset_3_drop.csv",
        f"{output_dir}/dataset_3_imputed.csv",
        attack_cols,
        attack_imputer,
    )
    os.makedirs("models_preprocessing", exist_ok=True)
    dump(saine_imputer, "models_preprocessing/imputer_saine.pkl")
    dump(attack_imputer, "models_preprocessing/imputer_attack.pkl")
    with open("models_preprocessing/saine_cols.json", "w") as f:
        json.dump(saine_cols, f, indent=2)
    with open("models_preprocessing/attack_cols.json", "w") as f:
        json.dump(attack_cols, f, indent=2)

    # Step 5: ENCODING
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

    df_attack = pd.read_csv(
        f"{output_dir}/dataset_2_imputed.csv", sep=";", low_memory=False
    )
    df_saine = pd.read_csv(
        f"{output_dir}/dataset_1_imputed.csv", sep=";", low_memory=False
    )
    df_attack2 = pd.read_csv(
        f"{output_dir}/dataset_3_imputed.csv", sep=";", low_memory=False
    )

    non_num_attack = df_attack.select_dtypes(include=["object"]).columns.tolist()
    non_num_saine = df_saine.select_dtypes(include=["object"]).columns.tolist()
    non_num_attack2 = df_attack2.select_dtypes(include=["object"]).columns.tolist()

    non_num_cols = sorted(
        set(non_num_attack).union(set(non_num_saine)).union(set(non_num_attack2))
    )

    # Nettoyage : on retire les colonnes déjà frequency-encodées, spéciales et temporelles
    non_num_cols = [
        col
        for col in non_num_cols
        if col not in freq_cols
        and col not in special_columns
        and col not in time_columns
    ]

    # forcer la conversion des colonnes object qui sont en réalité numériques
    fake_num_cols = []
    for col in list(non_num_cols):
        try:
            converted = pd.to_numeric(df_attack[col], errors="coerce")
            if converted.notna().sum() > 0 and converted.nunique() > 1:
                fake_num_cols.append(col)
                for df_tmp in [df_attack, df_saine, df_attack2]:
                    if col in df_tmp.columns:
                        df_tmp[col] = pd.to_numeric(df_tmp[col], errors="coerce")
                non_num_cols.remove(col)
        except Exception:
            pass

    if fake_num_cols:
        print("Colonnes object recastées en numériques :", fake_num_cols)

    print("=> Préparation du OneHotEncoder global...")
    df_attack_cat = df_attack[non_num_cols]
    df_saine_cat = df_saine[non_num_cols]
    df_attack2_cat = df_attack2[non_num_cols]

    df_cat_all = pd.concat([df_attack_cat, df_saine_cat, df_attack2_cat], axis=0)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(df_cat_all[non_num_cols])
    dump(encoder, "models_preprocessing/encoder.pkl")
    with open("models_preprocessing/non_num_cols.json", "w") as f:
        json.dump(non_num_cols, f, indent=2)

    del (
        non_num_attack,
        non_num_saine,
        non_num_attack2,
        df_attack_cat,
        df_saine_cat,
        df_attack2_cat,
        df_cat_all,
    )

    ### Boucle traitement attack/saine/attack2
    data_frames = ["df_attack", "df_saine", "df_attack2"]
    for data_frame in data_frames:
        if data_frame == "df_attack":
            df = df_attack
            timestamp_col = df[special_columns[0]] if special_columns[0] in df else None
            if special_columns[0] in df:
                df = df.drop(columns=[special_columns[0]])
        elif data_frame == "df_saine":
            df = df_saine.copy()
            timestamp_col = None
        else:  # df_attack2
            df = df_attack2
            timestamp_col = df[special_columns[0]] if special_columns[0] in df else None
            if special_columns[0] in df:
                df = df.drop(columns=[special_columns[0]])

        print("--> Frequency encoding des colonnes :")
        for col in freq_cols:
            if col in df.columns:
                print(f"   {data_frame}: {col}")
                df[col] = frequency_encode(df, col)

        df_freq_cols = df[[c for c in freq_cols if c in df.columns]]

        print("--> Time conversion:")
        for col in time_columns:
            if col in df.columns:
                print(f"    {data_frame}: {col}")
                df[col] = time_conversion(df, col)
        df_time_columns = df[[c for c in time_columns if c in df.columns]]
        df = df[
            [
                col
                for col in df.columns
                if col not in freq_cols and col not in time_columns
            ]
        ]

        print(f"Colonnes catégorielles à encoder : {len(non_num_cols)}")
        df[non_num_cols] = df[non_num_cols].fillna("NaN").astype(str)

        df_encoded = pd.DataFrame(
            encoder.transform(df[non_num_cols]),
            columns=encoder.get_feature_names_out(non_num_cols),
        )

        df = df.drop(columns=non_num_cols).reset_index(drop=True)
        print("print df.columns: ", df.columns)
        print("Sauvegarde finale des fichiers avec colonnes spéciales...")

        df.to_csv("df_main.csv", sep=";", index=False)
        df_freq_cols.to_csv("df_freq.csv", sep=";", index=False)
        df_encoded.to_csv("df_encoded.csv", sep=";", index=False)
        df_time_columns.to_csv("df_time_columns.csv", sep=";", index=False)

        del df, df_freq_cols, df_encoded, df_time_columns

        chunk_size = 100000
        if data_frame == "df_attack":
            if timestamp_col is not None:
                timestamp_col.to_csv("df_timestamp.csv", sep=";", index=False)
            header_written = False
            with open(f"{output_dir}/dataset_2_encoded.csv", "w") as f_out:
                for parts in zip(
                    pd.read_csv("df_main.csv", sep=";", chunksize=chunk_size),
                    pd.read_csv("df_freq.csv", sep=";", chunksize=chunk_size),
                    pd.read_csv("df_encoded.csv", sep=";", chunksize=chunk_size),
                    (
                        pd.read_csv("df_timestamp.csv", sep=";", chunksize=chunk_size)
                        if timestamp_col is not None
                        else [pd.DataFrame()]
                    ),
                    pd.read_csv("df_time_columns.csv", sep=";", chunksize=chunk_size),
                ):
                    merged = pd.concat([p for p in parts if not p.empty], axis=1)
                    merged.to_csv(
                        f_out, sep=";", index=False, header=not header_written
                    )
                    header_written = True
        elif data_frame == "df_saine":
            header_written = False
            with open(f"{output_dir}/dataset_1_encoded.csv", "w") as f_out:
                for parts in zip(
                    pd.read_csv("df_main.csv", sep=";", chunksize=chunk_size),
                    pd.read_csv("df_freq.csv", sep=";", chunksize=chunk_size),
                    pd.read_csv("df_encoded.csv", sep=";", chunksize=chunk_size),
                    pd.read_csv("df_time_columns.csv", sep=";", chunksize=chunk_size),
                ):
                    merged = pd.concat([p for p in parts if not p.empty], axis=1)
                    merged.to_csv(
                        f_out, sep=";", index=False, header=not header_written
                    )
                    header_written = True
        else:  # df_attack2
            if timestamp_col is not None:
                timestamp_col.to_csv("df_timestamp.csv", sep=";", index=False)
            header_written = False
            with open(f"{output_dir}/dataset_3_encoded.csv", "w") as f_out:
                for parts in zip(
                    pd.read_csv("df_main.csv", sep=";", chunksize=chunk_size),
                    pd.read_csv("df_freq.csv", sep=";", chunksize=chunk_size),
                    pd.read_csv("df_encoded.csv", sep=";", chunksize=chunk_size),
                    (
                        pd.read_csv("df_timestamp.csv", sep=";", chunksize=chunk_size)
                        if timestamp_col is not None
                        else [pd.DataFrame()]
                    ),
                    pd.read_csv("df_time_columns.csv", sep=";", chunksize=chunk_size),
                ):
                    merged = pd.concat([p for p in parts if not p.empty], axis=1)
                    merged.to_csv(
                        f_out, sep=";", index=False, header=not header_written
                    )
                    header_written = True

        for temp_file in [
            "df_main.csv",
            "df_freq.csv",
            "df_encoded.csv",
            "df_timestamp.csv",
            "df_time_columns.csv",
        ]:
            try:
                os.remove(temp_file)
            except FileNotFoundError:
                pass

    # Step 6: CORRELATION FILTERING
    compute_pearson_filter_multi(input_dir, output_dir)

    df1 = pd.read_csv(f"{output_dir}/dataset_1_filtered.csv", sep=";", nrows=1)
    df2 = pd.read_csv(f"{output_dir}/dataset_2_filtered.csv", sep=";", nrows=1)
    df3 = pd.read_csv(f"{output_dir}/dataset_3_filtered.csv", sep=";", nrows=1)
    common_filtered_cols = list(set(df1.columns) & set(df2.columns) & set(df3.columns))
    with open("models_preprocessing/common_filtered_cols.json", "w") as f:
        json.dump(common_filtered_cols, f, indent=2)

    # Step 7: Z-SCORE NORMALIZATION
    sep = ";"
    chunksize = 50000
    exclude_att = ["ip.opt.time_stamp", "frame.number", "source_file"]
    scaler, columns_to_scale = fit_scaler_on_file(
        f"{output_dir}/dataset_1_filtered.csv",
        exclude_cols=exclude_att,
        chunksize=chunksize,
        sep=sep,
    )
    dump(scaler, "models_preprocessing/scaler.pkl")
    with open("models_preprocessing/columns_to_scale.json", "w") as f:
        json.dump(columns_to_scale, f, indent=2)

    transform_file_with_scaler(
        f"{output_dir}/dataset_1_filtered.csv",
        f"{output_dir}/dataset_1_final.csv",
        scaler,
        columns_to_scale,
        exclude_cols=exclude_att,
        chunksize=chunksize,
        sep=sep,
    )
    transform_file_with_scaler(
        f"{output_dir}/dataset_2_filtered.csv",
        f"{output_dir}/dataset_2_final.csv",
        scaler,
        columns_to_scale,
        exclude_cols=exclude_att,
        chunksize=chunksize,
        sep=sep,
    )
    transform_file_with_scaler(
        f"{output_dir}/dataset_3_filtered.csv",
        f"{output_dir}/dataset_3_final.csv",
        scaler,
        columns_to_scale,
        exclude_cols=exclude_att,
        chunksize=chunksize,
        sep=sep,
    )


# SINGLE DATASET
def preprocessing_pipeline_partial(
    output_dir: str,
    dataset_name: str,
    input_file: str | None = None,
    df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Applica lo stesso preprocessing della pipeline completa, ma su un singolo dataset.
    Può ricevere direttamente un DataFrame o un percorso a file CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    work_dir = output_dir

    # === STEP 1: Caricamento input ===
    if df is None:
        if input_file is None:
            raise ValueError("Devi specificare input_file o df.")
        df = safe_read_csv(input_file)

    # Salva il DataFrame come file temporaneo per mantenere compatibilità con enrich_tcp_columns
    temp_input = os.path.join(work_dir, f"{dataset_name}_input_temp.csv")
    df.to_csv(temp_input, sep=";", index=False)
    source_path = temp_input if input_file is None else input_file

    # === STEP 2: TCP OPTIONS PARSING ===
    tcp_path = os.path.join(work_dir, f"{dataset_name}_tcp.csv")
    enrich_tcp_columns(source_path, tcp_path)

    # === STEP 3: ADVANCED CLEANING ===
    drop_path = os.path.join(work_dir, f"{dataset_name}_drop.csv")
    drop_columns_chunked(tcp_path, drop_path, is_attack=True)

    # === STEP 4: IMPUTE NUMERICAL ===
    imputer = joblib.load(Path(__file__).parent / "models_preprocessing/imputer_attack.pkl")
    # imputer = joblib.load("models_preprocessing/imputer_attack.pkl")
    with open(Path(__file__).parent / "models_preprocessing/attack_cols.json") as f:
        # with open("models_preprocessing/attack_cols.json") as f:
        valid_cols = json.load(f)
    imputed_path = os.path.join(work_dir, f"{dataset_name}_imputed.csv")
    impute_file(drop_path, imputed_path, valid_cols, imputer)
    # === STEP 5: ENCODING ===
    encoder = joblib.load(Path(__file__).parent / "models_preprocessing/encoder.pkl")
    # encoder = joblib.load("models_preprocessing/encoder.pkl")
    with open(Path(__file__).parent / "models_preprocessing/non_num_cols.json") as f:
        # with open("models_preprocessing/non_num_cols.json") as f:
        non_num_cols = json.load(f)

    df = safe_read_csv(imputed_path)

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

    # Frequency encoding
    for col in freq_cols:
        if col in df.columns:
            df[col] = frequency_encode(df, col)

    # Time conversion
    for col in time_columns:
        if col in df.columns:
            df[col] = time_conversion(df, col)

    # One-hot encoding (solo categorie già conosciute)
    cat_cols = [col for col in non_num_cols if col in df.columns]
    if cat_cols:
        df[cat_cols] = df[cat_cols].fillna("NaN").astype(str)
        df_encoded = pd.DataFrame(
            encoder.transform(df[cat_cols]),
            columns=encoder.get_feature_names_out(cat_cols),
            index=df.index,
        )
        df = pd.concat([df.drop(columns=cat_cols), df_encoded], axis=1)

    encoded_path = os.path.join(work_dir, f"{dataset_name}_encoded.csv")
    # Reinserisci ip.opt.time_stamp nel dataframe finale, come fa la pipeline completa
    if timestamp_col is not None:
        if special_columns[0] not in df.columns:
            # Trova la posizione originale leggendo la struttura del dataset originale
            df_ref = safe_read_csv(
                "preprocessing/final_datasets_from_preprocessing/dataset_3_encoded.csv",
                nrows=1,
            )
            # df_ref = safe_read_csv("final_datasets_from_preprocessing/dataset_3_encoded.csv", nrows=1)
            col_order = list(df_ref.columns)
            insert_pos = (
                col_order.index(special_columns[0])
                if special_columns[0] in col_order
                else len(df.columns)
            )
            df.insert(insert_pos, special_columns[0], timestamp_col)
    # Salva il file encoded completo
    df.to_csv(encoded_path, sep=";", index=False)

    # === STEP 6: CORRELATION FILTERING ===
    with open("preprocessing/models_preprocessing/cols_to_remove.json") as f:
        # with open("models_preprocessing/cols_to_remove.json") as f:
        cols_to_remove = json.load(f)
    filtered_path = os.path.join(work_dir, f"{dataset_name}_filtered.csv")
    apply_filter_and_save(encoded_path, filtered_path, cols_to_remove)

    # === STEP 7: Z-SCORE NORMALIZATION ===
    scaler = joblib.load("preprocessing/models_preprocessing/scaler.pkl")
    # scaler = joblib.load("models_preprocessing/scaler.pkl")
    with open("preprocessing/models_preprocessing/columns_to_scale.json") as f:
        # with open("models_preprocessing/columns_to_scale.json") as f:
        columns_to_scale = json.load(f)
    final_path = os.path.join(work_dir, f"{dataset_name}_final.csv")
    transform_file_with_scaler(
        filtered_path,
        final_path,
        scaler,
        columns_to_scale,
        exclude_cols=["ip.opt.time_stamp", "frame.number", "source_file"],
        chunksize=50000,
        sep=";",
    )

    if temp_input and os.path.exists(temp_input):
        os.remove(temp_input)
    final_df = safe_read_csv(final_path)
    print(f"✅ Dataset preprocessato e salvato in: {final_path}")

    if in_memory and temp_dir is not None:
        # Cancella la directory temporanea con tutti i file intermedi
        try:
            shutil.rmtree(temp_dir)
        except OSError:
            pass

    return final_df
