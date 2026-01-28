"""
Script to create initial train and test datasets from raw datasets.
It performs the following steps:
1. Merges raw datasets to create train and test datasets.
2. Filters out TCP and ICMP packets.
3. Drops useless and constant columns.
4. Converts categoric columns to numeric.
5. Imputes missing values.
6. Restores categoric columns.
"""

import logging
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.utils import (
    convert_to_numeric,
    drop_constant_columns,
    drop_useless_columns,
    load_imputers,
    restore_categoric_columns,
)

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger.setLevel(logging.INFO)


def drop_tcp_and_icmp_packets(packets: pd.DataFrame) -> pd.DataFrame:
    # Filter out TCP (ip.proto == 6) and ICMP (ip.proto == 1) packets
    mask = packets["ip.proto"].isin([1, 6])
    filtered_packets = packets[~mask].copy()
    return filtered_packets


if __name__ == "__main__":
    df1 = pd.read_csv(
        Path(__file__).parent.parent
        / "data/raw_datasets/dataset_1_cleaned.csv",
        sep=";",
        low_memory=False,
    )
    df2 = pd.read_csv(
        Path(__file__).parent.parent
        / "data/raw_datasets/dataset_2_cleaned.csv",
        sep=";",
        low_memory=False,
    )
    df3 = pd.read_csv(
        Path(__file__).parent.parent
        / "data/raw_datasets/dataset_3_cleaned.csv",
        sep=";",
        low_memory=False,
    )

    dataset_dir = Path(__file__).parent.parent / "data/datasets"
    dataset_dir.mkdir(exist_ok=True)

    # ----------------------------------------------------------
    # [Step 1] Merge datasets to create train and test datasets
    # ----------------------------------------------------------
    logger.info("Creating train and test datasets...")

    # Create train dataset concatenating dataset 1 and legitimate samples of dataset 2
    # legitimate samples of dataset 2 are those with label ip.opt.time_stamp = NaN,
    df_train = pd.concat(
        [df1, df2[df2["ip.opt.time_stamp"].isna()]], ignore_index=True
    )
    logger.info("Train dataset shape:", df_train.shape)

    # Create test dataset with all samples of dataset 3 and attack samples of dataset 2
    df_test = pd.concat(
        [df2[~df2["ip.opt.time_stamp"].isna()], df3], ignore_index=True
    )
    logger.info("Test dataset shape:", df_test.shape)

    # -----------------------------------------
    # [Step 2] Filter out TCP and ICMP packets
    # -----------------------------------------
    logger.info("Filtering out TCP and ICMP packets...")

    df_train_filtered = drop_tcp_and_icmp_packets(df_train)
    df_test_filtered = drop_tcp_and_icmp_packets(df_test)

    tcp_columns = [
        col for col in df_train_filtered.columns if col.startswith("tcp.")
    ]
    df_train_filtered.drop(columns=tcp_columns, inplace=True)
    df_test_filtered.drop(columns=tcp_columns, inplace=True)

    # --------------------------------------
    # [Step 3] Separate features and labels
    # --------------------------------------
    labels_train = df_train_filtered["ip.opt.time_stamp"].copy()
    labels_test = df_test_filtered["ip.opt.time_stamp"].copy()

    df_train_filtered.drop(columns=["ip.opt.time_stamp"], inplace=True)
    df_test_filtered.drop(columns=["ip.opt.time_stamp"], inplace=True)

    # ------------------------------
    # [Step 4] Drop useless columns
    # ------------------------------
    logger.info("Dropping useless and constant columns...")

    df_train_filtered = drop_useless_columns(df_train_filtered)
    df_test_filtered = drop_useless_columns(df_test_filtered)
    df_train_filtered = drop_constant_columns(df_train_filtered)
    df_test_filtered = drop_constant_columns(df_test_filtered)

    # ----------------------------------------------
    # [Step 5] Convert categoric columns to numeric
    # ----------------------------------------------
    logger.info("Converting categoric columns to numeric...")

    df_train_processed, cat_cols_train = convert_to_numeric(df_train_filtered)
    df_test_processed, cat_cols_test = convert_to_numeric(df_test_filtered)

    # -------------------------------
    # [Step 6] Impute missing values
    # -------------------------------
    logger.info("Imputing missing values...")

    simple_imputer, iter_imputer = load_imputers(random_state=42)

    # Train data
    cat_cols = df_train_processed.select_dtypes(include=["category"]).columns
    num_cols = df_train_processed.select_dtypes(exclude=["category"]).columns

    df_train_filtered[cat_cols] = simple_imputer.fit_transform(
        df_train_processed[cat_cols]
    )
    df_train_filtered[num_cols] = iter_imputer.fit_transform(
        df_train_processed[num_cols]
    )

    # Test data
    cat_cols = df_test_processed.select_dtypes(include=["category"]).columns
    num_cols = df_test_processed.select_dtypes(exclude=["category"]).columns

    df_test_filtered[cat_cols] = simple_imputer.transform(
        df_test_processed[cat_cols]
    )
    df_test_filtered[num_cols] = iter_imputer.transform(
        df_test_processed[num_cols]
    )

    round_cols = [
        "pfcp.duration_measurement",
        "pfcp.ie_type",
        "pfcp.msg_type",
        "pfcp.pdr_id",
        "pfcp.response_to",
        "pfcp.volume_measurement.dlnop",
        "pfcp.volume_measurement.dlvol",
        "pfcp.volume_measurement.tonop",
        "pfcp.volume_measurement.tovol",
    ]
    for col in round_cols:
        if col in df_train_filtered.columns:
            df_train_filtered[col] = (
                df_train_filtered[col].round().astype(float)
            )
        if col in df_test_filtered.columns:
            df_test_filtered[col] = df_test_filtered[col].round().astype(float)

    # -----------------------------------
    # [Step 7] Restore categoric columns
    # -----------------------------------
    logger.info("Restoring categoric columns...")

    df_train_filtered = restore_categoric_columns(
        df_train_filtered, cat_cols_train
    )
    df_test_filtered = restore_categoric_columns(
        df_test_filtered, cat_cols_test
    )

    df_train_filtered["ip.opt.time_stamp"] = labels_train
    df_test_filtered["ip.opt.time_stamp"] = labels_test

    df_train_filtered.to_csv(
        dataset_dir / "train_dataset2.csv", sep=";", index=False
    )
    df_test_filtered.to_csv(
        dataset_dir / "test_dataset2.csv", sep=";", index=False
    )

    logger.info("Datasets created and saved successfully.")
