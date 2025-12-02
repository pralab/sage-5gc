"""Script to create initial train and test datasets from raw datasets."""

from pathlib import Path

import pandas as pd


def drop_tcp_and_icmp_packets(packets: pd.DataFrame) -> pd.DataFrame:
    """
    Filters out TCP and ICMP packets from the given DataFrame of network packets.
    Drops columns related to TCP features.

    Parameters
    ----------
    packets : pd.DataFrame
        DataFrame containing network packet data.

    Returns
    -------
    pd.DataFrame
        DataFrame with TCP and ICMP packets removed and TCP columns dropped.
    """
    # Filter out TCP (ip.proto == 6) and ICMP (ip.proto == 1) packets
    mask = packets["ip.proto"].isin([1, 6])
    removed = int(mask.sum())
    filtered_packets = packets[~mask].copy()
    print(f"Removed {removed} rows with ip.proto == 1 or 6")
    return filtered_packets


df1 = pd.read_csv(
    Path(__file__).parent.parent / "data/raw_datasets/dataset_1_cleaned.csv",
    sep=";",
    low_memory=False,
)
df2 = pd.read_csv(
    Path(__file__).parent.parent / "data/raw_datasets/dataset_2_cleaned.csv",
    sep=";",
    low_memory=False,
)
df3 = pd.read_csv(
    Path(__file__).parent.parent / "data/raw_datasets/dataset_3_cleaned.csv",
    sep=";",
    low_memory=False,
)

dataset_dir = Path(__file__).parent.parent / "data/datasets"
dataset_dir.mkdir(exist_ok=True)

# Create train dataset concatenating dataset 1 and legitimate samples of dataset 2
# legitimate samples of dataset 2 are those with label ip.opt.time_stamp = NaN,
df_train = pd.concat([df1, df2[df2["ip.opt.time_stamp"].isna()]], ignore_index=True)
print("Train dataset shape:", df_train.shape)

# Create test dataset with all samples of dataset 3 and attack samples of dataset 2
df_test = pd.concat([df2[~df2["ip.opt.time_stamp"].isna()], df3], ignore_index=True)
print("Test dataset shape:", df_test.shape)

df_train_filtered = drop_tcp_and_icmp_packets(df_train)
df_test_filtered = drop_tcp_and_icmp_packets(df_test)

# Remove TCP-related columns
tcp_columns = [col for col in df_train_filtered.columns if col.startswith("tcp.")]
df_train_filtered.drop(columns=tcp_columns, inplace=True)
df_test_filtered.drop(columns=tcp_columns, inplace=True)

df_train_filtered.to_csv(dataset_dir / "train_dataset.csv", sep=";", index=False)
df_test_filtered.to_csv(dataset_dir / "test_dataset.csv", sep=";", index=False)
