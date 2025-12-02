"""
This module defines a function to filter out TCP and ICMP packets from a list of network packets,
and drop the columns features related to tcp.
TCP packets have ip.proto value 6, while ICMP packets have ip.proto value 1.
TCP columns are identified by the prefix 'tcp.'.
"""

import pandas as pd
import os

def drop_tcp_and_icmp_packets(packets: pd.DataFrame) -> pd.DataFrame:
    """
    Filters out TCP and ICMP packets from the given DataFrame of network packets.
    Drops columns related to TCP features.

    Parameters:
    packets (pd.DataFrame): DataFrame containing network packet data.

    Returns:
    pd.DataFrame: DataFrame with TCP and ICMP packets removed and TCP columns dropped.
    """
    # Filter out TCP (ip.proto == 6) and ICMP (ip.proto == 1) packets
    filtered_packets_tcp = packets[~packets['ip.proto'].isin([6])].copy()
    if filtered_packets_tcp.empty:
        print("Warning: No packets left after filtering out TCP packets.")
    if len(filtered_packets_tcp) == len(packets):
        print("Warning: No TCP packets were found to filter out.")
    if len(filtered_packets_tcp) < len(packets):
        print(f"Info: Filtered out {len(packets) - len(filtered_packets_tcp)} TCP packets.")

    filtered_packets_icmp = packets[~packets['ip.proto'].isin([1])].copy()
    if filtered_packets_icmp.empty:
        print("Warning: No packets left after filtering out ICMP packets.")
    if len(filtered_packets_icmp) == len(packets):
        print("Warning: No ICMP packets were found to filter out.")
    if len(filtered_packets_icmp) < len(packets):
        print(f"Info: Filtered out {len(packets) - len(filtered_packets_icmp)} ICMP packets.")


    filtered_packets = packets[~packets['ip.proto'].isin([1, 6])].copy()

    # Drop columns related to TCP features
    tcp_columns = [col for col in filtered_packets.columns if col.startswith('tcp.')]
    filtered_packets.drop(columns=tcp_columns, inplace=True)
    # Look the dropped columns
    if tcp_columns:
        print(f"Info: Dropped {len(tcp_columns)} TCP-related columns.")
        # print the dropped columns
        print("Dropped TCP columns:", tcp_columns)
        # print the remaining df shape
        print("Remaining DataFrame shape:", filtered_packets.shape)
    else:
        print("Info: No TCP-related columns found to drop.")

    return filtered_packets


df_train = pd.read_csv("new_datasets/merged_raw/train_dataset.csv", sep=";", low_memory=False)
df_test = pd.read_csv("new_datasets/merged_raw/test_dataset.csv", sep=";", low_memory=False)

df_train_filtered = drop_tcp_and_icmp_packets(df_train)
df_test_filtered = drop_tcp_and_icmp_packets(df_test)
os.makedirs("new_datasets/no_tcp_icmp", exist_ok=True)
df_train_filtered.to_csv("new_datasets/no_tcp_icmp/train_dataset.csv", sep=";", index=False)
df_test_filtered.to_csv("new_datasets/no_tcp_icmp/test_dataset.csv", sep=";", index=False)





