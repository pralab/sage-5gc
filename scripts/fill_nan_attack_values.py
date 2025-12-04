"""
Script to fill NaN PFCP fields in malicious samples with realistic values.
This is necessary because some attack samples have missing PFCP values,
due to how the attacks were generated.
"""

import logging
from pathlib import Path
import random

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


ATTACK_TYPE_MAP = {
    0.0: "DoS_Flood",  # PFCP Flooding
    1.0: "DoS_Deletion",  # PFCP Deletion
    2.0: "DoS_Modification",  # PFCP Modification
    3.0: "Reconnaissance",  # NMAP Scan
    4.0: "Lateral_Movement",  # Reverse Shell
    5.0: "DoS_Fault",  # UPF PDN-0 Fault
    6.0: "DoS_Restoration",  # PFCP Restoration-TEID
}

PFCP_FEATURES = [
    "pfcp.node_id_ipv4",
    "pfcp.f_seid.ipv4",
    "pfcp.f_teid.ipv4_addr",
    "pfcp.f_teid.teid",
    "pfcp.f_teid_flags.ch",
    "pfcp.f_teid_flags.ch_id",
    "pfcp.f_teid_flags.v6",
    "pfcp.outer_hdr_creation.ipv4",
    "pfcp.outer_hdr_creation.teid",
    "pfcp.dst_interface",
    "pfcp.pdr_id",
    "pfcp.apply_action.buff",
    "pfcp.apply_action.forw",
    "pfcp.apply_action.nocp",
    "pfcp.ue_ip_addr_ipv4",
    "pfcp.duration_measurement",
    "pfcp.volume_measurement.dlnop",
    "pfcp.volume_measurement.dlvol",
    "pfcp.volume_measurement.tonop",
    "pfcp.volume_measurement.tovol",
    "pfcp.time_of_first_packet",
    "pfcp.time_of_last_packet",
    "pfcp.end_time",
    "pfcp.recovery_time_stamp",
    "pfcp.ie_type",
    "pfcp.ie_len",
    "pfcp.response_time",
    "pfcp.response_to",
]


def generate_smart_attacker_row(attack_type: str = None):
    """
    Generate a realistic PFCP sample for the 5G Core Network.

    Parameters
    ----------
    attack_type : str, optional
        Type of attack (e.g., 'DoS_Flood', 'Reconnaissance').
        If None, one is selected randomly.

    Returns
    -------
    dict
        Dictionary with realistic PFCP values.
    """
    row = {}

    # If the attack type is not specified, pick one randomly
    if attack_type is None:
        attack_type = random.choice(list(ATTACK_TYPE_MAP.values()))

    # 5G topology
    row["pfcp.node_id_ipv4"] = (
        f"192.168.{random.randint(14, 130)}.{random.randint(2, 254)}"
    )
    row["pfcp.ue_ip_addr_ipv4"] = (
        f"10.45.{random.randint(0, 16)}.{random.randint(2, 254)}"
    )

    # GTP-U tunneling
    row["pfcp.f_teid.teid"] = hex(random.randint(29, 65507))
    row["pfcp.pdr_id"] = float(random.choice([1, 2]))

    row["pfcp.f_seid.ipv4"] = (
        f"192.168.{random.randint(14, 130)}.{random.randint(2, 254)}"
    )
    row["pfcp.f_teid.ipv4_addr"] = (
        f"192.168.{random.randint(14, 130)}.{random.randint(2, 254)}"
    )
    row["pfcp.outer_hdr_creation.ipv4"] = (
        f"192.168.{random.randint(14, 130)}.{random.randint(129, 186)}"
    )
    row["pfcp.outer_hdr_creation.teid"] = hex(random.randint(0x1, 0x18B6))

    # F-TEID flags
    row["pfcp.f_teid_flags.v6"] = False
    row["pfcp.f_teid_flags.ch"] = random.choice([True, False])
    row["pfcp.f_teid_flags.ch_id"] = random.choice([True, False])

    # Apply actions (varies by attack type)
    if attack_type in [
        "DoS_Flood",
        "DoS_Deletion",
        "DoS_Modification",
        "DoS_Fault",
        "DoS_Restoration",
    ]:
        row["pfcp.apply_action.forw"] = False
        row["pfcp.apply_action.buff"] = random.choice([True, False])
        row["pfcp.apply_action.nocp"] = random.choice([True, False])
    else:
        action_choice = random.choice([0, 1, 2])
        if action_choice == 0:
            row["pfcp.apply_action.forw"] = True
            row["pfcp.apply_action.buff"] = False
            row["pfcp.apply_action.nocp"] = False
        elif action_choice == 1:
            row["pfcp.apply_action.forw"] = False
            row["pfcp.apply_action.buff"] = True
            row["pfcp.apply_action.nocp"] = False
        else:
            row["pfcp.apply_action.forw"] = False
            row["pfcp.apply_action.buff"] = False
            row["pfcp.apply_action.nocp"] = True

    # Destination interface
    row["pfcp.dst_interface"] = float(random.choice([0, 1]))

    # Volume measurements
    simulated_packets = random.randint(0, 13195)
    avg_pkt_size = random.randint(500, 1200)
    dl_ratio = random.uniform(0.7, 0.9)

    row["pfcp.volume_measurement.tonop"] = float(simulated_packets)
    row["pfcp.volume_measurement.tovol"] = float(simulated_packets * avg_pkt_size)
    row["pfcp.volume_measurement.dlnop"] = float(int(simulated_packets * dl_ratio))
    row["pfcp.volume_measurement.dlvol"] = float(
        int(row["pfcp.volume_measurement.tovol"] * dl_ratio)
    )

    # PFCP timestamps
    row["pfcp.duration_measurement"] = float(random.uniform(1747212643.0, 1753894838.0))
    row["pfcp.time_of_first_packet"] = float(random.randint(1747212464, 1753894823))
    row["pfcp.time_of_last_packet"] = float(random.randint(1747212640, 1753894834))
    row["pfcp.end_time"] = float(random.randint(1747212642, 1753894838))
    row["pfcp.recovery_time_stamp"] = float(random.randint(1747207882, 1753892199))

    # IE type/length
    row["pfcp.ie_type"] = float(random.randint(10, 96))
    row["pfcp.ie_len"] = float(random.randint(1, 50))

    # Response time
    row["pfcp.response_time"] = random.uniform(2.0095e-05, 0.041239073)
    row["pfcp.response_to"] = float(random.randint(1, 2565))

    return row


def fill_nan_pfcp_fields(
    sample: pd.Series, seed: int = None
) -> tuple[pd.Series, int, str]:
    """
    Fill only PFCP NaN fields of a sample with realistic values.
    Existing values are preserved.

    Parameters
    ----------
    sample : pd.Series
        Malicious sample.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    filled_sample : pd.Series
        Sample with NaNs filled.
    nan_count : int
        Number of NaNs filled.
    attack_type : str
        Attack type inferred from the label.
    """
    if seed is not None:
        random.seed(seed)

    # Extract attack type from label
    label = sample.get("ip.opt.time_stamp", None)
    if pd.notna(label):
        attack_type = ATTACK_TYPE_MAP.get(float(label), "Unknown")
    else:
        attack_type = "Unknown"

    # Generate realistic values based on attack type
    generated_values = generate_smart_attacker_row(attack_type)

    # Fill only NaN fields (preserve existing values)
    filled_sample = sample.copy()
    nan_count = 0

    for field in PFCP_FEATURES:
        if field in filled_sample.index and pd.isna(filled_sample[field]):
            if field in generated_values:
                filled_sample[field] = generated_values[field]
                nan_count += 1

    return filled_sample, nan_count, attack_type


def fill_malicious_dataset_nans(
    input_path: str | Path,
    output_path: str | Path,
    label_col: str = "ip.opt.time_stamp",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load the malicious dataset, fill PFCP NaNs with realistic values, and save it.

    Parameters
    ----------
    input_path : str | Path
        Path to the malicious test CSV dataset.
    output_path : str | Path
        Path where the filled dataset will be saved.
    label_col : str
        Label column name (used to filter only malicious samples).
    seed : int
        Base random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Dataset with NaNs filled.
    """
    logger.info(f"Loading malicious dataset from: {input_path}")
    df = pd.read_csv(input_path, sep=";", low_memory=False)

    # Filter malicious samples (label not NaN)
    if label_col in df.columns:
        df_malicious = df[df[label_col].notna()].copy()
        logger.info(
            f"Found {len(df_malicious)} malicious samples (out of {len(df)} total)"
        )
    else:
        logger.warning(
            f"Label column '{label_col}' not found! Processing all samples..."
        )
        df_malicious = df.copy()

    # Initial NaN statistics for PFCP features
    pfcp_cols_present = [col for col in PFCP_FEATURES if col in df_malicious.columns]
    initial_nan_count = df_malicious[pfcp_cols_present].isna().sum().sum()
    logger.info(f"Initial NaN count in PFCP fields: {initial_nan_count}")

    # Statistics per attack type
    attack_type_stats = {}

    # Fill NaNs for each sample
    filled_samples = []
    total_filled = 0

    for idx, (row_idx, sample) in enumerate(df_malicious.iterrows()):
        filled_sample, nan_count, attack_type = fill_nan_pfcp_fields(
            sample, seed=seed + idx
        )
        filled_samples.append(filled_sample)
        total_filled += nan_count

        if attack_type not in attack_type_stats:
            attack_type_stats[attack_type] = {"samples": 0, "nans_filled": 0}
        attack_type_stats[attack_type]["samples"] += 1
        attack_type_stats[attack_type]["nans_filled"] += nan_count

        if (idx + 1) % 100 == 0:
            logger.info(f"Processed {idx + 1}/{len(df_malicious)} samples...")

    df_filled = pd.DataFrame(filled_samples)

    # Final NaN statistics
    final_nan_count = df_filled[pfcp_cols_present].isna().sum().sum()
    logger.info(f"Final NaN count in PFCP fields: {final_nan_count}")
    logger.info(f"Filled {total_filled} NaN values total")

    # Save dataset with filled NaNs
    df_filled.to_csv(output_path, sep=";", index=False)
    logger.info(f"Dataset with filled NaNs saved to: {output_path}")

    # Log statistics by attack type
    logger.info("\n" + "=" * 60)
    logger.info("FILLING STATISTICS BY ATTACK TYPE")
    logger.info("=" * 60)
    for attack_type, stats in sorted(attack_type_stats.items()):
        logger.info(
            f"{attack_type:20s} → {stats['samples']:4d} samples, {stats['nans_filled']:5d} NaNs filled"
        )

    return df_filled


def fill_malicious_dataset_nans_and_reassemble(
    input_path: str | Path,
    output_path: str | Path,
    label_col: str = "ip.opt.time_stamp",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Fill NaNs in malicious samples and reassemble them with the original benign ones.
    """
    logger.info(f"Loading mixed dataset from: {input_path}")
    df = pd.read_csv(input_path, sep=";", low_memory=False)

    # Separate benign and malicious samples
    if label_col in df.columns:
        df_benign = df[df[label_col].isna()].copy()
        df_malicious = df[df[label_col].notna()].copy()
        logger.info(
            f"Found {len(df_benign)} benign + {len(df_malicious)} malicious samples"
        )
    else:
        logger.warning(
            f"Label column '{label_col}' not found! Processing all samples..."
        )
        df_benign = pd.DataFrame()
        df_malicious = df.copy()

    # Initial NaN statistics
    pfcp_cols_present = [col for col in PFCP_FEATURES if col in df_malicious.columns]
    initial_nan_count = df_malicious[pfcp_cols_present].isna().sum().sum()
    logger.info(f"Initial NaN count in malicious PFCP fields: {initial_nan_count}")

    # Fill NaNs for each malicious sample
    filled_samples = []
    total_filled = 0
    attack_type_stats = {}

    for idx, (row_idx, sample) in enumerate(df_malicious.iterrows()):
        filled_sample, nan_count, attack_type = fill_nan_pfcp_fields(
            sample, seed=seed + idx
        )
        filled_samples.append(filled_sample)
        total_filled += nan_count

        if attack_type not in attack_type_stats:
            attack_type_stats[attack_type] = {"samples": 0, "nans_filled": 0}
        attack_type_stats[attack_type]["samples"] += 1
        attack_type_stats[attack_type]["nans_filled"] += nan_count

        if (idx + 1) % 100 == 0:
            logger.info(f"Processed {idx + 1}/{len(df_malicious)} malicious samples...")

    df_malicious_filled = pd.DataFrame(filled_samples)

    # Reassemble benign originals with filled malicious samples
    df_final = pd.concat([df_benign, df_malicious_filled], axis=0, ignore_index=True)

    # Final statistics
    final_nan_count = df_malicious_filled[pfcp_cols_present].isna().sum().sum()
    logger.info(f"Final NaN count in malicious PFCP fields: {final_nan_count}")
    logger.info(f"Filled {total_filled} NaN values total")

    # Save the full reassembled dataset
    df_final.to_csv(output_path, sep=";", index=False)
    logger.info(f"Full dataset (benign + malicious filled) saved to: {output_path}")

    # Log statistics by attack type
    logger.info("\n" + "=" * 60)
    logger.info("FILLING STATISTICS BY ATTACK TYPE")
    logger.info("=" * 60)
    for attack_type, stats in sorted(attack_type_stats.items()):
        logger.info(
            f"{attack_type:20s} → {stats['samples']:4d} samples, {stats['nans_filled']:5d} NaNs filled"
        )

    return df_final


if __name__ == "__main__":
    INPUT_PATH = Path(__file__).parent.parent / "data/datasets/test_dataset.csv"
    OUTPUT_PATH = Path(__file__).parent.parent / "data/datasets/test_dataset_filled.csv"

    df_filled = fill_malicious_dataset_nans_and_reassemble(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        label_col="ip.opt.time_stamp",
        seed=42,
    )

    logger.info("\n" + "=" * 60)
    logger.info("FILLING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total samples with filled NaNs: {len(df_filled)}")

    # Show label distribution
    if "ip.opt.time_stamp" in df_filled.columns:
        label_dist = df_filled["ip.opt.time_stamp"].value_counts().sort_index()
        logger.info("\nLabel distribution:")
        for label, count in label_dist.items():
            attack_name = ATTACK_TYPE_MAP.get(float(label), "Unknown")
            logger.info(f"  Label {label} ({attack_name:20s}): {count} samples")

    logger.info(f"\nFinal dataset shape: {df_filled.shape}")

    # Check remaining NaNs
    pfcp_cols = [col for col in PFCP_FEATURES if col in df_filled.columns]
    remaining_nans = df_filled[pfcp_cols].isna().sum().sum()
    if remaining_nans == 0:
        logger.info("All PFCP NaN values successfully filled.")
    else:
        logger.warning(f"{remaining_nans} PFCP NaN values still remain")
