#!/usr/bin/env python3
"""
Simple random feature-space attack for IP/TCP/UDP/PFCP fields.

- Reads a CSV exported from tshark (separator=';')
- For each row, replaces selected fields with random but semantically valid values
- Writes an attacked CSV

NOTE:
This preserves type/format semantics (IPv4, hex, bool, float, etc.),
but does NOT guarantee protocol-level validity (e.g., checksums).
"""

import ipaddress
from pathlib import Path
import random
import string
import sys
from typing import Any, Dict

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

import joblib

from ml_models import Detector

# ---------------------------------------------------------------------
# Original POSSIBLE_FEATURES (can be used to pick which columns to attack)
# ---------------------------------------------------------------------

"""
pfcp.duration_measurement;pfcp.end_time;pfcp.f_seid.ipv4;pfcp.f_teid.ipv4_addr;pfcp.f_teid.teid;pfcp.f_teid_flags.ch;pfcp.f_teid_flags.ch_id;pfcp.f_teid_flags.v6;pfcp.flags;pfcp.ie_len;pfcp.ie_type;pfcp.length;pfcp.msg_type;pfcp.node_id_ipv4;pfcp.outer_hdr_creation.ipv4;pfcp.outer_hdr_creation.teid;pfcp.pdr_id;pfcp.recovery_time_stamp;pfcp.response_time;pfcp.response_to;pfcp.s;pfcp.seid;pfcp.seqno;pfcp.time_of_first_packet;pfcp.time_of_last_packet;pfcp.ue_ip_addr_ipv4;pfcp.volume_measurement.dlnop;pfcp.volume_measurement.dlvol;pfcp.volume_measurement.tonop;pfcp.volume_measurement.tovol

"""


POSSIBLE_FEATURES = {
    # IP
    "ip.checksum",
    "ip.id",
    "ip.len",
    "ip.ttl",
    # UDP
    "udp.checksum",
    "udp.length",
    # PFCP
    "pfcp.apply_action.buff",
    "pfcp.apply_action.forw",
    "pfcp.apply_action.nocp",
    "pfcp.dst_interface",
    "pfcp.duration_measurement",
    "pfcp.end_time",
    "pfcp.f_seid.ipv4",
    "pfcp.f_teid.ipv4_addr",
    "pfcp.f_teid.teid",
    "pfcp.f_teid_flags.ch",
    "pfcp.f_teid_flags.ch_id",
    "pfcp.f_teid_flags.v6",
    "pfcp.flags",
    "pfcp.ie_len",
    "pfcp.ie_type",
    "pfcp.length",
    "pfcp.msg_type",
    "pfcp.node_id_ipv4",
    "pfcp.outer_hdr_creation.ipv4",
    "pfcp.outer_hdr_creation.teid",
    "pfcp.pdr_id",
    "pfcp.recovery_time_stamp",
    "pfcp.response_time",
    "pfcp.response_to",
    "pfcp.s",
    "pfcp.seid",
    "pfcp.seqno",
    "pfcp.time_of_first_packet",
    "pfcp.time_of_last_packet",
    "pfcp.ue_ip_addr_ipv4",
    "pfcp.volume_measurement.dlnop",
    "pfcp.volume_measurement.dlvol",
    "pfcp.volume_measurement.tonop",
    "pfcp.volume_measurement.tovol",
}

# ---------------------------------------------------------------------
# Helper generators for basic semantic types
# ---------------------------------------------------------------------


def rand_ipv4() -> str:
    """Generate a realistic private IPv4 address (e.g., 192.168.x.y)."""
    # Use RFC1918 spaces to stay "realistic"
    private_ranges = [
        ("10.0.0.0", "10.255.255.255"),
        ("172.16.0.0", "172.31.255.255"),
        ("192.168.0.0", "192.168.255.255"),
    ]
    start_str, end_str = random.choice(private_ranges)
    start = int(ipaddress.IPv4Address(start_str))
    end = int(ipaddress.IPv4Address(end_str))
    value = random.randint(start, end)
    return str(ipaddress.IPv4Address(value))


def rand_bool() -> bool:
    """Random boolean as 'True'/'False' (Wireshark-style exported)."""
    return True if random.random() < 0.5 else False


def rand_hex(bits: int = 16, prefix: str = "0x") -> str:
    """Random hex string with given number of bits."""
    max_val = 2**bits - 1
    val = random.randint(0, max_val)
    # width = bits / 4 hex digits
    width = bits // 4
    return f"{prefix}{val:0{width}x}"


def rand_int(min_val: int, max_val: int) -> int:
    return random.randint(min_val, max_val)


def rand_float(min_val: float, max_val: float) -> float:
    return random.uniform(min_val, max_val)


def rand_payload_hex(min_len_bytes: int = 8, max_len_bytes: int = 256) -> str:
    """Random payload encoded as hex string (no '0x' prefix)."""
    length = random.randint(min_len_bytes, max_len_bytes)
    return "".join(random.choice("0123456789abcdef") for _ in range(length * 2))


def rand_ascii_string(min_len: int = 5, max_len: int = 30) -> str:
    """Random printable ASCII string (e.g., for pfcp.flow_desc or network_instance)."""
    length = random.randint(min_len, max_len)
    chars = string.ascii_letters + string.digits + " .:/_-"
    return "".join(random.choice(chars) for _ in range(length))


def rand_timestamp_epoch() -> float:
    """
    Random realistic UNIX epoch time (Wireshark style float).
    E.g., between May 2024 and May 2025.
    """
    # 2024-05-01 and 2025-05-01 approximately
    min_epoch = 1714521600  # 2024-05-01
    max_epoch = 1746057600  # 2025-05-01
    return float(rand_int(min_epoch, max_epoch))


def rand_pfcp_response_time() -> float:
    """
    Random PFCP response time in seconds (Wireshark style).
    Use a small positive range, e.g., from 10µs to 1s.
    """
    return rand_float(1e-5, 1.0)


# ---------------------------------------------------------------------
# Field specifications (type + range)
# ---------------------------------------------------------------------

# For simplicity we encode semantics with a "type" tag and optional parameters.
FIELD_SPECS: Dict[str, Dict[str, Any]] = {
    # ------------- IP -------------
    "ip.checksum": {"type": "hex", "bits": 16},
    "ip.id": {"type": "hex", "bits": 16},
    "ip.len": {"type": "int", "min": 20, "max": 1500},
    "ip.ttl": {"type": "int", "min": 32, "max": 128},
    # ------------- UDP -------------
    "udp.checksum": {"type": "hex", "bits": 16},
    "udp.length": {"type": "int", "min": 8, "max": 1500},
    # ------------- PFCP booleans (flags/apply_action) -------------
    "pfcp.apply_action.buff": {"type": "bool_str"},
    "pfcp.apply_action.forw": {"type": "bool_str"},
    "pfcp.apply_action.nocp": {"type": "bool_str"},
    "pfcp.f_teid_flags.ch": {"type": "bool_str"},
    "pfcp.f_teid_flags.ch_id": {"type": "bool_str"},
    "pfcp.f_teid_flags.v6": {"type": "bool_str"},
    "pfcp.s": {"type": "bool_str"},
    # ------------- PFCP numeric identifiers -------------
    "pfcp.dst_interface": {"type": "float_int", "min": 0, "max": 64},
    "pfcp.duration_measurement": {"type": "float", "min": 0.0, "max": 10**6},
    "pfcp.ie_len": {"type": "float_int", "min": 1, "max": 1500},
    "pfcp.ie_type": {"type": "float_int", "min": 0, "max": 255},
    "pfcp.length": {"type": "float_int", "min": 0, "max": 1500},
    "pfcp.msg_type": {"type": "float_int", "min": 1, "max": 100},
    "pfcp.pdr_id": {"type": "float_int", "min": 1, "max": 65535},
    "pfcp.recovery_time_stamp": {"type": "float_int"},
    "pfcp.response_time": {"type": "pfcp_response_time"},
    "pfcp.response_to": {"type": "float_int"},
    "pfcp.seid": {"type": "float_int", "min": 0, "max": 2**32 - 1},
    "pfcp.seqno": {"type": "float_int", "min": 0, "max": 2**24 - 1},
    # ------------- PFCP IP / TEID / SEID hex fields -------------
    "pfcp.f_seid.ipv4": {"type": "ipv4"},
    "pfcp.f_teid.ipv4_addr": {"type": "ipv4"},
    "pfcp.f_teid.teid": {"type": "hex", "bits": 32},
    "pfcp.node_id_ipv4": {"type": "ipv4"},
    "pfcp.outer_hdr_creation.ipv4": {"type": "ipv4"},
    "pfcp.outer_hdr_creation.teid": {"type": "hex", "bits": 32},
    "pfcp.ue_ip_addr_ipv4": {"type": "ipv4"},
    # ------------- PFCP strings / descriptors -------------
    "pfcp.flags": {"type": "hex", "bits": 8},
    # ------------- PFCP timestamps -------------
    "pfcp.end_time": {"type": "timestamp_epoch"},
    "pfcp.time_of_first_packet": {"type": "timestamp_epoch"},
    "pfcp.time_of_last_packet": {"type": "timestamp_epoch"},
    # ------------- PFCP volumes (counters) -------------
    "pfcp.volume_measurement.dlnop": {"type": "float_int", "min": 0, "max": 10**9},
    "pfcp.volume_measurement.dlvol": {"type": "float_int", "min": 0, "max": 10**12},
    "pfcp.volume_measurement.tonop": {"type": "float_int", "min": 0, "max": 10**9},
    "pfcp.volume_measurement.tovol": {"type": "float_int", "min": 0, "max": 10**12},
}

# ---------------------------------------------------------------------
# Main random value dispatcher
# ---------------------------------------------------------------------


def generate_random_value(field: str) -> Any:
    """
    Given a field name, generate a random but semantically valid value.
    If the field is unknown or untyped, return the original value (None).
    """
    spec = FIELD_SPECS.get(field)
    if spec is None:
        # Unknown field: do not modify
        return None

    t = spec["type"]

    if t == "ipv4":
        return rand_ipv4()

    if t == "bool_str":
        return rand_bool()

    if t == "hex":
        bits = spec.get("bits", 16)
        return rand_hex(bits=bits)

    if t == "int":
        return rand_int(spec["min"], spec["max"])

    if t == "float":
        return rand_float(spec["min"], spec["max"])

    if t == "float_int":
        # Many Wireshark exports represent ints as floats with .0
        min_v = spec.get("min", 0)
        max_v = spec.get("max", 10**9)
        return float(rand_int(min_v, max_v))

    if t == "payload_hex":
        return rand_payload_hex(8, 256)

    if t == "payload_hex_small":
        return rand_payload_hex(4, 32)

    if t == "ascii":
        return rand_ascii_string(5, 40)

    if t == "ascii_imei":
        # Random IMEI-like numeric string (15 digits)
        return float("".join(random.choice(string.digits) for _ in range(15)))

    if t == "string_short":
        return rand_ascii_string(3, 15)

    if t == "str_enum":
        return random.choice(spec["values"])

    if t == "timestamp_epoch":
        return rand_timestamp_epoch()

    if t == "pfcp_response_time":
        return rand_pfcp_response_time()

    return None


# ---------------------------------------------------------------------
# Attack function: apply to a pandas DataFrame
# ---------------------------------------------------------------------


def random_attack(
    sample: pd.Series, fields_to_attack: Dict[str, list] = None, seed: int = None
) -> pd.Series:
    """
    Apply a random feature-space attack to the given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset (as loaded from tshark CSV).
    fields_to_attack : dict or None
        If None, uses POSSIBLE_FEATURES.
        Otherwise, dict with the same structure as POSSIBLE_FEATURES or a flat list.
    seed : int or None
        Optional seed for reproducibility.

    Returns
    -------
    attacked_df : pd.DataFrame
        Copy of df with attacked fields replaced by random values.
    """
    if seed is not None:
        random.seed(seed)

    adv_sample = sample.copy()

    for field in POSSIBLE_FEATURES:
        if field not in adv_sample.index:
            # Skip missing fields
            continue

        if FIELD_SPECS.get(field) is None:
            # Skip fields without a semantic spec
            continue

        value = generate_random_value(field)
        if value is not None:
            adv_sample[field] = value

    return adv_sample


if __name__ == "__main__":
    path = Path(__file__).parent.parent / "data/datasets/malicious_test_dataset.csv"
    dataset = pd.read_csv(path, sep=";", low_memory=False)
    dataset = dataset.drop(columns=["ip.opt.time_stamp"])

    detector: Detector = joblib.load(
        Path(__file__).parent.parent / "data/trained_models/IForest.pkl"
    )

    print(f"Threshold: {detector._detector.threshold_}")

    for idx, sample in dataset.iterrows():
        init_score = detector.decision_function(pd.DataFrame([sample]))
        print(f"Sample {idx} - Orig score: {init_score}")

        if init_score < detector._detector.threshold_:
            print(
                f"Sample {idx} is already misclassified. Skipping attack.\n"
            )
            continue

        adv_sample = random_attack(sample)
        adv_score = detector.decision_function(pd.DataFrame([adv_sample]))
        print(f"Sample {idx} - Adv score: {adv_score}\n")

        if adv_score < detector._detector.threshold_:
            print(f"Sample {idx} successfully attacked!")
