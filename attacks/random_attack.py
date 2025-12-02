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

POSSIBLE_FEATURES = {
    "IP": [
        "ip.checksum",
        "ip.dsfield",
        "ip.dsfield.dscp",
        "ip.flags",
        "ip.flags.df",
        "ip.hdr_len",
        "ip.id",
        "ip.len",
        "ip.ttl",
        "ip.src",
        # "ip.dst",
        # "ip.dst_host",
        # "ip.proto",
        # "ip.host",
        # "ip.addr",
        # "ip.src_host",
    ],
    "TCP": [
        "tcp.ack",
        "tcp.ack_raw",
        "tcp.analysis.bytes_in_flight",
        "tcp.analysis.push_bytes_sent",
        "tcp.checksum",
        "tcp.completeness",
        "tcp.completeness.ack",
        "tcp.completeness.data",
        "tcp.completeness.fin",
        "tcp.completeness.str",
        "tcp.completeness.syn-ack",
        #"tcp.dstport",
        "tcp.flags",
        "tcp.flags.ack",
        "tcp.flags.fin",
        "tcp.flags.push",
        "tcp.flags.reset",
        "tcp.flags.str",
        "tcp.flags.syn",
        "tcp.hdr_len",
        "tcp.len",
        "tcp.nxtseq",
        "tcp.option_kind",
        "tcp.option_len",
        "tcp.options",
        "tcp.options.timestamp.tsecr",
        "tcp.options.timestamp.tsval",
        "tcp.payload",
        "tcp.port",
        "tcp.seq",
        "tcp.seq_raw",
        "tcp.srcport",
        "tcp.stream",
        "tcp.time_delta",
        "tcp.time_relative",
        "tcp.window_size",
        "tcp.window_size_value",
    ],
    "UDP": [
        "udp.checksum",
        "udp.length",
        "udp.payload",
        "udp.port",
        "udp.srcport",
        #"udp.dstport",
        # "udp.stream",
        # "udp.time_delta",
        # "udp.time_relative",
    ],
    "PFCP": [
        "pfcp.apply_action.buff",
        "pfcp.apply_action.drop",
        "pfcp.apply_action.forw",
        "pfcp.apply_action.nocp",
        "pfcp.cause",
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
        "pfcp.flow_desc",
        "pfcp.flow_desc_len",
        "pfcp.ie_len",
        "pfcp.ie_type",
        "pfcp.length",
        "pfcp.msg_type",
        "pfcp.network_instance",
        "pfcp.node_id_ipv4",
        "pfcp.node_id_type",
        "pfcp.outer_hdr_creation.ipv4",
        "pfcp.outer_hdr_creation.teid",
        "pfcp.pdn_type",
        "pfcp.pdr_id",
        "pfcp.precedence",
        "pfcp.recovery_time_stamp",
        "pfcp.response_time",
        "pfcp.response_to",
        "pfcp.s",
        "pfcp.seid",
        "pfcp.seqno",
        "pfcp.source_interface",
        "pfcp.time_of_first_packet",
        "pfcp.time_of_last_packet",
        "pfcp.ue_ip_addr_ipv4",
        "pfcp.ue_ip_address_flag.sd",
        "pfcp.user_id.imei",
        "pfcp.volume_measurement.dlnop",
        "pfcp.volume_measurement.dlvol",
        "pfcp.volume_measurement.tonop",
        "pfcp.volume_measurement.tovol",
    ],
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
    "ip.addr": {"type": "ipv4"},
    "ip.src": {"type": "ipv4"},
    "ip.dst": {"type": "ipv4"},
    "ip.src_host": {"type": "ipv4"},
    "ip.dst_host": {"type": "ipv4"},
    "ip.host": {"type": "ipv4"},
    "ip.checksum": {"type": "hex", "bits": 16},
    "ip.dsfield": {"type": "hex", "bits": 8},
    "ip.dsfield.dscp": {"type": "int", "min": 0, "max": 63},
    "ip.flags": {"type": "hex", "bits": 8},  # e.g. 0x02
    "ip.flags.df": {"type": "bool_str"},
    "ip.hdr_len": {"type": "int", "min": 20, "max": 60},  # bytes
    "ip.id": {"type": "hex", "bits": 16},
    "ip.len": {"type": "int", "min": 20, "max": 1500},
    "ip.proto": {"type": "int", "min": 1, "max": 255},  # protocol number
    "ip.ttl": {"type": "int", "min": 32, "max": 128},
    # ------------- TCP -------------
    "tcp.ack": {"type": "int", "min": 0, "max": 2**32 - 1},
    "tcp.ack_raw": {"type": "float_int"},  # exported as float with .0
    "tcp.analysis.bytes_in_flight": {"type": "float", "min": 0.0, "max": 10**6},
    "tcp.analysis.push_bytes_sent": {"type": "float", "min": 0.0, "max": 10**5},
    "tcp.checksum": {"type": "hex", "bits": 16},
    "tcp.completeness": {"type": "str_enum", "values": ["", "1.0"]},
    "tcp.completeness.ack": {"type": "bool_str"},
    "tcp.completeness.data": {"type": "bool_str"},
    "tcp.completeness.fin": {"type": "bool_str"},
    "tcp.completeness.str": {"type": "string_short"},
    "tcp.completeness.syn-ack": {"type": "bool_str"},
    "tcp.dstport": {"type": "int", "min": 1, "max": 65535},
    "tcp.srcport": {"type": "int", "min": 1024, "max": 65535},
    "tcp.port": {"type": "int", "min": 1, "max": 65535},
    "tcp.flags": {"type": "hex", "bits": 8},  # e.g. 0x12
    "tcp.flags.ack": {"type": "bool_str"},
    "tcp.flags.fin": {"type": "bool_str"},
    "tcp.flags.push": {"type": "bool_str"},
    "tcp.flags.reset": {"type": "bool_str"},
    "tcp.flags.syn": {"type": "bool_str"},
    "tcp.flags.str": {"type": "string_short"},
    "tcp.hdr_len": {"type": "int", "min": 20, "max": 60},
    "tcp.len": {"type": "int", "min": 0, "max": 1460},
    "tcp.nxtseq": {"type": "float_int"},
    "tcp.option_kind": {"type": "int", "min": 0, "max": 255},
    "tcp.option_len": {"type": "float_int"},
    "tcp.options": {"type": "payload_hex_small"},
    "tcp.options.timestamp.tsecr": {"type": "float_int"},
    "tcp.options.timestamp.tsval": {"type": "float_int"},
    "tcp.payload": {"type": "payload_hex"},
    "tcp.seq": {"type": "int", "min": 0, "max": 2**32 - 1},
    "tcp.seq_raw": {"type": "float_int"},
    "tcp.stream": {"type": "float_int"},
    "tcp.time_delta": {"type": "float", "min": 0.0, "max": 1.0},
    "tcp.time_relative": {"type": "float", "min": 0.0, "max": 5.0},
    "tcp.window_size": {"type": "float_int"},
    "tcp.window_size_value": {"type": "float_int"},
    # ------------- UDP -------------
    "udp.checksum": {"type": "hex", "bits": 16},
    "udp.dstport": {"type": "int", "min": 1, "max": 65535},
    "udp.srcport": {"type": "int", "min": 1024, "max": 65535},
    "udp.port": {"type": "int", "min": 1, "max": 65535},
    "udp.length": {"type": "int", "min": 8, "max": 1500},
    "udp.stream": {"type": "float_int"},
    "udp.time_delta": {"type": "float", "min": 0.0, "max": 1.0},
    "udp.time_relative": {"type": "float", "min": 0.0, "max": 5.0},
    "udp.payload": {"type": "payload_hex"},
    # ------------- PFCP booleans (flags/apply_action) -------------
    "pfcp.apply_action.buff": {"type": "bool_str"},
    "pfcp.apply_action.drop": {"type": "bool_str"},
    "pfcp.apply_action.forw": {"type": "bool_str"},
    "pfcp.apply_action.nocp": {"type": "bool_str"},
    "pfcp.f_teid_flags.ch": {"type": "bool_str"},
    "pfcp.f_teid_flags.ch_id": {"type": "bool_str"},
    "pfcp.f_teid_flags.v6": {"type": "bool_str"},
    "pfcp.s": {"type": "bool_str"},
    "pfcp.ue_ip_address_flag.sd": {"type": "bool_str"},
    # ------------- PFCP numeric identifiers -------------
    "pfcp.cause": {"type": "float_int", "min": 0, "max": 64},
    "pfcp.dst_interface": {"type": "float_int", "min": 0, "max": 64},
    "pfcp.duration_measurement": {"type": "float", "min": 0.0, "max": 10**6},
    "pfcp.ie_len": {"type": "float_int", "min": 1, "max": 1500},
    "pfcp.ie_type": {"type": "float_int", "min": 0, "max": 255},
    "pfcp.length": {"type": "float_int", "min": 0, "max": 1500},
    "pfcp.msg_type": {"type": "float_int", "min": 1, "max": 100},
    "pfcp.node_id_type": {"type": "float_int", "min": 0, "max": 3},
    "pfcp.pdn_type": {"type": "float_int", "min": 0, "max": 3},
    "pfcp.pdr_id": {"type": "float_int", "min": 1, "max": 65535},
    "pfcp.precedence": {"type": "float_int", "min": 0, "max": 2**32 - 1},
    "pfcp.recovery_time_stamp": {"type": "float_int"},
    "pfcp.response_time": {"type": "pfcp_response_time"},
    "pfcp.response_to": {"type": "float_int"},
    "pfcp.seid": {"type": "float_int", "min": 0, "max": 2**32 - 1},
    "pfcp.seqno": {"type": "float_int", "min": 0, "max": 2**24 - 1},
    # ------------- PFCP IP / TEID / SEID hex fields -------------
    "pfcp.f_seid.ipv4": {"type": "hex", "bits": 64},
    "pfcp.f_teid.ipv4_addr": {"type": "ipv4"},
    "pfcp.f_teid.teid": {"type": "hex", "bits": 32},
    "pfcp.node_id_ipv4": {"type": "ipv4"},
    "pfcp.outer_hdr_creation.ipv4": {"type": "ipv4"},
    "pfcp.outer_hdr_creation.teid": {"type": "hex", "bits": 32},
    "pfcp.ue_ip_addr_ipv4": {"type": "ipv4"},
    # ------------- PFCP strings / descriptors -------------
    "pfcp.flags": {"type": "hex", "bits": 8},
    "pfcp.flow_desc": {"type": "ascii"},
    "pfcp.flow_desc_len": {"type": "float_int", "min": 0, "max": 1024},
    "pfcp.network_instance": {"type": "ascii"},
    "pfcp.user_id.imei": {"type": "ascii_imei"},
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

    if adv_sample["ip.proto"] == 6:  # TCP
        protocol_fields = (
            #POSSIBLE_FEATURES["TCP"]
            POSSIBLE_FEATURES["IP"]
            #+ POSSIBLE_FEATURES["PFCP"]
        )
    elif adv_sample["ip.proto"] == 17:  # UDP
        protocol_fields = (
            #POSSIBLE_FEATURES["UDP"]
            POSSIBLE_FEATURES["IP"]
            #POSSIBLE_FEATURES["PFCP"]
        )
    else:
        return adv_sample

    for field in protocol_fields:
        if field not in adv_sample.index:
            # Skip missing fields
            continue

        if FIELD_SPECS.get(field) is None:
            # Skip fields without a semantic spec
            continue

        value = generate_random_value(field)
        if value is not None:
            adv_sample[field] = value

    # Simple consistency fix: ip.addr, ip.src, ip.dst all exist
    # (example: set ip.addr = ip.src to keep something coherent)
    if "ip.addr" in adv_sample.index and "ip.src" in adv_sample.index:
        adv_sample["ip.addr"] = adv_sample["ip.src"]
        adv_sample["ip.src_host"] = adv_sample["ip.src"]
        adv_sample["ip.host"] = adv_sample["ip.src"]

    if "tcp.port" in adv_sample.index and "tcp.srcport" in adv_sample.index:
        adv_sample["tcp.port"] = adv_sample['tcp.srcport']

    if "udp.port" in adv_sample.index and "udp.srcport" in adv_sample.index:
        adv_sample["udp.port"] = adv_sample['udp.srcport']

    return adv_sample


if __name__ == "__main__":
    path = Path(__file__).parent.parent / "data/cleaned_datasets/dataset_3_cleaned.csv"
    dataset = pd.read_csv(path, sep=";", low_memory=False)

    detector: Detector = joblib.load(
        Path(__file__).parent.parent / "data/trained_models/IForest.pkl"
    )

    print(f"Threshold: {detector.detector.threshold_}")

    for idx, sample in dataset.iterrows():
        # if pd.isna(sample["ip.opt.time_stamp"]):
        #     continue

        print(f"Sample {idx} - Orig score: {detector.decision_function(dataset, idx)}")

        adv_sample = random_attack(sample=dataset.loc[idx])
        dataset.loc[idx] = adv_sample

        print(f"Sample {idx} - Adv score: {detector.decision_function(dataset, idx)}")
