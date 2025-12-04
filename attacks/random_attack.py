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

from datetime import datetime
import ipaddress
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

import joblib

from ml_models import Detector

# ---------------------------------------------------------------------
# Field specifications (type + range)
# ---------------------------------------------------------------------

FIELD_SPECS: Dict[str, Dict[str, Any]] = {
    # ------------- IP -------------
    "ip.checksum": {"type": "hex", "min": 0x36, "max": 0xFFF5},
    "ip.id": {"type": "hex", "min": 0x0, "max": 0xFFFE},
    "ip.len": {"type": "int", "min": 44, "max": 653},
    "ip.ttl": {"type": "int", "min": 2, "max": 200},
    # ------------- UDP -------------
    "udp.checksum": {"type": "hex", "min": 0x955, "max": 0xa118},
    "udp.length": {"type": "float_int", "min": 24.0, "max": 633.0},
    # ------------- PFCP booleans (flags/apply_action) -------------
    "pfcp.apply_action.buff": {"type": "bool_str"},
    "pfcp.apply_action.forw": {"type": "bool_str"},
    "pfcp.apply_action.nocp": {"type": "bool_str"},
    "pfcp.f_teid_flags.ch": {"type": "bool_str"},
    "pfcp.f_teid_flags.ch_id": {"type": "bool_str"},
    "pfcp.f_teid_flags.v6": {"type": "bool_str"},
    "pfcp.s": {"type": "bool_str"},
    # ------------- PFCP numeric identifiers -------------
    "pfcp.dst_interface": {"type": "float_int", "min": 0.0, "max": 1.0},
    "pfcp.duration_measurement": {
        "type": "float_int",
        "min": 1747212643.0,
        "max": 1753894838.0,
    },
    "pfcp.ie_len": {"type": "float_int", "min": 1.0, "max": 50.0},
    "pfcp.ie_type": {"type": "float_int", "min": 10.0, "max": 96.0},
    "pfcp.length": {"type": "float_int", "min": 12.0, "max": 621.0},
    "pfcp.msg_type": {"type": "float_int", "min": 1.0, "max": 57.0},
    "pfcp.pdr_id": {"type": "float_int", "min": 1.0, "max": 2.0},
    "pfcp.recovery_time_stamp": {
        "type": "timestamp",
        "min": 1747207882,
        "max": 1753892199,
    },
    "pfcp.response_time": {"type": "float", "min": 2.0095e-05, "max": 0.041239073},
    "pfcp.response_to": {"type": "float_int", "min": 1.0, "max": 2565.0},
    "pfcp.seid": {"type": "hex", "min": 0x00, "max": 0xFFF},
    "pfcp.seqno": {"type": "float_int", "min": 0.0, "max": 202364.0},
    # ------------- PFCP IP / TEID / SEID hex fields -------------
    "pfcp.f_seid.ipv4": {"type": "ipv4"},
    "pfcp.f_teid.ipv4_addr": {"type": "ipv4"},
    "pfcp.f_teid.teid": {"type": "hex", "min": 0x1D, "max": 0xFFE3},
    "pfcp.node_id_ipv4": {"type": "ipv4"},
    "pfcp.outer_hdr_creation.ipv4": {"type": "ipv4"},
    "pfcp.outer_hdr_creation.teid": {"type": "hex", "min": 0x1, "max": 0x18B6},
    "pfcp.ue_ip_addr_ipv4": {"type": "ipv4"},
    # ------------- PFCP strings / descriptors -------------
    "pfcp.flags": {"type": "hex", "min": 0x20, "max": 0x21},
    # ------------- PFCP volumes (counters) -------------
    "pfcp.volume_measurement.dlnop": {"type": "float_int", "min": 0.0, "max": 13195.0},
    "pfcp.volume_measurement.dlvol": {
        "type": "float_int",
        "min": 0.0,
        "max": 17834134.0,
    },
    "pfcp.volume_measurement.tonop": {"type": "float_int", "min": 0.0, "max": 13195.0},
    "pfcp.volume_measurement.tovol": {
        "type": "float_int",
        "min": 0.0,
        "max": 17834134.0,
    },
}

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
    # "pfcp.end_time",
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
    # "pfcp.time_of_first_packet",
    # "pfcp.time_of_last_packet",
    "pfcp.ue_ip_addr_ipv4",
    "pfcp.volume_measurement.dlnop",
    "pfcp.volume_measurement.dlvol",
    "pfcp.volume_measurement.tonop",
    "pfcp.volume_measurement.tovol",
}

# ---------------------------------------------------------------------
# Helper generators for basic semantic types
# ---------------------------------------------------------------------

def rand_bool() -> bool:
    """Random boolean as 'True'/'False'."""
    return True if random.random() < 0.5 else False

def rand_hex(min, max) -> str:
    """Random hex string within given integer range."""
    value = random.randint(min, max)
    hex_len = (max.bit_length() + 3) // 4  # Number of hex digits
    return f"0x{value:0{hex_len}X}"

def rand_int(min, max) -> int:
    """Random integer within given range."""
    return random.randint(min, max)

def rand_float(min, max) -> float:
    """Random float within given range."""
    return random.uniform(min, max)

def rand_float_int(min, max) -> float:
    """Random float that represents an integer within given range."""
    return float(random.randint(min, max))


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

    # if t == "ipv4":
    #     return rand_ipv4()

    if t == "bool_str":
        return rand_bool()

    if t == "hex":
        return rand_hex(spec["min"], spec["max"])

    if t == "int":
        return rand_int(spec["min"], spec["max"])

    if t == "float":
        return rand_float(spec["min"], spec["max"])

    if t == "float_int":
        # Many Wireshark exports represent ints as floats with .0
        min_v = spec.get("min", 0)
        max_v = spec.get("max", 10**9)
        return float(rand_int(min_v, max_v))

    if t == "timestamp":
        return datetime.fromtimestamp(
            rand_int(spec["min"], spec["max"])
        ).strftime("%Y-%m-%d %H:%M:%S")

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

    model_name = "INNE"
    detector: Detector = joblib.load(
        Path(__file__).parent.parent
        / f"data/trained_models/{model_name}.pkl"
    )

    print(f"Threshold: {detector._detector.threshold_}")

    results = {}

    for idx, sample in dataset.iterrows():
        y = detector.predict(pd.DataFrame([sample]), None)[0]
        init_score = detector.decision_function(pd.DataFrame([sample]))[0]
        print(f"Sample {idx} - Orig score: {init_score}")

        if y == 0:
            print(f"Sample {idx} is already misclassified. Skipping attack.\n")
            results[idx] = {
                "original_score": init_score,
                "attacked_score": None,
                "success": False,
            }
            continue

        adv_sample = random_attack(sample)
        adv_score = detector.decision_function(pd.DataFrame([adv_sample]))[0]

        if adv_score < detector._detector.threshold_:
            print(f"Sample {idx} - Adv score: {adv_score}")
            print(f"Sample {idx} successfully attacked!\n")
            results[idx] = {
                "original_score": init_score,
                "attacked_score": adv_score,
                "success": True,
            }
        else:
            print(f"Sample {idx} - Adv score: {adv_score}\n")
            results[idx] = {
                "original_score": init_score,
                "attacked_score": adv_score,
                "success": False,
            }

    with (Path(__file__).parent.parent / f"results/{model_name}.json").open("w") as f:
        json.dump(results, f, indent=4)
