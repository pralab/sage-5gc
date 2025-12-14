"""Simple random feature-space attack for network traffic classifiers."""

import argparse
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

ATTACK_TYPE_MAP = {
    0: "flooding",  # PFCP Flooding
    1: "session_deletion",  # PFCP Deletion
    2: "session_modification",  # PFCP Modification
    5: "upf_pdn0_fault",  # UPF PDN-0 Fault
    6: "restoration_teid",  # PFCP Restoration-TEID
}

ATTACK_FEATURES = {
    "flooding": ["pfcp.msg_type"],
    "session_deletion": [
        "pfcp.msg_type",
    ],
    "session_modification": [
        "pfcp.msg_type",
        "pfcp.seid",
    ],
    "upf_pdn0_fault": [
        "pfcp.node_id_ipv4",
        "pfcp.pdr_id",
        "pfcp.f_teid_flags.ch",
        "pfcp.f_teid_flags.ch_id",
        "pfcp.f_teid_flags.v6",
    ],
    "restoration_teid": [
        "pfcp.f_teid.teid",
        "pfcp.pdr_id",
    ],
}

FEAT_MAPPING: Dict[str, Dict[str, Any]] = {
    # ------------- IP -------------
    "ip.ttl": {"type": "int", "min": 2, "max": 200},
    "ip.id": {"type": "hex", "min": 0x0, "max": 0xFFFE},
    "ip.len": {"type": "int", "min": 44, "max": 653},
    "ip.checksum": {"type": "hex", "min": 0x36, "max": 0xFFF5},
    # ------------- UDP -------------
    "udp.checksum": {"type": "hex", "min": 0x955, "max": 0xA118},
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
    "pfcp.ie_type": {"type": "float_int", "min": 10.0, "max": 96.0},
    "pfcp.msg_type": {"type": "float_int", "min": 1.0, "max": 57.0},
    "pfcp.pdr_id": {"type": "float_int", "min": 1.0, "max": 2.0},
    "pfcp.response_time": {"type": "float", "min": 2.0095e-05, "max": 0.041239073},
    "pfcp.response_to": {"type": "float_int", "min": 1.0, "max": 2565.0},
    "pfcp.seid": {"type": "hex", "min": 0x00, "max": 0xFFF},
    "pfcp.seqno": {"type": "float_int", "min": 0.0, "max": 202364.0},
    "pfcp.f_seid.ipv4": {"type": "ipv4"},
    "pfcp.f_teid.ipv4_addr": {"type": "ipv4"},
    "pfcp.f_teid.teid": {"type": "hex", "min": 0x1D, "max": 0xFFE3},
    "pfcp.node_id_ipv4": {"type": "ipv4"},
    "pfcp.outer_hdr_creation.ipv4": {"type": "ipv4"},
    "pfcp.outer_hdr_creation.teid": {"type": "hex", "min": 0x1, "max": 0x18B6},
    "pfcp.ue_ip_addr_ipv4": {"type": "ipv4"},
    "pfcp.flags": {"type": "hex", "min": 0x20, "max": 0x21},
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


def rand_ipv4() -> str:
    """Random IPv4 address."""
    while True:
        ip_int = random.randint(1, 0xFFFFFFFF - 1)
        ip_addr = ipaddress.IPv4Address(ip_int)
        if not (
            ip_addr.is_multicast
            or ip_addr.is_reserved
            or ip_addr.is_loopback
            or ip_addr.is_unspecified
            or ip_addr.is_link_local
        ):
            return str(ip_addr)


def generate_random_value(mapping: dict) -> Any:
    """Generate a random value based on the feature mapping."""
    field_type = mapping["type"]

    if field_type == "ipv4":
        return rand_ipv4()

    if field_type == "bool_str":
        return rand_bool()

    if field_type == "hex":
        return rand_hex(mapping["min"], mapping["max"])

    if field_type == "int":
        return rand_int(mapping["min"], mapping["max"])

    if field_type == "float":
        return rand_float(mapping["min"], mapping["max"])

    if field_type == "float_int":
        return float(rand_int(mapping["min"], mapping["max"]))

    return None


def random_attack(sample: pd.Series, attack_type: int, seed: int = 42) -> pd.Series:
    """
    Apply a random feature-space attack to the given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Sample to perturb.
    attack_type : int
        Attack type of the sample.
    seed : int
        Seed for reproducibility. Default is 42.

    Returns
    -------
    pd.DataFrame
        Copy of df with attacked fields replaced by random values.
    """
    if seed is not None:
        random.seed(seed)

    adv_sample = sample.copy()

    for field, mapping in FEAT_MAPPING.items():
        # Skip features that should not be modified to preserve the attack intent
        if field in ATTACK_FEATURES[ATTACK_TYPE_MAP[attack_type]]:
            continue

        value = generate_random_value(mapping)
        if value is not None:
            adv_sample[field] = value

    return adv_sample


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Random feature-space attack for network traffic classifiers."
    )
    argparser.add_argument(
        "--model-name",
        type=str,
        required=True,
        choices=[
            "ABOD",
            "CBLOF",
            "COPOD",
            "ECOD",
            "FeatureBaggingHBOS",
            "GMM",
            "HBOS",
            "IForest",
            "INNE",
            "KNN",
            "LODA",
            "LOF",
            "PCA",
        ],
        help="Name of the trained model to attack.",
    )
    argparser.add_argument(
        "--ds-path",
        type=str,
        default=None,
        help="The path to the attacks dataset file",
        required=True,
    )
    args = argparser.parse_args()

    dataset = pd.read_csv(args.ds_path, sep=";", low_memory=False)
    labels = dataset["ip.opt.time_stamp"].copy()
    dataset = dataset.drop(columns=["ip.opt.time_stamp"])

    detector: Detector = joblib.load(
        Path(__file__).parent.parent / f"data/trained_models/{args.model_name}.pkl"
    )

    results = {}
    for idx, sample in dataset.iterrows():
        y_pred = detector.predict(pd.DataFrame([sample]))[0]
        init_score = detector.decision_function(pd.DataFrame([sample]))[0]

        print(f"Sample {idx} - Orig score: {init_score}")

        if y_pred == 0:
            print(f"Sample {idx} is already misclassified. Skipping attack.\n")
            results[idx] = {
                "original_score": init_score,
                "attack_type": int(labels.loc[idx]),
                "attacked_score": None,
                "success": False,
            }
            continue

        adv_sample = random_attack(sample, int(labels.loc[idx]))
        adv_score = detector.decision_function(pd.DataFrame([adv_sample]))[0]

        if adv_score < detector._detector.threshold_:
            print(f"Sample {idx} - Adv score: {adv_score}")
            print(f"Sample {idx} successfully attacked!\n")
            results[idx] = {
                "original_score": init_score,
                "attacked_score": adv_score,
                "attack_type": int(labels.loc[idx]),
                "success": True,
            }
        else:
            print(f"Sample {idx} - Adv score: {adv_score}\n")
            results[idx] = {
                "original_score": init_score,
                "attacked_score": adv_score,
                "attack_type": int(labels.loc[idx]),
                "success": False,
            }

    with (Path(__file__).parent.parent / f"results/random_attack/{args.model_name}.json").open(
        "w"
    ) as f:
        json.dump(results, f, indent=4)
