import json
import pandas as pd
from typing import Dict, Iterable, List, Optional


# -----------------------------
# SAFE CSV LOADER
# -----------------------------
def safe_read_csv(path: str, sep: str = ";") -> pd.DataFrame:
    """
    Attempts to read a CSV with UTF-8, falls back to Latin-1.
    """
    try:
        return pd.read_csv(path, sep=sep, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, sep=sep, low_memory=False, encoding="latin-1")


# -----------------------------
# PROTOCOL → FEATURE MAPPING
# -----------------------------
def get_protocol_feature_mapping(
    df: pd.DataFrame,
    modifiable_features: Optional[Iterable[str]] = None,
    protocol_column: str = "ip.proto",
) -> Dict[str, List[str]]:
    """
    Mapping:
        ip.proto value  →  list of feature names that appear at least once.
    """
    if protocol_column not in df.columns:
        raise ValueError(f"Column '{protocol_column}' not found in DataFrame")

    # restrict if needed
    if modifiable_features is not None:
        columns_to_check = [
            col for col in modifiable_features
            if col in df.columns and col != protocol_column
        ]
    else:
        columns_to_check = [col for col in df.columns if col != protocol_column]

    mapping: Dict[str, List[str]] = {}

    for proto in sorted(df[protocol_column].dropna().unique()):
        proto = str(proto)
        sub_df = df[df[protocol_column] == int(proto)]
        mapping[proto] = sorted(
            [col for col in columns_to_check if sub_df[col].notna().any()]
        )

    return mapping


# -----------------------------
# SAVE MAPPING TO FILE
# -----------------------------
def save_protocol_mapping(mapping: Dict[str, List[str]], output_path: str):
    with open(output_path, "w") as f:
        json.dump(mapping, f, indent=2)


# -----------------------------
# MAIN (NO argparse)
# -----------------------------
def main():
    """
    Configurable entry point.
    Modify the variables below and run:
        python protocol_feature_mapping.py
    """
    # === CONFIG SECTION ===
    DATASET_PATH = "preprocessing/cleaned_datasets/dataset_3_cleaned.csv"
    OUTPUT_PATH = "protocol_mapping.json"

    # If None → all features considered
    FEATURES_JSON = None
    # FEATURES_JSON = "modifiable_features.json"
    # ======================

    print(f"[INFO] Loading dataset: {DATASET_PATH}")
    df = safe_read_csv(DATASET_PATH)

    if FEATURES_JSON:
        print(f"[INFO] Loading feature list: {FEATURES_JSON}")
        with open(FEATURES_JSON, "r") as f:
            feature_list = json.load(f)
    else:
        feature_list = None

    print("[INFO] Computing protocol mapping...")
    mapping = get_protocol_feature_mapping(df, feature_list)

    print(f"[INFO] Saving mapping to: {OUTPUT_PATH}")
    save_protocol_mapping(mapping, OUTPUT_PATH)

    print("\nProtocol → Feature mapping:")
    print(json.dumps(mapping, indent=2))
    print("\n[OK] Done.")


if __name__ == "__main__":
    main()
