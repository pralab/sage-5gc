import json
import pandas as pd
from typing import Dict, Iterable, List, Optional


def safe_read_csv(path, **kwargs):
    """
    Safely load a CSV file by attempting UTF-8 first, then Latin-1.
    Supports both full DataFrame loading and chunked (TextFileReader) reading.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    **kwargs :
        Additional arguments passed to pandas.read_csv.

    Returns
    -------
    DataFrame or TextFileReader
        Loaded dataset or chunk iterator.
    """
    def load_and_preview(encoding):
        df = pd.read_csv(path, sep=';', low_memory=False, encoding=encoding, **kwargs)
        # If read_csv returns an iterator (when chunksize is used), consume first chunk to validate
        if isinstance(df, pd.DataFrame):
            return df
        else:
            first_chunk = next(df)  # preview first chunk
            return df  
    try:
        return load_and_preview("utf-8")
    except UnicodeDecodeError:
        return load_and_preview("latin-1")


def get_protocol_feature_mapping(
    df: pd.DataFrame,
    modifiable_features: Optional[Iterable[str]] = None,
    protocol_column: str = "ip.proto",
) -> Dict[str, List[str]]:
    """
    Build a mapping: protocol_value → list of features that exist for at least one row.

    Parameters
    ----------
    df : DataFrame
        Input dataframe containing protocol and feature columns.
    modifiable_features : Iterable[str] or None
        If provided, restricts the mapping to this subset of features.
    protocol_column : str
        Column containing the protocol identifier (e.g., ip.proto).

    Returns
    -------
    dict[str, list[str]]
        Keys are protocol codes (as strings), values are sorted lists of feature names.

    Raises
    ------
    ValueError
        If protocol_column is not found in the dataframe.
    """
    if protocol_column not in df.columns:
        raise ValueError(f"Column '{protocol_column}' not found in DataFrame")

    # Restrict feature set if requested
    if modifiable_features is not None:
        columns_to_check = [
            col for col in modifiable_features
            if col in df.columns and col != protocol_column
        ]
    else:
        columns_to_check = [col for col in df.columns if col != protocol_column]

    mapping: Dict[str, List[str]] = {}
    # Iterate over distinct protocol values
    for proto in sorted(df[protocol_column].dropna().unique()):
        proto = str(proto)
        sub_df = df[df[protocol_column] == int(proto)]
        # Keep only columns that have at least one non-NA value in this protocol subset
        mapping[proto] = sorted(
            [col for col in columns_to_check if sub_df[col].notna().any()]
        )

    return mapping


def save_protocol_mapping(mapping: Dict[str, List[str]], output_path: str):
    """
    Save protocol→feature mapping as JSON.

    Parameters
    ----------
    mapping : dict
        Mapping produced by get_protocol_feature_mapping().
    output_path : str
        Destination JSON path.

    Returns
    -------
    None
    """
    with open(output_path, "w") as f:
        json.dump(mapping, f, indent=2)


def main():
    # CONFIG SECTION
    DATASET_PATH = "preprocessing/cleaned_datasets/dataset_3_cleaned.csv"
    OUTPUT_PATH = "protocol_mapping.json"

    # If None → all features considered
    FEATURES_JSON = None
    # FEATURES_JSON = "modifiable_features.json"

    print(f"[INFO] Loading dataset: {DATASET_PATH}")
    df = safe_read_csv(DATASET_PATH)
    # Load the set of modifiable features if provided
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

