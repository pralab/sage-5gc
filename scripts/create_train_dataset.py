"""
Script to construct the training dataset by merging benign samples from
dataset1_cleaned.csv and dataset2_cleaned.csv into a dataset_training_benign.csv file.
"""

from pathlib import Path

import pandas as pd


def merge_benign_samples(
    dataset_source_path: str,
    dataset_target_path: str,
    output_path: str,
    label_col: str = "ip.opt.time_stamp",
) -> pd.DataFrame:
    """
    Extracts benign samples (label NaN) from a dataset and appends them to another
    benign dataset.

    Parameters
    ----------
    dataset_source_path : str
        Path of the dataset to extract benign samples from.
    dataset_target_path : str
        Path of the dataset that already contains only benign samples.
    output_path : str
        Path where to save the final dataset.
    label_col : str
        Name of the label column (default: ip.opt.time_stamp)

    Returns
    -------
    pd.DataFrame
        The merged final dataset.
    """
    print(f"Loading source dataset: {dataset_source_path}")
    df_source = pd.read_csv(dataset_source_path, sep=";", low_memory=False)

    print(f"Loading target dataset: {dataset_target_path}")
    df_target = pd.read_csv(dataset_target_path, sep=";", low_memory=False)

    # Extract benign samples from the source dataset
    print("Extracting benign samples (label NaN)...")
    benign_source = df_source[df_source[label_col].isna()].copy()

    print(f"Benign samples extracted: {len(benign_source)}")
    print(f"Benign samples in target: {len(df_target)}")

    # Concatenate
    print("Concatenating datasets...")
    df_final = pd.concat([df_target, benign_source], ignore_index=True)

    # Save
    print(f"Saving final dataset → {output_path}")
    df_final.to_csv(output_path, sep=";", index=False)

    print("Operation completed")

    return df_final


if __name__ == "__main__":
    merge_benign_samples(
        dataset_source_path=Path(__file__).parent.parent
        / "data/cleaned_datasets/dataset_2_cleaned.csv",
        dataset_target_path=Path(__file__).parent.parent
        / "data/cleaned_datasets/dataset_1_cleaned.csv",
        output_path=Path(__file__).parent.parent
        / "data/cleaned_datasets/dataset_training_benign.csv",
    )
