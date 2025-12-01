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
):
    """
    Estrae i campioni benigni (label NaN) da un dataset e li accorpa a un altro dataset benigno.

    Parameters
    ----------
    dataset_source_path : str
        Path del dataset da cui estrarre i benigni.
    dataset_target_path : str
        Path del dataset che contiene già solo benigni.
    output_path : str
        Path dove salvare il dataset finale.
    label_col : str
        Nome della colonna label (default: ip.opt.time_stamp)

    Returns
    -------
    pd.DataFrame
        Dataset finale unito.
    """
    print(f"Caricamento dataset sorgente: {dataset_source_path}")
    df_source = pd.read_csv(dataset_source_path, sep=";", low_memory=False)

    print(f"Caricamento dataset target: {dataset_target_path}")
    df_target = pd.read_csv(dataset_target_path, sep=";", low_memory=False)

    # Estrai benigni dal dataset sorgente
    print("Estrazione campioni benigni (label NaN)...")
    benign_source = df_source[df_source[label_col].isna()].copy()

    print(f"Campioni benigni estratti: {len(benign_source)}")
    print(f"Campioni benigni nel target: {len(df_target)}")

    # Accorpa
    print("Concatenazione dataset...")
    df_final = pd.concat([df_target, benign_source], ignore_index=True)

    # Salva
    print(f"Salvataggio dataset finale → {output_path}")
    df_final.to_csv(output_path, sep=";", index=False)

    print("Operazione completata ✓")

    return df_final


# =============================
# ESEMPIO DI UTILIZZO
# =============================
if __name__ == "__main__":
    merge_benign_samples(
        dataset_source_path=Path(__file__).parent.parent
        / "data/cleaned_datasets/dataset_2_cleaned.csv",
        dataset_target_path=Path(__file__).parent.parent
        / "data/cleaned_datasets/dataset_1_cleaned.csv",
        output_path=Path(__file__).parent.parent
        / "data/cleaned_datasets/dataset_training_benign.csv",
    )
