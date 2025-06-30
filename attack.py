import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def add_noise(
    df: pd.DataFrame,
    noise_level: float = 0.01,
    cols: list[str] = None,
    distribution: str = "normal",
) -> pd.DataFrame:
    """
    Add random noise to numeric columns of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    noise_level : float or dict
        If float, standard deviation of noise as a fraction of each column’s std.
        If dict, mapping column name → noise_level for that column.
    cols : list[str], optional
        Subset of columns to perturb. Defaults to all numeric columns.
    distribution : {'normal','uniform'}
        Type of noise distribution.

    Returns
    -------
    pd.DataFrame
        New DataFrame with noise applied.
    """
    df = df.copy()

    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in cols:
        if not is_numeric_dtype(df[col]):
            print(f"Skipping non-numeric column {col!r}")
            continue

        std = df[col].std(ddof=0)
        scale = std * noise_level

        if distribution == "normal":
            noise = np.random.normal(loc=0, scale=scale, size=len(df))
        elif distribution == "uniform":
            noise = np.random.uniform(low=-scale, high=+scale, size=len(df))
        else:
            raise ValueError("distribution must be 'normal' or 'uniform'")

        df[col] = df[col] + noise

    return df
