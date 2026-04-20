from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_features(path: str | Path, target: str) -> tuple[pd.DataFrame, pd.Series]:
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"unsupported extension: {path.suffix}")

    if target not in df.columns:
        raise KeyError(f"target column '{target}' missing from {path}")

    y = df[target].astype(float)
    X = df.drop(columns=[target]).select_dtypes(include=[np.number])
    if X.shape[1] == 0:
        raise ValueError("no numeric feature columns found after dropping target")
    return X, y
