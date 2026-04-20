from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported extension: {path.suffix}")


def load_features(
    path: str | Path,
    target: str,
    raw_path: str | Path | None = None,
    min_duration_hours: float | None = None,
    max_duration_days: float | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load features and target.

    Two modes:

    - **single-file** (legacy / synthetic demo): the features parquet contains
      the target column directly. Pass `raw_path=None`.

    - **btoe two-file** (production): features parquet contains only feature
      columns; the target column lives in a separate raw issue/PR parquet
      aligned by row index. Pass `raw_path` pointing at it.

    When the duration bounds are supplied, both DataFrames are filtered with
    the same boolean mask before being returned, matching btoe's
    `_apply_duration_filter`.
    """
    features_path = Path(path)
    df = _read_table(features_path)

    if raw_path is None:
        if target not in df.columns:
            raise KeyError(
                f"target column '{target}' missing from {features_path} and no raw_path was provided"
            )
        y = df[target].astype(float)
        X = df.drop(columns=[target]).select_dtypes(include=[np.number])
    else:
        raw = _read_table(Path(raw_path))
        if target not in raw.columns:
            raise KeyError(f"target column '{target}' missing from {raw_path}")
        if len(raw) != len(df):
            raise ValueError(
                f"row-count mismatch: features have {len(df)} rows, raw has {len(raw)} rows; "
                "btoe alignment relies on identical ordering"
            )
        y = raw[target].astype(float).reset_index(drop=True)
        X = df.select_dtypes(include=[np.number]).reset_index(drop=True)

    if X.shape[1] == 0:
        raise ValueError("no numeric feature columns found")

    mask = pd.Series(True, index=range(len(y)))
    if min_duration_hours is not None:
        mask &= y >= float(min_duration_hours)
    if max_duration_days is not None:
        mask &= y <= float(max_duration_days) * 24.0
    if not mask.all():
        X = X.loc[mask.values].reset_index(drop=True)
        y = y.loc[mask.values].reset_index(drop=True)

    return X, y
