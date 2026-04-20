from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.loader import load_features


def test_single_file_mode_with_target_column(tmp_path: Path):
    df = pd.DataFrame({"emb_0": [0.1, 0.2, 0.3], "emb_1": [1.0, 2.0, 3.0], "duration_hours": [4.0, 8.0, 12.0]})
    p = tmp_path / "features.parquet"
    df.to_parquet(p, index=False)

    X, y = load_features(p, target="duration_hours")
    assert X.shape == (3, 2)
    assert list(X.columns) == ["emb_0", "emb_1"]
    assert y.tolist() == [4.0, 8.0, 12.0]


def test_two_file_mode_aligns_by_row_index(tmp_path: Path):
    feat = pd.DataFrame({"emb_0": [0.1, 0.2, 0.3], "emb_1": [1.0, 2.0, 3.0]})
    raw = pd.DataFrame({"repo": ["a", "b", "c"], "duration_hours": [5.0, 10.0, 15.0]})
    fp = tmp_path / "nlp_features.parquet"
    rp = tmp_path / "issue_pr_pairs.parquet"
    feat.to_parquet(fp, index=False)
    raw.to_parquet(rp, index=False)

    X, y = load_features(fp, target="duration_hours", raw_path=rp)
    assert X.shape == (3, 2)
    assert y.tolist() == [5.0, 10.0, 15.0]


def test_two_file_mode_rejects_mismatched_lengths(tmp_path: Path):
    feat = pd.DataFrame({"emb_0": [0.1, 0.2, 0.3]})
    raw = pd.DataFrame({"duration_hours": [5.0, 10.0]})
    fp = tmp_path / "f.parquet"
    rp = tmp_path / "r.parquet"
    feat.to_parquet(fp, index=False)
    raw.to_parquet(rp, index=False)

    with pytest.raises(ValueError, match="row-count mismatch"):
        load_features(fp, target="duration_hours", raw_path=rp)


def test_duration_filter_drops_rows_outside_bounds(tmp_path: Path):
    feat = pd.DataFrame({"emb_0": [10.0, 20.0, 30.0, 40.0]})
    raw = pd.DataFrame({"duration_hours": [0.5, 5.0, 100.0, 5000.0]})  # h
    fp = tmp_path / "f.parquet"
    rp = tmp_path / "r.parquet"
    feat.to_parquet(fp, index=False)
    raw.to_parquet(rp, index=False)

    # 1h <= y <= 90 days (=2160h) keeps rows index 1 and 2 only
    X, y = load_features(fp, target="duration_hours", raw_path=rp,
                         min_duration_hours=1.0, max_duration_days=90.0)
    assert X.shape == (2, 1)
    assert y.tolist() == [5.0, 100.0]
    assert X["emb_0"].tolist() == [20.0, 30.0]


def test_missing_target_in_raw_raises(tmp_path: Path):
    feat = pd.DataFrame({"emb_0": [0.1]})
    raw = pd.DataFrame({"not_target": [1.0]})
    fp = tmp_path / "f.parquet"
    rp = tmp_path / "r.parquet"
    feat.to_parquet(fp, index=False)
    raw.to_parquet(rp, index=False)

    with pytest.raises(KeyError, match="duration_hours"):
        load_features(fp, target="duration_hours", raw_path=rp)
