import numpy as np
import pytest

from src.preprocessing.selector import XGBTopKSelector


def _synth(n=200, d=20, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    # Make features 3, 7, 11 predictive; everything else noise.
    y = 1.5 * X[:, 3] - 0.7 * X[:, 7] ** 2 + 0.4 * X[:, 11]
    y += rng.normal(scale=0.1, size=n)
    return X, y


def test_selects_k_columns():
    X, y = _synth()
    sel = XGBTopKSelector(k=5).fit(X, y)
    Xt = sel.transform(X)
    assert Xt.shape == (X.shape[0], 5)


def test_picks_predictive_features_not_noise():
    X, y = _synth(n=400, d=20, seed=0)
    sel = XGBTopKSelector(k=5).fit(X, y)
    chosen = set(sel._idx.tolist())
    # The 3 truly predictive features should be in the top 5.
    assert {3, 7, 11}.issubset(chosen)


def test_indexes_are_sorted_for_stable_column_order():
    X, y = _synth()
    sel = XGBTopKSelector(k=5).fit(X, y)
    assert list(sel._idx) == sorted(sel._idx)


def test_k_too_large_raises():
    X, y = _synth(n=100, d=20, seed=1)
    with pytest.raises(ValueError, match="exceeds feature count"):
        XGBTopKSelector(k=30).fit(X, y)


def test_transform_before_fit_raises():
    X, _ = _synth()
    with pytest.raises(RuntimeError, match="fit must be called"):
        XGBTopKSelector(k=3).transform(X)
