import numpy as np
import pytest

from src.preprocessing.reducer import (
    ReducerConfig,
    build_feature_pipeline,
    inverse_transform_target,
    transform_target,
)


def test_pca_pipeline_reduces_to_n_qubits_and_respects_range():
    cfg = ReducerConfig(n_qubits=4, mode="pca", scale_features=True)
    pipe = build_feature_pipeline(cfg)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 30))
    Xr = pipe.fit_transform(X)
    assert Xr.shape == (60, 4)
    assert Xr.min() >= 0.0 - 1e-9
    assert Xr.max() <= np.pi + 1e-9


def test_topk_pipeline_reduces_to_n_qubits_and_respects_range():
    cfg = ReducerConfig(n_qubits=4, mode="topk", scale_features=True)
    pipe = build_feature_pipeline(cfg)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 20))
    y = 2.0 * X[:, 5] + 0.5 * X[:, 11]  # only two features carry signal
    Xr = pipe.fit_transform(X, y)
    assert Xr.shape == (120, 4)
    assert Xr.min() >= 0.0 - 1e-9
    assert Xr.max() <= np.pi + 1e-9


def test_unknown_mode_raises():
    cfg = ReducerConfig(n_qubits=4, mode="bogus")
    with pytest.raises(ValueError, match="unknown reducer mode"):
        build_feature_pipeline(cfg)


def test_target_log_transform_roundtrips():
    cfg = ReducerConfig(n_qubits=4, log_transform_target=True)
    y = np.array([1.0, 5.0, 120.0])
    yt = transform_target(y, cfg)
    assert np.allclose(inverse_transform_target(yt, cfg), y)


def test_target_identity_when_disabled():
    cfg = ReducerConfig(n_qubits=4, log_transform_target=False)
    y = np.array([1.0, 5.0, 120.0])
    assert np.allclose(transform_target(y, cfg), y)
    assert np.allclose(inverse_transform_target(y, cfg), y)
