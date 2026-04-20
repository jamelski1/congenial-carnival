import numpy as np

from src.preprocessing.reducer import (
    ReducerConfig,
    build_feature_pipeline,
    inverse_transform_target,
    transform_target,
)


def test_pipeline_reduces_to_n_qubits_and_respects_range():
    cfg = ReducerConfig(n_qubits=4, scale_features=True)
    pipe = build_feature_pipeline(cfg)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 30))
    Xr = pipe.fit_transform(X)
    assert Xr.shape == (60, 4)
    assert Xr.min() >= 0.0 - 1e-9
    assert Xr.max() <= np.pi + 1e-9


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
