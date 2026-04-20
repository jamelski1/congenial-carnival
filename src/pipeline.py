from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from src.classical.baseline import XGBBaseline, XGBConfig
from src.data.loader import load_features
from src.evaluation.metrics import evaluate
from src.evaluation.stats import cliffs_delta, wilcoxon_vs_baseline
from src.preprocessing.reducer import (
    ReducerConfig,
    build_feature_pipeline,
    inverse_transform_target,
    transform_target,
)

# Quantum dependencies (qiskit, qiskit-machine-learning) are heavy and only
# needed for the quantum models. Import them lazily inside `run()` so the
# pipeline can execute in classical-only mode without qiskit installed.


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def run(
    cfg: dict[str, Any],
    backend: str,
    models: list[str],
    output_dir: Path,
    X_override: np.ndarray | None = None,
    y_override: np.ndarray | None = None,
    subsample: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    if X_override is not None and y_override is not None:
        X_raw, y_raw = X_override, y_override
    else:
        data_cfg = cfg["data"]
        X_df, y_s = load_features(
            path=data_cfg["path"],
            target=data_cfg["target"],
            raw_path=data_cfg.get("raw_path"),
            min_duration_hours=data_cfg.get("min_duration_hours"),
            max_duration_days=data_cfg.get("max_duration_days"),
        )
        X_raw, y_raw = X_df.to_numpy(dtype=float), y_s.to_numpy(dtype=float)
        print(f"[data] loaded {X_raw.shape[0]} rows x {X_raw.shape[1]} features")

    if subsample is not None and subsample < len(X_raw):
        rng = np.random.default_rng(int(cfg["split"]["random_seed"]))
        idx = rng.choice(len(X_raw), size=subsample, replace=False)
        X_raw, y_raw = X_raw[idx], y_raw[idx]
        print(f"[data] subsampled to {len(X_raw)} rows (seed={cfg['split']['random_seed']})")

    red_cfg = ReducerConfig(
        n_qubits=int(cfg["preprocessing"]["n_qubits"]),
        k_features=cfg["preprocessing"].get("k_features"),
        mode=str(cfg["preprocessing"].get("mode", "pca")),
        scale_features=bool(cfg["preprocessing"]["scale_features"]),
        log_transform_target=bool(cfg["preprocessing"]["log_transform_target"]),
        random_seed=int(cfg["split"]["random_seed"]),
    )

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_raw,
        y_raw,
        test_size=float(cfg["split"]["test_size"]),
        random_state=int(cfg["split"]["random_seed"]),
    )

    reducer = build_feature_pipeline(red_cfg)
    # topk mode needs y at fit time; pca mode ignores it. Pass log-transformed
    # target so feature importance is ranked against the same target the
    # downstream regressors will optimise for.
    y_train_for_reducer = transform_target(y_train_raw, red_cfg)
    X_train = reducer.fit_transform(X_train_raw, y_train_for_reducer)
    X_test = reducer.transform(X_test_raw)

    y_train = transform_target(y_train_raw, red_cfg)
    y_test = transform_target(y_test_raw, red_cfg)

    handles = None
    if any(m in models for m in ("vqr", "qkrr")):
        from src.quantum.backends import get_backend  # lazy

        shots = int(cfg["quantum"]["qkrr"].get("shots", 1024))
        handles = get_backend(backend, shots=shots)
        print(f"[backend] {handles.name}")

    fitted: dict[str, Any] = {}
    preds_real: dict[str, np.ndarray] = {}

    if "xgb" in models:
        xgb_cfg = XGBConfig(**cfg["classical"]["xgboost"], random_seed=int(cfg["split"]["random_seed"]))
        xgb = XGBBaseline(xgb_cfg).fit(X_train, y_train)
        fitted["xgb"] = xgb
        preds_real["xgb"] = inverse_transform_target(xgb.predict(X_test), red_cfg)

    if "qkrr" in models:
        from src.quantum.qkrr import QKRRConfig, QuantumKernelRidge  # lazy

        qkrr_yaml = cfg["quantum"]["qkrr"]
        qkrr_cfg = QKRRConfig(
            n_qubits=red_cfg.n_qubits,
            n_features=red_cfg.effective_k,
            alpha=float(qkrr_yaml["alpha"]),
            bandwidth=qkrr_yaml.get("bandwidth", 1.0),
            cv_folds=int(qkrr_yaml.get("cv_folds", 5)),
            cv_subsample=qkrr_yaml.get("cv_subsample"),
            cv_random_seed=int(cfg["split"]["random_seed"]),
            feature_map=dict(cfg["quantum"]["feature_map"]),
        )
        qkrr = QuantumKernelRidge(qkrr_cfg, handles).fit(X_train, y_train)
        fitted["qkrr"] = qkrr
        preds_real["qkrr"] = inverse_transform_target(qkrr.predict(X_test), red_cfg)

    if "vqr" in models:
        from src.quantum.vqr import VQRConfig, VQRegressor  # lazy

        vqr_cfg = VQRConfig(
            n_qubits=red_cfg.n_qubits,
            feature_map=dict(cfg["quantum"]["feature_map"]),
            ansatz=dict(cfg["quantum"]["ansatz"]),
            optimizer=str(cfg["quantum"]["vqr"]["optimizer"]),
            maxiter=int(cfg["quantum"]["vqr"]["maxiter"]),
            random_seed=int(cfg["split"]["random_seed"]),
        )
        vqr = VQRegressor(vqr_cfg, handles).fit(X_train, y_train)
        fitted["vqr"] = vqr
        preds_real["vqr"] = inverse_transform_target(vqr.predict(X_test), red_cfg)

    results: dict[str, Any] = {"backend": handles.name if handles else "classical-only", "models": {}}
    for name, y_pred in preds_real.items():
        metrics = evaluate(y_test_raw, y_pred, y_train_raw)
        results["models"][name] = {"metrics": metrics}

    if "xgb" in preds_real:
        base_res = y_test_raw - preds_real["xgb"]
        for name, y_pred in preds_real.items():
            if name == "xgb":
                continue
            m_res = y_test_raw - y_pred
            results["models"][name]["vs_xgb"] = {
                "wilcoxon": wilcoxon_vs_baseline(base_res, m_res),
                "cliffs_delta": cliffs_delta(np.abs(m_res), np.abs(base_res)),
            }

    (output_dir / "results.json").write_text(json.dumps(results, indent=2))
    for name, y_pred in preds_real.items():
        np.save(output_dir / f"{name}_predictions.npy", y_pred)
    np.save(output_dir / "y_test.npy", y_test_raw)
    return results


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--backend", choices=["aer", "ibm"], default="aer")
    parser.add_argument(
        "--models",
        default="xgb,qkrr,vqr",
        help="comma-separated subset of {xgb,qkrr,vqr}",
    )
    parser.add_argument("--output-dir", default="models/latest")
    parser.add_argument(
        "--subsample",
        type=int,
        default=None,
        help="randomly subsample N rows before split. Useful to keep the quantum-kernel Gram O(N^2) tractable (e.g. --subsample 500).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    results = run(cfg, args.backend, models, Path(args.output_dir), subsample=args.subsample)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
