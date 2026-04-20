"""End-to-end smoke test on synthetic data that mimics btoe's schema.

Generates 150 rows with 50 'emb_*' columns + 10 'repo_*' columns and a
nonlinear duration_hours target, then runs xgb + qkrr (vqr disabled by
default because it is slow even on the simulator — add 'vqr' to --models
if you want to wait).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.pipeline import load_config, run


def synthesise(n: int = 150, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(n, 50))
    repo = rng.normal(size=(n, 10))

    signal = (
        1.5 * emb[:, 0]
        + 0.8 * emb[:, 1] ** 2
        - 0.6 * emb[:, 2] * repo[:, 0]
        + 0.4 * np.sin(repo[:, 1])
        + 0.3 * repo[:, 2]
    )
    noise = rng.normal(scale=0.4, size=n)
    duration_hours = np.expm1(2.0 + signal + noise).clip(min=0.5, max=720)

    X = np.concatenate([emb, repo], axis=1)
    return X, duration_hours


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--models", default="xgb,qkrr")
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--output-dir", default="models/synthetic")
    args = parser.parse_args()

    cfg = load_config(args.config)
    X, y = synthesise(n=args.n)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    results = run(cfg, backend="aer", models=models, output_dir=Path(args.output_dir), X_override=X, y_override=y)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
