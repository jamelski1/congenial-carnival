# congenial-carnival

Quantum machine-learning angle on the [btoe](https://github.com/jamelski1/btoe)
(Beyond Text-Only Estimation) software-effort-estimation experiment.

Btoe trains XGBoost on CodeBERT embeddings + PyDriller repo-mined features
against `duration_hours`. This project re-uses the same feature tables and
asks: **can a quantum feature map or variational quantum circuit, run on
IBM's publicly available quantum hardware, learn a better representation of
the feature interactions than a classical model on the same inputs?**

## Models compared

| Model   | Type              | What it does                                                                 |
|---------|-------------------|------------------------------------------------------------------------------|
| `xgb`   | Classical         | XGBoost on the same reduced feature set — apples-to-apples baseline          |
| `qkrr`  | Quantum-Kernel    | Ridge regression with a kernel computed from a `ZZFeatureMap`                |
| `vqr`   | Variational       | Parameterised quantum circuit (`ZZ` map + `RealAmplitudes`) trained via COBYLA |

All three read the same `X`/`y`, use the same 80/20 split, and are scored
with btoe's metrics: MAE, MdAE, MMRE, PRED(25), PRED(50), R², SA.
Wilcoxon signed-rank + Cliff's delta compare residuals against `xgb`.

## Feature budget

CodeBERT embeddings come in at 50 PCA components (btoe's `pca.n_components`).
That's still too wide for near-term quantum circuits, so we apply a second
`PCA → n_qubits` projection (default 8). The classical baseline receives the
same reduced features to isolate the "quantum vs. classical" delta from
the "fewer features" delta.

## Backends

- **`aer`** (default): local `AerSimulator`. Use for iteration.
- **`ibm`**: IBM Quantum Runtime — least-busy real device for the final eval.
  Requires `IBM_QUANTUM_TOKEN` in `.env` (see `.env.example`).

## Input data

Drop btoe's feature tables into `data/`:

```
data/features.parquet
```

Expected columns:
- `duration_hours` — target (float, > 0)
- `emb_0 … emb_49` — PCA'd CodeBERT embeddings
- any number of repo-mined numeric columns (churn, coupling, tfidf_*, …)

If you don't have real data yet, run the synthetic demo:

```
python -m examples.synthetic_demo
```

## Quickstart

```
pip install -r requirements.txt
cp .env.example .env  # paste IBM_QUANTUM_TOKEN if using ibm backend
python -m src.pipeline --config configs/default.yaml --backend aer
```

## Notes on quantum realism

Real hardware is noisy and queue-limited. Expect to do 99% of iteration on
the simulator; only submit the final fit+predict to IBM. The VQR's training
loop issues many circuit evaluations per optimiser step — on real hardware
that means minutes to hours per run. The quantum-kernel model is cheaper at
inference but still `O(N²)` circuit evaluations to build the training
kernel. Budget accordingly.
