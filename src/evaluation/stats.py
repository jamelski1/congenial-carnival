from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def wilcoxon_vs_baseline(
    baseline_residuals: np.ndarray, model_residuals: np.ndarray
) -> dict[str, float]:
    """Paired Wilcoxon signed-rank on absolute residuals.

    p < alpha means the model's absolute errors differ significantly from
    the baseline's on the same test points.
    """
    b = np.abs(np.asarray(baseline_residuals, dtype=float))
    m = np.abs(np.asarray(model_residuals, dtype=float))
    if np.allclose(b, m):
        return {"statistic": 0.0, "pvalue": 1.0}
    stat, p = wilcoxon(b, m, zero_method="wilcox", alternative="two-sided")
    return {"statistic": float(stat), "pvalue": float(p)}


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta effect size. Positive = a > b on average."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    gt = np.sum(a[:, None] > b[None, :])
    lt = np.sum(a[:, None] < b[None, :])
    n = a.size * b.size
    return 0.0 if n == 0 else float((gt - lt) / n)
