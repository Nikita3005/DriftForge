from __future__ import annotations

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance


def _shared_histograms(a: np.ndarray, b: np.ndarray, bins: int = 20) -> tuple[np.ndarray, np.ndarray]:
    lo = float(min(np.min(a), np.min(b)))
    hi = float(max(np.max(a), np.max(b)))
    if np.isclose(lo, hi):
        return np.array([1.0]), np.array([1.0])
    edges = np.linspace(lo, hi, bins + 1)
    p, _ = np.histogram(a, bins=edges, density=False)
    q, _ = np.histogram(b, bins=edges, density=False)
    p = p.astype(float) + 1e-12
    q = q.astype(float) + 1e-12
    p /= p.sum()
    q /= q.sum()
    return p, q


def mean_js_divergence(X_ref: np.ndarray, X_cur: np.ndarray, bins: int = 20) -> float:
    """Average Jensen-Shannon divergence across numeric features."""
    vals = []
    for j in range(X_ref.shape[1]):
        p, q = _shared_histograms(X_ref[:, j], X_cur[:, j], bins=bins)
        d = jensenshannon(p, q, base=2.0)
        vals.append(float(d * d))  # scipy returns sqrt(JS divergence)
    return float(np.mean(vals))


def mean_wasserstein(X_ref: np.ndarray, X_cur: np.ndarray) -> float:
    """Average 1D Wasserstein distance across features."""
    return float(
        np.mean([
            wasserstein_distance(X_ref[:, j], X_cur[:, j])
            for j in range(X_ref.shape[1])
        ])
    )


def covariance_shift(X_ref: np.ndarray, X_cur: np.ndarray) -> float:
    """Relative Frobenius distance between covariance matrices."""
    cov_ref = np.cov(X_ref, rowvar=False)
    cov_cur = np.cov(X_cur, rowvar=False)
    denom = np.linalg.norm(cov_ref, ord="fro") + 1e-12
    return float(np.linalg.norm(cov_cur - cov_ref, ord="fro") / denom)


def class_entropy(y: np.ndarray) -> float:
    """Shannon entropy (bits) of class labels."""
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs + 1e-12)).sum())


def minority_share(y: np.ndarray) -> float:
    """Fraction of samples belonging to the least common class."""
    _, counts = np.unique(y, return_counts=True)
    return float(counts.min() / counts.sum())
