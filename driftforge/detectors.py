"""Lightweight baseline distribution-shift detectors for benchmark comparisons."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def population_stability_index(X_ref: np.ndarray, X_cur: np.ndarray, bins: int = 10) -> float:
    """Return mean feature-wise PSI using reference quantile bins."""
    values = []
    for column in range(X_ref.shape[1]):
        edges = np.unique(np.quantile(X_ref[:, column], np.linspace(0, 1, bins + 1)))
        if len(edges) < 2:
            continue
        edges[0], edges[-1] = -np.inf, np.inf
        ref_counts, _ = np.histogram(X_ref[:, column], bins=edges)
        cur_counts, _ = np.histogram(X_cur[:, column], bins=edges)
        ref = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-8, None)
        cur = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-8, None)
        values.append(float(np.sum((cur - ref) * np.log(cur / ref))))
    return float(np.mean(values)) if values else 0.0


def maximum_mean_discrepancy(X_ref: np.ndarray, X_cur: np.ndarray, max_samples: int = 256) -> float:
    """Return a deterministic RBF-kernel MMD estimate with a median bandwidth."""
    X = X_ref[:max_samples]
    Y = X_cur[:max_samples]
    combined = np.vstack([X, Y])
    squared_distances = np.sum((combined[:, None, :] - combined[None, :, :]) ** 2, axis=2)
    bandwidth = float(np.median(squared_distances[squared_distances > 0])) if np.any(squared_distances > 0) else 1.0
    bandwidth = max(bandwidth, 1e-8)

    def kernel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        distances = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=2)
        return np.exp(-distances / (2.0 * bandwidth))

    return float(kernel(X, X).mean() + kernel(Y, Y).mean() - 2.0 * kernel(X, Y).mean())


def classifier_two_sample_test(X_ref: np.ndarray, X_cur: np.ndarray, seed: int) -> float:
    """Return held-out logistic C2ST accuracy; chance is approximately 0.5."""
    n = min(len(X_ref), len(X_cur), 512)
    X = np.vstack([X_ref[:n], X_cur[:n]])
    y = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed
    )
    model = StandardScaler().fit(X_train)
    classifier = LogisticRegression(max_iter=500, random_state=seed)
    classifier.fit(model.transform(X_train), y_train)
    return float(accuracy_score(y_test, classifier.predict(model.transform(X_test))))
