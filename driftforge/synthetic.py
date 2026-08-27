from __future__ import annotations

import numpy as np

CONTAMINATION_METHODS = ("gaussian", "tail_suppression", "class_biased")


def _regularized_covariance(X: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Return a numerically stable covariance matrix."""
    cov = np.cov(X, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    scale = np.trace(cov) / max(cov.shape[0], 1)
    ridge = eps * (scale if np.isfinite(scale) and scale > 0 else 1.0)
    return cov + np.eye(cov.shape[0]) * ridge


def gaussian_class_generator(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    class_probabilities: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic tabular samples with a class-conditional Gaussian model.

    This intentionally simple generator is useful for a reproducible MVP: it
    preserves some first/second-order structure while gradually smoothing away
    tails and nonlinear relationships during recursive generation.
    """
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum() if class_probabilities is None else class_probabilities
    sampled_classes = rng.choice(classes, size=n_samples, p=probs)

    X_syn = np.empty((n_samples, X.shape[1]), dtype=float)
    y_syn = sampled_classes.copy()

    for cls in classes:
        idx = np.where(sampled_classes == cls)[0]
        if len(idx) == 0:
            continue
        X_cls = X[y == cls]
        mean = X_cls.mean(axis=0)
        cov = _regularized_covariance(X_cls)
        draws = rng.multivariate_normal(mean=mean, cov=cov, size=len(idx), method="svd")
        X_syn[idx] = draws

    return X_syn, y_syn


def _central_class_samples(X: np.ndarray, retention: float) -> np.ndarray:
    """Keep the central portion of a class as a controlled tail-loss proxy."""
    if len(X) <= 2:
        return X
    center = X.mean(axis=0)
    distances = np.sum((X - center) ** 2, axis=1)
    n_keep = max(2, int(np.ceil(len(X) * retention)))
    return X[np.argsort(distances)[:n_keep]]


def tail_suppression_generator(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    contamination: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate from progressively central class regions to suppress tails."""
    retention = max(0.25, 1.0 - 0.75 * contamination)
    classes = np.unique(y)
    X_core = np.vstack([_central_class_samples(X[y == cls], retention) for cls in classes])
    y_core = np.concatenate(
        [np.full(len(_central_class_samples(X[y == cls], retention)), cls, dtype=y.dtype) for cls in classes]
    )
    return gaussian_class_generator(X_core, y_core, n_samples, rng)


def class_biased_generator(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    contamination: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Progressively favor common classes when sampling synthetic labels."""
    _, counts = np.unique(y, return_counts=True)
    base_probabilities = counts / counts.sum()
    probabilities = base_probabilities ** (1.0 + 2.0 * contamination)
    probabilities /= probabilities.sum()
    return gaussian_class_generator(X, y, n_samples, rng, class_probabilities=probabilities)


def contaminate_training_set(
    X_real: np.ndarray,
    y_real: np.ndarray,
    contamination: float,
    rng: np.random.Generator,
    recursive_source: tuple[np.ndarray, np.ndarray] | None = None,
    method: str = "gaussian",
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """
    Replace a fraction of the training set with synthetic samples.

    Parameters
    ----------
    contamination:
        Fraction in [0, 1].
    recursive_source:
        If provided, fit the next synthetic generator on this previous mixed
        generation rather than on pristine real data. This simulates recursive
        synthetic-data feedback.

    Returns
    -------
    X_mix, y_mix, source_for_next_generation
    """
    contamination = float(np.clip(contamination, 0.0, 1.0))
    n_total = len(X_real)
    n_syn = int(round(n_total * contamination))
    n_real = n_total - n_syn

    if n_real > 0:
        real_idx = rng.choice(n_total, size=n_real, replace=False)
        X_keep = X_real[real_idx]
        y_keep = y_real[real_idx]
    else:
        X_keep = np.empty((0, X_real.shape[1]))
        y_keep = np.empty((0,), dtype=y_real.dtype)

    if method not in CONTAMINATION_METHODS:
        supported = ", ".join(CONTAMINATION_METHODS)
        raise ValueError(f"Unknown contamination method {method!r}. Supported methods: {supported}.")

    source_X, source_y = recursive_source if recursive_source is not None else (X_real, y_real)
    if n_syn > 0:
        if method == "gaussian":
            X_syn, y_syn = gaussian_class_generator(source_X, source_y, n_syn, rng)
        elif method == "tail_suppression":
            X_syn, y_syn = tail_suppression_generator(source_X, source_y, n_syn, contamination, rng)
        else:
            X_syn, y_syn = class_biased_generator(source_X, source_y, n_syn, contamination, rng)
    else:
        X_syn = np.empty((0, X_real.shape[1]))
        y_syn = np.empty((0,), dtype=y_real.dtype)

    X_mix = np.vstack([X_keep, X_syn])
    y_mix = np.concatenate([y_keep, y_syn])
    order = rng.permutation(n_total)
    X_mix, y_mix = X_mix[order], y_mix[order]

    return X_mix, y_mix, (X_mix.copy(), y_mix.copy())
