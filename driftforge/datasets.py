"""Local classification datasets used by the DriftForge benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_wine

DATASET_NAMES = ("digits", "wine", "breast_cancer")


@dataclass(frozen=True)
class ClassificationDataset:
    """A local tabular classification dataset with a stable benchmark name."""

    name: str
    X: np.ndarray
    y: np.ndarray


def load_dataset(name: str) -> ClassificationDataset:
    """Load one of the bundled scikit-learn benchmark datasets.

    These datasets ship with scikit-learn, so benchmark tests and experiments
    do not depend on network access or a download cache.
    """
    normalized = name.lower().replace("-", "_")
    loaders = {
        "digits": load_digits,
        "wine": load_wine,
        "breast_cancer": load_breast_cancer,
    }
    if normalized not in loaders:
        supported = ", ".join(DATASET_NAMES)
        raise ValueError(f"Unknown dataset {name!r}. Supported datasets: {supported}.")

    data = loaders[normalized]()
    return ClassificationDataset(
        name=normalized,
        X=np.asarray(data.data, dtype=float),
        y=np.asarray(data.target),
    )
