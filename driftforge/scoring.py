from __future__ import annotations

import numpy as np
import pandas as pd


def add_early_warning_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 0-100 heuristic early-warning score from dataset-level drift signals.

    The score is intentionally transparent for the MVP. A future research
    version can learn the mapping from drift features to future model collapse.
    """
    out = df.copy()
    signal_cols = ["js_divergence", "wasserstein", "covariance_shift"]

    normalized = []
    for col in signal_cols:
        x = out[col].to_numpy(dtype=float)
        lo, hi = np.min(x), np.max(x)
        z = (x - lo) / (hi - lo + 1e-12)
        normalized.append(z)

    # Class-entropy loss is another warning signal.
    entropy = out["class_entropy"].to_numpy(dtype=float)
    entropy_loss = np.maximum(0.0, entropy[0] - entropy)
    if entropy_loss.max() > 0:
        entropy_loss = entropy_loss / entropy_loss.max()

    score = 0.30 * normalized[0] + 0.25 * normalized[1] + 0.30 * normalized[2] + 0.15 * entropy_loss
    out["early_warning_score"] = np.clip(score * 100.0, 0.0, 100.0)
    return out
