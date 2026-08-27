"""Research-valid benchmark utilities shared by the cross-dataset runner and tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

COLLAPSE_THRESHOLD = 0.03
BASELINE_DETECTORS = ("js_divergence", "wasserstein", "covariance_shift", "psi", "mmd", "c2st_accuracy")
FULL_FEATURES = (*BASELINE_DETECTORS, "class_entropy", "minority_share")
ABLATIONS = {
    "Full DriftForge": FULL_FEATURES,
    "Without JS divergence": tuple(x for x in FULL_FEATURES if x != "js_divergence"),
    "Without Wasserstein": tuple(x for x in FULL_FEATURES if x != "wasserstein"),
    "Without covariance shift": tuple(x for x in FULL_FEATURES if x != "covariance_shift"),
    "Without minority/tail representation": BASELINE_DETECTORS,
    "Drift metrics only": BASELINE_DETECTORS,
    "Representation/minority signals only": ("class_entropy", "minority_share"),
}


def add_collapse_columns(observations: pd.DataFrame) -> pd.DataFrame:
    """Add baseline-relative current and next-step collapse labels per trajectory."""
    keys = ["dataset", "contamination_method", "seed"]
    out = observations.sort_values([*keys, "contamination"]).reset_index(drop=True).copy()
    baseline = out.groupby(keys)["accuracy"].transform("first")
    out["accuracy_drop"] = baseline - out["accuracy"]
    out["collapse_indicator"] = (out["accuracy_drop"] >= COLLAPSE_THRESHOLD).astype(int)
    out["next_collapse_indicator"] = out.groupby(keys)["collapse_indicator"].shift(-1)
    out["has_future_observation"] = out["next_collapse_indicator"].notna()
    out["next_collapse_indicator"] = out["next_collapse_indicator"].fillna(0).astype(int)
    return out


def select_warning_threshold(scores: pd.Series, labels: pd.Series, quantile: float = 0.90) -> float:
    """Calibrate a warning cutoff from training non-collapse rows only."""
    reference = scores.loc[labels == 0].dropna()
    return float(reference.quantile(quantile)) if not reference.empty else float("inf")


def warning_lead_time(trajectory: pd.DataFrame, score_column: str, threshold: float) -> dict[str, float | bool]:
    """Measure earliest warning relative to first experimental 3-point collapse."""
    ordered = trajectory.sort_values("contamination")
    collapsed = ordered.loc[ordered["collapse_indicator"] == 1, "contamination"]
    warned = ordered.loc[ordered[score_column] >= threshold, "contamination"]
    if collapsed.empty:
        return {"collapse_observed": False, "warning_observed": not warned.empty, "lead_time": np.nan}
    if warned.empty:
        return {"collapse_observed": True, "warning_observed": False, "lead_time": np.nan}
    return {
        "collapse_observed": True,
        "warning_observed": True,
        "lead_time": float(collapsed.iloc[0] - warned.iloc[0]),
    }


def classification_metrics(y: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Return discrimination and thresholded metrics without fitting or calibration."""
    pred = (scores >= threshold).astype(int)
    negatives = y == 0
    return {
        "roc_auc": float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else np.nan,
        "pr_auc": float(average_precision_score(y, scores)) if len(np.unique(y)) == 2 else np.nan,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "false_positive_rate": float(pred[negatives].mean()) if negatives.any() else np.nan,
    }
