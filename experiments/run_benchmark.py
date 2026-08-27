"""Run the five-seed DriftForge cross-dataset benchmark."""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from driftforge.benchmark import (ABLATIONS, BASELINE_DETECTORS, COLLAPSE_THRESHOLD, add_collapse_columns,
    classification_metrics, select_warning_threshold, warning_lead_time)
from driftforge.datasets import DATASET_NAMES
from driftforge.synthetic import CONTAMINATION_METHODS
from experiments.run_experiment import run

SEEDS = (42, 7, 21, 77, 101)


def _fit_scores(train: pd.DataFrame, test: pd.DataFrame, features: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    train = train.loc[train["has_future_observation"]]
    y_train = train["next_collapse_indicator"].to_numpy()
    model = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=2026))])
    model.fit(train[list(features)], y_train)
    return model.predict_proba(test[list(features)])[:, 1], y_train


def _bootstrap_ci(values: np.ndarray, seed: int = 2026) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    draws = [np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(300)]
    return tuple(float(x) for x in np.quantile(draws, [0.025, 0.975]))


def _plot_bar(table: pd.DataFrame, label: str, value: str, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(table[label], table[value], color="#1b4965")
    ax.set_title(title)
    ax.set_ylabel(value.replace("_", " "))
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)


def main() -> None:
    out_dir = ROOT / "results" / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for dataset in DATASET_NAMES:
        for method in CONTAMINATION_METHODS:
            for seed in SEEDS:
                frames.append(
                    run(
                        seed=seed,
                        dataset=dataset,
                        contamination_method=method,
                        n_estimators=30,
                        n_jobs=1,
                    )
                )
    results = add_collapse_columns(pd.concat(frames, ignore_index=True))
    results.to_csv(out_dir / "benchmark_results.csv", index=False)
    summary = results.groupby(["dataset", "contamination_method", "contamination"], as_index=False).mean(numeric_only=True)
    summary.to_csv(out_dir / "benchmark_summary.csv", index=False)

    lodo_rows, score_frames = [], []
    for held in DATASET_NAMES:
        train, test = results.loc[results.dataset != held], results.loc[results.dataset == held]
        scores, _ = _fit_scores(train, test, ABLATIONS["Full DriftForge"])
        target = test.loc[test.has_future_observation, "next_collapse_indicator"].to_numpy()
        metrics = classification_metrics(target, scores[test.has_future_observation.to_numpy()])
        lodo_rows.append({"held_out_dataset": held, "evaluation": "dataset", **metrics, "n_rows": len(target), "note": "Exploratory cross-dataset validation; five seeds per dataset."})
        scored = test.copy(); scored["driftforge_score"] = scores; score_frames.append(scored)
    lodo = pd.DataFrame(lodo_rows)
    macro = lodo[["roc_auc", "pr_auc", "precision", "recall", "f1", "false_positive_rate"]].mean().to_dict()
    lodo = pd.concat([lodo, pd.DataFrame([{ "held_out_dataset": "macro_average", "evaluation": "macro", **macro, "n_rows": np.nan, "note": "Unweighted mean across held-out datasets." }])], ignore_index=True)
    lodo.to_csv(out_dir / "leave_one_dataset_out.csv", index=False)

    scored_all = pd.concat(score_frames, ignore_index=True)
    detector_rows = []
    for detector in (*BASELINE_DETECTORS, "driftforge_score"):
        lead_rows = []
        for held in DATASET_NAMES:
            train = results.loc[results.dataset != held]
            threshold = select_warning_threshold(train[detector] if detector in train else pd.Series(dtype=float), train["next_collapse_indicator"]) if detector != "driftforge_score" else select_warning_threshold(scored_all.loc[scored_all.dataset != held, detector], scored_all.loc[scored_all.dataset != held, "next_collapse_indicator"])
            test = scored_all.loc[scored_all.dataset == held] if detector == "driftforge_score" else results.loc[results.dataset == held]
            for _, trajectory in test.groupby(["dataset", "contamination_method", "seed"]):
                lead_rows.append(warning_lead_time(trajectory, detector, threshold))
        lead = pd.DataFrame(lead_rows); collapse = lead[lead.collapse_observed]
        warned_before = collapse.lead_time.gt(0)
        metric_rows = scored_all.loc[scored_all.has_future_observation] if detector == "driftforge_score" else results.loc[results.has_future_observation]
        metric_target = metric_rows["next_collapse_indicator"].to_numpy(); metric_scores = metric_rows[detector].to_numpy()
        metrics = classification_metrics(metric_target, metric_scores, threshold=select_warning_threshold(metric_rows[detector], metric_rows["next_collapse_indicator"]))
        lo, hi = _bootstrap_ci(collapse.lead_time.dropna().to_numpy())
        detector_rows.append({"detector": detector, "mean_warning_lead_time": collapse.lead_time.mean(), "median_warning_lead_time": collapse.lead_time.median(), "lead_time_ci95_low": lo, "lead_time_ci95_high": hi, "false_warning_rate": lead.loc[~lead.collapse_observed, "warning_observed"].mean(), "missed_collapse_rate": 1 - collapse.warning_observed.mean(), "proportion_collapses_warned_before_failure": warned_before.mean(), **metrics, "threshold_protocol": "90th percentile of non-collapse calibration rows from non-held-out datasets"})
    comparison = pd.DataFrame(detector_rows); comparison.to_csv(out_dir / "detector_comparison.csv", index=False)

    ablation_rows = []
    for name, features in ABLATIONS.items():
        held_metrics = []
        for held in DATASET_NAMES:
            train, test = results.loc[results.dataset != held], results.loc[results.dataset == held]
            scores, _ = _fit_scores(train, test, features)
            held_metrics.append(classification_metrics(test.loc[test.has_future_observation, "next_collapse_indicator"].to_numpy(), scores[test.has_future_observation.to_numpy()]))
        average = pd.DataFrame(held_metrics).mean().to_dict()
        ablation_rows.append({"variant": name, **average})
    ablation = pd.DataFrame(ablation_rows); ablation.to_csv(out_dir / "ablation_results.csv", index=False)
    _plot_bar(comparison, "detector", "mean_warning_lead_time", out_dir / "detector_lead_time.png", "Detector warning lead time")
    _plot_bar(lodo[lodo.evaluation == "dataset"], "held_out_dataset", "roc_auc", out_dir / "cross_dataset_performance.png", "Cross-dataset DriftForge ROC-AUC")
    _plot_bar(ablation, "variant", "f1", out_dir / "ablation_performance.png", "DriftForge ablation F1")


if __name__ == "__main__":
    main()
