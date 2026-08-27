from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_experiment import run  # noqa: E402

SEEDS = [42, 7, 21, 77, 101, 123, 256, 512, 999, 2026]
METRIC_COLUMNS = [
    "accuracy",
    "macro_f1",
    "minority_recall",
    "js_divergence",
    "wasserstein",
    "covariance_shift",
    "class_entropy",
    "minority_share",
    "early_warning_score",
]
SIGNAL_COLUMNS = [
    "js_divergence",
    "wasserstein",
    "covariance_shift",
    "class_entropy",
    "minority_share",
    "early_warning_score",
]
PREDICTOR_FEATURES = SIGNAL_COLUMNS + ["contamination"]
COLLAPSE_THRESHOLD = 0.03


def compute_ci95(std: float, n: int) -> float:
    """Return the 95% confidence interval half-width from a sample standard deviation."""
    if n <= 1 or not np.isfinite(std):
        return 0.0
    return float(1.96 * std / np.sqrt(n))


def run_multiseed_experiment(seeds: list[int] | tuple[int, ...] = SEEDS, recursive: bool = True) -> pd.DataFrame:
    """Run the existing contamination experiment for each requested random seed."""
    frames = []
    for seed in seeds:
        seed_df = run(seed=seed, recursive=recursive).copy()
        seed_df["seed"] = seed
        seed_df["contamination"] = seed_df["contamination"].round(10)
        frames.append(seed_df[["seed", "contamination", *METRIC_COLUMNS]])
    return pd.concat(frames, ignore_index=True)


def _sample_std(series: pd.Series) -> float:
    return float(series.std(ddof=1))


def _mean_ci95(series: pd.Series) -> float:
    return compute_ci95(_sample_std(series), int(series.count()))


def summarize_multiseed_results(observations: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the per-seed observations at each contamination level."""
    aggregations: dict[str, tuple[str, str | Callable[[pd.Series], float]]] = {"n_seeds": ("seed", "count")}
    for metric in METRIC_COLUMNS:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, _sample_std)
        aggregations[f"{metric}_ci95"] = (metric, _mean_ci95)

    summary = (
        observations.groupby("contamination", as_index=False)
        .agg(**aggregations)
        .sort_values("contamination")
        .reset_index(drop=True)
    )
    return summary


def _future_max(values: pd.Series) -> np.ndarray:
    """For each position, return the maximum value available in later rows only."""
    arr = values.to_numpy(dtype=float)
    future = np.zeros_like(arr, dtype=float)
    running = 0.0
    for idx in range(len(arr) - 1, -1, -1):
        future[idx] = running
        running = max(running, arr[idx])
    return future


def add_drop_and_future_columns(observations: pd.DataFrame) -> pd.DataFrame:
    """Add baseline-referenced performance drops and future-collapse targets."""
    enriched = observations.sort_values(["seed", "contamination"]).reset_index(drop=True).copy()
    baseline = enriched.groupby("seed")[["accuracy", "minority_recall"]].transform("first")
    enriched["baseline_accuracy"] = baseline["accuracy"]
    enriched["baseline_minority_recall"] = baseline["minority_recall"]
    enriched["accuracy_drop"] = enriched["baseline_accuracy"] - enriched["accuracy"]
    enriched["minority_recall_drop"] = enriched["baseline_minority_recall"] - enriched["minority_recall"]
    enriched["next_accuracy_drop"] = enriched.groupby("seed")["accuracy_drop"].shift(-1)
    enriched["next_minority_recall_drop"] = enriched.groupby("seed")["minority_recall_drop"].shift(-1)
    enriched["future_accuracy_drop_max"] = enriched.groupby("seed")["accuracy_drop"].transform(_future_max)
    enriched["future_minority_recall_drop_max"] = enriched.groupby("seed")["minority_recall_drop"].transform(_future_max)

    group_size = enriched.groupby("seed")["seed"].transform("size")
    enriched["has_future_observation"] = enriched.groupby("seed").cumcount() < (group_size - 1)
    enriched["future_collapse"] = (
        enriched["has_future_observation"]
        & (enriched["next_accuracy_drop"] >= COLLAPSE_THRESHOLD)
    ).astype(int)
    return enriched


def _mark_first_crossing(analysis: pd.DataFrame, condition_column: str, marker_column: str) -> None:
    analysis[marker_column] = False
    matches = analysis.index[analysis[condition_column]]
    if len(matches) > 0:
        analysis.loc[matches[0], marker_column] = True


def build_early_warning_analysis(summary: pd.DataFrame) -> pd.DataFrame:
    """Create a per-contamination analysis table for experimental warning thresholds."""
    analysis = summary[
        ["contamination", "n_seeds", "accuracy_mean", "minority_recall_mean", "early_warning_score_mean"]
    ].copy()
    analysis["contamination_pct"] = analysis["contamination"] * 100.0

    baseline_accuracy = float(analysis.loc[analysis["contamination"] == 0.0, "accuracy_mean"].iloc[0])
    baseline_minority_recall = float(
        analysis.loc[analysis["contamination"] == 0.0, "minority_recall_mean"].iloc[0]
    )
    analysis["baseline_accuracy_mean"] = baseline_accuracy
    analysis["baseline_minority_recall_mean"] = baseline_minority_recall
    analysis["experimental_accuracy_collapse_threshold"] = COLLAPSE_THRESHOLD
    analysis["accuracy_drop"] = baseline_accuracy - analysis["accuracy_mean"]
    analysis["minority_recall_drop"] = baseline_minority_recall - analysis["minority_recall_mean"]
    analysis["accuracy_collapse_ge_0_03"] = analysis["accuracy_drop"] >= COLLAPSE_THRESHOLD
    analysis["minority_recall_drop_ge_0_03"] = analysis["minority_recall_drop"] >= COLLAPSE_THRESHOLD
    analysis["early_warning_ge_40"] = analysis["early_warning_score_mean"] >= 40.0
    analysis["early_warning_ge_50"] = analysis["early_warning_score_mean"] >= 50.0
    analysis["early_warning_ge_60"] = analysis["early_warning_score_mean"] >= 60.0

    _mark_first_crossing(analysis, "accuracy_collapse_ge_0_03", "is_first_accuracy_collapse")
    _mark_first_crossing(analysis, "minority_recall_drop_ge_0_03", "is_first_minority_recall_drop")
    _mark_first_crossing(analysis, "early_warning_ge_40", "is_first_early_warning_ge_40")
    _mark_first_crossing(analysis, "early_warning_ge_50", "is_first_early_warning_ge_50")
    _mark_first_crossing(analysis, "early_warning_ge_60", "is_first_early_warning_ge_60")
    return analysis


def build_correlation_analysis(observations: pd.DataFrame) -> pd.DataFrame:
    """Measure association between dataset-level signals and degradation metrics."""
    rows: list[dict[str, float | int | str]] = []
    target_columns = [
        "accuracy_drop",
        "minority_recall_drop",
        "next_accuracy_drop",
        "next_minority_recall_drop",
    ]

    for signal in SIGNAL_COLUMNS:
        for target in target_columns:
            subset = observations[[signal, target]].dropna()
            x = subset[signal].to_numpy(dtype=float)
            y = subset[target].to_numpy(dtype=float)

            pearson_r = np.nan
            pearson_pvalue = np.nan
            spearman_r = np.nan
            spearman_pvalue = np.nan
            if len(subset) >= 2 and np.ptp(x) > 0 and np.ptp(y) > 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pearson_r, pearson_pvalue = pearsonr(x, y)
                    spearman_r, spearman_pvalue = spearmanr(x, y)

            rows.append(
                {
                    "signal_metric": signal,
                    "target_metric": target,
                    "n_observations": len(subset),
                    "pearson_r": pearson_r,
                    "pearson_pvalue": pearson_pvalue,
                    "spearman_r": spearman_r,
                    "spearman_pvalue": spearman_pvalue,
                    "note": "Associative only; correlation does not establish causation.",
                }
            )

    return pd.DataFrame(rows)


def evaluate_collapse_predictor(observations: pd.DataFrame) -> pd.DataFrame:
    """Fit an exploratory grouped-CV logistic regression for future collapse risk."""
    dataset = observations.loc[observations["has_future_observation"]].copy()
    n_rows = len(dataset)
    n_seeds = int(dataset["seed"].nunique())
    positives = int(dataset["future_collapse"].sum())
    negatives = int(n_rows - positives)
    n_splits = min(5, n_seeds, positives, negatives) if positives > 0 and negatives > 0 else 0
    note = (
        "Exploratory only: one dataset, 10 seeds, a heuristic early-warning score, "
        "and a project-specific 3-point accuracy-drop threshold."
    )

    metrics = {
        "model": "logistic_regression",
        "target": "future_collapse",
        "threshold_accuracy_drop": COLLAPSE_THRESHOLD,
        "dataset_rows": n_rows,
        "n_seeds": n_seeds,
        "positive_rows": positives,
        "negative_rows": negatives,
        "positive_rate": (positives / n_rows) if n_rows else np.nan,
        "cv_strategy": "StratifiedGroupKFold by seed",
        "cv_folds": n_splits,
        "roc_auc": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "f1": np.nan,
        "features": ",".join(PREDICTOR_FEATURES),
        "note": note,
    }

    if n_rows == 0 or n_splits < 2:
        metrics["note"] = f"{note} Insufficient data for grouped cross-validation."
        return pd.DataFrame([metrics])

    X = dataset[PREDICTOR_FEATURES]
    y = dataset["future_collapse"].to_numpy(dtype=int)
    groups = dataset["seed"].to_numpy()

    estimator = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=2026)),
        ]
    )
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=2026)

    try:
        probabilities = cross_val_predict(
            estimator,
            X,
            y,
            groups=groups,
            cv=cv,
            method="predict_proba",
        )[:, 1]
    except ValueError as exc:
        metrics["note"] = f"{note} Grouped CV failed: {exc}"
        return pd.DataFrame([metrics])

    predictions = (probabilities >= 0.5).astype(int)
    metrics["roc_auc"] = float(roc_auc_score(y, probabilities))
    metrics["precision"] = float(precision_score(y, predictions, zero_division=0))
    metrics["recall"] = float(recall_score(y, predictions, zero_division=0))
    metrics["f1"] = float(f1_score(y, predictions, zero_division=0))
    return pd.DataFrame([metrics])


def save_multiseed_plot(summary: pd.DataFrame, out_path: Path) -> None:
    """Save a publication-oriented performance plot with 95% confidence bands."""
    pct = summary["contamination"] * 100.0
    fig, ax = plt.subplots(figsize=(9, 5.5))

    accuracy_mean = summary["accuracy_mean"].to_numpy(dtype=float)
    accuracy_ci = summary["accuracy_ci95"].to_numpy(dtype=float)
    recall_mean = summary["minority_recall_mean"].to_numpy(dtype=float)
    recall_ci = summary["minority_recall_ci95"].to_numpy(dtype=float)

    ax.plot(pct, accuracy_mean, color="#1b4965", linewidth=2.2, marker="o", label="Mean accuracy")
    ax.fill_between(
        pct,
        accuracy_mean - accuracy_ci,
        accuracy_mean + accuracy_ci,
        color="#1b4965",
        alpha=0.18,
        label="Accuracy 95% CI",
    )

    ax.plot(
        pct,
        recall_mean,
        color="#c1121f",
        linewidth=2.2,
        marker="s",
        label="Mean minority recall",
    )
    ax.fill_between(
        pct,
        recall_mean - recall_ci,
        recall_mean + recall_ci,
        color="#c1121f",
        alpha=0.18,
        label="Minority recall 95% CI",
    )

    ax.set_xlabel("Synthetic contamination (%)")
    ax.set_ylabel("Performance")
    ax.set_title("DriftForge multi-seed performance under synthetic contamination")
    ax.set_xticks(pct)
    ax.set_xticklabels([f"{int(value)}%" for value in pct])
    ax.set_ylim(0.0, 1.02)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _threshold_report(analysis: pd.DataFrame, marker_column: str) -> str:
    match = analysis.loc[analysis[marker_column], "contamination_pct"]
    if match.empty:
        return "not reached"
    return f"{match.iloc[0]:.0f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DriftForge multi-seed experiment.")
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Generate every contamination level from pristine real data.",
    )
    args = parser.parse_args()

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    observations = run_multiseed_experiment(recursive=not args.non_recursive)
    observations.to_csv(results_dir / "multiseed_results.csv", index=False)

    summary = summarize_multiseed_results(observations)
    summary.to_csv(results_dir / "multiseed_summary.csv", index=False)

    enriched = add_drop_and_future_columns(observations)
    early_warning = build_early_warning_analysis(summary)
    early_warning.to_csv(results_dir / "early_warning_analysis.csv", index=False)

    correlations = build_correlation_analysis(enriched)
    correlations.to_csv(results_dir / "correlation_analysis.csv", index=False)

    predictor_metrics = evaluate_collapse_predictor(enriched)
    predictor_metrics.to_csv(results_dir / "collapse_predictor_metrics.csv", index=False)

    plot_path = results_dir / "multiseed_performance.png"
    save_multiseed_plot(summary, plot_path)

    print("\nDriftForge multi-seed experiment complete.\n")
    print(f"Saved per-seed observations to: {results_dir / 'multiseed_results.csv'}")
    print(f"Saved summary statistics to: {results_dir / 'multiseed_summary.csv'}")
    print(f"Saved early-warning analysis to: {results_dir / 'early_warning_analysis.csv'}")
    print(f"Saved correlation analysis to: {results_dir / 'correlation_analysis.csv'}")
    print(f"Saved collapse predictor metrics to: {results_dir / 'collapse_predictor_metrics.csv'}")
    print(f"Saved performance plot to: {plot_path}")
    print("\nExperimental threshold crossings:")
    print(f"- Accuracy drop >= 0.03: {_threshold_report(early_warning, 'is_first_accuracy_collapse')}")
    print(
        f"- Minority recall drop >= 0.03: "
        f"{_threshold_report(early_warning, 'is_first_minority_recall_drop')}"
    )
    print(f"- Early-warning score >= 40: {_threshold_report(early_warning, 'is_first_early_warning_ge_40')}")
    print(f"- Early-warning score >= 50: {_threshold_report(early_warning, 'is_first_early_warning_ge_50')}")
    print(f"- Early-warning score >= 60: {_threshold_report(early_warning, 'is_first_early_warning_ge_60')}")


if __name__ == "__main__":
    main()
