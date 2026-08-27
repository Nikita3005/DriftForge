from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from driftforge.metrics import (  # noqa: E402
    class_entropy,
    covariance_shift,
    mean_js_divergence,
    mean_wasserstein,
    minority_share,
)
from driftforge.scoring import add_early_warning_score  # noqa: E402
from driftforge.synthetic import contaminate_training_set  # noqa: E402


def run(seed: int = 42, recursive: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    data = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.30,
        stratify=data.target,
        random_state=seed,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    minority_label = np.unique(y_train, return_counts=True)[0][
        np.argmin(np.unique(y_train, return_counts=True)[1])
    ]

    contamination_levels = np.arange(0.0, 1.01, 0.10)
    rows = []
    recursive_source = None

    for contamination in contamination_levels:
        X_mix, y_mix, next_source = contaminate_training_set(
            X_train,
            y_train,
            contamination=contamination,
            rng=rng,
            recursive_source=recursive_source if recursive else None,
        )
        if recursive:
            recursive_source = next_source

        model = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X_mix, y_mix)
        pred = model.predict(X_test)

        rows.append(
            {
                "contamination": contamination,
                "accuracy": accuracy_score(y_test, pred),
                "macro_f1": f1_score(y_test, pred, average="macro"),
                "minority_recall": recall_score(y_test, pred, labels=[minority_label], average="macro"),
                "js_divergence": mean_js_divergence(X_train, X_mix),
                "wasserstein": mean_wasserstein(X_train, X_mix),
                "covariance_shift": covariance_shift(X_train, X_mix),
                "class_entropy": class_entropy(y_mix),
                "minority_share": minority_share(y_mix),
            }
        )

    return add_early_warning_score(pd.DataFrame(rows))


def save_plots(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pct = df["contamination"] * 100

    fig = plt.figure(figsize=(8, 5))
    plt.plot(pct, df["accuracy"], marker="o", label="Accuracy")
    plt.plot(pct, df["macro_f1"], marker="o", label="Macro F1")
    plt.plot(pct, df["minority_recall"], marker="o", label="Minority recall")
    plt.xlabel("Synthetic contamination (%)")
    plt.ylabel("Model performance")
    plt.title("Model performance under synthetic contamination")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "performance_vs_contamination.png", dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(pct, df["early_warning_score"], marker="o", label="Early-warning score")
    plt.plot(pct, df["js_divergence"] * 100, marker="o", label="JS divergence ×100")
    plt.plot(pct, df["covariance_shift"] * 100, marker="o", label="Covariance shift ×100")
    plt.xlabel("Synthetic contamination (%)")
    plt.ylabel("Signal magnitude")
    plt.title("Dataset-level warning signals")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "warning_signals.png", dpi=160)
    plt.close(fig)

    baseline_accuracy = float(df.loc[df["contamination"] == 0, "accuracy"].iloc[0])
    accuracy_drop = (baseline_accuracy - df["accuracy"]) * 100
    fig = plt.figure(figsize=(8, 5))
    plt.scatter(df["early_warning_score"], accuracy_drop)
    for _, row in df.iterrows():
        plt.annotate(
            f"{int(row['contamination']*100)}%",
            (row["early_warning_score"], (baseline_accuracy - row["accuracy"]) * 100),
            fontsize=8,
        )
    plt.xlabel("Early-warning score")
    plt.ylabel("Accuracy drop (percentage points)")
    plt.title("Do warning signals rise before accuracy degrades?")
    plt.tight_layout()
    fig.savefig(out_dir / "warning_vs_accuracy_drop.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DriftForge MVP experiment.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Generate every contamination level from pristine real data.",
    )
    args = parser.parse_args()

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    df = run(seed=args.seed, recursive=not args.non_recursive)
    csv_path = results_dir / "experiment_results.csv"
    df.to_csv(csv_path, index=False)
    save_plots(df, results_dir)

    print("\nDriftForge experiment complete.\n")
    print(df.round(4).to_string(index=False))
    print(f"\nSaved results to: {csv_path}")


if __name__ == "__main__":
    main()
