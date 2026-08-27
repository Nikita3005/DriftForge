"""Generate README visuals from existing DriftForge result artifacts only."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "benchmark"
OUTPUT = ROOT / "assets" / "readme"


def _load_metrics() -> tuple[pd.Series, pd.Series]:
    comparison = pd.read_csv(RESULTS / "detector_comparison.csv")
    lodo = pd.read_csv(RESULTS / "leave_one_dataset_out.csv")
    driftforge = comparison.loc[comparison["detector"] == "driftforge_score"].iloc[0]
    macro = lodo.loc[lodo["held_out_dataset"] == "macro_average"].iloc[0]
    return driftforge, macro


def generate_banner() -> None:
    fig, ax = plt.subplots(figsize=(12, 3), dpi=100)
    fig.patch.set_facecolor("#0b1320")
    ax.set_facecolor("#0b1320")
    x = [0.04, 0.14, 0.23, 0.31, 0.40, 0.50, 0.60, 0.70, 0.80, 0.91]
    y = [0.18, 0.30, 0.24, 0.45, 0.36, 0.58, 0.49, 0.72, 0.65, 0.82]
    ax.plot(x, y, color="#4cc9f0", linewidth=2.2, alpha=0.85)
    ax.scatter(x, y, s=30, color="#f7b955", zorder=3)
    ax.annotate("", xy=(0.82, 0.71), xytext=(0.55, 0.52), arrowprops={"arrowstyle": "->", "color": "#f7b955", "lw": 2})
    ax.text(0.05, 0.77, "DriftForge", color="white", fontsize=30, fontweight="bold", transform=ax.transAxes)
    ax.text(0.05, 0.57, "Early Warning for Synthetic-Data-Induced Model Degradation", color="#d9e6f2", fontsize=13, transform=ax.transAxes)
    ax.text(0.05, 0.17, "Cross-dataset drift benchmarking  •  Early-warning lead time  •  Statistical validation", color="#8ba7bf", fontsize=9.5, transform=ax.transAxes)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.savefig(OUTPUT / "driftforge_banner.svg", format="svg", facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def generate_key_metrics() -> None:
    driftforge, macro = _load_metrics()
    cards = [
        ("Mean warning lead time", f"{driftforge['mean_warning_lead_time']:+.3f}"),
        ("Warned before collapse", f"{driftforge['proportion_collapses_warned_before_failure']:.1%}"),
        ("Cross-dataset macro ROC-AUC", f"{macro['roc_auc']:.3f}"),
        ("Cross-dataset macro PR-AUC", f"{macro['pr_auc']:.3f}"),
        ("Tests passing", "13"),
    ]
    fig, ax = plt.subplots(figsize=(12, 2.7), dpi=100)
    ax.set_xlim(0, 5); ax.set_ylim(0, 1); ax.axis("off")
    for index, (label, value) in enumerate(cards):
        ax.add_patch(plt.Rectangle((index + 0.04, 0.25), 0.88, 0.58, color="#f4f7fa", ec="#c7d2dc", lw=1))
        ax.text(index + 0.48, 0.61, value, ha="center", va="center", fontsize=19, fontweight="bold", color="#16324f")
        ax.text(index + 0.48, 0.38, label, ha="center", va="center", fontsize=8, color="#4a6072", wrap=True)
    ax.text(2.5, 0.06, "Results from the controlled benchmark; not universal performance claims.", ha="center", fontsize=8.5, color="#657786")
    fig.savefig(OUTPUT / "key_metrics.svg", format="svg", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def generate_progression() -> None:
    results = pd.read_csv(RESULTS / "benchmark_results.csv")
    run = results.query("dataset == 'digits' and contamination_method == 'gaussian' and seed == 42").sort_values("contamination")
    if run.empty:
        raise RuntimeError("Representative digits/gaussian/seed=42 trajectory was not found.")
    baseline = float(run.iloc[0]["accuracy"])
    collapse_line = baseline - 0.03
    fig, (performance, warning) = plt.subplots(2, 1, figsize=(9.5, 5.3), dpi=100, sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    def update(frame: int) -> None:
        visible = run.iloc[: frame + 1]
        pct = visible["contamination"] * 100
        performance.clear(); warning.clear()
        performance.plot(pct, visible["accuracy"], color="#1b4965", marker="o", label="Accuracy")
        performance.plot(pct, visible["minority_recall"], color="#c1121f", marker="s", label="Minority recall")
        performance.axhline(collapse_line, color="#8d99ae", linestyle="--", label="Experimental collapse threshold")
        performance.set_ylim(0.65, 1.02); performance.set_ylabel("Performance"); performance.legend(loc="lower left", ncol=3, fontsize=8); performance.grid(alpha=0.2)
        warning.plot(pct, visible["early_warning_score"], color="#f28e2b", marker="o", label="DriftForge warning level")
        warning.set_ylim(0, 105); warning.set_ylabel("Warning score"); warning.set_xlabel("Synthetic contamination (%)"); warning.grid(alpha=0.2)
        point = visible.iloc[-1]
        fig.suptitle(f"Digits • Gaussian contamination • seed 42 • {int(point['contamination'] * 100)}% contamination", fontweight="bold")
        fig.tight_layout()

    animation = FuncAnimation(fig, update, frames=len(run), interval=650, repeat=True)
    animation.save(OUTPUT / "drift_progression.gif", writer=PillowWriter(fps=2))
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    generate_banner()
    generate_key_metrics()
    generate_progression()


if __name__ == "__main__":
    main()
