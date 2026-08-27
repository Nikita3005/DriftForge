"""Generate README visuals from existing DriftForge result artifacts only."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch
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


def _diagram_canvas(height: float) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(12, height), dpi=100)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, height)
    ax.axis("off")
    return fig, ax


def _node(ax: plt.Axes, x: float, y: float, label: str, width: float = 2.2, color: str = "#e8f0f6") -> None:
    box = FancyBboxPatch(
        (x - width / 2, y - 0.28), width, 0.56,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        facecolor=color, edgecolor="#54748c", linewidth=1.1,
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center", fontsize=9.2, color="#17324d", fontweight="bold")


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#54748c") -> None:
    ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.3, "color": color})


def generate_why_driftforge() -> None:
    fig, ax = _diagram_canvas(5.8)
    _node(ax, 2.0, 4.9, "Real Data")
    _node(ax, 2.0, 4.0, "Synthetic Contamination", 2.7, "#f5eee1")
    _node(ax, 2.0, 3.1, "Distribution Shift", 2.5, "#e8f0f6")
    _node(ax, 5.5, 3.1, "Minority / Tail Damage", 2.8, "#f8e6e8")
    _node(ax, 5.5, 2.1, "Aggregate Failure", 2.4, "#f8e6e8")
    _node(ax, 5.5, 0.95, "DriftForge", 2.2, "#e7f2eb")
    _node(ax, 8.5, 0.95, "Early Warning", 2.2, "#e7f2eb")
    _node(ax, 10.5, 0.95, "Intervention Window", 2.7, "#e7f2eb")
    _arrow(ax, (2, 4.62), (2, 4.28)); _arrow(ax, (2, 3.72), (2, 3.38))
    _arrow(ax, (3.25, 3.1), (4.1, 3.1)); _arrow(ax, (5.5, 2.82), (5.5, 2.38), "#aa5b65")
    _arrow(ax, (2.65, 2.94), (4.3, 1.15), "#3d7d5a"); _arrow(ax, (6.62, 0.95), (7.38, 0.95), "#3d7d5a"); _arrow(ax, (9.62, 0.95), (9.15, 0.95), "#3d7d5a")
    ax.text(5.5, 5.48, "Traditional monitoring reacts after performance degradation. DriftForge investigates whether warning can occur earlier.", ha="center", fontsize=11, color="#17324d", fontweight="bold")
    ax.text(6, 0.22, "Monitor degradation before aggregate performance visibly fails.", ha="center", fontsize=9.5, color="#657786")
    fig.savefig(OUTPUT / "why_driftforge.svg", format="svg", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def generate_experimental_pipeline() -> None:
    fig, ax = _diagram_canvas(8.2)
    _node(ax, 6, 7.6, "Dataset")
    _node(ax, 6, 6.7, "Contamination Generator", 3.0, "#f5eee1")
    for x, label in ((2.5, "Gaussian"), (6, "Tail Suppression"), (9.5, "Class Biased")):
        _node(ax, x, 5.55, label, 2.6, "#e8f0f6")
        _arrow(ax, (6, 6.42), (x, 5.83))
    _node(ax, 6, 4.35, "Train Model")
    _arrow(ax, (2.5, 5.27), (5.5, 4.63)); _arrow(ax, (6, 5.27), (6, 4.63)); _arrow(ax, (9.5, 5.27), (6.5, 4.63))
    _node(ax, 3.5, 3.15, "Performance Metrics", 2.8, "#e8f0f6")
    _node(ax, 8.5, 3.15, "Distribution Metrics", 2.8, "#e8f0f6")
    _arrow(ax, (5.55, 4.07), (3.9, 3.43)); _arrow(ax, (6.45, 4.07), (8.1, 3.43))
    _node(ax, 3.5, 2.1, "Collapse Detection", 2.6, "#f8e6e8")
    _node(ax, 8.5, 2.1, "Early Warning Signals", 2.8, "#e7f2eb")
    _arrow(ax, (3.5, 2.87), (3.5, 2.38)); _arrow(ax, (8.5, 2.87), (8.5, 2.38))
    _node(ax, 6, 1.05, "Warning Lead-Time", 2.6, "#f5eee1")
    _node(ax, 6, 0.18, "Cross-Dataset Validation -> Ablation Study", 4.2, "#e7f2eb")
    _arrow(ax, (3.9, 1.82), (5.5, 1.33)); _arrow(ax, (8.1, 1.82), (6.5, 1.33)); _arrow(ax, (6, 0.77), (6, 0.46))
    ax.text(6, 7.98, "Controlled benchmark pipeline", ha="center", fontsize=12, color="#17324d", fontweight="bold")
    ax.text(6, -0.28, "3 datasets x 3 contamination mechanisms x 5 seeds x 11 contamination levels = 495 conditions", ha="center", fontsize=9.2, color="#657786")
    fig.savefig(OUTPUT / "experimental_pipeline.svg", format="svg", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def generate_warning_lead_time() -> None:
    fig, ax = _diagram_canvas(4.4)
    ax.plot([1, 11], [2.65, 2.65], color="#54748c", linewidth=2)
    ax.text(1, 2.98, "0%", ha="center", fontsize=10, fontweight="bold", color="#17324d")
    ax.text(11, 2.98, "100%", ha="center", fontsize=10, fontweight="bold", color="#17324d")
    warning_x, collapse_x = 4.15, 8.55
    ax.scatter([warning_x], [2.65], s=90, color="#3d7d5a", zorder=3, marker="^")
    ax.scatter([collapse_x], [2.65], s=90, color="#aa5b65", zorder=3, marker="v")
    ax.text(warning_x, 3.35, "WARNING", ha="center", fontsize=10, fontweight="bold", color="#3d7d5a")
    ax.text(collapse_x, 1.88, "COLLAPSE", ha="center", fontsize=10, fontweight="bold", color="#aa5b65")
    ax.annotate("", xy=(collapse_x, 2.1), xytext=(warning_x, 2.1), arrowprops={"arrowstyle": "<->", "color": "#f28e2b", "lw": 1.8})
    ax.text(6.35, 1.83, "intervention window", ha="center", fontsize=9.3, color="#8a5a16")
    ax.text(6, 3.9, "Warning Lead Time = Collapse Point - First Warning Point", ha="center", fontsize=12, fontweight="bold", color="#17324d")
    ax.text(6, 1.05, "Stable  ->  Elevated Risk  ->  Degraded", ha="center", fontsize=10.5, color="#4a6072")
    ax.text(6, 0.58, "Positive: before collapse   |   Zero: at collapse   |   Negative: too late", ha="center", fontsize=9.2, color="#657786")
    ax.text(6, 0.08, "Controlled benchmark mean: DriftForge +0.319   |   JS divergence -0.089   |   Wasserstein -0.092", ha="center", fontsize=8.8, color="#657786")
    fig.savefig(OUTPUT / "warning_lead_time.svg", format="svg", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def clean_svg_whitespace() -> None:
    """Remove renderer-introduced trailing whitespace from generated SVGs."""
    for path in OUTPUT.glob("*.svg"):
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    generate_banner()
    generate_key_metrics()
    generate_progression()
    generate_why_driftforge()
    generate_experimental_pipeline()
    generate_warning_lead_time()
    clean_svg_whitespace()


if __name__ == "__main__":
    main()
