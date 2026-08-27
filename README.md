# DriftForge

**Early detection of model degradation under synthetic-data contamination.**

DriftForge is a compact research prototype for testing a simple question:

> **Can dataset-level statistical warning signals rise before downstream model performance visibly degrades?**

As synthetic data increasingly enters training corpora, a dataset can become progressively less representative of the real distribution even while aggregate model metrics still look healthy. DriftForge creates controlled mixtures of real and synthetic tabular data, measures distributional warning signals, trains the same downstream model at each contamination level, and compares those signals with held-out real-world performance.

## Research hypothesis

Dataset-level signals such as Jensen-Shannon divergence, Wasserstein distance, covariance shift, class entropy, and minority representation can reveal degradation earlier than aggregate accuracy alone.

## MVP experiment

The current MVP uses the scikit-learn Digits dataset because it is local, multiclass, reproducible, and requires no API keys or external downloads.

1. Split the original dataset into a pristine train/test set.
2. Standardize features using only the real training split.
3. Generate class-conditional synthetic samples with a Gaussian generator.
4. Replace 0% to 100% of the training data with synthetic samples.
5. In recursive mode, each synthetic generation is based on the previous mixed generation, creating a simple feedback loop.
6. Train the same Random Forest classifier at every contamination level.
7. Evaluate on the untouched real test set.
8. Compare model performance with dataset-level drift metrics.

## Metrics

**Model metrics**
- Accuracy
- Macro F1
- Minority-class recall

**Dataset warning signals**
- Mean Jensen-Shannon divergence
- Mean Wasserstein distance
- Relative covariance shift
- Class entropy
- Minority-class share
- DriftForge early-warning score

The early-warning score in v0.1 is intentionally transparent and heuristic. A later version should learn the relationship between dataset statistics and future model degradation across multiple datasets and seeds.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python experiments/run_experiment.py
```

Optional non-recursive baseline:

```bash
python experiments/run_experiment.py --non-recursive
```

Run tests:

```bash
pytest -q
```

## Reference run (seed 42)

The checked-in MVP currently produces the following reference behavior on the untouched real test set:

| Synthetic contamination | Accuracy | Macro F1 | Minority-class recall | Early-warning score |
|---:|---:|---:|---:|---:|
| 0% | 96.85% | 96.81% | 86.54% | 0.0 |
| 50% | 94.07% | 94.10% | 84.62% | 39.6 |
| 70% | 93.33% | 93.23% | 71.15% | 54.1 |
| 100% | 90.56% | 90.53% | 71.15% | 100.0 |

This is a **single-seed proof of concept**, not evidence of a general law. The next research step is repeated-seed evaluation with confidence intervals and multiple datasets/generators.

## Output

The experiment writes:

```text
results/
├── experiment_results.csv
├── performance_vs_contamination.png
├── warning_signals.png
└── warning_vs_accuracy_drop.png
```

## Repository structure

```text
driftforge/
├── driftforge/
│   ├── __init__.py
│   ├── metrics.py
│   ├── scoring.py
│   └── synthetic.py
├── experiments/
│   └── run_experiment.py
├── tests/
│   └── test_metrics.py
├── results/
├── requirements.txt
├── LICENSE
└── README.md
```

## What makes this a research prototype rather than a demo?

The output is not predetermined. DriftForge asks whether statistical signals consistently anticipate downstream degradation. The important follow-up is to repeat the experiment across datasets, generators, seeds, models, contamination mechanisms, and subgroup definitions, then learn a calibrated collapse-risk estimator rather than hard-coding one.

## Next experiments

- Repeat each contamination level across 20+ random seeds and report confidence intervals.
- Add tabular generators such as CTGAN or Gaussian copulas.
- Add explicit tail erosion and minority-class under-generation.
- Compare recursive generation against one-shot synthetic mixing.
- Measure calibration error and subgroup performance.
- Train a model to predict future performance drop from dataset-level signals.
- Test whether an adaptive real-data injection policy can prevent collapse.

## Research direction

A stronger version of DriftForge should answer:

> Given only the current training dataset and its history, can we estimate the probability that downstream performance or subgroup performance will deteriorate beyond a predefined threshold in the next generation?

That turns the project from a drift dashboard into a predictive data-science problem.

## License

MIT
