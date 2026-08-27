# DriftForge

**Early warning for model degradation under synthetic-data contamination**

Research question:

> Can dataset-level statistical signals reveal synthetic-data-induced degradation before aggregate model accuracy substantially deteriorates?

## Key result

In this 10-seed, simulated-contamination experiment, the mean early-warning score first reached 40 at **50% contamination**. The mean minority-recall drop first reached the project's experimental 3-percentage-point threshold at **60%**, while mean aggregate accuracy first reached that threshold at **80%**. The 3-percentage-point threshold is a project-specific experimental definition, not a universal definition of model collapse.

![DriftForge multi-seed performance](results/multiseed_performance.png)

DriftForge is a compact research prototype for testing whether dataset-level warning signals can rise before downstream model performance clearly deteriorates. The current MVP uses the scikit-learn Digits dataset because it is local, multiclass, reproducible, and requires no external downloads or API keys.

The original single-seed experiment remains available in `experiments/run_experiment.py`. This repository now also includes a 10-seed repeated experiment with confidence intervals, threshold analysis, correlation analysis, and an exploratory collapse predictor in `experiments/run_multiseed.py`.

## Research question

The experiment evaluates the question stated above under one dataset, a simulated contamination mechanism, and the project's specified metrics and threshold.

## Hypotheses

- Minority-class performance may degrade before aggregate accuracy.
- Distributional drift metrics may provide earlier warning than aggregate accuracy.
- Combining multiple drift signals may better identify high-risk contamination regimes.

These are working hypotheses for the current MVP, not proven conclusions.

## Experimental design

The current experiment:

1. Splits the Digits dataset into a pristine train/test set.
2. Standardizes features using only the real training split.
3. Generates class-conditional synthetic tabular samples with a Gaussian generator.
4. Replaces 0% to 100% of the training data with synthetic samples in 10-point increments.
5. Supports recursive synthetic generation, where each later synthetic batch is fit on the previous mixed generation rather than only on pristine real data.
6. Trains the same Random Forest classifier at every contamination level.
7. Evaluates on the untouched real test set.
8. Repeats the full recursive experiment across 10 seeds:

```python
SEEDS = [42, 7, 21, 77, 101, 123, 256, 512, 999, 2026]
```

For each contamination level, DriftForge reports:

- Accuracy
- Macro F1
- Minority recall
- Jensen-Shannon divergence
- Wasserstein distance
- Covariance shift
- Class entropy
- Minority share
- Early-warning score

The multi-seed summary reports the mean, sample standard deviation, and 95% confidence interval using:

```text
CI95 = 1.96 * std / sqrt(n)
```

The current project also uses an **experimental** collapse threshold of a 3 percentage point accuracy drop relative to the 0% contamination baseline. This threshold is project-specific and should not be treated as a universal definition of model collapse.

## Results

The following results come from the local multi-seed run saved on **August 28, 2026** in `results/`.

Across 10 seeds, mean baseline accuracy at 0% contamination was **97.13%** and mean baseline minority recall was **93.46%**. The early-warning score crossed **40** at **50% contamination**, mean minority recall reached the project's 3-point drop threshold at **60% contamination**, and mean accuracy did not reach the same 3-point drop threshold until **80% contamination**.

Selected multi-seed summary points:

| Synthetic contamination | Mean accuracy (95% CI) | Mean minority recall (95% CI) | Mean early-warning score |
|---:|---:|---:|---:|
| 0% | 97.13% +/- 0.36 pp | 93.46% +/- 2.40 pp | 0.00 |
| 50% | 95.61% +/- 0.41 pp | 91.15% +/- 2.71 pp | 46.49 |
| 60% | 95.09% +/- 0.62 pp | 88.46% +/- 3.97 pp | 54.08 |
| 80% | 93.67% +/- 0.84 pp | 87.31% +/- 3.56 pp | 72.66 |
| 100% | 90.81% +/- 0.78 pp | 86.15% +/- 5.02 pp | 99.03 |

Threshold analysis from `results/early_warning_analysis.csv`:

- First mean early-warning score >= 40: **50% contamination**
- First mean early-warning score >= 50: **60% contamination**
- First mean early-warning score >= 60: **70% contamination**
- First mean minority-recall drop >= 0.03: **60% contamination**
- First mean accuracy drop >= 0.03: **80% contamination**

Correlation analysis from `results/correlation_analysis.csv` showed strong **associations** between drift metrics and both current and next-step degradation, without implying causation. For example:

- Pearson correlation between `js_divergence` and current `accuracy_drop`: **0.918**
- Pearson correlation between `early_warning_score` and next-step `accuracy_drop`: **0.851**
- Pearson correlation between `wasserstein` and next-step `accuracy_drop`: **0.877**

The exploratory collapse predictor uses grouped cross-validation by seed and a logistic regression model to predict whether the **next contamination step** will cross the project's 3-point accuracy-drop threshold. On this MVP dataset it achieved:

- ROC-AUC: **0.968**
- Precision: **0.774**
- Recall: **0.923**
- F1: **0.842**

These predictor results are encouraging but still exploratory because they come from one dataset, one contamination mechanism, and only 10 seeds.

The plot is designed to make it easy to inspect whether minority-class performance deteriorates before aggregate accuracy reaches the experimental collapse threshold.

## Output

The single-seed and multi-seed runs write:

```text
results/
|-- experiment_results.csv
|-- performance_vs_contamination.png
|-- warning_signals.png
|-- warning_vs_accuracy_drop.png
|-- multiseed_results.csv
|-- multiseed_summary.csv
|-- multiseed_performance.png
|-- early_warning_analysis.csv
|-- correlation_analysis.csv
`-- collapse_predictor_metrics.csv
```

## Limitations

- The MVP currently studies one dataset and effectively one domain.
- The contamination mechanism is simulated with a simple Gaussian class-conditional generator.
- Ten seeds provide a more credible estimate than a single-seed prototype, but they are still not exhaustive.
- The current early-warning score is heuristic and hand-crafted.
- Correlation analysis does not establish causation.
- The project's 3-point collapse threshold is experimental and project-specific.

## Reproducibility

Create and activate a virtual environment, install dependencies, then run:

```powershell
python experiments\run_experiment.py
python experiments\run_multiseed.py
python -m pytest -q
```

If you are starting from scratch on Windows:

```powershell
python -m venv ..\.venv
..\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Repository structure

```text
driftforge/
|-- driftforge/
|   |-- __init__.py
|   |-- metrics.py
|   |-- scoring.py
|   `-- synthetic.py
|-- experiments/
|   |-- run_experiment.py
|   `-- run_multiseed.py
|-- tests/
|   |-- test_metrics.py
|   `-- test_multiseed.py
|-- results/
|-- requirements.txt
|-- LICENSE
`-- README.md
```

## What makes this a research prototype rather than a demo?

The output is not predetermined. DriftForge asks whether statistical signals are associated with downstream degradation under this experimental setup. The important follow-up is to repeat the experiment across datasets, generators, seeds, models, contamination mechanisms, and subgroup definitions, then learn and validate a calibrated collapse-risk estimator rather than hard-coding one.

## Next experiments

- Add more datasets and domains.
- Compare recursive generation against one-shot synthetic mixing.
- Add stronger synthetic generators such as CTGAN or Gaussian copulas.
- Measure subgroup fairness and calibration effects.
- Replace the heuristic early-warning score with a learned and calibrated risk model.

## License

MIT
