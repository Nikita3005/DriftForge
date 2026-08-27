# Benchmark Results

Generated from the existing benchmark CSVs. Positive lead time indicates warning before the project-defined collapse threshold.

## Detector Comparison

| Detector | ROC-AUC | PR-AUC | F1 | Mean lead time | Warned before collapse |
|---|---:|---:|---:|---:|---:|
| JS divergence | 0.907 | 0.802 | 0.762 | -0.089 | 17.5% |
| Wasserstein | 0.891 | 0.766 | 0.688 | -0.092 | 17.5% |
| Covariance shift | 0.885 | 0.777 | 0.698 | -0.144 | 12.5% |
| PSI | 0.851 | 0.673 | 0.586 | -0.178 | 12.5% |
| MMD | 0.830 | 0.696 | 0.561 | -0.126 | 20.0% |
| C2ST | 0.824 | 0.702 | 0.610 | -0.152 | 10.0% |
| DriftForge | 0.831 | 0.718 | 0.656 | 0.319 | 42.5% |

## Leave-One-Dataset-Out

| Held-out dataset | ROC-AUC | PR-AUC | Precision | Recall | F1 | False-positive rate |
|---|---:|---:|---:|---:|---:|---:|
| Digits | 0.927 | 0.875 | 0.422 | 1.000 | 0.594 | 0.839 |
| Wine | 0.876 | 0.812 | 0.844 | 0.529 | 0.651 | 0.051 |
| Breast Cancer | 0.891 | 0.705 | 0.471 | 0.667 | 0.552 | 0.143 |
| Macro average | 0.898 | 0.797 | 0.579 | 0.732 | 0.599 | 0.344 |

## Ablation

| Variant | ROC-AUC | PR-AUC | F1 |
|---|---:|---:|---:|
| Full DriftForge | 0.898 | 0.797 | 0.599 |
| Without JS divergence | 0.898 | 0.790 | 0.599 |
| Without Wasserstein | 0.903 | 0.805 | 0.599 |
| Without covariance shift | 0.911 | 0.821 | 0.594 |
| Without minority/tail representation | 0.894 | 0.788 | 0.627 |
| Drift metrics only | 0.894 | 0.788 | 0.627 |
| Representation/minority signals only | 0.812 | 0.735 | 0.598 |
