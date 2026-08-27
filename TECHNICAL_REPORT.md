# DriftForge

## Early Warning of Synthetic-Data-Induced Model Degradation

### Abstract

DriftForge evaluates whether early warning is distinct from detecting current distribution shift. JS divergence was the strongest conventional degradation discriminator (ROC-AUC 0.907), while the combined DriftForge model was the only detector with positive mean warning lead time (0.319 contamination units) and warned before 42.5% of observed collapses. These results are exploratory.

### Research Question

Can dataset-level statistical signals provide warning before synthetic-data contamination reaches a project-defined model-degradation threshold?

### Motivation

Current-drift discrimination and actionable warning are different objectives. DriftForge evaluates both rather than assuming a strong drift discriminator provides advance warning.

### Hypotheses

- Individual metrics may distinguish future degradation risk.
- Combining signals may improve warning behavior.
- Results may vary across datasets and contamination mechanisms.

### Experimental Setup

The benchmark uses three datasets (`digits`, `wine`, `breast_cancer`), three contamination mechanisms, five seeds (`42, 7, 21, 77, 101`), and eleven 0%-to-100% contamination levels. It uses a reduced 30-tree Random Forest for the 495-run benchmark. Experimental collapse is a 3-percentage-point accuracy drop from baseline, a project-specific definition. Cross-dataset validation trains and calibrates on the other two datasets.

### Contamination Mechanisms

- `gaussian`: class-conditional Gaussian synthesis.
- `tail_suppression`: progressively central class samples as a tail/mode-loss proxy.
- `class_biased`: synthetic labels increasingly favor common classes.

They are controlled proxies, not observations of deployed generative models.

### Baselines

JS divergence, Wasserstein distance, covariance shift, PSI, MMD, and C2ST are compared. Their scales are not directly comparable. Warning thresholds are calibrated as the 90th percentile of non-collapse rows from non-held-out datasets.

### Evaluation Metrics

Discrimination uses ROC-AUC, PR-AUC, and F1. Early-warning evaluation uses warning lead time, warned-before-collapse rate, missed-collapse rate, and false-warning rate. Positive lead time means the first warning preceded first collapse; absent warnings/collapses are recorded rather than invented.

### Main Results

Best degradation discrimination and best early-warning behavior differ. JS had the best ROC-AUC and F1. DriftForge had the only positive lead time (95% bootstrap interval 0.193 to 0.445) and the largest warned-before-collapse rate.

| Detector | ROC-AUC | PR-AUC | F1 | Mean lead time | Warned before collapse |
|---|---:|---:|---:|---:|---:|
| JS divergence | 0.907 | 0.802 | 0.762 | -0.089 | 17.5% |
| Wasserstein | 0.891 | 0.766 | 0.688 | -0.092 | 17.5% |
| Covariance shift | 0.885 | 0.777 | 0.698 | -0.144 | 12.5% |
| PSI | 0.851 | 0.673 | 0.586 | -0.178 | 12.5% |
| MMD | 0.830 | 0.696 | 0.561 | -0.126 | 20.0% |
| C2ST | 0.824 | 0.702 | 0.610 | -0.152 | 10.0% |
| DriftForge | 0.831 | 0.718 | 0.656 | 0.319 | 42.5% |

### Cross-Dataset Generalization

Leave-one-dataset-out macro ROC-AUC was 0.898 and PR-AUC was 0.797, but false-positive rates were variable. In particular, Digits had FPR 0.839, compared with 0.051 for Wine and 0.143 for Breast Cancer. The high Digits FPR is a material limitation.

| Held-out dataset | ROC-AUC | PR-AUC | F1 | False-positive rate |
|---|---:|---:|---:|---:|
| Digits | 0.927 | 0.875 | 0.594 | 0.839 |
| Wine | 0.876 | 0.812 | 0.651 | 0.051 |
| Breast Cancer | 0.891 | 0.705 | 0.552 | 0.143 |
| Macro average | 0.898 | 0.797 | 0.599 | 0.344 |

### Ablation Study

Feature accumulation was not automatically beneficial. Removing covariance shift improved ROC-AUC from 0.898 to 0.911 and PR-AUC from 0.797 to 0.821. Drift-only features improved F1 from 0.599 to 0.627.

| Variant | ROC-AUC | PR-AUC | F1 |
|---|---:|---:|---:|
| Full DriftForge | 0.898 | 0.797 | 0.599 |
| Without covariance shift | 0.911 | 0.821 | 0.594 |
| Drift metrics only | 0.894 | 0.788 | 0.627 |

### Discussion

Conventional drift detectors may identify degraded regimes well while providing limited warning lead time. DriftForge explores whether optimizing directly for early warning is distinct from detecting current drift; these results do not establish a general combined-model advantage.

### Limitations

- Three small sklearn datasets, simulated proxies, and five seeds.
- Reduced-tree benchmark model and project-specific collapse threshold.
- Threshold transfer instability and variable false-positive rates.
- No causal claim and no real-world synthetic-generator deployment.

### Future Work

Priorities are larger real-world datasets, genuine generative-model contamination, threshold calibration, direct warning-lead-time optimization, and stronger statistical validation.

### Reproducibility

```powershell
python -m pytest -q
python experiments\run_benchmark.py
```
