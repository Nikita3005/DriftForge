# DriftForge

**Early warning for synthetic-data-induced model degradation**

Research question:

> Can dataset-level statistical signals provide warning before synthetic-data contamination reaches a project-defined model-degradation threshold?

## Key Finding

Across the controlled benchmark, Jensen-Shannon divergence achieved stronger conventional discrimination, while DriftForge produced the only positive mean warning lead time and warned before a larger share of observed collapse events.

| Method | ROC-AUC | Mean warning lead time | Warned before collapse |
|---|---:|---:|---:|
| JS divergence | 0.907 | -0.089 | 17.5% |
| Wasserstein distance | 0.891 | -0.092 | 17.5% |
| DriftForge | 0.831 | 0.319 | 42.5% |

![Detector warning lead time](results/benchmark/detector_lead_time.png)

The benchmark uses three local scikit-learn datasets, three controlled contamination mechanisms, five seeds, and eleven contamination levels. A collapse is an experimental, project-specific 3-percentage-point accuracy drop from baseline, not a universal definition of model collapse.

## Results and Reproduction

See [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md), [detector comparison](results/benchmark/detector_comparison.csv), [leave-one-dataset-out results](results/benchmark/leave_one_dataset_out.csv), and [ablations](results/benchmark/ablation_results.csv).

```powershell
python -m pytest -q
python experiments\run_benchmark.py
```

The benchmark command regenerates the existing results; it is not required to inspect them. DriftForge is exploratory: its controlled proxies do not establish causation, and cross-dataset validation is not cross-domain validation.

## License

MIT
