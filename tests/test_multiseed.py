import math

import pandas as pd

from experiments.run_multiseed import (
    METRIC_COLUMNS,
    add_drop_and_future_columns,
    compute_ci95,
    summarize_multiseed_results,
)


def _toy_observations() -> pd.DataFrame:
    rows = [
        (42, 0.0, 0.95, 0.94, 0.90, 0.01, 0.02, 0.03, 3.30, 0.10, 5.0),
        (42, 0.5, 0.92, 0.91, 0.87, 0.03, 0.05, 0.08, 3.25, 0.09, 35.0),
        (42, 1.0, 0.88, 0.87, 0.82, 0.06, 0.09, 0.14, 3.20, 0.08, 65.0),
        (7, 0.0, 0.96, 0.95, 0.91, 0.02, 0.03, 0.04, 3.31, 0.10, 6.0),
        (7, 0.5, 0.93, 0.92, 0.88, 0.04, 0.06, 0.09, 3.26, 0.09, 36.0),
        (7, 1.0, 0.89, 0.88, 0.83, 0.07, 0.10, 0.15, 3.21, 0.08, 66.0),
    ]
    columns = ["seed", "contamination", *METRIC_COLUMNS]
    return pd.DataFrame(rows, columns=columns)


def test_confidence_interval_calculation_is_sensible():
    ci95 = compute_ci95(std=0.10, n=4)
    assert math.isclose(ci95, 0.098, rel_tol=1e-9)
    assert compute_ci95(std=float("nan"), n=4) == 0.0


def test_multiseed_aggregation_creates_one_row_per_contamination_level():
    summary = summarize_multiseed_results(_toy_observations())
    assert summary["contamination"].tolist() == [0.0, 0.5, 1.0]
    assert summary["n_seeds"].tolist() == [2, 2, 2]


def test_baseline_accuracy_drop_is_zero_at_zero_contamination():
    enriched = add_drop_and_future_columns(_toy_observations())
    baseline_rows = enriched.loc[enriched["contamination"] == 0.0]
    assert baseline_rows["accuracy_drop"].eq(0.0).all()


def test_summary_contains_expected_metric_columns():
    summary = summarize_multiseed_results(_toy_observations())
    expected = {"contamination", "n_seeds"}
    for metric in METRIC_COLUMNS:
        expected.add(f"{metric}_mean")
        expected.add(f"{metric}_std")
        expected.add(f"{metric}_ci95")
    assert expected.issubset(set(summary.columns))
