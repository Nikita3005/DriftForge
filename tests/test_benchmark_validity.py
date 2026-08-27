import numpy as np
import pandas as pd

from driftforge.benchmark import ABLATIONS, add_collapse_columns, warning_lead_time
from driftforge.detectors import maximum_mean_discrepancy, population_stability_index


def test_psi_and_mmd_are_zero_for_identical_data():
    X = np.arange(40, dtype=float).reshape(20, 2)
    assert population_stability_index(X, X) < 1e-10
    assert maximum_mean_discrepancy(X, X) < 1e-10


def test_warning_lead_time_and_no_collapse_edge_case():
    trajectory = pd.DataFrame({"contamination": [0.0, 0.5, 0.8], "collapse_indicator": [0, 0, 1], "score": [0.0, 0.8, 0.9]})
    assert np.isclose(warning_lead_time(trajectory, "score", 0.7)["lead_time"], 0.3)
    no_collapse = trajectory.assign(collapse_indicator=0)
    assert np.isnan(warning_lead_time(no_collapse, "score", 0.7)["lead_time"])


def test_collapse_is_calculated_within_each_dataset_trajectory():
    rows = pd.DataFrame({"dataset": ["a", "a", "b", "b"], "contamination_method": ["g"] * 4, "seed": [1] * 4, "contamination": [0.0, 1.0, 0.0, 1.0], "accuracy": [0.9, 0.85, 0.7, 0.69]})
    assert add_collapse_columns(rows)["collapse_indicator"].tolist() == [0, 1, 0, 0]


def test_ablation_variants_select_distinct_feature_sets():
    assert "js_divergence" not in ABLATIONS["Without JS divergence"]
    assert ABLATIONS["Representation/minority signals only"] == ("class_entropy", "minority_share")
