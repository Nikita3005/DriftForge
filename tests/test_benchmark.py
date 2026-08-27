import numpy as np

from driftforge.datasets import DATASET_NAMES, load_dataset
from driftforge.synthetic import CONTAMINATION_METHODS, contaminate_training_set
from experiments.run_experiment import run


def test_local_datasets_load_with_expected_shapes():
    assert DATASET_NAMES == ("digits", "wine", "breast_cancer")
    for name in DATASET_NAMES:
        dataset = load_dataset(name)
        assert dataset.name == name
        assert dataset.X.ndim == 2
        assert len(dataset.X) == len(dataset.y)
        assert len(np.unique(dataset.y)) >= 2


def test_contamination_methods_are_deterministic_for_a_seed():
    dataset = load_dataset("wine")
    for method in CONTAMINATION_METHODS:
        first = contaminate_training_set(
            dataset.X, dataset.y, contamination=0.6, rng=np.random.default_rng(12), method=method
        )
        second = contaminate_training_set(
            dataset.X, dataset.y, contamination=0.6, rng=np.random.default_rng(12), method=method
        )
        assert np.array_equal(first[0], second[0])
        assert np.array_equal(first[1], second[1])


def test_run_records_cross_dataset_benchmark_schema():
    result = run(
        seed=5,
        dataset="wine",
        contamination_method="class_biased",
        contamination_levels=[0.0, 0.5],
        n_estimators=10,
    )
    expected = {
        "dataset",
        "contamination_method",
        "seed",
        "contamination",
        "accuracy",
        "macro_f1",
        "minority_recall",
        "early_warning_score",
    }
    assert expected.issubset(result.columns)
    assert result["dataset"].eq("wine").all()
    assert result["contamination_method"].eq("class_biased").all()
    assert result["seed"].eq(5).all()
