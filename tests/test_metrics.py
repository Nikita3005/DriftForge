import numpy as np

from driftforge.metrics import covariance_shift, mean_js_divergence, mean_wasserstein


def test_identical_data_has_zero_drift():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(200, 4))
    assert mean_js_divergence(X, X) < 1e-10
    assert mean_wasserstein(X, X) < 1e-10
    assert covariance_shift(X, X) < 1e-10


def test_shifted_data_has_positive_drift():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(200, 4))
    Y = X + 1.5
    assert mean_js_divergence(X, Y) > 0
    assert mean_wasserstein(X, Y) > 0
