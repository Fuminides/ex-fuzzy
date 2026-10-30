"""Fitted genetic estimators pickle without their training data."""
import pickle

import numpy as np
import pytest

from ex_fuzzy import evolutionary_backends as ev_backends
from ex_fuzzy import evolutionary_fit as evf
from ex_fuzzy import evolutionary_fit_regression as evr


def _data(n_rows, seed=0):
    X = np.random.default_rng(seed).normal(size=(n_rows, 4))
    return X, (X[:, 0] + X[:, 1] > 0).astype(int), 2.0 * X[:, 0] - X[:, 2]


def _fit_classifier(n_rows):
    X, y, _ = _data(n_rows)
    model = evf.BaseFuzzyRulesClassifier(nRules=4, nAnts=2, n_linguistic_variables=3)
    return model.fit(X, y, n_gen=3, pop_size=10, random_state=0)


def _fit_regressor(n_rows):
    X, _, y = _data(n_rows)
    model = evr.BaseFuzzyRulesRegressor(nRules=4, nAnts=2, n_linguistic_variables=3)
    return model.fit(X, y, n_gen=3, pop_size=10, random_state=0)


def test_regressor_pickle_does_not_grow_with_the_training_rows():
    small, large = _fit_regressor(200), _fit_regressor(8000)
    small_size, large_size = len(pickle.dumps(small)), len(pickle.dumps(large))
    # The training set alone is 8000 x 4 floats, over 250 kB.
    assert large_size < small_size * 1.5 and large_size < 100_000


def test_classifier_result_pickle_does_not_grow_with_the_training_rows():
    small, large = _fit_classifier(200), _fit_classifier(8000)
    small_size = len(pickle.dumps(small.optimization_result_))
    large_size = len(pickle.dumps(large.optimization_result_))
    assert large_size < small_size * 1.5 and large_size < 100_000


@pytest.mark.parametrize('fit', [_fit_classifier, _fit_regressor])
def test_stored_result_keeps_the_search_outcome_but_not_the_problem(fit):
    model = fit(200)
    result = model.optimization_result_
    assert result['algorithm'].problem is None and result['res'].problem is None
    assert result['algorithm'].evaluator.n_eval > 0
    assert len(result['pop']) == 10 and len(result['X']) > 0
    assert np.isfinite(model.performance)


def test_pickled_models_predict_bit_for_bit():
    X, _, _ = _data(300, seed=1)
    classifier, regressor = _fit_classifier(400), _fit_regressor(400)
    restored_classifier = pickle.loads(pickle.dumps(classifier))
    restored_regressor = pickle.loads(pickle.dumps(regressor))

    np.testing.assert_array_equal(restored_classifier.predict(X), classifier.predict(X))
    explained, restored = classifier.explainable_predict(X), restored_classifier.explainable_predict(X)
    for before, after in zip(explained, restored):
        np.testing.assert_array_equal(after, before)
    np.testing.assert_array_equal(restored_regressor.predict(X), regressor.predict(X))
    assert restored_regressor.performance_ == regressor.performance_
    assert restored_classifier.performance == classifier.performance


def test_release_problem_data_leaves_results_without_an_algorithm_alone():
    result = {'X': np.zeros(3), 'algorithm': None}
    assert ev_backends.release_problem_data(result) is result
    assert result['algorithm'] is None and ev_backends.release_problem_data({}) == {}
