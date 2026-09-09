"""Counting-based MCC must equal the np.unique reference and sklearn."""
import numpy as np
import pytest
from sklearn.metrics import matthews_corrcoef

import _fitness
from _fitness import _LabelDomain, _mcc, _mcc_encoded


def _check(y, prediction, n_classes):
    domain = _LabelDomain.build(y, n_classes)
    assert domain is not None
    expected = _mcc(y, prediction)
    assert _mcc_encoded(prediction, domain) == expected


@pytest.mark.parametrize('n_classes', [1, 2, 3, 7])
def test_encoded_mcc_matches_the_unique_reference(n_classes):
    rng = np.random.default_rng(3)
    for samples in (1, 5, 200, 3001):
        for _ in range(20):
            y = rng.integers(0, n_classes, size=samples)
            prediction = rng.integers(-1, n_classes, size=samples)
            _check(y, prediction, n_classes)


def test_encoded_mcc_handles_unknowns_absent_and_constant_labels():
    y = np.array([0, 0, 1, 1, 2, 2])
    # every prediction unknown; a class absent from y; perfect and constant cases
    for prediction in (np.full(6, -1), np.array([0, 0, 1, 1, 2, 2]),
                       np.zeros(6, dtype=int), np.array([2, 2, 2, 0, 0, 0]),
                       np.array([-1, 0, -1, 1, -1, 2])):
        _check(y, prediction, 3)
    # labels present in y but never predicted, and vice versa
    _check(np.zeros(5, dtype=int), np.array([-1, 0, 0, -1, 0]), 4)


def test_encoded_mcc_matches_sklearn_where_both_apply():
    rng = np.random.default_rng(8)
    y = rng.integers(0, 3, size=400)
    prediction = rng.integers(0, 3, size=400)
    domain = _LabelDomain.build(y, 3)
    assert _mcc_encoded(prediction, domain) == pytest.approx(
        matthews_corrcoef(y, prediction))


def test_label_domain_declines_unsupported_labels():
    assert _LabelDomain.build(np.array([0.0, 1.0]), 2) is None      # float labels
    assert _LabelDomain.build(np.array([], dtype=int), 2) is None   # no samples
    assert _LabelDomain.build(np.array([0, 5]), 3) is None          # label past n_classes
    assert _LabelDomain.build(np.array([-2, 0]), 3) is None         # below unknown
    assert _LabelDomain.build(np.array([-1, 0, 1]), 2) is not None  # unknown allowed


def test_problem_caches_one_label_domain_per_fit():
    import evolutionary_fit as evf
    import utils
    import fuzzy_sets as fs
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    problem = evf.FitRuleBase(X, y, 6, 2, 3,
                              linguistic_variables=utils.construct_partitions(X, fs.FUZZY_SETS.t1))
    assert problem._label_domain() is problem._label_domain()
