"""Counting-based class counts must equal the comparison reference, and macro F1 sklearn's."""
import numpy as np
import pytest
from sklearn.metrics import f1_score

from ex_fuzzy._fitness import _LabelDomain, _class_counts, _class_counts_encoded, _macro_f1


def _check(y, prediction, n_classes):
    domain = _LabelDomain.build(y, n_classes)
    assert domain is not None
    # Float labels have no counting layout, so they take the comparison path.
    expected = _class_counts(y.astype(float), prediction, n_classes)
    for counted, reference in zip(_class_counts_encoded(prediction, domain), expected):
        np.testing.assert_array_equal(counted, reference)


@pytest.mark.parametrize('n_classes', [1, 2, 3, 7])
def test_encoded_counts_match_the_comparison_reference(n_classes):
    rng = np.random.default_rng(3)
    for samples in (1, 5, 200, 3001):
        for _ in range(20):
            y = rng.integers(0, n_classes, size=samples)
            prediction = rng.integers(-1, n_classes, size=samples)
            _check(y, prediction, n_classes)


def test_encoded_counts_handle_unknowns_absent_and_constant_labels():
    y = np.array([0, 0, 1, 1, 2, 2])
    # every prediction unknown; a class absent from y; perfect and constant cases
    for prediction in (np.full(6, -1), np.array([0, 0, 1, 1, 2, 2]),
                       np.zeros(6, dtype=int), np.array([2, 2, 2, 0, 0, 0]),
                       np.array([-1, 0, -1, 1, -1, 2])):
        _check(y, prediction, 3)
    # labels present in y but never predicted, and vice versa
    _check(np.zeros(5, dtype=int), np.array([-1, 0, 0, -1, 0]), 4)


@pytest.mark.parametrize('n_classes', [2, 3, 7])
def test_macro_f1_matches_sklearn(n_classes):
    rng = np.random.default_rng(8)
    for samples in (5, 400):
        y = rng.integers(0, n_classes, size=samples)
        prediction = rng.integers(-1, n_classes, size=samples)
        expected = f1_score(y, prediction, labels=np.arange(n_classes), average='macro',
                            zero_division=0)
        assert _macro_f1(*_class_counts(y, prediction, n_classes)) == pytest.approx(expected)


def test_macro_f1_is_the_same_for_one_candidate_and_a_population():
    rng = np.random.default_rng(5)
    y = rng.integers(0, 4, size=300)
    predictions = rng.integers(-1, 4, size=(6, 300))
    counts = [_class_counts(y, prediction, 4) for prediction in predictions]
    batched = _macro_f1(np.stack([c[0] for c in counts]), np.stack([c[1] for c in counts]),
                        counts[0][2])
    for row, (tp, predicted, actual) in zip(batched, counts):
        assert row == _macro_f1(tp, predicted, actual)


def test_label_domain_declines_unsupported_labels():
    assert _LabelDomain.build(np.array([0.0, 1.0]), 2) is None      # float labels
    assert _LabelDomain.build(np.array([], dtype=int), 2) is None   # no samples
    assert _LabelDomain.build(np.array([0, 5]), 3) is None          # label past n_classes
    assert _LabelDomain.build(np.array([-2, 0]), 3) is None         # below unknown
    assert _LabelDomain.build(np.array([-1, 0, 1]), 2) is not None  # unknown allowed


def test_problem_caches_one_label_domain_per_fit():
    from ex_fuzzy import evolutionary_fit as evf
    from ex_fuzzy import utils
    from ex_fuzzy import fuzzy_sets as fs
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    problem = evf.FitRuleBase(X, y, 6, 2, 3,
                              linguistic_variables=utils.construct_partitions(X, fs.FUZZY_SETS.t1))
    assert problem._label_domain() is problem._label_domain()
