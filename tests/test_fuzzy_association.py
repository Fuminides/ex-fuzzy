"""Tests for the fuzzy association rule mining, prescreening and selection stages."""

import numpy as np
import pytest
from sklearn.datasets import load_iris

from ex_fuzzy import _fuzzy_association as association
from ex_fuzzy import fuzzy_sets as fs
from ex_fuzzy import utils


@pytest.fixture(scope='module')
def iris_memberships():
    X, y = load_iris(return_X_y=True)
    partitions = utils.construct_partitions(X, fs.FUZZY_SETS.t1)
    return association.membership_matrices(partitions, X), y


def test_callable_partitions_with_a_single_term():
    memberships = association.membership_matrices([lambda values: values], np.array([[0.2], [0.8]]))
    np.testing.assert_array_equal(memberships[0], [[0.2, 0.8]])


def test_mining_caps_each_class_and_falls_back_to_the_best_rules(iris_memberships):
    memberships, y = iris_memberships

    capped = association.mine_candidates(memberships, y, 3, max_conditions=2, min_support=0.0,
                                         min_confidence=0.0, per_class_cap=1)
    assert np.bincount(capped['consequent'], minlength=3).tolist() == [1, 1, 1]

    # No rule can be that confident, so every class takes its relaxed best rules.
    strict = association.mine_candidates(memberships, y, 3, max_conditions=2, min_confidence=1.01)
    assert set(strict['consequent']) == {0, 1, 2}
    assert np.all(strict['weight'] >= 1e-3)


def test_best_rules_for_a_class_that_nothing_covers():
    memberships = [np.zeros((2, 4))]
    empty = association.best_rules_for_class(memberships, np.array([0, 1, 0, 1]), 2, 1)
    assert len(empty['consequent']) == 0 and empty['firing'] is None


def test_prescreen_skips_missing_classes_and_keeps_one_rule_per_present_class():
    y = np.array([0, 0, 1, 1])
    pool = dict(features=[(0,), (0,)], terms=[(0,), (1,)], consequent=np.array([1, 1]), weight=np.ones(2),
                confidence=np.zeros(2), support=np.zeros(2), quality=np.array([0.2, 0.1]),
                firing=np.array([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]))

    screened = association.prescreen(pool, y, 2, per_class=2)
    # Class 0 has no candidates. Class 1's rules only cover class 0, so its best rule is kept anyway.
    assert screened['terms'] == [(0,)]
    assert screened['consequent'].tolist() == [1]


def test_selecting_from_an_empty_or_unsorted_pool():
    assert association.select_rules(association._empty_pool(), np.array([0, 1]), 2, 0).size == 0

    unsorted = dict(features=[(0,), (0,)], terms=[(0,), (1,)], consequent=np.array([1, 0]), weight=np.ones(2),
                    confidence=np.ones(2), support=np.ones(2), quality=np.ones(2), firing=np.ones((2, 2)))
    with pytest.raises(ValueError, match='sorted by consequent'):
        association.select_rules(unsorted, np.array([0, 1]), 2, 0)


def test_merging_ignores_empty_pools(iris_memberships):
    memberships, y = iris_memberships
    mined = association.mine_candidates(memberships, y, 3, max_conditions=1)

    merged = association._merge([association._empty_pool(), mined])
    assert merged['terms'] == mined['terms']
    np.testing.assert_array_equal(merged['firing'], mined['firing'])
