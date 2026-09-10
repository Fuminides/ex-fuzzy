"""Tests for the fuzzy partition validation guards."""
import numpy as np
import pandas as pd
import pytest

import fuzzy_sets as fs
import utils


@pytest.fixture(scope='module')
def numerical_data():
    rng = np.random.default_rng(0)
    return pd.DataFrame({'a': rng.normal(size=400), 'b': rng.normal(size=400)})


def test_type_1_partitions_are_validated(numerical_data):
    partitions = utils.construct_partitions(numerical_data, fs.FUZZY_SETS.t1,
                                            n_partitions=3, shape='triangular')

    assert utils.validate_partitions(numerical_data, partitions) == [True, True]


@pytest.mark.parametrize('fuzzy_type', [fs.FUZZY_SETS.t2, fs.FUZZY_SETS.gt2])
def test_type_2_variables_have_nothing_to_check(numerical_data, fuzzy_type):
    # The properties are defined over crisp memberships, so a variable that
    # answers with an interval per sample used to raise a numpy truth-value error.
    partitions = utils.construct_partitions(numerical_data, fuzzy_type,
                                            n_partitions=3, shape='triangular')

    assert utils.validate_partitions(numerical_data, partitions) == [True, True]

    with pytest.raises(NotImplementedError, match='Type-1'):
        partitions[0].validate(numerical_data['a'].values)


@pytest.mark.parametrize('fuzzy_type', [fs.FUZZY_SETS.t1, fs.FUZZY_SETS.t2])
def test_categorical_variables_are_valid_by_definition(fuzzy_type):
    frame = pd.DataFrame({'text': ['low', 'mid', 'high'] * 40,
                          'continuous': np.linspace(0, 1, 120)})
    partitions = utils.construct_partitions(frame, fuzzy_type, n_partitions=3,
                                            shape='triangular')

    # One crisp fuzzy set per category leaves nothing to check, whatever the
    # fuzzy type, so the detected categorical variable validates on its own.
    assert partitions[0].validate(frame['text'].values) is True


def test_a_mixed_partition_validates_every_variable():
    frame = pd.DataFrame({'text': ['low', 'mid', 'high'] * 40,
                          'flag': [0, 1] * 60,
                          'continuous': np.linspace(0, 1, 120)})
    partitions = utils.construct_partitions(frame, fs.FUZZY_SETS.t1, n_partitions=3,
                                            shape='triangular')

    assert utils.validate_partitions(frame, partitions) == [True, True, True]


def test_an_explicit_categorical_mask_reports_those_variables_as_valid(numerical_data):
    partitions = utils.construct_partitions(numerical_data, fs.FUZZY_SETS.t1,
                                            n_partitions=3, shape='triangular')

    assert utils.validate_partitions(numerical_data, partitions,
                                     categorical_mask=[1, 0]) == [True, True]


def _frame_with_one_broken_variable():
    """'bad' piles almost every sample on one point, which fails the properties.

    Only as a trapezoid: the same column passes every property as a triangle, so
    the shape is pinned below rather than left to the default.
    """
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        'text': rng.choice(['low', 'mid', 'high'], 400),
        'good': rng.normal(size=400),
        'bad': np.concatenate([np.full(396, 0.5), [0.0, 0.001, 0.002, 1.0]]),
    })


@pytest.mark.parametrize('categorical_mask', [None, [3, 0, 0]])
def test_results_stay_aligned_with_the_variables(categorical_mask):
    # A skipped variable used to consume its index without producing an entry, so
    # the shorter list blamed 'good' for the failure that belongs to 'bad'.
    frame = _frame_with_one_broken_variable()
    partitions = utils.construct_partitions(frame, fs.FUZZY_SETS.t1, n_partitions=3,
                                            shape='trapezoid')

    res = utils.validate_partitions(frame, partitions, categorical_mask=categorical_mask)

    assert len(res) == len(partitions)
    assert dict(zip(frame.columns, res)) == {'text': True, 'good': True, 'bad': False}
