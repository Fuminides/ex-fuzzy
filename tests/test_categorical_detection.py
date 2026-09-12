"""Tests for the automatic detection of categorical variables."""
import numpy as np
import pandas as pd

import evolutionary_fit as evf
import fuzzy_sets as fs
import utils


def _mixed_frame(n_samples=120):
    """A frame with one variable of every kind the detection has to separate."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        'text': rng.choice(['low', 'mid', 'high'], n_samples),
        'flag': rng.integers(0, 2, n_samples),
        'boolean': rng.integers(0, 2, n_samples).astype(bool),
        'counter': rng.integers(0, 40, n_samples),
        'continuous': rng.normal(size=n_samples),
    })


def test_detects_names_flags_and_leaves_numerical_alone():
    mask = utils.detect_categorical_mask(_mixed_frame())

    assert list(mask) == [3, 2, 2, 0, 0]


def test_whole_number_threshold_is_configurable():
    frame = pd.DataFrame({'ordinal': list(range(8)) * 10})

    assert utils.detect_categorical_mask(frame)[0] == 0
    assert utils.detect_categorical_mask(frame, max_unique_numeric=8)[0] == 8


def test_detection_matches_on_a_plain_object_array():
    frame = _mixed_frame()

    assert np.array_equal(utils.detect_categorical_mask(frame),
                          utils.detect_categorical_mask(frame.values))


def test_missing_values_are_not_categories():
    frame = pd.DataFrame({
        'continuous': np.concatenate((np.linspace(0, 1, 60), np.full(60, np.nan))),
        'text': ['a'] * 60 + [None] * 60,
        'empty': np.full(120, np.nan),
    })
    mask = utils.detect_categorical_mask(frame)

    # The NaNs neither turn a continuous variable categorical nor count as a
    # category of their own, and an all-NaN variable stays numerical.
    assert list(mask) == [0, 1, 0]


def test_construct_partitions_gives_categorical_variables_a_crisp_partition():
    frame = _mixed_frame()
    partitions = utils.construct_partitions(frame, fs.FUZZY_SETS.t1, n_partitions=3)

    names = [variable.linguistic_variable_names() for variable in partitions]
    assert names[0] == ['high', 'low', 'mid']       # one fuzzy set per category
    assert names[1] == ['0', '1']
    assert names[3] == ['Low', 'Medium', 'High']    # quantile partition
    assert names[4] == ['Low', 'Medium', 'High']
    assert [variable.name for variable in partitions] == list(frame.columns)


def test_construct_partitions_detection_can_be_switched_off():
    frame = _mixed_frame().drop(columns=['text'])
    partitions = utils.construct_partitions(frame, fs.FUZZY_SETS.t1, n_partitions=3,
                                            detect_categorical=False)

    assert all(variable.linguistic_variable_names() == ['Low', 'Medium', 'High']
               for variable in partitions)


def test_all_numerical_data_is_partitioned_exactly_as_before():
    # Nothing detected has to leave no trace: the detection must not send an
    # all-numerical dataset down the categorical reordering path.
    frame = _mixed_frame()[['counter', 'continuous']]
    points = np.linspace(-3, 40, 50)

    detected = utils.construct_partitions(frame, fs.FUZZY_SETS.t1, n_partitions=3)
    disabled = utils.construct_partitions(frame, fs.FUZZY_SETS.t1, n_partitions=3,
                                          detect_categorical=False)

    assert all(np.array_equal(left.compute_memberships(points),
                              right.compute_memberships(points))
               for left, right in zip(detected, disabled))


def test_an_explicit_mask_is_used_instead_of_the_detection():
    frame = _mixed_frame().drop(columns=['text'])
    mask = np.zeros(frame.shape[1], dtype=int)
    mask[frame.columns.get_loc('counter')] = 40  # detected as numerical

    partitions = utils.construct_partitions(frame, fs.FUZZY_SETS.t1, n_partitions=3,
                                            categorical_mask=mask)
    names = [variable.linguistic_variable_names() for variable in partitions]

    assert names[frame.columns.get_loc('counter')] != ['Low', 'Medium', 'High']
    assert names[frame.columns.get_loc('flag')] == ['Low', 'Medium', 'High']


def _fit(frame, labels, **kwargs):
    classifier = evf.BaseFuzzyRulesClassifier(nRules=6, nAnts=2, n_linguistic_variables=3,
                                              fuzzy_type=fs.FUZZY_SETS.t1, tolerance=0.001,
                                              **kwargs)
    classifier.fit(frame, labels, n_gen=3, pop_size=6, random_state=0, p_value_compute=False)
    return classifier


def test_classifier_optimizes_partitions_over_mixed_types():
    frame = _mixed_frame()
    labels = pd.Series((frame['text'] == 'high').astype(int))

    classifier = _fit(frame, labels)
    antecedents = classifier.rule_base.get_antecedents()

    assert antecedents[0].linguistic_variable_names() == ['high', 'low', 'mid']
    assert len(classifier.predict(frame)) == len(frame)


def test_detection_leaves_an_all_numerical_fit_untouched():
    frame = _mixed_frame().drop(columns=['text', 'flag', 'boolean'])
    labels = pd.Series((frame['continuous'] > 0).astype(int))

    detected = _fit(frame, labels)
    disabled = _fit(frame, labels, detect_categorical=False)

    assert detected.performance == disabled.performance
    assert str(detected.rule_base) == str(disabled.rule_base)
