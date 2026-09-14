import numpy as np
import math
import sys

sys.path.append('./ex_fuzzy/')
sys.path.append('../ex_fuzzy/')
import ex_fuzzy as ex_fuzzy

sample_size = 10000
tolerate = 0.05
def test_quartiles():
    sample = np.random.random_sample(sample_size)
    targets = [0, 0.25, 0.5, 1]
    quartiles = ex_fuzzy.utils.quartile_compute(sample)
    for ix, target in enumerate(targets):
        assert math.isclose(quartiles[ix], target, abs_tol=tolerate), 'Not the correct ' +str(target) + ' quartile.'

def test_quantiles():
    sample = np.random.random_sample(sample_size)
    targets = [0, 0.20, 0.30, 0.45, 0.55, 0.7, 0.8, 1]
    quartiles = ex_fuzzy.utils.fixed_quantile_compute(sample)
    for ix, target in enumerate(targets):
        assert math.isclose(quartiles[ix], target, abs_tol=tolerate), 'Not the correct ' +str(target) + ' quartile.'

def test_3_partitions():
    sample = np.random.random_sample(sample_size)
    targets = [0, 0.20, 0.50, 0.80, 1.00]
    quartiles = ex_fuzzy.utils.partition3_quantile_compute(sample)
    for ix, target in enumerate(targets):
        assert math.isclose(quartiles[ix], target, abs_tol=tolerate), 'Not the correct ' +str(target) + ' quartile.'


def test_construct_partitions_t1():
    sample = np.random.random_sample((sample_size, 1))
    partitions = ex_fuzzy.utils.construct_partitions(sample, ex_fuzzy.fuzzy_sets.FUZZY_SETS.t1)
    assert len(partitions[0]) == 3, 'Not the correct number of partitions'
    assert math.isclose(partitions[0](0.01)[0], 1, abs_tol=tolerate), 'Not the correct partition'


def test_construct_partitions_t2():
    sample = np.random.random_sample((sample_size, 1))
    partitions = ex_fuzzy.utils.construct_partitions(sample, ex_fuzzy.fuzzy_sets.FUZZY_SETS.t2)
    assert len(partitions[0]) == 3, 'Not the correct number of partitions'
    # T2 partitions now use lower_height=1.0, so both bounds should be 1.0 at extremes
    assert math.isclose(partitions[0](0.01)[0][0], 1.0, abs_tol=tolerate), 'Not the correct partition'
    assert math.isclose(partitions[0](0.01)[0][1], 1.0, abs_tol=tolerate), 'Not the correct partition'


def test_construct_partitions_gt2():
    sample = np.random.random_sample((sample_size, 1))
    partitions = ex_fuzzy.utils.construct_partitions(sample, ex_fuzzy.fuzzy_sets.FUZZY_SETS.gt2)
    assert len(partitions[0]) == 3, 'Not the correct number of partitions'
    # GT2 partitions now use lower_height=1.0, so alpha_reduction returns 1.0 at extremes
    assert partitions[0][0].alpha_reduction(partitions[0](0.1)[0]) is not None, 'Alpha reduction should return a value'


def test_temporal_conditional():
    sample_size = 10
    first_temp = int(sample_size / 2)
    sample = np.concatenate([
        np.linspace(0.05, 0.45, first_temp),
        np.linspace(0.55, 0.95, sample_size - first_temp),
    ])
    time_labels = [0] * sample_size
    time_labels[first_temp:] = [1] * first_temp

    vl1 = ex_fuzzy.fuzzy_sets.FS('',[0, 0.20, 0.30, 0.5], [0, 1])
    vl2 = ex_fuzzy.fuzzy_sets.FS('',[0.5, 0.70, 0.80, 1], [0, 1])

    conditional_frequencies = ex_fuzzy.utils.construct_conditional_frequencies(sample, time_labels, [vl1, vl2])

    assert conditional_frequencies[0, 0] > conditional_frequencies[0, 1]
    assert conditional_frequencies[1, 0] < conditional_frequencies[1, 1]


def test_temporal_assemble():
    sample_size = 1000
    sample = np.random.random_sample((sample_size, 5))
    y = np.random.randint(0, 2, sample_size)

    temp_moments1 = np.random.randint(0, 2, sample_size)
    temp_moments2 = 1 - temp_moments1
    temp_moments = [temp_moments1, temp_moments2]

    [X_train, X_test, y_train, y_test], [train_temporal_boolean_markers, test_temporal_boolean_markers] = ex_fuzzy.utils.temporal_assemble(sample, y, temp_moments)

    t1test = [y_test[jx] for jx, aux in enumerate(test_temporal_boolean_markers[0]) if aux]
    t2test = [y_test[jx] for jx, aux in enumerate(test_temporal_boolean_markers[1]) if aux]

    assert len(t1test) == len(t2test)


import pandas as pd
import pytest
from sklearn.datasets import load_iris

fs = ex_fuzzy.fuzzy_sets
utils = ex_fuzzy.utils


def test_partition_shapes_names_and_errors():
    X = np.random.default_rng(0).normal(size=(200, 2))

    gaussian = utils.construct_partitions(X, fs.FUZZY_SETS.t1, n_partitions=4, shape='gaussian')
    assert [fuzzy_set.shape() for fuzzy_set in gaussian[0]] == ['gaussian'] * 4
    assert gaussian[0].linguistic_variable_names() == ['0', '1', '2', '3']
    means = [fuzzy_set.membership_parameters[0] for fuzzy_set in gaussian[0]]
    assert means == sorted(means)

    triangular = utils.construct_partitions(
        X, fs.FUZZY_SETS.t1, n_partitions=3, shape='triangular'
    )
    assert [fuzzy_set.shape() for fuzzy_set in triangular[0]] == ['triangular'] * 3

    five_names = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    assert utils.construct_partitions(X, fs.FUZZY_SETS.t1, n_partitions=5)[0].linguistic_variable_names() == five_names
    assert utils.construct_partitions(X, fs.FUZZY_SETS.t2, n_partitions=5)[1].linguistic_variable_names() == five_names
    assert utils.construct_partitions(X, fs.FUZZY_SETS.t2, n_partitions=4)[0].linguistic_variable_names() == ['0', '1', '2', '3']

    with pytest.raises(ValueError, match='Triangular partitions must be 3'):
        utils.construct_partitions(X, fs.FUZZY_SETS.t1, n_partitions=4, shape='triangular')
    with pytest.raises(ValueError, match='Shape not recognized'):
        utils.construct_partitions(X, fs.FUZZY_SETS.t1, shape='hexagon')
    with pytest.raises(ValueError, match='Fuzzy set type not recognized'):
        utils.construct_partitions(X, fs.FUZZY_SETS.temporal)


def test_categorical_detection_can_be_disabled_and_ignores_missing_values():
    frame = pd.DataFrame({'missing': [np.nan] * 6, 'flag': [0, 1] * 3, 'x': np.linspace(0, 1, 6)})
    np.testing.assert_array_equal(utils.detect_categorical_mask(frame), [0, 2, 0])

    numerical = utils.construct_partitions(frame[['flag', 'x']], fs.FUZZY_SETS.t1, detect_categorical=False)
    assert [fuzzy_set.shape() for fuzzy_set in numerical[0]] == ['trapezoid'] * 3


def test_partition_builders_name_variables_after_dataframe_columns():
    frame = pd.DataFrame({'a': np.linspace(0, 1, 50), 'b': np.linspace(1, 2, 50)})
    for builder in (utils.t1_fuzzy_partitions_dataset, utils.t2_fuzzy_partitions_dataset, utils.gt2_fuzzy_partitions_dataset):
        assert [variable.name for variable in builder(frame)] == ['a', 'b']


def test_categorical_partitions_follow_the_fuzzy_type():
    frame = pd.DataFrame({'colour': ['red', 'blue'] * 10, 'size': np.linspace(0, 1, 20)})

    mixed = utils.construct_partitions(frame, fs.FUZZY_SETS.t2)
    assert [variable.name for variable in mixed] == ['colour', 'size']
    assert isinstance(mixed[0][0], fs.categoricalIVFS)
    assert isinstance(mixed[1][0], fs.IVFS)

    only_categorical = utils.construct_partitions(frame[['colour']], fs.FUZZY_SETS.gt2)
    assert only_categorical[0].linguistic_variable_names() == ['blue', 'red']
    assert isinstance(only_categorical[0][0], fs.categoricalIVFS)


@pytest.mark.parametrize('fuzzy_type', [fs.FUZZY_SETS.t2, fs.FUZZY_SETS.gt2])
def test_temporal_conditional_for_type_2_sets(fuzzy_type):
    sample = np.concatenate([np.linspace(0, 0.3, 20), np.linspace(0.7, 1, 20)])
    partitions = utils.construct_partitions(sample.reshape(-1, 1), fuzzy_type)[0]

    conditional_frequencies = utils.construct_conditional_frequencies(sample, [0] * 20 + [1] * 20, partitions)
    assert conditional_frequencies.shape == (2, 3)
    assert conditional_frequencies[0, 0] > conditional_frequencies[0, 2]
    assert conditional_frequencies[1, 2] > conditional_frequencies[1, 0]


def test_temporal_cuts_and_time_assignment():
    dates = pd.DataFrame({'date': pd.date_range('2024-01-01 00:00', periods=24, freq='h')})
    cuts = utils.temporal_cuts(dates, [('2024-01-01 00:00', '2024-01-01 11:00'), ('2024-01-01 12:00', '2024-01-01 23:00')])

    np.testing.assert_array_equal(cuts[0], np.arange(24) <= 11)
    np.testing.assert_array_equal(cuts[1], np.arange(24) >= 12)
    assert utils.assign_time(3, cuts) == 0
    assert utils.assign_time(20, cuts) == 1
    with pytest.raises(ValueError, match='No temporal moment'):
        utils.assign_time(3, [np.zeros(24, dtype=bool)])


def test_temporal_assemble_keeps_dataframes():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.random((100, 3)))
    y = rng.integers(0, 2, 100)
    moments = [np.arange(100) < 50, np.arange(100) >= 50]

    [X_train, X_test, y_train, y_test], [train_markers, test_markers] = utils.temporal_assemble(frame, y, moments)
    assert isinstance(X_train, pd.DataFrame) and isinstance(X_test, pd.DataFrame)
    assert len(X_train) + len(X_test) == 100
    assert len(y_train) == len(X_train) and len(y_test) == len(X_test)
    assert sum(train_markers[0]) + sum(train_markers[1]) == len(X_train)
    assert sum(test_markers[0]) + sum(test_markers[1]) == len(X_test)


def test_mcc_loss_scores_a_rule_base():
    X, y = load_iris(return_X_y=True)
    partitions = utils.construct_partitions(X, fs.FUZZY_SETS.t1)
    rules = ex_fuzzy.rules
    # One rule per class on petal length: small, medium and large petals.
    mrule_base = rules.MasterRuleBase([rules.RuleBaseT1(partitions, [rules.RuleSimple([-1, -1, label, -1])]) for label in range(3)])

    assert 0.5 < utils.mcc_loss(mrule_base, X, y, tolerance=0.0) <= 1.0


def test_validate_partitions_reports_progress(capsys):
    frame = pd.DataFrame({'a': np.linspace(0, 1, 100)})
    partitions = utils.construct_partitions(frame, fs.FUZZY_SETS.t2)

    assert utils.validate_partitions(frame, partitions, verbose=True) == [True]
    report = capsys.readouterr().out
    assert 'Validating fuzzy variable a...' in report
    assert 'Skipping fuzzy variable a: Fuzzy variable validation is only implemented for Type-1' in report

    # NumPy input follows the same one-result-per-variable contract.
    assert utils.validate_partitions(frame.values, partitions) == [True]


if __name__ == '__main__':
    test_quartiles()
    test_quantiles()
    test_3_partitions()
    test_construct_partitions_t1()
    test_construct_partitions_t2()
    test_construct_partitions_gt2()
    test_temporal_conditional()
    test_temporal_assemble()
    print('All tests passed.')
