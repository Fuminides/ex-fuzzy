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
    assert math.isclose(partitions[0](0.01)[0][0], 0.8, abs_tol=tolerate), 'Not the correct partition'
    assert math.isclose(partitions[0](0.01)[0][1], 1, abs_tol=tolerate), 'Not the correct partition'


def test_construct_partitions_gt2():
    sample = np.random.random_sample((sample_size, 1))
    partitions = ex_fuzzy.utils.construct_partitions(sample, ex_fuzzy.fuzzy_sets.FUZZY_SETS.gt2)
    assert len(partitions[0]) == 3, 'Not the correct number of partitions'
    assert math.isclose(np.mean(partitions[0][0].alpha_reduction(partitions[0](0.1)[0])), 0.9, abs_tol=tolerate), 'Not the correct partition'


def test_temporal_conditional():
    sample_size = 10
    sample = np.random.random_sample((sample_size, ))

    first_temp = int(sample_size / 2)

    sample[:first_temp] = sample[:first_temp] * 0.5
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