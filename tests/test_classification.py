
import numpy as np
import pandas as pd
import math
import sys
sys.path.append('../ex_fuzzy/')
sys.path.append('./ex_fuzzy/')

import ex_fuzzy

sample_size = 1000

def test_random_classification(fs_type=ex_fuzzy.fuzzy_sets.FUZZY_SETS.t1):
    global sample_size

    sample = np.random.random_sample((sample_size, 5))
    targets = np.random.randint(0, 2, sample_size)

    model = ex_fuzzy.evolutionary_fit.BaseFuzzyRulesClassifier(10, 3, fs_type, verbose=False, tolerance=0.0, n_linguist_variables=3)
    model.fit(sample, targets)
    predictions = model.predict(sample)
    assert math.isclose(np.mean(np.equal(predictions, targets)), 0.5, abs_tol=0.1)


def test_random_classification_precomputed(fs_type=ex_fuzzy.fuzzy_sets.FUZZY_SETS.t1):
    global sample_size

    sample = np.random.random_sample((sample_size, 5))
    targets = np.random.randint(0, 2, sample_size)
    vl_partitions = ex_fuzzy.utils.construct_partitions(sample, fs_type)
    model = ex_fuzzy.evolutionary_fit.BaseFuzzyRulesClassifier(10, 3, verbose=False, tolerance=0.0, linguistic_variables=vl_partitions)
    model.fit(sample, targets)
    predictions = model.predict(sample)
    assert math.isclose(np.mean(np.equal(predictions, targets)), 0.5, abs_tol=0.1)


def test_random_classification_t2():
    test_random_classification(ex_fuzzy.fuzzy_sets.FUZZY_SETS.t2)


def test_random_classification_t2_precomputed():
    test_random_classification_precomputed(ex_fuzzy.fuzzy_sets.FUZZY_SETS.t2)


def test_random_classification_gt2():
    test_random_classification(ex_fuzzy.fuzzy_sets.FUZZY_SETS.t2)


def test_random_classification_gt2_precomputed():
    test_random_classification_precomputed(ex_fuzzy.fuzzy_sets.FUZZY_SETS.t2)

if __name__ == '__main__':
    test_random_classification_t2()
    test_random_classification_t2_precomputed()
    test_random_classification_gt2()
    test_random_classification_gt2_precomputed()
