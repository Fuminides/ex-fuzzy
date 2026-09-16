"""Every nonconformity score type calibrates and predicts with the same quantity."""
import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from ex_fuzzy import BaseFuzzyRulesClassifier, fuzzy_sets as fs, utils
from ex_fuzzy.conformal import ConformalFuzzyClassifier, evaluate_conformal_coverage


@pytest.fixture(scope='module')
def fitted():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
    X_cal, X_test, y_cal, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=0, stratify=y_rest)
    model = BaseFuzzyRulesClassifier(nRules=8, nAnts=3, linguistic_variables=utils.construct_partitions(X_train, fs.FUZZY_SETS.t1),
                                     n_gen=10, pop_size=20, random_state=0).fit(X_train, y_train)
    return model, (X_cal, y_cal), (X_test, y_test)


@pytest.mark.parametrize('score_type', ['membership', 'association', 'entropy'])
def test_each_score_type_reaches_its_coverage(fitted, score_type):
    model, (X_cal, y_cal), (X_test, y_test) = fitted
    conformal = ConformalFuzzyClassifier(model, score_type=score_type).calibrate(X_cal, y_cal)
    coverage = evaluate_conformal_coverage(conformal, X_test, y_test, alpha=0.1)['coverage']
    # 114 test samples: allow sampling noise around the 0.9 target, but not a broken score.
    assert coverage >= 0.8


def test_test_scores_use_the_calibrated_score_type(fitted):
    model, (X_cal, y_cal), (X_test, y_test) = fitted
    entropy = ConformalFuzzyClassifier(model, score_type='entropy').calibrate(X_cal, y_cal)
    membership = ConformalFuzzyClassifier(model, score_type='membership').calibrate(X_cal, y_cal)
    entropy_scores = entropy._class_nonconformity(X_test)
    # Entropy does not depend on the candidate class; membership does.
    np.testing.assert_array_equal(entropy_scores[:, 0], entropy_scores[:, 1])
    assert not np.array_equal(membership._class_nonconformity(X_test)[:, 0], membership._class_nonconformity(X_test)[:, 1])
    # Calibration uses the very same matrix, indexed by the true class.
    np.testing.assert_array_equal(entropy._compute_nonconformity_scores(X_cal, y_cal),
                                  entropy._class_nonconformity(X_cal)[np.arange(len(y_cal)), y_cal])
