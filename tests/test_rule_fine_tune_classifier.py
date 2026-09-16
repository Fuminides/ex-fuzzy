"""Tests for the two-stage rule fine-tuning classifier."""

import numpy as np
from sklearn.datasets import load_iris

from ex_fuzzy import classifiers
from ex_fuzzy import evolutionary_fit as evf
from ex_fuzzy import fuzzy_sets as fs
from ex_fuzzy import utils


def test_fine_tuning_with_its_own_partitions():
    X, y = load_iris(return_X_y=True)
    model = classifiers.RuleFineTuneClassifier(nRules=6, nAnts=2, tolerance=0.01)

    assert model.fit(X, y, n_gen=3, pop_size=8, random_state=0) is model
    assert model.phase1_rules is model.fl_classifier1.rule_base
    assert isinstance(model.internal_classifier(), evf.BaseFuzzyRulesClassifier)
    assert model.internal_classifier() is model.fl_classifier2
    predictions = model.predict(X)
    assert predictions.shape == y.shape
    assert set(predictions) <= {0, 1, 2}


def test_fine_tuning_with_given_partitions():
    X, y = load_iris(return_X_y=True)
    partitions = utils.construct_partitions(X, fs.FUZZY_SETS.t1)
    model = classifiers.RuleFineTuneClassifier(nRules=6, nAnts=2, linguistic_variables=partitions, expansion_factor=2)

    model.fit(X, y, n_gen=3, pop_size=8, random_state=0)
    assert model.fl_classifier1.nRules == 12
    assert model.fl_classifier2.rule_base.antecedents[0].name == partitions[0].name
    assert np.mean(model.predict(X) == y) > 1 / 3
