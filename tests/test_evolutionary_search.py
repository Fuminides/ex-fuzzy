"""Tests for rule selection from a candidate pool."""

from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

from ex_fuzzy import evolutionary_fit as evf
from ex_fuzzy import evolutionary_search as es
from ex_fuzzy import fuzzy_sets as fs
from ex_fuzzy import rules as rl
from ex_fuzzy import utils


@pytest.fixture(scope='module')
def iris():
    return load_iris(return_X_y=True)


def _candidates(X, fuzzy_type=fs.FUZZY_SETS.t1, rule_base_class=rl.RuleBaseT1):
    '''Rules in class-major order: 0 and 1 for setosa, 2 for versicolor, 3 and 4 for virginica.'''
    partitions = utils.construct_partitions(X, fuzzy_type)
    rule_lists = [[rl.RuleSimple([-1, -1, 0, -1]), rl.RuleSimple([0, -1, -1, -1])],
                  [rl.RuleSimple([-1, -1, 1, -1])],
                  [rl.RuleSimple([-1, -1, 2, -1]), rl.RuleSimple([2, -1, -1, -1])]]
    return rl.MasterRuleBase([rule_base_class(partitions, rule_list) for rule_list in rule_lists])


def test_the_gene_chooses_the_rules(iris):
    X, y = iris
    candidates = _candidates(X)
    problem = es.ExploreRuleBases(pd.DataFrame(X, columns=list('abcd')), y, nRules=3, n_classes=3,
                                  candidate_rules=candidates)
    assert problem.var_names == ['a', 'b', 'c', 'd']

    one_per_class = problem._construct_ruleBase(np.array([0, 2, 3]), fs.FUZZY_SETS.t1)
    assert [len(rule_base) for rule_base in one_per_class] == [1, 1, 1]
    assert one_per_class[2].rules[0] is candidates[2].rules[0]
    assert one_per_class.get_consequents_names() == [0, 1, 2]

    # Classes without selected rules keep an empty rule base in their place.
    only_virginica = problem._construct_ruleBase(np.array([4, 4, 3]), fs.FUZZY_SETS.t1)
    assert [len(rule_base) for rule_base in only_virginica] == [0, 0, 2]

    good, bad = {}, {}
    problem._evaluate(np.array([0, 2, 3]), good)
    problem._evaluate(np.array([4, 4, 4]), bad)
    assert good['F'] < 0.5
    assert bad['F'] > good['F']


def test_size_penalties_are_added_to_the_fitness(iris):
    X, y = iris
    problem = es.ExploreRuleBases(X, y, nRules=3, n_classes=3, candidate_rules=_candidates(X))
    rule_base = problem._construct_ruleBase(np.array([0, 2, 3]), fs.FUZZY_SETS.t1)

    plain = problem.fitness_func(rule_base, problem.X, y, 0.0)
    penalised = problem.fitness_func(rule_base, problem.X, y, 0.0, alpha=0.5, beta=0.5)
    assert penalised > plain


@pytest.mark.parametrize('fuzzy_type, rule_base_class', [(fs.FUZZY_SETS.t2, rl.RuleBaseT2), (fs.FUZZY_SETS.gt2, rl.RuleBaseGT2)])
def test_type_2_candidates(iris, fuzzy_type, rule_base_class):
    X, y = iris
    problem = es.ExploreRuleBases(X, y, nRules=3, n_classes=3,
                                  candidate_rules=_candidates(X, fuzzy_type, rule_base_class))

    rule_base = problem._construct_ruleBase(np.array([0, 2, 3]), fuzzy_type)
    assert all(isinstance(class_rules, rule_base_class) for class_rules in rule_base)
    out = {}
    problem._evaluate(np.array([0, 2, 3]), out)
    assert out['F'] < 1


def test_parallel_runner(iris):
    X, y = iris
    with ThreadPool(2) as pool:
        runner = evf.StarmapParallelization(pool.starmap)
        problem = es.ExploreRuleBases(X, y, nRules=3, n_classes=3, candidate_rules=_candidates(X), thread_runner=runner)
    assert problem.elementwise_runner is runner


def test_classifier_selects_rules_from_candidates(iris):
    X, y = iris
    classifier = evf.BaseFuzzyRulesClassifier(nRules=3, nAnts=2, verbose=False)
    classifier.fit(X, y, candidate_rules=_candidates(X), n_gen=10, pop_size=10, random_state=0)
    assert np.mean(classifier.predict(X) == y) > 0.8
