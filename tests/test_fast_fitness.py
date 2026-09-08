"""Compare optimized fitness with the unchanged full evaluator."""
import copy

import numpy as np
import pytest
from sklearn.datasets import load_iris, make_classification
from sklearn.metrics import matthews_corrcoef

import evolutionary_fit as evf
import fuzzy_sets as fs
import utils
from _fitness import _mcc, score_rulebase


@pytest.mark.parametrize('labels', [[0], [0, 1], [-1, 0, 1, 2], [10, 30, 90]])
def test_integer_mcc_matches_sklearn(labels):
    rng = np.random.default_rng(6)
    for n in (1, 2, 30, 1000):
        y = rng.choice(labels, n)
        for prediction in (y, y[::-1], np.full(n, labels[0]), rng.choice(labels, n)):
            assert _mcc(y, prediction) == matthews_corrcoef(y, prediction)


@pytest.mark.parametrize('fixed', [False, True])
@pytest.mark.parametrize('fuzzy_type', [fs.FUZZY_SETS.t1, fs.FUZZY_SETS.t2])
@pytest.mark.parametrize('ds_mode', [0, 1, 2])
@pytest.mark.parametrize('allow_unknown', [False, True])
def test_random_population_matches_reference(fixed, fuzzy_type, ds_mode, allow_unknown):
    X, y = make_classification(n_samples=180, n_features=5, n_informative=4,
                               n_redundant=0, n_classes=3, random_state=16)
    partitions = utils.construct_partitions(X, fuzzy_type) if fixed else None
    problem = evf.FitRuleBase(X, y, 12, 3, 3, fuzzy_type=fuzzy_type,
                              linguistic_variables=partitions, ds_mode=ds_mode,
                              allow_unknown=allow_unknown)
    rng = np.random.default_rng(4)
    genes = rng.integers(problem.xl.astype(int), problem.xu.astype(int) + 1,
                         size=(8, problem.n_var))
    for tolerance in (0.0, 0.1, 1.0):
        problem.tolerance = tolerance
        for x in genes:
            slow, fast = {}, {}
            problem._evaluate_slow(x, slow)
            problem._evaluate(x, fast)
            assert fast['F'] == slow['F']


@pytest.mark.parametrize('ds_mode', [0, 1, 2])
def test_pruning_and_penalties_at_exact_dominance_threshold(ds_mode):
    X, y = load_iris(return_X_y=True)
    problem = evf.FitRuleBase(X, y, 10, 2, 3, ds_mode=ds_mode)
    rng = np.random.default_rng(21)
    gene = rng.integers(problem.xl.astype(int), problem.xu.astype(int) + 1)
    base = problem._construct_ruleBase(gene, problem.fuzzy_type)
    import eval_rules
    evaluator = eval_rules.evalRuleBase(copy.deepcopy(base), X, y)
    evaluator.add_full_evaluation()
    thresholds = [r.score for r in evaluator.mrule_base.get_rules()]
    for threshold in thresholds:
        for tolerance in (threshold, np.nextafter(threshold, np.inf)):
            left, right = copy.deepcopy(base), copy.deepcopy(base)
            slow = problem.fitness_func(left, X, y, tolerance, 0.2, 0.3)
            fast = score_rulebase(right, X, y, tolerance, 0.2, 0.3)
            assert fast == slow
            assert left.get_consequents() == right.get_consequents()
            np.testing.assert_array_equal(left.get_scores(), right.get_scores())
            for a, b in zip(left.get_rules(), right.get_rules()):
                np.testing.assert_array_equal(a.antecedents, b.antecedents)
                assert a.accuracy == b.accuracy


@pytest.mark.parametrize('fixed', [False, True])
def test_seeded_fit_retains_search_result(monkeypatch, fixed):
    X, y = load_iris(return_X_y=True)
    partitions = utils.construct_partitions(X) if fixed else None
    options = dict(nRules=10, nAnts=3, linguistic_variables=partitions)
    fast = evf.BaseFuzzyRulesClassifier(**options)
    fast.fit(X, y, n_gen=5, pop_size=12, random_state=8, patience=None)
    with monkeypatch.context() as reference:
        reference.setattr(evf.FitRuleBase, '_evaluate', evf.FitRuleBase._evaluate_slow)
        slow = evf.BaseFuzzyRulesClassifier(**options)
        slow.fit(X, y, n_gen=5, pop_size=12, random_state=8, patience=None)
    assert fast.performance == slow.performance
    np.testing.assert_array_equal(fast.optimization_result_['X'], slow.optimization_result_['X'])
    assert fast.rule_base.get_consequents() == slow.rule_base.get_consequents()
    np.testing.assert_array_equal(fast.predict(X), slow.predict(X))
    for a, b in zip(fast.rule_base.get_rulebase_matrix(), slow.rule_base.get_rulebase_matrix()):
        np.testing.assert_array_equal(a, b)


def test_fast_path_evaluates_firing_once(monkeypatch):
    X, y = load_iris(return_X_y=True)
    problem = evf.FitRuleBase(X, y, 10, 2, 3)
    rng = np.random.default_rng(42)
    gene = rng.integers(problem.xl.astype(int), problem.xu.astype(int) + 1)
    base = problem._construct_ruleBase(gene, problem.fuzzy_type)
    original = base.compute_firing_strenghts
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(base, 'compute_firing_strenghts', counted)
    score_rulebase(base, X, y, 0.0, 0.0, 0.0)
    assert len(calls) == 1
