"""Final-fit pruning preparation must equal the former full pre-purge pass."""
import copy

import numpy as np
import pytest
from sklearn.datasets import load_iris

import evolutionary_fit as evf
import eval_rules
import fuzzy_sets as fs
import utils


def _old_finalization(rule_base, X, y, tolerance):
    evaluator = eval_rules.evalRuleBase(rule_base, X, y)
    evaluator.add_full_evaluation()
    rule_base.purge_rules(tolerance)
    evaluator.add_full_evaluation()
    return evaluator


def _new_finalization(rule_base, X, y, tolerance):
    evaluator = eval_rules.evalRuleBase(rule_base, X, y)
    evaluator.add_rule_weights()
    evaluator.add_classification_metrics()
    rule_base.purge_rules(tolerance)
    evaluator.add_full_evaluation()
    return evaluator


def _assert_same_finalization(left_base, left_eval, right_base, right_eval):
    assert left_base.get_consequents() == right_base.get_consequents()
    np.testing.assert_array_equal(left_base.get_scores(), right_base.get_scores())
    for left, right in zip(left_base.get_rules(), right_base.get_rules()):
        np.testing.assert_array_equal(left.antecedents, right.antecedents)
        assert left.accuracy == right.accuracy
        assert left.support == right.support
        assert left.confidence == right.confidence
    assert left_eval.mcc == right_eval.mcc
    assert left_eval.acc == right_eval.acc


@pytest.mark.parametrize("kind", [fs.FUZZY_SETS.t1, fs.FUZZY_SETS.t2])
@pytest.mark.parametrize("tolerance", [0.0, 0.1, 1.1])
def test_preprune_minimal_work_matches_full_reference(kind, tolerance):
    X, y = load_iris(return_X_y=True)
    partitions = utils.construct_partitions(X, kind)
    problem = evf.FitRuleBase(X, y, 8, 3, 3, linguistic_variables=partitions,
                              fuzzy_type=kind)
    gene = np.random.default_rng(31).integers(problem.xl.astype(int),
                                               problem.xu.astype(int) + 1)
    left = problem._construct_ruleBase(gene.copy(), kind)
    right = copy.deepcopy(left)
    left_eval = _old_finalization(left, X, y, tolerance)
    right_eval = _new_finalization(right, X, y, tolerance)
    _assert_same_finalization(left, left_eval, right, right_eval)


@pytest.mark.parametrize("kind", [fs.FUZZY_SETS.t1, fs.FUZZY_SETS.t2])
def test_preprune_minimal_work_handles_no_rules(kind):
    X, y = load_iris(return_X_y=True)
    partitions = utils.construct_partitions(X, kind)
    problem = evf.FitRuleBase(X, y, 4, 2, 3, linguistic_variables=partitions,
                              fuzzy_type=kind)
    gene = np.zeros(problem.n_var, dtype=int)
    gene[2 * problem.nRules * problem.nAnts:] = -1
    left = problem._construct_ruleBase(gene.copy(), kind)
    right = copy.deepcopy(left)
    left_eval = _old_finalization(left, X, y, 0.1)
    right_eval = _new_finalization(right, X, y, 0.1)
    assert not left.get_rules() and not right.get_rules()
    assert left_eval.mcc == right_eval.mcc == 0.0


@pytest.mark.parametrize("kind", [fs.FUZZY_SETS.t1, fs.FUZZY_SETS.t2])
def test_seeded_fit_finalizes_global_metrics_once(monkeypatch, kind):
    X, y = load_iris(return_X_y=True)
    partitions = utils.construct_partitions(X, kind)
    calls = []
    original = eval_rules.evalRuleBase.classification_eval

    def counted(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(eval_rules.evalRuleBase, "classification_eval", counted)
    model = evf.BaseFuzzyRulesClassifier(
        nRules=6, nAnts=2, linguistic_variables=partitions,
        fuzzy_type=kind, tolerance=0.1)
    model.fit(X, y, n_gen=2, pop_size=6, random_state=19, patience=None)
    assert len(calls) == 1
    assert hasattr(model.eval_performance, "mcc")
    # Reconstruct the selected fixed-partition chromosome and apply the former
    # finalization sequence as an oracle for the actual seeded fit result.
    oracle_problem = evf.FitRuleBase(
        X, y, 6, 2, 3, linguistic_variables=partitions, fuzzy_type=kind)
    oracle_base = oracle_problem._construct_ruleBase(
        model.optimization_result_["X"].copy(), kind)
    oracle_eval = _old_finalization(oracle_base, X, y, 0.1)
    _assert_same_finalization(model.rule_base, model.eval_performance,
                              oracle_base, oracle_eval)
