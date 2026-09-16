"""Inference and evaluation compute memberships and firing once, with unchanged results."""
import contextlib
import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

from ex_fuzzy import eval_rules
from ex_fuzzy import evolutionary_fit as evf
from ex_fuzzy import fuzzy_sets as fs
from ex_fuzzy import rules
from ex_fuzzy import utils

KINDS = [fs.FUZZY_SETS.t1, fs.FUZZY_SETS.t2, fs.FUZZY_SETS.gt2]


def _random_master(kind, seed=5, ds_mode=0, allow_unknown=False):
    X, y = load_iris(return_X_y=True)
    partitions = utils.construct_partitions(X, kind)
    problem = evf.FitRuleBase(X, y, 8, 3, 3, linguistic_variables=partitions, fuzzy_type=kind,
                              ds_mode=ds_mode, allow_unknown=allow_unknown)
    gene = np.random.default_rng(seed).integers(problem.xl.astype(int), problem.xu.astype(int) + 1)
    master = problem._construct_ruleBase(gene, kind)
    eval_rules.evalRuleBase(master, X, y).add_rule_weights()
    return master, X, y


def _reference_firing(master, X):
    """Every rule base evaluates its own antecedents, as before the shared computation."""
    parts = [base.compute_rule_antecedent_memberships(X) for base in master.rule_bases]
    parts = [part for part in parts if part.size > 0]
    return np.concatenate(parts, axis=1) if parts else np.zeros((X.shape[0], 0))


def _reference_predict(master, X, out_class_names):
    """The former per-sample loops over the winning rules."""
    firing = _reference_firing(master, X)
    association = master.compute_association_degrees(X, firing_strengths=firing)
    winners = np.argmax(association, axis=1)
    if master.allow_unknown:
        winners[np.max(association, axis=1) == 0.0] = -1
    if out_class_names:
        consequents = sum([[master.consequent_names[ix]] * len(master[ix].rules)
                           for ix in range(len(master.rule_bases))], [])
        return np.array([consequents[w] if w != -1 else 'Unknown' for w in winners]), winners
    consequents = master.get_consequents()
    result = np.zeros(X.shape[0])
    for ix, winner in enumerate(winners):
        result[ix] = consequents[winner] if winner != -1 else -1
    return result, winners


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('frame', [False, True])
def test_firing_matches_per_rule_base_reference(kind, frame):
    master, X, _ = _random_master(kind)
    data = pd.DataFrame(X) if frame else X
    np.testing.assert_array_equal(master.compute_firing_strengths(data), _reference_firing(master, X))


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('ds_mode', [0, 1, 2])
@pytest.mark.parametrize('out_class_names', [False, True])
def test_predictions_match_reference(kind, ds_mode, out_class_names):
    master, X, _ = _random_master(kind, ds_mode=ds_mode)
    if ds_mode == 2:
        for index, rule in enumerate(master.get_rules()):
            rule.weight = 0.25 + 0.05 * index
    master.rename_cons(['a', 'b', 'c'])
    expected, winners = _reference_predict(master, X, out_class_names)

    predicted = master.winning_rule_predict(X, out_class_names=out_class_names)
    np.testing.assert_array_equal(predicted, expected)
    explained = master.explainable_predict(X, out_class_names=out_class_names)
    np.testing.assert_array_equal(explained[0], expected)
    np.testing.assert_array_equal(explained[1], winners)
    if not out_class_names:
        assert predicted.dtype.kind in 'iu'


@pytest.mark.parametrize('names', [None, ['low', 'high']])
def test_unknown_predictions_keep_their_markers(names):
    X = np.array([[0.0], [1.0]])
    partition = fs.fuzzyVariable('x', [fs.FS('low', [0, 0, 0.25, 0.5], [0, 1]),
                                       fs.FS('high', [0.5, 0.75, 1, 1], [0, 1])])
    rule = rules.RuleSimple([0])
    rule.score = 1.0
    master = rules.MasterRuleBase([rules.RuleBaseT1([partition], [rule]), rules.RuleBaseT1([partition], [])],
                                  names, allow_unknown=True)
    rule.boot_confidence_interval = np.array([0.6, 0.8])

    np.testing.assert_array_equal(master.winning_rule_predict(X), [0, -1])
    np.testing.assert_array_equal(
        master.winning_rule_predict(X, out_class_names=True), ['low' if names else 0, 'Unknown'])
    named, winners, association, confidence = master.explainable_predict(X, out_class_names=True)
    np.testing.assert_array_equal(winners, [0, -1])
    # A -1 winner selects the last interval, as the former list indexing did.
    np.testing.assert_array_equal(confidence, np.array([[0.6, 0.8], [0.0, 0.0]]))


@pytest.mark.parametrize('kind', KINDS)
def test_predict_evaluates_memberships_and_firing_once(kind, monkeypatch):
    master, X, _ = _random_master(kind)
    memberships, firings = [], []
    original_memberships = fs.fuzzyVariable.compute_memberships
    original_firing = rules.RuleBase.compute_rule_antecedent_memberships

    def counted_memberships(self, x):
        memberships.append(self.name)
        return original_memberships(self, x)

    def counted_firing(self, x, scaled=False, antecedents_memberships=None):
        firings.append(antecedents_memberships is not None)
        return original_firing(self, x, scaled, antecedents_memberships)

    monkeypatch.setattr(fs.fuzzyVariable, 'compute_memberships', counted_memberships)
    monkeypatch.setattr(rules.RuleBase, 'compute_rule_antecedent_memberships', counted_firing)
    master.winning_rule_predict(X)
    # One membership evaluation per variable, not one per variable and rule base.
    assert len(memberships) == X.shape[1]
    # One firing pass over the rule bases, every one of them fed the shared memberships.
    assert len(firings) <= len(master.rule_bases) and all(firings)


def test_rule_bases_with_their_own_antecedents_evaluate_themselves():
    X, _ = load_iris(return_X_y=True)
    first = utils.construct_partitions(X, fs.FUZZY_SETS.t1)
    second = utils.construct_partitions(X, fs.FUZZY_SETS.t1, n_partitions=5)
    master = rules.MasterRuleBase([rules.RuleBaseT1(first, [rules.RuleSimple([0, -1, -1, -1])]),
                                   rules.RuleBaseT1(second, [rules.RuleSimple([4, -1, -1, -1])])])
    assert master._shared_antecedent_memberships(X) is None
    np.testing.assert_array_equal(master.compute_firing_strengths(X), _reference_firing(master, X))


@pytest.mark.parametrize('kind', [fs.FUZZY_SETS.t1, fs.FUZZY_SETS.t2])
def test_evaluator_methods_compute_firing_once_and_exactly(kind, monkeypatch):
    master, X, y = _random_master(kind, seed=11)
    reference = copy.deepcopy(master)

    @contextlib.contextmanager
    def no_scope(self):
        yield

    # Reference values come from the unscoped, repeated computations.
    with monkeypatch.context() as patch:
        patch.setattr(rules.MasterRuleBase, '_firing_cache_scope', no_scope)
        expected = eval_rules.evalRuleBase(reference, X, y)
        expected.add_full_evaluation()

    calls = []
    original = rules.compute_antecedents_memberships

    def counted(antecedents, x):
        calls.append(1)
        return original(antecedents, x)

    monkeypatch.setattr(rules, 'compute_antecedents_memberships', counted)
    evaluator = eval_rules.evalRuleBase(master, X, y)
    evaluator.add_full_evaluation()
    assert len(calls) == 1
    for method in ('add_rule_weights', 'add_classification_metrics', 'classification_eval'):
        del calls[:]
        getattr(evaluator, method)()
        assert len(calls) == 1
    assert not hasattr(master, '_scoped_firing_cache')

    for actual, wanted in zip(master.get_rules(), reference.get_rules()):
        np.testing.assert_array_equal(actual.score, wanted.score)
        np.testing.assert_array_equal(actual.support, wanted.support)
        np.testing.assert_array_equal(actual.confidence, wanted.confidence)
        assert actual.accuracy == wanted.accuracy
    assert evaluator.mcc == expected.mcc and evaluator.acc == expected.acc


def test_firing_cache_scope_is_reentrant(monkeypatch):
    master, X, _ = _random_master(fs.FUZZY_SETS.t2, seed=13)
    calls = []
    original = rules._gather_rule_firing

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(rules, '_gather_rule_firing', counted)
    with master._firing_cache_scope():
        first = master.compute_firing_strengths(X)
        with master._firing_cache_scope():
            assert master.compute_firing_strengths(X) is first
        # Leaving the inner scope keeps the outer cache.
        assert master.compute_firing_strengths(X) is first
        assert len(calls) == 1
    assert not hasattr(master, '_scoped_firing_cache')
    assert master.compute_firing_strengths(X) is not first


def test_misspelled_firing_method_is_a_deprecated_alias():
    master, X, _ = _random_master(fs.FUZZY_SETS.t1)
    with pytest.warns(DeprecationWarning, match='compute_firing_strengths'):
        legacy = master.compute_firing_strenghts(X)
    np.testing.assert_array_equal(legacy, master.compute_firing_strengths(X))
