"""Correctness checks for the non-production array decoder prototype."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

import evolutionary_fit as evf
import eval_rules
import fuzzy_sets as fs
import rules
import utils


PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "prototype_array_decoder.py"
SPEC = importlib.util.spec_from_file_location("array_decoder_prototype", PATH)
prototype = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prototype)


@pytest.mark.parametrize("kind", [fs.FUZZY_SETS.t1, fs.FUZZY_SETS.t2])
@pytest.mark.parametrize("ds_mode", [0, 2])
def test_fixed_decoder_matches_constructed_rulebase(kind, ds_mode):
    X = np.array([[0.0, 0.2, 0.8], [0.3, 0.7, 0.4], [0.8, 0.1, 0.6]])
    y = np.array([0, 1, 1])
    lvs = utils.construct_partitions(X, kind)
    problem = evf.FitRuleBase(X, y, nRules=5, nAnts=3, n_classes=2,
                              linguistic_variables=lvs, fuzzy_type=kind,
                              ds_mode=ds_mode)
    fourth = 2 * problem.nRules * problem.nAnts
    gene = np.zeros(problem.n_var, dtype=int)
    # Rule 0: repeated feature 0; final slot must overwrite and term clips.
    gene[:3] = [0, 1, 0]
    gene[problem.nRules * problem.nAnts:fourth][:3] = [0, 1, 99]
    # Rule 1 duplicates rule 0 in class 0 and must be dropped.
    gene[3:6] = [0, 1, 0]
    gene[problem.nRules * problem.nAnts:fourth][3:6] = [0, 1, 99]
    # Rule 2 has a disabled consequent; Rule 3 belongs to class 1.
    gene[6:9] = [2, 2, 1]
    gene[problem.nRules * problem.nAnts:fourth][6:9] = [1, -1, -1]
    gene[9:12] = [2, 1, 0]
    gene[problem.nRules * problem.nAnts:fourth][9:12] = [0, -1, 1]
    gene[fourth:fourth + problem.nRules] = [0, 0, -1, 1, -1]
    if ds_mode == 2:
        gene[fourth + problem.nRules:fourth + 2 * problem.nRules] = [20, 80, 30, 65, 10]

    decoded = prototype.decode_fixed_partitions(
        gene, n_rules=problem.nRules, n_ants=problem.nAnts,
        n_features=X.shape[1], n_classes=problem.n_classes,
        n_lv_possible=problem.n_lv_possible, ds_mode=ds_mode)
    reference = problem._construct_ruleBase(gene.copy(), kind)
    expected_ants = np.asarray([rule.antecedents for rule in reference.get_rules()])
    np.testing.assert_array_equal(decoded.antecedents, expected_ants)
    np.testing.assert_array_equal(decoded.consequents, reference.get_consequents())
    np.testing.assert_array_equal(decoded.class_counts,
                                  [len(base.rules) for base in reference.rule_bases])
    if ds_mode == 2:
        np.testing.assert_allclose(decoded.weights, reference.get_weights())
    else:
        np.testing.assert_array_equal(decoded.weights, np.ones(len(expected_ants)))


def test_pruning_and_complexity_match_reference_formulae():
    X = np.array([[0.0, 0.2], [0.4, 0.8], [0.9, 0.1]])
    y = np.array([0, 1, 1])
    lvs = utils.construct_partitions(X, fs.FUZZY_SETS.t1)
    master = rules.MasterRuleBase([
        rules.RuleBaseT1(lvs, [rules.RuleSimple([0, -1]), rules.RuleSimple([-1, 1])]),
        rules.RuleBaseT1(lvs, [rules.RuleSimple([1, 0])]),
    ])
    all_rules = master.get_rules()
    scores = np.array([0.2, 0.1, 0.5])
    accuracies = np.array([1.0, 1.0, 0.0])
    for rule, score, accuracy in zip(all_rules, scores, accuracies):
        rule.score, rule.accuracy = score, accuracy
    decoded = prototype.DecodedRuleArrays(
        np.asarray([r.antecedents for r in all_rules]), np.asarray(master.get_consequents()),
        np.ones(len(all_rules)), np.array([2, 1]), np.arange(len(all_rules)))
    evaluator = eval_rules.evalRuleBase(master, X, y)
    expected = (evaluator.size_antecedents_eval(0.1), evaluator.effective_rulesize_eval(0.1))
    assert prototype.complexity_metrics(decoded, scores, 0.1) == expected
    expected_keep = np.array([True, True, False])
    np.testing.assert_array_equal(prototype.prune_mask(scores, accuracies, 0.1), expected_keep)
    master.purge_rules(0.1)
    assert len(master.get_rules()) == int(expected_keep.sum())


def test_prune_mask_matches_nan_and_exact_threshold_reference_behavior():
    X = np.array([[0.0], [1.0]])
    lvs = utils.construct_partitions(X, fs.FUZZY_SETS.t1)
    base = rules.RuleBaseT1(lvs, [rules.RuleSimple([0]), rules.RuleSimple([1]),
                                     rules.RuleSimple([2])])
    expected = np.array([True, True, False])
    scores = np.array([np.nan, 0.1, 0.09999999999999999])
    accuracies = np.ones(3)
    for rule, score, accuracy in zip(base.rules, scores, accuracies):
        rule.score, rule.accuracy = score, accuracy
    np.testing.assert_array_equal(prototype.prune_mask(scores, accuracies, 0.1), expected)
    base.prune_bad_rules(0.1)
    assert len(base.rules) == int(expected.sum())
    assert np.isnan(base.rules[0].score)


def test_empty_effective_rules_are_detected_without_membership_objects():
    decoded = prototype.decode_fixed_partitions(
        np.array([0, 1, -1, -1, -1, -1]), n_rules=2, n_ants=1,
        n_features=2, n_classes=2, n_lv_possible=[3, 3])
    assert decoded.antecedents.shape == (0, 2)
    np.testing.assert_array_equal(decoded.class_counts, [0, 0])
