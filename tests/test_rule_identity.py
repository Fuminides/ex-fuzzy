"""Rule equality and hashing depend on antecedents, consequent and modifiers only."""
import numpy as np
import pytest
from sklearn.datasets import load_iris

from ex_fuzzy import evolutionary_fit as evf
from ex_fuzzy import fuzzy_sets as fs
from ex_fuzzy import rules
from ex_fuzzy import utils
from ex_fuzzy._fitness import score_rulebase


def test_hash_is_consistent_with_equality_after_evaluation():
    first = rules.RuleSimple([0, 1, -1], consequent=0)
    second = rules.RuleSimple([0, 1, -1], consequent=0)
    first.score, first.weight, first.accuracy = 0.1, 0.2, 0.3
    second.score, second.weight, second.accuracy = 0.9, 0.8, 0.7
    assert first == second and hash(first) == hash(second)
    assert first != rules.RuleSimple([0, 1, -1], consequent=1)
    assert first != rules.RuleSimple([1, 0, -1], consequent=0)
    assert first != object() and first != None  # noqa: E711 - the comparison itself is under test
    # A rule is found again in a set after its metrics change.
    seen = {first}
    first.score = 0.5
    assert second in seen


def test_modifiers_distinguish_rules():
    plain = rules.RuleSimple([0], consequent=0, modifiers=np.array([1.0]))
    hedged = rules.RuleSimple([0], consequent=0, modifiers=np.array([2.0]))
    same = rules.RuleSimple([0], consequent=0, modifiers=[1.0])
    assert plain != hedged and hash(plain) != hash(hedged)
    assert plain == same and hash(plain) == hash(same)


def test_rule_base_keeps_the_first_of_equal_rules_with_different_weights():
    X, _ = load_iris(return_X_y=True)
    partitions = utils.construct_partitions(X, fs.FUZZY_SETS.t1)
    first = rules.RuleSimple([0, -1, -1, -1])
    second = rules.RuleSimple([0, -1, -1, -1])
    other = rules.RuleSimple([-1, 1, -1, -1])
    first.weight, second.weight, other.weight = 0.2, 0.8, 0.5
    base = rules.RuleBaseT1(partitions, [first, other, second])
    assert [rule is kept for rule, kept in zip(base.rules, (first, other))] == [True, True]
    assert len(base.rules) == 2


@pytest.mark.parametrize('kind', [fs.FUZZY_SETS.t1, fs.FUZZY_SETS.t2])
@pytest.mark.parametrize('fixed', [True, False])
def test_weighted_duplicates_are_removed_alike_on_both_evaluation_paths(kind, fixed):
    X, y = load_iris(return_X_y=True)
    partitions = utils.construct_partitions(X, kind) if fixed else None
    problem = evf.FitRuleBase(X, y, 6, 2, 3, linguistic_variables=partitions,
                              fuzzy_type=kind, ds_mode=2, tolerance=0.01)
    rng = np.random.default_rng(7)
    genes = rng.integers(problem.xl.astype(int), problem.xu.astype(int) + 1,
                         size=(30, problem.n_var))
    ants, slots = problem.nAnts, problem.nRules * problem.nAnts
    fourth = problem._consequent_pointer(kind)
    weights = slice(fourth + problem.nRules, fourth + 2 * problem.nRules)
    for gene in genes:
        # Rules 1 and 2 repeat rule 0 in its class with other weights; rule 3
        # repeats it in another class.
        for copy_index in (1, 2, 3):
            gene[copy_index * ants:(copy_index + 1) * ants] = gene[:ants]
            gene[slots + copy_index * ants:slots + (copy_index + 1) * ants] = gene[slots:slots + ants]
        gene[fourth:fourth + 3] = gene[fourth]
        gene[fourth + 3] = (gene[fourth] + 1) % 3
        gene[weights] = [10, 60, 90, 40, 50, 50]

    for gene in genes:
        master = problem._construct_ruleBase(gene.copy(), kind)
        for base in master.rule_bases:
            rows = [tuple(rule.antecedents) for rule in base.rules]
            assert len(rows) == len(set(rows))
        # The first occurrence keeps its weight; the dropped copies' weights are gone.
        assert {rule.weight for rule in master.get_rules()} <= {0.1, 0.4, 0.5}
        expected = score_rulebase(problem._construct_ruleBase(gene.copy(), kind), X, y,
                                  problem.tolerance, 0.0, 0.0, problem._precomputed_truth)
        assert problem._array_score(gene.copy()) == expected
