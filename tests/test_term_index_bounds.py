"""Term-gene bounds of the genetic encoding with unequal partition sizes.

A term gene sits next to the feature gene that decides which variable the slot
holds, so its upper bound cannot depend on that feature: it must allow every
term of the largest partition, and the decoders clip the index to the term
count of the chosen feature.
"""
import numpy as np
import pandas as pd
import pytest

from ex_fuzzy import _array_fitness as arrfit
from ex_fuzzy import evolutionary_fit as evf
from ex_fuzzy import evolutionary_fit_regression as evr
from ex_fuzzy import fuzzy_sets as fs
from ex_fuzzy import utils

N_RULES, N_ANTS = 6, 2


def _data(n_features=4):
    rng = np.random.default_rng(0)
    X = rng.random((150, n_features))
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)]), (X[:, -1] > 0.5).astype(int)


def _variables(X, counts):
    return [utils.construct_partitions(X[[column]], fs.FUZZY_SETS.t1, n_partitions=count)[0]
            for column, count in zip(X.columns, counts)]


def _classifier(counts, n_ants=N_ANTS, precomputed=True):
    X, y = _data(len(counts))
    if precomputed:
        return evf.FitRuleBase(X.to_numpy(), y, nRules=N_RULES, nAnts=n_ants, n_classes=2,
                               linguistic_variables=_variables(X, counts))
    return evf.FitRuleBase(X.to_numpy(), y, nRules=N_RULES, nAnts=n_ants, n_classes=2,
                           n_linguistic_variables=list(counts))


def _term_bounds(problem):
    start = problem.nRules * problem.nAnts
    return problem.xl[start:2 * start], problem.xu[start:2 * start]


@pytest.mark.parametrize("precomputed", [True, False])
@pytest.mark.parametrize("counts", [(3, 3, 5, 5), (5, 3, 3, 4), (2, 6, 2, 2)])
def test_every_term_gene_reaches_the_largest_partition(counts, precomputed):
    low, high = _term_bounds(_classifier(counts, precomputed=precomputed))
    assert np.all(low == -1)
    assert np.all(high == max(counts) - 1)


@pytest.mark.parametrize("precomputed", [True, False])
@pytest.mark.parametrize("counts, n_ants", [((3, 3, 3, 3), 2), ((4, 4, 4, 4), 4), ((5, 5, 5), 3)])
def test_equal_term_counts_keep_the_previous_bounds(counts, n_ants, precomputed):
    """The old bound for slot j was ``counts[j] - 1``; with equal counts it is the same."""
    problem = _classifier(counts, n_ants=n_ants, precomputed=precomputed)
    low, high = _term_bounds(problem)
    old_high = np.array([counts[slot] - 1 for slot in range(n_ants)] * N_RULES)
    np.testing.assert_array_equal(high, old_high)
    np.testing.assert_array_equal(low, -1)


def _gene(problem, rules):
    """Gene whose rule ``r`` holds the given ``(feature, term)`` pairs, class 0.

    Slots of a rule that are not listed repeat its last pair, so they change nothing.
    """
    gene = np.zeros(problem.n_var, dtype=int)
    slots = problem.nRules * problem.nAnts
    gene[:slots] = 1
    gene[slots:2 * slots] = -1
    gene[2 * slots:2 * slots + problem.nRules] = -1
    for rule, assignments in enumerate(rules):
        for slot in range(problem.nAnts):
            feature, term = assignments[min(slot, len(assignments) - 1)]
            gene[rule * problem.nAnts + slot] = feature
            gene[slots + rule * problem.nAnts + slot] = term
        gene[2 * slots + rule] = 0
    return gene


def test_each_term_of_each_variable_survives_decoding():
    """No valid term index of the feature a slot holds is clipped, in any slot."""
    counts = (3, 3, 5, 5)
    problem = _classifier(counts)
    term_counts = np.asarray(counts)
    pairs = [(feature, term) for feature, count in enumerate(counts) for term in range(count)]
    for slot in range(N_ANTS):
        for feature, term in pairs:
            pad = (1 if feature == 0 else 0, -1)  # the slot under test comes last, so it wins
            gene = _gene(problem, [[pad] * slot + [(feature, term)]])
            assert gene[N_RULES * N_ANTS:][slot] <= problem.xu[N_RULES * N_ANTS + slot]
            decoded = arrfit.decode_rule_arrays(
                gene, problem.nRules, problem.nAnts, len(counts), problem.n_classes,
                term_counts, 0, problem._consequent_pointer(problem.fuzzy_type))
            assert decoded.antecedents[0][feature] == term
            rule = problem._construct_ruleBase(gene.copy(), problem.fuzzy_type).get_rules()[0]
            assert rule.antecedents[feature] == term


def test_larger_variable_uses_its_extra_terms_after_smaller_ones():
    """Five-term variables placed after three-term ones: terms 3 and 4 are reachable."""
    problem = _classifier((3, 3, 5, 5))
    rng = np.random.default_rng(1)
    genes = rng.integers(problem.xl.astype(int), problem.xu.astype(int) + 1,
                         size=(400, problem.n_var))
    seen = {feature: set() for feature in range(4)}
    for gene in genes:
        base = problem._construct_ruleBase(gene.copy(), problem.fuzzy_type)
        for rule in base.get_rules():
            for feature, term in enumerate(rule.antecedents):
                if term != -1:
                    seen[feature].add(int(term))
    assert seen[2] == {0, 1, 2, 3, 4}
    assert seen[3] == {0, 1, 2, 3, 4}
    assert seen[0] == {0, 1, 2}


def test_index_beyond_a_smaller_partition_is_still_clipped():
    """Indexes a smaller variable does not have keep the documented clip, never an error."""
    problem = _classifier((3, 3, 5, 5))
    gene = _gene(problem, [[(0, 4)]])
    rule = problem._construct_ruleBase(gene.copy(), problem.fuzzy_type).get_rules()[0]
    assert rule.antecedents[0] == 2


def test_regression_bounds_use_the_largest_partition():
    """The regressor already bounds every term gene by the largest partition (it
    drops, rather than clips, an index beyond the feature's own term count)."""
    X, _ = _data()
    y = np.random.default_rng(2).random(len(X))
    problem = evr.FitRuleBaseRegression(
        X.to_numpy(), y, nRules=N_RULES, nAnts=N_ANTS,
        linguistic_variables=_variables(X, (3, 3, 5, 5)))
    start = N_RULES * N_ANTS
    assert np.all(problem.xu[start:2 * start] == 4)
    chromosome = np.zeros(problem.n_var, dtype=int)
    chromosome[:start] = 2
    chromosome[start:2 * start] = 4
    assert np.all(problem._rule_term_matrix(chromosome)[:, 2] == 4)
