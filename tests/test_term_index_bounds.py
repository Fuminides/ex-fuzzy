"""Term-gene bounds of the genetic encoding with unequal partition sizes.

A term gene sits next to the feature gene that decides which variable the slot
holds, so its upper bound cannot depend on that feature: it must allow every
term of the largest partition ``M``, and the decoders map the gene ``g`` onto
the ``n`` terms of the chosen feature as ``(g * n) // M`` (proportional and
order preserving, so every term is drawn equally often up to one gene value).
"""
import numpy as np
import pandas as pd
import pytest

from ex_fuzzy import _array_fitness as arrfit
from ex_fuzzy import evolutionary_fit as evf
from ex_fuzzy import evolutionary_fit_regression as evr
from ex_fuzzy import fuzzy_sets as fs
from ex_fuzzy import utils
from ex_fuzzy._fitness import (_fitness_cache_scope, remap_term, remap_terms, score_rulebase,
                               term_gene)

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
    """Every term of the feature a slot holds is reachable, in any slot, in every decoder."""
    counts = (3, 3, 5, 5)
    problem = _classifier(counts)
    term_counts = np.asarray(counts)
    pairs = [(feature, term) for feature, count in enumerate(counts) for term in range(count)]
    for slot in range(N_ANTS):
        for feature, term in pairs:
            pad = (1 if feature == 0 else 0, -1)  # the slot under test comes last, so it wins
            gene_value = term_gene(term, counts[feature], max(counts))
            gene = _gene(problem, [[pad] * slot + [(feature, gene_value)]])
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


def test_index_beyond_the_bound_is_still_clipped_to_the_last_term():
    """A gene above the bound (never drawn) keeps the documented clip, never an error."""
    problem = _classifier((3, 3, 5, 5))
    gene = _gene(problem, [[(0, 9)]])
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


# ---------------------------------------------------------------------------
# Proportional remap of a term gene onto the terms of the chosen variable.

TERM_COUNTS = [(n, m) for m in (1, 2, 3, 4, 5, 7) for n in range(1, m + 1)]


@pytest.mark.parametrize("n_terms, max_terms", TERM_COUNTS)
def test_remap_properties(n_terms, max_terms):
    genes = np.arange(max_terms)
    terms = np.asarray([remap_term(int(g), n_terms, max_terms) for g in genes])
    # The array form agrees with the scalar form.
    np.testing.assert_array_equal(remap_terms(genes, np.asarray(n_terms), max_terms), terms)
    # (b) every term of the variable is reachable, and nothing else.
    assert set(terms) == set(range(n_terms))
    # (c) a uniform draw over the genes gives every term floor or ceil of M / n values.
    counts = np.bincount(terms, minlength=n_terms)
    assert counts.max() - counts.min() <= 1
    assert counts.sum() == max_terms
    # (d) adjacent genes select equal or adjacent terms (and the map never decreases).
    assert set(np.diff(terms)) <= {0, 1}
    # (a) equal counts: identity.
    if n_terms == max_terms:
        np.testing.assert_array_equal(terms, genes)
    # The inverse picks the first gene of each term.
    for term in range(n_terms):
        gene = term_gene(term, n_terms, max_terms)
        assert remap_term(gene, n_terms, max_terms) == term
        assert gene == 0 or remap_term(gene - 1, n_terms, max_terms) == term - 1


def test_remap_keeps_dont_care_and_clips_out_of_range_genes():
    assert remap_term(-1, 3, 5) == -1 and term_gene(-1, 3, 5) == -1
    assert remap_term(-2, 3, 5) == -2  # as the old clip: nothing below -1 is repaired
    np.testing.assert_array_equal(remap_terms(np.array([-1, 0, 4, 9]), np.array(3), 5), [-1, 0, 2, 2])


def test_remap_distribution_for_three_four_and_five_terms():
    """Uniform draw over 0..4: clipping over-represented the last term of small variables."""
    expected = {5: [1, 1, 1, 1, 1], 4: [2, 1, 1, 1], 3: [2, 2, 1]}
    for n_terms, counts in expected.items():
        terms = [remap_term(g, n_terms, 5) for g in range(5)]
        assert np.bincount(terms, minlength=n_terms).tolist() == counts
        clipped = [min(g, n_terms - 1) for g in range(5)]
        assert np.bincount(clipped, minlength=n_terms).tolist()[-1] == 5 - n_terms + 1


@pytest.mark.parametrize("precomputed", [True, False])
def test_decoded_term_frequencies_under_uniform_genes(precomputed):
    """The object decoder, on random genes, draws every term of every variable evenly."""
    counts = (4, 4, 3, 5, 3)
    problem = _classifier(counts, n_ants=5, precomputed=precomputed)
    rng = np.random.default_rng(3)
    genes = rng.integers(problem.xl.astype(int), problem.xu.astype(int) + 1,
                         size=(600, problem.n_var))
    tally = {feature: np.zeros(count) for feature, count in enumerate(counts)}
    for gene in genes:
        for rule in problem._construct_ruleBase(gene.copy(), problem.fuzzy_type).get_rules():
            for feature, term in enumerate(rule.antecedents):
                if term != -1:
                    tally[feature][int(term)] += 1
    for feature, count in enumerate(counts):
        share = tally[feature] / tally[feature].sum()
        # Gene values 0..M-1 plus -1 (don't care): expected share is 1/M or 2/M of the cares.
        assert np.all(share > 0), (feature, share)
    # Terms are drawn in proportion to their gene values: 3-term variables see 2:2:1.
    three = tally[2] / tally[2].sum()
    assert three[2] < 0.3 and three[0] > 0.3 and three[1] > 0.3


@pytest.mark.parametrize("counts", [(3, 3, 5, 5), (5, 4, 3, 5), (2, 6, 2, 3)])
def test_encode_rulebase_round_trips_with_unequal_term_counts(counts):
    """Only fixed partitions can be encoded: the gene is the inverse of the decoder's map."""
    problem = _classifier(counts, n_ants=len(counts), precomputed=True)
    rng = np.random.default_rng(4)
    for _ in range(20):
        gene = rng.integers(problem.xl.astype(int), problem.xu.astype(int) + 1,
                            size=problem.n_var)
        # One slot per feature, in feature order, as encode_rulebase needs.
        gene[:problem.nRules * problem.nAnts] = np.tile(np.arange(len(counts)), problem.nRules)
        base = problem._construct_ruleBase(gene.copy(), problem.fuzzy_type)
        encoded = problem.encode_rulebase(base, optimize_lv=False)
        again = problem._construct_ruleBase(encoded.copy(), problem.fuzzy_type)
        assert ([list(rule.antecedents) for rule in base.get_rules()]
                == [list(rule.antecedents) for rule in again.get_rules()])
        assert base.get_consequents() == again.get_consequents()


@pytest.mark.parametrize("precomputed", [True, False])
@pytest.mark.parametrize("counts", [(3, 3, 3, 3), (4, 4, 4, 4), (5, 5, 5, 5)])
def test_equal_term_counts_decode_exactly_as_the_clip_did(counts, precomputed):
    """With equal counts the remap is the identity, so every decoder keeps the old phenotype."""
    problem = _classifier(counts, n_ants=3, precomputed=precomputed)
    pointer = problem._consequent_pointer(problem.fuzzy_type)
    slots = problem.nRules * problem.nAnts
    for gene in _population(problem, count=60, seed=11):
        base = problem._construct_ruleBase(gene.copy(), problem.fuzzy_type)
        expected = np.full((problem.nRules, len(counts)), -1)
        features = gene[:slots].reshape(problem.nRules, problem.nAnts)
        terms = np.minimum(gene[slots:2 * slots], np.asarray(counts)[features.reshape(-1)] - 1)
        for rule in range(problem.nRules):
            for slot in range(problem.nAnts):
                expected[rule, features[rule, slot]] = terms[rule * problem.nAnts + slot]
        decoded = arrfit.decode_rule_arrays(gene, problem.nRules, problem.nAnts, len(counts),
                                            problem.n_classes, np.asarray(counts), 0, pointer)
        rows = {tuple(row) for row, c in zip(expected.tolist(), gene[pointer:pointer + problem.nRules])
                if c != -1 and any(v != -1 for v in row)}
        assert {tuple(row) for row in decoded.antecedents.tolist()} == rows
        assert {tuple(int(t) for t in rule.antecedents) for rule in base.get_rules()} == rows


# Parity of every decoder with unequal term counts.

def _reference(problem, gene):
    rulebase = problem._construct_ruleBase(gene.copy(), problem.fuzzy_type)
    return score_rulebase(rulebase, problem.X, problem.y, problem.tolerance,
                          problem.alpha_, problem.beta_, problem._precomputed_truth,
                          max_conditions=problem._max_conditions)


def _population(problem, count=40, seed=7):
    return np.random.default_rng(seed).integers(
        problem.xl.astype(int), problem.xu.astype(int) + 1, size=(count, problem.n_var))


@pytest.mark.parametrize("precomputed", [True, False])
@pytest.mark.parametrize("counts", [(3, 4, 5, 3), (5, 4, 3, 5), (2, 6, 2, 3)])
def test_decoders_agree_with_unequal_term_counts(counts, precomputed):
    """Object decoder, array decoder, population batch and torch objective decode alike."""
    problem = _classifier(counts, n_ants=3, precomputed=precomputed)
    genes = _population(problem)
    scores = np.asarray([problem._array_score(gene.copy()) for gene in genes])
    assert all(problem._array_score(gene.copy()) == _reference(problem, gene)
               for gene in genes[:15])
    if precomputed:  # the population batch only serves fixed partitions
        with _fitness_cache_scope(problem, True):
            population = problem._population_scores(genes)
        assert population is not None
        np.testing.assert_array_equal(population, scores)

    # Effective phenotypes: array decoder vs the rules the object decoder builds.
    term_counts = np.asarray(counts)
    pointer = problem._consequent_pointer(problem.fuzzy_type)
    for gene in genes[:15]:
        decoded = arrfit.decode_rule_arrays(gene, problem.nRules, problem.nAnts, len(counts),
                                            problem.n_classes, term_counts, 0, pointer)
        base = problem._construct_ruleBase(gene.copy(), problem.fuzzy_type)
        objects = sorted(tuple(int(t) for t in rule.antecedents) for rule in base.get_rules())
        assert sorted(tuple(row) for row in decoded.antecedents.tolist()) == objects


@pytest.mark.parametrize("counts", [(3, 4, 5, 3), (5, 4, 3, 5), (2, 6, 2, 3)])
def test_torch_objective_agrees_with_unequal_term_counts(counts):
    pytest.importorskip("torch")
    from ex_fuzzy import _torch_fitness
    problem = _classifier(counts, n_ants=3, precomputed=True)
    genes = _population(problem, count=60)
    objective = _torch_fitness.TorchObjective.build(problem, "cpu")
    assert objective is not None
    torch_scores, declined = objective.score(genes)
    assert not declined.any()
    expected = np.asarray([1 - problem._array_score(gene.copy()) for gene in genes])
    np.testing.assert_array_equal(1 - torch_scores, expected)
