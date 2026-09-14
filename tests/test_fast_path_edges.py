"""Edge cases of the object-free and population evaluators.

Both evaluators decline unsupported inputs by returning None so the object path
runs instead; these tests pin those refusals and the exact agreement between the
population route and the scalar route on the cases they do support.
"""
import numpy as np

import _array_fitness as af
import _population_fitness as pf
import rules
from _fitness import _FiringCache, _LabelDomain


TERM_COUNTS = np.array([3, 3])
Y = np.array([0, 1, 0, 1, 0, 1])


def _truth(n_samples=6, n_features=2, n_terms=3, seed=0):
    rng = np.random.default_rng(seed)
    return [[rng.random(n_samples) for _ in range(n_terms)] for _ in range(n_features)]


def test_decoding_refuses_invalid_chromosomes():
    # Two rules of one antecedent: features, terms, then consequents from position 4.
    assert af.decode_rule_arrays(np.array([0, 1, 0, 2, 0, 2]), 2, 1, 2, 2, TERM_COUNTS, 0, 4) is None
    assert af.decode_rule_arrays(np.array([0, 5, 0, 2, 0, 1]), 2, 1, 2, 2, TERM_COUNTS, 0, 4) is None


def test_weighted_duplicates_are_dropped_like_rule_objects():
    # Two identical rules of class 0 with the same weight: the second one is a duplicate.
    x = np.array([0, 0, 1, 1, 0, 0, 50, 50])
    decoded = af.decode_rule_arrays(x, 2, 1, 2, 2, TERM_COUNTS, 2, 4)
    assert decoded.consequents.tolist() == [0]
    assert decoded.weights.tolist() == [0.5]


def test_t1_firing_without_gatherable_memberships_matches_the_gathered_path():
    truth = _truth(n_samples=5)
    antecedents = np.array([[0, -1], [2, 1], [-1, -1]])
    gathered = af.firing_strengths(antecedents, truth, 5, interval=False)

    # Plain lists cannot be gathered, so the per-rule reference loop runs instead.
    as_lists = [[column.tolist() for column in feature] for feature in truth]
    np.testing.assert_array_equal(af.firing_strengths(antecedents, as_lists, 5, interval=False), gathered)

    unusable = [as_lists[0], 'not a feature']
    assert af.firing_strengths(np.empty((2, 0), dtype=int), as_lists, 5, False) is None
    assert af.firing_strengths(np.array([[3, 0]]), as_lists, 5, False) is None
    assert af.firing_strengths(antecedents, unusable, 5, False) is None
    assert af.firing_strengths(antecedents, as_lists, 5, False, cache=_FiringCache()) is None

    decoded = af._DecodedCandidate(antecedents, np.array([0, 1, 1]), np.ones(3))
    assert af.score_candidate(decoded, unusable, np.zeros((5, 2)), Y[:5], 2, 0, False, 0.0, 0.0, 0.0, False) is None


def _population_arguments(**overrides):
    arguments = dict(n_rules=2, n_ants=1, n_features=2, n_classes=2, term_counts=TERM_COUNTS,
                     fourth_pointer=4, ds_mode=0, allow_unknown=False, tolerance=0.0,
                     alpha=0.0, beta=0.0, labels=_LabelDomain.build(Y, 2))
    arguments.update(overrides)
    return arguments


def test_population_scoring_declines_unsupported_candidates():
    truth = _truth()
    packed = rules.pack_membership_table(truth, len(Y), tail=())
    good = np.array([[0, 1, 0, 1, 0, 1], [0, 1, 2, 2, 1, 0]])

    assert pf._decode_population(good[0], 2, 1, 2, 2, TERM_COUNTS, 4) is None
    bad_consequent = good.copy()
    bad_consequent[0, 5] = 2
    assert pf.score_population(bad_consequent, packed, Y, **_population_arguments()) is None

    table, offsets, lengths = packed
    assert pf.score_population(good, (table, offsets[:1], lengths[:1]), Y, **_population_arguments()) is None


def test_population_without_active_rules_scores_zero():
    packed = rules.pack_membership_table(_truth(), len(Y), tail=())
    inactive = np.array([[0, 1, 0, 1, -1, -1], [1, 0, 2, 2, -1, -1]])
    np.testing.assert_array_equal(pf.score_population(inactive, packed, Y, **_population_arguments()), [0.0, 0.0])


def test_population_scores_match_the_scalar_route_with_each_penalty():
    truth = _truth()
    packed = rules.pack_membership_table(truth, len(Y), tail=())
    genes = np.array([[0, 1, 0, 1, 0, 1], [0, 1, 2, 2, 1, 0], [1, 1, 0, 2, 1, 1]])
    X = np.zeros((len(Y), 2))

    for penalties in ({}, {'alpha': 0.5}, {'beta': 0.5}):
        arguments = _population_arguments(**penalties)
        scores = pf.score_population(genes, packed, Y, **arguments)
        for gene, score in zip(genes, scores):
            decoded = af.decode_rule_arrays(gene, 2, 1, 2, 2, TERM_COUNTS, 0, 4)
            expected = af.score_candidate(decoded, truth, X, Y, 2, 0, False, 0.0,
                                          arguments['alpha'], arguments['beta'], False,
                                          labels=arguments['labels'])
            assert score == expected
