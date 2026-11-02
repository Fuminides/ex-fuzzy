"""
Tests for the rule-set stability diagnostics of ex_fuzzy.pattern_stability:
rule keys, selection frequencies, pairwise Jaccard, consensus rules and the
resampled stabilizer run.
"""
import numpy as np
import pytest
from sklearn.datasets import load_iris

from ex_fuzzy import fuzzy_sets as fs
from ex_fuzzy import pattern_stability as ps
from ex_fuzzy import rules as rl
from ex_fuzzy import utils


@pytest.fixture(scope='module')
def iris():
    return load_iris(return_X_y=True)


@pytest.fixture(scope='module')
def lvs(iris):
    return utils.construct_partitions(iris[0], fs.FUZZY_SETS.t1)


def master(lvs, per_class, names=('a', 'b', 'c')):
    """MasterRuleBase from one list of antecedent lists per class."""
    bases = [rl.RuleBaseT1(lvs, [rl.RuleSimple(ant, consequent=ix) for ant in antecedents])
             for ix, antecedents in enumerate(per_class)]
    return rl.MasterRuleBase(bases, list(names))


@pytest.fixture
def rb_a(lvs):
    return master(lvs, [[[0, -1, 1, -1], [-1, 2, -1, -1]], [[1, -1, -1, 0]], []])


@pytest.fixture
def rb_disjoint(lvs):
    return master(lvs, [[[2, -1, -1, -1]], [[-1, 0, -1, -1]], [[-1, -1, 2, 2]]])


class TestRuleKeys:
    def test_keys_use_names_sorted_and_class_labels(self, rb_a, lvs):
        keys = ps.rule_keys(rb_a)
        n0, n2 = lvs[0].name, lvs[2].name
        t0, t2 = lvs[0].linguistic_variable_names(), lvs[2].linguistic_variable_names()
        assert ((tuple(sorted([(n0, t0[0]), (n2, t2[1])])), 'a')) in keys
        assert len(keys) == 3
        assert {label for _, label in keys} == {'a', 'b'}
        assert keys == sorted(keys, key=lambda key: (str(key[1]), key[0]))

    def test_same_antecedents_in_different_classes_are_different_rules(self, lvs):
        keys = ps.rule_keys(master(lvs, [[[0, -1, -1, -1]], [[0, -1, -1, -1]], []]))
        assert len(keys) == 2

    def test_rule_repeated_inside_one_base_has_one_key(self, lvs):
        mrb = master(lvs, [[[0, -1, -1, -1]], [], []])
        mrb.add_rule(rl.RuleSimple([0, -1, -1, -1]), 0)  # add_rule does not deduplicate
        assert len(mrb.get_rules()) == 2
        assert len(ps.rule_keys(mrb)) == 1

    def test_score_does_not_change_the_key(self, lvs):
        first, second = master(lvs, [[[0, -1, -1, -1]], [], []]), master(lvs, [[[0, -1, -1, -1]], [], []])
        first.get_rules()[0].score = 0.9
        second.get_rules()[0].score = 0.1
        assert ps.rule_keys(first) == ps.rule_keys(second)

    def test_numpy_class_labels_become_plain(self, lvs):
        mrb = master(lvs, [[[0, -1, -1, -1]], [], []], names=np.array([0, 1, 2]))
        label = ps.rule_keys(mrb)[0][1]
        assert label == 0 and type(label) is int

    def test_ambiguous_names_are_rejected(self, lvs):
        clones = [fs.fuzzyVariable.__new__(fs.fuzzyVariable) for _ in range(2)]
        for clone in clones:
            clone.__dict__.update(lvs[0].__dict__)
        mrb = rl.MasterRuleBase([rl.RuleBaseT1(clones, [rl.RuleSimple([0, 0])])])
        with pytest.raises(ValueError, match='unique'):
            ps.rule_keys(mrb)


class TestSelectionFrequencies:
    def test_identical_fits_give_frequency_one(self, rb_a, lvs):
        twin = master(lvs, [[[0, -1, 1, -1], [-1, 2, -1, -1]], [[1, -1, -1, 0]], []])
        freqs = ps.selection_frequencies([rb_a, twin, rb_a])
        assert set(freqs) == set(ps.rule_keys(rb_a))
        assert all(f == 1.0 for f in freqs.values())

    def test_disjoint_fits_give_frequency_one_over_n(self, rb_a, rb_disjoint):
        freqs = ps.selection_frequencies([rb_a, rb_disjoint])
        assert len(freqs) == 3 + 3 and all(f == 0.5 for f in freqs.values())

    def test_repeated_rule_counts_once(self, lvs, rb_disjoint):
        repeated = master(lvs, [[[0, -1, -1, -1]], [], []])
        repeated.add_rule(rl.RuleSimple([0, -1, -1, -1]), 0)
        freqs = ps.selection_frequencies([repeated, rb_disjoint])
        assert freqs[ps.rule_keys(repeated)[0]] == 0.5

    def test_partial_overlap(self, lvs):
        shared = [[0, -1, -1, -1]]
        fits = [master(lvs, [shared, [], []]), master(lvs, [shared, [[-1, 1, -1, -1]], []]),
                master(lvs, [shared, [], []]), master(lvs, [[], [], []])]
        freqs = ps.selection_frequencies(fits)
        assert sorted(freqs.values()) == [0.25, 0.75]

    def test_no_fits(self):
        with pytest.raises(ValueError):
            ps.selection_frequencies([])


class TestPairwiseJaccard:
    def test_identical_fits(self, rb_a):
        res = ps.pairwise_jaccard([rb_a, rb_a, rb_a])
        assert res == dict(n_pairs=3, mean=1.0, sd=0.0, min=1.0, max=1.0)

    def test_disjoint_fits(self, rb_a, rb_disjoint):
        res = ps.pairwise_jaccard([rb_a, rb_disjoint])
        assert res['mean'] == res['min'] == res['max'] == 0.0 and res['n_pairs'] == 1

    def test_known_value_and_spread(self, lvs):
        one = master(lvs, [[[0, -1, -1, -1], [1, -1, -1, -1]], [], []])
        two = master(lvs, [[[0, -1, -1, -1]], [], []])
        three = master(lvs, [[], [[0, -1, -1, -1]], []])
        res = ps.pairwise_jaccard([one, two, three])
        assert res['n_pairs'] == 3
        assert res['max'] == 0.5 and res['min'] == 0.0
        assert res['mean'] == pytest.approx(1 / 6)

    def test_empty_rule_bases_count_as_identical(self, lvs):
        empty = master(lvs, [[], [], []])
        assert ps.pairwise_jaccard([empty, empty])['mean'] == 1.0

    def test_needs_two_fits(self, rb_a):
        with pytest.raises(ValueError):
            ps.pairwise_jaccard([rb_a])


class TestConsensusKeys:
    FREQS = {(((('x', 'Low')),), 'a'): 0.3, (((('x', 'High')),), 'a'): 0.7, (((('y', 'Low')),), 'b'): 1.0}

    def test_threshold_is_inclusive_and_ordered_by_frequency(self):
        assert [self.FREQS[k] for k in ps.consensus_keys(self.FREQS, 0.3)] == [1.0, 0.7, 0.3]
        assert [self.FREQS[k] for k in ps.consensus_keys(self.FREQS, 0.7)] == [1.0, 0.7]
        assert len(ps.consensus_keys(self.FREQS, 1.0)) == 1

    def test_float_noise_does_not_drop_a_rule(self):
        key = ((('x', 'Low'),), 'a')
        assert ps.consensus_keys({key: 3 / 10}, 0.3) == [key]

    @pytest.mark.parametrize('pi', [0.0, -0.1, 1.01])
    def test_pi_bounds(self, pi):
        with pytest.raises(ValueError, match='pi'):
            ps.consensus_keys(self.FREQS, pi)


class TestSummary:
    def test_summary_is_plain_data(self, rb_a, rb_disjoint, lvs):
        twin = master(lvs, [[[0, -1, 1, -1], [-1, 2, -1, -1]], [[1, -1, -1, 0]], []])
        res = ps.rule_set_stability_summary([rb_a, twin, rb_disjoint], pi=0.6)
        assert res['n_fits'] == 3 and res['n_empty_fits'] == 0 and res['pi'] == 0.6
        assert [row['count'] for row in res['frequencies']] == [2, 2, 2, 1, 1, 1]
        assert res['frequencies'][0]['frequency'] == pytest.approx(2 / 3)
        assert len(res['consensus']) == 3
        assert res['jaccard']['n_pairs'] == 3 and res['jaccard']['max'] == 1.0
        assert ps.rule_set_stability_summary([rb_a, twin])['consensus'] is None

    def test_empty_fits_are_counted(self, rb_a, lvs):
        res = ps.rule_set_stability_summary([rb_a, master(lvs, [[], [], []])])
        assert res['n_empty_fits'] == 1 and res['jaccard']['mean'] == 0.0

    def test_different_variables_are_rejected(self, rb_a, iris):
        other = master(utils.construct_partitions(iris[0] * 2, fs.FUZZY_SETS.t1), [[[0, -1, -1, -1]], [], []])
        with pytest.raises(ValueError, match='share'):
            ps.rule_set_stability_summary([rb_a, other])

    def test_invalid_arguments(self, rb_a):
        with pytest.raises(ValueError):
            ps.rule_set_stability_summary([rb_a])
        with pytest.raises(ValueError, match='pi'):
            ps.rule_set_stability_summary([rb_a, rb_a], pi=0.0)


class TestConsensusRuleBase:
    def test_round_trip_and_recomputed_scores(self, rb_a, iris):
        X, y = iris
        keys = ps.rule_keys(rb_a)
        consensus = ps.consensus_rule_base(keys, rb_a, X, y)
        assert ps.rule_keys(consensus) == keys
        assert len(consensus) == len(rb_a) and consensus.get_consequents_names() == rb_a.get_consequents_names()
        assert len(consensus.get_scores()) == 3 and np.all(consensus.get_scores() >= 0)
        assert consensus.winning_rule_predict(X).shape == y.shape
        assert len(rb_a.get_rules()) == 3  # the template is untouched

    def test_subset_and_unscored(self, rb_a):
        keys = ps.rule_keys(rb_a)[:1]
        consensus = ps.consensus_rule_base(keys, rb_a)
        assert ps.rule_keys(consensus) == keys
        assert not hasattr(consensus.get_rules()[0], 'score')

    def test_rejects_foreign_keys_and_partial_data(self, rb_a, iris):
        with pytest.raises(ValueError, match='Class'):
            ps.consensus_rule_base([((), 'zzz')], rb_a)
        with pytest.raises(ValueError, match='term'):
            ps.consensus_rule_base([((('nope', 'Low'),), 'a')], rb_a)
        with pytest.raises(ValueError, match='both|neither'):
            ps.consensus_rule_base([], rb_a, X=iris[0])


class TestStabilizerRun:
    def test_end_to_end_on_iris(self, iris, lvs):
        X, y = iris
        stabilizer = ps.pattern_stabilizer(X, y, nRules=6, nAnts=2, linguistic_variables=lvs)
        res = stabilizer.rule_set_stability(n=4, n_gen=3, pop_size=8, pi=0.5)

        assert res['n_fits'] == 4 and len(res['accuracies']) == 4
        assert all(0.0 <= acc <= 1.0 for acc in res['accuracies'])
        assert res['frequencies'] and res['jaccard']['n_pairs'] == 6
        for row in res['frequencies']:
            assert 1 <= row['count'] <= 4
            assert row['frequency'] == row['count'] / 4
            assert row['consequent'] in (0, 1, 2)
        freqs = [row['frequency'] for row in res['frequencies']]
        assert freqs == sorted(freqs, reverse=True)
        jaccard = res['jaccard']
        assert 0.0 <= jaccard['min'] <= jaccard['mean'] <= jaccard['max'] <= 1.0 and jaccard['sd'] >= 0.0
        assert all(row['frequency'] >= 0.5 for row in res['consensus'])
        assert len(res['consensus']) == sum(f >= 0.5 for f in freqs)

        # The consensus loads as a rule base with the library's own scores.
        keys = [(row['antecedents'], row['consequent']) for row in res['consensus']]
        if keys:
            fit_bases = stabilizer.generate_solutions(2, n_gen=3, pop_size=8)[0]
            consensus = ps.consensus_rule_base(keys, fit_bases[0], X, y)
            assert ps.rule_keys(consensus) == sorted(keys, key=lambda key: (str(key[1]), key[0]))
            assert consensus.winning_rule_predict(X).shape == y.shape

    def test_requires_shared_variables_and_valid_arguments(self, iris, lvs):
        X, y = iris
        with pytest.raises(ValueError, match='linguistic_variables'):
            ps.pattern_stabilizer(X, y).rule_set_stability(n=2)
        stabilizer = ps.pattern_stabilizer(X, y, linguistic_variables=lvs)
        with pytest.raises(ValueError, match='at least 2'):
            stabilizer.rule_set_stability(n=1)
        with pytest.raises(ValueError, match='pi'):
            stabilizer.rule_set_stability(n=2, pi=1.5)
