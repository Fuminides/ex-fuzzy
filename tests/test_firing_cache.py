"""Fit-local firing reuse must be exact, bounded and scoped to one fit."""
import numpy as np
import pytest
from sklearn.datasets import load_iris

import _array_fitness as arrfit
import evolutionary_fit as evf
import fuzzy_sets as fs
import rules
import utils
from _fitness import _FiringCache, _fitness_cache_scope


@pytest.mark.parametrize('kind', [fs.FUZZY_SETS.t1, fs.FUZZY_SETS.t2])
def test_cached_columns_equal_a_fresh_gather(kind):
    """A cached column must not depend on which rules shared its gather."""
    rng = np.random.default_rng(21)
    X = rng.normal(size=(400, 6))
    truth = rules.compute_antecedents_memberships(
        utils.construct_partitions(X, kind), X)
    interval = kind == fs.FUZZY_SETS.t2
    antecedents = rng.integers(-1, 3, size=(40, 6))
    antecedents[0] = -1
    cache = _FiringCache()
    expected = arrfit.firing_strengths(antecedents, truth, 400, interval)
    # Warm the cache from differently sized rule subsets, then reassemble.
    for block in (slice(0, 7), slice(7, 33), slice(5, 40), slice(0, 40)):
        actual = arrfit.firing_strengths(antecedents[block], truth, 400,
                                         interval, cache)
        np.testing.assert_array_equal(actual, expected[:, block])
    np.testing.assert_array_equal(
        arrfit.firing_strengths(antecedents, truth, 400, interval, cache), expected)
    # A reordered rule set must reuse the same columns.
    order = rng.permutation(40)
    np.testing.assert_array_equal(
        arrfit.firing_strengths(antecedents[order], truth, 400, interval, cache),
        expected[:, order])


def test_firing_cache_respects_its_byte_budget():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(500, 4))
    truth = rules.compute_antecedents_memberships(
        utils.construct_partitions(X, fs.FUZZY_SETS.t1), X)
    antecedents = rng.integers(0, 3, size=(60, 4))
    cache = _FiringCache(max_bytes=500 * 8 * 4)  # room for four columns
    arrfit.firing_strengths(antecedents, truth, 500, False, cache)
    assert cache.bytes <= cache.max_bytes
    assert 0 < len(cache.entries) <= 4
    tiny = _FiringCache(max_bytes=1)
    arrfit.firing_strengths(antecedents, truth, 500, False, tiny)
    assert not tiny.entries  # nothing fits, and nothing is retained
    # Repeated antecedents, which different classes can share, must not inflate
    # the accounted size.
    repeated = _FiringCache(max_bytes=500 * 8 * 4)
    duplicated = np.repeat(antecedents[:2], 30, axis=0)
    arrfit.firing_strengths(duplicated, truth, 500, False, repeated)
    assert repeated.bytes == sum(c.nbytes for c in repeated.entries.values())
    assert len(repeated.entries) == 2


@pytest.mark.parametrize('kind', [fs.FUZZY_SETS.t1, fs.FUZZY_SETS.t2])
def test_cached_and_uncached_candidates_score_identically(kind):
    X, y = load_iris(return_X_y=True)
    problem = evf.FitRuleBase(X, y, 12, 3, 3, fuzzy_type=kind,
                              linguistic_variables=utils.construct_partitions(X, kind))
    rng = np.random.default_rng(31)
    genes = rng.integers(problem.xl.astype(int), problem.xu.astype(int) + 1,
                         size=(40, problem.n_var))
    expected = [problem._array_score(gene.copy()) for gene in genes]
    with _fitness_cache_scope(problem, True):
        assert isinstance(problem._firing_cache, _FiringCache)
        actual = [problem._array_score(gene.copy()) for gene in genes]
        # Score every gene twice so most rules come from the cache.
        again = [problem._array_score(gene.copy()) for gene in genes]
    assert actual == expected == again
    assert not hasattr(problem, '_firing_cache')


def test_optimized_partitions_get_no_firing_cache():
    """Optimized memberships change per candidate, so reuse would be wrong."""
    X, y = load_iris(return_X_y=True)
    problem = evf.FitRuleBase(X, y, 8, 2, 3, fuzzy_type=fs.FUZZY_SETS.t1)
    with _fitness_cache_scope(problem, True):
        assert not hasattr(problem, '_firing_cache')


def test_firing_cache_is_discarded_when_optimization_fails():
    X, y = load_iris(return_X_y=True)
    problem = evf.FitRuleBase(X, y, 8, 2, 3, fuzzy_type=fs.FUZZY_SETS.t1,
                              linguistic_variables=utils.construct_partitions(X, fs.FUZZY_SETS.t1))
    with pytest.raises(RuntimeError):
        with _fitness_cache_scope(problem, True):
            raise RuntimeError('optimizer failed')
    assert not hasattr(problem, '_firing_cache')


@pytest.mark.parametrize('kind', [fs.FUZZY_SETS.t1, fs.FUZZY_SETS.t2])
def test_seeded_fits_match_with_the_firing_cache_disabled(monkeypatch, kind):
    X, y = load_iris(return_X_y=True)
    partitions = utils.construct_partitions(X, kind)
    outcomes = []
    for enabled in (True, False):
        if not enabled:
            monkeypatch.setattr(evf.FitRuleBase, '_array_score',
                                lambda self, x, **kw: _uncached(self, x, **kw))
        model = evf.BaseFuzzyRulesClassifier(
            nRules=10, nAnts=3, linguistic_variables=partitions, fuzzy_type=kind)
        model.fit(X, y, n_gen=6, pop_size=12, random_state=5, patience=None)
        outcomes.append((model.performance,
                         tuple(np.asarray(model.optimization_result_['X']).tolist()),
                         tuple(np.asarray(model.predict(X)).tolist()),
                         tuple(np.asarray(model.rule_base.get_scores()).tolist())))
    assert outcomes[0] == outcomes[1]


_original_array_score = evf.FitRuleBase._array_score


def _uncached(self, x, **kwargs):
    """Evaluate with the firing cache hidden from this problem."""
    cache = getattr(self, '_firing_cache', None)
    if cache is not None:
        del self._firing_cache
    try:
        return _original_array_score(self, x, **kwargs)
    finally:
        if cache is not None:
            self._firing_cache = cache
