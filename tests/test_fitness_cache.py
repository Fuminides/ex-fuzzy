"""Memoization preserves logical evaluations and the seeded search."""
from contextlib import nullcontext

import numpy as np
import pytest
from sklearn.datasets import load_iris

import _fitness
import evolutionary_fit as evf
import fuzzy_sets as fs
import utils


def test_lru_eviction_and_byte_limit():
    cache = _fitness._FitnessCache(capacity=2, max_key_bytes=8)
    cache.put(b'a', 0.)
    cache.put(b'b', 1.)
    assert cache.get(b'a') == 0.
    cache.put(b'c', 2.)
    assert cache.get(b'b') is None
    cache.put(b'12345678', 3.)
    assert list(cache.entries) == [b'12345678']
    assert cache.key_bytes == 8
    cache.put(b'oversized', 4.)
    assert cache.key_bytes == 8
    assert cache.key(np.ones(100)) is None


def test_keys_preserve_dtype_and_float_encoding():
    cache = _fitness._FitnessCache()
    assert cache.key(np.array([1])) != cache.key(np.array([1.]))
    assert cache.key(np.array([0.])) != cache.key(np.array([-0.]))
    assert cache.key(np.array([1, 2])[::2]) == cache.key(np.array([1]))
    assert cache.key(np.array(['a'])) is None
    assert cache.key(np.ones((2, 2))) is None


def test_scope_discards_cache_on_exception():
    problem = type('Problem', (), {})()
    with pytest.raises(RuntimeError):
        with _fitness._fitness_cache_scope(problem, True):
            problem._fitness_cache.put(b'a', 1.)
            raise RuntimeError('optimizer failed')
    assert not hasattr(problem, '_fitness_cache')


@pytest.mark.parametrize('fixed', [False, True])
@pytest.mark.parametrize('kind', [fs.FUZZY_SETS.t1, fs.FUZZY_SETS.t2])
@pytest.mark.parametrize('ds_mode', [0, 1, 2])
def test_cached_fit_matches_uncached_search(monkeypatch, fixed, kind, ds_mode):
    X, y = load_iris(return_X_y=True)
    partitions = utils.construct_partitions(X, kind) if fixed else None
    options = dict(nRules=10, nAnts=3, linguistic_variables=partitions, fuzzy_type=kind,
                   ds_mode=ds_mode, allow_unknown=True)
    trace = []
    problems = []
    evaluate = evf.FitRuleBase._evaluate
    construct = evf.FitRuleBase._construct_ruleBase
    array_score = evf.FitRuleBase._array_score
    # Count objective computations, whichever evaluator produced them: a cache
    # hit must skip the work entirely.
    decodes = []

    def observed(self, gene, out, *args, **kwargs):
        evaluate(self, gene, out, *args, **kwargs)
        trace.append((gene.copy(), out['F']))
        if not problems or problems[-1] is not self:
            problems.append(self)

    def decoded(self, *args, **kwargs):
        decodes.append(1)
        return construct(self, *args, **kwargs)

    def scored(self, *args, **kwargs):
        result = array_score(self, *args, **kwargs)
        if result is not None:
            decodes.append(1)
        return result

    monkeypatch.setattr(evf.FitRuleBase, '_evaluate', observed)
    monkeypatch.setattr(evf.FitRuleBase, '_construct_ruleBase', decoded)
    monkeypatch.setattr(evf.FitRuleBase, '_array_score', scored)
    fast = evf.BaseFuzzyRulesClassifier(**options)
    fast.fit(X, y, n_gen=20, pop_size=20, random_state=7, patience=5)
    cached_trace, cached_decodes = trace[:], len(decodes)
    assert all(not hasattr(p, '_fitness_cache') for p in problems)
    trace.clear()
    decodes.clear()
    monkeypatch.setattr(_fitness, '_fitness_cache_scope', lambda *args: nullcontext())
    slow = evf.BaseFuzzyRulesClassifier(**options)
    slow.fit(X, y, n_gen=20, pop_size=20, random_state=7, patience=5)
    assert len(trace) == len(cached_trace)
    assert cached_decodes < len(decodes)
    for (a, fa), (b, fb) in zip(trace, cached_trace):
        np.testing.assert_array_equal(a, b)
        assert fa == fb
    assert fast.n_generations_run_ == slow.n_generations_run_
    assert fast.stopped_early_ == slow.stopped_early_
    assert fast.optimization_result_['algorithm'].evaluator.n_eval == slow.optimization_result_['algorithm'].evaluator.n_eval
    np.testing.assert_array_equal(fast.optimization_result_['pop'].get('X'), slow.optimization_result_['pop'].get('X'))
    np.testing.assert_array_equal(fast.optimization_result_['pop'].get('F'), slow.optimization_result_['pop'].get('F'))
    np.testing.assert_array_equal(fast.optimization_result_['X'], slow.optimization_result_['X'])
    np.testing.assert_array_equal(fast.predict(X), slow.predict(X))
    np.testing.assert_array_equal(fast.rule_base.get_scores(), slow.rule_base.get_scores())


def test_custom_loss_never_uses_cache(monkeypatch):
    X, y = load_iris(return_X_y=True)
    model = evf.BaseFuzzyRulesClassifier(nRules=8, nAnts=3)
    calls = []

    def loss(*args, **kwargs):
        calls.append(1)
        return .5

    def unexpected(*args):
        pytest.fail('Custom loss reached memoization')

    model.custom_loss = loss
    monkeypatch.setattr(_fitness._FitnessCache, 'get', unexpected)
    model.fit(X, y, n_gen=3, pop_size=10, random_state=7, patience=None)
    assert calls


@pytest.mark.parametrize('mode', ['checkpoints', 'workers'])
def test_external_execution_paths_bypass_cache(monkeypatch, mode):
    from multiprocessing.pool import ThreadPool
    X, y = load_iris(return_X_y=True)
    model = evf.BaseFuzzyRulesClassifier(nRules=8, nAnts=3)

    def unexpected(*args):
        pytest.fail('External execution path reached memoization')

    monkeypatch.setattr(_fitness._FitnessCache, 'get', unexpected)
    if mode == 'checkpoints':
        checkpoints = []
        model.fit(X, y, n_gen=3, pop_size=10, random_state=7, patience=None,
                  checkpoints=1, checkpoint_callback=lambda gen, base: checkpoints.append(gen))
        assert checkpoints == [0, 1, 2]
    else:
        with ThreadPool(2) as pool:
            model.thread_runner = evf.StarmapParallelization(pool.starmap)
            model.fit(X, y, n_gen=3, pop_size=10, random_state=7, patience=None)
