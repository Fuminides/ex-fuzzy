"""Parity and bounds tests for evaluation-local dominance masks."""
import numpy as np
import pytest
from sklearn.datasets import load_iris

import _fitness
import evolutionary_fit as evf
from _fitness import _ClassMaskCache, _dominance, score_rulebase


def _scalar_dominance(firing, y, consequents):
    """The pre-mask-cache reduction, kept as a small numerical oracle."""
    scores = np.empty(len(consequents))
    for index, consequent in enumerate(consequents):
        values = firing[:, index]
        match = y == consequent
        if firing.ndim == 3:
            support = np.mean([
                np.mean(values[:, 0] * match),
                np.mean(values[:, 1] * match),
            ])
        else:
            support = np.mean(values * match)
        denominator = np.sum(values)
        confidence = np.sum(values[match]) / denominator if denominator != 0 else 0.0
        scores[index] = support * confidence
    return scores


@pytest.mark.parametrize('interval', [False, True])
@pytest.mark.parametrize('layout', ['c', 'f', 'strided'])
@pytest.mark.parametrize('labels', [
    np.array([0, 1, 2, 0, 2, 1]),
    np.array([-1, 10, 30, -1, 30, 10]),
])
def test_dominance_mask_cache_matches_scalar_oracle(interval, layout, labels):
    rng = np.random.default_rng(12)
    shape = (len(labels), 7, 2) if interval else (len(labels), 7)
    firing = rng.random(shape)
    if layout == 'f':
        firing = np.asfortranarray(firing)
    elif layout == 'strided':
        source_shape = (len(labels) * 2, 7, 2) if interval else (len(labels) * 2, 7)
        source = rng.random(source_shape)
        firing = source[::2]
    firing[:, 5] = 0.0
    consequents = np.array([0, 1, 0, 2, 1, 99, 88])
    if labels.min() < 0:
        consequents = np.array([-1, 10, -1, 30, 10, 99, 88])

    expected = _scalar_dominance(firing, labels, consequents)
    actual = _dominance(firing, labels, consequents, _ClassMaskCache(labels))

    np.testing.assert_array_equal(actual, expected)


def test_dominance_three_argument_call_remains_supported():
    y = np.array([0, 1, 0])
    firing = np.array([[0.2, 0.8], [0.4, 0.6], [0.3, 0.7]])
    consequents = np.array([0, 1])
    np.testing.assert_array_equal(
        _dominance(firing, y, consequents),
        _scalar_dominance(firing, y, consequents),
    )


def test_class_mask_cache_reuses_masks_and_honors_byte_cap():
    y = np.array([0, 1, 2, 0, 1, 2])
    cache = _ClassMaskCache(y, max_bytes=y.size)

    first = cache.get(0)
    assert cache.get(0) is first
    assert cache.bytes == first.nbytes == y.size
    uncached = cache.get(1)
    assert cache.get(1) is not uncached
    assert cache.bytes == y.size
    assert set(cache.entries) == {0}


def test_class_mask_caches_are_evaluation_local():
    y = np.array([0, 1, 0, 1])
    first = _ClassMaskCache(y)
    second = _ClassMaskCache(y)
    first.get(0)

    assert set(first.entries) == {0}
    assert second.entries == {}


def test_score_rulebase_shares_masks_only_within_one_evaluation(monkeypatch):
    X, y = load_iris(return_X_y=True)
    problem = evf.FitRuleBase(X, y, 10, 2, 3)
    rng = np.random.default_rng(43)
    gene = rng.integers(problem.xl.astype(int), problem.xu.astype(int) + 1)
    caches = []
    original = _fitness._dominance

    def observed(firing, labels, consequents, mask_cache=None):
        caches.append(mask_cache)
        return original(firing, labels, consequents, mask_cache)

    monkeypatch.setattr(_fitness, '_dominance', observed)
    score_rulebase(problem._construct_ruleBase(gene, problem.fuzzy_type), X, y, 0.0, 0.0, 0.0)
    score_rulebase(problem._construct_ruleBase(gene, problem.fuzzy_type), X, y, 0.0, 0.0, 0.0)

    assert len(caches) == 4
    assert caches[0] is caches[1]
    assert caches[2] is caches[3]
    assert caches[0] is not caches[2]
