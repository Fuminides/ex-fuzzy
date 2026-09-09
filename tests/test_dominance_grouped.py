"""Grouped dominance scoring must equal the per-rule reference bit for bit."""
import numpy as np
import pytest

import _fitness


def _random_case(rng, samples, rules, classes, interval, layout):
    shape = (samples, rules, 2) if interval else (samples, rules)
    firing = rng.random(shape)
    firing[rng.random(shape) < 0.3] = 0.0
    if rules > 1 and rng.random() < 0.3:
        firing[:, 0] = 0.0  # a rule that never fires: zero denominator
    if layout == 'fortran':
        firing = np.asfortranarray(firing)
    elif layout == 'strided':
        padded = np.zeros(tuple(2 * s for s in shape))
        padded[tuple(slice(None, None, 2) for _ in shape)] = firing
        firing = padded[tuple(slice(None, None, 2) for _ in shape)]
    y = rng.integers(0, classes, size=samples)
    consequents = np.sort(rng.integers(0, classes, size=rules))
    return firing, y, consequents


@pytest.mark.parametrize('interval', [False, True])
@pytest.mark.parametrize('layout', ['c', 'fortran', 'strided'])
def test_grouped_dominance_matches_reference(interval, layout):
    rng = np.random.default_rng(11)
    for samples in (1, 3, 17, 180, 999, 4097):
        for rules in (1, 2, 5, 20, 33):
            for classes in (1, 2, 3, 5):
                firing, y, consequents = _random_case(
                    rng, samples, rules, classes, interval, layout)
                expected = _fitness._dominance_reference(firing, y, consequents)
                actual = _fitness._dominance(firing, y, consequents)
                np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('interval', [False, True])
def test_grouped_dominance_uses_the_shared_mask_cache(interval):
    rng = np.random.default_rng(5)
    firing, y, consequents = _random_case(rng, 250, 12, 3, interval, 'c')
    cache = _fitness._ClassMaskCache(y)
    expected = _fitness._dominance_reference(firing, y, consequents)
    np.testing.assert_array_equal(
        _fitness._dominance(firing, y, consequents, cache), expected)
    assert cache.entries  # the grouped path still reuses cached masks


@pytest.mark.parametrize('interval', [False, True])
def test_grouped_dominance_handles_absent_classes_and_no_rules(interval):
    y = np.array([0, 0, 1, 1, 2])
    shape = (5, 3, 2) if interval else (5, 3)
    firing = np.zeros(shape)
    firing[0] = 0.5
    consequents = np.array([0, 4, 4])  # class 4 never appears in y
    np.testing.assert_array_equal(
        _fitness._dominance(firing, y, consequents),
        _fitness._dominance_reference(firing, y, consequents))
    empty = np.zeros((5, 0, 2) if interval else (5, 0))
    assert _fitness._dominance(empty, y, np.array([], dtype=int)).shape == (0,)


@pytest.mark.parametrize('interval', [False, True])
@pytest.mark.parametrize('consequents', [
    [2, 0, 1, 2, 0, 1, 2, 0],   # never grouped
    [0, 0, 1, 1, 0, 0, 2, 2],   # one class split across two runs
    [2, 2, 2, 1, 1, 0, 0, 0],   # grouped but descending
    [1, 1, 1, 1, 1, 1, 1, 1],   # a single run
])
def test_grouped_dominance_handles_any_consequent_ordering(interval, consequents):
    rng = np.random.default_rng(23)
    firing, y, _ = _random_case(rng, 120, 8, 3, interval, 'c')
    consequents = np.array(consequents)
    np.testing.assert_array_equal(
        _fitness._dominance(firing, y, consequents),
        _fitness._dominance_reference(firing, y, consequents))


def test_consequent_runs_splits_maximal_equal_runs():
    assert list(_fitness._consequent_runs(np.array([0, 0, 1, 1, 1, 0]))) == [
        (0, 2), (2, 5), (5, 6)]
    assert list(_fitness._consequent_runs(np.array([7]))) == [(0, 1)]
