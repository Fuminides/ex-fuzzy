"""Fit-scoped ownership tests for the optional PyMoo thread runner."""
from multiprocessing.pool import ThreadPool

import pytest
from sklearn.datasets import load_iris

import _fitness
import evolutionary_fit as evf


pytestmark = pytest.mark.skipif(
    evf.StarmapParallelization is None,
    reason="Installed pymoo version has no starmap runner",
)


class TrackingThreadPool(ThreadPool):
    """Real worker pool with observable lifecycle calls."""
    instances = []

    def __init__(self, *args, **kwargs):
        self.close_calls = 0
        self.join_calls = 0
        super().__init__(*args, **kwargs)
        self.instances.append(self)

    def close(self):
        self.close_calls += 1
        return super().close()

    def join(self):
        self.join_calls += 1
        return super().join()


@pytest.fixture
def iris():
    return load_iris(return_X_y=True)


@pytest.fixture
def tracked_pools(monkeypatch):
    TrackingThreadPool.instances = []
    monkeypatch.setattr(evf, 'ThreadPool', TrackingThreadPool)
    return TrackingThreadPool.instances


def _classifier():
    return evf.BaseFuzzyRulesClassifier(nRules=6, nAnts=3, runner=2)


def _assert_closed(pools, expected):
    assert len(pools) == expected
    for pool in pools:
        assert pool.close_calls == 1
        assert pool.join_calls == 1


def test_constructor_does_not_allocate_workers(tracked_pools):
    classifier = _classifier()
    assert classifier.runner == 2
    assert classifier.thread_runner is None
    assert tracked_pools == []


def test_repeated_normal_fits_close_owned_pools(iris, tracked_pools):
    X, y = iris
    classifier = _classifier()

    classifier.fit(X, y, n_gen=2, pop_size=10, random_state=7, patience=None)
    assert classifier.thread_runner is None
    classifier.fit(X, y, n_gen=2, pop_size=10, random_state=8, patience=None)

    assert classifier.thread_runner is None
    _assert_closed(tracked_pools, 2)


def test_checkpoint_fit_closes_owned_pool(iris, tracked_pools):
    X, y = iris
    checkpoints = []
    classifier = _classifier()

    classifier.fit(
        X, y, n_gen=2, pop_size=10, random_state=7, patience=None,
        checkpoints=1,
        checkpoint_callback=lambda generation, _: checkpoints.append(generation),
    )

    assert checkpoints == [0, 1]
    assert classifier.thread_runner is None
    _assert_closed(tracked_pools, 1)


def test_checkpoint_callback_exception_closes_owned_pool(iris, tracked_pools):
    X, y = iris
    classifier = _classifier()

    def fail_callback(*args):
        raise RuntimeError('checkpoint callback failed')

    with pytest.raises(RuntimeError, match='checkpoint callback failed'):
        classifier.fit(
            X, y, n_gen=2, pop_size=10, random_state=7, patience=None,
            checkpoints=1, checkpoint_callback=fail_callback,
        )

    assert classifier.thread_runner is None
    _assert_closed(tracked_pools, 1)


def test_optimizer_exception_closes_owned_pool(iris, tracked_pools, monkeypatch):
    X, y = iris
    classifier = _classifier()

    def fail(*args, **kwargs):
        raise RuntimeError('optimizer failed')

    monkeypatch.setattr(classifier.backend, 'optimize', fail)
    with pytest.raises(RuntimeError, match='optimizer failed'):
        classifier.fit(X, y, n_gen=2, pop_size=10, random_state=7)

    assert classifier.thread_runner is None
    _assert_closed(tracked_pools, 1)


def test_finalization_exception_closes_owned_pool(iris, tracked_pools, monkeypatch):
    X, y = iris
    classifier = _classifier()

    def fail_finalization(*args, **kwargs):
        raise RuntimeError('finalization failed')

    monkeypatch.setattr(evf.evr, 'evalRuleBase', fail_finalization)
    with pytest.raises(RuntimeError, match='finalization failed'):
        classifier.fit(X, y, n_gen=2, pop_size=10, random_state=7, patience=None)

    assert classifier.thread_runner is None
    _assert_closed(tracked_pools, 1)


def test_external_runner_is_not_closed_or_memoized(iris, tracked_pools, monkeypatch):
    X, y = iris
    classifier = _classifier()
    external_pool = TrackingThreadPool(2)
    classifier.thread_runner = evf.StarmapParallelization(external_pool.starmap)

    def unexpected_cache_lookup(*args, **kwargs):
        pytest.fail('Worker execution must bypass the serial fitness cache')

    monkeypatch.setattr(_fitness._FitnessCache, 'get', unexpected_cache_lookup)
    try:
        classifier.fit(X, y, n_gen=2, pop_size=10, random_state=7, patience=None)
        assert classifier.thread_runner is not None
        assert external_pool.close_calls == 0
        assert external_pool.join_calls == 0
        # The manually supplied runner suppresses creation of an owned pool.
        assert tracked_pools == [external_pool]
    finally:
        external_pool.close()
        external_pool.join()


def test_sklearn_parameter_introspection_retains_runner_when_supported():
    """New ``runner`` state participates in clone when this estimator supports it."""
    from sklearn.base import clone

    classifier = _classifier()
    try:
        parameters = classifier.get_params(deep=False)
    except AttributeError as error:
        pytest.skip(f'pre-existing estimator parameter introspection is unavailable: {error}')

    assert parameters['runner'] == 2
    cloned = clone(classifier)
    assert cloned.runner == 2
    assert cloned.thread_runner is None
