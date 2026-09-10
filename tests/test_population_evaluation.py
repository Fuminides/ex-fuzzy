"""C01 population evaluation stays exact and keeps scalar fallbacks."""
from contextlib import nullcontext
from multiprocessing.pool import ThreadPool
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.datasets import load_iris, make_classification

import evolutionary_fit as evf
import fuzzy_sets as fs
import utils
from _fitness import _fitness_cache_scope
from _population_fitness import supports_shape


def _problem(samples=150, **kwargs):
    X, y = make_classification(
        n_samples=samples, n_features=10, n_informative=9,
        n_redundant=0, n_classes=3, random_state=42)
    partitions = utils.construct_partitions(X, fs.FUZZY_SETS.t1)
    return evf.FitRuleBase(
        X, y, kwargs.pop('nRules', 20), kwargs.pop('nAnts', 4), 3,
        linguistic_variables=partitions, **kwargs)


def _population(problem, count=40, seed=7):
    return np.random.default_rng(seed).integers(
        problem.xl.astype(int), problem.xu.astype(int) + 1,
        size=(count, problem.n_var))


def _batched(problem):
    with _fitness_cache_scope(problem, True):
        return problem._population_scores(_population(problem))


@pytest.mark.parametrize('ds_mode', [0, 1])
@pytest.mark.parametrize('allow_unknown', [False, True])
@pytest.mark.parametrize('tolerance', [0.0, 0.01, 0.5, 2.0])
@pytest.mark.parametrize('alpha,beta', [(0.0, 0.0), (0.1, 0.2)])
def test_population_scores_equal_scalar_scores(
        ds_mode, allow_unknown, tolerance, alpha, beta):
    problem = _problem(
        ds_mode=ds_mode, allow_unknown=allow_unknown, tolerance=tolerance,
        alpha=alpha, beta=beta)
    genes = _population(problem)
    expected = np.asarray([problem._array_score(gene) for gene in genes])
    with _fitness_cache_scope(problem, True):
        actual = problem._population_scores(genes)
    assert actual is not None
    np.testing.assert_array_equal(actual, expected)


def test_population_handles_empty_and_duplicate_phenotypes():
    problem = _problem(nRules=6, nAnts=2)
    random = _population(problem, count=6)
    empty = np.zeros(problem.n_var, dtype=int)
    empty[2 * problem.nRules * problem.nAnts:] = -1
    duplicate = np.zeros(problem.n_var, dtype=int)
    genes = np.vstack((empty, duplicate, random))
    expected = [problem._array_score(gene) for gene in genes]
    with _fitness_cache_scope(problem, True):
        actual = problem._population_scores(genes)
    np.testing.assert_array_equal(actual, expected)


def test_population_returns_entirely_cached_population_without_rescoring():
    problem = _problem()
    genes = _population(problem)
    with _fitness_cache_scope(problem, True):
        first = {}
        problem._evaluate_elementwise(genes, first)
        with patch.object(problem, '_population_scores',
                          side_effect=AssertionError('should not rescore cache hits')):
            second = {}
            problem._evaluate_elementwise(genes, second)
    np.testing.assert_array_equal(second['F'], first['F'])


def test_population_path_declines_unsupported_execution_routes():
    ordinary = _problem()

    weighted = _problem(ds_mode=2)
    assert _batched(weighted) is None

    large = _problem(samples=600)
    assert _batched(large) is None

    X, y = load_iris(return_X_y=True)
    optimized = evf.FitRuleBase(X, y, 10, 3, 3)
    assert _batched(optimized) is None
    interval = evf.FitRuleBase(
        X, y, 10, 3, 3, fuzzy_type=fs.FUZZY_SETS.t2,
        linguistic_variables=utils.construct_partitions(X, fs.FUZZY_SETS.t2))
    assert _batched(interval) is None

    custom = _problem()
    custom.fitness_func = lambda *args: 0.5
    assert _batched(custom) is None

    class Derived(evf.FitRuleBase):
        pass

    derived = Derived(
        ordinary.X, ordinary.y, 20, 4, 3,
        linguistic_variables=utils.construct_partitions(
            ordinary.X, fs.FUZZY_SETS.t1))
    assert _batched(derived) is None

    with ThreadPool(1) as pool:
        runner = evf.StarmapParallelization(pool.starmap)
        external = evf.FitRuleBase(
            ordinary.X, ordinary.y, 20, 4, 3,
            linguistic_variables=utils.construct_partitions(
                ordinary.X, fs.FUZZY_SETS.t1), thread_runner=runner)
        assert _batched(external) is None

    # The packed MCC layout only represents the classifier's integer class
    # domain.  Direct Problem users can supply other numeric labels, for which
    # the scalar evaluator remains the oracle.
    non_contiguous = _problem()
    non_contiguous.y = non_contiguous.y * 2 + 10
    assert _batched(non_contiguous) is None
    floating_labels = _problem()
    floating_labels.y = floating_labels.y.astype(float)
    assert _batched(floating_labels) is None


def test_population_memory_boundary_at_512_samples():
    # This is the largest supported sample count.  Forty 20-rule, 10-feature
    # candidates fit the conservative gather budget; one more does not.
    assert supports_shape(40, 512, 20, 10)
    assert not supports_shape(41, 512, 20, 10)


def test_seeded_fit_preserves_search_and_logical_evaluation_count():
    X, y = load_iris(return_X_y=True)
    partitions = utils.construct_partitions(X, fs.FUZZY_SETS.t1)
    outcomes = []
    for batched in (False, True):
        context = (nullcontext() if batched else patch.object(
            evf.FitRuleBase, '_evaluate_elementwise',
            evf.Problem._evaluate_elementwise))
        model = evf.BaseFuzzyRulesClassifier(
            nRules=12, nAnts=3, linguistic_variables=partitions,
            ds_mode=0, tolerance=0.01)
        with context:
            model.fit(X, y, n_gen=8, pop_size=20, random_state=7,
                      patience=None)
        outcomes.append((
            model.performance,
            model.optimization_result_['algorithm'].evaluator.n_eval,
            model.optimization_result_['pop'].get('X'),
            model.optimization_result_['pop'].get('F'),
            model.optimization_result_['X'],
            model.rule_base.get_scores(),
            model.predict(X),
        ))
    for left, right in zip(*outcomes):
        np.testing.assert_array_equal(left, right)
