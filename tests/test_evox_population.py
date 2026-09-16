"""EvoX population scoring stays exact: caches, batching and the device objective."""
import sys
import types
from contextlib import nullcontext
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.datasets import make_classification

from ex_fuzzy import _fitness
from ex_fuzzy import evolutionary_backends as eb
from ex_fuzzy import evolutionary_fit as evf
from ex_fuzzy import fuzzy_sets as fs
from ex_fuzzy import utils
from ex_fuzzy._fitness import _dominance, _fitness_cache_scope


def _data(samples=150, classes=3):
    return make_classification(
        n_samples=samples, n_features=6, n_informative=4, n_redundant=0,
        n_classes=classes, n_clusters_per_class=1, random_state=42)


def _problem(samples=150, fixed=True, **kwargs):
    X, y = _data(samples)
    partitions = utils.construct_partitions(X, fs.FUZZY_SETS.t1) if fixed else None
    return evf.FitRuleBase(
        X, y, kwargs.pop('nRules', 10), kwargs.pop('nAnts', 3), 3,
        linguistic_variables=partitions, **kwargs)


def _population(problem, count=24, seed=7):
    """Random chromosomes plus an empty and a duplicate-rule phenotype."""
    genes = np.random.default_rng(seed).integers(
        problem.xl.astype(int), problem.xu.astype(int) + 1, size=(count, problem.n_var))
    fourth = problem._consequent_pointer(problem.fuzzy_type)
    slots, ants = problem.nRules * problem.nAnts, problem.nAnts
    empty = genes[0].copy()
    empty[fourth:fourth + problem.nRules] = -1
    duplicate = genes[1].copy()
    duplicate[ants:2 * ants] = duplicate[:ants]
    duplicate[slots + ants:slots + 2 * ants] = duplicate[slots:slots + ants]
    duplicate[fourth + 1] = duplicate[fourth]
    return np.vstack((genes, empty, duplicate))


def _fitness_values(problem, genes):
    with np.errstate(all='ignore'):
        return np.asarray([1 - problem._array_score(gene) for gene in genes])


def test_scope_sizes_the_genotype_cache_to_the_population():
    problem = _problem()
    with _fitness_cache_scope(problem, True):
        assert problem._fitness_cache.capacity == 256
    with _fitness_cache_scope(problem, True, population=200):
        cache = problem._fitness_cache
        assert cache.capacity == 800
        assert cache.max_key_bytes >= 800 * 8 * problem.n_var
        assert problem._torch_route is None
    for name in ('_fitness_cache', '_route_probe', '_torch_route', '_firing_cache'):
        assert not hasattr(problem, name)


def test_repeated_genotypes_within_a_population_are_scored_once():
    problem = _problem()
    genes = _population(problem, 20)
    repeated = np.vstack((genes, genes[:5], genes[3:4]))
    with _fitness_cache_scope(problem, True):
        with patch.object(problem, '_score_fresh', wraps=problem._score_fresh) as spy:
            out = {}
            problem._evaluate_elementwise(repeated, out)
    assert spy.call_count == 1
    assert len(spy.call_args.args[0]) == len(genes)
    np.testing.assert_array_equal(out['F'], _fitness_values(problem, repeated))


@pytest.mark.parametrize('fixed', [True, False])
def test_gene_population_matches_scalar_evaluation(fixed):
    problem = _problem(fixed=fixed)
    genes = _population(problem)
    genes = np.vstack((genes, genes[::3]))
    expected = _fitness_values(problem, genes)

    unscoped, on_device = problem._evaluate_gene_population(genes)
    np.testing.assert_array_equal(unscoped, expected)
    assert not on_device

    with _fitness_cache_scope(problem, True, population=len(genes)):
        first, _ = problem._evaluate_gene_population(genes)
        with patch.object(problem, '_score_on_best_route',
                          side_effect=AssertionError('cached genotypes were rescored')):
            second, _ = problem._evaluate_gene_population(genes)
    np.testing.assert_array_equal(first, expected)
    np.testing.assert_array_equal(second, expected)


def test_batched_complexity_penalty_when_survivors_score_exactly_the_tolerance():
    # With one class every survivor must clear the strict tolerance to count,
    # so a tolerance equal to the best rule score leaves no selected rule.
    X, _ = _data()
    y = np.zeros(len(X), dtype=int)
    partitions = utils.construct_partitions(X, fs.FUZZY_SETS.t1)
    probe = evf.FitRuleBase(X, y, 6, 3, 1, linguistic_variables=partitions)
    checked = 0
    for gene in _population(probe, 30, seed=3):
        problem, score = _problem_at_best_score(probe, gene, X, y, partitions)
        if problem is None:
            continue
        with _fitness_cache_scope(problem, True):
            batched = problem._population_scores(np.vstack((gene, gene)))
        np.testing.assert_array_equal(batched, [score, score])
        checked += 1
    assert checked


def _problem_at_best_score(probe, gene, X, y, partitions):
    from ex_fuzzy import _array_fitness as arrfit
    decoded = arrfit.decode_rule_arrays(
        gene, probe.nRules, probe.nAnts, X.shape[1], 1,
        np.asarray([len(variable) for variable in partitions]), 0,
        probe._consequent_pointer(probe.fuzzy_type))
    if decoded is None or not len(decoded.consequents):
        return None, None
    firing = arrfit.firing_strengths(decoded.antecedents, probe._precomputed_truth,
                                     len(X), False)
    scores = _dominance(firing, y, decoded.consequents)
    if not np.any(scores > 0):
        return None, None
    problem = evf.FitRuleBase(X, y, 6, 3, 1, linguistic_variables=partitions,
                              tolerance=float(scores.max()), alpha=0.1, beta=0.2)
    return problem, problem._array_score(gene)


# ---------------------------------------------------------------------------
# Exact PyTorch objective, exercised on CPU tensors.

@pytest.fixture
def torchfit():
    pytest.importorskip('torch')
    from ex_fuzzy import _torch_fitness
    return _torch_fitness


def test_pairwise_sum_reproduces_numpy_bit_for_bit(torchfit):
    import torch
    rng = np.random.default_rng(11)
    for length in (0, 1, 7, 8, 9, 15, 16, 127, 128, 129, 255, 256, 257, 1000, 8193, 20011):
        values = rng.random((3, 2, length))
        if length:
            values[..., ::3] *= 1e-12
            values[..., 1::7] *= 1e12
        expected = np.sum(values, axis=2)
        actual = torchfit.pairwise_sum(torch.as_tensor(values)).numpy()
        np.testing.assert_array_equal(expected.view(np.uint64), actual.view(np.uint64))


@pytest.mark.parametrize('fixed', [True, False])
@pytest.mark.parametrize('ds_mode', [0, 1])
@pytest.mark.parametrize('allow_unknown', [False, True])
@pytest.mark.parametrize('tolerance', [0.0, 0.01, 0.3])
@pytest.mark.parametrize('alpha,beta', [(0.0, 0.0), (0.1, 0.2)])
def test_torch_objective_equals_array_evaluator(
        torchfit, fixed, ds_mode, allow_unknown, tolerance, alpha, beta):
    problem = _problem(fixed=fixed, ds_mode=ds_mode, allow_unknown=allow_unknown,
                       tolerance=tolerance, alpha=alpha, beta=beta)
    genes = _population(problem)
    objective = torchfit.TorchObjective.build(problem, 'cpu')
    scores, declined = objective.score(genes)
    assert not declined.any()
    np.testing.assert_array_equal(1 - scores, _fitness_values(problem, genes))


@pytest.mark.parametrize('samples', [1, 7, 1000])
def test_torch_objective_handles_tiny_and_larger_datasets(torchfit, samples):
    for fixed in (True, False):
        problem = (_problem(samples=samples, fixed=fixed, alpha=0.1) if samples >= 60
                   else _subset_problem(samples, fixed))
        genes = _population(problem, 12)
        objective = torchfit.TorchObjective.build(problem, 'cpu')
        scores, declined = objective.score(genes)
        np.testing.assert_array_equal((1 - scores)[~declined],
                                      _fitness_values(problem, genes)[~declined])


def _subset_problem(samples, fixed):
    X, y = _data(60)
    X, y = X[:samples], y[:samples]
    partitions = utils.construct_partitions(X, fs.FUZZY_SETS.t1) if fixed else None
    return evf.FitRuleBase(X, y, 10, 3, 3, linguistic_variables=partitions, alpha=0.1)


def test_torch_objective_chunking_and_declined_candidates(torchfit):
    problem = _problem(fixed=False)
    genes = _population(problem)
    flat = genes[2].copy()
    flat[2 * problem.nRules * problem.nAnts:problem._consequent_pointer(problem.fuzzy_type)] = 0
    genes = np.vstack((genes, flat))
    objective = torchfit.TorchObjective.build(problem, 'cpu')
    whole, declined = objective.score(genes)
    objective.chunk = 5
    chunked, chunk_declined = objective.score(genes)
    np.testing.assert_array_equal(whole[~declined], chunked[~declined])
    np.testing.assert_array_equal(declined, chunk_declined)
    # Every coinciding partition parameter normalizes by zero on the CPU.
    assert declined.tolist() == [False] * (len(genes) - 1) + [True]
    assert objective.score(genes[:, :-1])[1].all()


def test_torch_objective_matches_the_tolerance_edge(torchfit):
    X, _ = _data()
    y = np.zeros(len(X), dtype=int)
    partitions = utils.construct_partitions(X, fs.FUZZY_SETS.t1)
    probe = evf.FitRuleBase(X, y, 6, 3, 1, linguistic_variables=partitions)
    checked = 0
    for gene in _population(probe, 30, seed=3):
        problem, score = _problem_at_best_score(probe, gene, X, y, partitions)
        if problem is None:
            continue
        scores, declined = torchfit.TorchObjective.build(problem, 'cpu').score(gene[None, :])
        assert not declined[0] and scores[0] == score
        checked += 1
    assert checked


def test_torch_objective_declines_unsupported_problems(torchfit):
    X, y = _data()
    t2 = evf.FitRuleBase(X, y, 10, 3, 3, fuzzy_type=fs.FUZZY_SETS.t2,
                         linguistic_variables=utils.construct_partitions(X, fs.FUZZY_SETS.t2))
    weighted = _problem(ds_mode=2)
    custom = _problem()
    custom.fitness_func = lambda *args: 0.5
    categorical = evf.FitRuleBase(np.column_stack((X, np.arange(len(X)) % 3)), y, 10, 3, 3,
                                  categorical_mask=np.array([0] * 6 + [1]))
    for problem in (t2, weighted, custom, categorical):
        with patch.object(evf.FitRuleBase, 'torch_devices', ('cpu',)), \
                _fitness_cache_scope(problem, True):
            assert problem._device_route_state('cpu') is None
    ordinary = _problem()
    with _fitness_cache_scope(ordinary, True):
        # CPU tensors are not a default device route.
        assert ordinary._device_route_state('cpu') is None
    assert ordinary._device_route_state('cpu') is None  # no fit scope


def test_device_route_is_verified_before_use_and_stays_exact(torchfit):
    problem = _problem()
    used = []
    with patch.object(evf.FitRuleBase, 'torch_devices', ('cpu',)), \
            _fitness_cache_scope(problem, True, population=26):
        for generation in range(7):
            genes = _population(problem, 24, seed=100 + generation)
            fitness, on_device = problem._evaluate_gene_population(genes, device='cpu')
            np.testing.assert_array_equal(fitness, _fitness_values(problem, genes))
            used.append(on_device)
        route = problem._torch_route
        assert route.verified and not route.rejected
        assert route.probe.decision is not None
    # Once its CPU sample matches, the verification generation keeps the device scores.
    assert used[0] is True


def test_device_route_is_rejected_after_one_difference(torchfit):
    problem = _problem()

    def wrong(self, genes):
        return np.full(len(genes), 0.25), np.zeros(len(genes), dtype=bool)

    with patch.object(evf.FitRuleBase, 'torch_devices', ('cpu',)), \
            patch.object(torchfit.TorchObjective, 'score', wrong), \
            _fitness_cache_scope(problem, True, population=26):
        for generation in range(4):
            genes = _population(problem, 24, seed=200 + generation)
            fitness, on_device = problem._evaluate_gene_population(genes, device='cpu')
            np.testing.assert_array_equal(fitness, _fitness_values(problem, genes))
            assert not on_device
        assert problem._torch_route.rejected


def test_declined_device_candidates_are_scored_on_the_cpu(torchfit):
    problem = _problem(fixed=False)
    genes = _population(problem)
    objective = torchfit.TorchObjective.build(problem, 'cpu')

    def all_declined(genes):
        return np.zeros(len(genes)), np.ones(len(genes), dtype=bool)

    with patch.object(objective, 'score', all_declined):
        fitness = problem._device_fitness(objective, genes)
    np.testing.assert_array_equal(fitness, _fitness_values(problem, genes))


def test_decisive_verification_timing_settles_on_the_device(torchfit):
    objective = torchfit.TorchObjective.build(_problem(), 'cpu')
    values = np.arange(4.0)
    fast = torchfit.DeviceRoute(objective)
    assert fast.verify(values, values.copy(), cpu_seconds=4.0, device_seconds=1.0)
    assert fast.probe.decision == fast.DEVICE
    close = torchfit.DeviceRoute(objective)
    assert close.verify(values, values.copy(), cpu_seconds=2.5, device_seconds=1.0)
    assert close.probe.decision is None
    wrong = torchfit.DeviceRoute(objective)
    assert not wrong.verify(values, values + 1, cpu_seconds=9.0, device_seconds=1.0)
    assert wrong.rejected and wrong.probe.decision is None


def test_verification_scores_only_a_sample_on_the_cpu(torchfit):
    problem = _problem()
    genes = _population(problem, 24, seed=300)
    expected = _fitness_values(problem, genes)
    rows = []
    original = evf.FitRuleBase._scalar_scores

    def counting(self, genes):
        rows.append(len(genes))
        return original(self, genes)

    with patch.object(evf.FitRuleBase, 'torch_devices', ('cpu',)), \
            patch.object(evf.FitRuleBase, '_scalar_scores', counting), \
            _fitness_cache_scope(problem, True, population=len(genes)):
        fitness, on_device = problem._evaluate_gene_population(genes, device='cpu')
        assert problem._torch_route.verified
    np.testing.assert_array_equal(fitness, expected)
    assert on_device and rows == [torchfit.DeviceRoute.VERIFY_CANDIDATES]


def test_generations_too_small_to_verify_stay_on_the_cpu(torchfit):
    problem = _problem()
    genes = _population(problem, 24, seed=301)[:torchfit.DeviceRoute.MIN_CANDIDATES - 1]
    with patch.object(evf.FitRuleBase, 'torch_devices', ('cpu',)), \
            _fitness_cache_scope(problem, True, population=len(genes)):
        fitness, on_device = problem._evaluate_gene_population(genes, device='cpu')
        assert not problem._torch_route.verified
    assert not on_device
    np.testing.assert_array_equal(fitness, _fitness_values(problem, genes))


# ---------------------------------------------------------------------------
# Complete EvoX fits keep the search unchanged.

@pytest.fixture
def fake_evox(monkeypatch):
    """Minimal EvoX operators whose mutation varies offspring with torch RNG."""
    torch = pytest.importorskip('torch')
    evox = types.ModuleType('evox')
    operators = types.ModuleType('evox.operators')
    operators.crossover = types.SimpleNamespace(
        simulated_binary=lambda population, **kwargs: population.clone())
    operators.mutation = types.SimpleNamespace(
        polynomial_mutation=lambda population, **kwargs: population + torch.randint(
            -2, 3, population.shape, device=population.device).float())
    evox.operators = operators
    monkeypatch.setitem(sys.modules, 'evox', evox)
    monkeypatch.setitem(sys.modules, 'evox.operators', operators)
    return torch


@pytest.mark.parametrize('fixed', [True, False])
def test_evox_fit_search_is_identical_on_every_route(fake_evox, fixed):
    X, y = _data()
    partitions = utils.construct_partitions(X, fs.FUZZY_SETS.t1) if fixed else None

    def unscoped(problem, enabled, population=None):
        return nullcontext()

    routes = {
        'per individual': (patch.object(_fitness, '_fitness_cache_scope', unscoped), nullcontext()),
        'cached': (nullcontext(), nullcontext()),
        'device': (nullcontext(), patch.object(evf.FitRuleBase, 'torch_devices', ('cpu',))),
    }
    outcomes = {}
    for name, (scope, devices) in routes.items():
        model = evf.BaseFuzzyRulesClassifier(nRules=10, nAnts=3, backend='evox',
                                             linguistic_variables=partitions)
        with scope, devices:
            model.fit(X, y, n_gen=8, pop_size=16, random_state=5, patience=None)
        result = model.optimization_result_
        assert result['device'] == 'cpu' and not result['gpu_accelerated']
        outcomes[name] = (result['history']['best_fitness'], result['pop'],
                          result['fitness'], result['X'], model.predict(X))
    reference = outcomes.pop('per individual')
    for outcome in outcomes.values():
        for left, right in zip(reference, outcome):
            np.testing.assert_array_equal(left, right)


def test_evox_evaluates_generic_problems_with_one_host_transfer(fake_evox):
    torch = fake_evox

    class Squares:
        n_var = 3
        xl = np.zeros(3)
        xu = np.full(3, 9)
        calls = 0

        def _evaluate(self, x, out, *args, **kwargs):
            assert isinstance(x, np.ndarray) and x.shape == (3,)
            Squares.calls += 1
            out['F'] = float(np.sum(x ** 2))

    backend = object.__new__(eb.EvoXBackend)
    genes = torch.as_tensor([[1, 2, 3], [0, 0, 0]])
    fitness = backend._evaluate_population(genes, Squares(), torch.device('cpu'))
    assert fitness.tolist() == [14.0, 0.0] and Squares.calls == 2
