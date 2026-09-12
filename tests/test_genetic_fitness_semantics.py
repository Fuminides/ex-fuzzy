"""All genetic backends must optimize the classifier's reference objective."""
from multiprocessing.pool import ThreadPool

import numpy as np
import pytest
from sklearn.datasets import load_iris

import evolutionary_fit as evf
import evolutionary_backends as eb
import fuzzy_sets as fs
import utils


def make_problem(fixed=False, ds_mode=0, fuzzy_type=fs.FUZZY_SETS.t1):
    X, y = load_iris(return_X_y=True)
    partitions = utils.construct_partitions(X, fuzzy_type) if fixed else None
    return evf.FitRuleBase(
        X, y, nRules=10, nAnts=2, n_classes=3,
        linguistic_variables=partitions, fuzzy_type=fuzzy_type,
        ds_mode=ds_mode, tolerance=0.01, alpha=0.1, beta=0.2,
    )


def population(problem, size=5):
    return np.random.default_rng(42).integers(
        problem.xl.astype(int), problem.xu.astype(int) + 1,
        size=(size, problem.n_var),
    )


def objective(problem, chromosome):
    out = {}
    problem._evaluate(chromosome, out)
    return out['F']


def reference(problem, chromosome):
    rulebase = problem._construct_ruleBase(chromosome, problem.fuzzy_type)
    if not rulebase.get_rules():
        return 1.0
    return 1 - problem.fitness_func(
        rulebase, problem.X, problem.y, problem.tolerance,
        problem.alpha_, problem.beta_, problem._precomputed_truth,
    )


@pytest.mark.parametrize('fixed', [False, True])
@pytest.mark.parametrize('ds_mode', [0, 1, 2])
@pytest.mark.parametrize('fuzzy_type', [fs.FUZZY_SETS.t1, fs.FUZZY_SETS.t2])
def test_objective_matches_decoded_classifier(fixed, ds_mode, fuzzy_type):
    problem = make_problem(fixed, ds_mode, fuzzy_type)
    genes = population(problem)
    original = genes.copy()
    expected = [reference(problem, x) for x in genes]
    actual = [objective(problem, x) for x in genes]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(genes, original)


def test_membership_search_is_independent_of_evaluation_order_and_threads():
    problem = make_problem()
    genes = population(problem, 12)
    expected = [reference(problem, x) for x in genes]
    np.testing.assert_array_equal(
        [objective(problem, x) for x in genes[::-1]], expected[::-1])
    with ThreadPool(3) as pool:
        actual = pool.map(lambda x: objective(problem, x), genes)
    np.testing.assert_array_equal(actual, expected)


def test_custom_loss_receives_normalized_rulebase_and_penalties():
    problem = make_problem()
    calls = []

    def loss(rulebase, X, y, tolerance, alpha, beta, precomputed):
        calls.append((tolerance, alpha, beta))
        for feature, variable in enumerate(rulebase.antecedents):
            params = np.asarray([s.membership_parameters for s in variable])
            assert params.min() == pytest.approx(X[:, feature].min())
            assert params.max() == pytest.approx(X[:, feature].max())
        return 0.37

    problem.fitness_func = loss
    assert objective(problem, population(problem, 1)[0]) == pytest.approx(0.63)
    assert calls == [(0.01, 0.1, 0.2)]


def test_disabled_rules_do_not_vote_for_last_class():
    problem = make_problem(fixed=True)
    chromosome = population(problem, 1)[0]
    chromosome[-problem.nRules:] = -1
    assert objective(problem, chromosome) == 1.0


@pytest.mark.parametrize('fixed', [False, True])
def test_evox_population_uses_same_objective_including_custom_loss(fixed):
    torch = pytest.importorskip('torch')
    problem = make_problem(fixed=fixed, ds_mode=2)
    problem.fitness_func = lambda *args: 0.37
    genes = population(problem)
    # Exercise real dispatch without requiring EvoX or a GPU on the test host.
    backend = object.__new__(eb.EvoXBackend)
    actual = backend._evaluate_population(
        torch.as_tensor(genes), problem, torch.device('cpu'))
    # EvoX's custom-loss bridge returns float32; CPU evaluator parity above is exact.
    np.testing.assert_allclose(actual.numpy(), [reference(problem, x) for x in genes])
