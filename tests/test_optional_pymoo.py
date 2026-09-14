"""pymoo is imported only by the code that runs pymoo."""
import os
import pickle
import subprocess
import sys
import textwrap
from multiprocessing.pool import ThreadPool

import numpy as np
import pytest
from sklearn.datasets import load_iris

import _problem as problem_module
import evolutionary_backends as eb
import evolutionary_fit as evf
import fuzzy_sets as fs
import utils

PACKAGE_PARENT = os.path.join(os.path.dirname(__file__), '..', 'ex_fuzzy')


def _run_isolated(code: str) -> subprocess.CompletedProcess:
    script = f"import sys\nsys.path.insert(0, {PACKAGE_PARENT!r})\n" + textwrap.dedent(code)
    return subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)


def test_importing_ex_fuzzy_does_not_import_pymoo():
    result = _run_isolated("""
        import ex_fuzzy
        loaded = sorted(name for name in sys.modules if name == 'pymoo' or name.startswith('pymoo.'))
        assert not loaded, loaded
    """)
    assert result.returncode == 0, result.stderr


def test_evox_fits_run_when_pymoo_cannot_be_imported():
    pytest.importorskip('torch')
    result = _run_isolated("""
        sys.modules['pymoo'] = None  # every 'import pymoo...' now raises ImportError
        import types, torch
        evox = types.ModuleType('evox')
        operators = types.ModuleType('evox.operators')
        operators.crossover = types.SimpleNamespace(
            simulated_binary=lambda population, **kwargs: population.clone())
        operators.mutation = types.SimpleNamespace(
            polynomial_mutation=lambda population, **kwargs: population + torch.randint(
                -1, 2, population.shape).float())
        evox.operators = operators
        sys.modules['evox'] = evox
        sys.modules['evox.operators'] = operators

        from sklearn.datasets import load_iris, make_regression
        from ex_fuzzy import evolutionary_fit as evf, evolutionary_fit_regression as efr, utils
        X, y = load_iris(return_X_y=True)
        classifier = evf.BaseFuzzyRulesClassifier(
            nRules=6, nAnts=2, backend='evox', linguistic_variables=utils.construct_partitions(X))
        classifier.fit(X, y, n_gen=3, pop_size=8, random_state=0, patience=None)
        assert len(classifier.predict(X)) == len(X)

        Xr, yr = make_regression(n_samples=60, n_features=3, random_state=0)
        regressor = efr.BaseFuzzyRulesRegressor(nRules=4, nAnts=2, backend='evox')
        regressor.fit(Xr, yr, n_gen=3, pop_size=8, random_state=0)
        assert len(regressor.predict(Xr)) == len(Xr)

        try:
            evf.BaseFuzzyRulesClassifier(nRules=6, nAnts=2).fit(X, y, n_gen=1, pop_size=4)
        except ImportError as error:
            assert 'pip install' in str(error), error
        else:
            raise AssertionError('the pymoo backend ran without pymoo')
    """)
    assert result.returncode == 0, result.stderr


def _iris_problem(**kwargs):
    X, y = load_iris(return_X_y=True)
    return evf.FitRuleBase(X, y, 6, 2, 3, linguistic_variables=utils.construct_partitions(
        X, fs.FUZZY_SETS.t1), **kwargs)


def test_pymoo_wrapper_delegates_evaluation():
    from pymoo.core.problem import Problem as PymooProblem
    from pymoo.core.variable import Integer as PymooInteger

    problem = _iris_problem()
    wrapped = eb.as_pymoo_problem(problem)
    assert isinstance(wrapped, PymooProblem) and wrapped.problem is problem
    assert not isinstance(problem, PymooProblem)
    assert wrapped.n_var == problem.n_var and wrapped.elementwise
    np.testing.assert_array_equal(wrapped.xl, problem.xl)
    np.testing.assert_array_equal(wrapped.xu, problem.xu)
    assert all(isinstance(var, PymooInteger) and np.array_equal(var.bounds, problem.vars[name].bounds)
               for name, var in wrapped.vars.items())

    genes = np.random.default_rng(3).integers(
        problem.xl.astype(int), problem.xu.astype(int) + 1, size=(12, problem.n_var))
    expected = [1 - problem._array_score(gene) for gene in genes]
    np.testing.assert_array_equal(wrapped.evaluate(genes).ravel(), expected)
    np.testing.assert_array_equal(problem.evaluate(genes).ravel(), expected)
    np.testing.assert_array_equal(problem.evaluate(genes[0]), [expected[0]])
    assert eb.as_pymoo_problem(wrapped) is wrapped


def test_wrapped_problems_pickle():
    problem = _iris_problem()
    restored = pickle.loads(pickle.dumps(eb.as_pymoo_problem(problem)))
    genes = np.zeros((2, problem.n_var), dtype=int)
    np.testing.assert_array_equal(restored.evaluate(genes), problem.evaluate(genes))


def test_starmap_runner_evaluates_in_order_and_drops_its_pool_when_pickled():
    with ThreadPool(2) as pool:
        runner = problem_module.StarmapParallelization(pool.starmap)
        problem = _iris_problem(thread_runner=runner)
        genes = np.random.default_rng(5).integers(
            problem.xl.astype(int), problem.xu.astype(int) + 1, size=(8, problem.n_var))
        out = {}
        problem._evaluate_elementwise(genes, out)
    np.testing.assert_array_equal(out['F'], [1 - problem._array_score(gene) for gene in genes])
    assert 'starmap' not in pickle.loads(pickle.dumps(runner)).__dict__
    assert evf.StarmapParallelization is problem_module.StarmapParallelization
