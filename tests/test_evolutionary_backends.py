"""
Tests for the evolutionary_backends module.

Tests PyMoo and EvoX backends for genetic optimization,
including backend selection and fallback behavior.
"""
import sys
import types

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import evolutionary_backends as eb
import evolutionary_fit as evf
import fuzzy_sets as fs

# EvoX may fail to import for reasons other than a missing package, so catch everything.
try:
    import evox  # noqa: F401
    HAS_EVOX = True
except Exception:
    HAS_EVOX = False

needs_evox = pytest.mark.skipif(not HAS_EVOX, reason="EvoX not installed")


@pytest.fixture
def fake_evox(monkeypatch):
    """Provide the small operator surface used by :class:`EvoXBackend`."""
    torch = pytest.importorskip('torch')
    evox = types.ModuleType('evox')
    operators = types.ModuleType('evox.operators')
    operators.crossover = types.SimpleNamespace(
        simulated_binary=lambda population, **kwargs: population.clone()
    )
    operators.mutation = types.SimpleNamespace(
        polynomial_mutation=lambda population, **kwargs: population.clone()
    )
    evox.operators = operators
    monkeypatch.setitem(sys.modules, 'evox', evox)
    monkeypatch.setitem(sys.modules, 'evox.operators', operators)
    return torch


@pytest.fixture(scope='module')
def dataset():
    X, y = make_classification(n_samples=80, n_features=4, n_informative=3, n_redundant=1,
                               n_classes=2, random_state=42)
    return train_test_split(X, y, test_size=0.25, random_state=42)


class SquaresProblem:
    '''A small integer problem whose optimum is the all-zero individual.'''

    n_var = 3
    xl = np.zeros(3)
    xu = np.full(3, 9)

    def _evaluate(self, x, out, *args, **kwargs):
        out['F'] = float(np.sum(np.asarray(x) ** 2))


class TestBackendSelection:

    def test_registry(self):
        backend = eb.get_backend('pymoo')
        assert backend.name() == 'pymoo' and backend.is_available()
        with pytest.raises(ValueError, match='Unknown backend'):
            eb.get_backend('invalid_backend')
        assert eb.list_available_backends() == (['pymoo', 'evox'] if HAS_EVOX else ['pymoo'])

    def test_unavailable_evox_is_reported(self, monkeypatch):
        monkeypatch.setitem(sys.modules, 'evox', None)
        backend = eb.EvoXBackend()
        assert not backend.is_available() and backend.name() == 'evox'
        with pytest.raises(ValueError, match='not available'):
            eb.get_backend('evox')
        assert eb.list_available_backends() == ['pymoo']

    def test_classifier_falls_back_to_pymoo(self, capsys):
        clf = evf.BaseFuzzyRulesClassifier(nRules=4, nAnts=2, backend='quantum', verbose=True)
        assert clf.backend.name() == 'pymoo'
        assert 'Falling back to pymoo backend' in capsys.readouterr().out

    def test_default_backend_is_pymoo(self, capsys):
        assert evf.BaseFuzzyRulesClassifier(nRules=10, nAnts=3, verbose=True).backend.name() == 'pymoo'
        assert 'Using evolutionary backend: pymoo' in capsys.readouterr().out


class TestEvoXBackendWithoutOptionalPackage:

    def test_available_backend_sets_up_cpu(self, fake_evox, capsys):
        backend = eb.EvoXBackend()

        assert backend.is_available()
        assert backend._device.type == 'cpu'
        assert 'EvoX backend using CPU' in capsys.readouterr().out

    def test_setup_reports_gpu(self, fake_evox, monkeypatch, capsys):
        monkeypatch.setattr(fake_evox.cuda, 'is_available', lambda: True)
        monkeypatch.setattr(fake_evox.cuda, 'get_device_name', lambda index: 'Test GPU')
        monkeypatch.setattr(fake_evox, 'device', lambda name: name)

        backend = eb.EvoXBackend()

        assert backend._device == 'cuda'
        assert 'EvoX backend using GPU: Test GPU' in capsys.readouterr().out

    def test_optimize_generic_and_torch_population_paths(self, fake_evox, capsys):
        backend = eb.EvoXBackend()
        seeded = np.array([[3, 2, 1], [0, 0, 0], [2, 2, 2], [1, 1, 1]])

        result = backend.optimize(
            SquaresProblem(), n_gen=3, pop_size=4, random_state=0,
            verbose=True, sampling=seeded, patience=1, min_delta=100.0,
        )

        assert result['X'].tolist() == [0, 0, 0]
        assert result['F'] == 0.0
        assert result['n_gen_run'] == 2
        assert result['stopped_early']
        assert result['device'] == 'cpu'
        assert not result['gpu_accelerated']
        output = capsys.readouterr().out
        assert 'Gen    0' in output
        assert 'Early stopping at generation 2' in output
        assert 'Optimization complete' in output

        class TorchProblem(SquaresProblem):
            def _evaluate_torch_population(self, population, device):
                # Returning NumPy also covers the backend's tensor conversion.
                return np.sum(population.cpu().numpy() ** 2, axis=1)

        random_result = backend.optimize(
            TorchProblem(), n_gen=2, pop_size=4, random_state=1,
            verbose=False, patience=None,
        )

        assert random_result['n_gen_run'] == 2
        assert not random_result['stopped_early']
        assert random_result['X'].shape == (3,)

        silent_stop = backend.optimize(
            SquaresProblem(), n_gen=3, pop_size=4, random_state=2,
            verbose=False, sampling=seeded, patience=1, min_delta=100.0,
        )
        assert silent_stop['stopped_early']


class TestPyMooBackend:

    def test_fit_and_predict(self, dataset):
        X_train, X_test, y_train, y_test = dataset
        clf = evf.BaseFuzzyRulesClassifier(nRules=10, nAnts=3, backend='pymoo')
        clf.fit(X_train, y_train, n_gen=5, pop_size=10)
        assert set(clf.predict(X_test)) <= {0, 1}

    @pytest.mark.parametrize('fuzzy_type', [fs.FUZZY_SETS.t1, fs.FUZZY_SETS.t2])
    def test_fuzzy_types(self, dataset, fuzzy_type):
        X_train, X_test, y_train, y_test = dataset
        clf = evf.BaseFuzzyRulesClassifier(nRules=10, nAnts=3, fuzzy_type=fuzzy_type)
        clf.fit(X_train, y_train, n_gen=5, pop_size=10)
        assert len(clf.predict(X_test)) == len(y_test)

    def test_progress_checkpoints_and_early_stopping(self, dataset, capsys):
        X_train, _, y_train, _ = dataset
        problem = evf.FitRuleBase(X_train, y_train, 4, 2, 2)
        checkpoints = []

        result = eb.get_backend('pymoo').optimize_with_checkpoints(
            problem, n_gen=6, pop_size=8, random_state=0, verbose=True, checkpoint_freq=2,
            checkpoint_callback=lambda gen, best: checkpoints.append((gen, best.shape)),
            patience=1, min_delta=10.0)

        output = capsys.readouterr().out
        assert 'n_gen  |  n_eval' in output
        assert 'Early stopping at generation 2' in output
        assert result['n_gen_run'] == 2 and result['stopped_early']
        assert checkpoints == [(0, (problem.n_var,))]

    def test_classifier_early_stopping(self, dataset):
        X_train, _, y_train, _ = dataset
        clf = evf.BaseFuzzyRulesClassifier(nRules=10, nAnts=3, backend='pymoo')
        clf.fit(X_train, y_train, n_gen=6, pop_size=10, patience=1, min_delta=10.0)
        assert clf.n_generations_run_ == 2
        assert clf.stopped_early_ is True

    def test_explicit_sampling_is_preserved(self):
        sampling = np.zeros((4, SquaresProblem.n_var), dtype=int)

        algorithm = eb.PyMooBackend()._build_ga_algorithm(
            pop_size=4, var_prob=0.3, sbx_eta=3.0, mutation_eta=7.0,
            tournament_size=3, sampling=sampling,
        )

        assert algorithm.initialization.sampling is sampling


@needs_evox
class TestEvoXBackend:

    def test_setup_reports_the_device(self, monkeypatch, capsys):
        import torch
        monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
        monkeypatch.setattr(torch.cuda, 'get_device_name', lambda index: 'Fake GPU')

        backend = eb.EvoXBackend()
        assert backend._device.type == 'cuda'
        assert 'EvoX backend using GPU: Fake GPU' in capsys.readouterr().out

    def test_generic_problems_with_seeded_and_random_populations(self, capsys):
        backend = eb.get_backend('evox')
        problem = SquaresProblem()

        seeded = backend.optimize(problem, n_gen=5, pop_size=6, random_state=0, verbose=True,
                                  sampling=np.zeros((6, 3), dtype=int))
        assert seeded['F'] == 0.0 and seeded['X'].tolist() == [0, 0, 0]
        assert seeded['device'] == 'cpu' and not seeded['gpu_accelerated']
        output = capsys.readouterr().out
        assert 'Gen    0' in output and 'Optimization complete' in output

        sampled = backend.optimize(problem, n_gen=20, pop_size=10, random_state=0, verbose=True,
                                   patience=2, min_delta=100.0)
        assert sampled['stopped_early'] and sampled['n_gen_run'] == 3
        assert 'Early stopping at generation 3' in capsys.readouterr().out

    def test_numpy_fitness_from_torch_problems(self):
        class TorchStyle(SquaresProblem):
            def _evaluate_torch_population(self, population, device):
                return np.sum(population.cpu().numpy() ** 2, axis=1)

        result = eb.get_backend('evox').optimize(TorchStyle(), n_gen=2, pop_size=4, random_state=0, verbose=False)
        assert result['F'] >= 0

    def test_classifier(self, dataset):
        X_train, X_test, y_train, y_test = dataset
        clf = evf.BaseFuzzyRulesClassifier(nRules=10, nAnts=3, backend='evox')
        clf.fit(X_train, y_train, n_gen=6, pop_size=10, patience=1, min_delta=10.0)

        assert set(clf.predict(X_test)) <= {0, 1}
        assert clf.n_generations_run_ == 2
        assert clf.stopped_early_ is True
