"""Minibatch fitness: stratified batches, exact row swapping and the generation protocol."""
import warnings
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.datasets import make_classification

from ex_fuzzy import evolutionary_backends as eb
from ex_fuzzy import evolutionary_fit as evf
from ex_fuzzy import fuzzy_sets as fs
from ex_fuzzy import utils
from ex_fuzzy._fitness import MinibatchSchedule, _batch_quotas, _fitness_cache_scope


def _data(samples=400, classes=3, seed=3, features=6):
    return make_classification(
        n_samples=samples, n_features=features, n_informative=5, n_redundant=0,
        n_classes=classes, weights=[0.7, 0.2, 0.1][:classes] if classes == 3 else None,
        random_state=seed)


def _problem(X, y, fixed=True, partitions=None, **kwargs):
    """A problem on (X, y); ``partitions`` lets a problem on a row subset share the full data's variables."""
    if fixed and partitions is None:
        partitions = utils.construct_partitions(X, fs.FUZZY_SETS.t1)
    elif not fixed:
        partitions = None
    return evf.FitRuleBase(X, y, kwargs.pop('nRules', 12), kwargs.pop('nAnts', 3), 3,
                           linguistic_variables=partitions, **kwargs)


def _genes(problem, count=24, seed=7):
    return np.random.default_rng(seed).integers(
        problem.xl.astype(int), problem.xu.astype(int) + 1, size=(count, problem.n_var))


def _classifier(**kwargs):
    options = dict(nRules=10, nAnts=3, n_gen=5, pop_size=14, patience=None, batch_size=120)
    options.update(kwargs)
    return evf.BaseFuzzyRulesClassifier(**options)


class _RecordingSchedule(MinibatchSchedule):
    """A schedule that remembers the rows of every batch it activated."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batches = []

    def activate_next(self):
        rows = super().activate_next()
        self.batches.append(rows)
        return rows


# --------------------------------------------------------------------------- validation

class _OtherBackend(eb.EvolutionaryBackend):
    def optimize(self, *args, **kwargs):  # pragma: no cover - never reached
        raise AssertionError('validation must run first')

    def is_available(self):
        return True

    def name(self):
        return 'other'


class TestValidation:
    X, y = _data()

    @pytest.mark.parametrize('batch_size', [2, 400, 401, 0, -5, 1.5, '100', True])
    def test_batch_size_must_be_an_integer_between_classes_and_rows(self, batch_size):
        with pytest.raises(ValueError, match='batch_size'):
            _classifier(batch_size=batch_size).fit(self.X, self.y)

    def test_bounds_are_inclusive_below_and_exclusive_above(self):
        _classifier(batch_size=3, n_gen=1).fit(self.X, self.y)
        _classifier(batch_size=399, n_gen=1).fit(self.X, self.y)
        _classifier(batch_size=np.int64(100), n_gen=1).fit(self.X, self.y)

    def test_nsga2_is_rejected(self):
        with pytest.raises(ValueError, match='nsga2'):
            _classifier(algorithm='nsga2').fit(self.X, self.y)

    def test_other_backends_are_rejected(self):
        with pytest.raises(ValueError, match='pymoo backend'):
            _classifier(backend=_OtherBackend()).fit(self.X, self.y)

    def test_candidate_rules_are_rejected(self):
        with pytest.raises(ValueError, match='candidate_rules'):
            _classifier().fit(self.X, self.y, candidate_rules=object())

    def test_checkpoints_are_rejected(self):
        with pytest.raises(ValueError, match='checkpoints'):
            _classifier().fit(self.X, self.y, checkpoints=2)

    def test_patience_is_ignored_with_one_warning(self):
        with pytest.warns(UserWarning, match='patience is ignored') as caught:
            clf = _classifier(patience=3).fit(self.X, self.y)
        assert len([w for w in caught if 'patience is ignored' in str(w.message)]) == 1
        assert clf.n_generations_run_ == 5 and not clf.stopped_early_

    def test_no_patience_warning_without_patience_or_batches(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            _classifier(patience=None).fit(self.X, self.y)
            _classifier(patience=3, batch_size=None).fit(self.X, self.y)

    def test_fit_argument_overrides_the_constructor(self):
        clf = _classifier(batch_size=None)
        assert clf.batch_size is None
        clf.fit(self.X, self.y, batch_size=100)
        assert clf.optimization_result_['algorithm'].evaluator.n_eval == 14 * 5 + 14 * 4 + 14
        clf = _classifier(batch_size=100).fit(self.X, self.y, batch_size=None)
        assert clf.batch_size == 100
        assert clf.optimization_result_['algorithm'].evaluator.n_eval == 14 * 5

    def test_get_params_round_trip(self):
        clf = _classifier(batch_size=77)
        assert clf.get_params()['batch_size'] == 77
        assert clf.set_params(batch_size=None).batch_size is None


# --------------------------------------------------------------------------- the sampler

def _stub(class_sizes):
    y = np.repeat(np.arange(len(class_sizes)), class_sizes)
    np.random.default_rng(0).shuffle(y)
    swaps = []
    return SimpleNamespace(y=y, _activate_rows=swaps.append, _restore_rows=lambda: None), swaps


class TestSampler:
    def test_quotas_are_proportional_by_largest_remainder(self):
        np.testing.assert_array_equal(_batch_quotas([7400, 2300, 200, 100], 300), [222, 69, 6, 3])
        # Ties of the remainders go to the lower class.
        np.testing.assert_array_equal(_batch_quotas([5, 5, 5, 5], 6), [2, 2, 1, 1])

    def test_every_class_gets_a_row(self):
        # The proportional shares of the rare classes round to zero.
        np.testing.assert_array_equal(_batch_quotas([7400, 2300, 200, 100], 40), [29, 9, 1, 1])
        np.testing.assert_array_equal(_batch_quotas([1000, 1, 1], 3), [1, 1, 1])

    def test_quotas_properties_hold_for_random_class_sizes(self):
        rng = np.random.default_rng(1)
        for _ in range(200):
            sizes = rng.integers(1, 500, size=rng.integers(2, 8))
            batch = int(rng.integers(len(sizes), sizes.sum()))
            quotas = _batch_quotas(sizes, batch)
            assert quotas.sum() == batch and quotas.min() >= 1 and np.all(quotas <= sizes)
            assert np.all(np.abs(quotas - sizes * batch / sizes.sum()) < 1 + len(sizes))

    def test_draw_is_stratified_sorted_and_without_replacement(self):
        problem, _ = _stub([7400, 2300, 200, 100])
        schedule = MinibatchSchedule(problem, 300, seed=5)
        for _ in range(5):
            rows = schedule.draw()
            assert len(rows) == 300 and len(np.unique(rows)) == 300
            assert np.all(np.diff(rows) > 0)
            np.testing.assert_array_equal(np.bincount(problem.y[rows]), [222, 69, 6, 3])

    def test_batches_differ_and_cover_the_data_over_time(self):
        problem, _ = _stub([900, 100])
        schedule = MinibatchSchedule(problem, 100, seed=5)
        draws = [schedule.draw() for _ in range(30)]
        assert any(not np.array_equal(draws[0], other) for other in draws[1:])
        assert len(np.unique(np.concatenate(draws))) > 700

    def test_same_seed_same_sequence(self):
        problem, _ = _stub([700, 200, 100])
        first = MinibatchSchedule(problem, 60, seed=11)
        second = MinibatchSchedule(problem, 60, seed=11)
        other = MinibatchSchedule(problem, 60, seed=12)
        for _ in range(4):
            a, b, c = first.draw(), second.draw(), other.draw()
            np.testing.assert_array_equal(a, b)
            assert not np.array_equal(a, c)

    def test_global_random_state_is_untouched(self):
        problem, _ = _stub([700, 200, 100])
        np.random.seed(123)
        before = np.random.get_state()
        schedule = MinibatchSchedule(problem, 60, seed=11)
        for _ in range(3):
            schedule.draw()
        after = np.random.get_state()
        assert before[0] == after[0] and before[2:] == after[2:]
        np.testing.assert_array_equal(before[1], after[1])

    def test_activate_next_hands_the_rows_to_the_problem(self):
        problem, swaps = _stub([70, 30])
        schedule = MinibatchSchedule(problem, 20, seed=0)
        rows = schedule.activate_next()
        assert len(swaps) == 1 and swaps[0] is rows

    def test_invalid_sizes_are_rejected(self):
        problem, _ = _stub([70, 30])
        for bad in (1, 100, 0):
            with pytest.raises(ValueError, match='batch_size'):
                MinibatchSchedule(problem, bad, seed=0)


# --------------------------------------------------------------------------- exact row swapping

def _rows(y, seed=2, size=90):
    return MinibatchSchedule(SimpleNamespace(y=y), size, seed).draw()


class TestRowSwap:
    @pytest.mark.parametrize('ds_mode', [0, 1])
    def test_fixed_partitions_equal_a_fresh_problem_on_the_rows(self, ds_mode):
        X, y = _data()
        problem = _problem(X, y, ds_mode=ds_mode, tolerance=0.01)
        genes = _genes(problem)
        with _fitness_cache_scope(problem, True):
            # Warm every cache on the full data first: the swap has to discard them.
            full_scalar = problem._scalar_scores(genes)
            full_batched = 1 - problem._batched_scores(genes)
            np.testing.assert_array_equal(full_scalar, full_batched)
            rows = _rows(y)
            problem._activate_rows(rows)
            assert len(problem.X) == len(rows) == len(problem.y)
            scalar = problem._scalar_scores(genes)
            batched = 1 - problem._batched_scores(genes)
            assert not np.array_equal(scalar, full_scalar)

            fresh = _problem(X[rows], y[rows], partitions=problem.lvs, ds_mode=ds_mode, tolerance=0.01)
            with _fitness_cache_scope(fresh, True):
                fresh_scalar = fresh._scalar_scores(genes)
                fresh_batched = 1 - fresh._batched_scores(genes)
        np.testing.assert_array_equal(scalar, fresh_scalar)
        np.testing.assert_array_equal(batched, fresh_batched)
        np.testing.assert_array_equal(scalar, batched)

    def test_population_entry_point_matches_a_fresh_problem(self):
        X, y = _data()
        problem = _problem(X, y)
        genes = _genes(problem)
        rows = _rows(y)
        with _fitness_cache_scope(problem, True):
            warm = {}
            problem._evaluate_elementwise(genes, warm)
            problem._activate_rows(rows)
            out = {}
            problem._evaluate_elementwise(genes, out)
        fresh = _problem(X[rows], y[rows], partitions=problem.lvs)
        with _fitness_cache_scope(fresh, True):
            expected = {}
            fresh._evaluate_elementwise(genes, expected)
        np.testing.assert_array_equal(out['F'], expected['F'])

    def test_restoring_gives_the_original_values_bit_for_bit(self):
        X, y = _data()
        problem = _problem(X, y)
        genes = _genes(problem)
        with _fitness_cache_scope(problem, True):
            original_scalar = problem._scalar_scores(genes)
            original_batched = problem._batched_scores(genes)
            problem._activate_rows(_rows(y))
            problem._scalar_scores(genes)
            problem._activate_rows(_rows(y, seed=3))
            problem._restore_rows()
            assert problem.X is X and problem.y is y
            np.testing.assert_array_equal(problem._scalar_scores(genes), original_scalar)
            np.testing.assert_array_equal(problem._batched_scores(genes), original_batched)
        problem._restore_rows()  # a second restore is a no-op

    def test_label_domain_and_packed_table_follow_the_rows(self):
        X, y = _data()
        problem = _problem(X, y)
        assert problem._label_domain().class_sizes.sum() == len(y)
        assert len(problem._packed_memberships()[0]) == len(y)
        rows = _rows(y)
        problem._activate_rows(rows)
        np.testing.assert_array_equal(problem._label_domain().class_sizes, np.bincount(y[rows]))
        assert len(problem._packed_memberships()[0]) == len(rows)
        problem._restore_rows()
        assert problem._label_domain().class_sizes.sum() == len(y)
        assert len(problem._packed_memberships()[0]) == len(y)

    def test_fit_local_caches_are_cleared_on_every_swap(self):
        X, y = _data()
        problem = _problem(X, y)
        genes = _genes(problem)
        with _fitness_cache_scope(problem, True):
            problem._scalar_scores(genes)
            assert problem._fitness_cache.entries and problem._firing_cache.entries
            problem._activate_rows(_rows(y))
            assert not problem._fitness_cache.entries and problem._fitness_cache.key_bytes == 0
            assert not problem._firing_cache.entries and problem._firing_cache.bytes == 0

    def test_route_probe_and_device_route_follow_the_row_count(self):
        X, y = _data()
        problem = _problem(X, y)
        with _fitness_cache_scope(problem, True):
            problem._torch_route = object()
            problem._route_probe = probe = object()
            problem._activate_rows(_rows(y))
            assert problem._route_probe is None and problem._torch_route is None
            problem._route_probe = probe
            problem._activate_rows(_rows(y, seed=4))  # same row count: the choice stays valid
            assert problem._route_probe is probe
            problem._restore_rows()
            assert problem._route_probe is None

    def test_searched_partitions_decode_identically_whatever_the_batch(self):
        X, y = _data()
        problem = _problem(X, y, fixed=False)
        genes = _genes(problem, count=6)
        domain = problem._normalization_domain()
        full = [problem._decode_antecedents(gene, fs.FUZZY_SETS.t1) for gene in genes]
        problem._activate_rows(_rows(y))
        assert problem._normalization_domain() is domain
        for gene, expected in zip(genes, full):
            for variable, reference in zip(problem._decode_antecedents(gene, fs.FUZZY_SETS.t1), expected):
                assert len(variable.linguistic_variables) == len(reference.linguistic_variables)
                for term, expected_term in zip(variable.linguistic_variables, reference.linguistic_variables):
                    np.testing.assert_array_equal(term.membership_parameters,
                                                  expected_term.membership_parameters)

    def test_searched_partitions_score_like_a_fresh_problem_with_the_full_domain(self):
        X, y = _data()
        problem = _problem(X, y, fixed=False)
        genes = _genes(problem, count=10)
        rows = _rows(y)
        problem._activate_rows(rows)
        fresh = _problem(X[rows], y[rows], fixed=False)
        fresh._normalization_domain_cache = problem._normalization_domain()
        with _fitness_cache_scope(problem, True), _fitness_cache_scope(fresh, True):
            np.testing.assert_array_equal(problem._scalar_scores(genes), fresh._scalar_scores(genes))
        problem._restore_rows()
        with _fitness_cache_scope(problem, True):
            full = problem._scalar_scores(genes)
        reference = _problem(X, y, fixed=False)
        with _fitness_cache_scope(reference, True):
            np.testing.assert_array_equal(full, reference._scalar_scores(genes))


# --------------------------------------------------------------------------- the generation protocol

class TestGenerationProtocol:
    def _run(self, fixed=True, n_gen=4, pop_size=12, batch=90, seed=21):
        X, y = _data()
        problem = _problem(X, y, fixed=fixed)
        schedule = _RecordingSchedule(problem, batch, seed)
        backend = eb.PyMooBackend()
        algorithm = backend._build_algorithm('ga', pop_size, 0.3, 3.0, 7.0, 3, None)
        checks = []
        original_next = None

        def checked_next():
            original_next()
            rows = schedule.batches[-1]
            pop = algorithm.pop
            genes = pop.get('X').astype(int)
            fresh = _problem(X[rows], y[rows], fixed=fixed, partitions=problem.lvs)
            if not fixed:
                fresh._normalization_domain_cache = problem._normalization_domain()
            on_batch = fresh._scalar_scores(genes)
            checks.append(np.array_equal(pop.get('F')[:, 0], on_batch))
            if len(schedule.batches) > 1:
                earlier = _problem(X[schedule.batches[-2]], y[schedule.batches[-2]], fixed=fixed,
                                   partitions=problem.lvs)
                if not fixed:
                    earlier._normalization_domain_cache = problem._normalization_domain()
                checks.append(not np.array_equal(earlier._scalar_scores(genes), on_batch))

        original_next = algorithm.next
        algorithm.next = checked_next
        result = backend._run_ga_loop(eb.as_pymoo_problem(problem), algorithm, n_gen, 5, False,
                                      minibatch=schedule)
        return X, y, problem, schedule, algorithm, result, checks

    @pytest.mark.parametrize('fixed', [True, False])
    def test_population_is_scored_on_the_generations_batch(self, fixed):
        _, _, _, schedule, _, _, checks = self._run(fixed=fixed)
        assert len(schedule.batches) == 4
        assert all(checks), 'parents and offspring must be scored on the same, current batch'

    def test_counts_and_full_data_restoration(self):
        X, y, problem, schedule, algorithm, result, _ = self._run(n_gen=4, pop_size=12)
        # Initial population, then per later generation the parents and the offspring, then the final scoring.
        assert algorithm.evaluator.n_eval == 12 + 3 * (12 + 12) + 12
        assert problem.X is X and problem.y is y
        assert result['n_gen_run'] == 4 and not result['stopped_early']

    @pytest.mark.parametrize('fixed', [True, False])
    def test_result_is_the_best_of_the_exact_full_data_scoring(self, fixed):
        X, y, problem, _, algorithm, result, _ = self._run(fixed=fixed)
        reference = _problem(X, y, fixed=fixed, partitions=problem.lvs)
        genes = algorithm.pop.get('X').astype(int)
        exact = reference._scalar_scores(genes)
        np.testing.assert_array_equal(algorithm.pop.get('F')[:, 0], exact)
        best = int(np.argmin(exact))
        assert result['F'] == exact[best]
        np.testing.assert_array_equal(result['X'], genes[best])
        np.testing.assert_array_equal(algorithm.opt.get('F')[:, 0].min(), exact.min())

    def test_without_a_schedule_the_loop_is_unchanged(self):
        X, y = _data()
        problem = _problem(X, y)
        backend = eb.PyMooBackend()
        algorithm = backend._build_algorithm('ga', 12, 0.3, 3.0, 7.0, 3, None)
        backend._run_ga_loop(eb.as_pymoo_problem(problem), algorithm, 4, 5, False)
        assert algorithm.evaluator.n_eval == 12 * 4


# --------------------------------------------------------------------------- complete fits

def _rule_text(clf):
    return clf.print_rules(return_rules=True)


@pytest.mark.parametrize('partitions', ['searched', 'fixed'])
class TestFit:
    X, y = _data()

    def _clf(self, partitions, **kwargs):
        if partitions == 'fixed':
            kwargs['linguistic_variables'] = utils.construct_partitions(self.X, fs.FUZZY_SETS.t1)
        return _classifier(**kwargs)

    def test_performance_is_the_exact_full_training_objective(self, partitions):
        clf = self._clf(partitions).fit(self.X, self.y)
        result = clf.optimization_result_
        gene = result['X']
        reference = _problem(self.X, self.y, fixed=partitions == 'fixed', nRules=10, nAnts=3,
                             tolerance=clf.tolerance, alpha=clf.alpha_, beta=clf.beta_)
        exact = reference._scalar_scores(gene[None, :])[0]
        assert result['F'] == exact
        assert clf.performance == 1 - exact
        assert len(clf.predict(self.X)) == len(self.y)

    def test_same_seed_same_model_whatever_the_global_numpy_state(self, partitions):
        np.random.seed(1)
        first = self._clf(partitions).fit(self.X, self.y)
        np.random.seed(2)
        second = self._clf(partitions).fit(self.X, self.y)
        assert first.performance == second.performance
        np.testing.assert_array_equal(first.optimization_result_['X'], second.optimization_result_['X'])
        assert _rule_text(first) == _rule_text(second)

    def test_the_search_does_not_consume_the_global_random_state(self, partitions):
        np.random.seed(9)
        before = np.random.get_state()
        self._clf(partitions).fit(self.X, self.y)
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        assert before[2] == after[2]

    def test_a_different_seed_changes_the_search(self, partitions):
        first = self._clf(partitions).fit(self.X, self.y)
        second = self._clf(partitions, random_state=99).fit(self.X, self.y)
        assert not np.array_equal(first.optimization_result_['X'], second.optimization_result_['X'])

    def test_custom_loss_sees_the_batch_and_then_the_full_data(self, partitions):
        sizes = []

        def loss(rule_base, X, y, tolerance, alpha=0.0, beta=0.0, precomputed_truth=None):
            sizes.append(len(X))
            assert len(y) == len(X)
            return float(np.mean(rule_base.predict(X) == y))

        clf = self._clf(partitions, n_gen=3, pop_size=8)
        clf.customized_loss(loss)
        clf.fit(self.X, self.y)
        assert set(sizes) == {120, len(self.y)}
        assert sizes[0] == 120 and sizes[-1] == len(self.y)
        assert len(clf.predict(self.X)) == len(self.y)

    def test_thread_runner_matches_the_serial_fit(self, partitions):
        serial = self._clf(partitions, n_gen=3, pop_size=8).fit(self.X, self.y)
        threaded = self._clf(partitions, n_gen=3, pop_size=8, runner=2).fit(self.X, self.y)
        assert serial.performance == threaded.performance
        np.testing.assert_array_equal(serial.optimization_result_['X'], threaded.optimization_result_['X'])


def test_batch_mode_leaves_the_estimator_reusable():
    X, y = _data()
    # Fixed partitions: a refit of searched partitions reuses the first fit's, with or without batches.
    clf = _classifier(linguistic_variables=utils.construct_partitions(X, fs.FUZZY_SETS.t1))
    clf.fit(X, y)
    first = clf.performance
    clf.fit(X, y)
    assert clf.performance == first
    assert clf.batch_size == 120
    assert len(clf.predict(X)) == len(y)


def test_default_fit_does_not_build_a_schedule():
    X, y = _data()
    with patch('ex_fuzzy._fitness.MinibatchSchedule', side_effect=AssertionError('unexpected')):
        clf = _classifier(batch_size=None).fit(X, y)
    assert clf.optimization_result_['algorithm'].evaluator.n_eval == 14 * 5
