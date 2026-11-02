"""A training subset that grows during the genetic search (initial_data_fraction)."""
import numpy as np
import pytest
from sklearn.datasets import make_classification

from ex_fuzzy import evolutionary_fit as evf
from ex_fuzzy import fuzzy_sets as fs
from ex_fuzzy import rules
from ex_fuzzy import utils
from ex_fuzzy._fitness import _fitness_cache_scope, score_rulebase
from ex_fuzzy._growing_data import GrowingData


@pytest.fixture(scope='module')
def data():
    return make_classification(n_samples=1200, n_features=5, n_informative=4, n_redundant=0,
                               n_classes=3, n_clusters_per_class=1, weights=[0.1, 0.3, 0.6],
                               random_state=2)


def _problem(X, y, fixed=True, **kwargs):
    partitions = utils.construct_partitions(X, fs.FUZZY_SETS.t1) if fixed else None
    return evf.FitRuleBase(X, y, 8, 3, 3, linguistic_variables=partitions, tolerance=0.01, **kwargs)


def _genes(problem, count=8, seed=4):
    return np.random.default_rng(seed).integers(problem.xl.astype(int), problem.xu.astype(int) + 1,
                                                size=(count, problem.n_var))


def _full_performance(model, X, y, partitions):
    """``performance`` of the returned chromosome scored on every training sample."""
    problem = evf.FitRuleBase(X, y, model.nRules, model.nAnts, 3, linguistic_variables=partitions,
                              tolerance=model.tolerance)
    score = problem._array_score(np.asarray(model.optimization_result_['X']).astype(int))
    return 1 - problem._objectives(score)  # fit's own round trip through the minimized value


def test_subsets_grow_linearly_to_every_sample(data):
    X, y = data
    schedule = GrowingData(_problem(X, y), 0.1, 0.5, 20, seed=0)
    assert schedule.ramp == 10
    sizes = [schedule.size_at(gen) for gen in range(20)]
    assert sizes[0] == int(np.ceil(0.1 * len(y)))
    assert sizes[5] == int(np.ceil(0.55 * len(y)))
    assert sizes[9] < len(y) and sizes[10:] == [len(y)] * 10
    assert sizes == sorted(sizes)


def test_subsets_are_nested_and_stratified(data):
    X, y = data
    schedule = GrowingData(_problem(X, y), 0.1, 0.5, 20, seed=0)
    shares = np.bincount(y) / len(y)
    previous = set()
    for gen in range(11):
        rows = schedule.order[:schedule.size_at(gen)]
        assert previous <= set(rows)
        previous = set(rows)
        assert np.all(np.abs(np.bincount(y[rows], minlength=3) - shares * len(rows)) <= 1)


def test_every_subset_holds_every_class():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 3))
    y = np.zeros(2000, dtype=int)
    y[:3] = 1  # a class of three samples
    schedule = GrowingData(_problem(X, y), 0.001, 0.5, 10, seed=1)
    assert 1 in y[schedule.order[:schedule.size_at(0)]]


def test_subsets_follow_the_seed(data):
    X, y = data
    problem = _problem(X, y)
    first, again = GrowingData(problem, 0.2, 0.5, 10, 3), GrowingData(problem, 0.2, 0.5, 10, 3)
    np.testing.assert_array_equal(first.order, again.order)
    assert not np.array_equal(first.order, GrowingData(problem, 0.2, 0.5, 10, 4).order)
    assert GrowingData(problem, 0.2, 0.5, 10, None).size_at(0) == first.size_at(0)


@pytest.mark.parametrize('fraction, full_at, n_gen, ramp', [(0.2, 1.0, 10, 9), (0.2, 0.5, 1, 0),
                                                              (0.2, 0.01, 50, 0), (1.0, 0.5, 10, 5)])
def test_the_ramp_ends_by_the_last_generation(data, fraction, full_at, n_gen, ramp):
    X, y = data
    schedule = GrowingData(_problem(X, y), fraction, full_at, n_gen, 0)
    assert schedule.ramp == ramp
    assert schedule.size_at(n_gen - 1) == len(y)
    if ramp == 0 or fraction == 1.0:
        assert not any(schedule.advance(gen) for gen in range(n_gen))
        assert schedule.complete


@pytest.mark.parametrize('value', [0, -0.1, 1.5, True, 'half', None])
def test_invalid_fractions_are_rejected(data, value):
    X, y = data
    problem = _problem(X, y)
    with pytest.raises(ValueError, match='initial_data_fraction'):
        GrowingData(problem, value, 0.5, 10, 0)
    with pytest.raises(ValueError, match='full_data_at'):
        GrowingData(problem, 0.5, value, 10, 0)


def test_activated_rows_score_like_a_problem_built_on_them(data):
    X, y = data
    partitions = utils.construct_partitions(X, fs.FUZZY_SETS.t1)
    problem = evf.FitRuleBase(X, y, 8, 3, 3, linguistic_variables=partitions, tolerance=0.01)
    rows = np.sort(np.random.default_rng(0).choice(len(y), size=400, replace=False))
    subset = evf.FitRuleBase(X[rows], y[rows], 8, 3, 3, linguistic_variables=partitions, tolerance=0.01)
    genes = _genes(problem)
    full = [problem._array_score(gene.copy()) for gene in genes]
    problem._activate_rows(rows)
    assert [problem._array_score(gene.copy()) for gene in genes] == \
        [subset._array_score(gene.copy()) for gene in genes]
    problem._restore_rows()
    assert [problem._array_score(gene.copy()) for gene in genes] == full


def test_optimized_partitions_keep_the_full_data_domain(data):
    X, y = data
    problem = _problem(X, y, fixed=False)
    rows = np.arange(0, len(y), 3)
    gene = _genes(problem, 1)[0]
    variables = problem._decode_membership_functions(gene, problem.fuzzy_type)
    problem._activate_rows(rows)
    after = problem._decode_membership_functions(gene, problem.fuzzy_type)
    for before_variable, after_variable in zip(variables, after):
        for before_set, after_set in zip(before_variable, after_variable):
            np.testing.assert_array_equal(before_set.membership_parameters, after_set.membership_parameters)
    rulebase = problem._construct_ruleBase(gene.copy(), problem.fuzzy_type)
    assert problem._array_score(gene.copy()) == score_rulebase(
        rulebase, problem.X, problem.y, problem.tolerance, problem.alpha_, problem.beta_,
        max_conditions=problem._max_conditions)


def test_cached_scores_last_until_the_data_is_complete(data):
    X, y = data
    problem = _problem(X, y)
    first, second = _genes(problem, 2)
    with _fitness_cache_scope(problem, True):
        out = {}
        problem._activate_rows(np.arange(0, len(y), 4))
        problem._evaluate(first.copy(), out)
        quarter = out['F']
        problem._activate_rows(np.arange(0, len(y), 2))
        problem._evaluate(first.copy(), out)
        assert out['F'] == quarter  # kept, like a parent's score
        problem._evaluate(second.copy(), out)
        assert out['F'] == problem._objectives(problem._array_score(second.copy()))
        problem._restore_rows()
        problem._evaluate(first.copy(), out)
        assert out['F'] == problem._objectives(problem._array_score(first.copy()))
        assert out['F'] != quarter


def test_streamed_scores_on_a_subset_are_exact(data):
    X, y = data
    plain, streamed = _problem(X, y), _problem(X, y, batch_size=97)
    rows = np.arange(1, len(y), 2)
    plain._activate_rows(rows)
    streamed._activate_rows(rows)
    for gene in _genes(plain):
        assert streamed._array_score(gene.copy()) == plain._array_score(gene.copy())


def _fit(X, y, partitions, **kwargs):
    fit_options = {key: kwargs.pop(key) for key in ('n_gen', 'patience', 'checkpoints',
                                                     'checkpoint_callback') if key in kwargs}
    model = evf.BaseFuzzyRulesClassifier(nRules=8, nAnts=3, linguistic_variables=partitions,
                                         tolerance=0.01, **kwargs)
    model.fit(X, y, **{'n_gen': 12, 'pop_size': 10, 'random_state': 5, 'patience': None, **fit_options})
    return model


@pytest.mark.parametrize('fraction', [0.1, 0.5])
def test_the_result_is_scored_on_every_sample(data, fraction):
    X, y = data
    partitions = utils.construct_partitions(X, fs.FUZZY_SETS.t1)
    model = _fit(X, y, partitions, initial_data_fraction=fraction)
    assert model.performance == _full_performance(model, X, y, partitions)


def test_the_pareto_front_is_scored_on_every_sample(data):
    X, y = data
    partitions = utils.construct_partitions(X, fs.FUZZY_SETS.t1)
    model = _fit(X, y, partitions, initial_data_fraction=0.2, algorithm='nsga2')
    problem = evf.FitRuleBase(X, y, 8, 3, 3, linguistic_variables=partitions, tolerance=0.01, pareto=True)
    front = model.optimization_result_['front']
    for gene, objectives in zip(front['X'], front['F']):
        score = problem._array_score(np.asarray(gene).astype(int))
        np.testing.assert_array_equal(objectives, problem._objectives(score))


def test_patience_waits_for_the_full_data(data):
    X, y = data
    partitions = utils.construct_partitions(X, fs.FUZZY_SETS.t1)
    model = _fit(X, y, partitions, initial_data_fraction=0.1, full_data_at=0.5, n_gen=20, patience=1)
    assert model.n_generations_run_ >= 11  # generation 10 is the first on every sample


def test_a_full_initial_fraction_changes_nothing(data):
    X, y = data
    partitions = utils.construct_partitions(X, fs.FUZZY_SETS.t1)
    plain, full = _fit(X, y, partitions), _fit(X, y, partitions, initial_data_fraction=1.0)
    assert plain.performance == full.performance
    np.testing.assert_array_equal(plain.optimization_result_['X'], full.optimization_result_['X'])
    np.testing.assert_array_equal(plain.predict(X), full.predict(X))


def test_seeded_fits_repeat(data):
    X, y = data
    partitions = utils.construct_partitions(X, fs.FUZZY_SETS.t1)
    first = _fit(X, y, partitions, initial_data_fraction=0.2)
    again = _fit(X, y, partitions, initial_data_fraction=0.2)
    assert first.performance == again.performance
    np.testing.assert_array_equal(first.optimization_result_['X'], again.optimization_result_['X'])


def test_a_custom_loss_sees_the_subsets(data):
    X, y = data
    sizes = set()

    def loss(rulebase, X, y, *args):
        sizes.add(len(y))
        return 0.5

    model = evf.BaseFuzzyRulesClassifier(nRules=6, nAnts=2, initial_data_fraction=0.2)
    model.customized_loss(loss)
    model.fit(X, y, n_gen=6, pop_size=8, patience=None)
    assert len(y) in sizes and min(sizes) < len(y) // 2


def test_checkpoints_and_batch_size_combine(data):
    X, y = data
    partitions = utils.construct_partitions(X, fs.FUZZY_SETS.t1)
    generations = []
    model = _fit(X, y, partitions, initial_data_fraction=0.2, batch_size=128, checkpoints=3,
                 checkpoint_callback=lambda gen, rule_base: generations.append(gen))
    assert generations == [0, 3, 6, 9]
    assert model.performance == _full_performance(model, X, y, partitions)


def test_fit_can_override_the_schedule(data):
    X, y = data
    model = evf.BaseFuzzyRulesClassifier(nRules=6, nAnts=2, initial_data_fraction=0.2, full_data_at=0.8)
    assert model.get_params()['initial_data_fraction'] == 0.2
    assert model.get_params()['full_data_at'] == 0.8
    with pytest.raises(ValueError, match='full_data_at'):
        model.fit(X, y, n_gen=4, pop_size=6, full_data_at=0)
    model.fit(X, y, n_gen=4, pop_size=6, initial_data_fraction=None, full_data_at=0)


def test_growing_data_needs_pymoo_and_no_candidate_rules(data):
    X, y = data
    model = evf.BaseFuzzyRulesClassifier(nRules=6, nAnts=2, initial_data_fraction=0.2)
    model.backend = object()
    with pytest.raises(ValueError, match='pymoo'):
        model.fit(X, y, n_gen=4, pop_size=6)
    model = evf.BaseFuzzyRulesClassifier(nRules=6, nAnts=2, initial_data_fraction=0.2)
    with pytest.raises(ValueError, match='candidate_rules'):
        model.fit(X, y, n_gen=4, pop_size=6, candidate_rules=rules.MasterRuleBase([rules.RuleBaseT1([], [])]))
