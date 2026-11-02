"""Streaming the samples in batches (batch_size) must not change any score."""
import numpy as np
import pytest
from sklearn.datasets import load_iris, make_classification

from ex_fuzzy import evolutionary_fit as evf
from ex_fuzzy import fuzzy_sets as fs
from ex_fuzzy import rules
from ex_fuzzy import utils
from ex_fuzzy import _streamed_fitness as streamfit
from ex_fuzzy._fitness import _dominance, _fitness_cache_scope

#: More samples than one NumPy buffer, so sums cross the buffer boundaries where
#: NumPy's summation order changes between releases.
N_SAMPLES = np.getbufsize() + 808


@pytest.fixture(scope='module')
def summation():
    measured = streamfit.Summation.measure()
    assert measured is not None, f'NumPy {np.__version__} sums in an unrecognized order'
    return measured


@pytest.fixture(scope='module')
def data():
    return make_classification(n_samples=N_SAMPLES, n_features=5, n_informative=4, n_redundant=0,
                               n_classes=3, n_clusters_per_class=1, weights=[0.25, 0.25, 0.5],
                               random_state=3)


def _streamed_total(values, layout, batch):
    stream = streamfit._StreamingSum(layout, len(values))
    for start in range(0, values.shape[1], batch):
        stream.feed(values[:, start:start + batch])
    return stream.total()


def test_summation_order_is_recognized(summation):
    assert set(summation.buffers) == set(streamfit.Summation.LAYOUTS)
    assert set(summation.buffers.values()) <= {None, np.getbufsize()}


@pytest.mark.parametrize('length', [0, 1, 7, 8, 128, 129, 1000, 8192, 8193, 3 * 8192 + 5])
@pytest.mark.parametrize('batch', [1, 13, 128, 1000, 10 ** 6])
def test_streamed_sum_matches_numpy(summation, length, batch):
    if batch == 1 and length > 1000:
        pytest.skip('slow and covered by the shorter lengths')
    rng = np.random.default_rng(length)
    values = rng.random((length, 3, 2))
    rows = np.ascontiguousarray(values[:, :, 0].T)
    pairs = np.ascontiguousarray(values.transpose(1, 0, 2).reshape(3, -1))
    np.testing.assert_array_equal(
        _streamed_total(rows, summation.layout(length, 'rows'), batch), np.sum(rows, axis=1))
    np.testing.assert_array_equal(
        _streamed_total(pairs, summation.layout(2 * length, 'interval'), batch),
        [np.sum(values[:, index]) for index in range(3)])
    single = [np.sum(np.ascontiguousarray(values[:, index])) for index in range(3)]
    np.testing.assert_array_equal(
        _streamed_total(pairs, summation.layout(2 * length, 'single_interval'), batch), single)


@pytest.mark.parametrize('interval', [False, True])
@pytest.mark.parametrize('n_rules', [1, 2, 7])
def test_dominance_sums_match_dominance(summation, interval, n_rules):
    rng = np.random.default_rng(n_rules)
    y = rng.choice(4, size=N_SAMPLES, p=[0.3, 0.3, 0.4, 0.0])  # class 3 has no samples
    consequents = np.sort(rng.choice(4, size=n_rules))
    firing = rng.random((N_SAMPLES, n_rules) + ((2,) if interval else ()))
    firing[rng.random(firing.shape) < 0.3] = 0.0
    sums = streamfit._DominanceSums(consequents, y, interval, summation)
    for start in range(0, N_SAMPLES, 999):
        sums.feed(firing[start:start + 999], y[start:start + 999])
    np.testing.assert_array_equal(sums.scores(), _dominance(firing, y, consequents))
    # The scores of a subset are those of the subset's own firing array, a
    # single rule included (its Type-2 denominator has another layout).
    for keep in [np.arange(n_rules) == index for index in range(n_rules)] + [rng.random(n_rules) < 0.5]:
        if keep.any():
            np.testing.assert_array_equal(
                sums.scores(keep),
                _dominance(np.ascontiguousarray(firing[:, keep]), y, consequents[keep]))


def _problems(X, y, batch_size, kind=fs.FUZZY_SETS.t1, fixed=True, **kwargs):
    partitions = utils.construct_partitions(X, kind) if fixed else None
    options = dict(linguistic_variables=partitions, fuzzy_type=kind, **{'tolerance': 0.01, **kwargs})
    return (evf.FitRuleBase(X, y, 8, 3, 3, **options),
            evf.FitRuleBase(X, y, 8, 3, 3, batch_size=batch_size, **options))


def _genes(problem, count, seed=5):
    return np.random.default_rng(seed).integers(problem.xl.astype(int), problem.xu.astype(int) + 1,
                                                size=(count, problem.n_var))


@pytest.mark.parametrize('kind', [fs.FUZZY_SETS.t1, fs.FUZZY_SETS.t2])
@pytest.mark.parametrize('fixed', [True, False])
@pytest.mark.parametrize('ds_mode', [0, 1, 2])
@pytest.mark.parametrize('allow_unknown', [False, True])
def test_streamed_objective_matches_unbatched(data, kind, fixed, ds_mode, allow_unknown):
    X, y = data
    plain, streamed = _problems(X, y, 1000, kind, fixed, ds_mode=ds_mode, allow_unknown=allow_unknown)
    assert streamed._streamed_summation is not None
    for gene in _genes(plain, 6):
        expected = plain._array_score(gene.copy())
        assert expected is not None
        assert streamed._array_score(gene.copy()) == expected


@pytest.mark.parametrize('batch_size', [1, 37, 4099])
def test_batch_size_never_changes_the_objective(data, batch_size):
    X, y = data
    X, y = (X, y) if batch_size > 1 else (X[:300], y[:300])
    plain, streamed = _problems(X, y, batch_size, fs.FUZZY_SETS.t2, tolerance=0.0)
    for gene in _genes(plain, 8, seed=batch_size):
        assert streamed._array_score(gene.copy()) == plain._array_score(gene.copy())


@pytest.mark.parametrize('tolerance', [0.0, 0.05, 0.5])
def test_streamed_pareto_objectives_match(data, tolerance):
    X, y = data
    plain, streamed = _problems(X, y, 2048, pareto=True, tolerance=tolerance)
    for gene in _genes(plain, 6):
        expected = plain._array_score(gene.copy())
        assert isinstance(expected, tuple)
        assert streamed._array_score(gene.copy()) == expected


def test_streaming_turns_the_population_routes_off(data):
    X, y = data
    plain, streamed = _problems(X, y, 1000)
    assert plain._standard_population_objective()
    assert not streamed._standard_population_objective()
    # Population entry points (EvoX) still score every candidate, streamed.
    genes = _genes(plain, 10)
    with _fitness_cache_scope(plain, True), _fitness_cache_scope(streamed, True):
        expected, _ = plain._evaluate_gene_population(genes)
        actual, on_device = streamed._evaluate_gene_population(genes)
    assert not on_device
    np.testing.assert_array_equal(actual, expected)


def test_a_single_batch_is_not_streamed(data):
    X, y = data
    problem = evf.FitRuleBase(X, y, 8, 3, 3, linguistic_variables=utils.construct_partitions(X, fs.FUZZY_SETS.t1),
                              batch_size=len(X))
    assert problem.batch_size == len(X)
    assert problem._streamed_summation is None
    assert problem._standard_population_objective()


@pytest.mark.parametrize('kind', [fs.FUZZY_SETS.t1, fs.FUZZY_SETS.t2])
@pytest.mark.parametrize('fixed', [True, False])
def test_seeded_fits_are_identical_with_batch_size(kind, fixed):
    X, y = load_iris(return_X_y=True)
    partitions = utils.construct_partitions(X, kind) if fixed else None
    results = []
    for batch_size in (None, 16):
        model = evf.BaseFuzzyRulesClassifier(nRules=10, nAnts=3, linguistic_variables=partitions,
                                             fuzzy_type=kind, tolerance=0.01, batch_size=batch_size)
        model.fit(X, y, n_gen=5, pop_size=12, random_state=7, patience=None)
        results.append((model.performance, model.optimization_result_['X'].copy(),
                        model.rule_base.get_consequents(), np.asarray(model.rule_base.get_scores()),
                        np.asarray(model.predict(X))))
    (left, right) = results
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2] == right[2]
    np.testing.assert_array_equal(left[3], right[3])
    np.testing.assert_array_equal(left[4], right[4])


def test_fit_can_override_batch_size():
    X, y = load_iris(return_X_y=True)
    model = evf.BaseFuzzyRulesClassifier(nRules=6, nAnts=2, batch_size=16)
    assert model.get_params()['batch_size'] == 16
    model.fit(X, y, n_gen=2, pop_size=6, batch_size=None, patience=None)
    assert model.batch_size == 16


@pytest.mark.parametrize('batch_size', [0, -5, 2.5, True, '100'])
def test_invalid_batch_sizes_are_rejected(batch_size):
    X, y = load_iris(return_X_y=True)
    with pytest.raises(ValueError, match='batch_size'):
        evf.FitRuleBase(X, y, 6, 2, 3, batch_size=batch_size)


def test_batch_size_rejects_general_type2():
    X, y = load_iris(return_X_y=True)
    partitions = utils.construct_partitions(X, fs.FUZZY_SETS.gt2)
    with pytest.raises(ValueError, match='Type-1 and interval Type-2'):
        evf.FitRuleBase(X, y, 6, 2, 3, linguistic_variables=partitions, batch_size=10)


def test_batch_size_rejects_custom_losses_and_candidate_rules():
    X, y = load_iris(return_X_y=True)
    model = evf.BaseFuzzyRulesClassifier(nRules=6, nAnts=2, batch_size=16)
    model.customized_loss(lambda *args: 0.0)
    with pytest.raises(ValueError, match='custom loss'):
        model.fit(X, y, n_gen=2, pop_size=6)
    model = evf.BaseFuzzyRulesClassifier(nRules=6, nAnts=2, batch_size=16)
    candidates = rules.MasterRuleBase([rules.RuleBaseT1([], [])])
    with pytest.raises(ValueError, match='candidate_rules'):
        model.fit(X, y, n_gen=2, pop_size=6, candidate_rules=candidates)


def test_unrecognized_summation_order_is_refused(monkeypatch):
    X, y = load_iris(return_X_y=True)
    monkeypatch.setattr(streamfit.Summation, 'measure', classmethod(lambda cls: None))
    with pytest.raises(ValueError, match='NumPy'):
        evf.FitRuleBase(X, y, 6, 2, 3, batch_size=10)
