"""The object-free evaluator must return the object decoder's exact objective."""
import numpy as np
import pytest
from sklearn.datasets import load_iris, make_classification

import evolutionary_fit as evf
import fuzzy_sets as fs
import utils
from _fitness import score_rulebase


def _reference(problem, gene):
    """The objective as computed through rule objects."""
    rulebase = problem._construct_ruleBase(gene.copy(), problem.fuzzy_type)
    return score_rulebase(rulebase, problem.X, problem.y, problem.tolerance,
                          problem.alpha_, problem.beta_, problem._precomputed_truth)


def _genes(problem, count, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(problem.xl.astype(int), problem.xu.astype(int) + 1,
                        size=(count, problem.n_var))


def _problem(X, y, kind, fixed, **kwargs):
    partitions = utils.construct_partitions(X, kind) if fixed else None
    return evf.FitRuleBase(X, y, kwargs.pop('nRules', 12), kwargs.pop('nAnts', 3),
                           len(np.unique(y)), linguistic_variables=partitions,
                           fuzzy_type=kind, **kwargs)


@pytest.mark.parametrize('kind', [fs.FUZZY_SETS.t1, fs.FUZZY_SETS.t2])
@pytest.mark.parametrize('fixed', [True, False])
@pytest.mark.parametrize('ds_mode', [0, 1, 2])
def test_array_objective_matches_object_decoder(kind, fixed, ds_mode):
    X, y = load_iris(return_X_y=True)
    problem = _problem(X, y, kind, fixed, ds_mode=ds_mode, tolerance=0.01)
    used = 0
    for gene in _genes(problem, 40, 3):
        actual = problem._array_score(gene.copy())
        assert actual is not None
        used += 1
        assert actual == _reference(problem, gene)
    assert used == 40


@pytest.mark.parametrize('kind', [fs.FUZZY_SETS.t1, fs.FUZZY_SETS.t2])
@pytest.mark.parametrize('allow_unknown', [True, False])
@pytest.mark.parametrize('tolerance', [0.0, 0.01, 0.5, 1.5])
def test_array_objective_matches_across_tolerances(kind, allow_unknown, tolerance):
    X, y = load_iris(return_X_y=True)
    problem = _problem(X, y, kind, True, tolerance=tolerance,
                       allow_unknown=allow_unknown)
    for gene in _genes(problem, 25, 9):
        assert problem._array_score(gene.copy()) == _reference(problem, gene)


@pytest.mark.parametrize('kind', [fs.FUZZY_SETS.t1, fs.FUZZY_SETS.t2])
@pytest.mark.parametrize('alpha,beta', [(0.1, 0.0), (0.0, 0.2), (0.3, 0.4)])
def test_array_objective_matches_with_complexity_penalties(kind, alpha, beta):
    X, y = load_iris(return_X_y=True)
    for fixed in (True, False):
        problem = _problem(X, y, kind, fixed, alpha=alpha, beta=beta, tolerance=0.01)
        for gene in _genes(problem, 20, 17):
            assert problem._array_score(gene.copy()) == _reference(problem, gene)


@pytest.mark.parametrize('kind', [fs.FUZZY_SETS.t1, fs.FUZZY_SETS.t2])
def test_array_objective_matches_on_wider_random_data(kind):
    X, y = make_classification(n_samples=300, n_features=8, n_informative=7,
                               n_redundant=0, n_classes=4, random_state=1)
    for fixed in (True, False):
        problem = _problem(X, y, kind, fixed, nRules=16, nAnts=4, tolerance=0.005)
        for gene in _genes(problem, 30, 41):
            assert problem._array_score(gene.copy()) == _reference(problem, gene)


@pytest.mark.parametrize('kind', [fs.FUZZY_SETS.t1, fs.FUZZY_SETS.t2])
def test_array_objective_handles_degenerate_candidates(kind):
    X, y = load_iris(return_X_y=True)
    problem = _problem(X, y, kind, True, nRules=6, nAnts=2, tolerance=0.01)
    empty = np.zeros(problem.n_var, dtype=int)
    empty[2 * problem.nRules * problem.nAnts:] = -1  # every rule disabled
    assert problem._array_score(empty.copy()) == 0.0 == _reference(problem, empty)
    # Every rule identical: duplicate removal must leave a single rule per class.
    duplicated = np.zeros(problem.n_var, dtype=int)
    assert problem._array_score(duplicated.copy()) == _reference(problem, duplicated)
    # An impossible tolerance prunes everything.
    strict = _problem(X, y, kind, True, nRules=6, nAnts=2, tolerance=2.0)
    for gene in _genes(strict, 5, 2):
        assert strict._array_score(gene.copy()) == 0.0 == _reference(strict, gene)


@pytest.mark.parametrize('kind', [fs.FUZZY_SETS.t1, fs.FUZZY_SETS.t2])
def test_array_decoder_reproduces_the_object_phenotype(kind):
    """Rule order, antecedents, consequents and weights must all match."""
    import _array_fitness as arrfit
    X, y = load_iris(return_X_y=True)
    for ds_mode in (0, 1, 2):
        problem = _problem(X, y, kind, True, ds_mode=ds_mode)
        term_counts = np.asarray([len(lv) for lv in problem.lvs])
        for gene in _genes(problem, 30, 77):
            decoded = arrfit.decode_rule_arrays(
                gene.astype(int), problem.nRules, problem.nAnts, X.shape[1],
                problem.n_classes, term_counts, ds_mode,
                problem._consequent_pointer(kind))
            base = problem._construct_ruleBase(gene.copy(), kind)
            expected = base.get_rules()
            assert len(expected) == len(decoded.consequents)
            for index, rule in enumerate(expected):
                assert rule.antecedents == list(decoded.antecedents[index])
            assert base.get_consequents() == list(decoded.consequents)
            if ds_mode == 2:
                np.testing.assert_array_equal(base.get_weights(), decoded.weights)


def test_array_path_declines_unsupported_problems():
    X, y = load_iris(return_X_y=True)
    problem = _problem(X, y, fs.FUZZY_SETS.t1, True)
    gene = _genes(problem, 1, 4)[0]
    assert problem._array_score(gene.copy(), time_moment=0) is None
    problem.array_evaluation = False
    assert problem._array_score(gene.copy()) is None
    problem.array_evaluation = True

    class Subclassed(evf.FitRuleBase):
        pass

    other = Subclassed(X, y, 6, 2, 3,
                       linguistic_variables=utils.construct_partitions(X, fs.FUZZY_SETS.t1))
    assert other._array_score(gene.copy()) is None


@pytest.mark.parametrize('kind', [fs.FUZZY_SETS.t1, fs.FUZZY_SETS.t2])
@pytest.mark.parametrize('fixed', [True, False])
def test_seeded_fits_are_identical_with_and_without_the_array_path(kind, fixed):
    X, y = load_iris(return_X_y=True)
    partitions = utils.construct_partitions(X, kind) if fixed else None
    results = {}
    for enabled in (True, False):
        evf.FitRuleBase.array_evaluation = enabled
        try:
            model = evf.BaseFuzzyRulesClassifier(
                nRules=10, nAnts=3, linguistic_variables=partitions,
                fuzzy_type=kind, tolerance=0.01)
            model.fit(X, y, n_gen=5, pop_size=12, random_state=7, patience=None)
        finally:
            evf.FitRuleBase.array_evaluation = True
        results[enabled] = (model.performance, model.optimization_result_['X'].copy(),
                            model.rule_base.get_consequents(),
                            np.asarray(model.rule_base.get_scores()),
                            np.asarray(model.predict(X)))
    left, right = results[True], results[False]
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2] == right[2]
    np.testing.assert_array_equal(left[3], right[3])
    np.testing.assert_array_equal(left[4], right[4])


def test_optimized_fit_matches_with_a_categorical_variable():
    """A categorical feature packs like any other and stays exact."""
    rng = np.random.default_rng(4)
    X = np.column_stack([rng.normal(size=120), rng.integers(0, 3, size=120)])
    y = rng.integers(0, 2, size=120)
    mask = np.array([False, True])
    problem = evf.FitRuleBase(X, y, 8, 2, 2, fuzzy_type=fs.FUZZY_SETS.t1,
                              categorical_mask=mask)
    genes = np.random.default_rng(6).integers(
        problem.xl.astype(int), problem.xu.astype(int) + 1, size=(10, problem.n_var))
    for gene in genes:
        assert problem._array_score(gene.copy()) == _reference(problem, gene)
