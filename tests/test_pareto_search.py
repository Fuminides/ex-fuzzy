"""The NSGA-II option: exact Pareto objectives, the stored front and its selector."""
import numpy as np
import pytest
from sklearn.datasets import load_iris

from ex_fuzzy import evolutionary_backends as eb
from ex_fuzzy import evolutionary_fit as evf
from ex_fuzzy import evolutionary_search as es
from ex_fuzzy import fuzzy_sets as fs
from ex_fuzzy import rules as rl
from ex_fuzzy import utils
from ex_fuzzy._fitness import score_rulebase


@pytest.fixture(scope='module')
def iris():
    return load_iris(return_X_y=True)


def _problems(X, y, fixed, kind=fs.FUZZY_SETS.t1):
    partitions = utils.construct_partitions(X, kind) if fixed else None
    options = dict(linguistic_variables=partitions, fuzzy_type=kind, tolerance=0.01,
                   alpha=0.1, beta=0.2)
    return (evf.FitRuleBase(X, y, 10, 3, 3, **options),
            evf.FitRuleBase(X, y, 10, 3, 3, pareto=True, **options))


def _genes(problem, count=20):
    return np.random.default_rng(7).integers(
        problem.xl.astype(int), problem.xu.astype(int) + 1, size=(count, problem.n_var))


@pytest.mark.parametrize('kind', [fs.FUZZY_SETS.t1, fs.FUZZY_SETS.t2])
@pytest.mark.parametrize('fixed', [False, True])
def test_pareto_objectives_recombine_into_the_scalar_objective(iris, kind, fixed):
    scalar, pareto = _problems(*iris, fixed, kind)
    assert pareto.n_obj == 2
    for gene in _genes(scalar):
        accuracy, compactness = pareto._array_score(gene.copy())
        assert accuracy - scalar.alpha_ * compactness == scalar._array_score(gene.copy())
        reference = score_rulebase(pareto._construct_ruleBase(gene.copy(), kind), *iris,
                                   pareto.tolerance, pareto.alpha_, pareto.beta_,
                                   pareto._precomputed_truth,
                                   max_conditions=pareto._max_conditions, pareto=True)
        assert reference == (accuracy, compactness)


@pytest.mark.parametrize('fixed', [False, True])
def test_pareto_objectives_agree_on_the_fast_and_reference_routes(iris, fixed):
    _, pareto = _problems(*iris, fixed)
    for gene in _genes(pareto):
        fast, slow = {}, {}
        pareto._evaluate(gene.copy(), fast)
        pareto._evaluate_slow(gene.copy(), slow)
        np.testing.assert_array_equal(fast['F'], slow['F'])
        assert fast['F'].shape == (2,)


def test_pareto_problems_skip_the_batched_population_route(iris):
    scalar, pareto = _problems(*iris, True)
    assert scalar._standard_population_objective()
    assert not pareto._standard_population_objective()


@pytest.fixture(scope='module')
def nsga2(iris):
    X, y = iris
    return evf.BaseFuzzyRulesClassifier(nRules=10, nAnts=3, n_gen=15, pop_size=30, patience=None,
                                        random_state=3, algorithm='nsga2').fit(X, y)


def test_nsga2_keeps_a_front_from_most_accurate_to_most_compact(nsga2):
    front = nsga2.pareto_front_
    assert len(front) > 1
    fitness = [solution['fitness'] for solution in front]
    conditions = [solution['conditions'] for solution in front]
    # Distinct non-dominated solutions: each one trades accuracy for size.
    assert fitness == sorted(fitness, reverse=True) and len(set(fitness)) == len(fitness)
    assert conditions == sorted(conditions, reverse=True) and len(set(conditions)) == len(conditions)
    assert nsga2.pareto_index_ == 0
    assert nsga2.rule_base is front[0]['rule_base']
    assert nsga2.performance == front[0]['fitness']
    for solution in front:
        rules = solution['rule_base'].get_rules()
        assert solution['rules'] == len(rules)


def test_select_pareto_solution_switches_the_fitted_model(nsga2, iris):
    X, _ = iris
    most_accurate = nsga2.predict(X)
    try:
        assert nsga2.select_pareto_solution(-1) is nsga2
        assert nsga2.pareto_index_ == len(nsga2.pareto_front_) - 1
        assert nsga2.rule_base is nsga2.pareto_front_[-1]['rule_base']
        assert nsga2.performance == nsga2.pareto_front_[-1]['fitness']
        # Partitions were optimized, so each solution brings its own.
        assert nsga2.lvs is nsga2.pareto_front_[-1]['partitions']
        np.testing.assert_array_equal(
            nsga2.predict(X), nsga2.pareto_front_[-1]['rule_base'].winning_rule_predict(X))
        with pytest.raises(ValueError):
            nsga2.select_pareto_solution(len(nsga2.pareto_front_))
    finally:
        nsga2.select_pareto_solution(0)
    np.testing.assert_array_equal(nsga2.predict(X), most_accurate)


def test_select_pareto_solution_needs_a_pareto_fit(iris):
    X, y = iris
    ga = evf.BaseFuzzyRulesClassifier(nRules=6, nAnts=2, n_gen=2, pop_size=10).fit(X, y)
    assert ga.pareto_front_ is None and ga.pareto_index_ is None
    with pytest.raises(ValueError, match='nsga2'):
        ga.select_pareto_solution(0)


class _OtherBackend(eb.EvolutionaryBackend):
    def optimize(self, *args, **kwargs):
        raise AssertionError('the search must not start')

    def is_available(self):
        return True

    def name(self):
        return 'other'


@pytest.mark.parametrize('options, message', [
    (dict(algorithm='nsga3'), 'Unknown algorithm'),
    (dict(algorithm='nsga2', backend=_OtherBackend()), 'pymoo backend'),
])
def test_unsupported_searches_are_rejected(iris, options, message):
    with pytest.raises(ValueError, match=message):
        evf.BaseFuzzyRulesClassifier(n_gen=2, pop_size=10, **options).fit(*iris)


def test_nsga2_rejects_a_custom_loss(iris):
    classifier = evf.BaseFuzzyRulesClassifier(n_gen=2, pop_size=10, algorithm='nsga2')
    classifier.customized_loss(lambda *args: 0.5)
    with pytest.raises(ValueError, match='custom loss'):
        classifier.fit(*iris)


def test_nsga2_selects_from_candidate_rules(iris):
    X, y = iris
    partitions = utils.construct_partitions(X, fs.FUZZY_SETS.t1)
    candidates = rl.MasterRuleBase([
        rl.RuleBaseT1(partitions, [rl.RuleSimple([-1, -1, 0, -1]), rl.RuleSimple([0, -1, -1, -1])]),
        rl.RuleBaseT1(partitions, [rl.RuleSimple([-1, -1, 1, -1]), rl.RuleSimple([-1, 1, 1, -1])]),
        rl.RuleBaseT1(partitions, [rl.RuleSimple([-1, -1, 2, -1]), rl.RuleSimple([2, -1, 2, -1])])])
    classifier = evf.BaseFuzzyRulesClassifier(nRules=4, linguistic_variables=partitions, n_gen=5,
                                              pop_size=12, patience=None, algorithm='nsga2')
    classifier.fit(X, y, candidate_rules=candidates)
    assert isinstance(classifier.pareto_front_, list) and classifier.pareto_front_
    problem = es.ExploreRuleBases(X, y, nRules=4, n_classes=3, candidate_rules=candidates, pareto=True)
    assert problem.n_obj == 2
