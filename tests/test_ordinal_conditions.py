"""Ordinal ``at least`` / ``at most`` conditions: encoding, membership, search, estimators and persistence."""
import json
import os
import pickle

import numpy as np
import pytest
from sklearn.datasets import load_diabetes, load_iris, make_classification

from ex_fuzzy import evolutionary_fit as evf
from ex_fuzzy import evolutionary_fit_regression as efr
from ex_fuzzy import fuzzy_sets as fs
from ex_fuzzy import pattern_stability as ps
from ex_fuzzy import persistence, rule_mixture, rules, utils, vis_rules
from ex_fuzzy._fitness import _fitness_cache_scope, remap_term, score_rulebase

GOLDEN = os.path.join(os.path.dirname(__file__), 'data', 'ordinal_default_golden.json')


@pytest.fixture(scope='module')
def iris():
    return load_iris(return_X_y=True)


def _strong_variable(n_terms=4, ordinal=True, name='x'):
    """Strong trapezoidal partition of [0, 1] with ``n_terms`` terms (memberships add up to 1)."""
    edges = np.linspace(0, 1, n_terms + 1)
    width = 1.0 / (n_terms * 4)
    sets = []
    for k in range(n_terms):
        left = edges[k]
        right = edges[k + 1]
        a = left - width if k else left
        b = left + width if k else left
        c = right - width if k < n_terms - 1 else right
        d = right + width if k < n_terms - 1 else right
        sets.append(fs.FS(f'T{k}', [a, b, c, d], [0.0, 1.0]))
    return fs.fuzzyVariable(name, sets, ordinal_conditions=ordinal)


# ---------------------------------------------------------------------------
# Alphabet, codes and names.

@pytest.mark.parametrize('n_terms,size', [(1, 1), (2, 2), (3, 5), (4, 8), (5, 11)])
def test_alphabet_size_is_terms_plus_two_per_inner_term(n_terms, size):
    assert fs.ordinal_alphabet_size(n_terms) == size


def test_four_terms_give_eight_options_and_exclude_vacuous_ones():
    variable = _strong_variable(4)
    assert variable.n_conditions() == 8
    assert variable.condition_names() == ['T0', 'T1', 'T2', 'T3', 'at least T1', 'at least T2', 'at most T1', 'at most T2']
    covered = {name: variable.condition_terms(code) for code, name in enumerate(variable.condition_names())}
    assert covered['at least T1'] == [1, 2, 3] and covered['at most T2'] == [0, 1, 2]
    # at least T0 (= don't care) and at least T3 (= T3), at most T3 and at most T0 are not offered.
    assert 'at least T0' not in covered and 'at least T3' not in covered
    assert 'at most T3' not in covered and 'at most T0' not in covered
    # No two codes describe the same set of terms, so a set has one code and rules over it cannot be duplicates.
    sets = [tuple(terms) for terms in covered.values()]
    assert len(set(sets)) == len(sets)


@pytest.mark.parametrize('n_terms', [3, 4, 5, 7])
def test_condition_codes_round_trip(n_terms):
    variable = _strong_variable(n_terms)
    for code in range(variable.n_conditions()):
        operator, term = variable.condition(code)
        assert variable.condition_code(operator, term) == code
        assert fs.decode_ordinal_condition(code, n_terms) == (operator, term)
    with pytest.raises(ValueError):
        variable.condition(variable.n_conditions())
    with pytest.raises(ValueError):
        variable.condition_code('at_least', 0)
    with pytest.raises(ValueError):
        variable.condition_code('at_most', n_terms - 1)


def test_a_variable_without_ordinal_conditions_is_unchanged():
    variable = _strong_variable(4, ordinal=False)
    assert variable.ordinal is False and variable.n_conditions() == 4
    assert variable.condition_names() == variable.linguistic_variable_names()
    assert variable.compute_memberships(np.linspace(0, 1, 7)).shape == (4, 7)
    with pytest.raises(ValueError):
        variable.condition_code('at_least', 1)


def test_ordinal_conditions_need_three_ordered_non_categorical_terms():
    two = [fs.FS('lo', [0, 0, 1, 2], [0, 4]), fs.FS('hi', [1, 2, 4, 4], [0, 4])]
    with pytest.raises(ValueError):
        fs.fuzzyVariable('x', two, ordinal_conditions=True)
    categorical = [fs.categoricalFS(name, name) for name in 'abc']
    with pytest.raises(ValueError):
        fs.fuzzyVariable('c', categorical, ordinal_conditions=True)
    # The search skips them quietly: categories are not ordered.
    cat = fs.fuzzyVariable('c', categorical)
    assert fs.enable_ordinal_conditions([cat])[0] is cat
    # General Type-2 sets are refused clearly.
    gt2 = utils.construct_partitions(np.random.default_rng(0).normal(size=(50, 1)), fs.FUZZY_SETS.gt2)
    with pytest.raises(ValueError, match='general Type-2'):
        fs.enable_ordinal_conditions(gt2)
    with pytest.raises(ValueError, match='general Type-2'):
        evf.FitRuleBase(np.zeros((5, 1)), np.zeros(5), 2, 1, 1, linguistic_variables=gt2, ordinal_conditions=True)


def test_unordered_terms_warn():
    sets = [fs.FS('b', [3, 4, 5, 6], [0, 9]), fs.FS('a', [0, 1, 2, 3], [0, 9]), fs.FS('c', [6, 7, 8, 9], [0, 9])]
    with pytest.warns(UserWarning, match='not listed from low to high'):
        fs.fuzzyVariable('x', sets, ordinal_conditions=True)


def test_enable_ordinal_conditions_copies_and_keeps_the_originals():
    originals = [_strong_variable(3, ordinal=False, name='a'), _strong_variable(4, ordinal=False, name='b')]
    enabled = fs.enable_ordinal_conditions(originals)
    assert [v.ordinal for v in originals] == [False, False]
    assert [v.ordinal for v in enabled] == [True, True]
    assert enabled[0].linguistic_variables is not None and enabled[0].name == 'a'
    assert fs.enable_ordinal_conditions(enabled)[0] is enabled[0]


# ---------------------------------------------------------------------------
# Membership: bounded sum.

def test_union_on_a_strong_partition_is_the_complement_of_the_other_terms():
    variable = _strong_variable(5)
    x = np.linspace(0, 1, 401)
    base = variable._term_memberships(x)
    assert np.allclose(base.sum(axis=0), 1.0)  # strong
    memberships = variable.compute_memberships(x)
    assert memberships.shape == (11, 401)
    np.testing.assert_array_equal(memberships[:5], base)
    for j in range(1, 4):
        at_least = memberships[variable.condition_code('at_least', j)]
        at_most = memberships[variable.condition_code('at_most', j)]
        np.testing.assert_allclose(at_least, base[j:].sum(axis=0), atol=1e-12)
        np.testing.assert_allclose(at_least, 1 - base[:j].sum(axis=0), atol=1e-12)
        np.testing.assert_allclose(at_most, 1 - base[j + 1:].sum(axis=0), atol=1e-12)
        # The bounded sum is a shouldered trapezoid: 0, then linear, then 1 (monotone, one ramp).
        assert np.all(np.diff(at_least) >= -1e-12) and np.all(np.diff(at_most) <= 1e-12)
        assert at_least.max() == pytest.approx(1.0) and at_least.min() == pytest.approx(0.0)


def test_union_saturates_at_one_when_memberships_overlap_beyond_one():
    X = np.random.default_rng(0).normal(size=(300, 1))
    variable = fs.fuzzyVariable('g', utils.construct_partitions(X, n_partitions=4, shape='gaussian',
                                                                detect_categorical=False)[0].linguistic_variables,
                                ordinal_conditions=True)
    x = np.linspace(X.min(), X.max(), 500)
    base = variable._term_memberships(x)
    assert base.sum(axis=0).max() > 1.2  # Gaussians do not form a strong partition
    memberships = variable.compute_memberships(x)
    assert memberships.max() <= 1.0 and memberships.min() >= 0.0
    np.testing.assert_allclose(memberships[variable.condition_code('at_least', 1)],
                               np.minimum(1.0, base[1:].sum(axis=0)))


def test_interval_type_2_bounds_are_summed_apart():
    X = np.random.default_rng(1).normal(size=(200, 1))
    parts = utils.construct_partitions(X, fs.FUZZY_SETS.t2, n_partitions=3, detect_categorical=False)
    variable = fs.fuzzyVariable('v', parts[0].linguistic_variables, ordinal_conditions=True)
    x = np.linspace(X.min(), X.max(), 300)
    base = variable._term_memberships(x)  # (terms, samples, 2)
    memberships = variable.compute_memberships(x)
    assert memberships.shape == (5, 300, 2)
    expected = np.minimum(1.0, base[1:].sum(axis=0))
    np.testing.assert_allclose(memberships[variable.condition_code('at_least', 1)], expected)
    assert np.all(memberships[..., 0] <= memberships[..., 1] + 1e-12)


def test_product_t_norm_multiplies_the_union_with_the_other_conditions():
    a, b = _strong_variable(4, name='a'), _strong_variable(3, name='b')
    X = np.random.default_rng(2).random((40, 2))
    rule = rules.RuleSimple([a.condition_code('at_least', 2), b.condition_code('at_most', 1)])
    base = rules.RuleBaseT1([a, b], [rule])
    firing = base.compute_rule_antecedent_memberships(X)[:, 0]
    mu_a = a.compute_memberships(X[:, 0])[a.condition_code('at_least', 2)]
    mu_b = b.compute_memberships(X[:, 1])[b.condition_code('at_most', 1)]
    np.testing.assert_allclose(firing, mu_a * mu_b)
    assert firing.max() <= 1.0


def test_default_partitions_of_the_library_are_strong(iris):
    X, _ = iris
    for n in (3, 4, 5):
        variable = utils.construct_partitions(X, n_partitions=n)[0]
        grid = np.linspace(X[:, 0].min(), X[:, 0].max(), 500)
        np.testing.assert_allclose(variable.compute_memberships(grid).sum(axis=0), 1.0)


# ---------------------------------------------------------------------------
# Default behaviour is bit-for-bit the 3.4.0 one.

def _golden_run():
    import importlib.util
    spec = importlib.util.spec_from_file_location('golden_ordinal_default',
                                                  os.path.join(os.path.dirname(__file__), 'golden_ordinal_default.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return json.loads(json.dumps(module.record(), sort_keys=True))


def test_default_fits_equal_the_recorded_release_fits():
    """Genes, rules, predictions and objective of nine fits recorded on 3.4.0 (commit 10d811b) are reproduced."""
    with open(GOLDEN) as handle:
        recorded = json.load(handle)
    current = _golden_run()
    assert sorted(current) == sorted(recorded)
    for key in recorded:
        assert current[key] == recorded[key], key


def test_default_encoding_has_the_release_bounds_and_codes(iris):
    X, y = iris
    partitions = utils.construct_partitions(X, n_partitions=4)
    off = evf.FitRuleBase(X, y, 6, 3, 3, linguistic_variables=partitions)
    assert off.n_term_codes == off.n_lv_possible == [4, 4, 4, 4]
    assert int(off.xu[18:36].max()) == 3 and off.lvs is partitions
    explicit = evf.FitRuleBase(X, y, 6, 3, 3, linguistic_variables=partitions, ordinal_conditions=False)
    np.testing.assert_array_equal(off.xu, explicit.xu)
    on = evf.FitRuleBase(X, y, 6, 3, 3, linguistic_variables=partitions, ordinal_conditions=True)
    assert on.n_term_codes == [8, 8, 8, 8] and int(on.xu[18:36].max()) == 7
    assert on.n_var == off.n_var  # same number of genes: the term gene is wider, no gene is added
    assert not any(v.ordinal for v in partitions)  # the caller's variables are not modified


def test_old_pickles_of_variables_and_estimators_load_as_off(iris):
    X, y = iris
    variable = _strong_variable(3, ordinal=False)
    state = dict(variable.__dict__)
    state.pop('ordinal', None)
    restored = fs.fuzzyVariable.__new__(fs.fuzzyVariable)
    restored.__dict__.update(state)
    assert restored.ordinal is False and restored.compute_memberships(np.array([0.5])).shape == (3, 1)
    clf = evf.BaseFuzzyRulesClassifier(nRules=4, nAnts=2, n_gen=2, pop_size=6)
    del clf.__dict__['ordinal_conditions']
    assert clf.ordinal_conditions is False
    reg = efr.BaseFuzzyRulesRegressor()
    del reg.__dict__['ordinal_conditions']
    assert reg.ordinal_conditions is False


# ---------------------------------------------------------------------------
# The search: encoding, decoding, routes.

def _problem(fixed=True, kind=fs.FUZZY_SETS.t1, samples=120, n_terms=4, nRules=10, nAnts=3, **kwargs):
    X, y = make_classification(n_samples=samples, n_features=5, n_informative=4, n_redundant=0, n_classes=3,
                               random_state=3)
    if fixed:
        partitions = utils.construct_partitions(X, kind, n_partitions=n_terms)
        return evf.FitRuleBase(X, y, nRules, nAnts, 3, linguistic_variables=partitions,
                               ordinal_conditions=True, **kwargs)
    return evf.FitRuleBase(X, y, nRules, nAnts, 3, fuzzy_type=kind, n_linguistic_variables=n_terms,
                           ordinal_conditions=True, **kwargs)


def _genes(problem, count=30, seed=4):
    return np.random.default_rng(seed).integers(problem.xl.astype(int), problem.xu.astype(int) + 1,
                                                size=(count, problem.n_var))


def _object_score(problem, gene):
    rule_base = problem._construct_ruleBase(gene, problem.fuzzy_type)
    return score_rulebase(rule_base, problem.X, problem.y, problem.tolerance, problem.alpha_, problem.beta_,
                          problem._precomputed_truth, max_conditions=problem._max_conditions, pareto=problem.pareto)


@pytest.mark.parametrize('fixed', [True, False])
@pytest.mark.parametrize('kind', [fs.FUZZY_SETS.t1, fs.FUZZY_SETS.t2])
@pytest.mark.parametrize('batch_size', [None, 37])
def test_array_route_equals_the_object_oracle(fixed, kind, batch_size):
    problem = _problem(fixed=fixed, kind=kind, batch_size=batch_size, alpha=0.05, beta=0.1)
    used_ordinal = False
    for gene in _genes(problem):
        used_ordinal |= any(
            any(a >= len(v.linguistic_variables) for a, v in zip(r.antecedents, rb.antecedents))
            for rb in [problem._construct_ruleBase(gene, kind)] for r in rb.get_rules())
        assert problem._array_score(gene) == _object_score(problem, gene)
    assert used_ordinal


def test_population_route_equals_the_scalar_route():
    problem = _problem(fixed=True)
    genes = _genes(problem, 40)
    expected = np.asarray([problem._array_score(gene) for gene in genes])
    with _fitness_cache_scope(problem, True):
        actual = problem._population_scores(genes)
    assert actual is not None
    np.testing.assert_array_equal(actual, expected)


def test_torch_objective_equals_the_array_route_and_declines_optimized_partitions():
    pytest.importorskip('torch')
    from ex_fuzzy import _torch_fitness
    problem = _problem(fixed=True)
    genes = _genes(problem, 24)
    objective = _torch_fitness.TorchObjective.build(problem, 'cpu')
    assert objective is not None
    scores, declined = objective.score(genes)
    assert not declined.any()
    expected = np.asarray([problem._array_score(gene) for gene in genes])
    np.testing.assert_array_equal(1 - scores, 1 - expected)
    assert _torch_fitness.TorchObjective.build(_problem(fixed=False), 'cpu') is None


def test_gene_bounds_cover_every_code_and_remap_proportionally():
    X = np.random.default_rng(0).random((60, 3))
    y = np.arange(60) % 2
    partitions = [_strong_variable(3, name='a'), _strong_variable(5, name='b'), _strong_variable(2, ordinal=False, name='c')]
    problem = evf.FitRuleBase(X, y, 4, 3, 2, linguistic_variables=partitions, ordinal_conditions=True)
    assert problem.n_term_codes == [5, 11, 2]  # a variable with 2 terms has nothing to add
    assert problem.n_lv_possible == [3, 5, 2]
    assert int(problem.xu[12:24].max()) == 10
    seen = {0: set(), 1: set(), 2: set()}
    for term_gene in range(-1, 11):
        for feature in range(3):
            code = remap_term(term_gene, problem.n_term_codes[feature], 11)
            if term_gene >= 0:
                seen[feature].add(code)
            assert code == -1 or 0 <= code < problem.n_term_codes[feature]
    assert [len(seen[f]) for f in range(3)] == [5, 11, 2]  # every code is reachable


def test_rule_base_encodes_back_to_the_same_rules():
    X, y = load_iris(return_X_y=True)
    partitions = [_strong_variable(n, name=str(i)) for i, n in enumerate((3, 4, 5, 3))]
    problem = evf.FitRuleBase(X / X.max(0), y, 3, 4, 3, linguistic_variables=partitions, ordinal_conditions=True)
    codes = [[problem.lvs[0].condition_code('at_least', 1), -1, problem.lvs[2].condition_code('at_most', 3), 2],
             [-1, problem.lvs[1].condition_code('at_most', 2), 4, -1]]
    base = [rules.RuleBaseT1(problem.lvs, [rules.RuleSimple(c) for c in codes[:1]]),
            rules.RuleBaseT1(problem.lvs, [rules.RuleSimple(c) for c in codes[1:]]),
            rules.RuleBaseT1(problem.lvs, [])]
    master = rules.MasterRuleBase(base, ['a', 'b', 'c'])
    gene = problem.encode_rulebase(master, optimize_lv=False)
    rebuilt = problem._construct_ruleBase(gene, fs.FUZZY_SETS.t1)
    assert [r.antecedents for r in rebuilt.get_rules()] == codes


def test_equal_condition_sets_are_duplicate_rules():
    problem = _problem(fixed=True, nRules=4, nAnts=2)
    variable = problem.lvs[0]
    code = variable.condition_code('at_least', 1)
    assert variable.condition_terms(code) == [1, 2, 3]
    gene = np.full(problem.n_var, -1)
    gene[:8] = [0, 1, 0, 1, 0, 1, 0, 1]  # feature 0 then 1 in every rule
    term_start = 8
    n_codes = problem.n_term_codes[0]
    gene[term_start:term_start + 8] = [code, 2, code, 2, code, 3, code, 2]  # rules 0, 1 and 3 repeat; 2 differs
    consequent = problem._consequent_pointer(problem.fuzzy_type)
    gene[consequent:consequent + 4] = [1, 1, 1, 0]
    base = problem._construct_ruleBase(gene, problem.fuzzy_type)
    assert [len(b.get_rules()) for b in base.get_rulebases()] == [1, 2, 0]  # class 1: rule 0 == rule 1, rule 2 differs
    assert problem._array_score(gene) == _object_score(problem, gene)
    for class_base in base.get_rulebases():  # rules are identified by their codes, which are canonical
        keys = [rule._identity() for rule in class_base.get_rules()]
        assert len(set(keys)) == len(keys)
    assert n_codes == 8


def test_random_sampling_crossover_and_mutation_stay_in_bounds_and_reach_ordinal_codes(iris):
    X, y = iris
    clf = evf.BaseFuzzyRulesClassifier(nRules=12, nAnts=3, linguistic_variables=utils.construct_partitions(X, n_partitions=4),
                                       ordinal_conditions=True, n_gen=8, pop_size=40, random_state=2, patience=None)
    clf.fit(X, y)
    gene = np.asarray(clf.optimization_result_['X'])
    assert gene.min() >= -1 and gene.max() <= 7
    term_genes = gene[36:72]
    assert term_genes.max() <= 7
    # the initial random sampling draws from the whole alphabet
    problem = _problem(fixed=True)
    assert (_genes(problem, 200)[:, 30:60] >= 4).any()
    from pymoo.core.population import Population
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PolynomialMutation
    from pymoo.operators.repair.rounding import RoundingRepair
    from ex_fuzzy._problem import as_pymoo_problem
    pymoo_problem = as_pymoo_problem(problem)
    population = Population.new(X=_genes(problem, 20))
    pairs = np.arange(20).reshape(10, 2)
    children = SBX(prob=1.0, eta=3.0, repair=RoundingRepair()).do(pymoo_problem, population, pairs).get('X')
    mutants = PolynomialMutation(prob=1.0, eta=7.0, repair=RoundingRepair()).do(pymoo_problem, population).get('X')
    for values in (children, mutants):
        assert (values >= problem.xl).all() and (values <= problem.xu).all()
        assert (values[:, 30:60] >= 4).any()  # offspring and mutants also land on ordinal codes


# ---------------------------------------------------------------------------
# Estimators.

def _rules_use_ordinal(rule_base):
    return any(c.operator != 'is' for r in rule_base.get_rules() for c in r.conditions(rule_base.antecedents))


@pytest.mark.parametrize('partitions', ['optimized', 'fixed'])
def test_winning_rule_classifier(iris, partitions):
    X, y = iris
    options = dict(linguistic_variables=utils.construct_partitions(X, n_partitions=4)) if partitions == 'fixed' else {}
    clf = evf.BaseFuzzyRulesClassifier(nRules=12, nAnts=3, ordinal_conditions=True, n_gen=25, pop_size=40,
                                       random_state=1, patience=None, **options)
    clf.fit(X, y)
    assert (clf.predict(X) == y).mean() > 0.85
    text = clf.print_rules(True)
    assert _rules_use_ordinal(clf.rule_base) == (' IS at least ' in text or ' IS at most ' in text)
    assert clf.get_params()['ordinal_conditions'] is True
    # every rule's conditions are valid codes of its variables
    for rule in clf.rule_base.get_rules():
        for condition, code in zip(rule.conditions(clf.rule_base.antecedents), [c for c in rule.antecedents if c != -1]):
            assert condition.code == code


def test_a_fit_with_ordinal_conditions_uses_them_and_the_option_is_cloneable(iris):
    from sklearn.base import clone
    X, y = iris
    clf = evf.BaseFuzzyRulesClassifier(nRules=15, nAnts=2, ordinal_conditions=True, n_gen=30, pop_size=50,
                                       random_state=0, patience=None)
    assert clone(clf).ordinal_conditions is True
    clf.fit(X, y)
    assert _rules_use_ordinal(clf.rule_base)
    assert ' IS at least ' in clf.print_rules(True) or ' IS at most ' in clf.print_rules(True)


def test_mixture_classifier_with_mixture_search(iris):
    X, y = iris
    clf = evf.BaseFuzzyRulesClassifier(
        nRules=10, nAnts=2, linguistic_variables=utils.construct_partitions(X, n_partitions=4),
        inference='mixture', mixture_search=True, mixture_non_negative=True, ordinal_conditions=True,
        n_gen=6, pop_size=16, random_state=3, patience=None)
    clf.fit(X, y)
    assert (clf.predict(X) == y).mean() > 0.7
    text = clf.print_mixture(True)
    assert text.count('Rule ') == len(clf.rule_base.get_rules())
    assert clf.mixture_scores(X).shape == (150, 3)
    # the mixture text names the operators too
    problem_codes = {c for r in clf.rule_base.get_rules() for c in r.antecedents if c >= 4}
    assert ('at least' in text or 'at most' in text) == bool(problem_codes)
    restored = pickle.loads(pickle.dumps(clf))
    np.testing.assert_array_equal(restored.predict(X), clf.predict(X))


def test_type_2_classifier(iris):
    X, y = iris
    clf = evf.BaseFuzzyRulesClassifier(nRules=8, nAnts=2, fuzzy_type=fs.FUZZY_SETS.t2, ordinal_conditions=True,
                                       n_gen=6, pop_size=16, random_state=1, patience=None)
    clf.fit(X, y)
    assert (clf.predict(X) == y).mean() > 0.5


def test_incompatible_options_are_refused(iris):
    X, y = iris
    with pytest.raises(ValueError, match='candidate_rules'):
        template = evf.BaseFuzzyRulesClassifier(nRules=4, nAnts=2, n_gen=2, pop_size=8, patience=None).fit(X, y)
        evf.BaseFuzzyRulesClassifier(nRules=4, nAnts=2, ordinal_conditions=True).fit(
            X, y, candidate_rules=template.rule_base)
    with pytest.raises(ValueError, match='general Type-2'):
        evf.BaseFuzzyRulesClassifier(nRules=4, nAnts=2, ordinal_conditions=True,
                                     linguistic_variables=utils.construct_partitions(X, fs.FUZZY_SETS.gt2)).fit(X, y)


def test_categorical_variables_keep_single_terms():
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.random(200), rng.integers(0, 3, 200), rng.random(200)])
    y = (X[:, 0] + (X[:, 1] == 1) > 0.9).astype(int)
    problem = evf.FitRuleBase(X, y, 6, 3, 2, n_linguistic_variables=3, categorical_mask=np.array([0, 1, 0]),
                              ordinal_conditions=True)
    assert problem.n_term_codes == [5, 3, 5]
    gene = _genes(problem, 5)[0]
    antecedents = problem._decode_antecedents(gene, fs.FUZZY_SETS.t1)
    assert [v.ordinal for v in antecedents] == [True, False, True]


def _regression_data():
    X, y = load_diabetes(return_X_y=True)
    return X, y


@pytest.mark.parametrize('options', [{}, dict(consequent_type='fuzzy'), dict(rule_mode='sufficient', unique_antecedents=True)])
def test_regression_with_ordinal_conditions(options):
    X, y = _regression_data()
    reg = efr.BaseFuzzyRulesRegressor(nRules=12, nAnts=3, n_linguistic_variables=4, ordinal_conditions=True, **options)
    reg.fit(X, y, n_gen=25, pop_size=40, random_state=3)
    assert reg.score(X, y) > 0.1
    text = reg.print_rules(True)
    for rule in reg.get_rulebase().get_rules():
        assert all(c.operator in fs.CONDITION_OPERATORS for c in rule.conditions(reg.rule_base.antecedents))
    assert ('at least' in text or 'at most' in text) == any(
        c.operator != 'is' for r in reg.get_rulebase().get_rules() for c in r.conditions(reg.rule_base.antecedents))
    restored = pickle.loads(pickle.dumps(reg))
    np.testing.assert_array_equal(restored.predict(X), reg.predict(X))
    assert restored.print_rules(True) == text


def test_regression_problem_scores_ordinal_rule_bases_consistently():
    X, y = _regression_data()
    partitions = utils.construct_partitions(X, n_partitions=4)
    problem = efr.FitRuleBaseRegression(X, y, 8, 3, partitions, ordinal_conditions=True)
    assert problem.n_lv.tolist() == [8] * 10 and int(problem.xu[24:48].max()) == 7
    genes = np.random.default_rng(1).integers(problem.xl.astype(int), problem.xu.astype(int) + 1, size=(15, problem.n_var))
    for gene in genes:
        rule_base = problem._construct_ruleBase(gene)
        fast = problem._fast_predict(gene)
        np.testing.assert_allclose(rule_base.predict(X), fast, rtol=1e-9, atol=1e-9)
    with pytest.raises(ValueError, match='invalid term index'):
        efr.RuleBaseT1Regression(partitions, [rules.RuleSimple([8] + [-1] * 9)], np.array([1.0]))
    ok = efr.RuleBaseT1Regression(fs.enable_ordinal_conditions(partitions), [rules.RuleSimple([7] + [-1] * 9)], np.array([1.0]))
    assert 'IS at most' in ok.print_rules(True)


def test_regression_ordinal_off_keeps_the_problem_unchanged():
    X, y = _regression_data()
    partitions = utils.construct_partitions(X, n_partitions=4)
    off = efr.FitRuleBaseRegression(X, y, 8, 3, partitions)
    assert off.n_lv.tolist() == [4] * 10 and off.lvs is partitions and int(off.xu[24:48].max()) == 3
    with pytest.raises(ValueError):
        efr.BaseFuzzyRulesRegressor(ordinal_conditions='yes').fit(X, y, n_gen=2, pop_size=4)


# ---------------------------------------------------------------------------
# Reading, printing, saving and loading.

def _hand_made_rule_base():
    a, b = _strong_variable(4, name='a'), _strong_variable(3, name='b')
    first = rules.RuleSimple([a.condition_code('at_least', 2), b.condition_code('at_most', 1)])
    second = rules.RuleSimple([1, b.condition_code('at_least', 1)])
    third = rules.RuleSimple([a.condition_code('at_most', 1), -1])
    for rule, score in ((first, 0.5), (second, 0.25), (third, 0.125)):
        rule.score, rule.accuracy = score, 0.75
    base = [rules.RuleBaseT1([a, b], [first, second]), rules.RuleBaseT1([a, b], [third])]
    master = rules.MasterRuleBase(base, ['low', 'high'])
    return master, (a, b)


def test_printing_names_the_operator_and_the_term():
    master, _ = _hand_made_rule_base()
    text = master.print_rules(True)
    assert 'IF a IS at least T2 AND b IS at most T1 WITH DS 0.5, ACC 0.75' in text
    assert 'IF a IS T1 AND b IS at least T1 WITH' in text
    assert 'IF a IS at most T1 WITH' in text
    assert 'IS at least T2' in rule_mixture.antecedent_text(master.get_rules()[0], master.antecedents)


def test_conditions_are_readable_without_parsing_text():
    master, (a, b) = _hand_made_rule_base()
    first = master.get_rules()[0]
    conditions = first.conditions(master.antecedents)
    assert [(c.variable, c.operator, c.term, c.term_index, c.terms) for c in conditions] == [
        ('a', 'at_least', 'T2', 2, ['T2', 'T3']), ('b', 'at_most', 'T1', 1, ['T0', 'T1'])]
    assert conditions[0].feature == 0 and conditions[1].feature == 1
    assert conditions[0].code == first.antecedents[0]
    assert rules.rule_conditions(master.get_rules()[1], master.antecedents)[0].operator == 'is'
    assert [c.variable for c in master.get_rules()[2].conditions(master.antecedents)] == ['a']
    # without ordinal conditions the same API gives 'is' only
    plain = rules.RuleBaseT1([_strong_variable(3, False, 'p')], [rules.RuleSimple([2])])
    assert plain.rules[0].conditions(plain.antecedents)[0][2:5] == ('is', 'T2', 2)


def test_rule_text_round_trips_and_the_loader_enables_the_conditions():
    master, (a, b) = _hand_made_rule_base()
    text = master.print_rules(True)
    plain = [_strong_variable(4, False, 'a'), _strong_variable(3, False, 'b')]
    loaded = persistence.load_fuzzy_rules(text, plain)
    assert [r.antecedents for r in loaded.get_rules()] == [r.antecedents for r in master.get_rules()]
    assert [v.ordinal for v in loaded.antecedents] == [True, True]
    assert not any(v.ordinal for v in plain)  # the given list and variables are untouched
    X = np.random.default_rng(0).random((30, 2))
    np.testing.assert_array_equal(loaded.compute_firing_strengths(X), master.compute_firing_strengths(X))
    assert loaded.print_rules(True) == text
    with pytest.raises(ValueError):
        persistence.load_fuzzy_rules('IF a IS at least T0 WITH DS 0.1', plain)  # vacuous: not offered
    with pytest.raises(ValueError):
        persistence.load_fuzzy_rules('IF a IS at least Nope WITH DS 0.1', plain)


def test_variable_text_round_trips_the_flag_and_is_unchanged_without_it():
    on, off = _strong_variable(4, True, 'on'), _strong_variable(4, False, 'off')
    printed = persistence.save_fuzzy_variables([on, off])
    assert printed.count('Ordinal conditions') == 1
    loaded = persistence.load_fuzzy_variables(printed)
    assert [v.ordinal for v in loaded] == [True, False]
    assert persistence.save_fuzzy_variables(loaded) == printed
    assert persistence.print_fuzzy_variable(off) == persistence.print_fuzzy_variable(_strong_variable(4, False, 'off'))
    assert 'Ordinal' not in persistence.print_fuzzy_variable(off)


def test_fitted_models_round_trip_through_text_and_pickle(iris):
    X, y = iris
    clf = evf.BaseFuzzyRulesClassifier(nRules=15, nAnts=2, ordinal_conditions=True, n_gen=30, pop_size=50,
                                       random_state=0, patience=None).fit(X, y)
    assert _rules_use_ordinal(clf.rule_base)
    variables_text = persistence.save_fuzzy_variables(clf.rule_base.antecedents)
    loaded = persistence.load_fuzzy_rules(clf.rule_base.print_rules(True), persistence.load_fuzzy_variables(variables_text))
    assert [r.antecedents for r in loaded.get_rules()] == [r.antecedents for r in clf.rule_base.get_rules()]
    np.testing.assert_array_equal(loaded.compute_firing_strengths(X), clf.rule_base.compute_firing_strengths(X))
    restored = pickle.loads(pickle.dumps(clf))
    np.testing.assert_array_equal(restored.predict(X), clf.predict(X))
    assert restored.print_rules(True) == clf.print_rules(True)
    copy = clf.rule_base.copy()
    np.testing.assert_array_equal(copy.compute_firing_strengths(X), clf.rule_base.compute_firing_strengths(X))


def test_pattern_stability_keys_name_the_condition_and_rebuild_the_rules():
    master, _ = _hand_made_rule_base()
    keys = ps.rule_keys(master)
    assert (('a', 'at least T2'), ('b', 'at most T1')) in [k[0] for k in keys]
    rebuilt = ps.consensus_rule_base(keys, master)
    assert sorted(r.antecedents for r in rebuilt.get_rules()) == sorted(r.antecedents for r in master.get_rules())


def test_rule_matrix_and_latex_helpers_handle_the_extra_codes(capsys):
    master, _ = _hand_made_rule_base()
    frame = vis_rules.matrix_rule_base_form(master.get_rulebases()[0])
    assert frame.shape == (2, 2) and frame.iloc[0, 0] == master.get_rules()[0].antecedents[0]
    latex = vis_rules.rules_to_latex(master)
    assert '$\\geq$ T2' in latex and '$\\leq$ T1' in latex
    capsys.readouterr()


def test_explainable_predict_reports_the_winning_ordinal_rule(iris):
    X, y = iris
    clf = evf.BaseFuzzyRulesClassifier(nRules=10, nAnts=2, ordinal_conditions=True, n_gen=10, pop_size=24,
                                       random_state=0, patience=None).fit(X, y)
    explained = clf.explainable_predict(X)
    winners = np.asarray(explained.winning_rule)
    assert winners.min() >= -1 and winners.max() < len(clf.rule_base.get_rules())


def test_p_value_tools_read_the_variables_the_rules_refer_to(iris):
    X, y = iris
    clf = evf.BaseFuzzyRulesClassifier(
        nRules=12, nAnts=2, linguistic_variables=utils.construct_partitions(X, n_partitions=4), ordinal_conditions=True,
        n_gen=20, pop_size=40, random_state=0, patience=None).fit(X, y, p_value_compute=True, bootstrap_size=5)
    assert _rules_use_ordinal(clf.rule_base)
    assert all(v.ordinal for v in clf.lvs)
    assert hasattr(clf.rule_base.get_rules()[0], 'boot_p_value')


def test_pattern_stabilizer_reports_and_charts_name_the_conditions(iris, capsys):
    import matplotlib
    matplotlib.use('Agg')
    X, y = iris
    variables = fs.enable_ordinal_conditions(utils.construct_partitions(X, n_partitions=4))
    stabilizer = ps.pattern_stabilizer(X, y, nRules=15, nAnts=2, linguistic_variables=variables)
    stabilizer.stability_report(n=3, n_gen=25, pop_size=40)
    printed = capsys.readouterr().out
    assert 'Pattern' in printed
    used = {c.operator for rb in stabilizer.rule_bases for r in rb.get_rules() for c in r.conditions(rb.antecedents)}
    assert used - {'is'}  # this seeded run uses ordinal conditions
    assert 'IS at least' in printed or 'IS at most' in printed
    assert 'at least 2 appears' in printed or 'at most 2 appears' in printed
    stabilizer.pie_chart_class(0)
    stabilizer.pie_chart_var(2)
    stabilizer.pie_chart_basic(2, 1)
    summary = stabilizer.rule_set_stability(n=3, n_gen=25, pop_size=40, pi=0.5)
    assert summary['frequencies']
    import matplotlib.pyplot as plt
    plt.close('all')
