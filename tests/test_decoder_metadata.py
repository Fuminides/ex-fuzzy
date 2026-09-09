"""Fit-local normalization must preserve empirical decoder semantics."""
import numpy as np
import pytest
from sklearn.datasets import load_iris

import evolutionary_fit as evf
import fuzzy_sets as fs
import utils


@pytest.mark.parametrize('kind', [fs.FUZZY_SETS.t1, fs.FUZZY_SETS.t2])
def test_normalization_uses_empirical_bounds_and_new_problem_data(kind):
    X, y = load_iris(return_X_y=True)
    for data in (X, X * 3 + 50):
        problem = evf.FitRuleBase(data, y, 10, 3, 3, fuzzy_type=kind,
                                 domain=(np.full(4, -100.), np.full(4, 100.)))
        minimum, maximum, span = problem._normalization_domain()
        np.testing.assert_array_equal(minimum, np.min(data, axis=0))
        np.testing.assert_array_equal(maximum, np.max(data, axis=0))
        np.testing.assert_array_equal(span, np.max(data, axis=0) - np.min(data, axis=0))
        assert all(not a.flags.writeable for a in (minimum, maximum, span))
        gene = np.random.default_rng(7).integers(problem.xl.astype(int), problem.xu.astype(int) + 1)
        base = problem._construct_ruleBase(gene, kind)
        for i, variable in enumerate(base.antecedents):
            for term in variable.linguistic_variables:
                np.testing.assert_array_equal(term.domain, [minimum[i], maximum[i]])


def test_normalization_nan_and_categorical_semantics():
    # Isolate normalization from partition builders' separate NaN policies.
    problem = object.__new__(evf.FitRuleBase)
    problem.X = np.array([[np.nan, 3], [2, 3], [5, 3]])
    minimum, maximum, span = problem._normalization_domain()
    np.testing.assert_array_equal(minimum, [2, 3])
    np.testing.assert_array_equal(maximum, [5, 3])
    np.testing.assert_array_equal(span, [3, 0])
    categorical = object.__new__(evf.FitRuleBase)
    categorical.X = np.array([['a', 'x'], ['b', None], ['a', 'y']], dtype=object)
    minimum, maximum, span = categorical._normalization_domain()
    np.testing.assert_array_equal(minimum, [0, 0])
    np.testing.assert_array_equal(maximum, [2, 2])


@pytest.mark.parametrize('kind', [fs.FUZZY_SETS.t1, fs.FUZZY_SETS.t2])
def test_refit_on_same_shaped_data_preserves_partition_reuse(kind):
    X, y = load_iris(return_X_y=True)
    options = dict(nRules=8, nAnts=3, fuzzy_type=kind)
    budget = dict(n_gen=2, pop_size=10, random_state=7, patience=None)
    reused = evf.BaseFuzzyRulesClassifier(**options)
    reused.fit(X, y, **budget)
    # Existing fit behavior retains learned partitions on subsequent fits.
    # Compare to a fresh classifier supplied with those same partitions.
    partitions = reused.rule_base.antecedents
    changed = X * 2 + 30
    reused.fit(changed, y, **budget)
    fresh = evf.BaseFuzzyRulesClassifier(**options, linguistic_variables=partitions)
    fresh.fit(changed, y, **budget)
    np.testing.assert_array_equal(reused.optimization_result_['X'], fresh.optimization_result_['X'])
    np.testing.assert_array_equal(reused.predict(changed), fresh.predict(changed))
    np.testing.assert_array_equal(reused.rule_base.get_scores(), fresh.rule_base.get_scores())
    for left, right in zip(reused.rule_base.antecedents, fresh.rule_base.antecedents):
        for a, b in zip(left.linguistic_variables, right.linguistic_variables):
            np.testing.assert_array_equal(a.domain, b.domain)


@pytest.mark.parametrize('fixed', [False, True])
def test_candidate_decoding_does_not_rescan_data(monkeypatch, fixed):
    X, y = load_iris(return_X_y=True)
    problem = evf.FitRuleBase(X, y, 10, 3, 3,
                             linguistic_variables=utils.construct_partitions(X) if fixed else None)
    genes = np.random.default_rng(7).integers(problem.xl.astype(int), problem.xu.astype(int) + 1,
                                             size=(3, problem.n_var))

    def unexpected_scan(*args, **kwargs):
        pytest.fail('Candidate decoding rescanned training data')

    monkeypatch.setattr(np, 'nanmin', unexpected_scan)
    monkeypatch.setattr(np, 'nanmax', unexpected_scan)
    for gene in genes:
        problem._construct_ruleBase(gene, problem.fuzzy_type)
    assert hasattr(problem, '_normalization_domain_cache') is not fixed
