"""Missing values are rejected, ds_mode names are accepted, and search settings live in the constructor."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import load_iris

from ex_fuzzy import evolutionary_fit as evf
from ex_fuzzy import fuzzy_sets as fs
from ex_fuzzy import rules
from ex_fuzzy import utils
from ex_fuzzy.conformal import ConformalFuzzyClassifier, evaluate_conformal_coverage

NAMES = np.array(['setosa', 'versicolor', 'virginica'])


@pytest.fixture(scope='module')
def iris():
    return load_iris(return_X_y=True)


@pytest.fixture(scope='module')
def fitted(iris):
    X, y = iris
    model = evf.BaseFuzzyRulesClassifier(nRules=8, nAnts=3, n_gen=3, pop_size=10, random_state=0,
                                         linguistic_variables=utils.construct_partitions(X, fs.FUZZY_SETS.t1))
    return model.fit(X, y)


def test_missing_values_are_rejected_at_fit_and_predict(iris, fitted):
    X, y = iris
    with_nan = X.copy()
    with_nan[3, 1] = np.nan
    with_nan[7, 3] = np.inf
    with pytest.raises(ValueError, match=r'columns \[1, 3\]'):
        evf.BaseFuzzyRulesClassifier(nRules=4, nAnts=2).fit(with_nan, y, n_gen=1, pop_size=4)
    for method in (fitted.predict, fitted.predict_proba, fitted.predict_proba_rules,
                   fitted.predict_membership_class, fitted.explainable_predict):
        with pytest.raises(ValueError, match='missing or infinite'):
            method(with_nan)
    # Frames with categorical text columns are complete data.
    frame = pd.DataFrame({'size': np.linspace(0, 1, 30), 'colour': ['red', 'green', 'blue'] * 10})
    labels = (frame['size'] > 0.5).astype(int).to_numpy()
    model = evf.BaseFuzzyRulesClassifier(nRules=4, nAnts=2, n_gen=1, pop_size=4).fit(frame, labels)
    assert len(model.predict(frame)) == len(frame)
    frame.loc[2, 'colour'] = None
    with pytest.raises(ValueError, match=r'columns \[1\]'):
        model.predict(frame)


def test_search_settings_live_in_the_constructor(iris):
    X, y = iris
    partitions = utils.construct_partitions(X, fs.FUZZY_SETS.t1)
    configured = evf.BaseFuzzyRulesClassifier(nRules=6, nAnts=2, linguistic_variables=partitions,
                                              n_gen=3, pop_size=10, random_state=0, patience=None)
    params = configured.get_params()
    assert params['n_gen'] == 3 and params['pop_size'] == 10 and params['random_state'] == 0
    assert params['patience'] is None and clone(configured).get_params()['n_gen'] == 3

    configured.fit(X, y)
    assert configured.n_generations_run_ == 3
    explicit = evf.BaseFuzzyRulesClassifier(nRules=6, nAnts=2, linguistic_variables=partitions)
    explicit.fit(X, y, n_gen=3, pop_size=10, random_state=0, patience=None)
    np.testing.assert_array_equal(configured.predict(X), explicit.predict(X))

    # A fit argument overrides the constructor for that call only.
    configured.fit(X, y, n_gen=2)
    assert configured.n_generations_run_ == 2 and configured.n_gen == 3


@pytest.mark.parametrize('name,code', [('dominance', 0), ('unweighted', 1), ('optimized', 2), ('OPTIMIZED', 2)])
def test_ds_mode_accepts_names(name, code):
    assert rules.resolve_ds_mode(name) == code
    assert rules.resolve_ds_mode(code) == code
    assert rules.resolve_ds_mode(np.int64(code)) == code


def test_ds_mode_rejects_unknown_values(iris):
    X, y = iris
    for value in ('weights', 3, -1, True, 1.5):
        with pytest.raises(ValueError, match='Unknown ds_mode'):
            evf.BaseFuzzyRulesClassifier(ds_mode=value)
        with pytest.raises(ValueError, match='Unknown ds_mode'):
            rules.MasterRuleBase([rules.RuleBaseT1(utils.construct_partitions(X), [])], ds_mode=value)


def test_named_ds_mode_fits_like_its_code(iris):
    X, y = iris
    partitions = utils.construct_partitions(X, fs.FUZZY_SETS.t1)
    named = evf.BaseFuzzyRulesClassifier(nRules=6, nAnts=2, linguistic_variables=partitions, ds_mode='optimized',
                                         n_gen=3, pop_size=10, random_state=0).fit(X, y)
    coded = evf.BaseFuzzyRulesClassifier(nRules=6, nAnts=2, linguistic_variables=partitions, ds_mode=2,
                                         n_gen=3, pop_size=10, random_state=0).fit(X, y)
    assert named.get_params()['ds_mode'] == 'optimized' and named.rule_base.ds_mode == 2
    np.testing.assert_array_equal(named.predict(X), coded.predict(X))


def test_explainable_predict_is_a_named_tuple(iris, fitted):
    X, _ = iris
    explained = fitted.explainable_predict(X[:5])
    assert isinstance(explained, rules.ExplainedPrediction)
    prediction, winning_rule, association, confidence = explained
    np.testing.assert_array_equal(explained.prediction, prediction)
    np.testing.assert_array_equal(explained.winning_rule, winning_rule)
    assert explained.association_degree.shape == (5, 1) and explained.confidence_interval.shape[0] == 5
    np.testing.assert_array_equal(prediction, fitted.predict(X[:5]))


def test_conformal_sets_use_the_fitted_labels(iris):
    X, y = iris
    labels = NAMES[y]
    conformal = ConformalFuzzyClassifier(nRules=8, nAnts=3, linguistic_variables=utils.construct_partitions(X, fs.FUZZY_SETS.t1))
    conformal.fit(X, labels, cal_size=0.3, n_gen=3, pop_size=10, random_state=0)

    sets = conformal.predict_set(X, alpha=0.1)
    assert all(prediction_set <= set(NAMES) for prediction_set in sets)
    # The same search on index labels gives the same sets, mapped through the labels.
    indexed = ConformalFuzzyClassifier(nRules=8, nAnts=3, linguistic_variables=utils.construct_partitions(X, fs.FUZZY_SETS.t1))
    indexed.fit(X, y, cal_size=0.3, n_gen=3, pop_size=10, random_state=0)
    assert sets == [{NAMES[index] for index in prediction_set} for prediction_set in indexed.predict_set(X, alpha=0.1)]
    explained = conformal.predict_set_with_rules(X[:3], alpha=0.1)
    assert set(explained[0]['class_p_values']) == set(NAMES)
    assert all(contribution['class'] in NAMES for result in explained for contribution in result['rule_contributions'])
    assert set(conformal.get_calibration_info()['samples_per_class']) <= set(NAMES)

    metrics = evaluate_conformal_coverage(conformal, X, labels, alpha=0.1)
    assert metrics['coverage'] >= 0.75
    assert set(metrics['coverage_by_class']) == set(NAMES)
    with pytest.raises(ValueError, match='not among the fitted classes'):
        conformal.calibrate(X[:10], np.array(['iris'] * 10))


def test_conformal_sets_with_shifted_integer_labels(iris):
    X, y = iris
    labels = y + 5
    conformal = ConformalFuzzyClassifier(nRules=8, nAnts=3, linguistic_variables=utils.construct_partitions(X, fs.FUZZY_SETS.t1))
    conformal.fit(X, labels, cal_size=0.3, n_gen=3, pop_size=10, random_state=0)
    sets = conformal.predict_set(X, alpha=0.1)
    assert all(prediction_set <= {5, 6, 7} for prediction_set in sets)
    assert evaluate_conformal_coverage(conformal, X, labels, alpha=0.1)['coverage'] >= 0.75
