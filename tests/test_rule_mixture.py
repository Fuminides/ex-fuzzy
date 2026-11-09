"""Mixture-of-rules inference (``inference='mixture'``): scores, parity of the default path, persistence, edges."""
import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from ex_fuzzy import evolutionary_fit as evf
from ex_fuzzy import fuzzy_sets as fs
from ex_fuzzy import rule_mixture as rm
from ex_fuzzy import rules
from ex_fuzzy import utils
from ex_fuzzy._fitness import _class_counts, _macro_f1, _penalized_objective

NAMES = np.array(['setosa', 'versicolor', 'virginica'])


@pytest.fixture(scope='module')
def iris():
    return load_iris(return_X_y=True)


def _fit(X, y, **kwargs):
    model = evf.BaseFuzzyRulesClassifier(nRules=10, nAnts=2, **kwargs)
    return model.fit(X, y, n_gen=8, pop_size=16, random_state=0)


def _softmax(scores):
    exponentials = np.exp(scores - scores.max(axis=1, keepdims=True))
    return exponentials / exponentials.sum(axis=1, keepdims=True)


@pytest.mark.parametrize('non_negative', [False, True])
def test_scores_are_the_weighted_sum_of_the_firing_strengths(iris, non_negative):
    X, y = iris
    model = _fit(X, y, inference='mixture', mixture_non_negative=non_negative)
    firing = model.rule_base.compute_firing_strengths(X)
    W, b = model.mixture_.coef_, model.mixture_.intercept_
    assert W.shape == (len(model.rule_base.get_rules()), 3) and b.shape == (3,)

    manual = np.zeros((len(X), 3))
    for r in range(W.shape[0]):
        for c in range(3):
            manual[:, c] += W[r, c] * firing[:, r]
    manual += b
    scores = model.mixture_scores(X)
    np.testing.assert_allclose(scores, manual, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(model.predict(X), np.argmax(manual, axis=1))
    np.testing.assert_allclose(model.predict_proba(X), _softmax(manual), atol=1e-12)
    np.testing.assert_allclose(model.mixture_contributions(X).sum(axis=1) + b, scores, atol=1e-12)
    if non_negative:
        assert W.min() >= 0
        # A constant added to a rule's weights changes nothing, so the optimum leaves a zero in each row.
        assert np.all(W.min(axis=1) == 0)
    else:
        # With the L2 penalty the weights of each rule add up to zero over the classes.
        np.testing.assert_allclose(W.sum(axis=1), 0, atol=1e-6)
    assert abs(b.mean()) < 1e-12


def test_unconstrained_weights_are_scikit_learns_logistic_regression(iris):
    X, y = iris
    model = _fit(X, y, inference='mixture', mixture_C=0.5)
    firing = rm.rule_firing(model.rule_base, X)
    reference = LogisticRegression(C=0.5, tol=1e-10, max_iter=10_000).fit(firing, y)
    np.testing.assert_allclose(model.predict_proba(X), reference.predict_proba(firing), atol=1e-4)
    np.testing.assert_array_equal(model.predict(X), reference.predict(firing))


def test_default_inference_is_unchanged(iris):
    X, y = iris
    default = _fit(X, y)
    explicit = _fit(X, y, inference='winning_rule')
    mixture = _fit(X, y, inference='mixture')
    assert default.inference == 'winning_rule' and default.mixture_ is None and explicit.mixture_ is None
    # The search, its pruning and the rules are those of the winning-rule fit.
    text = default.rule_base.print_rules(True, bootstrap_results=False)
    assert explicit.rule_base.print_rules(True, bootstrap_results=False) == text
    assert mixture.rule_base.print_rules(True, bootstrap_results=False) == text
    assert np.array_equal(default.performance, mixture.performance)
    np.testing.assert_array_equal(explicit.predict(X), default.predict(X))
    np.testing.assert_array_equal(explicit.predict_proba(X), default.predict_proba(X))
    np.testing.assert_array_equal(mixture.rule_base.winning_rule_predict(X), default.predict(X))


def test_fit_mixture_gives_an_existing_model_the_weights_of_a_mixture_fit(iris):
    X, y = iris
    model = _fit(X, y)
    winning = model.predict(X)
    scores = model.rule_base.get_scores().copy()
    with pytest.raises(ValueError, match="set it first"):
        model.fit_mixture(X, y)
    model.set_params(inference='mixture').fit_mixture(X, y)
    fitted = _fit(X, y, inference='mixture')
    np.testing.assert_array_equal(model.mixture_.coef_, fitted.mixture_.coef_)
    np.testing.assert_array_equal(model.mixture_.intercept_, fitted.mixture_.intercept_)
    # The rules and their dominance scores stay; switching back gives the winning-rule predictions again.
    np.testing.assert_array_equal(model.rule_base.get_scores(), scores)
    model.set_params(inference='winning_rule')
    np.testing.assert_array_equal(model.predict(X), winning)


def test_pickle_round_trip_keeps_the_weights(iris):
    X, y = iris
    model = _fit(X, NAMES[y], inference='mixture', mixture_non_negative=True)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.mixture_.coef_, model.mixture_.coef_)
    np.testing.assert_array_equal(restored.mixture_.intercept_, model.mixture_.intercept_)
    np.testing.assert_array_equal(restored.mixture_scores(X), model.mixture_scores(X))
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))
    assert set(model.predict(X)) <= set(NAMES)
    assert restored.get_params()['inference'] == 'mixture' and restored.mixture_non_negative


def test_a_classifier_pickled_before_the_option_existed_still_predicts(iris):
    X, y = iris
    model = _fit(X, y)
    expected = model.predict(X)
    for name in ('inference', 'mixture_C', 'mixture_non_negative', 'mixture_search', 'mixture_'):
        model.__dict__.pop(name, None)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict(X), expected)
    assert restored.get_params()['inference'] == 'winning_rule'
    restored.set_params(inference='mixture').fit_mixture(X, y)
    assert restored.mixture_.coef_.shape == (len(restored.rule_base.get_rules()), 3)


def _hand_built(X):
    """Two rules of class 0, one of class 1, none of class 2; rule 1 uses a term outside the data (unweighted)."""
    variables = utils.construct_partitions(X, fs.FUZZY_SETS.t1, n_partitions=3)
    empty = fs.FS('Far away', [5.0, 6.0, 7.0, 8.0], [0.0, 8.0])
    variables[1] = fs.fuzzyVariable('x1', [*variables[1].linguistic_variables, empty])
    rule_lists = [[rules.RuleSimple([0, -1], 0), rules.RuleSimple([-1, 3], 0)],
                  [rules.RuleSimple([1, -1], 1)], []]
    master = rules.MasterRuleBase([rules.RuleBaseT1(variables, rule_list) for rule_list in rule_lists], ds_mode=1)
    return evf.BaseFuzzyRulesClassifier(precomputed_rules=master, inference='mixture')


def test_a_rule_that_never_fires_gets_no_weight_and_a_class_without_rules_is_predicted():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(600, 2))
    y = np.digitize(X[:, 0], [1 / 3, 2 / 3])  # class 2 = large x0, for which no rule exists
    model = _hand_built(X).fit_mixture(X, y)
    firing = rm.rule_firing(model.rule_base, X)
    assert np.all(firing[:, 1] == 0)
    np.testing.assert_array_equal(model.mixture_.coef_[1], 0.0)
    prediction = model.predict(X)
    assert np.any(prediction == 2) and np.mean(prediction == y) > 0.8
    assert not np.any(model.rule_base.winning_rule_predict(X) == 2)
    # Without the silent rule the scores are the same.
    alone = rm.RuleMixture().fit(firing[:, [0, 2]], y, 3)
    np.testing.assert_allclose(alone.scores(firing[:, [0, 2]]), model.mixture_scores(X), atol=1e-6)
    text = model.print_mixture(return_text=True)
    assert 'x1 IS Far away; strongest class: none' in text and 'Bias:' in text


def test_samples_that_fire_no_rule_follow_allow_unknown():
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, size=(400, 2))
    y = np.digitize(X[:, 0], [1 / 3, 2 / 3])
    variables = utils.construct_partitions(X, fs.FUZZY_SETS.t1, n_partitions=3)
    rule_lists = [[rules.RuleSimple([0, -1], 0)], [rules.RuleSimple([1, -1], 1)], []]
    for allow_unknown in (False, True):
        master = rules.MasterRuleBase([rules.RuleBaseT1(variables, r) for r in rule_lists],
                                      allow_unknown=allow_unknown)
        model = evf.BaseFuzzyRulesClassifier(precomputed_rules=master, inference='mixture').fit_mixture(X, y)
        silent = ~np.any(rm.rule_firing(model.rule_base, X) > 0, axis=1)
        assert silent.any()
        expected = -1 if allow_unknown else int(np.argmax(model.mixture_.intercept_))
        assert np.all(model.predict(X)[silent] == expected)


def test_rules_with_the_same_antecedents_share_one_row_of_weights():
    rng = np.random.default_rng(2)
    X = rng.uniform(0, 1, size=(500, 2))
    y = np.digitize(X[:, 0], [0.5])
    variables = utils.construct_partitions(X, fs.FUZZY_SETS.t1, n_partitions=3)
    rule_lists = [[rules.RuleSimple([0, -1], 0)], [rules.RuleSimple([0, -1], 1), rules.RuleSimple([2, -1], 1)]]
    master = rules.MasterRuleBase([rules.RuleBaseT1(variables, r) for r in rule_lists])
    model = evf.BaseFuzzyRulesClassifier(precomputed_rules=master, inference='mixture').fit_mixture(X, y)
    np.testing.assert_array_equal(model.mixture_.duplicate_of_, [-1, 0, -1])
    np.testing.assert_array_equal(model.mixture_.coef_[1], 0.0)
    assert np.any(model.mixture_.coef_[0] != 0)
    assert 'same antecedents as rule 0' in model.print_mixture(return_text=True)


def test_changed_rules_need_new_weights(iris):
    X, y = iris
    model = _fit(X, y, inference='mixture')
    next(base for base in model.rule_base.rule_bases if base.rules).remove_rule(0)
    with pytest.raises(ValueError, match='call fit_mixture again'):
        model.predict(X)
    model.fit_mixture(X, y)
    assert len(model.predict(X)) == len(X)


def test_a_class_without_samples_cannot_be_fitted():
    with pytest.raises(ValueError, match='no training sample'):
        rm.RuleMixture().fit(np.ones((4, 1)), np.array([0, 0, 1, 1]), 3)


def test_mixture_search_scores_candidates_as_the_final_mixture(iris):
    X, y = iris
    model = _fit(X, y, inference='mixture', mixture_search=True, mixture_non_negative=True)
    prediction = model.mixture_.predict(rm.rule_firing(model.rule_base, X))
    survivors = model.rule_base.get_rules()
    covered = np.count_nonzero(np.bincount(model.rule_base.get_consequents(), minlength=3))
    conditions = sum(np.count_nonzero(np.asarray(rule.antecedents) != -1) for rule in survivors)
    objective = _penalized_objective(_macro_f1(*_class_counts(y, prediction, 3)), covered, conditions, 3,
                                     10 * 2, model.alpha_, model.beta_)
    assert float(np.ravel(model.performance)[0]) == pytest.approx(float(objective), abs=1e-12)
    assert model.mixture_.coef_.min() >= 0


@pytest.mark.parametrize('kwargs, message', [
    (dict(inference='voting'), 'Unknown inference'),
    (dict(inference='mixture', mixture_C=0), 'mixture_C'),
    (dict(mixture_search=True), "needs inference='mixture'"),
])
def test_invalid_options_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        evf.BaseFuzzyRulesClassifier(**kwargs)


def test_mixture_search_rejects_other_objectives(iris):
    X, y = iris
    for kwargs in (dict(algorithm='nsga2'), dict(batch_size=50)):
        model = evf.BaseFuzzyRulesClassifier(nRules=4, nAnts=2, inference='mixture', mixture_search=True, **kwargs)
        with pytest.raises(ValueError, match='mixture_search replaces the built-in objective'):
            model.fit(X, y, n_gen=2, pop_size=6)
    model = evf.BaseFuzzyRulesClassifier(nRules=4, nAnts=2, inference='mixture', mixture_search=True)
    model.customized_loss(lambda *args: 0.0)
    with pytest.raises(ValueError, match='a custom loss'):
        model.fit(X, y, n_gen=2, pop_size=6)


def test_explainable_predict_refuses_the_mixture_and_clone_keeps_its_options(iris):
    X, y = iris
    model = _fit(X, y, inference='mixture', mixture_C=0.3, mixture_non_negative=True)
    with pytest.raises(ValueError, match='mixture_contributions'):
        model.explainable_predict(X)
    copied = clone(model)
    assert copied.get_params()['mixture_C'] == 0.3 and copied.mixture_non_negative and copied.mixture_ is None
