"""Tests for FuzzyRulesClassifier and its fuzzy association rule engine."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import load_iris, make_classification

import _fuzzy_association as association
import fuzzy_sets as fs
import utils
from classifiers import FuzzyRulesClassifier


@pytest.fixture(scope="module")
def iris():
    X, y = load_iris(return_X_y=True)
    frame = pd.DataFrame(X, columns=["sepal_len", "sepal_wid", "petal_len", "petal_wid"])
    labels = np.array(["setosa", "versicolor", "virginica"])[y]
    return frame, labels


@pytest.fixture(scope="module")
def fitted(iris):
    frame, labels = iris
    model = FuzzyRulesClassifier(max_features=3, feature_selection="global", nRules=30,
                                 n_gen=30, pop_size=30, random_state=0)
    assert model.fit(frame, labels) is model
    return model


def test_is_a_scikit_learn_classifier(fitted, iris):
    frame, labels = iris
    predictions = fitted.predict(frame)
    probabilities = fitted.predict_proba(frame)

    assert set(predictions) <= set(labels)
    assert probabilities.shape == (len(frame), 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert fitted.score(frame, labels) > 0.85
    assert clone(fitted).get_params() == fitted.get_params()
    assert np.array_equal(fitted.feature_names_in_, frame.columns.to_numpy())
    with pytest.raises(ValueError, match="features"):
        fitted.predict(frame.to_numpy()[:, :2])


def test_feature_cap_and_rule_budget_are_respected(fitted):
    assert len(fitted.selected_features_) == 3
    assert len(fitted.linguistic_variables_) == 3
    assert 1 <= fitted.n_rules_ <= fitted.nRules
    assert len(fitted.rule_base_.get_rules()) == fitted.n_rules_


def test_fit_is_reproducible_for_a_seed(iris):
    frame, labels = iris
    first = FuzzyRulesClassifier(max_features=3, n_gen=20, pop_size=20, random_state=3).fit(frame, labels)
    second = FuzzyRulesClassifier(max_features=3, n_gen=20, pop_size=20, random_state=3).fit(frame, labels)
    np.testing.assert_array_equal(first.predict_proba(frame), second.predict_proba(frame))
    np.testing.assert_array_equal(first.selected_features_, second.selected_features_)


def test_rules_export_to_ex_fuzzy_objects(fitted):
    text = fitted.print_rules(return_rules=True)
    assert "IF" in text and "petal" in text
    internal = fitted.internal_classifier()
    assert len(internal.rule_base.get_rules()) == fitted.n_rules_
    assert np.all(internal.rule_base.get_weights() > 0)


def test_legacy_fit_signature_still_works():
    X, y = make_classification(n_samples=120, n_features=6, n_informative=4, random_state=0)
    model = FuzzyRulesClassifier(nRules=10, nAnts=3, verbose=False, expansion_factor=2)
    assert model.fit(X, y, n_gen=5, pop_size=10) is model
    assert model.n_rules_ <= 10
    assert len(model.predict(X)) == len(y)
    with pytest.raises(TypeError, match="Unexpected fit arguments"):
        model.fit(X, y, candidate_rules=None)


def test_additive_and_sufficient_rule_modes_both_predict(iris):
    frame, labels = iris
    for rule_mode in ("additive", "sufficient"):
        model = FuzzyRulesClassifier(rule_mode=rule_mode, n_gen=10, pop_size=20, random_state=0)
        assert model.fit(frame, labels).score(frame, labels) > 0.8


def test_user_partitions_are_subset_with_the_selected_features(iris):
    frame, labels = iris
    partitions = utils.construct_partitions(frame, fs.FUZZY_SETS.t1, n_partitions=3)
    model = FuzzyRulesClassifier(linguistic_variables=partitions, max_features=2, n_gen=10,
                                 pop_size=20, random_state=0).fit(frame, labels)
    assert [partitions[i] for i in model.selected_features_] == model.linguistic_variables_
    with pytest.raises(ValueError, match="linguistic_variables"):
        FuzzyRulesClassifier(linguistic_variables=partitions[:2]).fit(frame, labels)


@pytest.mark.parametrize("kwargs, message", [
    ({"fuzzy_type": fs.FUZZY_SETS.t2}, "Type-1"),
    ({"rule_mode": "winner"}, "rule_mode"),
    ({"max_features": 0}, "max_features"),
    ({"n_linguistic_variables": 1}, "n_linguistic_variables"),
    ({"feature_selector": "chi2"}, "feature_selector"),
    ({"nAnts": 0}, "nAnts"),
])
def test_rejects_invalid_parameters(iris, kwargs, message):
    frame, labels = iris
    with pytest.raises(ValueError, match=message):
        FuzzyRulesClassifier(**kwargs).fit(frame, labels)


def _toy_memberships():
    rng = np.random.default_rng(0)
    X = rng.random((90, 3))
    y = np.repeat([0, 1, 2], 30)
    X[y == 1, 0] += 1.0
    X[y == 2, 1] += 1.0
    partitions = utils.construct_partitions(X, fs.FUZZY_SETS.t1, n_partitions=3)
    return association.membership_matrices(partitions, X), y


def test_every_class_keeps_candidates_even_when_thresholds_reject_it():
    memberships, y = _toy_memberships()
    pool = association.mine_candidates(memberships, y, 3, max_conditions=2,
                                       min_support=0.05, min_confidence=0.99)
    assert set(np.unique(pool["consequent"])) == {0, 1, 2}
    screened = association.prescreen(pool, y, 3, per_class=4)
    assert set(np.unique(screened["consequent"])) == {0, 1, 2}
    assert np.all(np.diff(screened["consequent"]) >= 0)
    assert np.bincount(screened["consequent"]).max() <= 4


def test_mined_firing_matches_recomputed_firing():
    memberships, y = _toy_memberships()
    pool = association.mine_candidates(memberships, y, 3, max_conditions=2)
    recomputed = association.rule_firing(memberships, pool["features"], pool["terms"], len(y))
    np.testing.assert_allclose(recomputed, pool["firing"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(pool["weight"], 2 * pool["confidence"] - 1)


def test_selection_respects_the_rule_cap_and_is_seeded():
    memberships, y = _toy_memberships()
    pool = association.prescreen(association.mine_candidates(memberships, y, 3, max_conditions=2),
                                 y, 3, per_class=10)
    first = association.select_rules(pool, y, 3, 0, max_rules=4, n_gen=15, pop_size=12, rng=5)
    second = association.select_rules(pool, y, 3, 0, max_rules=4, n_gen=15, pop_size=12, rng=5)
    assert first.sum() <= 4
    np.testing.assert_array_equal(first, second)


def test_scores_and_fallback():
    # rules x samples; the third sample's class-1 rule (0.4) beats class 0's best
    # weighted firing (0.3), so no arg-max tie decides the expected label.
    firing = np.array([[0.9, 0.0, 0.2], [0.5, 0.0, 0.6], [0.0, 0.0, 0.4]])
    weights = np.array([1.0, 0.5, 1.0])
    consequents = np.array([0, 0, 1])
    sufficient = association.class_scores(firing, weights, consequents, 3, "sufficient")
    additive = association.class_scores(firing, weights, consequents, 3, "additive")
    np.testing.assert_allclose(sufficient[:, 0], [0.9, 0.0, 0.3])
    np.testing.assert_allclose(additive[:, 0], [1.15, 0.0, 0.5])
    np.testing.assert_array_equal(association.predict_from_scores(sufficient, default_class=2), [0, 2, 1])


def test_per_class_feature_selection_mines_each_class_on_its_own_features():
    X, y = make_classification(n_samples=300, n_features=20, n_informative=8, n_classes=4,
                               n_clusters_per_class=1, random_state=0)
    model = FuzzyRulesClassifier(max_features=4, feature_selection="per_class", n_gen=15,
                                 pop_size=20, random_state=0).fit(X, y)

    assert len(model.class_features_) == 4
    assert all(len(features) == 4 for features in model.class_features_)
    np.testing.assert_array_equal(model.selected_features_,
                                  np.unique(np.concatenate(model.class_features_)))
    # Each selected rule only uses features its own class was mined on.
    for features, consequent in zip(model._rules["features"], model._rules["consequent"]):
        original = {int(model.selected_features_[f]) for f in features}
        assert original <= set(model.class_features_[consequent].tolist())

    again = FuzzyRulesClassifier(max_features=4, feature_selection="per_class", n_gen=15,
                                 pop_size=20, random_state=0).fit(X, y)
    np.testing.assert_array_equal(model.predict_proba(X), again.predict_proba(X))


def test_global_feature_selection_shares_features_across_classes(fitted):
    assert all(np.array_equal(features, fitted.selected_features_) for features in fitted.class_features_)


def test_rejects_unknown_feature_selection(iris):
    frame, labels = iris
    with pytest.raises(ValueError, match="feature_selection"):
        FuzzyRulesClassifier(feature_selection="local").fit(frame, labels)


def test_auto_feature_cap_scales_with_the_number_of_classes():
    few, few_y = make_classification(n_samples=240, n_features=24, n_informative=10, n_classes=3,
                                     n_clusters_per_class=1, random_state=1)
    many, many_y = make_classification(n_samples=420, n_features=24, n_informative=12, n_classes=7,
                                       n_clusters_per_class=1, random_state=1)
    small = FuzzyRulesClassifier(max_features="auto", feature_selection="global", n_gen=5,
                                 pop_size=10, random_state=0).fit(few, few_y)
    large = FuzzyRulesClassifier(max_features="auto", feature_selection="per_class", n_gen=5,
                                 pop_size=10, random_state=0).fit(many, many_y)
    assert small.max_features_ == 8 and len(small.selected_features_) == 8
    assert large.max_features_ == 16
    assert all(len(features) == 16 for features in large.class_features_)


def test_auto_depth_allows_longer_rules_on_few_features():
    X, y = make_classification(n_samples=240, n_features=5, n_informative=4, n_redundant=0,
                               n_classes=3, n_clusters_per_class=1, random_state=2)
    shallow = FuzzyRulesClassifier(nAnts="auto", n_gen=5, pop_size=10, random_state=0).fit(X, y)
    assert shallow.n_conditions_ == [4, 4, 4]
    assert max(len(features) for features in shallow._rules["features"]) <= 4
    wide, wide_y = make_classification(n_samples=240, n_features=12, n_informative=6, random_state=2)
    deep = FuzzyRulesClassifier(nAnts="auto", max_features=8, n_gen=5, pop_size=10, random_state=0).fit(wide, wide_y)
    assert deep.n_conditions_ == [3, 3]
    with pytest.raises(ValueError, match="nAnts"):
        FuzzyRulesClassifier(nAnts="deep").fit(X, y)
    with pytest.raises(ValueError, match="max_features"):
        FuzzyRulesClassifier(max_features="many").fit(X, y)


def test_defaults_are_the_development_selected_configuration():
    params = FuzzyRulesClassifier().get_params()
    assert params["max_features"] == 8
    assert params["feature_selection"] == "per_class"
    assert params["n_linguistic_variables"] == "auto"
    assert params["nAnts"] == 3
    assert params["nRules"] is None
    assert params["rule_mode"] == "additive"


def _assert_same_pool(left, right):
    # Same rules in the same order with identical firing; the derived statistics may differ
    # only by BLAS round-off, because the two paths score matrices of different shapes.
    assert left["features"] == right["features"]
    assert left["terms"] == right["terms"]
    np.testing.assert_array_equal(left["consequent"], right["consequent"])
    np.testing.assert_array_equal(left["firing"], right["firing"])
    for key in ("weight", "confidence", "support", "quality"):
        np.testing.assert_allclose(left[key], right[key], rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("depth, n_terms, min_support", [(3, 3, 0.05), (4, 3, 0.05), (4, 5, 0.1), (3, 4, 0.0)])
def test_support_pruning_keeps_exactly_the_exhaustive_candidates(depth, n_terms, min_support):
    X, y = make_classification(n_samples=260, n_features=6, n_informative=4, n_redundant=1,
                               n_classes=3, n_clusters_per_class=1, random_state=4)
    partitions = utils.construct_partitions(X, fs.FUZZY_SETS.t1, n_partitions=n_terms)
    memberships = association.membership_matrices(partitions, X)
    pruned = association.mine_candidates(memberships, y, 3, max_conditions=depth,
                                         min_support=min_support, prune=True)
    exhaustive = association.mine_candidates(memberships, y, 3, max_conditions=depth,
                                             min_support=min_support, prune=False)
    _assert_same_pool(pruned, exhaustive)


def test_five_condition_rules_are_supported():
    X, y = make_classification(n_samples=300, n_features=10, n_informative=6, n_classes=3,
                               n_clusters_per_class=1, random_state=5)
    model = FuzzyRulesClassifier(nAnts=5, max_features=6, n_gen=10, pop_size=20, random_state=0).fit(X, y)
    assert model.n_rules_ >= 1
    assert max(len(features) for features in model._rules["features"]) <= 5
    assert model.score(X, y) > 0.5


@pytest.mark.parametrize('params, message', [
    (dict(fuzzy_type=fs.FUZZY_SETS.t2), 'Type-1'),
    (dict(rule_mode='voting'), 'rule_mode'),
    (dict(nAnts=0), 'nAnts'),
    (dict(nRules=0), 'nRules'),
    (dict(max_features='all'), 'max_features'),
    (dict(n_linguistic_variables=1), 'n_linguistic_variables'),
    (dict(rules_per_class=0), 'rules_per_class'),
    (dict(candidates_per_class=0), 'candidates_per_class'),
    (dict(feature_selection='both'), 'feature_selection'),
    (dict(feature_selector='chi2'), 'feature_selector'),
    (dict(pop_size=1), 'pop_size'),
])
def test_invalid_parameters_are_rejected(iris, params, message):
    frame, labels = iris
    with pytest.raises(ValueError, match=message):
        FuzzyRulesClassifier(**params).fit(frame, labels)


def test_invalid_training_data_is_rejected(iris):
    frame, labels = iris
    X = frame.to_numpy()
    model = FuzzyRulesClassifier(n_gen=1, pop_size=4)

    with pytest.raises(ValueError, match='two-dimensional'):
        model.fit(X[:, 0], labels)
    with pytest.raises(ValueError, match='same, non-zero'):
        model.fit(X[:10], labels)
    with pytest.raises(ValueError, match='NaN'):
        model.fit(np.where(X == X[0, 0], np.nan, X), labels)
    with pytest.raises(ValueError, match='two classes'):
        model.fit(X, np.zeros(len(X)))


def test_feature_selectors_term_count_and_label_shape(iris, capsys):
    frame, labels = iris
    X = frame.to_numpy()

    by_f_score = FuzzyRulesClassifier(max_features=2, feature_selector='f_classif', feature_selection='global',
                                      n_linguistic_variables=4, n_gen=3, pop_size=10, verbose=True, random_state=0)
    by_f_score.fit(X, labels.reshape(-1, 1))
    assert len(by_f_score.selected_features_) == 2
    assert len(by_f_score.linguistic_variables_[0]) == 4
    assert 'rules selected' in capsys.readouterr().out
    assert by_f_score.predict(X[0]).shape == (1,)

    # A callable scores the features; per class it sees the same variances, so one feature is kept.
    by_variance = FuzzyRulesClassifier(max_features=1, feature_selector=lambda values, target: values.var(0),
                                       n_gen=3, pop_size=10, random_state=0)
    by_variance.fit(X, labels)
    assert by_variance.selected_features_.tolist() == [int(np.argmax(X.var(0)))]


def test_partitions_that_never_fire_select_no_rules(iris, capsys):
    frame, labels = iris
    far_away = [fs.fuzzyVariable(name, [fs.FS('far', [100, 101, 102, 103], [100, 103])]) for name in frame.columns]
    model = FuzzyRulesClassifier(linguistic_variables=far_away, max_features=2, n_gen=3, pop_size=10, random_state=0)
    model.fit(frame, labels)

    assert model.n_rules_ == 0 and model.rule_base_ is None
    assert set(model.predict(frame)) == {model.classes_[model.majority_class_]}
    np.testing.assert_allclose(model.predict_proba(frame), 1 / 3)
    assert model.print_rules(return_rules=True).startswith('No rules were selected')
    assert model.print_rules() is None
    assert 'No rules were selected' in capsys.readouterr().out
    with pytest.raises(ValueError, match='No rules were selected'):
        model.internal_classifier()
