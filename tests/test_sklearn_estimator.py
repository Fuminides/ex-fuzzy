"""BaseFuzzyRulesClassifier follows the scikit-learn estimator contract."""
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ex_fuzzy import evolutionary_fit as evf
from ex_fuzzy import fuzzy_sets as fs
from ex_fuzzy import utils

NAMES = np.array(['setosa', 'versicolor', 'virginica'])


@pytest.fixture(scope='module')
def iris():
    return load_iris(return_X_y=True)


def _fit(X, y, **kwargs):
    model = evf.BaseFuzzyRulesClassifier(nRules=6, nAnts=2, **kwargs)
    return model.fit(X, y, n_gen=3, pop_size=10, random_state=0)


def test_get_params_and_clone_round_trip(iris):
    X, _ = iris
    partitions = utils.construct_partitions(X, fs.FUZZY_SETS.t1)
    model = evf.BaseFuzzyRulesClassifier(
        nRules=7, nAnts=2, class_names=list(NAMES), linguistic_variables=partitions,
        n_class=3, ds_mode=2, n_linguistic_variables=5, backend='pymoo')

    params = model.get_params()
    assert params['class_names'] == list(NAMES)
    assert params['linguistic_variables'] is partitions
    assert params['n_linguistic_variables'] == 5
    assert params['n_class'] == 3 and params['nRules'] == 7 and params['ds_mode'] == 2
    assert params['backend'].name() == 'pymoo'
    # The derived attributes the library reads are unchanged.
    assert model.classes_names == list(NAMES) and model.lvs is partitions and model.nclasses_ == 3

    copied = clone(model)
    assert copied.get_params()['nRules'] == 7
    assert copied.classes_names == list(NAMES)
    assert copied.backend.name() == 'pymoo'
    assert copied.lvs is not partitions and len(copied.lvs) == len(partitions)
    assert 'BaseFuzzyRulesClassifier(' in repr(model)


def test_fit_returns_self_and_sets_fitted_attributes(iris):
    X, y = iris
    model = evf.BaseFuzzyRulesClassifier(nRules=6, nAnts=2)
    assert model.fit(X, y, n_gen=2, pop_size=6, random_state=0) is model
    np.testing.assert_array_equal(model.classes_, [0, 1, 2])
    assert model.n_features_in_ == 4

    prediction = model.predict(X)
    assert prediction.dtype.kind in 'iu'
    assert set(prediction) <= {0, 1, 2}
    assert model.score(X, y) == np.mean(prediction == y)
    np.testing.assert_array_equal(model.explainable_predict(X)[0], prediction)


def test_string_labels_round_trip(iris):
    X, y = iris
    labels = NAMES[y]
    model = _fit(X, labels)
    np.testing.assert_array_equal(model.classes_, NAMES)

    prediction = model.predict(X)
    assert set(prediction) <= set(NAMES)
    np.testing.assert_array_equal(prediction, model.predict(X, out_class_names=True))
    np.testing.assert_array_equal(model.explainable_predict(X)[0], prediction)
    assert model.score(X, labels) == np.mean(prediction == labels) > 0.5
    # The rule base keeps one consequent per class, in classes_ order.
    assert model.rule_base.get_consequents_names() == list(NAMES)


def test_non_contiguous_integer_labels_search_like_indexes(iris):
    X, y = iris
    labels = y * 10 + 5
    model = _fit(X, labels)
    np.testing.assert_array_equal(model.classes_, [5, 15, 25])

    prediction = model.predict(X)
    assert set(prediction) <= {5, 15, 25}
    assert model.score(X, labels) == np.mean(prediction == labels) > 0.5
    # The labels are encoded as consequent indexes, so the seeded search is
    # the one that runs on the labels 0, 1 and 2.
    reference = _fit(X, y)
    np.testing.assert_array_equal(prediction, reference.predict(X) * 10 + 5)


def test_class_names_for_integer_labels(iris):
    X, y = iris
    model = _fit(X, y, class_names=list(NAMES))
    np.testing.assert_array_equal(model.classes_, [0, 1, 2])

    prediction = model.predict(X)
    assert set(prediction) <= {0, 1, 2}
    np.testing.assert_array_equal(model.predict(X, out_class_names=True), NAMES[prediction])


def test_unknown_labels_are_rejected(iris):
    X, y = iris
    model = evf.BaseFuzzyRulesClassifier(nRules=6, nAnts=2, class_names=['a', 'b'])
    with pytest.raises(ValueError, match='not among the class names'):
        model.fit(X, NAMES[y], n_gen=1, pop_size=6)


def test_decode_handles_unknown_predictions():
    model = evf.BaseFuzzyRulesClassifier()
    model.classes_ = np.array([3, 7])
    np.testing.assert_array_equal(model._decode_predictions(np.array([1, -1, 0])), [7, -1, 3])
    model.classes_ = NAMES
    np.testing.assert_array_equal(
        model._decode_predictions(np.array([2, -1])), ['virginica', 'Unknown'])
    # Without fitted classes, as with precomputed rules, indexes pass through.
    unfitted = evf.BaseFuzzyRulesClassifier()
    np.testing.assert_array_equal(unfitted._decode_predictions(np.array([1., -1.])), [1., -1.])


def test_cross_validation_and_pipeline(iris):
    X, y = iris
    labels = NAMES[y]
    model = evf.BaseFuzzyRulesClassifier(nRules=6, nAnts=2)
    scores = cross_val_score(
        model, X, labels, cv=2, params={'n_gen': 3, 'pop_size': 10, 'random_state': 0})
    assert scores.shape == (2,) and np.all((scores >= 0.3) & (scores <= 1.0))

    pipeline = make_pipeline(StandardScaler(), clone(model))
    pipeline.fit(X, labels, basefuzzyrulesclassifier__n_gen=3,
                 basefuzzyrulesclassifier__pop_size=10, basefuzzyrulesclassifier__random_state=0)
    assert set(pipeline.predict(X)) <= set(NAMES)


def test_wrapper_classifiers_are_estimators(iris):
    from ex_fuzzy import classifiers
    from sklearn.exceptions import NotFittedError
    X, y = iris
    labels = NAMES[y]
    partitions = utils.construct_partitions(X, fs.FUZZY_SETS.t1)
    for factory in (classifiers.RuleMineClassifier, classifiers.RuleFineTuneClassifier):
        model = factory(nRules=6, nAnts=2, linguistic_variables=partitions)
        params = model.get_params()
        assert params['nRules'] == 6 and params['linguistic_variables'] is partitions
        assert clone(model).get_params()['nRules'] == 6
        with pytest.raises(NotFittedError):
            model.internal_classifier()

        assert model.fit(X, labels, n_gen=2, pop_size=8, random_state=0) is model
        np.testing.assert_array_equal(model.classes_, NAMES)
        assert model.n_features_in_ == 4
        prediction = model.predict(X)
        assert set(prediction) <= set(NAMES)
        assert model.score(X, labels) == np.mean(prediction == labels)
        antecedents = model.internal_classifier().rule_base.antecedents
        assert [antecedent.name for antecedent in antecedents] == [partition.name for partition in partitions]

        # The search settings can be given to the constructor, where GridSearchCV sees them.
        configured = factory(nRules=6, nAnts=2, linguistic_variables=partitions, n_gen=2, pop_size=8, random_state=0)
        assert configured.get_params()['n_gen'] == 2 and clone(configured).get_params()['random_state'] == 0
        configured.fit(X, labels)
        np.testing.assert_array_equal(configured.predict(X), prediction)


def test_conformal_classifier_is_an_estimator(iris):
    from ex_fuzzy.conformal import ConformalFuzzyClassifier
    X, y = iris
    model = ConformalFuzzyClassifier(nRules=6, nAnts=2, verbose=False)
    params = model.get_params()
    assert params['estimator'] is model.clf and model.clf.nRules == 6
    assert params['score_type'] == 'membership'
    copied = clone(model)
    assert copied.clf.nRules == 6 and copied.clf is not model.clf
    assert model.fit(X, y, cal_size=0.3, n_gen=2, pop_size=8, random_state=0) is model

    # A wrapped classifier that is not fitted yet is fitted by fit.
    inner = evf.BaseFuzzyRulesClassifier(nRules=6, nAnts=2)
    wrapped = ConformalFuzzyClassifier(inner)
    assert wrapped.fit(X, y, cal_size=0.3, n_gen=2, pop_size=8, random_state=0) is wrapped
    assert wrapped.clf is inner and inner.rule_base is not None
    assert set(wrapped.predict(X)) <= {0, 1, 2}

    assert ConformalFuzzyClassifier(clf_or_nRules=inner).clf is inner
    with pytest.raises(TypeError):
        ConformalFuzzyClassifier(inner, nAnts=3)


def test_temporal_classifier_predicts_and_returns_self(iris):
    from ex_fuzzy import temporal
    X, y = iris
    time_moments = np.arange(len(y)) % 2
    variables = [temporal.temporalFuzzyVariable(
                     lv.name, [temporal.temporalFS(fuzzy_set, np.array([1.0, 0.25]))
                               for fuzzy_set in lv.linguistic_variables])
                 for lv in utils.construct_partitions(X, fs.FUZZY_SETS.t1)]
    model = temporal.TemporalFuzzyRulesClassifier(nRules=4, nAnts=2, linguistic_variables=variables)
    assert clone(model).get_params()['nRules'] == 4
    assert model.fit(X, y, n_gen=2, pop_size=6, time_moments=time_moments) is model
    assert model.n_features_in_ == 4
    np.testing.assert_array_equal(model.predict(X, time_moments), model.forward(X, time_moments))
