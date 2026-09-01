"""Tests for Fast Evidential Rule Learning (FERL)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from ferl import FERL
from ferl_partitions import learn_partitions_mdlp
import fuzzy_sets as fs
import utils


@pytest.fixture(scope="module")
def iris_split():
    dataset = load_iris()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    return train_test_split(
        X, dataset.target, test_size=0.25, random_state=0, stratify=dataset.target
    )


@pytest.fixture(scope="module")
def compact_ferl(iris_split):
    X_train, _, y_train, _ = iris_split
    model = FERL(max_rules=6, min_improvement=0.0, random_state=0)
    assert model.fit(X_train, y_train, patience=5) is model
    return model


def test_ferl_is_public_sklearn_estimator(compact_ferl, iris_split):
    _, X_test, _, y_test = iris_split
    model = compact_ferl

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    assert predictions.shape == y_test.shape
    assert probabilities.shape == (len(X_test), len(model.classes_))
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert model.score(X_test, y_test) >= 0.80
    assert model.n_rules() <= model.max_rules
    assert model.n_features_in_ == X_test.shape[1]
    assert np.array_equal(model.feature_names_in_, X_test.columns.to_numpy())
    assert clone(model).get_params()["max_rules"] == model.max_rules


def test_ferl_native_evidence_is_well_formed(compact_ferl, iris_split):
    _, X_test, _, _ = iris_split
    betp, belief, plausibility, ignorance = compact_ferl.predict_credal(
        X_test.iloc[:12]
    )
    prediction_sets = compact_ferl.predict_set(X_test.iloc[:12])

    assert betp.shape == belief.shape == plausibility.shape == (12, 3)
    assert ignorance.shape == (12,)
    assert prediction_sets.shape == (12, 3)
    assert prediction_sets.dtype == np.bool_
    assert np.all(prediction_sets.any(axis=1))
    assert np.allclose(betp.sum(axis=1), 1.0)
    assert np.all(belief >= -1e-12)
    assert np.all(belief <= plausibility + 1e-12)
    assert np.allclose(plausibility - belief, ignorance[:, None])
    assert np.all((ignorance >= 0.0) & (ignorance <= 1.0))


def test_mdlp_partitions_are_native_fuzzy_variables():
    X, y = load_iris(return_X_y=True)
    partitions = learn_partitions_mdlp(X, y)

    assert len(partitions) == X.shape[1]
    assert all(isinstance(variable, fs.fuzzyVariable) for variable in partitions)
    assert all(len(variable) >= 1 for variable in partitions)

    model = FERL(
        partition="mdlp", max_rules=5, min_improvement=0.0, random_state=0
    ).fit(X, y, patience=4)
    assert model.predict_proba(X[:5]).shape == (5, 3)


def test_learned_ferl_is_reproducible_and_does_not_mutate_input_partitions():
    X, y = load_iris(return_X_y=True)
    partitions = utils.construct_partitions(X, fs.FUZZY_SETS.t1, n_partitions=3)
    original_sizes = [len(variable) for variable in partitions]
    model = FERL(
        fuzzy_partitions=partitions,
        split_mode="learned",
        learned_n_boot=3,
        max_rules=5,
        max_depth=3,
        min_improvement=0.0,
        random_state=4,
    )

    model.fit(X, y, patience=4)
    first = model.predict_proba(X[:15])
    model.fit(X, y, patience=4)
    second = model.predict_proba(X[:15])

    assert np.allclose(first, second)
    assert [len(variable) for variable in partitions] == original_sizes
    assert any(
        len(fitted) > original
        for fitted, original in zip(model.fuzzy_partitions_, original_sizes)
    )


def test_ferl_supports_missing_feature_masks(compact_ferl, iris_split):
    _, X_test, _, _ = iris_split
    X = X_test.iloc[:6].to_numpy()
    observed = np.ones_like(X, dtype=bool)
    observed[:, 0] = False

    probabilities = compact_ferl.predict_proba(X, observed_mask=observed)
    sets = compact_ferl.predict_set(X, observed_mask=observed)

    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert sets.any(axis=1).all()


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"partition": "unknown"}, "partition"),
        ({"split_mode": "unknown"}, "split_mode"),
        ({"prediction_mode": "unknown"}, "prediction_mode"),
    ],
)
def test_ferl_rejects_unknown_modes(kwargs, message):
    X, y = load_iris(return_X_y=True)
    with pytest.raises(ValueError, match=message):
        FERL(**kwargs).fit(X, y)
