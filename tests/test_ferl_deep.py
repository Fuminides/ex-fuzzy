"""Tests for DeepFERL, the deep learned-split member of the FERL family."""

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import load_iris

from ex_fuzzy.ferl_deep import DeepFERL

GOLDEN = json.loads((Path(__file__).parent / "data" / "deep_ferl_golden.json").read_text())
TIGHT = dict(rtol=1e-10, atol=1e-12)
RULES = ("dempster", "cautious", "hybrid", "incremental", "incremental_local", "mixture")


def _flatten(node):
    """Preorder list of node records, in the golden fixture's layout."""
    records = []

    def walk(current):
        record = {"leaf": bool(current["leaf"]), "dist": np.asarray(current["dist"]).tolist(),
                  "support": float(current["support"])}
        if not current["leaf"]:
            record.update(f=int(current["f"]), center=float(current["center"]),
                          h=float(current["h"]), lo=float(current["lo"]), hi=float(current["hi"]))
        records.append(record)
        if not current["leaf"]:
            walk(current["L"])
            walk(current["R"])

    walk(node)
    return records


def _load_reference_tree():
    """LearnedFuzzyTree from a fuzzy_greedy_tree checkout, if one is available."""
    root = Path(os.environ.get(
        "FGRT_REPO", Path(__file__).resolve().parents[2] / "fuzzy_greedy_tree"))
    path = root / "fgrt" / "core" / "learned_tree.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("_reference_learned_tree", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LearnedFuzzyTree


ReferenceTree = _load_reference_tree()


@pytest.mark.parametrize("case", GOLDEN["cases"],
                         ids=[f"{c['dataset']}-{c['config']}" for c in GOLDEN["cases"]])
def test_matches_fuzzy_greedy_tree_golden_outputs(case):
    X_train, X_test = np.array(case["X_train"]), np.array(case["X_test"])
    model = DeepFERL(**case["params"]).fit(X_train, np.array(case["y_train"]))

    assert model.n_rules() == case["n_rules"]
    tree = _flatten(model.root_)
    assert len(tree) == len(case["tree"])
    for ours, theirs in zip(tree, case["tree"]):
        assert ours.keys() == theirs.keys()
        assert ours["leaf"] == theirs["leaf"]
        assert ours.get("f") == theirs.get("f")
        for key in ("dist", "support", "center", "h", "lo", "hi"):
            if key in theirs:
                np.testing.assert_allclose(ours[key], theirs[key], **TIGHT)

    np.testing.assert_array_equal(model.predict(X_test), case["predict"])
    np.testing.assert_allclose(model.predict_proba(X_test), case["predict_proba"], **TIGHT)
    np.testing.assert_array_equal(model.predict_set(X_test).astype(int), case["predict_set"])
    for key, expected in case["predict_ds"].items():
        rule, leaves = key.split("|")
        result = model.predict_ds(X_test, rule=rule, leaves_only=leaves == "leaves_only=True")
        for name, value in zip(("betp", "bel", "pl", "ignorance"), result):
            np.testing.assert_allclose(value, expected[name], **TIGHT, err_msg=f"{key} {name}")


@pytest.mark.skipif(ReferenceTree is None,
                    reason="fuzzy_greedy_tree checkout not found; set FGRT_REPO to compare live")
@pytest.mark.parametrize("params", [
    dict(random_state=5),
    dict(random_state=1, criterion="cci", max_depth=6),
    dict(random_state=2, width=0.3, bounded_support=False, n_boot=4),
])
def test_is_bit_identical_to_the_reference_implementation(params):
    rng = np.random.default_rng(11)
    X = rng.normal(size=(160, 5))
    y = (X[:, 0] + 0.5 * X[:, 1] ** 2 > 0.3).astype(int) + (X[:, 2] > 1.0)
    X_test = np.vstack([rng.normal(size=(30, 5)), rng.normal(size=(4, 5)) * 20.0])

    ours = DeepFERL(**params).fit(X, y)
    theirs = ReferenceTree(**params).fit(X, y)

    assert ours.n_rules() == theirs.n_rules()
    np.testing.assert_array_equal(ours.predict_proba(X_test), theirs.predict_proba(X_test))
    np.testing.assert_array_equal(ours.predict_set(X_test), theirs.predict_set(X_test))
    for rule in RULES:
        for leaves_only in (False, True):
            for mine, reference in zip(ours.predict_ds(X_test, rule=rule, leaves_only=leaves_only),
                                       theirs.predict_ds(X_test, rule=rule, leaves_only=leaves_only)):
                np.testing.assert_array_equal(mine, reference)


@pytest.fixture(scope="module")
def iris_frame():
    dataset = load_iris()
    return pd.DataFrame(dataset.data, columns=dataset.feature_names), dataset.target


@pytest.fixture(scope="module")
def fitted(iris_frame):
    X, y = iris_frame
    model = DeepFERL(random_state=0)
    assert model.fit(X, y) is model
    return model


def test_is_a_scikit_learn_classifier(fitted, iris_frame):
    X, y = iris_frame
    probabilities = fitted.predict_proba(X)

    assert probabilities.shape == (len(X), 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert fitted.score(X, y) > 0.9
    assert fitted.n_features_in_ == 4
    assert np.array_equal(fitted.feature_names_in_, X.columns.to_numpy())
    assert clone(fitted).get_params() == fitted.get_params()
    stats = fitted.get_tree_stats()
    assert stats["leaves"] == fitted.n_rules()
    assert stats["max_depth"] <= fitted.max_depth
    assert stats["internal_nodes"] == stats["leaves"] - 1
    with pytest.raises(ValueError, match="features"):
        fitted.predict(X.to_numpy()[:, :3])


def test_fit_is_reproducible_for_a_seed(iris_frame):
    X, y = iris_frame
    first = DeepFERL(random_state=7, n_boot=5).fit(X, y).predict_proba(X)
    second = DeepFERL(random_state=7, n_boot=5).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(first, second)


def test_evidence_is_well_formed(fitted, iris_frame):
    X, _ = iris_frame
    sample = X.iloc[:15]
    for rule in ("dempster", "cautious", "hybrid", "incremental", "incremental_local"):
        betp, belief, plausibility, ignorance = fitted.predict_credal(sample, rule=rule)
        assert np.allclose(betp.sum(axis=1), 1.0)
        assert np.all(belief >= -1e-12)
        assert np.all(belief <= plausibility + 1e-12)
        assert np.allclose(plausibility - belief, ignorance[:, None])
        assert np.all((ignorance >= 0.0) & (ignorance <= 1.0))
    sets = fitted.predict_set(sample)
    assert sets.dtype == np.bool_ and sets.shape == (15, 3) and sets.any(axis=1).all()
    assert np.all(fitted.predict_dirichlet(sample) >= 1.0)


def test_bounded_support_flags_far_out_of_distribution_inputs(iris_frame):
    X, y = iris_frame
    inside = X.to_numpy()[:5]
    far = inside * 50.0
    bounded = DeepFERL(random_state=0).fit(X, y)
    unbounded = DeepFERL(random_state=0, bounded_support=False).fit(X, y)

    np.testing.assert_allclose(bounded.firing_strength(inside), 1.0)
    assert np.all(bounded.firing_strength(far) < 1e-6)
    assert np.all(bounded.predict_credal(far)[3] > 0.99)
    assert np.mean(unbounded.predict_credal(far)[3]) < np.mean(bounded.predict_credal(far)[3])
    np.testing.assert_allclose(bounded.predict_proba(far), 1.0 / 3)


def test_missing_features_split_evenly(fitted, iris_frame):
    X, _ = iris_frame
    values = X.to_numpy()[:8]

    nothing_observed = np.zeros_like(values, dtype=bool)
    blind = fitted.predict_proba(values, observed_mask=nothing_observed)
    assert np.allclose(blind.sum(axis=1), 1.0)
    assert np.allclose(blind, blind[0])            # no information, same answer for all

    used, stack = set(), [fitted.root_]
    while stack:
        node = stack.pop()
        if not node["leaf"]:
            used.add(node["f"])
            stack.extend((node["L"], node["R"]))
    for feature in set(range(values.shape[1])) - used:
        mask = np.ones_like(values, dtype=bool)
        mask[:, feature] = False
        np.testing.assert_array_equal(fitted.predict_proba(values, observed_mask=mask),
                                      fitted.predict_proba(values))

    with_gaps = values.copy()
    with_gaps[:, 2] = np.nan
    mask = np.ones_like(values, dtype=bool)
    mask[:, 2] = False
    assert np.all(np.isfinite(fitted.predict_proba(with_gaps, observed_mask=mask)))
    with pytest.raises(ValueError, match="observed_mask"):
        fitted.predict_proba(values, observed_mask=mask[:, :3])


@pytest.mark.parametrize("kwargs, message", [
    ({"criterion": "entropy"}, "criterion"),
    ({"width": "wide"}, "width"),
    ({"width": -1.0}, "width"),
    ({"n_boot": 1}, "n_boot"),
    ({"max_depth": -1}, "max_depth"),
    ({"min_leaf_w": 0.0}, "min_leaf_w"),
    ({"oob_margin": -0.1}, "oob_margin"),
])
def test_rejects_invalid_parameters(kwargs, message):
    X, y = load_iris(return_X_y=True)
    with pytest.raises(ValueError, match=message):
        DeepFERL(**kwargs).fit(X, y)


def test_rejects_unknown_evidence_rules(fitted, iris_frame):
    X, _ = iris_frame
    with pytest.raises(ValueError, match="rule"):
        fitted.predict_ds(X.iloc[:2], rule="yager")


def test_print_tree_names_features(fitted, capsys):
    fitted.print_tree()
    printed = capsys.readouterr().out
    assert printed.startswith("root:")
    assert any(name in printed for name in fitted.feature_names_in_)


def test_best_cut_without_usable_or_useful_splits():
    from ex_fuzzy import ferl_deep
    x = np.array([0.0, 1.0, 2.0])

    # Every cut leaves one side without weight.
    unusable = ferl_deep._best_cut(x, np.eye(2)[[0, 1, 0]], np.array([1.0, 0.0, 0.0]), 0.5, 1.0)
    assert unusable == (0.0, None)
    # A pure node cannot gain from a cut.
    assert ferl_deep._best_cut(x, np.eye(2)[[0, 0, 0]], np.ones(3), 0.0, 3.0) == (0.0, None)


def test_invalid_training_data_is_rejected():
    X, y = load_iris(return_X_y=True)
    model = DeepFERL(max_depth=1, n_boot=2)

    with pytest.raises(ValueError, match="two-dimensional"):
        model.fit(X[:, 0], y)
    with pytest.raises(ValueError, match="same number"):
        model.fit(X[:10], y)
    with pytest.raises(ValueError, match="at least one"):
        model.fit(X[:0], y[:0])
    with pytest.raises(ValueError, match="NaN"):
        model.fit(np.where(X == X[0, 0], np.nan, X), y)


def test_feature_names_follow_the_last_fit_and_labels_can_be_a_column():
    X, y = load_iris(return_X_y=True)
    assert not hasattr(DeepFERL(max_depth=1, n_boot=2).fit(pd.DataFrame(X), y), "feature_names_in_")

    model = DeepFERL(max_depth=1, n_boot=2, random_state=0)
    model.fit(pd.DataFrame(X, columns=list("abcd")), y.reshape(-1, 1))
    assert model.feature_names_in_.tolist() == list("abcd")
    model.fit(pd.DataFrame(X), y)
    assert not hasattr(model, "feature_names_in_")


def test_small_or_unsplittable_nodes_become_leaves():
    # Three samples are too few to split.
    assert DeepFERL(min_leaf_w=0.5, n_boot=2).fit(np.array([[0.0], [1.0], [2.0]]), [0, 1, 0]).n_rules() == 1
    # A constant feature offers no cut.
    assert DeepFERL(n_boot=2).fit(np.zeros((10, 1)), [0, 1] * 5).n_rules() == 1
    # With seed 0 a bootstrap resample misses the only positive sample, so the split is too unstable to keep.
    unstable = DeepFERL(n_boot=2, min_leaf_w=0.5, random_state=0)
    assert unstable.fit(np.array([[0.0], [0.0], [0.0], [1.0]]), [0, 0, 0, 1]).n_rules() == 1


def test_a_root_leaf_tree(capsys):
    X, y = load_iris(return_X_y=True)
    stump = DeepFERL(max_depth=0).fit(X, y)

    M, cons, names, support = stump.node_activation_matrix(X)
    assert M.shape == (150, 0) and cons.shape == (0, 3) and names == [] and support.shape == (0,)
    np.testing.assert_array_equal(stump.firing_strength(X), np.ones(150))
    betp, _, _, ignorance = stump.predict_ds(X[:2])
    assert betp.shape == (2, 3) and ignorance.shape == (2,)

    stump.print_tree()
    assert capsys.readouterr().out.startswith("root: ")


def test_single_samples_masks_routing_and_reliability(capsys):
    X, y = load_iris(return_X_y=True)
    model = DeepFERL(max_depth=3, n_boot=5, random_state=0).fit(X, y)

    assert model.predict(X[0]).shape == (1,)
    assert model.predict_proba(X[0], observed_mask=np.array([True, False, True, True])).shape == (1, 3)
    with pytest.raises(ValueError, match="one- or two-dimensional"):
        model.predict(X[None, :, :])

    _, _, names, _ = model.node_activation_matrix(X)
    depths = DeepFERL.node_depths(names)
    assert depths.min() == 1 and depths.max() <= 3

    assert model.predict_ds(X, top_p=0.9)[0].shape == (150, 3)
    discounted = model.predict_ds(X, reliability_vec=np.full(len(names), 0.5))[3]
    assert np.all(discounted >= model.predict_ds(X)[3] - 1e-12)

    model.print_tree(feature_names=list("abcd"))
    tree = capsys.readouterr().out
    assert tree.startswith("root: ")
    assert any(f"{name} below" in tree for name in "abcd")
