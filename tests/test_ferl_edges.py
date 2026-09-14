"""Deterministic edge coverage for FERL's public and legacy helper paths."""

import copy

import numpy as np
import pandas as pd
import pytest

import ferl as ferl_module
from ferl import FERL, LearnedRampSet


X_SMALL = np.array([[0.0], [0.2], [0.8], [1.0]])
Y_SMALL = np.array([0, 0, 1, 1])


@pytest.fixture()
def small_tree():
    """A stable root with two non-overlapping leaves."""
    return FERL(
        n_partitions=2,
        max_rules=4,
        min_improvement=-1.0,
        target_metric="purity",
        random_state=0,
    ).fit(X_SMALL, Y_SMALL, patience=2)


def test_scalar_split_metrics_and_learned_ramps_cover_edge_cases():
    assert np.isinf(ferl_module._weighted_gini_index(np.array([]), np.array([])))
    assert np.isinf(ferl_module._weighted_gini_index(np.zeros(2), np.array([0, 1])))
    assert ferl_module._gini_index(np.array([])) == 0.0
    assert ferl_module._gini_index(np.array([0, 1])) == 0.5
    assert ferl_module._complete_classification_index(
        np.array([]), np.array([]), np.array([])
    ) == 0.0
    assert ferl_module._complete_classification_index(
        np.array([0, 1]), np.array([1, 1]), np.array([0, 1])
    ) == 1.0
    assert ferl_module.compute_purity(np.array([False, False]), np.array([0, 1])) == 0.0
    assert ferl_module.compute_purity(np.array([True, True]), np.array([0, 1])) == 0.5
    assert np.isinf(ferl_module.compute_fuzzy_purity(
        np.ones(2), np.array([0, 1]), minimum_coverage_threshold=1.1
    ))
    assert ferl_module.compute_fuzzy_cci(
        np.array([0, 1]), np.ones(2), np.array([0, 0]), np.array([0, 1]), 1.1
    ) == -1.0

    below = LearnedRampSet(0.5, 0.0, "below", name="low")
    above = LearnedRampSet(0.5, 0.1, "above")
    np.testing.assert_allclose(below(np.array([0.0, 1.0])), [1.0, 0.0])
    np.testing.assert_allclose(
        above.membership(np.array([0.4, 0.6])), [0.0, 1.0], atol=1e-15
    )

    no_gain = ferl_module._learned_best_cut(
        np.array([0.0, 1.0]), np.ones((2, 1)), np.ones(2), 0.0, 2.0
    )
    assert no_gain == (0.0, None)
    invalid_weights = ferl_module._learned_best_cut(
        np.array([0.0, 1.0]), np.eye(2), np.array([2.0, 0.0]), 0.5, 2.0
    )
    assert invalid_weights == (0.0, None)


@pytest.mark.parametrize(
    "X, y, message",
    [
        (np.array([0.0, 1.0]), np.array([0, 1]), "two-dimensional"),
        (np.ones((2, 1)), np.array([0]), "same number"),
        (np.empty((0, 1)), np.array([]), "at least one"),
        (np.array([[0.0], [np.inf]]), np.array([0, 1]), "NaN or infinite"),
        (np.ones((2, 1)), np.array([0, 0]), "two target classes"),
    ],
)
def test_fit_validates_training_arrays(X, y, message):
    with pytest.raises(ValueError, match=message):
        FERL().fit(X, y)


def test_fit_flattens_targets_checks_partitions_and_refreshes_feature_names():
    model = FERL(max_rules=1).fit(X_SMALL, Y_SMALL[:, None])
    assert model.classes_.tolist() == [0, 1]

    one_partition = copy.deepcopy(model.fuzzy_partitions_)
    with pytest.raises(ValueError, match="one fuzzy variable per feature"):
        FERL(fuzzy_partitions=one_partition).fit(np.c_[X_SMALL, X_SMALL], Y_SMALL)

    frame = pd.DataFrame(X_SMALL, columns=["amount"])
    named = FERL(max_rules=1).fit(frame, Y_SMALL)
    assert named.feature_names_in_.tolist() == ["amount"]
    named.fit(X_SMALL, Y_SMALL)
    assert not hasattr(named, "feature_names_in_")


def test_prediction_input_validation_and_simple_helpers(small_tree):
    model = small_tree
    np.testing.assert_array_equal(model._as_array(np.array([0.2])), [[0.2]])
    with pytest.raises(ValueError, match="one- or two-dimensional"):
        model.predict(np.zeros((1, 1, 1)))
    with pytest.raises(ValueError, match="1 features"):
        model.predict(np.zeros((2, 2)))

    assert model._majority_class(np.array([])) == 0
    assert model._majority_class(np.array([9, 1]), np.array([1.0, 0.0])) == 0
    np.testing.assert_allclose(model._class_probabilities(np.array([])), [0.5, 0.5])
    np.testing.assert_allclose(
        model._class_probabilities(np.array([0, 1]), np.zeros(2)), [0.5, 0.5]
    )
    assert model._get_best_possible_coverage(np.empty((0, 1)), np.array([])) == 0.0


def test_legacy_leaf_prediction_and_path_apis(small_tree, monkeypatch):
    model = small_tree
    X = np.array([[0.1], [0.9]])
    observed = np.array([[True], [False]])

    probabilities = model._predict_proba_direct_leaves(X, observed)
    assert probabilities.shape == (2, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    memberships, predictions = model.predict_all_leaves(X, observed)
    matrix, predictions_array, names = model.predict_all_leaves_matrix(X, observed)
    assert names == list(memberships)
    np.testing.assert_allclose(matrix, np.column_stack(list(memberships.values())))
    np.testing.assert_array_equal(predictions_array, list(predictions.values()))

    pred, strength, paths = model._predict_direct_leaves(X)
    assert pred.shape == strength.shape == paths.shape == (2,)
    model._cached_leaves = []
    pred, strength, paths = model._predict_direct_leaves(X)
    np.testing.assert_array_equal(pred, model._root["prediction"])
    np.testing.assert_array_equal(paths, "root")
    model._invalidate_leaf_cache()

    leaf = next(iter(model._get_all_leaf_nodes().values()))
    assert model._get_path_to_leaf(model._root) == []
    assert model._get_node_path(model._root) == []
    assert model._get_node_path({"name": "root_F0_bad_F0_L1"}) == [(0, 1)]
    assert model._get_path_to_leaf({"name": "root_bad_F0_L1"})[-1] == {
        "type": "split", "feature": 0, "fuzzy_set": 1
    }
    uncached = model._calculate_membership_to_leaf(X, leaf, observed)
    monkeypatch.setattr(model, "_get_cached_memberships", lambda _X: (_ for _ in ()).throw(RuntimeError()))
    fallback = model._calculate_membership_to_leaf(X, leaf, observed)
    np.testing.assert_allclose(fallback, uncached)


def test_recursive_probability_fallbacks_and_prediction_modes(small_tree):
    model = small_tree
    X = np.array([[0.1], [0.9]])
    observed = np.ones_like(X, dtype=bool)

    result = model._predict_proba(X, model._root)
    assert result.shape == (2, 2)
    root_only = copy.deepcopy(model._root)
    root_only.pop("children")
    root_only.pop("class_probabilities")
    np.testing.assert_allclose(model._predict_proba(X, root_only), 0.5)

    child = next(iter(model._root["children"].values()))
    saved_child_probs = child.pop("class_probabilities")
    saved_root_probs = model._root.pop("class_probabilities")
    try:
        result = model._predict_proba(X, model._root)
        assert np.isfinite(result).all()
    finally:
        child["class_probabilities"] = saved_child_probs
        model._root["class_probabilities"] = saved_root_probs

    for mode in ("winner", "soft", "soft_gate", "hard_gate"):
        model.prediction_mode = mode
        assert model.predict(X).shape == (2,)
        np.testing.assert_allclose(model.predict_proba(X).sum(axis=1), 1.0)
    assert model.predict_with_path(X)[0].shape == (2,)
    assert model.firing_strength(X).shape == (2,)


def test_node_activation_and_evidence_empty_rule_paths(small_tree):
    model = small_tree
    X = np.array([[0.5]])
    observed = np.array([[False]])
    child = next(iter(model._root["children"].values()))
    child["class_probabilities"] = None
    model._invalidate_leaf_cache()
    M, consequents, names = model.node_activation_matrix(
        X, observed_mask=observed, membership_floor=0.2
    )
    assert M.shape[1] == len(consequents) == len(names)
    assert np.all(M >= 0.2)
    np.testing.assert_allclose(consequents.sum(axis=1), 1.0)

    root_only = FERL(max_rules=1, random_state=0).fit(X_SMALL, Y_SMALL)
    betp, belief, plausibility, ignorance = root_only.predict_ds(X)
    np.testing.assert_allclose(betp, 0.5)
    np.testing.assert_allclose(belief, 0.0)
    np.testing.assert_allclose(plausibility, 1.0)
    np.testing.assert_allclose(ignorance, 1.0)
    assert root_only.predict_dirichlet(X).shape == (1, 2)
    assert root_only.predict_credal(X)[0].shape == (1, 2)
    assert root_only.predict_set(X).all()


def test_tree_introspection_printing_and_lookup(small_tree, capsys):
    model = small_tree
    children = list(model._root["children"].values())
    children[0]["aux_purity_cache"] = {}
    children[0]["quality_improvement"] = 0.125

    model.print_tree()
    output = capsys.readouterr().out
    assert "FERL tree" in output and "split_criterion=0.125" in output
    assert model.get_tree_stats() == {
        "total_nodes": 3, "leaves": 2, "internal": 1, "depth": 1
    }
    assert model._find_node_by_name(children[0]["name"]) is children[0]
    assert model._node_has_children("missing") is False
    assert model._get_node_children_names("missing") == []
    assert model._get_node_children_names(children[0]["name"]) == []


def test_dummy_nodes_real_splits_and_multiway_expansion(small_tree):
    model = small_tree
    root = model._root
    prediction = model._split_node_dummy(root, 0, 0, X_SMALL, Y_SMALL)
    dummy_name = "root_F0_L0_dummy"
    assert prediction in model.classes_ and dummy_name in root["children"]
    with pytest.raises(ValueError, match="already exists"):
        model._split_node_dummy(root, 0, 0, X_SMALL, Y_SMALL)
    model.node_dict_access[dummy_name] = root["children"][dummy_name]
    model._delete_node_dummy(root, 0, 0)
    assert dummy_name not in root["children"] and dummy_name not in model.node_dict_access
    model._delete_node_dummy({}, 0, 0)

    root["aux_purity_cache"] = {"feature": -1}
    model._expand_multiway(root, X_SMALL, Y_SMALL)
    root["aux_purity_cache"] = {"feature": 0}
    model._expand_multiway(root, X_SMALL, Y_SMALL)  # existing names are skipped

    fresh = FERL(n_partitions=2, max_rules=1).fit(X_SMALL, Y_SMALL)
    fresh._root["aux_purity_cache"] = {
        "split_criterion": 0.5,
        "feature": 0,
        "fuzzy_set": 0,
        "coverage": 0.5,
        "child_decision": 0,
    }
    fresh._split_node(fresh._root, X_SMALL, Y_SMALL)
    fresh._root["child_splits"][0][0] = True
    with pytest.raises(ValueError, match="already exists"):
        fresh._split_node(fresh._root, X_SMALL, Y_SMALL)


def test_split_checks_sampling_depth_and_recursive_selection(small_tree, monkeypatch):
    model = small_tree
    root = model._root
    depth_node = copy.deepcopy(root)
    depth_node["depth"] = model.max_depth
    assert model._node_purity_checks(depth_node, X_SMALL, Y_SMALL) == -np.inf
    assert model._node_cci_checks(depth_node, X_SMALL, Y_SMALL)[0] == -np.inf

    blocked = copy.deepcopy(root)
    blocked.pop("children", None)
    blocked["father_path"] = [np.zeros(2, dtype=bool)]
    blocked["child_splits"] = [np.ones(2, dtype=bool)]
    assert model._node_purity_checks(blocked, X_SMALL, Y_SMALL) == -np.inf
    assert blocked["aux_purity_cache"]["feature"] == -1

    model.sample_for_splits = True
    model.sample_size = 2
    model._rng = np.random.default_rng(0)
    sample_node = copy.deepcopy(root)
    sample_node["child_splits"] = [np.ones(2, dtype=bool)]
    assert np.isfinite(model._node_purity_checks(sample_node, X_SMALL, Y_SMALL))

    scores = {"root": 0.0, **{name: 1.0 for name in root["children"]}}
    monkeypatch.setattr(model, "_node_purity_checks", lambda node, X, y: scores[node["name"]])
    assert model._get_best_node_split(root, X_SMALL, Y_SMALL)[1] != "root"

    values = {"root": (0.0, 1.0)}
    values.update({name: (0.0, 0.0) for name in root["children"]})
    monkeypatch.setattr(
        model, "_node_cci_checks", lambda node, X, y, ctx=None: values[node["name"]]
    )
    ctx = {"sentinel": True}
    assert model._get_best_node_split_cci(root, X_SMALL, Y_SMALL, ctx)[1] != "root"


def test_cci_context_and_candidate_coverage_edges(small_tree):
    model = small_tree
    model.prediction_mode = "winner"
    model.consistent_cci = False
    model.coverage_weight = 0.5
    context = model._build_cci_context(X_SMALL, Y_SMALL)
    assert context["base_votes"] is None
    assert context["uncovered_mask"].shape == Y_SMALL.shape

    candidate = copy.deepcopy(model._root)
    candidate.pop("children", None)
    candidate["father_path"] = [np.ones(2, dtype=bool)]
    candidate["child_splits"] = [np.ones(2, dtype=bool)]
    model.coverage_threshold = 2.0
    cci, purity = model._node_cci_checks(candidate, X_SMALL, Y_SMALL, context)
    assert cci == -np.inf and purity == np.inf
    assert candidate["aux_purity_cache"]["feature"] == -1


def _root_only_learned(**kwargs):
    model = FERL(
        n_partitions=2, max_rules=1, split_mode="learned", random_state=0,
        **kwargs,
    ).fit(X_SMALL, Y_SMALL)
    return model, model._build_cci_context(X_SMALL, Y_SMALL)


def test_learned_candidate_sampling_coverage_and_cached_results():
    model, context = _root_only_learned(learned_width=0.2, coverage_weight=0.5)
    context["sample_indices"] = np.arange(len(Y_SMALL))
    context["uncovered_mask"] = np.ones(len(Y_SMALL), dtype=bool)
    score, purity = model._learned_cci_candidates(model._root, context)
    assert np.isfinite(score) and np.isfinite(purity)
    assert model._learned_cci_candidates(model._root, context) == (score, purity)

    model._root.pop("_learned_aux")
    model.coverage_threshold = 2.0
    score, purity = model._learned_cci_candidates(model._root, context)
    assert score == -np.inf and purity == np.inf

    model._root["_learned_exhausted"] = True
    assert model._learned_cci_candidates(model._root, context) == (-np.inf, np.inf)


def test_bootstrap_learned_candidate_rejects_too_few_cut_replicates():
    model, context = _root_only_learned(
        learned_width="bootstrap", learned_n_boot=1
    )
    score, purity = model._learned_cci_candidates(model._root, context)
    assert score == -np.inf and purity == np.inf


def test_remaining_path_coverage_and_dummy_branches(small_tree):
    model = small_tree
    root = model._root
    np.testing.assert_array_equal(
        model._get_node_samples_mask(root, X_SMALL), np.ones(4, dtype=bool)
    )
    # Invalid legacy path components are ignored independently.
    invalid_path = {"name": "legacy", "_cached_path": [(2, 0), (0, 9)]}
    np.testing.assert_array_equal(
        model._get_node_samples_mask(invalid_path, X_SMALL), np.ones(4, dtype=bool)
    )
    assert model._get_node_path({"name": "root"}) == []

    model._get_node_samples_mask = lambda node, X: np.zeros(len(X), dtype=bool)
    assert model._get_best_possible_coverage(X_SMALL, Y_SMALL) == 0.0

    weighted = small_tree
    weighted._get_node_samples_mask = FERL._get_node_samples_mask.__get__(weighted)
    assert weighted._get_best_possible_coverage(
        X_SMALL, Y_SMALL, sample_weight=np.zeros(len(Y_SMALL))
    ) == 0.0

    fresh = FERL(n_partitions=2, max_rules=1).fit(X_SMALL, Y_SMALL)
    prediction = fresh._split_node_dummy(fresh._root, 0, 0, X_SMALL, Y_SMALL)
    assert prediction in fresh.classes_
    dummy = "root_F0_L0_dummy"
    del fresh._root["children"][dummy]
    fresh.node_dict_access[dummy] = {"name": dummy}
    fresh._delete_node_dummy(fresh._root, 0, 0)
    assert dummy not in fresh.node_dict_access
    fresh._root["children"][dummy] = {"name": dummy}
    fresh._delete_node_dummy(fresh._root, 0, 0)


def test_prediction_defaults_zero_membership_and_internal_fallbacks(small_tree):
    model = small_tree
    outside = np.array([[10.0]])
    observed = np.ones_like(outside, dtype=bool)

    model._invalidate_leaf_cache()
    assert model._predict_direct_leaves(outside)[0].shape == (1,)
    np.testing.assert_allclose(model._predict_proba_direct_leaves(outside), 0.5)
    assert model.predict_all_leaves(outside)[0]

    with_root_prior = model._predict_proba(outside, model._root)
    np.testing.assert_allclose(with_root_prior, [model._root["class_probabilities"]])
    prior = model._root.pop("class_probabilities")
    try:
        np.testing.assert_allclose(model._predict_proba(outside, model._root), 0.5)
    finally:
        model._root["class_probabilities"] = prior

    model._cached_all_nodes = []
    pred, memberships, paths = model._predict_all_nodes(outside, observed)
    np.testing.assert_array_equal(pred, model._root["prediction"])
    np.testing.assert_array_equal(memberships, 1.0)
    np.testing.assert_array_equal(paths, "root")
    model._invalidate_leaf_cache()

    child = next(iter(model._root["children"].values()))
    child["class_probabilities"] = None
    model.prediction_mode = "soft_gate"
    np.testing.assert_allclose(model.predict_proba(outside), 0.5)

    assert model.predict(outside, observed_mask=observed).shape == (1,)
    assert model.predict_with_path(outside, observed_mask=observed)[0].shape == (1,)
    assert model.firing_strength(outside, observed_mask=observed).shape == (1,)
    assert model.predict_credal(outside, leaves_only=True)[0].shape == (1, 2)

    # A deliberately stale cache exercises the defensive child-lookup guards.
    model._cached_all_nodes = [{
        "prediction": model._root["prediction"],
        "name": "root",
        "path_features": [],
        "path_fuzzy_sets": [],
        "path_length": 0,
    }]
    model._predict_all_nodes(outside, observed)
    model._predict_proba_all_nodes(outside, observed)
    model._invalidate_leaf_cache()

    leaf = next(iter(model._get_all_leaf_nodes().values()))
    original_path = model._get_path_to_leaf
    model._get_path_to_leaf = lambda node: [{"type": "metadata"}]
    try:
        np.testing.assert_array_equal(
            model._calculate_membership_to_leaf(outside, leaf, observed), [1.0]
        )
    finally:
        model._get_path_to_leaf = original_path


def test_print_tree_without_optional_child_quality(small_tree, capsys):
    model = small_tree
    children = list(model._root["children"].values())
    children[0].pop("aux_purity_cache", None)
    children[1]["aux_purity_cache"] = {}
    children[1].pop("quality_improvement", None)
    model.print_tree()
    assert "split_criterion" not in capsys.readouterr().out


def test_multiway_duplicate_and_new_sibling_paths():
    model = FERL(n_partitions=2, max_rules=1).fit(X_SMALL, Y_SMALL)
    root = model._root
    root["aux_purity_cache"] = {"feature": 0}
    root["children"] = {}
    duplicate_name = "root_F0_L0"
    model.node_dict_access[duplicate_name] = {"name": duplicate_name}
    model._expand_multiway(root, X_SMALL, Y_SMALL)
    assert duplicate_name not in root["children"]
    assert "root_F0_L1" in root["children"]


def test_learned_split_accepts_an_existing_children_dictionary():
    model, _ = _root_only_learned(learned_width=0.2)
    root = model._root
    root["children"] = {}
    model._split_node_learned(root, X_SMALL, Y_SMALL, {
        "feature": 0,
        "learned_center": 0.5,
        "learned_h": 0.1,
        "split_criterion": 0.25,
    })
    assert len(root["children"]) == 2


def test_pruning_one_child_recursive_removal_and_default_alpha(small_tree, monkeypatch):
    model = small_tree
    child = next(iter(model._root["children"].values()))
    one_child = copy.deepcopy(model._root)
    one_child["children"] = {child["name"]: copy.deepcopy(child)}
    assert model._calculate_complexity_measure(one_child, X_SMALL, Y_SMALL) == np.inf

    deep = {
        "name": "parent",
        "children": {"nested": {"name": "nested", "children": {"gone": {"name": "gone"}}}},
    }
    model.node_dict_access.update({name: {"name": name} for name in ("parent", "nested", "gone")})
    model._remove_from_node_dict(deep)
    assert not {"parent", "nested", "gone"} & set(model.node_dict_access)
    model._remove_from_node_dict({"name": "already-absent"})
    model._prune_subtree({"name": "leaf"}, X_SMALL, Y_SMALL)

    model.ccp_alpha = -np.inf
    assert model.cost_complexity_pruning(X_SMALL, Y_SMALL) == [0.0]

    nested_parent = next(iter(model._root["children"].values()))
    nested_parent["children"] = {"nested-leaf": {"name": "nested-leaf"}}
    monkeypatch.setattr(model, "_calculate_complexity_measure", lambda *args: 0.0)
    weakest, alpha = model._find_weakest_link(X_SMALL, Y_SMALL)
    assert weakest is model._root and alpha == 0.0

    sequence = iter([(nested_parent, 0.0), (None, np.inf)])
    monkeypatch.setattr(model, "_find_weakest_link", lambda X, y: next(sequence))
    monkeypatch.setattr(model, "_prune_subtree", lambda *args: None)
    assert model.cost_complexity_pruning(X_SMALL, Y_SMALL, alpha=1.0) == [0.0, 0.0]


def test_pruning_helpers_and_tree_restore(small_tree):
    model = small_tree
    root = model._root
    child = next(iter(root["children"].values()))
    empty_child = copy.deepcopy(child)
    empty_child["existing_membership"] = np.zeros(len(Y_SMALL))
    assert model._calculate_node_impurity(empty_child, X_SMALL, Y_SMALL) == 0.0
    assert model._calculate_subtree_impurity(child, X_SMALL, Y_SMALL) >= 0.0
    assert model._calculate_subtree_impurity(root, X_SMALL, Y_SMALL) >= 0.0
    assert model._count_leaves(root) == 2
    assert model._calculate_complexity_measure(child, X_SMALL, Y_SMALL) == np.inf

    weakest, alpha = model._find_weakest_link(X_SMALL, Y_SMALL)
    assert weakest is root and np.isfinite(alpha)
    backup = model._deep_copy_tree()
    old_names = set(model.node_dict_access)
    model._prune_subtree(root, X_SMALL, Y_SMALL)
    assert "children" not in root and set(model.node_dict_access) == {"root"}
    model._restore_tree(backup)
    assert set(model.node_dict_access) == old_names
    assert model.cost_complexity_pruning(X_SMALL, Y_SMALL, alpha=np.inf)


def test_fit_with_pruning_uses_training_or_explicit_validation_data():
    train_model = FERL(
        n_partitions=2, max_rules=3, min_improvement=-1.0,
        target_metric="purity", random_state=0,
    )
    alpha, score = train_model.fit_with_pruning(X_SMALL, Y_SMALL)
    assert np.isfinite(alpha) and 0.0 <= score <= 1.0

    validation_model = FERL(
        n_partitions=2, max_rules=3, min_improvement=-1.0,
        target_metric="purity", random_state=0,
    )
    alpha, score = validation_model.fit_with_pruning(
        X_SMALL, Y_SMALL, X_val=X_SMALL[::-1], y_val=Y_SMALL[::-1]
    )
    assert np.isfinite(alpha) and 0.0 <= score <= 1.0
