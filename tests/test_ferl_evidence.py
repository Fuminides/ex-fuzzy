"""FERL's evidence combination, now shared with DeepFERL through ``_evidence``.

``data/ferl_predict_ds_golden.json`` was recorded from ``FERL.predict_ds``
before the combination rules moved into the shared module, so these tests pin
that the move changed no output.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import _evidence
from ferl import FERL

GOLDEN = json.loads((Path(__file__).parent / "data" / "ferl_predict_ds_golden.json").read_text())
TIGHT = dict(rtol=1e-10, atol=1e-12)


@pytest.fixture(scope="module")
def models():
    X, y = np.array(GOLDEN["X_train"]), np.array(GOLDEN["y_train"])
    return {name: FERL(**spec["params"]).fit(X, y, **spec["fit_params"])
            for name, spec in GOLDEN["models"].items()}


def test_pinned_models_still_grow_the_same_trees(models):
    for name, spec in GOLDEN["models"].items():
        assert models[name].n_rules() == spec["n_rules"]


def _options(model, X_test, leaves_only, option):
    if option == "plain":
        return {}
    if option == "reliability_k":
        return {"reliability_k": 5.0}
    if option == "prior_strength":
        return {"prior_strength": 1.5}
    if option == "top_p":
        return {"top_p": 0.8}
    if option == "observed_mask":
        return {"observed_mask": np.array(GOLDEN["observed_mask"], dtype=bool)}
    names = model.node_activation_matrix(X_test)[2]
    count = sum(1 for n in names if not model._node_has_children(n)) if leaves_only else len(names)
    return {"reliability_vec": np.linspace(0.55, 0.95, count)}


@pytest.mark.parametrize(
    "case", GOLDEN["cases"],
    ids=[f"{c['model']}-{c['rule']}-{c['option']}-leaves{c['leaves_only']}" for c in GOLDEN["cases"]])
def test_predict_ds_matches_the_pre_refactor_outputs(models, case):
    model = models[case["model"]]
    X_test = np.array(GOLDEN["X_test"])
    kwargs = _options(model, X_test, case["leaves_only"], case["option"])
    result = model.predict_ds(X_test, leaves_only=case["leaves_only"], rule=case["rule"], **kwargs)
    for name, value in zip(("betp", "bel", "pl", "ignorance"), result):
        np.testing.assert_allclose(value, case[name], **TIGHT, err_msg=name)
    if "incremental_diag" in case:
        assert model._last_incremental_diag == case["incremental_diag"]


def test_unknown_rules_are_rejected_instead_of_falling_back_to_dempster(models):
    with pytest.raises(ValueError, match="rule"):
        models["compact"].predict_ds(np.array(GOLDEN["X_test"]), rule="yager")
    with pytest.raises(ValueError, match="rule"):
        _evidence.check_rule("ds")


def test_no_active_rules_means_total_ignorance():
    betp, belief, plausibility, ignorance, diagnostics = _evidence.combine_evidence(
        np.zeros((3, 0)), np.zeros((0, 2)), [], 2)
    assert np.allclose(betp, 0.5) and np.allclose(belief, 0.0)
    assert np.allclose(plausibility, 1.0) and np.allclose(ignorance, 1.0)
    assert diagnostics is None


def test_mixture_is_a_support_weighted_vote():
    M = np.array([[1.0, 0.0], [0.25, 0.75]])
    cons = np.array([[0.9, 0.1], [0.2, 0.8]])
    support = np.array([30.0, 10.0])
    _, belief, plausibility, ignorance, _ = _evidence.combine_evidence(
        M, cons, ["r_0", "r_1"], 2, rule="mixture", support=support, beta=10.0)
    expected = M @ ((support / (support + 10.0))[:, None] * cons)
    np.testing.assert_allclose(belief, expected)
    np.testing.assert_allclose(ignorance, 1.0 - expected.sum(axis=1))
    np.testing.assert_allclose(plausibility, expected + ignorance[:, None])


def test_top_p_routing_moves_the_tail_to_ignorance():
    M = np.array([[1.0]])
    cons = np.array([[0.7, 0.2, 0.1]])
    routed_M, routed_cons = _evidence.route_top_p(M, cons, 0.8)
    np.testing.assert_allclose(routed_cons, [[0.7 / 0.9, 0.2 / 0.9, 0.0]])
    np.testing.assert_allclose(routed_M, [[0.9]])
