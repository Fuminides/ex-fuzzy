"""Parity and optional-installation checks for FERL's compiled backend."""
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import load_iris, load_wine

from ferl import FERL
import _ferl_backend


def require_native():
    return pytest.importorskip('_ferl_kernels')


def test_default_backend_and_clone_need_no_extension(monkeypatch):
    def unavailable(*args, **kwargs):
        raise ImportError('extension unavailable')

    monkeypatch.setattr(_ferl_backend, 'import_module', unavailable)
    X, y = load_iris(return_X_y=True)
    model = FERL(max_rules=3, random_state=0).fit(X, y)
    assert clone(model).backend == 'python'
    assert clone(FERL(backend='cython')).backend == 'cython'
    with pytest.raises(ImportError, match='EX_FUZZY_BUILD_FERL=1'):
        FERL(backend='cython').fit(X, y)
    with pytest.raises(ValueError, match='backend'):
        FERL(backend='invalid').fit(X, y)


def test_native_vote_argmax_matches_numpy_at_ties_and_boundaries():
    kernels = require_native()
    rng = np.random.default_rng(12)
    for n, classes in [(0, 3), (1, 1), (150, 3), (200, 17)]:
        votes = rng.random((n, classes))
        membership = rng.random(n)
        probs = rng.dirichlet(np.ones(classes))
        if n:
            votes[0] = 0
            membership[0] = 0  # Exact tie selects first class.
        expected = np.argmax(votes + membership[:, None] * probs, axis=1)
        np.testing.assert_array_equal(
            kernels.added_vote_argmax(votes, membership, probs), expected)
    votes = np.array([[1.0, np.nextafter(1.0, 2.0)]])
    np.testing.assert_array_equal(
        kernels.added_vote_argmax(votes, np.zeros(1), np.ones(2)), [1])
    votes = np.array([[1.0, np.nan, np.nan], [np.nan, 1.0, 2.0]])
    np.testing.assert_array_equal(
        kernels.added_vote_argmax(votes, np.zeros(2), np.ones(3)),
        np.argmax(votes, axis=1))
    with pytest.raises(ValueError, match='shapes'):
        kernels.added_vote_argmax(votes, np.zeros(2), np.ones(2))
    with pytest.raises(ValueError, match='class'):
        kernels.added_vote_argmax(np.empty((1, 0)), np.ones(1), np.empty(0))


@pytest.mark.parametrize('load_data', [load_iris, load_wine])
@pytest.mark.parametrize('options', [
    {},
    {'partition': 'mdlp'},
    {'split_mode': 'learned', 'learned_width': 0.15},
    {'split_mode': 'learned', 'learned_n_boot': 3},
    {'sample_for_splits': True, 'sample_size': 60},
    {'coverage_weight': 0.2, 'multiway_splits': True},
    {'target_metric': 'purity'},
    {'consistent_cci': False, 'prediction_mode': 'winner'},
    {'prediction_mode': 'soft_gate'},
    {'ccp_alpha': 0.01},
])
def test_backends_preserve_tree_and_evidence(load_data, options):
    require_native()
    X, y = load_data(return_X_y=True)
    # Exercise labels that cannot be used as raw native array offsets.
    y = 10 + 3 * y
    models = [FERL(max_rules=6, min_improvement=0.0, random_state=42,
                   backend=backend, **options).fit(X, y, patience=4)
              for backend in ('python', 'cython')]
    python, native = models
    assert list(python.node_dict_access) == list(native.node_dict_access)
    for name in python.node_dict_access:
        left, right = python.node_dict_access[name], native.node_dict_access[name]
        assert left['prediction'] == right['prediction']
        np.testing.assert_array_equal(left['existing_membership'], right['existing_membership'])
    np.testing.assert_array_equal(python.predict(X), native.predict(X))
    np.testing.assert_array_equal(python.predict_proba(X), native.predict_proba(X))
    for left, right in zip(python.predict_credal(X[:20]), native.predict_credal(X[:20])):
        np.testing.assert_array_equal(left, right)
    mask = np.ones_like(X[:20], dtype=bool)
    mask[:, 0] = False
    np.testing.assert_array_equal(
        python.predict_proba(X[:20], observed_mask=mask),
        native.predict_proba(X[:20], observed_mask=mask))
    # Refit with a different dataset of the same shape: no backend cache leakage.
    native.fit(X[::-1], y[::-1], patience=4)
    fresh = clone(native).fit(X[::-1], y[::-1], patience=4)
    np.testing.assert_array_equal(native.predict_proba(X), fresh.predict_proba(X))
