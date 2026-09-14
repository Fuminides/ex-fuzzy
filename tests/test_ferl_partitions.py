"""Tests for the supervised MDLP fuzzy partitions."""

import numpy as np
import pytest

import ferl_partitions as fp


def test_term_names():
    assert fp._term_name(0, 1) == 'all'
    assert [fp._term_name(k, 3) for k in range(3)] == ['low_0', 'med_1', 'high_2']
    assert fp._term_name(6, 7) == 'term_6'


def test_mdlp_cuts_between_classes():
    x = np.r_[np.linspace(0, 1, 30), np.linspace(2, 3, 30)]
    y = np.r_[np.zeros(30), np.ones(30)]
    assert fp.mdlp_cuts(x, y) == [pytest.approx(1.5)]


def test_mdlp_has_nothing_to_cut():
    # A single sample, samples of one class, and equal values that cannot be split.
    assert fp.mdlp_cuts([1.0], [0]) == []
    assert fp.mdlp_cuts(np.arange(6.0), [1] * 6) == []
    assert fp.mdlp_cuts([2.0, 2.0, 2.0, 2.0], [0, 1, 0, 1]) == []


def test_partitions_for_informative_constant_and_noisy_features():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(7), 20)
    informative = y + rng.uniform(0, 0.5, len(y))
    constant = np.full(len(y), 3.0)
    noise = rng.uniform(size=len(y))
    X = np.column_stack([informative, constant, noise])

    variables = fp.learn_partitions_mdlp(X, y)
    assert [variable.name for variable in variables] == ['feature_0', 'feature_1', 'feature_2']
    assert variables[0].linguistic_variable_names() == [f'term_{k}' for k in range(7)]
    # A constant feature keeps one set over a tiny non-empty domain.
    assert variables[1].linguistic_variable_names() == ['all']
    assert np.diff(variables[1][0].domain)[0] == pytest.approx(1e-6)
    # An uninformative feature is split at its median.
    assert len(variables[2]) == 2

    without_fallback = fp.learn_partitions_mdlp(X[:, 2:], y, fallback_median=False)
    assert without_fallback[0].linguistic_variable_names() == ['all']
