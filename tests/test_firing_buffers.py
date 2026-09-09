"""Exact parity with the original per-rule allocation and ordered reduction."""
import numpy as np
import pytest

import fuzzy_sets as fs
import rules
import utils


def reference_firing(base, X, truth):
    """Independent copy of the original T1/T2 firing algorithm."""
    tail = (2,) if base.fuzzy_type() == fs.FUZZY_SETS.t2 else ()
    result = np.zeros((len(X), len(base.rules)) + tail)
    for j, rule in enumerate(base.rules):
        membership = np.zeros((len(X), len(rule.antecedents)) + tail)
        active = 0
        for i, term in enumerate(rule.antecedents):
            if term >= 0:
                values = list(truth[i])[term]
                if rule.modifiers is not None and rule.modifiers[i] != -1:
                    values = values ** rule.modifiers[i]
                membership[:, i] = values
                active += 1
            else:
                membership[:, i] = 1.0
        if active == 0:
            membership[:, i] = 0.0
        result[:, j] = base.tnorm(membership, axis=1)
    return result


@pytest.mark.parametrize('kind', [fs.FUZZY_SETS.t1, fs.FUZZY_SETS.t2])
@pytest.mark.parametrize('tnorm', [np.prod, np.min])
@pytest.mark.parametrize('layout', ['C', 'F', 'strided'])
@pytest.mark.parametrize('container', [list, tuple, np.asarray])
def test_firing_matches_original(kind, tnorm, layout, container):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(67, 7))
    if layout == 'F':
        X = np.asfortranarray(X)
    elif layout == 'strided':
        X = X[::2]
    variables = utils.construct_partitions(X, kind)
    # Unequal term counts, all don't-cares, duplicates and varied modifiers.
    variables[0].linguistic_variables = variables[0].linguistic_variables[:2]
    antecedents = rng.integers(-1, 2, size=(23, 7))
    antecedents[0] = -1
    antecedents[1] = 0
    antecedents[2] = antecedents[1]
    rule_list = [rules.RuleSimple(a, modifiers=rng.choice([-1, .5, 1, 2, 3], 7))
                 for a in antecedents]
    cls = rules.RuleBaseT1 if kind == fs.FUZZY_SETS.t1 else rules.RuleBaseT2
    base = cls(variables, rule_list, tnorm=tnorm)
    truth = [container(t) for t in rules.compute_antecedents_memberships(variables, X)]
    before = [[v.copy() for v in t] for t in truth]
    np.testing.assert_array_equal(
        base.compute_rule_antecedent_memberships(X, antecedents_memberships=truth),
        reference_firing(base, X, truth))
    for old, current in zip(before, truth):
        np.testing.assert_array_equal(old, current)
    # A second call must not reuse stale data or rule metadata.
    base.rules[1].antecedents[0] = -1
    changed = [container([v * .7 for v in t]) for t in truth]
    np.testing.assert_array_equal(
        base.compute_rule_antecedent_memberships(X, antecedents_memberships=changed),
        reference_firing(base, X, changed))


def test_custom_tnorm_receives_independent_buffers():
    X = np.arange(24).reshape(8, 3)
    captured = []

    def tnorm(values, axis):
        captured.append(values)
        return np.prod(values, axis=axis)

    base = rules.RuleBaseT1(utils.construct_partitions(X),
                            [rules.RuleSimple([0, -1, 1]), rules.RuleSimple([1, 1, -1])],
                            tnorm=tnorm)
    base.compute_rule_antecedent_memberships(X)
    assert len(captured) == 2
    assert not np.shares_memory(*captured)
