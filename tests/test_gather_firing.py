"""Batched gathering must match the original ordered feature reductions."""
import numpy as np
import pytest

import fuzzy_sets as fs
import rules
import utils


@pytest.mark.parametrize('samples', [1, 180, 4097])
@pytest.mark.parametrize('features', [1, 7, 33])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_gather_matches_existing_firing(monkeypatch, samples, features, dtype):
    kind = fs.FUZZY_SETS.t2
    rng = np.random.default_rng(71)
    X = rng.normal(size=(samples, features))
    variables = utils.construct_partitions(X, kind)
    cls = rules.RuleBaseT1 if kind == fs.FUZZY_SETS.t1 else rules.RuleBaseT2
    bases = []
    for count in (0, 1, 23):
        ants = rng.integers(-1, 3, size=(count, features))
        if count:
            ants[0] = -1
        bases.append(cls(variables, [rules.RuleSimple(a) for a in ants]))
    base = rules.MasterRuleBase(bases)
    tail = (2,) if kind == fs.FUZZY_SETS.t2 else ()
    # Unequal term counts and strided sample arrays, including exact zeros.
    truth = [[rng.random((samples * 2,) + tail).astype(dtype)[::2]
              for _ in range(3 + i % 2)] for i in range(features)]
    truth[0][0][:] = 0.
    original = [[v.copy() for v in feature] for feature in truth]
    actual = rules._gather_rule_firing(bases, X, truth)
    assert actual is not None
    np.testing.assert_array_equal(base.compute_firing_strenghts(X, precomputed_truth=truth), actual)
    with monkeypatch.context() as reference:
        reference.setattr(rules, '_gather_rule_firing', lambda *args: None)
        expected = base.compute_firing_strenghts(X, precomputed_truth=truth)
    np.testing.assert_array_equal(actual, expected)
    for before, after in zip(original, truth):
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize('case', ['modifier', 'tnorm', 'override', 'empty', 't1'])
def test_gather_falls_back_for_unsupported_rules(case):
    X = np.arange(30).reshape(10, 3)
    kind = fs.FUZZY_SETS.t1 if case == 't1' else fs.FUZZY_SETS.t2
    variables = utils.construct_partitions(X, kind)
    rule = rules.RuleSimple([0, -1, 1])
    cls = rules.RuleBaseT1 if case == 't1' else rules.RuleBaseT2
    base = cls(variables, [rule])
    if case == 'modifier':
        rule.modifiers = [2, -1, .5]
    elif case == 'tnorm':
        base.tnorm = np.min
    elif case == 'override':
        base.compute_rule_antecedent_memberships = lambda *a, **kw: np.ones((len(X), 1))
    elif case == 'empty':
        base.rules = []
    truth = rules.compute_antecedents_memberships(variables, X)
    assert rules._gather_rule_firing([base], X, truth) is None


def _reference_firing(antecedents, truth, samples, tail):
    """The per-rule scratch-and-reduce firing the array kernel must reproduce."""
    result = np.zeros((samples, len(antecedents)) + tail)
    for jx, row in enumerate(antecedents):
        membership = np.zeros((samples, len(row)) + tail)
        used = 0
        for ix, term in enumerate(row):
            if term >= 0:
                membership[:, ix] = truth[ix][term]
                used += 1
            else:
                membership[:, ix] = 1.0
        if used == 0:
            membership[:, len(row) - 1] = 0.0
        result[:, jx] = np.prod(membership, axis=1)
    return result


@pytest.mark.parametrize('samples', [1, 180, 4097])
@pytest.mark.parametrize('features', [1, 7, 33])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('interval', [False, True])
def test_array_gather_kernel_matches_the_per_rule_reduction(samples, features,
                                                            dtype, interval):
    rng = np.random.default_rng(19)
    tail = (2,) if interval else ()
    truth = [[rng.random((samples * 2,) + tail).astype(dtype)[::2]
              for _ in range(3 + index % 2)] for index in range(features)]
    truth[0][0][:] = 0.  # a term that never fires
    antecedents = rng.integers(-1, 3, size=(23, features))
    antecedents[0] = -1  # a completely disabled rule
    antecedents = np.clip(antecedents,
                          -1, np.array([len(f) for f in truth]) - 1)
    actual = rules._gather_firing_from_arrays(antecedents, truth, samples, tail)
    assert actual is not None
    np.testing.assert_array_equal(
        actual, _reference_firing(antecedents, truth, samples, tail))


@pytest.mark.parametrize('interval', [False, True])
def test_array_gather_kernel_declines_unsupported_inputs(interval):
    tail = (2,) if interval else ()
    truth = [[np.ones((6,) + tail) for _ in range(2)] for _ in range(3)]
    good = np.zeros((4, 3), dtype=int)
    assert rules._gather_firing_from_arrays(good, truth, 6, tail) is not None
    # No rules, wrong feature count, out-of-range term, float antecedents and a
    # membership container that is not a sequence all fall back.
    assert rules._gather_firing_from_arrays(np.zeros((0, 3), dtype=int), truth, 6, tail) is None
    assert rules._gather_firing_from_arrays(np.zeros((4, 2), dtype=int), truth, 6, tail) is None
    assert rules._gather_firing_from_arrays(np.full((4, 3), 9), truth, 6, tail) is None
    assert rules._gather_firing_from_arrays(np.zeros((4, 3)), truth, 6, tail) is None
    assert rules._gather_firing_from_arrays(good, [object()] * 3, 6, tail) is None
    assert rules._gather_firing_from_arrays(good, [[np.ones(5)] * 2] * 3, 6, tail) is None


@pytest.mark.parametrize('samples', [1, 180, 4097])
@pytest.mark.parametrize('features', [1, 7, 33])
@pytest.mark.parametrize('interval', [False, True])
def test_packed_table_gather_matches_the_chunked_gather(samples, features, interval):
    rng = np.random.default_rng(37)
    tail = (2,) if interval else ()
    truth = [[rng.random((samples,) + tail) for _ in range(3 + index % 2)]
             for index in range(features)]
    truth[0][0][:] = 0.
    antecedents = rng.integers(-1, 3, size=(23, features))
    antecedents[0] = -1
    antecedents = np.clip(antecedents, -1, np.array([len(f) for f in truth]) - 1)
    packed = rules.pack_membership_table(truth, samples, tail)
    assert packed is not None
    expected = rules._gather_firing_from_arrays(antecedents, truth, samples, tail)
    actual = rules._gather_firing_from_arrays(antecedents, truth, samples, tail, packed)
    np.testing.assert_array_equal(actual, expected)
    # A rule subset must gather the same columns from the shared table.
    subset = antecedents[3:9]
    np.testing.assert_array_equal(
        rules._gather_firing_from_arrays(subset, truth, samples, tail, packed),
        expected[:, 3:9])


def test_pack_membership_table_declines_unsupported_or_oversized_input():
    truth = [[np.ones(50) for _ in range(3)] for _ in range(4)]
    assert rules.pack_membership_table(truth, 50, ()) is not None
    assert rules.pack_membership_table(truth, 50, (), max_bytes=8) is None
    assert rules.pack_membership_table([object()], 50, ()) is None
    assert rules.pack_membership_table([[np.ones(7)]], 50, ()) is None
    assert rules.pack_membership_table([], 50, ()) is None
    assert rules.pack_membership_table([[np.ones(50).astype('U3')]], 50, ()) is None
    # A packed table must be read-only: candidates share it for the whole fit.
    packed, _, _ = rules.pack_membership_table(truth, 50, ())
    assert not packed.flags.writeable


def test_packed_gather_declines_out_of_range_terms():
    truth = [[np.ones(20) for _ in range(2)] for _ in range(3)]
    packed = rules.pack_membership_table(truth, 20, ())
    assert rules._gather_firing_from_arrays(np.full((2, 3), 5), truth, 20, (), packed) is None
    assert rules._gather_firing_from_arrays(np.zeros((2, 2), dtype=int), truth, 20, (), packed) is None


@pytest.mark.parametrize('kind', [fs.FUZZY_SETS.t1, fs.FUZZY_SETS.t2])
def test_variable_packing_matches_membership_then_packing(kind):
    """Packing straight from the variables must equal the two-step reference."""
    rng = np.random.default_rng(53)
    X = rng.normal(size=(240, 5)) * 3
    variables = utils.construct_partitions(X, kind)
    tail = (2,) if kind == fs.FUZZY_SETS.t2 else ()
    truth = rules.compute_antecedents_memberships(variables, X)
    expected = rules.pack_membership_table(truth, 240, tail)
    actual = rules.pack_membership_table_from_variables(variables, X, tail)
    for left, right in zip(actual, expected):
        np.testing.assert_array_equal(left, right)


def test_variable_packing_handles_categorical_and_absent_domains():
    X = np.column_stack([np.arange(20.0), np.tile([0.0, 1.0, 2.0, 0.0], 5)])
    numeric = fs.fuzzyVariable('n', [fs.FS('a', [0, 3, 6, 9], (0.0, 19.0)),
                                     fs.FS('b', [6, 9, 15, 19], (0.0, 19.0))])
    categorical = fs.fuzzyVariable('c', [fs.categoricalFS(str(v), float(v))
                                         for v in range(3)])
    variables = [numeric, categorical]
    truth = rules.compute_antecedents_memberships(variables, X)
    expected = rules.pack_membership_table(truth, 20, ())
    actual = rules.pack_membership_table_from_variables(variables, X, ())
    for left, right in zip(actual, expected):
        np.testing.assert_array_equal(left, right)
    # A variable with no domain still packs, matching compute_memberships.
    undomained = fs.fuzzyVariable('u', [fs.FS('a', [0, 3, 6, 9], None)])
    single = X[:, :1]
    np.testing.assert_array_equal(
        rules.pack_membership_table_from_variables([undomained], single, ())[0],
        rules.pack_membership_table(
            rules.compute_antecedents_memberships([undomained], single), 20, ())[0])


def test_variable_packing_declines_oversized_and_mismatched_input():
    X = np.zeros((30, 2))
    variables = utils.construct_partitions(X + np.arange(30)[:, None], fs.FUZZY_SETS.t1)
    assert rules.pack_membership_table_from_variables(variables, X, ()) is not None
    assert rules.pack_membership_table_from_variables(variables, X, (), max_bytes=8) is None
    assert rules.pack_membership_table_from_variables(variables[:1], X, ()) is None
    assert rules.pack_membership_table_from_variables(variables, X, (2,)) is None
