"""Deterministic edge coverage for the rule and rule-base protocols."""

import numpy as np
import pytest

import fuzzy_sets as fs
import rules as rl
import utils


def _data():
    return np.array([[0.05], [0.5], [0.95]])


def _partitions(fuzzy_type=fs.FUZZY_SETS.t1):
    return utils.construct_partitions(_data(), fuzzy_type)


def test_gather_guards_reject_mismatched_rules_and_membership_payloads():
    X = np.column_stack([_data()[:, 0], _data()[::-1, 0]])
    t2 = utils.construct_partitions(X, fs.FUZZY_SETS.t2)
    base = rl.RuleBaseT2(t2, [rl.RuleSimple([0])])
    truth = [variable.compute_memberships(X[:, ix]) for ix, variable in enumerate(t2)]

    assert rl._gather_rule_firing([base], X, truth) is None
    assert rl._gather_firing_from_arrays(
        np.array([[0]]), [[np.array(['a', 'b', 'c'], dtype=object)]], 3, tail=()
    ) is None
    assert rl._gather_firing_from_arrays(np.array([[0]]), [[]], 3, tail=()) is None


def test_legacy_rule_membership_and_both_centroid_shapes():
    low = fs.FS('low', [0.0, 0.0, 0.4, 0.7], [0.0, 1.0])
    high = fs.FS('high', [0.3, 0.6, 1.0, 1.0], [0.0, 1.0])
    rule = rl.Rule([low, high], low)
    X = np.array([0.2, 0.8])

    membership = rule.membership(X)
    np.testing.assert_allclose(
        membership,
        low.membership(np.array([0.2])) * high.membership(np.array([0.8])),
    )
    np.testing.assert_allclose(rule.membership(X[None, :]), membership)
    assert rl._myprod(np.array([2.0]), np.array([3.0]))[0] == 6.0
    first_centroid = rule.consequent_centroid()
    assert np.isfinite(first_centroid)
    assert rule.consequent_centroid() == first_centroid

    interval = fs.IVFS(
        'interval', [0.0, 0.0, 0.3, 0.6], [0.0, 0.0, 0.5, 0.8], [0.0, 1.0]
    )
    interval_rule = rl.Rule([interval], interval)
    interval_centroid = interval_rule.consequent_centroid()
    assert np.asarray(interval_centroid).shape == (2,)


def test_direct_rulebase_protocol_mutation_scaling_and_failures(capsys):
    X = _data()
    t1 = _partitions()
    first = rl.RuleSimple([0])
    first.score = 0.8
    first.weight = 0.6
    second = rl.RuleSimple([1])
    second.score = 0.2
    second.weight = 0.4
    base = rl.RuleBase(t1, [first])

    assert base.get_rules() == [first]
    base.add_rule(second)
    base.add_rules([rl.RuleSimple([2])])
    base.remove_rule(2)
    np.testing.assert_array_equal(base.get_rulebase_matrix(), [[0], [1]])
    np.testing.assert_allclose(base.get_scores(), [0.8, 0.2])
    np.testing.assert_allclose(base.get_weights(), [0.6, 0.4])
    base.remove_rules([1])
    assert len(base) == 1
    replacement = rl.RuleSimple([1])
    base[0] = replacement
    assert base[0] is replacement
    assert list(base) == [replacement]
    assert 'Rule:' in str(base)
    assert base == rl.RuleBase(t1, [rl.RuleSimple([1])])
    assert isinstance(hash(base), int)
    assert base.n_linguistic_variables() == [3]
    combined = base + rl.RuleBase(t1, [rl.RuleSimple([2])])
    assert len(combined) == 2

    assert base.print_rules() is None
    assert 'IF ' in capsys.readouterr().out
    with pytest.raises(NotImplementedError):
        base.inference(X)
    with pytest.raises(NotImplementedError):
        base.forward(X)
    with pytest.raises(NotImplementedError):
        base.fuzzy_type()

    copied = base.copy()
    assert type(copied) is rl.RuleBase
    assert copied is not base and copied.rules is not base.rules

    missing_modifiers = rl.RuleSimple([0])
    del missing_modifiers.modifiers
    custom = rl.RuleBaseT1(t1, [missing_modifiers], tnorm=lambda values, axis: np.min(values, axis=axis))
    assert custom.compute_rule_antecedent_memberships(X).shape == (3, 1)

    scaled_t1 = rl.RuleBaseT1(t1, [rl.RuleSimple([0]), rl.RuleSimple([1])])
    firing_t1 = scaled_t1.compute_rule_antecedent_memberships(X, scaled=True)
    np.testing.assert_allclose(firing_t1.sum(axis=1), [1.0, 1.0, 0.0])

    low_score = rl.RuleSimple([0])
    low_score.score = 0.0
    pruned = rl.RuleBaseT1(t1, [low_score])
    pruned.prune_bad_rules(0.1)
    assert len(pruned) == 0

    unscored = rl.RuleBaseT1(t1, [rl.RuleSimple([0])])
    with pytest.raises(AssertionError, match='Dominance scores'):
        unscored.prune_bad_rules()


def test_type2_and_gt2_scaling_empty_memberships_and_copy_protocols():
    X = _data()
    t2 = _partitions(fs.FUZZY_SETS.t2)
    custom_t2 = rl.RuleBaseT2(
        t2,
        [rl.RuleSimple([0]), rl.RuleSimple([1])],
        tnorm=lambda values, axis: np.prod(values, axis=axis),
    )
    firing_t2 = custom_t2.compute_rule_antecedent_memberships(X, scaled=True)
    np.testing.assert_allclose(
        firing_t2.sum(axis=1), [[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]]
    )
    assert isinstance(custom_t2.copy(), rl.RuleBaseT2)

    gt2 = _partitions(fs.FUZZY_SETS.gt2)
    empty_gt2 = rl.RuleBaseGT2(gt2, [])
    assert empty_gt2.compute_antecedents_memberships(X)[0].shape[2] == 2
    gt2_base = rl.RuleBaseGT2(gt2, [rl.RuleSimple([0]), rl.RuleSimple([1])])
    alpha_firing = gt2_base.alpha_compute_rule_antecedent_memberships(X, scaled=True)
    assert alpha_firing.shape[2:] == (len(gt2_base.alpha_cuts), 2)
    assert gt2_base.compute_rule_antecedent_memberships(X).shape == (3, 2, 2)
    assert isinstance(gt2_base.copy(), rl.RuleBaseGT2)
    with pytest.raises(NotImplementedError):
        gt2_base.inference(X)
    with pytest.raises(NotImplementedError):
        gt2_base.forward(X)


def test_t1_and_t2_rulebase_regression_inference_and_copy():
    X = _data()
    t1 = _partitions()
    output_t1 = fs.fuzzyVariable(
        'output',
        [
            fs.FS('low', [0.0, 0.0, 0.3, 0.6], [0.0, 1.0]),
            fs.FS('high', [0.4, 0.7, 1.0, 1.0], [0.0, 1.0]),
        ],
    )
    t1_base = rl.RuleBaseT1(
        t1, [rl.RuleSimple([0], consequent=0), rl.RuleSimple([2], consequent=1)], output_t1
    )
    assert t1_base.inference(X).shape == (3,)
    np.testing.assert_allclose(t1_base.forward(X), t1_base.inference(X), equal_nan=True)
    assert isinstance(t1_base.copy(), rl.RuleBaseT1)

    t2 = _partitions(fs.FUZZY_SETS.t2)
    output_t2 = fs.fuzzyVariable(
        'output',
        [
            fs.IVFS('low', [0.0, 0.0, 0.2, 0.5], [0.0, 0.0, 0.4, 0.7], [0.0, 1.0]),
            fs.IVFS('high', [0.5, 0.7, 1.0, 1.0], [0.3, 0.6, 1.0, 1.0], [0.0, 1.0]),
        ],
    )
    t2_base = rl.RuleBaseT2(
        t2, [rl.RuleSimple([0], consequent=0), rl.RuleSimple([2], consequent=1)], output_t2
    )
    assert t2_base.inference(X).shape == (3, 2)
    assert t2_base.forward(X).shape == (3,)


def _classification_master(fuzzy_type=fs.FUZZY_SETS.t1, *, allow_unknown=True, ds_mode=0):
    partitions = _partitions(fuzzy_type)
    base_class = {
        fs.FUZZY_SETS.t1: rl.RuleBaseT1,
        fs.FUZZY_SETS.t2: rl.RuleBaseT2,
        fs.FUZZY_SETS.gt2: rl.RuleBaseGT2,
    }[fuzzy_type]
    rule = rl.RuleSimple([0])
    rule.score = 1.0
    rule.weight = 0.75
    return rl.MasterRuleBase(
        [base_class(partitions, [rule]), base_class(partitions, [])],
        consequent_names=['low', 'other'], ds_mode=ds_mode, allow_unknown=allow_unknown,
    )


def test_master_rulebase_empty_predictions_mutation_and_accessors(capsys):
    with pytest.raises(rl.RuleError, match='No rule bases'):
        rl.MasterRuleBase([])

    partitions = _partitions()
    master = rl.MasterRuleBase([rl.RuleBaseT1(partitions, []), rl.RuleBaseT1(partitions, [])])
    X = _data()
    np.testing.assert_array_equal(master.compute_firing_strenghts(X), np.empty((3, 0)))
    np.testing.assert_array_equal(master.compute_association_degrees(X), np.empty((3, 0)))
    winners, strengths = master._winning_rules(X)
    np.testing.assert_array_equal(winners, [-1, -1, -1])
    np.testing.assert_array_equal(strengths, [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(master.winning_rule_predict(X), [-1, -1, -1])
    assert master.winning_rule_predict(X, out_class_names=True) == ['Unknown'] * 3

    explained = master.explainable_predict(X)
    np.testing.assert_array_equal(explained[0], [-1, -1, -1])
    explained_names = master.explainable_predict(X, out_class_names=True)
    np.testing.assert_array_equal(explained_names[0], ['Unknown'] * 3)
    assert master.get_scores().size == 0
    assert master.get_weights().size == 0

    rule = rl.RuleSimple([0])
    master.add_rule(rule, 0)
    assert master.get_rulebase_matrix()[0].shape == (1, 1)
    master.remove_rule(0, 0)
    assert len(master.get_rules()) == 0
    added = rl.RuleBaseT1(partitions, [])
    master.add_rule_base(added)
    assert master.get_consequents_names() == [0, 1, 2]
    master.rename_cons(['a', 'b', 'c'])
    assert master.get_rulebases()[-1] is added
    assert master.get_antecedents() is partitions
    assert master.n_linguistic_variables() == [3]
    assert master.print_rules() is None
    assert 'Rules for consequent' in capsys.readouterr().out
    assert 'Rules for consequent' in str(master)
    assert master == master.copy(deep=False)
    assert master == master.copy()
    assert master != object()


def test_master_named_known_and_unknown_predictions_and_confidence_intervals():
    master = _classification_master()
    X = np.array([[0.05], [0.95]])
    rule = master.get_rules()[0]
    rule.boot_confidence_interval = np.array([0.7, 0.9])

    np.testing.assert_array_equal(master.winning_rule_predict(X), [0, -1])
    np.testing.assert_array_equal(
        master.winning_rule_predict(X, out_class_names=True), ['low', 'Unknown']
    )
    named, winners, association, confidence = master.explainable_predict(
        X, out_class_names=True
    )
    np.testing.assert_array_equal(named, ['low', 'Unknown'])
    np.testing.assert_array_equal(winners, [0, -1])
    assert association.shape == (2, 1)
    assert confidence.shape == (2, 2)
    np.testing.assert_array_equal(master(X), master.predict(X))

    ds_one = _classification_master(ds_mode=1)
    assert ds_one.compute_association_degrees(X).shape == (2, 1)
    ds_two_t1 = _classification_master(ds_mode=2)
    assert ds_two_t1.compute_association_degrees(X).shape == (2, 1)
    t2 = _classification_master(fs.FUZZY_SETS.t2, ds_mode=2)
    assert t2.compute_association_degrees(X).shape == (2, 1)
    assert t2.explainable_predict(X)[2].ndim == 2


@pytest.mark.parametrize('fuzzy_type', [fs.FUZZY_SETS.t1, fs.FUZZY_SETS.t2, fs.FUZZY_SETS.gt2])
def test_construct_rule_base_supports_each_fuzzy_type(fuzzy_type):
    antecedents = _partitions(fuzzy_type)
    matrix = np.array([[0], [-1], [2]])
    consequents = np.array([0, 0, 1])
    weights = np.array([0.7, 0.1, 0.8])

    master = rl.construct_rule_base(
        matrix, 2, consequents, antecedents, weights, class_names=['a', 'b']
    )

    assert master.fuzzy_type() == fuzzy_type
    assert master.get_consequents_names() == ['a', 'b']
    assert len(master.get_rules()) == 2
    if fuzzy_type == fs.FUZZY_SETS.t1:
        assert rl.construct_rule_base(matrix, 2, consequents, antecedents, weights)


def test_rule_string_formats_modifiers_metrics_and_missing_attributes():
    antecedents = _partitions()
    rule = rl.RuleSimple([0], modifiers=np.array([1.0]))
    rule.score = 0.7
    rule.p_value_class_structure = 0.0005
    rule.p_value_feature_coalitions = 0.005
    rule.boot_p_value = 0.02
    rule.boot_confidence_interval = [0.5, 0.8]
    rule.boot_support_interval = [0.4, 0.9]
    text = rl.generate_rule_string(rule, antecedents)
    assert '***' in text and '**' in text and '*' in text
    assert ', ACC ' not in text

    known_modifier = rl.RuleSimple([0], modifiers=np.array([2.0]))
    assert '(MOD Very)' in rl.generate_rule_string(known_modifier, antecedents)
    numeric_modifier = rl.RuleSimple([0], modifiers=np.array([1.25]))
    assert '(MOD 1.25)' in rl.generate_rule_string(numeric_modifier, antecedents)

    no_score = rl.RuleSimple([0], consequent=2)
    assert 'THEN consequent vl is 2' in rl.generate_rule_string(no_score, antecedents)

    class MinimalRule:
        def __getitem__(self, index):
            return 0

    assert rl.generate_rule_string(MinimalRule(), antecedents).startswith('IF ')


def test_bootstrap_result_printers(capsys):
    master = _classification_master()
    rule = master.get_rules()[0]
    rule.boot_confidence_interval = [0.6, 0.8]
    rule.boot_support_interval = [0.4, 0.7]

    master.print_rule_bootstrap_results()

    output = capsys.readouterr().out
    assert 'Rules for consequent: low' in output
    assert 'Confidence: [0.6, 0.8]' in output
