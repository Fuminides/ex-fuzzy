"""Tests for temporal fuzzy sets, rule bases and the temporal classifier."""

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

import fuzzy_sets as fs
import rules as rl
import temporal
import utils


CONDITIONAL = np.array([1.0, 0.25])


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


@pytest.fixture(scope='module')
def iris():
    X, y = load_iris(return_X_y=True)
    return X, y, np.arange(len(y)) % 2


def _temporal_variables(X, fuzzy_type):
    lvs = utils.construct_partitions(X, fuzzy_type)
    return [temporal.temporalFuzzyVariable(lv.name, [temporal.temporalFS(fuzzy_set, CONDITIONAL) for fuzzy_set in lv.linguistic_variables])
            for lv in lvs]


@pytest.fixture(scope='module')
def fitted(iris):
    X, y, time_moments = iris
    np.random.seed(0)
    model = temporal.TemporalFuzzyRulesClassifier(nRules=6, nAnts=2, linguistic_variables=_temporal_variables(X, fs.FUZZY_SETS.t1))
    model.fit(X, y, n_gen=4, pop_size=10, time_moments=time_moments)
    return model


def test_legacy_enums():
    assert [member.value for member in temporal.TMP_FUZZY_SETS] == [1, 2, 3]
    assert temporal.NEW_FUZZY_SETS is fs.FUZZY_SETS


def test_temporal_fuzzy_set_scales_memberships_by_time():
    base = fs.FS('Low', [0, 0, 0.3, 0.6], [0, 1])
    temporal_set = temporal.temporalFS(base, CONDITIONAL)
    x = np.array([0.1, 0.45, 0.9])

    assert temporal_set.name == 'Low'
    assert temporal_set.type() == fs.FUZZY_SETS.temporal
    assert temporal_set.inside_type() == fs.FUZZY_SETS.t1
    assert temporal_set.membership_parameters == base.membership_parameters
    np.testing.assert_allclose(temporal_set.membership(x, 1), base.membership(x) * 0.25)

    with pytest.raises(AssertionError, match='no fixed time'):
        temporal_set.membership(x)
    temporal_set.fix_time(0)
    np.testing.assert_allclose(temporal_set.membership(x), base.membership(x))


def test_temporal_fuzzy_sets_keep_type_2_parameters():
    iv_set = fs.IVFS('Low', [0, 0, 0.3, 0.5], [0, 0, 0.4, 0.6], [0, 1])
    temporal_iv = temporal.temporalFS(iv_set, CONDITIONAL)
    assert temporal_iv.inside_type() == fs.FUZZY_SETS.t2
    assert temporal_iv.secondMF_lower == iv_set.secondMF_lower
    assert temporal_iv.secondMF_upper == iv_set.secondMF_upper

    gt2_set = utils.construct_partitions(np.linspace(0, 1, 20).reshape(-1, 1), fs.FUZZY_SETS.gt2)[0][0]
    temporal_gt2 = temporal.temporalFS(gt2_set, CONDITIONAL)
    assert temporal_gt2.inside_type() == fs.FUZZY_SETS.gt2
    assert temporal_gt2.alpha_cuts is gt2_set.alpha_cuts
    assert temporal_gt2.secondary_memberships is gt2_set.secondary_memberships


def test_temporal_fuzzy_variable():
    sets = [temporal.temporalFS(fs.FS(name, params, [0, 1]), CONDITIONAL)
            for name, params in [('Low', [0, 0, 0.3, 0.6]), ('High', [0.4, 0.7, 1, 1])]]
    variable = temporal.temporalFuzzyVariable('x', sets)
    x = np.array([0.2, 0.8])

    assert variable.fs_type == fs.FUZZY_SETS.temporal
    assert variable.n_time_moments() == 2
    with pytest.raises(AssertionError, match='no fixed time'):
        variable.compute_memberships(x)

    at_time_1 = variable.compute_memberships(x, 1)
    variable.fix_time(1)
    for explicit, fixed in zip(at_time_1, variable.compute_memberships(x)):
        np.testing.assert_allclose(explicit, fixed)
    np.testing.assert_allclose(at_time_1[0], sets[0].std_set.membership(x) * 0.25)


def test_temporal_variables_from_conditional_frequencies(iris):
    X, _, time_moments = iris
    partitions = utils.construct_partitions(X, fs.FUZZY_SETS.t1)

    variables = utils.create_tempVariables(X, time_moments, partitions)
    assert [variable.name for variable in variables] == [partition.name for partition in partitions]
    assert all(variable.n_time_moments() == 2 for variable in variables)

    per_moment = utils.create_multi_tempVariables(X, time_moments, fs.FUZZY_SETS.t1)
    assert len(per_moment) == 2
    assert all(len(variables) == X.shape[1] for variables in per_moment)


def test_classifier_needs_temporal_linguistic_variables(iris):
    X, y, time_moments = iris
    with pytest.raises(ValueError, match='temporal linguistic variables'):
        temporal.TemporalFuzzyRulesClassifier(nRules=4).fit(X, y, n_gen=1, pop_size=4, time_moments=time_moments)


def test_fitted_classifier_predicts_per_time_moment(iris, fitted):
    X, y, time_moments = iris
    rule_base = fitted.rule_base

    assert set(fitted.performance) == {0, 1}
    assert len(rule_base) == 2
    assert rule_base.time_step_names == ['0', '1']
    assert rule_base.fuzzy_type() == fs.FUZZY_SETS.t1

    predictions = fitted.forward(X, time_moments)
    assert predictions.shape == y.shape
    assert np.mean(predictions == y) > 0.5
    np.testing.assert_array_equal(fitted.forward(pd.DataFrame(X), time_moments), predictions)

    names = rule_base.winning_rule_predict(X, time_moments, out_class_names=True)
    assert set(names) <= {0, 1, 2, 'Unknown'}
    assert fitted.eval_performance.association_degree().shape == (len(y), len(rule_base.get_rules()))

    # A rule only fires for the samples of its own time moment.
    firing = rule_base.compute_firing_strenghts(X, time_moments)
    rules_at_time_0 = sum(len(rb) for rb in rule_base[0])
    assert np.all(firing[time_moments == 1, :rules_at_time_0] == 0)
    assert np.all(firing[time_moments == 0, rules_at_time_0:] == 0)


def test_temporal_master_rule_base_accessors(fitted):
    rule_base = fitted.rule_base

    n_rules = len(rule_base.get_rules())
    assert n_rules == sum(len(rb) for mrb in rule_base.time_mrule_bases for rb in mrb)
    assert len(rule_base.get_scores()) == n_rules
    assert len(rule_base.get_rulebases()) == 6
    assert rule_base[1] is rule_base.time_mrule_bases[1]
    assert len(rule_base.get_consequents()) == n_rules
    assert len(fitted.get_rulebase()) == 6

    text = rule_base.print_rules(return_rules=True)
    assert text.count('Rules for time step') == 2
    assert 'Consequent: 2' in text


def test_temporal_master_rule_base_editing(iris):
    X, y, time_moments = iris
    lvs = _temporal_variables(X, fs.FUZZY_SETS.t1)

    def master(rule_lists):
        return rl.MasterRuleBase([rl.RuleBaseT1(lvs, rule_list) for rule_list in rule_lists])

    rule_base = temporal.temporalMasterRuleBase([master([[], []]), master([[], []])], time_step_names=['morning', 'evening'])
    assert rule_base.time_step_names == ['morning', 'evening']
    assert rule_base.compute_firing_strenghts(X, time_moments).shape == (len(y), 0)
    assert len(rule_base.get_scores()) == 0
    winners, degrees = rule_base._winning_rules(X, time_moments)
    assert np.all(winners == -1) and np.all(degrees == 0)

    rule = rl.RuleSimple([-1, -1, 0, -1])
    rule.score = 0.5
    rule_base.add_rule(rule, consequent=0, time_step=1)
    rule_base.add_rule_base(rl.RuleBaseT1(lvs, []), time=0)
    assert len(rule_base[0]) == 3
    # Rule bases are listed per time moment: three at time 0, then the one holding the new rule.
    assert rule_base.get_rulebase_matrix()[3].shape == (1, 4)

    # Time moment 0 has no rules: its samples fall back to the first rule unless unknowns are allowed.
    predictions = rule_base.winning_rule_predict(X, time_moments)
    assert np.all(predictions == 0)
    rule_base.allow_unknown = True
    predictions = rule_base.winning_rule_predict(X, time_moments)
    assert np.all(predictions[time_moments == 0] == -1)
    assert set(predictions[time_moments == 1]) == {-1, 0}

    rule_base.time_mrule_bases[0].consequent_names = ['only one name']
    text = rule_base.print_rules(return_rules=True)
    assert 'only one name' in text

    rule_base.purge_rules(tolerance=0.9)
    assert len(rule_base.get_rules()) == 0


def test_print_rules_goes_to_stdout(fitted, capsys):
    assert fitted.rule_base.print_rules() is None
    assert 'Rules for time step: 1' in capsys.readouterr().out


@pytest.mark.parametrize('fuzzy_type', [fs.FUZZY_SETS.t2, fs.FUZZY_SETS.gt2])
def test_type_2_temporal_classifier(iris, fuzzy_type):
    X, y, time_moments = iris
    np.random.seed(0)
    model = temporal.TemporalFuzzyRulesClassifier(nRules=6, nAnts=2, linguistic_variables=_temporal_variables(X, fuzzy_type))
    model.fit(X, y, n_gen=3, pop_size=8, time_moments=time_moments)

    assert model.rule_base.fuzzy_type() == fuzzy_type
    predictions = model.forward(X, time_moments)
    assert np.mean(predictions == y) > 0.4


def test_fit_with_checkpoints_writes_rules(iris, tmp_path, monkeypatch, capsys):
    X, y, time_moments = iris
    monkeypatch.chdir(tmp_path)
    np.random.seed(0)
    model = temporal.TemporalFuzzyRulesClassifier(nRules=4, nAnts=2, verbose=True,
                                                  linguistic_variables=_temporal_variables(X, fs.FUZZY_SETS.t1))
    model.fit(X, y, n_gen=2, pop_size=6, time_moments=time_moments, checkpoints=1)

    checkpoints = sorted(tmp_path.iterdir())
    assert [path.name.rsplit('_', 1)[0] for path in checkpoints] == ['checkpoint_0', 'checkpoint_0', 'checkpoint_1', 'checkpoint_1']
    assert all(path.read_text() for path in checkpoints)
    output = capsys.readouterr().out
    assert 'n_gen' in output and 'Rule based fit for time 1 completed.' in output


def test_fit_on_a_dataframe_with_sparse_checkpoints(iris, tmp_path, monkeypatch, capsys):
    X, y, time_moments = iris
    monkeypatch.chdir(tmp_path)
    frame = pd.DataFrame(X, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    np.random.seed(0)
    model = temporal.TemporalFuzzyRulesClassifier(nRules=4, nAnts=2, linguistic_variables=_temporal_variables(X, fs.FUZZY_SETS.t1))
    model.fit(frame, y, n_gen=3, pop_size=6, time_moments=time_moments, checkpoints=2)

    assert model.var_names == list(frame.columns)
    # Generations 0 and 2 are saved for each of the two time moments, silently.
    assert len(list(tmp_path.iterdir())) == 4
    assert capsys.readouterr().out == ''


def test_evaluation_report(iris, fitted, capsys):
    X, y, time_moments = iris

    rules_text = temporal.eval_temporal_fuzzy_model(fitted, X, y, X, y, time_moments, time_moments,
                                                    plot_rules=True, plot_partitions=True, print_rules=False, return_rules=True)
    report = capsys.readouterr().out
    assert 'Rules for time step: 0' in rules_text
    assert 'ACCURACY' in report and 'MOMENT 1' in report
    assert 'Rules for time step' not in report

    frame = pd.DataFrame(X)
    assert temporal.eval_temporal_fuzzy_model(fitted, frame, y, frame, y, time_moments, time_moments, plot_rules=False,
                                              plot_partitions=False, print_accuracy=False, print_matthew=False) is None
    report = capsys.readouterr().out
    assert 'Rules for time step: 1' in report and 'ACCURACY' not in report

    assert temporal.eval_temporal_fuzzy_model(fitted, X, y, X, y, time_moments, time_moments, plot_rules=False,
                                              plot_partitions=False, print_rules=False, print_accuracy=False,
                                              print_matthew=False) is None
    assert capsys.readouterr().out == ''
