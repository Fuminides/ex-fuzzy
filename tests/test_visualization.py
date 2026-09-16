"""Tests for the rule and fuzzy partition visualization module (vis_rules.py)."""
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

from ex_fuzzy import fuzzy_sets as fs
from ex_fuzzy import rules as rl
from ex_fuzzy import utils
from ex_fuzzy import vis_rules


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


@pytest.fixture(scope='module')
def iris():
    X, y = load_iris(return_X_y=True)
    return X, y


@pytest.fixture(scope='module')
def partitions(iris):
    return utils.construct_partitions(iris[0], fs.FUZZY_SETS.t1)


def _rule(antecedents, score=0.5, accuracy=0.75):
    rule = rl.RuleSimple(antecedents)
    rule.score = score
    rule.accuracy = accuracy
    return rule


@pytest.fixture
def master_rule_base(partitions):
    class_0 = rl.RuleBaseT1(partitions, [_rule([0, -1, 0, -1]), _rule([0, 1, -1, -1]), _rule([-1, 2, 1, 2])])
    class_1 = rl.RuleBaseT1(partitions, [_rule([1, -1, 1, -1], score=0.25)])
    return rl.MasterRuleBase([class_0, class_1])


def _legend_labels():
    return [text.get_text() for text in plt.gcf().axes[0].get_legend().get_texts()]


@pytest.mark.parametrize('fuzzy_type', [fs.FUZZY_SETS.t1, fs.FUZZY_SETS.t2])
def test_plot_partitions(iris, fuzzy_type):
    variable = utils.construct_partitions(iris[0], fuzzy_type)[0]
    vis_rules.plot_fuzzy_variable(variable)

    assert _legend_labels() == variable.linguistic_variable_names()
    assert plt.gcf().axes[0].get_title() == variable.name


def test_plot_general_type_2_partition(iris):
    variable = utils.construct_partitions(iris[0], fs.FUZZY_SETS.gt2)[0]
    vis_rules.plot_fuzzy_variable(variable)

    axes = plt.gcf().axes[0]
    assert axes.name == '3d'
    # Each set draws one line per secondary membership.
    n_lines = sum(len(fuzzy_set.secondary_memberships) for fuzzy_set in variable.linguistic_variables)
    assert len(axes.get_lines()) == n_lines


def test_plot_gaussian_sets_with_units():
    variable = fs.fuzzyVariable('speed', [fs.gaussianFS('Slow', [0.2, 0.1], [0, 1]), fs.gaussianFS('Fast', [0.8, 0.1], [0, 1])], units='m/s')
    vis_rules.plot_fuzzy_variable(variable)

    axes = plt.gcf().axes[0]
    assert axes.get_xlabel() == 'm/s'
    assert _legend_labels() == ['Slow', 'Fast']
    assert len(axes.get_lines()[0].get_xdata()) == 101


def test_plot_many_sets_cycles_colors_and_skips_sets_without_domain():
    sets = [fs.FS(f'set {ix}', [ix / 10, ix / 10 + 0.05, ix / 10 + 0.1, ix / 10 + 0.15], [0, 1]) for ix in range(6)]
    sets.append(fs.triangularFS('triangle', [0.2, 0.3, 0.3, 0.4], [0, 1]))
    sets.append(fs.FS('no domain', [0, 0.1, 0.2, 0.3]))
    vis_rules.plot_fuzzy_variable(fs.fuzzyVariable('x', sets))

    lines = plt.gcf().axes[0].get_lines()
    assert _legend_labels() == [f'set {ix}' for ix in range(6)] + ['triangle']
    assert lines[5].get_color() == lines[0].get_color()


def test_plot_interval_type_2_shapes():
    distinct = fs.IVFS('distinct', [0.1, 0.2, 0.3, 0.4], [0.05, 0.15, 0.35, 0.45], [0, 1])
    shared = fs.IVFS('shared', [0.5, 0.6, 0.7, 0.8], [0.5, 0.6, 0.7, 0.9], [0, 1], lower_height=0.8)
    vis_rules.plot_fuzzy_variable(fs.fuzzyVariable('x', [distinct, shared]))
    assert _legend_labels() == ['distinct', 'shared']
    plt.close('all')

    gaussian = fs.gaussianIVFS('bell', [0.5, 0.1], [0.5, 0.2], [0, 1], lower_height=0.9)
    vis_rules.plot_fuzzy_variable(fs.fuzzyVariable('y', [gaussian]))
    assert _legend_labels() == ['bell']


def test_plot_categorical_variable_as_a_table():
    variable = fs.fuzzyVariable('colour', [fs.categoricalFS('red', 'red'), fs.categoricalFS('blue', 'blue')])
    vis_rules.plot_fuzzy_variable(variable)

    axes = plt.gcf().axes[0]
    assert axes.get_title() == 'colour'
    cells = [cell.get_text().get_text() for cell in axes.tables[0].get_celld().values()]
    assert {'Categories', 'red', 'blue'} <= set(cells)


def test_rule_matrix_views(master_rule_base, partitions):
    frame = vis_rules.matrix_rule_base_form(master_rule_base[0])
    assert list(frame.columns) == [partition.name for partition in partitions]
    np.testing.assert_array_equal(frame.to_numpy(), [[0, -1, 0, -1], [0, 1, -1, -1], [-1, 2, 1, 2]])

    filtered = vis_rules.filter_useless_columns(vis_rules.matrix_rule_base_form(master_rule_base[1]))
    assert list(filtered.columns) == [partitions[0].name, partitions[2].name]


def test_rules_to_latex(master_rule_base, capsys):
    latex = vis_rules.rules_to_latex(master_rule_base)

    assert capsys.readouterr().out.strip() == latex
    lines = latex.split('\n')
    assert lines[0] == '\\begin{tabular}{ccccc|cc}'
    assert lines[-1] == '\\end{tabular}'
    assert '\\multirow{3}{*}{0.0}' in latex and '\\multirow{1}{*}{1.0}' in latex
    assert all(command in latex for command in ['\\dc', '\\low', '\\med', '\\hig', '\\cellcolor{gray!25}', '0.75', '0.25'])
    # One rule between the header and the first consequent, one between consequents.
    assert latex.count('\\midrule') == 2


def test_rules_to_latex_summarises_interval_scores(iris):
    partitions = utils.construct_partitions(iris[0], fs.FUZZY_SETS.t2)
    rule = _rule([0, -1, 1, -1], score=np.array([0.3, 0.4]))
    latex = vis_rules.rules_to_latex(rl.MasterRuleBase([rl.RuleBaseT2(partitions, [rule])]))
    assert '0.35' in latex
