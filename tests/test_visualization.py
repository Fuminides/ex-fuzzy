"""Tests for the rule and fuzzy partition visualization module (vis_rules.py)."""
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

import evolutionary_fit as evf
import fuzzy_sets as fs
import rules as rl
import utils
import vis_rules


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


def test_histograms_ignore_dont_care_values():
    matrix = np.array([[0, -1], [0, 1], [2, 1]])

    assert vis_rules._column_histogram(matrix[:, 0]) == {0: 2, 2: 1}
    assert vis_rules._histogram(matrix) == [{0: 2, 2: 1}, {1: 2}]
    # Ties keep the first antecedent found.
    assert vis_rules._max_values(vis_rules._histogram(matrix)) == (0, 0)
    assert vis_rules._max_values([{}, {}]) == (None, None)
    np.testing.assert_array_equal(vis_rules.choose_popular_rules(matrix), [True, True, False])


def test_graph_connection_counts_co_occurrences():
    graph = vis_rules.create_graph_connection(np.array([[0, 1], [0, -1]]), 3)

    assert graph.shape == (6, 6)
    assert graph[0, 0] == 1.0
    assert graph[0, 4] == graph[4, 0] == graph[4, 4] == 0.5
    assert graph.sum() == 2.5


def test_connect_rule_bases(master_rule_base, partitions, capsys):
    graphs = vis_rules.connect_rulebase(master_rule_base[0])

    # The two rules sharing the most popular antecedent form the first graph, the remaining rule the second.
    assert len(graphs) == 2
    assert graphs[0].shape == (12, 12)
    first_label = partitions[0].name + ' ' + partitions[0].linguistic_variable_names()[0]
    assert graphs[0].loc[first_label, first_label] == 1.0
    assert graphs[1].loc[first_label, first_label] == 0.0

    assert [len(graphs) for graphs in vis_rules.connect_master_rulebase(master_rule_base)] == [2, 1]
    assert capsys.readouterr().out == ''


def test_connect_rule_bases_reports_unplottable_rules(partitions, capsys):
    with_empty_rule = rl.RuleBaseT1(partitions, [_rule([0, -1, -1, -1]), _rule([-1, -1, -1, -1])])
    assert len(vis_rules.connect_rulebase(with_empty_rule)) == 1
    assert 'too small' in capsys.readouterr().out

    mrule_base = rl.MasterRuleBase([rl.RuleBaseT1(partitions, []), with_empty_rule], consequent_names=['no', 'yes'])
    assert [len(graphs) for graphs in vis_rules.connect_master_rulebase(mrule_base)] == [0, 1]
    assert '"no", probably because there are no rules' in capsys.readouterr().out


def test_visualize_rulebase_exports_one_graph_per_consequent(master_rule_base, tmp_path):
    pytest.importorskip('networkx')
    classifier = evf.BaseFuzzyRulesClassifier()
    classifier.rule_base = master_rule_base

    vis_rules.visualize_rulebase(classifier, export_path=str(tmp_path))

    assert sorted(path.name for path in tmp_path.iterdir()) == ['consequent_0.gexf', 'consequent_1.gexf']
    assert 'Medium' in (tmp_path / 'consequent_0.gexf').read_text() or ' M' in (tmp_path / 'consequent_0.gexf').read_text()
    assert len(plt.get_fignums()) == 2


def test_visualize_rulebase_without_export(master_rule_base, tmp_path, monkeypatch):
    pytest.importorskip('networkx')
    monkeypatch.chdir(tmp_path)
    vis_rules.visualize_rulebase(master_rule_base)
    assert list(tmp_path.iterdir()) == []


def test_visualize_rulebase_names_graphs_after_their_consequent(master_rule_base, partitions, tmp_path, capsys):
    pytest.importorskip('networkx')
    with_empty_class = rl.MasterRuleBase([rl.RuleBaseT1(partitions, []), master_rule_base[1]])

    vis_rules.visualize_rulebase(with_empty_class, export_path=str(tmp_path))

    assert [path.name for path in tmp_path.iterdir()] == ['consequent_1.gexf']
    assert plt.gcf().axes[0].get_title() == 'Consequent: 1'
    assert 'no rules in the rule base' in capsys.readouterr().out


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
