"""Tests for the KEEL benchmark harness in ``benchmarks/``.

The collection itself is not redistributed, so the parser is exercised against
KEEL files written into a temporary directory. The measurement scripts are
opt-in and are not run here; what is checked is the parsing, the task grid, the
size accounting and the aggregation that turn into published numbers.
"""
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

BENCHMARKS = Path(__file__).resolve().parent.parent / 'benchmarks'
sys.path.insert(0, str(BENCHMARKS))

import keel_datasets  # noqa: E402
import benchmark_keel  # noqa: E402
import aggregate_keel  # noqa: E402


IRIS_LIKE = """@relation toy
@attribute Length real [0.0, 10.0]
@attribute Shade {small, medium, large}
@attribute Class {yes, no}
@inputs Length, Shade
@outputs Class
@data
1.0, small, yes
2.0, medium, yes
3.0, large, no
4.0, small, no
"""


@pytest.fixture
def collection(tmp_path):
    """A two-dataset KEEL collection on disk."""
    for name, text in (('toy', IRIS_LIKE), ('tiny', IRIS_LIKE.replace('@relation toy',
                                                                      '@relation tiny'))):
        directory = tmp_path / name
        directory.mkdir()
        (directory / f'{name}.dat').write_text(text)
    return tmp_path


def test_loads_numeric_and_nominal_columns(collection):
    data = keel_datasets.load_dataset('toy', collection)
    assert data.n_samples == 4 and data.n_features == 2 and data.n_classes == 2
    np.testing.assert_allclose(data.X[:, 0], [1.0, 2.0, 3.0, 4.0])
    # Nominal levels take the codes of their declared order, not of appearance.
    np.testing.assert_array_equal(data.X[:, 1], [0, 1, 2, 0])
    np.testing.assert_array_equal(data.categorical_mask, [0, 1])
    assert data.class_names == ['no', 'yes']
    np.testing.assert_array_equal(data.y, [1, 1, 0, 0])
    assert data.dropped_rows == 0


def test_accepts_singular_output_and_tight_type_spec(tmp_path):
    (tmp_path / 'one').mkdir()
    (tmp_path / 'one' / 'one.dat').write_text(
        '@relation one\n'
        '@attribute A real[-1.87,0.965]\n'
        '@attribute Class {a, b}\n'
        '@inputs A\n'
        '@output Class\n'
        '@data\n0.5, a\n-1.0, b\n')
    data = keel_datasets.load_dataset('one', tmp_path)
    assert data.feature_names == ['A'] and data.n_classes == 2
    np.testing.assert_allclose(data.X[:, 0], [0.5, -1.0])


def test_missing_markers_are_dropped_and_counted(tmp_path):
    (tmp_path / 'gappy').mkdir()
    (tmp_path / 'gappy' / 'gappy.dat').write_text(
        IRIS_LIKE.replace('3.0, large, no', '?, large, no'))
    data = keel_datasets.load_dataset('gappy', tmp_path)
    assert data.dropped_rows == 1 and data.n_samples == 3


def test_short_row_is_rejected(tmp_path):
    (tmp_path / 'ragged').mkdir()
    (tmp_path / 'ragged' / 'ragged.dat').write_text(
        IRIS_LIKE.replace('4.0, small, no', '4.0, small'))
    with pytest.raises(ValueError, match='expected 3 values'):
        keel_datasets.load_dataset('ragged', tmp_path)


def test_available_datasets_and_root_resolution(collection, monkeypatch):
    assert keel_datasets.available_datasets(collection) == ['tiny', 'toy']
    monkeypatch.setenv('EX_FUZZY_KEEL_ROOT', str(collection))
    assert keel_datasets.keel_root() == collection
    monkeypatch.setenv('EX_FUZZY_KEEL_ROOT', str(collection / 'absent'))
    monkeypatch.setattr(keel_datasets, 'DEFAULT_ROOTS', ())
    with pytest.raises(FileNotFoundError, match='EX_FUZZY_KEEL_ROOT'):
        keel_datasets.keel_root()


def test_task_grid_covers_every_pair_once(collection, capsys):
    benchmark_keel.main(['--list-tasks', '--root', str(collection)])
    printed = [line.split() for line in capsys.readouterr().out.splitlines()]
    assert len(printed) == 2 * len(benchmark_keel.METHODS)
    assert len(set(map(tuple, printed))) == len(printed)
    assert {method for _, method in printed} == set(benchmark_keel.METHODS)


def test_task_index_selects_the_listed_pair(collection, tmp_path, capsys):
    benchmark_keel.main(['--list-tasks', '--root', str(collection)])
    grid = [line.split() for line in capsys.readouterr().out.splitlines()]
    dataset, method = grid[2]
    output = tmp_path / 'out'
    benchmark_keel.main(['--task-index', '3', '--root', str(collection), '--folds', '2',
                         '--output-dir', str(output)])
    assert (output / f'{dataset}__{method}.json').exists()


def test_run_task_records_folds_and_sizes(collection, tmp_path):
    output = tmp_path / 'out'
    assert benchmark_keel.main(['--dataset', 'toy', '--method', 'sklearn-tree',
                                '--root', str(collection), '--folds', '2',
                                '--output-dir', str(output)]) == 0
    record = json.loads((output / 'toy__sklearn-tree.json').read_text())
    assert record['status'] == 'ok'
    assert len(record['folds_detail']) == 2
    assert record['mean_rules'] >= 1 and record['mean_conditions'] >= 1
    assert 0.0 <= record['mean_accuracy'] <= 1.0
    assert record['environment']['packages']['numpy']


def test_existing_result_is_kept_unless_forced(collection, tmp_path):
    output = tmp_path / 'out'
    output.mkdir()
    destination = output / 'toy__sklearn-tree.json'
    destination.write_text('{"status": "sentinel"}\n')
    benchmark_keel.main(['--dataset', 'toy', '--method', 'sklearn-tree', '--root',
                         str(collection), '--folds', '2', '--output-dir', str(output)])
    assert json.loads(destination.read_text())['status'] == 'sentinel'
    benchmark_keel.main(['--dataset', 'toy', '--method', 'sklearn-tree', '--root',
                         str(collection), '--folds', '2', '--output-dir', str(output),
                         '--force'])
    assert json.loads(destination.read_text())['status'] == 'ok'


def test_failure_is_recorded_rather_than_lost(collection, tmp_path, monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError('no estimator today')

    monkeypatch.setattr(benchmark_keel, 'build_method', explode)
    output = tmp_path / 'out'
    assert benchmark_keel.main(['--dataset', 'toy', '--method', 'sklearn-tree', '--root',
                                str(collection), '--folds', '2',
                                '--output-dir', str(output)]) == 1
    record = json.loads((output / 'toy__sklearn-tree.json').read_text())
    assert record['status'] == 'failed' and 'no estimator today' in record['error']


def test_tree_size_counts_leaves_and_their_depths():
    from sklearn.tree import DecisionTreeClassifier

    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    model = DecisionTreeClassifier(random_state=0).fit(X, y)
    size = benchmark_keel._size_sklearn_tree(model)
    # One split makes two leaves, each one condition deep.
    assert size == dict(rules=2, conditions=2)


def test_gradient_boosting_size_sums_leaves_over_every_tree():
    from sklearn.ensemble import HistGradientBoostingClassifier

    X = np.repeat(np.arange(4.0), 10)[:, None]
    y = np.repeat([0, 1, 2, 2], 10)
    model = HistGradientBoostingClassifier(max_iter=3, max_depth=1, min_samples_leaf=5,
                                           random_state=0).fit(X, y)
    size = benchmark_keel._size_sklearn_hgb(model)
    # Three iterations of one stump per class: two leaves, one condition each.
    trees = model.n_iter_ * model.n_trees_per_iteration_
    assert trees == 9
    assert size == dict(rules=2 * trees, conditions=2 * trees)


def test_rank_within_averages_ties():
    ranks = aggregate_keel.rank_within({'a': 0.9, 'b': 0.9, 'c': 0.5})
    assert ranks == {'a': 1.5, 'b': 1.5, 'c': 3.0}
    ascending = aggregate_keel.rank_within({'a': 2.0, 'b': 1.0}, higher_is_better=False)
    assert ascending == {'b': 1.0, 'a': 2.0}


def _result(dataset, method, accuracy, rules):
    return dict(status='ok', dataset=dataset, method=method, folds=5, seed=0,
                configuration={}, n_samples=10, n_features=2, n_classes=2,
                environment={'packages': {}}, mean_accuracy=accuracy, std_accuracy=0.0,
                mean_balanced_accuracy=accuracy, mean_macro_f1=accuracy,
                mean_rules=rules, mean_conditions=rules * 2, mean_fit_seconds=1.0,
                mean_unclassified=0.0)


def test_aggregate_compares_only_complete_datasets(tmp_path):
    methods = list(benchmark_keel.METHODS)
    for index, method in enumerate(methods):
        (tmp_path / f'full__{method}.json').write_text(
            json.dumps(_result('full', method, 0.5 + index / 100, 10 * (index + 1))))
    # ``half`` is missing the last method and must not reach the comparison.
    for method in methods[:-1]:
        (tmp_path / f'half__{method}.json').write_text(
            json.dumps(_result('half', method, 0.9, 5)))
    successes, failures = aggregate_keel.read_results(tmp_path)
    report = aggregate_keel.aggregate(successes, failures, methods)
    assert report['datasets_compared'] == ['full']
    assert report['datasets_incomplete'] == ['half']
    assert report['summary'][methods[-1]]['best_on'] == 1
    assert report['summary'][methods[0]]['mean_rank'] == len(methods)


def test_aggregate_reports_failed_pairs(tmp_path):
    (tmp_path / 'boom__sklearn-tree.json').write_text(
        json.dumps(dict(status='failed', dataset='boom', method='sklearn-tree',
                        error='RuntimeError: boom')))
    successes, failures = aggregate_keel.read_results(tmp_path)
    assert successes == {}
    assert failures == [dict(dataset='boom', method='sklearn-tree',
                             error='RuntimeError: boom')]


def test_publishing_refuses_an_incomplete_grid(tmp_path):
    methods = list(benchmark_keel.METHODS)
    results = tmp_path / 'results'
    results.mkdir()
    for method in methods[:-1]:
        (results / f'half__{method}.json').write_text(
            json.dumps(_result('half', method, 0.9, 5)))
    with pytest.raises(SystemExit):
        aggregate_keel.main(['--results', str(results),
                             '--output', str(tmp_path / 'keel.json')])


def test_publishing_writes_figure_table_and_json(tmp_path):
    methods = list(benchmark_keel.METHODS)
    results = tmp_path / 'results'
    results.mkdir()
    for name in ('alpha', 'beta'):
        for index, method in enumerate(methods):
            (results / f'{name}__{method}.json').write_text(
                json.dumps(_result(name, method, 0.5 + index / 10, 10 ** (index + 1))))
    destination = tmp_path / 'keel.json'
    assert aggregate_keel.main(['--results', str(results),
                                '--output', str(destination)]) == 0
    report = json.loads(destination.read_text())
    assert report['n_datasets_compared'] == 2
    assert destination.with_suffix('.svg').stat().st_size > 0
    table = destination.with_suffix('.md').read_text()
    assert 'alpha' in table and 'beta' in table
    # The winner of every row is the one marked bold in the accuracy table.
    assert f'**{0.5 + (len(methods) - 1) / 10:.4f}**' in table


def test_ferl_presets_record_their_configuration():
    compact = benchmark_keel.method_configuration('exfuzzy-ferl-compact')
    deep = benchmark_keel.method_configuration('exfuzzy-ferl-deep')
    medium = benchmark_keel.method_configuration('exfuzzy-ferl-medium')
    # Written out explicitly: Ex-Fuzzy's FERL defaults to 15 rules, fgrt to 20.
    assert compact['max_rules'] == 20 and compact['fit'] == {'patience': 3}
    # Medium is the paper's learned-threshold preset (fgrt-performance).
    assert medium['split_mode'] == 'learned' and medium['max_rules'] == 150
    assert medium['fit'] == {'patience': 16}
    # Deep is the separate learned-tree estimator, not a FERL configuration.
    assert deep == {'estimator': 'DeepFERL', 'library_defaults': True}
    assert 'exfuzzy-ferl-deep' not in benchmark_keel.FERL_PRESETS
    assert benchmark_keel.EXFUZZY_METHODS >= set(benchmark_keel.FERL_PRESETS) | {'exfuzzy-ferl-deep'}


def test_logistic_regression_records_parameters_not_rules(collection, tmp_path):
    output = tmp_path / 'out'
    assert benchmark_keel.main(['--dataset', 'toy', '--method', 'sklearn-logreg',
                                '--root', str(collection), '--folds', '2',
                                '--output-dir', str(output)]) == 0
    record = json.loads((output / 'toy__sklearn-logreg.json').read_text())
    assert record['status'] == 'ok'
    assert record['mean_rules'] is None and record['mean_conditions'] is None
    # Two features, binary target: two coefficients and one intercept.
    assert record['mean_parameters'] == 3.0


def test_rule_less_method_is_aggregated_tabled_and_plotted(tmp_path):
    methods = list(benchmark_keel.METHODS)
    results = tmp_path / 'results'
    results.mkdir()
    for name, classes in (('alpha', 2), ('beta', 3), ('gamma', 8)):
        for index, method in enumerate(methods):
            record = _result(name, method, 0.5 + index / 20, 10 ** (index % 4 + 1))
            record['n_classes'] = classes
            if method in benchmark_keel.RULELESS_METHODS:
                record.update(mean_rules=None, mean_conditions=None, mean_parameters=9.0)
            (results / f'{name}__{method}.json').write_text(json.dumps(record))
    destination = tmp_path / 'keel.json'
    assert aggregate_keel.main(['--results', str(results),
                                '--output', str(destination)]) == 0
    report = json.loads(destination.read_text())
    entry = report['summary']['sklearn-logreg']
    assert entry['median_rules'] is None and entry['rules_q1'] is None
    assert report['summary']['sklearn-tree']['median_rules'] is not None
    assert len(report['class_bands']) == 3
    table = destination.with_suffix('.md').read_text()
    assert '| Logistic regression |' in table and '—' in table
    assert destination.with_suffix('.svg').stat().st_size > 0


def test_association_rule_methods_report_both_rule_modes():
    additive = benchmark_keel.method_configuration('exfuzzy-frc-additive')
    sufficient = benchmark_keel.method_configuration('exfuzzy-frc-sufficient')
    assert additive['rule_mode'] == 'additive' and sufficient['rule_mode'] == 'sufficient'
    assert {k: v for k, v in additive.items() if k != 'rule_mode'} == \
        {k: v for k, v in sufficient.items() if k != 'rule_mode'}
    assert {'exfuzzy-frc-additive', 'exfuzzy-frc-sufficient'} <= benchmark_keel.EXFUZZY_METHODS


def test_association_rule_method_runs_and_counts_rules(collection, tmp_path):
    output = tmp_path / 'out'
    assert benchmark_keel.main(['--dataset', 'toy', '--method', 'exfuzzy-frc-sufficient',
                                '--root', str(collection), '--folds', '2',
                                '--output-dir', str(output)]) == 0
    record = json.loads((output / 'toy__exfuzzy-frc-sufficient.json').read_text())
    assert record['status'] == 'ok'
    assert record['configuration']['rule_mode'] == 'sufficient'
    assert record['mean_rules'] >= 0 and record['mean_conditions'] >= record['mean_rules']
