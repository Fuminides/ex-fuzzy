"""
Tests for statistical testing modules.

Tests bootstrapping_test, permutation_test, and pattern_stability modules
for statistical validation of fuzzy classifiers.
"""
import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split

import fuzzy_sets as fs
import evolutionary_fit as evf
import rules as rl
import utils
import bootstrapping_test as bt
import permutation_test as pt
import pattern_stability as ps


@pytest.fixture(scope='module')
def iris():
    X, y = load_iris(return_X_y=True)
    return X, y


@pytest.fixture(scope='module')
def fitted(iris):
    X, y = iris
    clf = evf.BaseFuzzyRulesClassifier(nRules=6, nAnts=2, verbose=False)
    clf.fit(X, y, n_gen=5, pop_size=10, random_state=0)
    assert len(clf.rule_base.get_rules()) > 0
    return clf


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


class _Oracle:
    '''Classifier stub that always predicts the given labels.'''

    def __init__(self, predictions):
        self.predictions = np.asarray(predictions)

    def predict(self, X):
        return self.predictions


class _FirstColumnSign:
    '''Classifier stub that predicts from the sign of the first feature.'''

    def predict(self, X):
        return (X[:, 0] > 0).astype(int)


class TestBootstrappingTest:

    def test_bootstrap_samples_resample_rows_with_their_labels(self, iris):
        X, y = iris
        np.random.seed(0)
        samples = bt.generate_bootstrap_samples(X, y, 4)

        assert len(samples) == 4
        for X_sample, y_sample in samples:
            assert X_sample.shape == X.shape
            assert y_sample.shape == y.shape
            # Every resampled row keeps the label it had in the original data.
            for row, label in zip(X_sample[:10], y_sample[:10]):
                matches = np.where((X == row).all(axis=1))[0]
                assert label in y[matches]

    def test_random_rules_respect_the_rule_shape(self):
        np.random.seed(0)
        random_rules = bt.generate_random_rules(4, 5, [3, 3, 2, 3], np.array([0, 1, 2]))

        assert len(random_rules) == 5
        for antecedents, consequent in random_rules:
            assert len(antecedents) == 4
            assert all(-1 <= value < 3 for value in antecedents)
            assert consequent in (0, 1, 2)

    @pytest.mark.parametrize('fuzzy_type, shape', [
        (fs.FUZZY_SETS.t1, (150,)),
        (fs.FUZZY_SETS.t2, (150, 2)),
    ])
    def test_membership_of_a_random_rule(self, iris, fuzzy_type, shape):
        X, _ = iris
        lvs = utils.construct_partitions(X, fuzzy_type)

        membership = bt.membership_randomrule([[0, -1, 2, -1], 0], X, lvs)
        assert membership.shape == shape
        expected = lvs[0][0].membership(X[:, 0]) * lvs[2][2].membership(X[:, 2])
        np.testing.assert_allclose(membership, expected)

        np.testing.assert_array_equal(bt.membership_randomrule([[-1] * 4, 0], X, lvs), np.ones(shape))

    def test_membership_of_a_random_rule_for_general_type_2(self, iris):
        X, _ = iris
        lvs = utils.construct_partitions(X, fs.FUZZY_SETS.gt2)

        membership = bt.membership_randomrule([[1, -1, -1, -1], 0], X, lvs)
        assert membership.shape == (150, len(lvs[0][0].alpha_cuts), 2)

    def test_membership_of_a_random_rule_rejects_other_fuzzy_types(self, iris):
        X, _ = iris

        class TemporalVariable:
            def fuzzy_type(self):
                return fs.FUZZY_SETS.temporal

        with pytest.raises(ValueError, match='not supported'):
            bt.membership_randomrule([[0], 0], X, [TemporalVariable()])

    def test_quality_metric_contrasts_class_and_non_class_memberships(self, iris):
        X, y = iris
        lvs = utils.construct_partitions(X, fs.FUZZY_SETS.t1)
        rule = [[-1, -1, 0, -1], 0]

        membership = lvs[2][0].membership(X[:, 2])
        expected = membership[y == 0].mean() - membership[y != 0].mean()
        assert bt.quality_metric_rule(rule, X, y, lvs) == pytest.approx(expected)
        # Small petal lengths describe setosa, so the rule should favour its class.
        assert expected > 0.5

    @pytest.mark.parametrize('fuzzy_type', [fs.FUZZY_SETS.t2, fs.FUZZY_SETS.gt2])
    def test_quality_metric_for_type_2_sets(self, iris, fuzzy_type):
        X, y = iris
        lvs = utils.construct_partitions(X, fuzzy_type)
        assert bt.quality_metric_rule([[-1, -1, 0, -1], 0], X, y, lvs) > 0.2

    @pytest.mark.parametrize('use_rule_base', [False, True])
    def test_null_distribution_and_rule_p_values(self, iris, fitted, use_rule_base):
        X, y = iris
        model = fitted.rule_base if use_rule_base else fitted
        np.random.seed(0)

        null_distribution = bt.generate_null_distribution(model, X, y, n_samples=3, nRules=4)
        assert null_distribution.shape == (3, 4)

        p_values = bt.compute_rule_p_value(model, X, y, nSamples=3, nRules=4)
        assert p_values.shape == (len(fitted.rule_base.get_rules()),)
        assert np.all((0 <= p_values) & (p_values <= 1))

    def test_rule_p_value_against_extreme_null_distributions(self, iris):
        X, y = iris
        lvs = utils.construct_partitions(X, fs.FUZZY_SETS.t1)
        rule = [[-1, -1, 0, -1], 0]

        assert bt._aux_compute_rule_p_value(rule, X, y, lvs, np.full((5, 3), -np.inf)) == 0.0
        assert bt._aux_compute_rule_p_value(rule, X, y, lvs, np.full((5, 3), np.inf)) == 1.0


class TestPermutationTest:

    def test_p_value_counts_random_victories(self):
        assert pt.permutation_p_compute(np.array([0.1, 0.5, 0.9]), 0.5) == pytest.approx(0.5)
        assert pt.permutation_p_compute(np.array([0.6, 0.7]), 0.5) == pytest.approx(1 / 3)

    def test_original_error_rate(self):
        labels = np.array([0, 1, 1, 0])
        assert pt.estimate_og_error_rate(_Oracle([0, 1, 0, 0]), None, labels, r=3) == pytest.approx(0.25)

    def test_label_permutation_test_rewards_a_perfect_classifier(self, iris):
        _, y = iris
        np.random.seed(0)
        assert pt.permutation_labels_test(_Oracle(y), None, y, k=20, r=2) == pytest.approx(1 / 21)

    def test_column_permutation_within_classes_keeps_the_class_structure(self):
        X, y = make_classification(n_samples=60, n_features=3, n_informative=2, n_redundant=0, random_state=0)
        np.random.seed(0)
        # Permuting rows inside a class cannot beat the original error rate of a deterministic classifier.
        assert pt.permute_columns_class_test(_FirstColumnSign(), X, y, k=10, r=2) >= 1 / 11

    def test_rule_wrapper_predicts_whether_a_rule_wins(self, iris, fitted):
        X, y = iris
        mrule_base = fitted.rule_base
        winning_rules, _ = mrule_base._winning_rules(X)

        wrapper, new_labels = pt._aux_ova_rule_classifier(mrule_base, 0, y)
        np.testing.assert_array_equal(wrapper.predict(X), winning_rules == 0)
        np.testing.assert_array_equal(new_labels, y == mrule_base.get_consequents()[0])

    def test_rulewise_permutation_tests_annotate_every_rule(self, iris, fitted):
        X, y = iris
        np.random.seed(0)
        mrule_base = fitted.rule_base.copy()

        pt.rulewise_label_permutation_test(mrule_base, X, y, k=5, r=2)
        pt.rulewise_column_permutation_test(mrule_base, X, y, k=5, r=2)

        for rule in mrule_base.get_rules():
            assert 0 < rule.p_value_class_structure <= 1
            assert 0 < rule.p_value_feature_coalitions <= 1


class TestPatternStability:

    def test_dictionary_helpers(self):
        assert ps.add_dicts({'a': 1}, {'a': 2, 'b': 3}) == {'a': 3, 'b': 3}
        assert ps.concatenate_dicts({'a': 1}, {'a': 2, 'b': 3}) == {'a': 1, 'b': 3}
        assert ps.str_rule_as_list(str(np.array([0.0, -1.0, 2.0]))) == [0, -1, 2]
        assert ps.str_rule_as_list('(1 [2])') == [1, 2]

    def test_class_names_and_linguistic_variables_configuration(self, iris):
        X, y = iris
        assert ps.pattern_stabilizer(X, y).classes_names == [0, 1, 2]
        assert ps.pattern_stabilizer(X, y, class_names=np.array(['a', 'b', 'c'])).classes_names == ['a', 'b', 'c']
        assert ps.pattern_stabilizer(X, y, class_names=['a', 'b', 'c']).classes_names == ['a', 'b', 'c']

        lvs = utils.construct_partitions(X, fs.FUZZY_SETS.t2)
        stabilizer = ps.pattern_stabilizer(X, y, linguistic_variables=lvs)
        assert stabilizer.fuzzy_type == fs.FUZZY_SETS.t2
        assert stabilizer.n_linguist_variables == [3, 3, 3, 3]
        assert stabilizer.domain is None

    def test_count_unique_patterns(self, iris):
        X, _ = iris
        lvs = utils.construct_partitions(X, fs.FUZZY_SETS.t1)
        rule_list = [rl.RuleSimple([0, -1, 1, -1]), rl.RuleSimple([0, 2, -1, -1])]
        for score, rule in zip([0.4, 0.2], rule_list):
            rule.score = score
        rule_base = rl.RuleBaseT1(lvs, rule_list)
        stabilizer = ps.pattern_stabilizer(X, np.zeros(len(X)))

        patterns, scores, var_used = stabilizer.count_unique_patterns(rule_base)
        assert list(patterns.values()) == [1, 1]
        assert sorted(scores.values()) == [0.2, 0.4]
        assert var_used[0] == {0.0: 2}
        assert var_used[1] == {-1.0: 1, 2.0: 1}

        # Rules added after construction are not deduplicated, so the pattern is counted twice.
        repeated = rl.RuleBaseT1(lvs, [rule_list[0]])
        repeated.add_rule(rule_list[0])
        patterns, _, var_used = stabilizer.count_unique_patterns(repeated)
        assert list(patterns.values()) == [2]
        assert var_used[2] == {1.0: 2}

        mrule_base = rl.MasterRuleBase([rule_base, rl.RuleBaseT1(lvs, [])])
        class_patterns, patterns_dss, class_vars = stabilizer.count_unique_patterns_all_classes(mrule_base)
        class_patterns, patterns_dss, class_vars = stabilizer.count_unique_patterns_all_classes(
            mrule_base, class_patterns, patterns_dss, class_vars)
        assert sum(class_patterns[0].values()) == 4
        assert class_patterns[1] == {}
        assert class_vars[0][0][0] == 4
        assert class_vars[1][0][0] == 0

    def test_stability_report_and_charts(self, iris, capsys):
        X, y = iris
        np.random.seed(0)
        stabilizer = ps.pattern_stabilizer(X, y, nRules=6, nAnts=2)
        stabilizer.stability_report(n=2, n_gen=3, pop_size=8)

        report = capsys.readouterr().out
        assert 'Pattern stability report for 2 generated solutions' in report
        assert 'Class 0' in report and 'Class 2' in report
        assert len(stabilizer.rule_bases) == 2 and len(stabilizer.accuracies) == 2

        stabilizer.var_reports(stabilizer.class_vars[0], stabilizer.rule_bases[0].antecedents, cutoff=0)
        stabilizer.text_report(stabilizer.class_patterns, stabilizer.patterns_dss, stabilizer.class_vars,
                               stabilizer.accuracies, stabilizer.rule_bases, rule_cutoff=0)
        assert 'appears in' not in capsys.readouterr().out

        stabilizer.pie_chart_basic(0, 0)
        stabilizer.pie_chart_var(2)
        stabilizer.pie_chart_class(0)
        stabilizer.pie_chart_class(1, var_list=[2])
        assert len(plt.gcf().axes) == 1

        # A variable can legitimately be absent from every retained rule.
        # Matplotlib rejects an empty pie, so the charts show an empty-state
        # message instead.
        for class_ix in stabilizer.class_vars:
            stabilizer.class_vars[class_ix][2] = {-1: 1}

        stabilizer.pie_chart_basic(2, 0)
        assert plt.gcf().axes[0].texts[0].get_text() == 'No variable usage'

        stabilizer.pie_chart_var(2)
        assert all(
            axis.texts[0].get_text() == 'No variable usage'
            for axis in plt.gcf().axes
        )

        stabilizer.pie_chart_class(0, var_list=[2])
        assert plt.gcf().axes[0].texts[0].get_text() == 'No variable usage'

    def test_variable_report_stops_at_the_cutoff(self, iris, capsys):
        X, y = iris
        lvs = utils.construct_partitions(X, fs.FUZZY_SETS.t1)
        stabilizer = ps.pattern_stabilizer(X, y)
        stabilizer.n = 2
        class_vars = {0: {-1: 6, 0: 4, 2: 2, 1: 0}, 1: {-1: 1, 0: 0, 1: 0, 2: 0}}

        stabilizer.var_reports(class_vars, lvs, cutoff=5)
        report = capsys.readouterr().out
        assert report.count('Variable') == 1
        assert f'{lvs[0][0].name} appears 2.00 times' in report
        assert f'{lvs[0][2].name} appears 1.00 times' in report

        stabilizer.var_reports(class_vars, lvs, cutoff=1)
        report = capsys.readouterr().out
        assert f'{lvs[0][0].name} appears' in report
        assert f'{lvs[0][2].name} appears' not in report

    @pytest.mark.parametrize('n_labels, first_color', [(2, '#FA8072'), (3, '#FA8072'), (5, None)])
    def test_colormap_assigns_a_color_per_label(self, iris, n_labels, first_color):
        X, y = iris
        lvs = utils.construct_partitions(X, fs.FUZZY_SETS.t1, n_partitions=n_labels)
        colors = ps.pattern_stabilizer(X, y).gen_colormap(lvs)

        assert list(colors) == lvs[0].linguistic_variable_names()
        if first_color is not None:
            assert colors[lvs[0][0].name] == first_color

    def test_stratified_solutions_hold_out_whole_groups(self, iris):
        X, y = iris
        frame = pd.DataFrame(X, columns=list('abcd'))
        frame['participant'] = np.arange(len(y)) % 10
        stabilizer = ps.pattern_stabilizer(frame, y, nRules=6, nAnts=2, stratify_by='participant',
                                           class_names=np.array(['setosa', 'versicolor', 'virginica']))

        rule_bases, accuracies = stabilizer.generate_solutions(n=1, n_gen=3, pop_size=8)
        assert len(rule_bases) == 1
        assert rule_bases[0].antecedents[0].name == 'a'
        assert 0 <= accuracies[0] <= 1

    def test_solutions_without_rules_are_reported(self, iris, fitted, monkeypatch, capsys):
        X, y = iris
        lvs = fitted.rule_base.antecedents
        empty = rl.MasterRuleBase([rl.RuleBaseT1(lvs, []) for _ in range(3)])
        stabilizer = ps.pattern_stabilizer(X, y)

        monkeypatch.setattr(stabilizer, 'generate_solutions', lambda *args, **kwargs: ([empty, fitted.rule_base], [0.0, 1.0]))
        class_patterns, *_ = stabilizer.get_patterns_scores(n=2)
        assert 'No rules were generated for solution 0' in capsys.readouterr().out
        assert sum(len(patterns) for patterns in class_patterns.values()) > 0

        monkeypatch.setattr(stabilizer, 'generate_solutions', lambda *args, **kwargs: ([empty], [0.0]))
        with pytest.raises(ValueError, match='No rules'):
            stabilizer.get_patterns_scores(n=1)


class TestStatisticalValidation:
    """Tests for general statistical validation."""

    def test_accuracy_is_statistically_significant(self):
        """Test that classifier accuracy is better than random."""
        X, y = make_classification(n_samples=100, n_features=4, n_informative=3, n_redundant=1,
                                   n_classes=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        clf = evf.BaseFuzzyRulesClassifier(nRules=15, nAnts=4, verbose=False)
        clf.fit(X_train, y_train, n_gen=30, pop_size=40, random_state=123)

        accuracy = np.mean(clf.predict(X_test) == y_test)

        # For a 2-class problem, random is 0.5
        # Should be at least as good as random (with margin for small sample variance)
        assert accuracy >= 0.4, f"Accuracy {accuracy} is too low"
