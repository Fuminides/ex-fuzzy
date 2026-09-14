"""
Comprehensive tests for eval_tools and eval_rules modules.

Tests rule evaluation, dominance scores, accuracy computation,
statistical validation and the evaluation report.
"""
import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

import fuzzy_sets as fs
import rules as rl
import evolutionary_fit as evf
import utils
import eval_tools
import eval_rules


CLASS_NAMES = np.array(['setosa', 'versicolor', 'virginica'])
RULE_BASE_TYPES = [(fs.FUZZY_SETS.t1, rl.RuleBaseT1), (fs.FUZZY_SETS.t2, rl.RuleBaseT2), (fs.FUZZY_SETS.gt2, rl.RuleBaseGT2)]


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


@pytest.fixture(scope='module')
def iris():
    return load_iris(return_X_y=True)


def _petal_rule_base(X, fuzzy_type=fs.FUZZY_SETS.t1, rule_base_class=rl.RuleBaseT1):
    '''One petal length rule per class, plus a sepal rule for versicolor.'''
    partitions = utils.construct_partitions(X, fuzzy_type)
    rule_lists = [[rl.RuleSimple([-1, -1, 0, -1])],
                  [rl.RuleSimple([-1, -1, 1, -1]), rl.RuleSimple([1, 1, -1, -1])],
                  [rl.RuleSimple([-1, -1, 2, -1])]]
    return rl.MasterRuleBase([rule_base_class(partitions, rule_list) for rule_list in rule_lists])


@pytest.mark.parametrize('fuzzy_type, rule_base_class', RULE_BASE_TYPES)
def test_rule_weights_and_classification_metrics(iris, fuzzy_type, rule_base_class):
    X, y = iris
    mrule_base = _petal_rule_base(X, fuzzy_type, rule_base_class)
    evaluator = eval_rules.evalRuleBase(mrule_base, X, y)

    evaluator.add_full_evaluation()
    rules = mrule_base.get_rules()
    for rule in rules:
        assert 0 <= rule.support <= 1 and 0 <= rule.confidence <= 1
        assert rule.score == pytest.approx(rule.support * rule.confidence)
        assert 0 <= rule.accuracy <= 1
    assert evaluator.acc > 0.5

    # The support of a pattern alone bounds the support of the pattern with its class.
    antecedent_support = evaluator.compute_antecedent_pattern_support()
    assert antecedent_support.shape == (4,)
    assert np.all(antecedent_support >= evaluator.compute_pattern_support() - 1e-12)

    evaluator.add_auxiliary_rule_weights()
    for rule in rules:
        assert rule.aux_score.shape == rule.aux_support.shape == rule.aux_confidence.shape == (3,)
    # Each pattern's confidence is split across the classes.
    np.testing.assert_allclose(evaluator.compute_aux_pattern_confidence().sum(axis=1), 1.0)

    assert evaluator.association_degree().shape == (len(y), 4)


def test_string_labels_and_explicit_evaluation_data(iris):
    X, y = iris
    mrule_base = _petal_rule_base(X)
    mrule_base.rename_cons(list(CLASS_NAMES))

    evaluator = eval_rules.evalRuleBase(mrule_base, X, CLASS_NAMES[y])
    np.testing.assert_array_equal(evaluator.y, y)

    # Metrics on a subset given as a list of names, before any dominance score exists.
    evaluator.add_classification_metrics(X[:100], list(CLASS_NAMES[y][:100]))
    assert all(hasattr(rule, 'score') for rule in mrule_base.get_rules())
    assert mrule_base[0].rules[0].accuracy == 1.0
    assert mrule_base[2].rules[0].accuracy == 0.0


def test_temporal_moments_are_used_by_all_support_helpers():
    class TemporalRuleBase:
        def __init__(self):
            self.rule = rl.RuleSimple([0])
            self.calls = []

        def compute_firing_strenghts(self, X, time_moments, **kwargs):
            self.calls.append((X, time_moments))
            return np.array([[0.2], [0.4], [0.6], [0.8]])

        def get_rules(self):
            return [self.rule]

        def fuzzy_type(self):
            return fs.FUZZY_SETS.t1

    X = np.arange(4.0)[:, None]
    y = np.array([0, 1, 0, 1])
    moments = np.array([0, 0, 1, 1])
    rule_base = TemporalRuleBase()
    evaluator = eval_rules.evalRuleBase(rule_base, X, y, time_moments=moments)

    np.testing.assert_allclose(evaluator.compute_antecedent_pattern_support(), [0.5])
    np.testing.assert_allclose(evaluator.compute_aux_pattern_support(), [[0.2, 0.3]])
    np.testing.assert_allclose(evaluator.compute_aux_pattern_confidence(), [[0.4, 0.6]])
    assert len(rule_base.calls) == 3
    assert all(call[0] is X and call[1] is moments for call in rule_base.calls)


def test_rule_size_metrics(iris):
    X, y = iris
    mrule_base = _petal_rule_base(X)
    evaluator = eval_rules.evalRuleBase(mrule_base, X, y)
    for rule in mrule_base.get_rules():
        rule.score = 0.5

    # Five antecedents used out of four rules of four antecedents.
    assert evaluator.size_antecedents_eval(tolerance=0.1) == pytest.approx(1 - 5 / 16)
    assert evaluator.effective_rulesize_eval(tolerance=0.1) == 1.0
    # No rule is dominant enough.
    assert evaluator.size_antecedents_eval(tolerance=0.9) == 0.0
    assert evaluator.effective_rulesize_eval(tolerance=0.9) == 0.0

    # A rule without antecedents counts as using all of them, and is not effective.
    empty_rule = rl.RuleSimple([-1, -1, -1, -1])
    empty_rule.score = 0.5
    mrule_base[0].rules.append(empty_rule)
    assert evaluator.size_antecedents_eval(tolerance=0.1) == pytest.approx(1 - 9 / 20)
    assert evaluator.effective_rulesize_eval(tolerance=0.1) == pytest.approx(4 / 5)

    # A consequent without rules scores zero.
    partitions = mrule_base.antecedents
    without_rules = rl.MasterRuleBase([rl.RuleBaseT1(partitions, []), rl.RuleBaseT1(partitions, [rl.RuleSimple([0, -1, -1, -1])])])
    evaluator = eval_rules.evalRuleBase(without_rules, X, y)
    assert evaluator.size_antecedents_eval() == 0.0
    assert evaluator.effective_rulesize_eval() == 0.0


def test_statistical_validation_annotates_the_rules(iris):
    X, y = iris
    np.random.seed(0)
    mrule_base = _petal_rule_base(X)
    evaluator = eval_rules.evalRuleBase(mrule_base, X, y)
    evaluator.add_full_evaluation()

    p_labels, p_columns = evaluator.p_permutation_classifier_validation(n=5, r=2)
    assert 0 < p_labels <= 1 and 0 < p_columns <= 1
    assert mrule_base.p_value_class_structure == p_labels
    assert mrule_base.p_value_feature_coalition == p_columns

    evaluator.p_bootstrapping_rules_validation(n=5)
    for rule in mrule_base.get_rules():
        assert 0 <= rule.boot_p_value <= 1
        assert rule.boot_confidence_interval.shape == (2,)
        assert rule.boot_support_interval[0] <= rule.boot_support_interval[1]

    text = mrule_base.print_rules(return_rules=True, bootstrap_results=True)
    assert 'p-value Permutation Class Structure' in text and 'Support Interval' in text


class TestFuzzyEvaluator:

    @pytest.fixture(scope='class')
    def fitted(self, iris):
        X, y = iris
        classifier = evf.BaseFuzzyRulesClassifier(nRules=6, nAnts=2, verbose=False)
        classifier.fit(X, CLASS_NAMES[y], n_gen=5, pop_size=10, random_state=0)
        return classifier, X, CLASS_NAMES[y]

    def test_metrics(self, fitted):
        classifier, X, labels = fitted
        evaluator = eval_tools.FuzzyEvaluator(classifier)

        accuracy = evaluator.get_metric('accuracy_score', X, labels)
        assert accuracy == pytest.approx(np.mean(classifier.predict(X) == np.searchsorted(classifier.classes_names, labels)))
        assert evaluator.get_metric('not_a_metric', X, labels) == "Metric 'not_a_metric' not found in sklearn.metrics."
        assert evaluator.get_metric('accuracy_score', X, labels, bogus=1) == "Invalid arguments passed for the metric 'accuracy_score'."

    def test_full_report(self, fitted, tmp_path, capsys):
        pytest.importorskip('networkx')
        classifier, X, labels = fitted
        evaluator = eval_tools.FuzzyEvaluator(classifier)

        rules_text = evaluator.eval_fuzzy_model(X, labels, X, labels, plot_rules=True, print_rules=True, plot_partitions=True,
                                                return_rules=True, export_path=str(tmp_path), bootstrap_results_print=False)
        report = capsys.readouterr().out
        assert 'ACCURACY' in report and 'MATTHEW CORRCOEF' in report
        assert rules_text in report
        consequents_with_rules = [ix for ix, rule_base in enumerate(classifier.rule_base) if len(rule_base) > 0]
        assert sorted(path.name for path in tmp_path.iterdir()) == [f'consequent_{ix}.gexf' for ix in consequents_with_rules]

    def test_quiet_report(self, fitted, capsys):
        classifier, X, labels = fitted
        evaluator = eval_tools.FuzzyEvaluator(classifier)

        assert evaluator.eval_fuzzy_model(X, labels, X, labels, plot_rules=False, print_rules=False, plot_partitions=False,
                                          print_accuracy=False, print_matthew=False) is None
        assert capsys.readouterr().out == ''

        rules_text = eval_tools.eval_fuzzy_model(classifier, X, labels, X, labels, print_rules=False)
        assert rules_text.startswith('Rules for consequent')
        assert 'ACCURACY' in capsys.readouterr().out
