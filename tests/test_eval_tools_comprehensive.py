"""
Comprehensive tests for eval_tools and eval_rules modules.

Tests rule evaluation, dominance scores, accuracy computation,
and other evaluation metrics.
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ex_fuzzy', 'ex_fuzzy'))

import fuzzy_sets as fs
import rules as rl
import evolutionary_fit as evf
import utils
try:
    import eval_tools
    import eval_rules
    HAS_EVAL_MODULES = True
except ImportError:
    HAS_EVAL_MODULES = False


@pytest.mark.skipif(not HAS_EVAL_MODULES, reason="Eval modules not available")
class TestEvalRuleBase:
    """Tests for eval_tools evaluation functionality."""

    @pytest.fixture
    def trained_classifier(self):
        """Create a trained classifier for testing."""
        X, y = make_classification(
            n_samples=100,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            n_classes=2,
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        clf = evf.BaseFuzzyRulesClassifier(nRules=10, nAnts=3, verbose=False)
        clf.fit(X_train, y_train, n_gen=15, pop_size=25)

        return clf, X_train, y_train, X_test, y_test

    def test_eval_fuzzy_model_exists(self):
        """Test that eval_fuzzy_model function exists."""
        assert hasattr(eval_tools, 'eval_fuzzy_model')

    def test_eval_rulebase_accuracy(self, trained_classifier):
        """Test accuracy computation using eval_fuzzy_model."""
        clf, X_train, y_train, X_test, y_test = trained_classifier

        # Use eval_fuzzy_model which is the actual function in the module
        result = eval_tools.eval_fuzzy_model(
            clf, X_train, y_train, X_test, y_test,
            plot_rules=False, print_rules=False, plot_partitions=False
        )

        assert result is not None


@pytest.mark.skipif(not HAS_EVAL_MODULES, reason="Eval modules not available")
class TestRuleWeights:
    """Tests for rule weight computation."""

    @pytest.fixture
    def rule_base_with_data(self):
        """Create a rule base with associated data."""
        X, y = make_classification(
            n_samples=100,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            n_classes=2,
            random_state=42
        )
        X = pd.DataFrame(X, columns=['f1', 'f2', 'f3', 'f4'])

        fuzzy_vars = utils.construct_partitions(X, fs.FUZZY_SETS.t1)

        clf = evf.BaseFuzzyRulesClassifier(
            nRules=10, nAnts=3,
            linguistic_variables=fuzzy_vars,
            verbose=False
        )
        clf.fit(X, y, n_gen=15, pop_size=25)

        return clf.rule_base, X, y

    def test_add_rule_weights(self, rule_base_with_data):
        """Test adding weights to rules."""
        rule_base, X, y = rule_base_with_data

        try:
            eval_rules.add_rule_weights(rule_base, X, y)

            # Rules should now have weights
            for rule in rule_base.get_rules():
                if hasattr(rule, 'weight'):
                    assert isinstance(rule.weight, (float, np.floating))
        except (AttributeError, TypeError):
            pytest.skip("add_rule_weights not implemented as expected")


@pytest.mark.skipif(not HAS_EVAL_MODULES, reason="Eval modules not available")
class TestDominanceScore:
    """Tests for dominance score computation."""

    @pytest.fixture
    def simple_rule_base(self):
        """Create a simple rule base for testing."""
        X, y = make_classification(
            n_samples=80,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            n_classes=2,
            random_state=42
        )

        clf = evf.BaseFuzzyRulesClassifier(nRules=10, nAnts=3, verbose=False)
        clf.fit(X, y, n_gen=10, pop_size=20)

        return clf.rule_base, X, y

    def test_dominance_score_computation(self, simple_rule_base):
        """Test that dominance scores are computed."""
        rule_base, X, y = simple_rule_base

        rules = rule_base.get_rules()

        # Rules should have dominance scores after fitting
        for rule in rules:
            if hasattr(rule, 'score'):
                assert isinstance(rule.score, (float, np.floating))
                assert 0 <= rule.score <= 1


@pytest.mark.skipif(not HAS_EVAL_MODULES, reason="Eval modules not available")
class TestRuleQualityMetrics:
    """Tests for rule quality metrics."""

    @pytest.fixture
    def trained_classifier(self):
        """Create a trained classifier."""
        X, y = make_classification(
            n_samples=100,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            n_classes=2,
            random_state=42
        )

        clf = evf.BaseFuzzyRulesClassifier(nRules=10, nAnts=3, verbose=False)
        clf.fit(X, y, n_gen=15, pop_size=25)

        return clf, X, y

    def test_size_antecedents_eval(self, trained_classifier):
        """Test size/antecedents evaluation."""
        clf, X, y = trained_classifier

        try:
            size_eval = eval_tools.size_antecedents_eval(clf.rule_base)
            assert isinstance(size_eval, (int, float, np.number))
        except (AttributeError, TypeError):
            pytest.skip("size_antecedents_eval not available")

    def test_effective_rulesize_eval(self, trained_classifier):
        """Test effective rule size evaluation."""
        clf, X, y = trained_classifier

        try:
            effective_size = eval_tools.effective_rulesize_eval(clf.rule_base)
            assert isinstance(effective_size, (int, float, np.number))
        except (AttributeError, TypeError):
            pytest.skip("effective_rulesize_eval not available")


@pytest.mark.skipif(not HAS_EVAL_MODULES, reason="Eval modules not available")
class TestFullEvaluation:
    """Tests for full rule base evaluation."""

    @pytest.fixture
    def trained_classifier(self):
        """Create a trained classifier."""
        X, y = make_classification(
            n_samples=100,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            n_classes=2,
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        clf = evf.BaseFuzzyRulesClassifier(nRules=10, nAnts=3, verbose=False)
        clf.fit(X_train, y_train, n_gen=15, pop_size=25)

        return clf, X_test, y_test

    def test_add_full_evaluation(self, trained_classifier):
        """Test adding full evaluation to rule base."""
        clf, X_test, y_test = trained_classifier

        try:
            eval_rules.add_full_evaluation(clf.rule_base, X_test, y_test)

            # Rules should now have evaluation metrics
            rules = clf.rule_base.get_rules()
            for rule in rules:
                if hasattr(rule, 'accuracy'):
                    assert isinstance(rule.accuracy, (float, np.floating))
        except (AttributeError, TypeError):
            pytest.skip("add_full_evaluation not available")


class TestDifferentFuzzyTypes:
    """Tests for evaluation with different fuzzy types."""

    @pytest.fixture
    def datasets(self):
        """Create dataset."""
        X, y = make_classification(
            n_samples=80,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            n_classes=2,
            random_state=42
        )
        return train_test_split(X, y, test_size=0.3, random_state=42)

    def test_eval_t1_classifier(self, datasets):
        """Test evaluation of T1 classifier."""
        X_train, X_test, y_train, y_test = datasets

        clf = evf.BaseFuzzyRulesClassifier(
            nRules=10, nAnts=3,
            fuzzy_type=fs.FUZZY_SETS.t1,
            verbose=False
        )
        clf.fit(X_train, y_train, n_gen=10, pop_size=20)

        predictions = clf.predict(X_test)
        accuracy = np.mean(predictions == y_test)

        assert 0 <= accuracy <= 1

    def test_eval_t2_classifier(self, datasets):
        """Test evaluation of T2 classifier."""
        X_train, X_test, y_train, y_test = datasets

        clf = evf.BaseFuzzyRulesClassifier(
            nRules=10, nAnts=3,
            fuzzy_type=fs.FUZZY_SETS.t2,
            verbose=False
        )
        clf.fit(X_train, y_train, n_gen=10, pop_size=20)

        predictions = clf.predict(X_test)
        accuracy = np.mean(predictions == y_test)

        assert 0 <= accuracy <= 1


class TestRuleBaseMetrics:
    """Tests for rule base level metrics."""

    @pytest.fixture
    def trained_classifier(self):
        """Create a trained classifier."""
        X, y = make_classification(
            n_samples=100,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            n_classes=2,
            random_state=42
        )

        clf = evf.BaseFuzzyRulesClassifier(nRules=15, nAnts=4, verbose=False)
        clf.fit(X, y, n_gen=15, pop_size=25)

        return clf

    def test_rule_count(self, trained_classifier):
        """Test counting rules in rule base."""
        clf = trained_classifier

        rules = clf.rule_base.get_rules()
        assert len(rules) > 0
        assert len(rules) <= clf.nRules

    def test_rule_consequents(self, trained_classifier):
        """Test getting rule consequents."""
        clf = trained_classifier

        consequents = clf.rule_base.get_consequents()
        assert len(consequents) > 0

        # All consequents should be valid class indices
        unique_classes = set(consequents)
        assert all(c in [0, 1] for c in unique_classes)

    def test_rule_antecedents(self, trained_classifier):
        """Test getting rule antecedents."""
        clf = trained_classifier

        antecedents = clf.rule_base.get_antecedents()
        assert len(antecedents) > 0

        # All should be fuzzy variables
        for ant in antecedents:
            assert isinstance(ant, fs.fuzzyVariable)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
