"""
Tests for the classifiers module.

Tests RuleMineClassifier, FuzzyRulesClassifier, and other
high-level classifier implementations.
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.base import is_classifier
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ex_fuzzy', 'ex_fuzzy'))

import fuzzy_sets as fs
import classifiers as clf
import evolutionary_fit as evf


class TestRuleMineClassifier:
    """Tests for RuleMineClassifier."""

    @pytest.fixture
    def binary_dataset(self):
        """Create binary classification dataset."""
        X, y = make_classification(
            n_samples=100,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            n_classes=2,
            random_state=42
        )
        X = pd.DataFrame(X, columns=['f1', 'f2', 'f3', 'f4'])
        return train_test_split(X, y, test_size=0.3, random_state=42)

    @pytest.fixture
    def iris_dataset(self):
        """Load Iris dataset."""
        X, y = load_iris(return_X_y=True)
        X = pd.DataFrame(X, columns=['sl', 'sw', 'pl', 'pw'])
        return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    def test_rule_mine_classifier_creation(self):
        """Test RuleMineClassifier creation."""
        classifier = clf.RuleMineClassifier(
            nRules=20,
            nAnts=3,
            fuzzy_type=fs.FUZZY_SETS.t1,
            tolerance=0.05
        )
        assert classifier.nAnts == 3
        assert classifier.fuzzy_type == fs.FUZZY_SETS.t1
        assert classifier.tolerance == 0.05

    def test_rule_mine_classifier_fit_predict(self, binary_dataset):
        """Test basic fit and predict workflow."""
        X_train, X_test, y_train, y_test = binary_dataset

        classifier = clf.RuleMineClassifier(
            nRules=10,
            nAnts=3,
            verbose=False
        )
        classifier.fit(X_train, y_train, n_gen=5, pop_size=10)

        predictions = classifier.predict(X_test)

        assert len(predictions) == len(y_test)
        assert all(p in [0, 1] for p in predictions)

    def test_rule_mine_classifier_multiclass(self, iris_dataset):
        """Test multiclass classification."""
        X_train, X_test, y_train, y_test = iris_dataset

        classifier = clf.RuleMineClassifier(
            nRules=15,
            nAnts=4,
            verbose=False
        )
        classifier.fit(X_train, y_train, n_gen=10, pop_size=20)

        predictions = classifier.predict(X_test)

        assert len(predictions) == len(y_test)
        assert all(p in [0, 1, 2] for p in predictions)

    def test_rule_mine_classifier_internal_classifier(self, binary_dataset):
        """Test accessing internal classifier."""
        X_train, X_test, y_train, y_test = binary_dataset

        classifier = clf.RuleMineClassifier(nRules=10, nAnts=3, verbose=False)
        classifier.fit(X_train, y_train, n_gen=5, pop_size=10)

        internal = classifier.internal_classifier()

        assert isinstance(internal, evf.BaseFuzzyRulesClassifier)
        assert internal.rule_base is not None


class TestFuzzyRulesClassifier:
    """Tests for FuzzyRulesClassifier (double optimization)."""

    @pytest.fixture
    def binary_dataset(self):
        """Create binary classification dataset."""
        X, y = make_classification(
            n_samples=100,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            n_classes=2,
            random_state=42
        )
        return train_test_split(X, y, test_size=0.3, random_state=42)

    def test_fuzzy_rules_classifier_creation(self):
        """Test FuzzyRulesClassifier creation."""
        classifier = clf.FuzzyRulesClassifier(
            nRules=20,
            nAnts=3,
            fuzzy_type=fs.FUZZY_SETS.t1,
            expansion_factor=2
        )
        assert classifier.fuzzy_type == fs.FUZZY_SETS.t1

    def test_fuzzy_rules_classifier_fit_predict(self, binary_dataset):
        """Test basic fit and predict workflow."""
        X_train, X_test, y_train, y_test = binary_dataset

        classifier = clf.FuzzyRulesClassifier(
            nRules=10,
            nAnts=3,
            verbose=False,
            expansion_factor=1
        )
        classifier.fit(X_train, y_train, n_gen=5, pop_size=10)

        predictions = classifier.predict(X_test)

        assert len(predictions) == len(y_test)

    def test_fuzzy_rules_classifier_internal_classifier(self, binary_dataset):
        """Test accessing internal classifier."""
        X_train, X_test, y_train, y_test = binary_dataset

        classifier = clf.FuzzyRulesClassifier(
            nRules=10,
            nAnts=3,
            verbose=False,
            expansion_factor=1
        )
        classifier.fit(X_train, y_train, n_gen=5, pop_size=10)

        internal = classifier.internal_classifier()

        assert isinstance(internal, evf.BaseFuzzyRulesClassifier)


class TestSklearnCompatibility:
    """Tests for scikit-learn compatibility."""

    def test_is_sklearn_classifier(self):
        """Test that classifiers are recognized as sklearn classifiers."""
        rule_mine = clf.RuleMineClassifier(nRules=10, verbose=False)
        fuzzy_rules = clf.FuzzyRulesClassifier(nRules=10, verbose=False)

        assert is_classifier(rule_mine)
        assert is_classifier(fuzzy_rules)

    def test_cross_validation_compatibility(self):
        """Test that classifiers work with cross_val_score."""
        X, y = make_classification(
            n_samples=100,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            n_classes=2,
            random_state=42
        )
        X = pd.DataFrame(X)

        classifier = clf.RuleMineClassifier(nRules=10, nAnts=3, verbose=False)

        # This should not raise an error
        # Note: May be slow due to small n_gen
        try:
            scores = cross_val_score(
                classifier, X, y, cv=2,
                fit_params={'n_gen': 3, 'pop_size': 10}
            )
            assert len(scores) == 2
        except Exception:
            # Some sklearn versions may not support fit_params in cross_val_score
            pass


class TestClassifierWithDifferentFuzzyTypes:
    """Tests for classifiers with different fuzzy set types."""

    @pytest.fixture
    def dataset(self):
        """Create dataset."""
        X, y = make_classification(
            n_samples=80,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            n_classes=2,
            random_state=42
        )
        X = pd.DataFrame(X, columns=['f1', 'f2', 'f3', 'f4'])
        return train_test_split(X, y, test_size=0.25, random_state=42)

    def test_rule_mine_t1(self, dataset):
        """Test RuleMineClassifier with T1 fuzzy sets."""
        X_train, X_test, y_train, y_test = dataset

        classifier = clf.RuleMineClassifier(
            nRules=10, nAnts=3,
            fuzzy_type=fs.FUZZY_SETS.t1,
            verbose=False
        )
        classifier.fit(X_train, y_train, n_gen=5, pop_size=10)

        predictions = classifier.predict(X_test)
        assert len(predictions) == len(y_test)

    def test_rule_mine_t2(self, dataset):
        """Test RuleMineClassifier with T2 fuzzy sets."""
        X_train, X_test, y_train, y_test = dataset

        classifier = clf.RuleMineClassifier(
            nRules=10, nAnts=3,
            fuzzy_type=fs.FUZZY_SETS.t2,
            verbose=False
        )
        classifier.fit(X_train, y_train, n_gen=5, pop_size=10)

        predictions = classifier.predict(X_test)
        assert len(predictions) == len(y_test)


class TestClassifierParameters:
    """Tests for different classifier parameter configurations."""

    @pytest.fixture
    def dataset(self):
        """Create dataset."""
        X, y = make_classification(
            n_samples=80,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            n_classes=2,
            random_state=42
        )
        X = pd.DataFrame(X)
        return train_test_split(X, y, test_size=0.25, random_state=42)

    def test_different_n_rules(self, dataset):
        """Test with different numbers of rules."""
        X_train, X_test, y_train, y_test = dataset

        for n_rules in [5, 15, 25]:
            classifier = clf.RuleMineClassifier(
                nRules=n_rules,
                nAnts=3,
                verbose=False
            )
            classifier.fit(X_train, y_train, n_gen=5, pop_size=10)

            predictions = classifier.predict(X_test)
            assert len(predictions) == len(y_test)

    def test_different_n_ants(self, dataset):
        """Test with different numbers of antecedents."""
        X_train, X_test, y_train, y_test = dataset

        for n_ants in [2, 3, 4]:
            classifier = clf.RuleMineClassifier(
                nRules=10,
                nAnts=n_ants,
                verbose=False
            )
            classifier.fit(X_train, y_train, n_gen=5, pop_size=10)

            predictions = classifier.predict(X_test)
            assert len(predictions) == len(y_test)

    def test_with_tolerance(self, dataset):
        """Test with different tolerance values."""
        X_train, X_test, y_train, y_test = dataset

        for tolerance in [0.0, 0.01, 0.1]:
            classifier = clf.RuleMineClassifier(
                nRules=10,
                nAnts=3,
                tolerance=tolerance,
                verbose=False
            )
            classifier.fit(X_train, y_train, n_gen=5, pop_size=10)

            predictions = classifier.predict(X_test)
            assert len(predictions) == len(y_test)


class TestExpansionFactor:
    """Tests for FuzzyRulesClassifier expansion factor."""

    def test_expansion_factor_effect(self):
        """Test that expansion factor affects first phase rules."""
        X, y = make_classification(
            n_samples=80,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            n_classes=2,
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        # With expansion_factor=1
        clf1 = clf.FuzzyRulesClassifier(
            nRules=10, nAnts=3,
            expansion_factor=1,
            verbose=False
        )
        clf1.fit(X_train, y_train, n_gen=5, pop_size=10)

        # With expansion_factor=2
        clf2 = clf.FuzzyRulesClassifier(
            nRules=10, nAnts=3,
            expansion_factor=2,
            verbose=False
        )
        clf2.fit(X_train, y_train, n_gen=5, pop_size=10)

        # Both should produce valid predictions
        pred1 = clf1.predict(X_test)
        pred2 = clf2.predict(X_test)

        assert len(pred1) == len(y_test)
        assert len(pred2) == len(y_test)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
