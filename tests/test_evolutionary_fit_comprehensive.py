"""
Comprehensive tests for the evolutionary_fit module.

Tests BaseFuzzyRulesClassifier training, prediction, and various
configuration options including fuzzy types, backends, and parameters.
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ex_fuzzy', 'ex_fuzzy'))

import fuzzy_sets as fs
import evolutionary_fit as evf
import utils
import rules as rl


class TestBaseFuzzyRulesClassifierBasic:
    """Basic tests for BaseFuzzyRulesClassifier."""

    @pytest.fixture
    def simple_dataset(self):
        """Create a simple binary classification dataset."""
        X, y = make_classification(
            n_samples=100,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            n_classes=2,
            random_state=42
        )
        return train_test_split(X, y, test_size=0.3, random_state=42)

    @pytest.fixture
    def iris_dataset(self):
        """Load Iris dataset."""
        X, y = load_iris(return_X_y=True)
        return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    def test_classifier_creation_default(self):
        """Test classifier creation with default parameters."""
        clf = evf.BaseFuzzyRulesClassifier()
        assert clf.nRules == 30
        assert clf.nAnts == 4
        assert clf.fuzzy_type == fs.FUZZY_SETS.t1
        assert clf.tolerance == 0.0
        assert clf.verbose == False

    def test_classifier_creation_custom(self):
        """Test classifier creation with custom parameters."""
        clf = evf.BaseFuzzyRulesClassifier(
            nRules=20,
            nAnts=3,
            fuzzy_type=fs.FUZZY_SETS.t2,
            tolerance=0.01,
            verbose=True,
            n_linguistic_variables=5
        )
        assert clf.nRules == 20
        assert clf.nAnts == 3
        assert clf.fuzzy_type == fs.FUZZY_SETS.t2
        assert clf.tolerance == 0.01
        assert clf.verbose == True

    def test_fit_basic(self, simple_dataset):
        """Test basic fitting with minimal generations."""
        X_train, X_test, y_train, y_test = simple_dataset
        clf = evf.BaseFuzzyRulesClassifier(nRules=10, nAnts=3, verbose=False)
        clf.fit(X_train, y_train, n_gen=5, pop_size=10)

        assert hasattr(clf, 'rule_base')
        assert clf.rule_base is not None
        assert hasattr(clf, 'nclasses_')
        assert clf.nclasses_ == 2

    def test_predict_basic(self, simple_dataset):
        """Test basic prediction."""
        X_train, X_test, y_train, y_test = simple_dataset
        clf = evf.BaseFuzzyRulesClassifier(nRules=10, nAnts=3, verbose=False)
        clf.fit(X_train, y_train, n_gen=5, pop_size=10)

        predictions = clf.predict(X_test)

        assert len(predictions) == len(y_test)
        assert all(p in [0, 1] for p in predictions)

    def test_predict_proba(self, simple_dataset):
        """Test probability prediction."""
        X_train, X_test, y_train, y_test = simple_dataset
        clf = evf.BaseFuzzyRulesClassifier(nRules=10, nAnts=3, verbose=False)
        clf.fit(X_train, y_train, n_gen=5, pop_size=10)

        proba = clf.predict_proba(X_test)

        assert proba.shape == (len(y_test), 2)  # 2 classes
        # Probabilities should sum to 1 for each sample
        row_sums = np.sum(proba, axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(len(y_test)), decimal=5)

    def test_predict_membership_class(self, simple_dataset):
        """Test class membership prediction."""
        X_train, X_test, y_train, y_test = simple_dataset
        clf = evf.BaseFuzzyRulesClassifier(nRules=10, nAnts=3, verbose=False)
        clf.fit(X_train, y_train, n_gen=5, pop_size=10)

        memberships = clf.predict_membership_class(X_test)

        assert memberships.shape == (len(y_test), 2)
        assert np.all(memberships >= 0)
        assert np.all(memberships <= 1)

    def test_predict_proba_rules(self, simple_dataset):
        """Test per-rule probability prediction."""
        X_train, X_test, y_train, y_test = simple_dataset
        clf = evf.BaseFuzzyRulesClassifier(nRules=10, nAnts=3, verbose=False)
        clf.fit(X_train, y_train, n_gen=5, pop_size=10)

        # With truth degrees
        rule_proba_truth = clf.predict_proba_rules(X_test, truth_degrees=True)
        assert rule_proba_truth.shape[0] == len(y_test)

        # Without truth degrees (association degrees)
        rule_proba_assoc = clf.predict_proba_rules(X_test, truth_degrees=False)
        assert rule_proba_assoc.shape[0] == len(y_test)


class TestClassifierWithDifferentFuzzyTypes:
    """Tests for classifier with different fuzzy set types."""

    @pytest.fixture
    def dataset(self):
        """Create a small dataset for testing."""
        X, y = make_classification(
            n_samples=80,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            n_classes=2,
            random_state=42
        )
        return train_test_split(X, y, test_size=0.25, random_state=42)

    def test_classifier_t1(self, dataset):
        """Test classifier with Type-1 fuzzy sets."""
        X_train, X_test, y_train, y_test = dataset
        clf = evf.BaseFuzzyRulesClassifier(
            nRules=10, nAnts=3,
            fuzzy_type=fs.FUZZY_SETS.t1,
            verbose=False
        )
        clf.fit(X_train, y_train, n_gen=5, pop_size=10)

        predictions = clf.predict(X_test)
        assert len(predictions) == len(y_test)
        assert clf.rule_base.fuzzy_type() == fs.FUZZY_SETS.t1

    def test_classifier_t2(self, dataset):
        """Test classifier with Type-2 fuzzy sets."""
        X_train, X_test, y_train, y_test = dataset
        clf = evf.BaseFuzzyRulesClassifier(
            nRules=10, nAnts=3,
            fuzzy_type=fs.FUZZY_SETS.t2,
            verbose=False
        )
        clf.fit(X_train, y_train, n_gen=5, pop_size=10)

        predictions = clf.predict(X_test)
        assert len(predictions) == len(y_test)
        assert clf.rule_base.fuzzy_type() == fs.FUZZY_SETS.t2


class TestClassifierWithPrecomputedVariables:
    """Tests for classifier with precomputed linguistic variables."""

    @pytest.fixture
    def dataset_with_variables(self):
        """Create dataset with precomputed fuzzy variables."""
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

        # Create fuzzy variables
        fuzzy_vars = utils.construct_partitions(X_train, fs.FUZZY_SETS.t1)

        return X_train, X_test, y_train, y_test, fuzzy_vars

    def test_classifier_with_precomputed_lvs(self, dataset_with_variables):
        """Test classifier using precomputed linguistic variables."""
        X_train, X_test, y_train, y_test, fuzzy_vars = dataset_with_variables

        clf = evf.BaseFuzzyRulesClassifier(
            nRules=10, nAnts=3,
            linguistic_variables=fuzzy_vars,
            verbose=False
        )
        clf.fit(X_train, y_train, n_gen=5, pop_size=10)

        predictions = clf.predict(X_test)
        assert len(predictions) == len(y_test)

    def test_classifier_uses_provided_variables(self, dataset_with_variables):
        """Test that classifier actually uses the provided variables."""
        X_train, X_test, y_train, y_test, fuzzy_vars = dataset_with_variables

        clf = evf.BaseFuzzyRulesClassifier(
            nRules=10, nAnts=3,
            linguistic_variables=fuzzy_vars,
            verbose=False
        )
        clf.fit(X_train, y_train, n_gen=5, pop_size=10)

        # The rule base should use the same antecedents
        rb_antecedents = clf.rule_base.get_antecedents()
        assert len(rb_antecedents) == len(fuzzy_vars)


class TestClassifierWithPandasInput:
    """Tests for classifier with pandas DataFrame input."""

    def test_fit_with_dataframe(self):
        """Test fitting with pandas DataFrame."""
        X, y = make_classification(
            n_samples=80,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            n_classes=2,
            random_state=42
        )
        X_df = pd.DataFrame(X, columns=['f1', 'f2', 'f3', 'f4'])
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y, test_size=0.25, random_state=42
        )

        clf = evf.BaseFuzzyRulesClassifier(nRules=10, nAnts=3, verbose=False)
        clf.fit(X_train, y_train, n_gen=5, pop_size=10)

        predictions = clf.predict(X_test)
        assert len(predictions) == len(y_test)

    def test_column_names_preserved(self):
        """Test that column names are preserved from DataFrame."""
        X, y = make_classification(
            n_samples=80,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            n_classes=2,
            random_state=42
        )
        column_names = ['Temperature', 'Pressure', 'Humidity', 'Wind']
        X_df = pd.DataFrame(X, columns=column_names)

        clf = evf.BaseFuzzyRulesClassifier(nRules=10, nAnts=3, verbose=False)
        clf.fit(X_df, y, n_gen=5, pop_size=10)

        # Variable names should be from DataFrame columns
        antecedents = clf.rule_base.get_antecedents()
        for i, ant in enumerate(antecedents):
            assert ant.name == column_names[i]


class TestClassifierMulticlass:
    """Tests for multiclass classification."""

    @pytest.fixture
    def multiclass_dataset(self):
        """Load Iris dataset for multiclass testing."""
        X, y = load_iris(return_X_y=True)
        return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    def test_multiclass_fit_predict(self, multiclass_dataset):
        """Test multiclass classification."""
        X_train, X_test, y_train, y_test = multiclass_dataset

        clf = evf.BaseFuzzyRulesClassifier(nRules=15, nAnts=4, verbose=False)
        clf.fit(X_train, y_train, n_gen=10, pop_size=20)

        predictions = clf.predict(X_test)

        assert len(predictions) == len(y_test)
        assert clf.nclasses_ == 3
        assert all(p in [0, 1, 2] for p in predictions)

    def test_multiclass_predict_proba(self, multiclass_dataset):
        """Test multiclass probability prediction."""
        X_train, X_test, y_train, y_test = multiclass_dataset

        clf = evf.BaseFuzzyRulesClassifier(nRules=15, nAnts=4, verbose=False)
        clf.fit(X_train, y_train, n_gen=10, pop_size=20)

        proba = clf.predict_proba(X_test)

        assert proba.shape == (len(y_test), 3)  # 3 classes
        row_sums = np.sum(proba, axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(len(y_test)), decimal=5)


class TestClassifierReproducibility:
    """Tests for classifier reproducibility."""

    def test_same_random_state_same_results(self):
        """Test that same random_state produces same results."""
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

        # Train two classifiers with same random state
        clf1 = evf.BaseFuzzyRulesClassifier(nRules=10, nAnts=3, verbose=False)
        clf1.fit(X_train, y_train, n_gen=5, pop_size=10, random_state=123)
        pred1 = clf1.predict(X_test)

        clf2 = evf.BaseFuzzyRulesClassifier(nRules=10, nAnts=3, verbose=False)
        clf2.fit(X_train, y_train, n_gen=5, pop_size=10, random_state=123)
        pred2 = clf2.predict(X_test)

        np.testing.assert_array_equal(pred1, pred2)


class TestClassifierWithDifferentParameters:
    """Tests for classifier with different parameter configurations."""

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
        return train_test_split(X, y, test_size=0.25, random_state=42)

    def test_different_n_linguistic_variables(self, dataset):
        """Test with different numbers of linguistic variables."""
        X_train, X_test, y_train, y_test = dataset

        for n_lv in [2, 3, 5]:
            clf = evf.BaseFuzzyRulesClassifier(
                nRules=10, nAnts=3,
                n_linguistic_variables=n_lv,
                verbose=False
            )
            clf.fit(X_train, y_train, n_gen=5, pop_size=10)

            predictions = clf.predict(X_test)
            assert len(predictions) == len(y_test)

    def test_different_tolerance(self, dataset):
        """Test with different tolerance values."""
        X_train, X_test, y_train, y_test = dataset

        for tolerance in [0.0, 0.01, 0.1]:
            clf = evf.BaseFuzzyRulesClassifier(
                nRules=10, nAnts=3,
                tolerance=tolerance,
                verbose=False
            )
            clf.fit(X_train, y_train, n_gen=5, pop_size=10)

            predictions = clf.predict(X_test)
            assert len(predictions) == len(y_test)

    def test_ds_mode_options(self, dataset):
        """Test different dominance score modes."""
        X_train, X_test, y_train, y_test = dataset

        for ds_mode in [0, 1, 2]:
            clf = evf.BaseFuzzyRulesClassifier(
                nRules=10, nAnts=3,
                ds_mode=ds_mode,
                verbose=False
            )
            clf.fit(X_train, y_train, n_gen=5, pop_size=10)

            predictions = clf.predict(X_test)
            assert len(predictions) == len(y_test)


class TestClassifierRuleBase:
    """Tests for rule base inspection and manipulation."""

    @pytest.fixture
    def trained_classifier(self):
        """Create a trained classifier."""
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
        return clf

    def test_get_rulebase(self, trained_classifier):
        """Test getting rule base matrix."""
        rb_matrix = trained_classifier.get_rulebase()
        assert isinstance(rb_matrix, list)

    def test_print_rules(self, trained_classifier):
        """Test printing rules."""
        rules_str = trained_classifier.print_rules(return_rules=True)
        assert isinstance(rules_str, str)
        assert len(rules_str) > 0
        assert 'IF' in rules_str or 'Rules for' in rules_str

    def test_load_master_rule_base(self, trained_classifier):
        """Test loading a master rule base."""
        original_rb = trained_classifier.rule_base

        new_clf = evf.BaseFuzzyRulesClassifier(nRules=10, nAnts=3, verbose=False)
        new_clf.load_master_rule_base(original_rb)

        assert new_clf.rule_base is original_rb
        assert new_clf.nRules == len(original_rb.get_rules())


class TestClassifierCustomLoss:
    """Tests for custom loss functions."""

    def test_customized_loss(self):
        """Test using a custom loss function."""
        X, y = make_classification(
            n_samples=80,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            n_classes=2,
            random_state=42
        )

        def custom_loss(ruleBase, X, y, tolerance, alpha=0.0, beta=0.0, precomputed_truth=None):
            # Simple custom loss: just return accuracy
            predictions = ruleBase.predict(X)
            return np.mean(predictions == y)

        clf = evf.BaseFuzzyRulesClassifier(nRules=10, nAnts=3, verbose=False)
        clf.customized_loss(custom_loss)
        clf.fit(X, y, n_gen=5, pop_size=10)

        predictions = clf.predict(X)
        assert len(predictions) == len(y)

    def test_reparametrice_loss(self):
        """Test loss function reparametrization."""
        clf = evf.BaseFuzzyRulesClassifier(nRules=10, nAnts=3, verbose=False)
        clf.reparametrice_loss(alpha=0.1, beta=0.2)

        assert clf.alpha_ == 0.1
        assert clf.beta_ == 0.2


class TestClassifierCallable:
    """Tests for callable interface."""

    def test_classifier_callable(self):
        """Test calling classifier directly."""
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

        clf = evf.BaseFuzzyRulesClassifier(nRules=10, nAnts=3, verbose=False)
        clf.fit(X_train, y_train, n_gen=5, pop_size=10)

        # Call directly
        predictions_call = clf(X_test)
        predictions_predict = clf.predict(X_test)

        np.testing.assert_array_equal(predictions_call, predictions_predict)


class TestExplainablePrediction:
    """Tests for explainable prediction functionality."""

    def test_explainable_predict(self):
        """Test explainable prediction output."""
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

        clf = evf.BaseFuzzyRulesClassifier(nRules=10, nAnts=3, verbose=False)
        clf.fit(X_train, y_train, n_gen=10, pop_size=20)

        result = clf.explainable_predict(X_test)

        # Should return tuple with multiple elements
        assert isinstance(result, tuple)
        assert len(result) >= 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
