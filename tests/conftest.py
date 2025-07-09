"""
Test configuration and fixtures for the Ex-Fuzzy library test suite.

This module provides common fixtures and configuration for pytest,
including sample datasets, test parameters, and utility functions.
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.model_selection import train_test_split

# Import the library modules
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ex_fuzzy', 'ex_fuzzy'))

import fuzzy_sets as fs
import evolutionary_fit as evf
import utils
import rules as rl


@pytest.fixture(scope="session")
def iris_dataset():
    """Load and prepare the Iris dataset for testing."""
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': load_iris().feature_names,
        'target_names': load_iris().target_names
    }


@pytest.fixture(scope="session")
def binary_dataset():
    """Generate a binary classification dataset for testing."""
    X, y = make_classification(
        n_samples=200,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        n_clusters_per_class=1,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


@pytest.fixture(scope="session")
def regression_dataset():
    """Generate a regression dataset for testing."""
    X, y = make_regression(
        n_samples=200,
        n_features=4,
        noise=0.1,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


@pytest.fixture
def sample_fuzzy_sets():
    """Create sample fuzzy sets for testing."""
    # Type-1 fuzzy sets
    t1_low = fs.FS('Low', [0, 0, 0.3, 0.5], [0, 1])
    t1_medium = fs.FS('Medium', [0.3, 0.5, 0.5, 0.7], [0, 1])
    t1_high = fs.FS('High', [0.5, 0.7, 1, 1], [0, 1])
    
    # Type-2 fuzzy sets
    t2_low = fs.IVFS('Low_T2', [0, 0, 0.3, 0.5], [0, 0.1, 0.4, 0.6], [0, 1])
    t2_medium = fs.IVFS('Medium_T2', [0.3, 0.5, 0.5, 0.7], [0.2, 0.4, 0.6, 0.8], [0, 1])
    t2_high = fs.IVFS('High_T2', [0.5, 0.7, 1, 1], [0.4, 0.6, 1, 1], [0, 1])
    
    return {
        't1_sets': [t1_low, t1_medium, t1_high],
        't2_sets': [t2_low, t2_medium, t2_high]
    }


@pytest.fixture
def sample_fuzzy_variables(sample_fuzzy_sets):
    """Create sample fuzzy variables for testing."""
    t1_variable = fs.fuzzyVariable('Test_Variable_T1', sample_fuzzy_sets['t1_sets'])
    t2_variable = fs.fuzzyVariable('Test_Variable_T2', sample_fuzzy_sets['t2_sets'])
    
    return {
        't1_variable': t1_variable,
        't2_variable': t2_variable
    }


@pytest.fixture
def sample_rules():
    """Create sample rules for testing."""
    # Simple rules for testing
    rule1 = rl.RuleSimple([0, 1], consequent=0)
    rule2 = rl.RuleSimple([1, 2], consequent=1)
    rule3 = rl.RuleSimple([2, 0], consequent=2)
    
    return [rule1, rule2, rule3]


@pytest.fixture(params=[fs.FUZZY_SETS.t1, fs.FUZZY_SETS.t2])
def fuzzy_type(request):
    """Parametrized fixture for different fuzzy set types."""
    return request.param


@pytest.fixture
def test_parameters():
    """Common test parameters for classifiers."""
    return {
        'n_rules': 10,
        'n_antecedents': 3,
        'n_linguistic_variables': 3,
        'tolerance': 0.0,
        'n_gen': 5,  # Small for testing
        'pop_size': 10  # Small for testing
    }


class TestDataGenerator:
    """Utility class for generating test data."""
    
    @staticmethod
    def generate_random_data(n_samples=100, n_features=4, random_state=42):
        """Generate random data for testing."""
        np.random.seed(random_state)
        X = np.random.random((n_samples, n_features))
        y = np.random.randint(0, 3, n_samples)
        return X, y
    
    @staticmethod
    def generate_separable_data(n_samples=100, n_features=2, n_classes=2, random_state=42):
        """Generate linearly separable data for testing."""
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features,
            n_redundant=0,
            n_clusters_per_class=1,
            n_classes=n_classes,
            random_state=random_state
        )
        return X, y


# Test utility functions
def assert_fuzzy_set_properties(fuzzy_set, expected_name=None, expected_type=None):
    """Assert common properties of fuzzy sets."""
    if expected_name:
        assert fuzzy_set.name == expected_name
    if expected_type:
        assert fuzzy_set.type() == expected_type
    
    # Test that membership function works
    test_input = np.array([0.0, 0.5, 1.0])
    membership_values = fuzzy_set.membership(test_input)
    assert isinstance(membership_values, np.ndarray)
    assert len(membership_values) == len(test_input)
    
    # Test that membership values are in valid range
    if expected_type == fs.FUZZY_SETS.t1:
        assert np.all(membership_values >= 0) and np.all(membership_values <= 1)
    elif expected_type == fs.FUZZY_SETS.t2:
        assert membership_values.shape[1] == 2  # Lower and upper bounds
        assert np.all(membership_values >= 0) and np.all(membership_values <= 1)


def assert_classifier_properties(classifier, X_test, y_test, min_accuracy=0.0):
    """Assert common properties of trained classifiers."""
    # Test prediction capability
    predictions = classifier.predict(X_test)
    assert len(predictions) == len(y_test)
    
    # Test accuracy is reasonable
    accuracy = np.mean(predictions == y_test)
    assert accuracy >= min_accuracy
    
    # Test that classifier has rule base
    assert hasattr(classifier, 'rule_base')
    assert classifier.rule_base is not None


# Constants for testing
TEST_RANDOM_SEED = 42
SMALL_DATASET_SIZE = 50
MEDIUM_DATASET_SIZE = 200
LARGE_DATASET_SIZE = 1000

# Tolerance values for numerical comparisons
FLOAT_TOLERANCE = 1e-6
ACCURACY_TOLERANCE = 0.1
