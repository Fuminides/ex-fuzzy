"""
Test suite for the conformal prediction module.

Tests the ConformalFuzzyClassifier class and related functions
for coverage guarantees, prediction sets, and rule-wise explanations.
"""
import pytest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ex_fuzzy', 'ex_fuzzy'))

import evolutionary_fit as evf
from conformal import ConformalFuzzyClassifier, evaluate_conformal_coverage


class TestConformalFuzzyClassifier:
    """Tests for ConformalFuzzyClassifier class."""

    @pytest.fixture
    def iris_data(self):
        """Prepare Iris dataset split into train, calibration, and test sets."""
        X, y = load_iris(return_X_y=True)
        # First split: train vs temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=42, stratify=y
        )
        # Second split: cal vs test
        X_cal, X_test, y_cal, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        return X_train, y_train, X_cal, y_cal, X_test, y_test

    @pytest.fixture
    def trained_classifier(self, iris_data):
        """Create a pre-trained fuzzy classifier."""
        X_train, y_train, _, _, _, _ = iris_data
        clf = evf.BaseFuzzyRulesClassifier(nRules=10, nAnts=3, verbose=False)
        clf.fit(X_train, y_train, n_gen=10, pop_size=20)
        return clf

    def test_init_with_classifier(self, trained_classifier):
        """Test initialization with existing trained classifier."""
        conf_clf = ConformalFuzzyClassifier(trained_classifier)
        assert conf_clf.clf is trained_classifier
        assert conf_clf._owns_clf is False
        assert conf_clf.score_type == 'membership'

    def test_init_with_nrules(self):
        """Test initialization with nRules parameter."""
        conf_clf = ConformalFuzzyClassifier(nRules=15, nAnts=4)
        assert conf_clf.clf.nRules == 15
        assert conf_clf.clf.nAnts == 4
        assert conf_clf._owns_clf is True

    def test_init_default(self):
        """Test initialization with default parameters."""
        conf_clf = ConformalFuzzyClassifier()
        assert conf_clf.clf.nRules == 30  # Default
        assert conf_clf._owns_clf is True

    def test_init_invalid_score_type(self, trained_classifier):
        """Test that invalid score_type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown score_type"):
            ConformalFuzzyClassifier(trained_classifier, score_type='invalid')

    def test_wrapper_mode(self, iris_data, trained_classifier):
        """Test wrapping an existing trained classifier."""
        X_train, y_train, X_cal, y_cal, X_test, y_test = iris_data

        # Wrap with conformal
        conf_clf = ConformalFuzzyClassifier(trained_classifier)
        conf_clf.calibrate(X_cal, y_cal)

        # Verify calibration happened
        assert conf_clf._calibrated is True

        # Predict sets
        pred_sets = conf_clf.predict_set(X_test, alpha=0.1)
        assert len(pred_sets) == len(X_test)
        assert all(isinstance(s, set) for s in pred_sets)

    def test_creation_mode_with_separate_calibration(self, iris_data):
        """Test creating classifier from parameters with separate cal set."""
        X_train, y_train, X_cal, y_cal, X_test, y_test = iris_data

        conf_clf = ConformalFuzzyClassifier(nRules=10, nAnts=3, verbose=False)
        conf_clf.fit(X_train, y_train, X_cal, y_cal, n_gen=10, pop_size=20)

        assert conf_clf._calibrated is True
        pred_sets = conf_clf.predict_set(X_test, alpha=0.1)
        assert len(pred_sets) == len(X_test)

    def test_creation_mode_auto_split(self, iris_data):
        """Test creating classifier with automatic calibration split."""
        X_train, y_train, X_cal, y_cal, X_test, y_test = iris_data

        # Combine train and cal for this test
        X_combined = np.vstack([X_train, X_cal])
        y_combined = np.concatenate([y_train, y_cal])

        conf_clf = ConformalFuzzyClassifier(nRules=10, nAnts=3, verbose=False)
        conf_clf.fit(X_combined, y_combined, cal_size=0.25, n_gen=10, pop_size=20)

        assert conf_clf._calibrated is True
        pred_sets = conf_clf.predict_set(X_test, alpha=0.1)
        assert len(pred_sets) == len(X_test)

    def test_calibrate_without_fit_raises(self):
        """Test that calibrate() raises error if classifier not fitted."""
        conf_clf = ConformalFuzzyClassifier(nRules=10)
        X_cal = np.random.rand(20, 4)
        y_cal = np.random.randint(0, 3, 20)

        with pytest.raises(ValueError, match="Classifier not fitted"):
            conf_clf.calibrate(X_cal, y_cal)

    def test_predict_without_calibrate_raises(self, trained_classifier):
        """Test that predict_set() raises error if not calibrated."""
        conf_clf = ConformalFuzzyClassifier(trained_classifier)
        X_test = np.random.rand(10, 4)

        with pytest.raises(ValueError, match="not calibrated"):
            conf_clf.predict_set(X_test, alpha=0.1)

    def test_predict_delegates_to_classifier(self, iris_data, trained_classifier):
        """Test that standard predict() delegates to underlying classifier."""
        X_train, y_train, X_cal, y_cal, X_test, y_test = iris_data

        conf_clf = ConformalFuzzyClassifier(trained_classifier)
        conf_clf.calibrate(X_cal, y_cal)

        # Compare predictions
        conf_preds = conf_clf.predict(X_test)
        clf_preds = trained_classifier.predict(X_test)
        np.testing.assert_array_equal(conf_preds, clf_preds)

    def test_coverage_guarantee(self, iris_data):
        """Test that empirical coverage >= 1-alpha (approximately)."""
        X_train, y_train, X_cal, y_cal, X_test, y_test = iris_data

        conf_clf = ConformalFuzzyClassifier(nRules=15, nAnts=4, verbose=False)
        conf_clf.fit(X_train, y_train, X_cal, y_cal, n_gen=20, pop_size=30)

        alpha = 0.1
        metrics = evaluate_conformal_coverage(conf_clf, X_test, y_test, alpha=alpha)

        # Coverage should be close to or above 1-alpha
        # Allow margin for finite sample effects
        assert metrics['coverage'] >= (1 - alpha) - 0.15, \
            f"Coverage {metrics['coverage']:.3f} too low (expected >= {1-alpha-0.15:.3f})"

    def test_prediction_sets_format(self, iris_data, trained_classifier):
        """Test that prediction sets have correct format."""
        X_train, y_train, X_cal, y_cal, X_test, y_test = iris_data

        conf_clf = ConformalFuzzyClassifier(trained_classifier)
        conf_clf.calibrate(X_cal, y_cal)

        pred_sets = conf_clf.predict_set(X_test, alpha=0.1)

        # Check format
        assert isinstance(pred_sets, list)
        assert len(pred_sets) == len(X_test)
        for ps in pred_sets:
            assert isinstance(ps, set)
            # All elements should be valid class indices
            for elem in ps:
                assert isinstance(elem, (int, np.integer))
                assert 0 <= elem < conf_clf.nclasses_

    def test_rule_wise_predictions(self, iris_data):
        """Test rule-wise conformal predictions."""
        X_train, y_train, X_cal, y_cal, X_test, y_test = iris_data

        conf_clf = ConformalFuzzyClassifier(nRules=10, nAnts=3, verbose=False)
        conf_clf.fit(X_train, y_train, X_cal, y_cal, n_gen=10, pop_size=20)

        results = conf_clf.predict_set_with_rules(X_test[:5], alpha=0.1)

        assert len(results) == 5
        for r in results:
            assert 'prediction_set' in r
            assert 'rule_contributions' in r
            assert 'class_p_values' in r
            assert isinstance(r['prediction_set'], set)
            assert isinstance(r['rule_contributions'], list)
            assert isinstance(r['class_p_values'], dict)

            # Check rule contributions format
            for contrib in r['rule_contributions']:
                assert 'rule_index' in contrib
                assert 'class' in contrib
                assert 'firing_strength' in contrib
                assert 'rule_confidence' in contrib
                assert 0 <= contrib['firing_strength'] <= 1
                assert 0 <= contrib['rule_confidence'] <= 1

    def test_different_score_types(self, iris_data):
        """Test different nonconformity score types."""
        X_train, y_train, X_cal, y_cal, X_test, y_test = iris_data

        for score_type in ['membership', 'association', 'entropy']:
            conf_clf = ConformalFuzzyClassifier(
                nRules=10, nAnts=3, score_type=score_type, verbose=False
            )
            conf_clf.fit(X_train, y_train, X_cal, y_cal, n_gen=10, pop_size=20)

            pred_sets = conf_clf.predict_set(X_test, alpha=0.1)
            assert len(pred_sets) == len(X_test)
            assert conf_clf.score_type == score_type

    def test_alpha_effect_on_set_size(self, iris_data, trained_classifier):
        """Test that larger alpha leads to smaller prediction sets."""
        X_train, y_train, X_cal, y_cal, X_test, y_test = iris_data

        conf_clf = ConformalFuzzyClassifier(trained_classifier)
        conf_clf.calibrate(X_cal, y_cal)

        # Smaller alpha = more conservative = larger sets
        pred_sets_01 = conf_clf.predict_set(X_test, alpha=0.01)
        pred_sets_10 = conf_clf.predict_set(X_test, alpha=0.10)
        pred_sets_30 = conf_clf.predict_set(X_test, alpha=0.30)

        avg_size_01 = np.mean([len(s) for s in pred_sets_01])
        avg_size_10 = np.mean([len(s) for s in pred_sets_10])
        avg_size_30 = np.mean([len(s) for s in pred_sets_30])

        # On average, sets should get smaller as alpha increases
        assert avg_size_01 >= avg_size_10 - 0.5  # Allow small tolerance
        assert avg_size_10 >= avg_size_30 - 0.5

    def test_get_calibration_info(self, iris_data, trained_classifier):
        """Test get_calibration_info method."""
        X_train, y_train, X_cal, y_cal, X_test, y_test = iris_data

        conf_clf = ConformalFuzzyClassifier(trained_classifier)

        # Before calibration
        info = conf_clf.get_calibration_info()
        assert info == {}

        # After calibration
        conf_clf.calibrate(X_cal, y_cal)
        info = conf_clf.get_calibration_info()

        assert 'n_calibration_samples' in info
        assert 'samples_per_class' in info
        assert 'score_type' in info
        assert 'n_rules_calibrated' in info

        assert info['n_calibration_samples'] == len(X_cal)
        assert info['score_type'] == 'membership'

    def test_properties_delegate_to_classifier(self, iris_data, trained_classifier):
        """Test that properties correctly delegate to underlying classifier."""
        X_train, y_train, X_cal, y_cal, X_test, y_test = iris_data

        conf_clf = ConformalFuzzyClassifier(trained_classifier)
        conf_clf.calibrate(X_cal, y_cal)

        assert conf_clf.rule_base is trained_classifier.rule_base
        assert conf_clf.nclasses_ == trained_classifier.nclasses_


class TestEvaluateConformalCoverage:
    """Tests for evaluate_conformal_coverage function."""

    @pytest.fixture
    def calibrated_classifier(self):
        """Create a calibrated conformal classifier."""
        X, y = load_iris(return_X_y=True)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=42, stratify=y
        )
        X_cal, X_test, y_cal, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        conf_clf = ConformalFuzzyClassifier(nRules=15, nAnts=4, verbose=False)
        conf_clf.fit(X_train, y_train, X_cal, y_cal, n_gen=15, pop_size=25)

        return conf_clf, X_test, y_test

    def test_metrics_format(self, calibrated_classifier):
        """Test that metrics have correct format."""
        conf_clf, X_test, y_test = calibrated_classifier

        metrics = evaluate_conformal_coverage(conf_clf, X_test, y_test, alpha=0.1)

        expected_keys = [
            'coverage', 'expected_coverage', 'avg_set_size',
            'efficiency', 'empty_sets', 'singleton_sets', 'coverage_by_class'
        ]
        for key in expected_keys:
            assert key in metrics

        # Check value ranges
        assert 0 <= metrics['coverage'] <= 1
        assert metrics['expected_coverage'] == 0.9  # 1 - alpha
        assert metrics['avg_set_size'] >= 0
        assert 0 <= metrics['empty_sets'] <= 1
        assert 0 <= metrics['singleton_sets'] <= 1

    def test_coverage_by_class(self, calibrated_classifier):
        """Test per-class coverage calculation."""
        conf_clf, X_test, y_test = calibrated_classifier

        metrics = evaluate_conformal_coverage(conf_clf, X_test, y_test, alpha=0.1)

        # Should have coverage for each class in test set
        unique_classes = np.unique(y_test)
        for c in unique_classes:
            assert int(c) in metrics['coverage_by_class']
            assert 0 <= metrics['coverage_by_class'][int(c)] <= 1


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_sample_prediction(self):
        """Test prediction on a single sample."""
        X, y = load_iris(return_X_y=True)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        X_cal, X_test, y_cal, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

        conf_clf = ConformalFuzzyClassifier(nRules=10, nAnts=3, verbose=False)
        conf_clf.fit(X_train, y_train, X_cal, y_cal, n_gen=10, pop_size=20)

        # Predict on single sample
        single_sample = X_test[0:1]
        pred_sets = conf_clf.predict_set(single_sample, alpha=0.1)

        assert len(pred_sets) == 1
        assert isinstance(pred_sets[0], set)

    def test_extreme_alpha_values(self):
        """Test prediction with extreme alpha values."""
        X, y = load_iris(return_X_y=True)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        X_cal, X_test, y_cal, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

        conf_clf = ConformalFuzzyClassifier(nRules=10, nAnts=3, verbose=False)
        conf_clf.fit(X_train, y_train, X_cal, y_cal, n_gen=10, pop_size=20)

        # Very small alpha - should give large prediction sets
        pred_sets_small = conf_clf.predict_set(X_test, alpha=0.001)
        avg_size_small = np.mean([len(s) for s in pred_sets_small])

        # Very large alpha - should give small prediction sets
        pred_sets_large = conf_clf.predict_set(X_test, alpha=0.99)
        avg_size_large = np.mean([len(s) for s in pred_sets_large])

        # Small alpha should give larger sets on average
        assert avg_size_small >= avg_size_large

    def test_print_rules(self):
        """Test that print_rules delegates correctly."""
        X, y = load_iris(return_X_y=True)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        X_cal, X_test, y_cal, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

        conf_clf = ConformalFuzzyClassifier(nRules=10, nAnts=3, verbose=False)
        conf_clf.fit(X_train, y_train, X_cal, y_cal, n_gen=10, pop_size=20)

        # Should not raise
        rules_str = conf_clf.print_rules(return_rules=True)
        assert isinstance(rules_str, str)
        assert len(rules_str) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
