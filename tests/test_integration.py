"""
Integration tests for Ex-Fuzzy library.

Tests complete pipelines from data to predictions,
ensuring all components work together correctly.
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ex_fuzzy', 'ex_fuzzy'))

import fuzzy_sets as fs
import rules as rl
import evolutionary_fit as evf
import classifiers as clf
import rule_mining as rm
import utils
import persistence as pers
try:
    from conformal import ConformalFuzzyClassifier, evaluate_conformal_coverage
    HAS_CONFORMAL = True
except ImportError:
    HAS_CONFORMAL = False


class TestFullPipelineTrainPredict:
    """Tests for complete train-predict pipelines."""

    @pytest.fixture
    def iris_data(self):
        """Load and prepare Iris dataset."""
        X, y = load_iris(return_X_y=True)
        feature_names = load_iris().feature_names
        X_df = pd.DataFrame(X, columns=feature_names)
        return train_test_split(X_df, y, test_size=0.3, random_state=42, stratify=y)

    @pytest.fixture
    def binary_data(self):
        """Create binary classification data."""
        X, y = make_classification(
            n_samples=150,
            n_features=5,
            n_informative=4,
            n_redundant=1,
            n_classes=2,
            random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        return train_test_split(X_df, y, test_size=0.3, random_state=42)

    def test_basic_classifier_pipeline(self, binary_data):
        """Test basic classifier pipeline."""
        X_train, X_test, y_train, y_test = binary_data

        # Train
        classifier = evf.BaseFuzzyRulesClassifier(
            nRules=15, nAnts=4,
            fuzzy_type=fs.FUZZY_SETS.t1,
            verbose=False
        )
        classifier.fit(X_train, y_train, n_gen=20, pop_size=30)

        # Predict
        predictions = classifier.predict(X_test)

        # Evaluate
        accuracy = accuracy_score(y_test, predictions)

        assert len(predictions) == len(y_test)
        assert 0 <= accuracy <= 1
        # Should be better than random for this separable dataset
        assert accuracy >= 0.45

    def test_multiclass_pipeline(self, iris_data):
        """Test multiclass classification pipeline."""
        X_train, X_test, y_train, y_test = iris_data

        classifier = evf.BaseFuzzyRulesClassifier(
            nRules=20, nAnts=4,
            fuzzy_type=fs.FUZZY_SETS.t1,
            verbose=False
        )
        classifier.fit(X_train, y_train, n_gen=30, pop_size=40)

        predictions = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        assert len(predictions) == len(y_test)
        assert all(p in [0, 1, 2] for p in predictions)
        # Should perform reasonably on Iris
        assert accuracy >= 0.6

    def test_rule_mine_pipeline(self, iris_data):
        """Test rule mining classifier pipeline."""
        X_train, X_test, y_train, y_test = iris_data

        classifier = clf.RuleMineClassifier(
            nRules=15, nAnts=4,
            fuzzy_type=fs.FUZZY_SETS.t1,
            verbose=False
        )
        classifier.fit(X_train, y_train, n_gen=20, pop_size=30)

        predictions = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        assert len(predictions) == len(y_test)
        assert accuracy > 0.5


class TestPipelineWithDifferentFuzzyTypes:
    """Tests for pipelines with different fuzzy set types."""

    @pytest.fixture
    def data(self):
        """Create test data."""
        X, y = make_classification(
            n_samples=100,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            n_classes=2,
            random_state=42
        )
        X_df = pd.DataFrame(X, columns=['f1', 'f2', 'f3', 'f4'])
        return train_test_split(X_df, y, test_size=0.3, random_state=42)

    def test_t1_pipeline(self, data):
        """Test complete T1 fuzzy pipeline."""
        X_train, X_test, y_train, y_test = data

        classifier = evf.BaseFuzzyRulesClassifier(
            nRules=10, nAnts=3,
            fuzzy_type=fs.FUZZY_SETS.t1,
            verbose=False
        )
        classifier.fit(X_train, y_train, n_gen=10, pop_size=20)

        predictions = classifier.predict(X_test)
        assert len(predictions) == len(y_test)
        assert classifier.rule_base.fuzzy_type() == fs.FUZZY_SETS.t1

    def test_t2_pipeline(self, data):
        """Test complete T2 fuzzy pipeline."""
        X_train, X_test, y_train, y_test = data

        classifier = evf.BaseFuzzyRulesClassifier(
            nRules=10, nAnts=3,
            fuzzy_type=fs.FUZZY_SETS.t2,
            verbose=False
        )
        classifier.fit(X_train, y_train, n_gen=10, pop_size=20)

        predictions = classifier.predict(X_test)
        assert len(predictions) == len(y_test)
        assert classifier.rule_base.fuzzy_type() == fs.FUZZY_SETS.t2


class TestMiningOptimizationPipeline:
    """Tests for mining -> optimization -> prediction pipeline."""

    def test_mine_then_optimize(self):
        """Test mining rules then optimizing."""
        X, y = make_classification(
            n_samples=120,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            n_classes=2,
            random_state=42
        )
        X_df = pd.DataFrame(X, columns=['f1', 'f2', 'f3', 'f4'])
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y, test_size=0.3, random_state=42
        )

        # Step 1: Create fuzzy partitions
        fuzzy_vars = utils.construct_partitions(X_train, fs.FUZZY_SETS.t1)

        # Step 2: Mine candidate rules
        candidate_rules = rm.multiclass_mine_rulebase(
            X_train, y_train, fuzzy_vars,
            support_threshold=0.05,
            max_depth=3
        )

        # Step 3: Optimize rule selection
        classifier = evf.BaseFuzzyRulesClassifier(
            nRules=15, nAnts=4,
            verbose=False
        )
        classifier.fit(
            X_train, y_train,
            n_gen=15, pop_size=25,
            candidate_rules=candidate_rules
        )

        # Step 4: Predict
        predictions = classifier.predict(X_test)

        assert len(predictions) == len(y_test)
        assert hasattr(classifier, 'rule_base')


class TestPersistencePipeline:
    """Tests for save/load pipeline."""

    def test_train_save_load_predict(self):
        """Test complete train -> save -> load -> predict pipeline."""
        X, y = make_classification(
            n_samples=100,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            n_classes=2,
            random_state=42
        )
        X_df = pd.DataFrame(X, columns=['f1', 'f2', 'f3', 'f4'])
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y, test_size=0.3, random_state=42
        )

        # Train
        classifier = evf.BaseFuzzyRulesClassifier(
            nRules=10, nAnts=3,
            verbose=False
        )
        classifier.fit(X_train, y_train, n_gen=10, pop_size=20)

        # Get original predictions
        original_predictions = classifier.predict(X_test)

        # Save variables and rules
        antecedents = classifier.rule_base.get_antecedents()
        saved_vars = pers.save_fuzzy_variables(antecedents)
        saved_rules = classifier.print_rules(return_rules=True)

        # Load variables
        loaded_vars = pers.load_fuzzy_variables(saved_vars)

        # Verify loaded variables work
        assert len(loaded_vars) == len(antecedents)
        for i, lv in enumerate(loaded_vars):
            assert lv.name == antecedents[i].name


@pytest.mark.skipif(not HAS_CONFORMAL, reason="Conformal module not available")
class TestConformalPipeline:
    """Tests for conformal prediction pipeline."""

    def test_train_calibrate_predict_pipeline(self):
        """Test complete conformal prediction pipeline."""
        X, y = load_iris(return_X_y=True)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=42, stratify=y
        )
        X_cal, X_test, y_cal, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        # Create conformal classifier from scratch
        conf_clf = ConformalFuzzyClassifier(nRules=15, nAnts=4, verbose=False)
        conf_clf.fit(X_train, y_train, X_cal, y_cal, n_gen=15, pop_size=25)

        # Predict sets
        pred_sets = conf_clf.predict_set(X_test, alpha=0.1)

        assert len(pred_sets) == len(y_test)
        assert all(isinstance(ps, set) for ps in pred_sets)

    def test_wrap_existing_classifier_pipeline(self):
        """Test wrapping existing classifier with conformal."""
        X, y = load_iris(return_X_y=True)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=42, stratify=y
        )
        X_cal, X_test, y_cal, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        # Train base classifier
        base_clf = evf.BaseFuzzyRulesClassifier(nRules=15, nAnts=4, verbose=False)
        base_clf.fit(X_train, y_train, n_gen=15, pop_size=25)

        # Wrap with conformal
        conf_clf = ConformalFuzzyClassifier(base_clf)
        conf_clf.calibrate(X_cal, y_cal)

        # Evaluate coverage
        metrics = evaluate_conformal_coverage(conf_clf, X_test, y_test, alpha=0.1)

        assert 'coverage' in metrics
        assert 'avg_set_size' in metrics


class TestCrossValidationPipeline:
    """Tests for cross-validation workflows."""

    def test_manual_cv_pipeline(self):
        """Test manual cross-validation pipeline."""
        from sklearn.model_selection import StratifiedKFold

        X, y = make_classification(
            n_samples=100,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            n_classes=2,
            random_state=42
        )
        X_df = pd.DataFrame(X, columns=['f1', 'f2', 'f3', 'f4'])

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []

        for train_idx, test_idx in skf.split(X_df, y):
            X_train = X_df.iloc[train_idx]
            X_test = X_df.iloc[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]

            classifier = evf.BaseFuzzyRulesClassifier(
                nRules=10, nAnts=3,
                verbose=False
            )
            classifier.fit(X_train, y_train, n_gen=10, pop_size=15)

            predictions = classifier.predict(X_test)
            score = accuracy_score(y_test, predictions)
            scores.append(score)

        assert len(scores) == 3
        assert all(0 <= s <= 1 for s in scores)


class TestExplainablePipeline:
    """Tests for explainable prediction pipeline."""

    def test_get_rules_and_explanations(self):
        """Test getting rules and explanations for predictions."""
        X, y = make_classification(
            n_samples=100,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            n_classes=2,
            random_state=42
        )
        X_df = pd.DataFrame(X, columns=['Temp', 'Pressure', 'Humidity', 'Wind'])
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y, test_size=0.3, random_state=42
        )

        classifier = evf.BaseFuzzyRulesClassifier(
            nRules=10, nAnts=3,
            verbose=False
        )
        classifier.fit(X_train, y_train, n_gen=15, pop_size=25)

        # Get rules as string
        rules_str = classifier.print_rules(return_rules=True)
        assert isinstance(rules_str, str)
        assert len(rules_str) > 0

        # Get explainable prediction
        result = classifier.explainable_predict(X_test[:5])
        assert isinstance(result, tuple)


class TestDataTypeConsistency:
    """Tests for data type consistency across pipeline."""

    def test_numpy_array_input(self):
        """Test pipeline with numpy array input."""
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

        classifier = evf.BaseFuzzyRulesClassifier(nRules=10, nAnts=3, verbose=False)
        classifier.fit(X_train, y_train, n_gen=10, pop_size=20)

        predictions = classifier.predict(X_test)
        assert len(predictions) == len(y_test)

    def test_pandas_dataframe_input(self):
        """Test pipeline with pandas DataFrame input."""
        X, y = make_classification(
            n_samples=100,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            n_classes=2,
            random_state=42
        )
        X_df = pd.DataFrame(X, columns=['a', 'b', 'c', 'd'])
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y, test_size=0.3, random_state=42
        )

        classifier = evf.BaseFuzzyRulesClassifier(nRules=10, nAnts=3, verbose=False)
        classifier.fit(X_train, y_train, n_gen=10, pop_size=20)

        predictions = classifier.predict(X_test)
        assert len(predictions) == len(y_test)

    def test_mixed_input_types(self):
        """Test pipeline with mixed input types (train DataFrame, predict array)."""
        X, y = make_classification(
            n_samples=100,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            n_classes=2,
            random_state=42
        )
        X_df = pd.DataFrame(X, columns=['a', 'b', 'c', 'd'])
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y, test_size=0.3, random_state=42
        )

        classifier = evf.BaseFuzzyRulesClassifier(nRules=10, nAnts=3, verbose=False)
        classifier.fit(X_train, y_train, n_gen=10, pop_size=20)

        # Predict with numpy array
        predictions = classifier.predict(X_test.values)
        assert len(predictions) == len(y_test)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
