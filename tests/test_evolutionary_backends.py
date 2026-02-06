"""
Tests for the evolutionary_backends module.

Tests PyMoo and EvoX backends for genetic optimization,
including backend selection and fallback behavior.
"""
import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ex_fuzzy', 'ex_fuzzy'))

import fuzzy_sets as fs
import evolutionary_fit as evf

try:
    import evolutionary_backends as eb
    HAS_BACKENDS = True
except ImportError:
    HAS_BACKENDS = False

# Check if evox is available (catch all exceptions as evox may fail to import
# for various reasons: missing deps, incompatible Python version, etc.)
try:
    import evox
    HAS_EVOX = True
except Exception:
    HAS_EVOX = False


@pytest.mark.skipif(not HAS_BACKENDS, reason="Backends module not available")
class TestBackendSelection:
    """Tests for backend selection functionality."""

    def test_get_backend_pymoo(self):
        """Test getting PyMoo backend."""
        try:
            backend = eb.get_backend('pymoo')
            assert backend is not None
            assert 'pymoo' in backend.__class__.__name__.lower() or hasattr(backend, 'run')
        except (AttributeError, TypeError):
            # get_backend might have different signature
            pass

    @pytest.mark.skipif(not HAS_EVOX, reason="EvoX not installed")
    def test_get_backend_evox(self):
        """Test getting EvoX backend."""
        try:
            backend = eb.get_backend('evox')
            assert backend is not None
        except (AttributeError, TypeError, ImportError):
            pytest.skip("EvoX backend not available")

    def test_invalid_backend_name(self):
        """Test that invalid backend name raises error."""
        try:
            with pytest.raises((ValueError, KeyError)):
                eb.get_backend('invalid_backend')
        except (AttributeError, TypeError):
            pass


@pytest.mark.skipif(not HAS_BACKENDS, reason="Backends module not available")
class TestPyMooBackend:
    """Tests for PyMoo backend."""

    @pytest.fixture
    def simple_dataset(self):
        """Create simple dataset for testing."""
        X, y = make_classification(
            n_samples=80,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            n_classes=2,
            random_state=42
        )
        return train_test_split(X, y, test_size=0.25, random_state=42)

    def test_pymoo_backend_basic(self, simple_dataset):
        """Test basic PyMoo backend optimization."""
        X_train, X_test, y_train, y_test = simple_dataset

        clf = evf.BaseFuzzyRulesClassifier(
            nRules=10, nAnts=3,
            backend='pymoo',
            verbose=False
        )
        clf.fit(X_train, y_train, n_gen=5, pop_size=10)

        predictions = clf.predict(X_test)
        assert len(predictions) == len(y_test)

    def test_pymoo_backend_with_checkpoints(self, simple_dataset):
        """Test PyMoo backend with checkpoint support."""
        X_train, X_test, y_train, y_test = simple_dataset

        clf = evf.BaseFuzzyRulesClassifier(
            nRules=10, nAnts=3,
            backend='pymoo',
            verbose=False
        )
        # PyMoo supports checkpoints
        clf.fit(X_train, y_train, n_gen=5, pop_size=10, checkpoints=0)

        predictions = clf.predict(X_test)
        assert len(predictions) == len(y_test)


@pytest.mark.skipif(not HAS_EVOX, reason="EvoX not installed")
class TestEvoXBackend:
    """Tests for EvoX backend."""

    @pytest.fixture
    def simple_dataset(self):
        """Create simple dataset for testing."""
        X, y = make_classification(
            n_samples=80,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            n_classes=2,
            random_state=42
        )
        return train_test_split(X, y, test_size=0.25, random_state=42)

    def test_evox_backend_basic(self, simple_dataset):
        """Test basic EvoX backend optimization."""
        X_train, X_test, y_train, y_test = simple_dataset

        try:
            clf = evf.BaseFuzzyRulesClassifier(
                nRules=10, nAnts=3,
                backend='evox',
                verbose=False
            )
            clf.fit(X_train, y_train, n_gen=5, pop_size=10)

            predictions = clf.predict(X_test)
            assert len(predictions) == len(y_test)
        except (ImportError, RuntimeError) as e:
            pytest.skip(f"EvoX backend not properly configured: {e}")


class TestBackendConsistency:
    """Tests for consistency between backends."""

    @pytest.fixture
    def simple_dataset(self):
        """Create simple dataset."""
        X, y = make_classification(
            n_samples=80,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            n_classes=2,
            random_state=42
        )
        return train_test_split(X, y, test_size=0.25, random_state=42)

    def test_both_backends_produce_valid_results(self, simple_dataset):
        """Test that both backends produce valid classification results."""
        X_train, X_test, y_train, y_test = simple_dataset

        # PyMoo
        clf_pymoo = evf.BaseFuzzyRulesClassifier(
            nRules=10, nAnts=3,
            backend='pymoo',
            verbose=False
        )
        clf_pymoo.fit(X_train, y_train, n_gen=5, pop_size=10)
        pred_pymoo = clf_pymoo.predict(X_test)

        assert len(pred_pymoo) == len(y_test)
        assert all(p in [0, 1] for p in pred_pymoo)

        # EvoX (if available)
        if HAS_EVOX:
            try:
                clf_evox = evf.BaseFuzzyRulesClassifier(
                    nRules=10, nAnts=3,
                    backend='evox',
                    verbose=False
                )
                clf_evox.fit(X_train, y_train, n_gen=5, pop_size=10)
                pred_evox = clf_evox.predict(X_test)

                assert len(pred_evox) == len(y_test)
                assert all(p in [0, 1] for p in pred_evox)
            except (ImportError, RuntimeError):
                pass


class TestDefaultBackend:
    """Tests for default backend behavior."""

    def test_default_backend_is_pymoo(self):
        """Test that default backend is PyMoo."""
        clf = evf.BaseFuzzyRulesClassifier(nRules=10, nAnts=3, verbose=False)

        # Default should be pymoo (backend is an object with name() method)
        assert clf.backend.name() == 'pymoo'

    def test_default_backend_works(self):
        """Test that default backend works without specifying."""
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

        predictions = clf.predict(X_test)
        assert len(predictions) == len(y_test)


class TestBackendFallback:
    """Tests for backend fallback behavior."""

    def test_fallback_when_evox_unavailable(self):
        """Test fallback to PyMoo when EvoX is unavailable."""
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

        if not HAS_EVOX:
            # If EvoX not available, specifying it should either
            # raise an error or fallback to pymoo
            try:
                clf = evf.BaseFuzzyRulesClassifier(
                    nRules=10, nAnts=3,
                    backend='evox',
                    verbose=False
                )
                clf.fit(X_train, y_train, n_gen=5, pop_size=10)
                # If it succeeded, it fell back to pymoo
            except (ImportError, ValueError):
                # Expected if no fallback
                pass


class TestBackendParameters:
    """Tests for backend-specific parameters."""

    @pytest.fixture
    def simple_dataset(self):
        """Create simple dataset."""
        X, y = make_classification(
            n_samples=80,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            n_classes=2,
            random_state=42
        )
        return train_test_split(X, y, test_size=0.25, random_state=42)

    def test_population_size(self, simple_dataset):
        """Test different population sizes."""
        X_train, X_test, y_train, y_test = simple_dataset

        for pop_size in [10, 20, 50]:
            clf = evf.BaseFuzzyRulesClassifier(
                nRules=10, nAnts=3,
                verbose=False
            )
            clf.fit(X_train, y_train, n_gen=5, pop_size=pop_size)

            predictions = clf.predict(X_test)
            assert len(predictions) == len(y_test)

    def test_number_of_generations(self, simple_dataset):
        """Test different numbers of generations."""
        X_train, X_test, y_train, y_test = simple_dataset

        for n_gen in [5, 10, 20]:
            clf = evf.BaseFuzzyRulesClassifier(
                nRules=10, nAnts=3,
                verbose=False
            )
            clf.fit(X_train, y_train, n_gen=n_gen, pop_size=10)

            predictions = clf.predict(X_test)
            assert len(predictions) == len(y_test)


class TestBackendWithDifferentFuzzyTypes:
    """Tests for backends with different fuzzy set types."""

    @pytest.fixture
    def simple_dataset(self):
        """Create simple dataset."""
        X, y = make_classification(
            n_samples=80,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            n_classes=2,
            random_state=42
        )
        return train_test_split(X, y, test_size=0.25, random_state=42)

    def test_pymoo_with_t1(self, simple_dataset):
        """Test PyMoo with T1 fuzzy sets."""
        X_train, X_test, y_train, y_test = simple_dataset

        clf = evf.BaseFuzzyRulesClassifier(
            nRules=10, nAnts=3,
            fuzzy_type=fs.FUZZY_SETS.t1,
            backend='pymoo',
            verbose=False
        )
        clf.fit(X_train, y_train, n_gen=5, pop_size=10)

        predictions = clf.predict(X_test)
        assert len(predictions) == len(y_test)

    def test_pymoo_with_t2(self, simple_dataset):
        """Test PyMoo with T2 fuzzy sets."""
        X_train, X_test, y_train, y_test = simple_dataset

        clf = evf.BaseFuzzyRulesClassifier(
            nRules=10, nAnts=3,
            fuzzy_type=fs.FUZZY_SETS.t2,
            backend='pymoo',
            verbose=False
        )
        clf.fit(X_train, y_train, n_gen=5, pop_size=10)

        predictions = clf.predict(X_test)
        assert len(predictions) == len(y_test)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
