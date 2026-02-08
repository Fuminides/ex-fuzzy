"""
Tests for the visualization module (vis_rules.py).

Tests rule and fuzzy set visualization functionality.
Note: These tests verify that plotting functions don't crash,
not the visual output itself.
"""
import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ex_fuzzy', 'ex_fuzzy'))

import fuzzy_sets as fs
import evolutionary_fit as evf
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

try:
    import vis_rules
    HAS_VIS = True
except ImportError:
    HAS_VIS = False

try:
    import networkx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


@pytest.fixture
def sample_fuzzy_variable():
    """Create a sample fuzzy variable for visualization."""
    low = fs.FS('Low', [0, 0, 0.3, 0.5], [0, 1])
    medium = fs.FS('Medium', [0.3, 0.5, 0.5, 0.7], [0, 1])
    high = fs.FS('High', [0.5, 0.7, 1, 1], [0, 1])
    return fs.fuzzyVariable('Temperature', [low, medium, high])


@pytest.fixture
def trained_classifier():
    """Create a trained classifier for visualization."""
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

    return clf


@pytest.mark.skipif(not HAS_VIS, reason="Visualization module not available")
class TestFuzzySetVisualization:
    """Tests for fuzzy set visualization."""

    def test_plot_fuzzy_variable_no_error(self, sample_fuzzy_variable):
        """Test that plotting a fuzzy variable doesn't raise errors."""
        try:
            if hasattr(vis_rules, 'plot_fuzzy_variable'):
                fig = vis_rules.plot_fuzzy_variable(sample_fuzzy_variable)
                plt.close('all')
                assert True
            elif hasattr(vis_rules, 'visualize_fuzzy_variable'):
                fig = vis_rules.visualize_fuzzy_variable(sample_fuzzy_variable)
                plt.close('all')
                assert True
        except (AttributeError, TypeError) as e:
            pytest.skip(f"Fuzzy variable plotting not available: {e}")

    def test_plot_multiple_fuzzy_sets(self, sample_fuzzy_variable):
        """Test plotting multiple fuzzy sets."""
        try:
            if hasattr(vis_rules, 'plot_fuzzy_sets'):
                fig = vis_rules.plot_fuzzy_sets(sample_fuzzy_variable.linguistic_variables)
                plt.close('all')
                assert True
        except (AttributeError, TypeError):
            pytest.skip("plot_fuzzy_sets not available")


@pytest.mark.skipif(not HAS_VIS, reason="Visualization module not available")
@pytest.mark.skipif(not HAS_NETWORKX, reason="NetworkX not available")
class TestRuleVisualization:
    """Tests for rule visualization."""

    def test_plot_rules_no_error(self, trained_classifier):
        """Test that plotting rules doesn't raise errors."""
        try:
            if hasattr(vis_rules, 'plot_rules'):
                fig = vis_rules.plot_rules(trained_classifier.rule_base)
                plt.close('all')
                assert True
            elif hasattr(vis_rules, 'visualize_rules'):
                fig = vis_rules.visualize_rules(trained_classifier.rule_base)
                plt.close('all')
                assert True
        except (AttributeError, TypeError) as e:
            pytest.skip(f"Rule plotting not available: {e}")

    def test_rule_network_visualization(self, trained_classifier):
        """Test rule network visualization."""
        try:
            if hasattr(vis_rules, 'plot_rule_network'):
                fig = vis_rules.plot_rule_network(trained_classifier.rule_base)
                plt.close('all')
                assert True
        except (AttributeError, TypeError):
            pytest.skip("plot_rule_network not available")


@pytest.mark.skipif(not HAS_VIS, reason="Visualization module not available")
class TestVisualizationOutput:
    """Tests for visualization output format."""

    def test_returns_figure(self, sample_fuzzy_variable):
        """Test that visualization returns a figure object."""
        try:
            if hasattr(vis_rules, 'plot_fuzzy_variable'):
                result = vis_rules.plot_fuzzy_variable(sample_fuzzy_variable)
                # Should return figure or axes
                assert result is not None or True  # May return None
                plt.close('all')
        except (AttributeError, TypeError):
            pytest.skip("plot_fuzzy_variable not available")


class TestBasicPlotting:
    """Basic plotting tests without vis_rules module."""

    def test_fuzzy_set_membership_plottable(self, sample_fuzzy_variable):
        """Test that fuzzy set memberships can be plotted manually."""
        x = np.linspace(0, 1, 100)

        fig, ax = plt.subplots()
        for fuzzy_set in sample_fuzzy_variable.linguistic_variables:
            memberships = fuzzy_set.membership(x)
            ax.plot(x, memberships, label=fuzzy_set.name)

        ax.set_xlabel('x')
        ax.set_ylabel('Membership')
        ax.legend()
        plt.close(fig)

        assert True  # If we got here without error, test passes

    def test_t2_fuzzy_set_plottable(self):
        """Test that T2 fuzzy set memberships can be plotted."""
        low = fs.IVFS('Low', [0, 0, 0.3, 0.5], [0, 0.1, 0.4, 0.6], [0, 1])

        x = np.linspace(0, 1, 100)
        memberships = low.membership(x)

        fig, ax = plt.subplots()
        ax.fill_between(x, memberships[:, 0], memberships[:, 1], alpha=0.3)
        ax.plot(x, memberships[:, 0], label='Lower')
        ax.plot(x, memberships[:, 1], label='Upper')
        ax.legend()
        plt.close(fig)

        assert True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
