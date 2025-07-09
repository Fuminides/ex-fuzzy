"""
Comprehensive test suite for ex_fuzzy.pattern_stability module.

This test suite provides complete coverage for:
- Pattern frequency analysis and counting
- Variable usage tracking and analysis
- Stability metrics and statistical measures
- Multi-run experiment execution
- Visualization components
- Pattern stabilizer class functionality
- Utility functions for pattern analysis
- Integration with fuzzy classifiers and rule bases

Test Categories:
    - Unit tests for individual functions
    - Integration tests for complete workflows
    - Performance tests for large datasets
    - Error handling and edge cases
    - Visualization and plotting tests
    - Statistical analysis validation
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
import warnings
import tempfile
import os

# Import the module under test
from ex_fuzzy import pattern_stability as ps
from ex_fuzzy import fuzzy_sets as fs
from ex_fuzzy import rules as rl
from ex_fuzzy import evolutionary_fit as evf


class TestUtilityFunctions:
    """Test utility functions for pattern analysis."""
    
    def test_add_dicts_empty_dicts(self):
        """Test adding empty dictionaries."""
        dict1 = {}
        dict2 = {}
        result = ps.add_dicts(dict1, dict2)
        assert result == {}
        assert dict1 == {}
    
    def test_add_dicts_existing_keys(self):
        """Test adding dictionaries with existing keys."""
        dict1 = {'a': 1, 'b': 2}
        dict2 = {'a': 3, 'b': 4}
        result = ps.add_dicts(dict1, dict2)
        
        assert result == {'a': 4, 'b': 6}
        assert dict1 == {'a': 4, 'b': 6}  # dict1 is modified
    
    def test_add_dicts_new_keys(self):
        """Test adding dictionaries with new keys."""
        dict1 = {'a': 1}
        dict2 = {'b': 2, 'c': 3}
        result = ps.add_dicts(dict1, dict2)
        
        assert result == {'a': 1, 'b': 2, 'c': 3}
        assert dict1 == {'a': 1, 'b': 2, 'c': 3}
    
    def test_add_dicts_mixed_keys(self):
        """Test adding dictionaries with mixed existing and new keys."""
        dict1 = {'a': 1, 'b': 2}
        dict2 = {'b': 3, 'c': 4, 'd': 5}
        result = ps.add_dicts(dict1, dict2)
        
        assert result == {'a': 1, 'b': 5, 'c': 4, 'd': 5}
    
    def test_concatenate_dicts_empty_dicts(self):
        """Test concatenating empty dictionaries."""
        dict1 = {}
        dict2 = {}
        result = ps.concatenate_dicts(dict1, dict2)
        assert result == {}
    
    def test_concatenate_dicts_existing_keys(self):
        """Test concatenating dictionaries with existing keys."""
        dict1 = {'a': 1, 'b': 2}
        dict2 = {'a': 3, 'b': 4}
        result = ps.concatenate_dicts(dict1, dict2)
        
        # Should not overwrite existing keys
        assert result == {'a': 1, 'b': 2}
    
    def test_concatenate_dicts_new_keys(self):
        """Test concatenating dictionaries with new keys."""
        dict1 = {'a': 1}
        dict2 = {'b': 2, 'c': 3}
        result = ps.concatenate_dicts(dict1, dict2)
        
        assert result == {'a': 1, 'b': 2, 'c': 3}
    
    def test_str_rule_as_list_simple(self):
        """Test string rule conversion to list."""
        rule_str = "1 2 3"
        result = ps.str_rule_as_list(rule_str)
        assert result == [1, 2, 3]
    
    def test_str_rule_as_list_with_brackets(self):
        """Test string rule conversion with brackets."""
        rule_str = "[1 2 3]"
        result = ps.str_rule_as_list(rule_str)
        assert result == [1, 2, 3]
    
    def test_str_rule_as_list_with_parentheses(self):
        """Test string rule conversion with parentheses."""
        rule_str = "(1 2 3)"
        result = ps.str_rule_as_list(rule_str)
        assert result == [1, 2, 3]
    
    def test_str_rule_as_list_with_dots(self):
        """Test string rule conversion with dots."""
        rule_str = "1.0 2.0 3.0"
        result = ps.str_rule_as_list(rule_str)
        assert result == [10, 20, 30]
    
    def test_str_rule_as_list_complex(self):
        """Test string rule conversion with complex formatting."""
        rule_str = "[(1.0 2.0 3.0)]"
        result = ps.str_rule_as_list(rule_str)
        assert result == [10, 20, 30]
    
    def test_str_rule_as_list_empty(self):
        """Test string rule conversion with empty string."""
        rule_str = ""
        result = ps.str_rule_as_list(rule_str)
        assert result == []


class TestPatternStabilizer:
    """Test the pattern_stabilizer class."""
    
    def test_init_basic(self, sample_data):
        """Test basic initialization."""
        X, y = sample_data
        stabilizer = ps.pattern_stabilizer(X, y)
        
        assert stabilizer.X is X
        assert stabilizer.y is y
        assert stabilizer.nRules == 30
        assert stabilizer.nAnts == 4
        assert stabilizer.fuzzy_type == fs.FUZZY_SETS.t1
        assert stabilizer.tolerance == 0.0
        assert stabilizer.verbose == False
    
    def test_init_with_parameters(self, sample_data):
        """Test initialization with custom parameters."""
        X, y = sample_data
        class_names = ['Class1', 'Class2']
        
        stabilizer = ps.pattern_stabilizer(
            X, y, 
            nRules=20, 
            nAnts=3, 
            fuzzy_type=fs.FUZZY_SETS.t2,
            tolerance=0.1,
            class_names=class_names,
            n_linguistic_variables=5,
            verbose=True
        )
        
        assert stabilizer.nRules == 20
        assert stabilizer.nAnts == 3
        assert stabilizer.fuzzy_type == fs.FUZZY_SETS.t2
        assert stabilizer.tolerance == 0.1
        assert stabilizer.classes_names == class_names
        assert stabilizer.n_linguist_variables == 5
        assert stabilizer.verbose == True
    
    def test_init_with_linguistic_variables(self, sample_data, sample_fuzzy_variables):
        """Test initialization with precomputed linguistic variables."""
        X, y = sample_data
        
        stabilizer = ps.pattern_stabilizer(
            X, y,
            linguistic_variables=sample_fuzzy_variables
        )
        
        assert stabilizer.lvs == sample_fuzzy_variables
        assert stabilizer.domain is None
        assert stabilizer.fuzzy_type == sample_fuzzy_variables[0].fuzzy_type()
        assert isinstance(stabilizer.n_linguist_variables, list)
    
    def test_init_class_names_inference(self, sample_data):
        """Test automatic class name inference."""
        X, y = sample_data
        stabilizer = ps.pattern_stabilizer(X, y)
        
        expected_classes = list(np.unique(y))
        assert stabilizer.classes_names == expected_classes
    
    def test_init_numpy_class_names(self, sample_data):
        """Test initialization with numpy array class names."""
        X, y = sample_data
        class_names = np.array(['A', 'B'])
        
        stabilizer = ps.pattern_stabilizer(X, y, class_names=class_names)
        assert stabilizer.classes_names == ['A', 'B']
    
    def test_init_stratify_parameter(self, sample_data):
        """Test initialization with stratification parameter."""
        X, y = sample_data
        stabilizer = ps.pattern_stabilizer(X, y, stratify_by='participant')
        
        assert stabilizer.stratify_by == 'participant'
    
    @patch('ex_fuzzy.evolutionary_fit.BaseFuzzyRulesClassifier')
    @patch('sklearn.model_selection.train_test_split')
    def test_generate_solutions_basic(self, mock_split, mock_classifier, sample_data):
        """Test basic solution generation."""
        X, y = sample_data
        stabilizer = ps.pattern_stabilizer(X, y)
        
        # Mock the train_test_split
        mock_split.return_value = (X[:80], X[80:], y[:80], y[80:])
        
        # Mock the classifier
        mock_classifier_instance = Mock()
        mock_classifier_instance.rule_base = Mock()
        mock_classifier_instance.forward.return_value = y[80:]
        mock_classifier.return_value = mock_classifier_instance
        
        rule_bases, accs = stabilizer.generate_solutions(n=2, n_gen=5, pop_size=5)
        
        assert len(rule_bases) == 2
        assert len(accs) == 2
        assert mock_classifier.call_count == 2
    
    @patch('ex_fuzzy.evolutionary_fit.BaseFuzzyRulesClassifier')
    def test_generate_solutions_with_stratification(self, mock_classifier, sample_data):
        """Test solution generation with stratification."""
        X, y = sample_data
        # Add stratification column
        X_strat = X.copy()
        X_strat['participant'] = [0, 0, 1, 1] * 25  # 100 samples
        
        stabilizer = ps.pattern_stabilizer(X_strat, y, stratify_by='participant')
        
        # Mock the classifier
        mock_classifier_instance = Mock()
        mock_classifier_instance.rule_base = Mock()
        mock_classifier_instance.forward.return_value = [0, 1] * 8  # Matching test size
        mock_classifier.return_value = mock_classifier_instance
        
        rule_bases, accs = stabilizer.generate_solutions(n=2, n_gen=5, pop_size=5, test_size=0.33)
        
        assert len(rule_bases) == 2
        assert len(accs) == 2
        assert mock_classifier.call_count == 2
    
    def test_count_unique_patterns_empty(self):
        """Test pattern counting with empty rule base."""
        X, y = np.array([[1, 2], [3, 4]]), np.array([0, 1])
        stabilizer = ps.pattern_stabilizer(X, y)
        
        # Mock empty rule base
        mock_rule_base = Mock()
        mock_rule_base.get_rulebase_matrix.return_value = []
        
        unique_patterns, patterns_ds, var_used = stabilizer.count_unique_patterns(mock_rule_base)
        
        assert unique_patterns == {}
        assert patterns_ds == {}
        assert var_used == {}
    
    def test_count_unique_patterns_single_rule(self):
        """Test pattern counting with single rule."""
        X, y = np.array([[1, 2], [3, 4]]), np.array([0, 1])
        stabilizer = ps.pattern_stabilizer(X, y)
        
        # Mock rule base with single rule
        mock_rule_base = Mock()
        mock_rule = Mock()
        mock_rule.score = 0.8
        mock_rule_base.get_rulebase_matrix.return_value = [[1, 2, 3]]
        mock_rule_base.__getitem__.return_value = mock_rule
        
        unique_patterns, patterns_ds, var_used = stabilizer.count_unique_patterns(mock_rule_base)
        
        assert len(unique_patterns) == 1
        assert "[1, 2, 3]" in unique_patterns
        assert unique_patterns["[1, 2, 3]"] == 1
        assert patterns_ds["[1, 2, 3]"] == 0.8
        assert len(var_used) == 3
    
    def test_count_unique_patterns_multiple_rules(self):
        """Test pattern counting with multiple rules."""
        X, y = np.array([[1, 2], [3, 4]]), np.array([0, 1])
        stabilizer = ps.pattern_stabilizer(X, y)
        
        # Mock rule base with multiple rules
        mock_rule_base = Mock()
        mock_rule1 = Mock()
        mock_rule1.score = 0.8
        mock_rule2 = Mock()
        mock_rule2.score = 0.6
        mock_rule3 = Mock()
        mock_rule3.score = 0.9
        
        mock_rule_base.get_rulebase_matrix.return_value = [[1, 2], [3, 4], [1, 2]]
        mock_rule_base.__getitem__.side_effect = [mock_rule1, mock_rule2, mock_rule3]
        
        unique_patterns, patterns_ds, var_used = stabilizer.count_unique_patterns(mock_rule_base)
        
        assert len(unique_patterns) == 2
        assert unique_patterns["[1, 2]"] == 2
        assert unique_patterns["[3, 4]"] == 1
        assert patterns_ds["[1, 2]"] == 0.8  # First occurrence
        assert patterns_ds["[3, 4]"] == 0.6
    
    def test_count_unique_patterns_all_classes_empty(self):
        """Test pattern counting for all classes with empty rule base."""
        X, y = np.array([[1, 2], [3, 4]]), np.array([0, 1])
        stabilizer = ps.pattern_stabilizer(X, y)
        
        # Mock empty master rule base
        mock_master_rule_base = Mock()
        mock_master_rule_base.__len__.return_value = 2
        mock_master_rule_base.__iter__.return_value = [[], []]
        mock_master_rule_base.n_linguistic_variables.return_value = [3, 3]
        
        class_patterns, patterns_dss, class_vars = stabilizer.count_unique_patterns_all_classes(mock_master_rule_base)
        
        assert len(class_patterns) == 2
        assert len(patterns_dss) == 2
        assert len(class_vars) == 2
        assert class_patterns[0] == {}
        assert class_patterns[1] == {}
    
    @patch.object(ps.pattern_stabilizer, 'count_unique_patterns')
    def test_count_unique_patterns_all_classes_with_rules(self, mock_count, sample_data):
        """Test pattern counting for all classes with rules."""
        X, y = sample_data
        stabilizer = ps.pattern_stabilizer(X, y)
        
        # Mock return values
        mock_count.return_value = (
            {'[1, 2]': 2}, 
            {'[1, 2]': 0.8}, 
            {0: {1: 1, 2: 1}, 1: {1: 1, 2: 1}}
        )
        
        # Mock master rule base
        mock_master_rule_base = Mock()
        mock_master_rule_base.__len__.return_value = 2
        mock_rule_base = Mock()
        mock_rule_base.__len__.return_value = 1
        mock_master_rule_base.__iter__.return_value = [mock_rule_base, mock_rule_base]
        mock_master_rule_base.n_linguistic_variables.return_value = [3, 3]
        
        class_patterns, patterns_dss, class_vars = stabilizer.count_unique_patterns_all_classes(mock_master_rule_base)
        
        assert len(class_patterns) == 2
        assert mock_count.call_count == 2
    
    @patch.object(ps.pattern_stabilizer, 'generate_solutions')
    def test_get_patterns_scores_basic(self, mock_generate, sample_data):
        """Test pattern scores generation."""
        X, y = sample_data
        stabilizer = ps.pattern_stabilizer(X, y)
        
        # Mock generated solutions
        mock_rule_bases = [Mock(), Mock()]
        mock_accuracies = [0.8, 0.7]
        mock_generate.return_value = (mock_rule_bases, mock_accuracies)
        
        # Mock master rule base
        for rule_base in mock_rule_bases:
            rule_base.__len__.return_value = 2
            rule_base.__iter__.return_value = [Mock(), Mock()]
            rule_base.n_linguistic_variables.return_value = [3, 3]
        
        result = stabilizer.get_patterns_scores(n=2, n_gen=10, pop_size=10)
        class_patterns, patterns_dss, class_vars, accuracies, rule_bases = result
        
        assert len(class_patterns) == 2
        assert len(patterns_dss) == 2
        assert len(class_vars) == 2
        assert accuracies == mock_accuracies
        assert rule_bases == mock_rule_bases
    
    @patch.object(ps.pattern_stabilizer, 'generate_solutions')
    def test_get_patterns_scores_with_faults(self, mock_generate, sample_data):
        """Test pattern scores generation with faulty solutions."""
        X, y = sample_data
        stabilizer = ps.pattern_stabilizer(X, y)
        
        # Mock generated solutions with empty rule base
        mock_rule_bases = [Mock(), []]  # Second one is empty
        mock_accuracies = [0.8, 0.7]
        mock_generate.return_value = (mock_rule_bases, mock_accuracies)
        
        # Mock first rule base
        mock_rule_bases[0].__len__.return_value = 2
        mock_rule_bases[0].__iter__.return_value = [Mock(), Mock()]
        mock_rule_bases[0].n_linguistic_variables.return_value = [3, 3]
        
        with patch('builtins.print') as mock_print:
            result = stabilizer.get_patterns_scores(n=2, n_gen=10, pop_size=10)
            mock_print.assert_called()
    
    @patch.object(ps.pattern_stabilizer, 'get_patterns_scores')
    @patch.object(ps.pattern_stabilizer, 'text_report')
    def test_stability_report(self, mock_text_report, mock_get_patterns, sample_data):
        """Test stability report generation."""
        X, y = sample_data
        stabilizer = ps.pattern_stabilizer(X, y)
        
        # Mock pattern scores
        mock_data = ({}, {}, {}, [0.8, 0.7], [Mock(), Mock()])
        mock_get_patterns.return_value = mock_data
        
        stabilizer.stability_report(n=2, n_gen=10, pop_size=10)
        
        mock_get_patterns.assert_called_once_with(2, n_gen=10, pop_size=10, test_size=0.33)
        mock_text_report.assert_called_once()
        
        # Check that attributes are set
        assert hasattr(stabilizer, 'class_patterns')
        assert hasattr(stabilizer, 'patterns_dss')
        assert hasattr(stabilizer, 'class_vars')
        assert hasattr(stabilizer, 'accuracies')
        assert hasattr(stabilizer, 'rule_bases')
    
    def test_var_reports_basic(self, sample_data):
        """Test variable reports generation."""
        X, y = sample_data
        stabilizer = ps.pattern_stabilizer(X, y)
        stabilizer.n = 10
        
        # Mock class variables
        class_vars = {
            0: {0: {0: 5, 1: 3, -1: 0}, 1: {0: 2, 1: 4, -1: 0}},
            1: {0: {0: 1, 1: 7, -1: 0}, 1: {0: 6, 1: 2, -1: 0}}
        }
        
        # Mock antecedents
        mock_antecedent = Mock()
        mock_antecedent.name = 'Variable1'
        mock_linguistic_var = Mock()
        mock_linguistic_var.name = 'Low'
        mock_antecedent.__getitem__.return_value = mock_linguistic_var
        
        antecedents = [mock_antecedent, mock_antecedent]
        
        with patch('builtins.print') as mock_print:
            stabilizer.var_reports(class_vars, antecedents, cutoff=10)
            mock_print.assert_called()
    
    def test_var_reports_with_cutoff(self, sample_data):
        """Test variable reports with cutoff."""
        X, y = sample_data
        stabilizer = ps.pattern_stabilizer(X, y)
        stabilizer.n = 10
        
        # Mock class variables with more entries
        class_vars = {
            0: {0: {i: 1 for i in range(15)}}  # 15 entries
        }
        class_vars[0][0][-1] = 0  # Set -1 to 0
        
        # Mock antecedents
        mock_antecedent = Mock()
        mock_antecedent.name = 'Variable1'
        mock_linguistic_var = Mock()
        mock_linguistic_var.name = 'Low'
        mock_antecedent.__getitem__.return_value = mock_linguistic_var
        
        antecedents = [mock_antecedent]
        
        with patch('builtins.print') as mock_print:
            stabilizer.var_reports(class_vars, antecedents, cutoff=3)
            # Should only print first 3 entries (excluding -1)
            print_calls = mock_print.call_args_list
            content_calls = [call for call in print_calls if call[0] and 'appears' in str(call[0][0])]
            assert len(content_calls) <= 4  # 3 entries + header
    
    @patch('builtins.print')
    @patch('ex_fuzzy.rules.generate_rule_string')
    def test_text_report_basic(self, mock_generate_rule, mock_print, sample_data):
        """Test text report generation."""
        X, y = sample_data
        stabilizer = ps.pattern_stabilizer(X, y)
        stabilizer.n = 10
        
        # Mock data
        class_patterns = {0: {'[1, 2]': 5, '[3, 4]': 3}}
        patterns_dss = {0: {'[1, 2]': 0.8, '[3, 4]': 0.6}}
        class_vars = {0: {0: {0: 2, 1: 3}}}
        accuracies = [0.8, 0.7, 0.9]
        
        # Mock rule bases
        mock_rule_base = Mock()
        mock_rule_base.antecedents = [Mock()]
        rule_bases = [mock_rule_base]
        
        mock_generate_rule.return_value = "IF Variable1 is Low THEN ..."
        
        with patch.object(stabilizer, 'var_reports') as mock_var_reports:
            stabilizer.text_report(class_patterns, patterns_dss, class_vars, accuracies, rule_bases, rule_cutoff=5)
            
            # Check that print was called with expected content
            mock_print.assert_called()
            mock_var_reports.assert_called()
    
    def test_text_report_with_cutoff(self, sample_data):
        """Test text report with rule cutoff."""
        X, y = sample_data
        stabilizer = ps.pattern_stabilizer(X, y)
        stabilizer.n = 10
        
        # Mock data with many patterns
        class_patterns = {0: {f'[{i}, {i+1}]': 1 for i in range(10)}}
        patterns_dss = {0: {f'[{i}, {i+1}]': 0.5 for i in range(10)}}
        class_vars = {0: {0: {0: 1}}}
        accuracies = [0.8]
        rule_bases = [Mock()]
        
        with patch('builtins.print') as mock_print:
            with patch.object(stabilizer, 'var_reports'):
                with patch('ex_fuzzy.rules.generate_rule_string', return_value='Rule'):
                    stabilizer.text_report(class_patterns, patterns_dss, class_vars, accuracies, rule_bases, rule_cutoff=3)
                    
                    # Should only process first 3 rules
                    print_calls = mock_print.call_args_list
                    rule_calls = [call for call in print_calls if call[0] and 'Pattern' in str(call[0][0])]
                    assert len(rule_calls) <= 3


class TestVisualization:
    """Test visualization components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create sample data for visualization tests
        self.X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        self.y = np.array([0, 1, 0, 1])
        
        # Create stabilizer with mock data
        self.stabilizer = ps.pattern_stabilizer(self.X, self.y)
        self.stabilizer.n = 10
        
        # Mock rule bases and class variables
        self.mock_rule_base = Mock()
        self.mock_antecedent = Mock()
        self.mock_antecedent.name = 'Variable1'
        self.mock_linguistic_var = Mock()
        self.mock_linguistic_var.name = 'Low'
        self.mock_antecedent.__getitem__.return_value = self.mock_linguistic_var
        self.mock_rule_base.antecedents = [self.mock_antecedent]
        
        self.stabilizer.rule_bases = [[self.mock_rule_base, self.mock_rule_base]]
        self.stabilizer.classes_names = ['Class0', 'Class1']
        self.stabilizer.class_vars = {
            0: {0: {0: 5, 1: 3, -1: 0}},
            1: {0: {0: 2, 1: 4, -1: 0}}
        }
        self.stabilizer.n_linguist_variables = [2, 2]
    
    @patch('matplotlib.pyplot.show')
    def test_pie_chart_basic(self, mock_show):
        """Test basic pie chart generation."""
        self.stabilizer.pie_chart_basic(var_ix=0, class_ix=0)
        mock_show.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    def test_pie_chart_basic_no_data(self, mock_show):
        """Test pie chart with no data."""
        self.stabilizer.class_vars = {0: {0: {-1: 10}}}  # Only -1 values
        self.stabilizer.pie_chart_basic(var_ix=0, class_ix=0)
        mock_show.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    @patch.object(ps.pattern_stabilizer, 'gen_colormap')
    def test_pie_chart_var(self, mock_colormap, mock_show):
        """Test variable pie chart generation."""
        mock_colormap.return_value = {'Low': 'red', 'High': 'blue'}
        self.stabilizer.pie_chart_var(var_ix=0)
        mock_show.assert_called_once()
        mock_colormap.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    @patch.object(ps.pattern_stabilizer, 'gen_colormap')
    def test_pie_chart_class(self, mock_colormap, mock_show):
        """Test class pie chart generation."""
        mock_colormap.return_value = {'Low': 'red', 'High': 'blue'}
        self.stabilizer.pie_chart_class(class_ix=0)
        mock_show.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    @patch.object(ps.pattern_stabilizer, 'gen_colormap')
    def test_pie_chart_class_with_var_list(self, mock_colormap, mock_show):
        """Test class pie chart with variable list."""
        mock_colormap.return_value = {'Low': 'red', 'High': 'blue'}
        self.stabilizer.pie_chart_class(class_ix=0, var_list=[0])
        mock_show.assert_called_once()
    
    def test_gen_colormap_two_variables(self):
        """Test colormap generation for 2 linguistic variables."""
        self.stabilizer.n_linguist_variables = [2, 2]
        
        # Mock antecedents
        mock_antecedent = Mock()
        mock_lv1 = Mock()
        mock_lv1.name = 'Low'
        mock_lv2 = Mock()
        mock_lv2.name = 'High'
        mock_antecedent.__iter__.return_value = [mock_lv1, mock_lv2]
        
        antecedents = [mock_antecedent]
        
        colors = self.stabilizer.gen_colormap(antecedents)
        
        assert isinstance(colors, dict)
        assert len(colors) == 2
        assert 'Low' in colors
        assert 'High' in colors
        assert colors['Low'] == '#FA8072'
        assert colors['High'] == 'Green'
    
    def test_gen_colormap_three_variables(self):
        """Test colormap generation for 3 linguistic variables."""
        self.stabilizer.n_linguist_variables = [3, 3]
        
        # Mock antecedents
        mock_antecedent = Mock()
        mock_lv1 = Mock()
        mock_lv1.name = 'Low'
        mock_lv2 = Mock()
        mock_lv2.name = 'Medium'
        mock_lv3 = Mock()
        mock_lv3.name = 'High'
        mock_antecedent.__iter__.return_value = [mock_lv1, mock_lv2, mock_lv3]
        
        antecedents = [mock_antecedent]
        
        colors = self.stabilizer.gen_colormap(antecedents)
        
        assert isinstance(colors, dict)
        assert len(colors) == 3
        assert 'Low' in colors
        assert 'Medium' in colors
        assert 'High' in colors
        assert colors['Low'] == '#FA8072'
        assert colors['Medium'] == '#EEE8AA'
        assert colors['High'] == 'Green'
    
    @patch('matplotlib.colormaps')
    def test_gen_colormap_many_variables(self, mock_colormaps):
        """Test colormap generation for many linguistic variables."""
        self.stabilizer.n_linguist_variables = [5, 5]
        
        # Mock colormap
        mock_colormap = Mock()
        mock_colormap.return_value = (1.0, 0.0, 0.0, 1.0)  # Red color
        mock_colormaps.__getitem__.return_value = mock_colormap
        
        # Mock antecedents
        mock_antecedent = Mock()
        mock_antecedent.name = 'Variable1'
        antecedents = [mock_antecedent]
        
        colors = self.stabilizer.gen_colormap(antecedents)
        
        assert isinstance(colors, dict)


class TestIntegration:
    """Test integration between components."""
    
    def test_full_workflow_mock(self, sample_data):
        """Test full workflow with mocked components."""
        X, y = sample_data
        
        with patch('ex_fuzzy.evolutionary_fit.BaseFuzzyRulesClassifier') as mock_classifier:
            # Mock classifier
            mock_classifier_instance = Mock()
            mock_rule_base = Mock()
            mock_rule_base.get_rulebase_matrix.return_value = [[1, 2], [3, 4]]
            mock_rule_base.__getitem__.side_effect = [Mock(score=0.8), Mock(score=0.6)]
            mock_classifier_instance.rule_base = [mock_rule_base]
            mock_classifier_instance.forward.return_value = y[80:]
            mock_classifier.return_value = mock_classifier_instance
            
            stabilizer = ps.pattern_stabilizer(X, y)
            
            with patch('sklearn.model_selection.train_test_split') as mock_split:
                mock_split.return_value = (X[:80], X[80:], y[:80], y[80:])
                
                with patch('builtins.print'):
                    stabilizer.stability_report(n=2, n_gen=5, pop_size=5)
                
                # Check that all components were called
                assert mock_classifier.called
                assert mock_split.called
                assert hasattr(stabilizer, 'class_patterns')
                assert hasattr(stabilizer, 'accuracies')
    
    def test_pattern_analysis_workflow(self, sample_data):
        """Test pattern analysis workflow."""
        X, y = sample_data
        stabilizer = ps.pattern_stabilizer(X, y)
        
        # Mock rule base with patterns
        mock_rule_base = Mock()
        mock_rule_base.get_rulebase_matrix.return_value = [[1, 2], [1, 2], [3, 4]]
        mock_rule1 = Mock(score=0.8)
        mock_rule2 = Mock(score=0.7)
        mock_rule3 = Mock(score=0.9)
        mock_rule_base.__getitem__.side_effect = [mock_rule1, mock_rule2, mock_rule3]
        
        unique_patterns, patterns_ds, var_used = stabilizer.count_unique_patterns(mock_rule_base)
        
        # Check pattern counting
        assert len(unique_patterns) == 2
        assert unique_patterns["[1, 2]"] == 2
        assert unique_patterns["[3, 4]"] == 1
        
        # Check dominance scores
        assert patterns_ds["[1, 2]"] == 0.8  # First occurrence
        assert patterns_ds["[3, 4]"] == 0.9
        
        # Check variable usage
        assert 0 in var_used
        assert 1 in var_used
    
    def test_statistical_analysis_workflow(self, sample_data):
        """Test statistical analysis workflow."""
        X, y = sample_data
        stabilizer = ps.pattern_stabilizer(X, y)
        
        # Mock multiple rule bases for statistical analysis
        mock_rule_bases = []
        for i in range(5):
            mock_rule_base = Mock()
            mock_rule_base.get_rulebase_matrix.return_value = [[1, 2], [3, 4]]
            mock_rule_base.__getitem__.side_effect = [Mock(score=0.8), Mock(score=0.6)]
            mock_rule_bases.append([mock_rule_base])
        
        # Mock master rule base
        mock_master_rule_base = Mock()
        mock_master_rule_base.__len__.return_value = 1
        mock_master_rule_base.__iter__.return_value = [Mock()]
        mock_master_rule_base.n_linguistic_variables.return_value = [3]
        
        # Test pattern aggregation across multiple runs
        class_patterns = None
        patterns_dss = None
        class_vars = None
        
        for rule_base in mock_rule_bases:
            if class_patterns is None:
                class_patterns, patterns_dss, class_vars = stabilizer.count_unique_patterns_all_classes(rule_base[0])
            else:
                class_patterns, patterns_dss, class_vars = stabilizer.count_unique_patterns_all_classes(
                    rule_base[0], class_patterns, patterns_dss, class_vars
                )
        
        # Check aggregation
        assert len(class_patterns) >= 1
        assert len(patterns_dss) >= 1
        assert len(class_vars) >= 1


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_parameters(self):
        """Test handling of invalid parameters."""
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        
        # Test with invalid nRules
        with pytest.raises((ValueError, TypeError)):
            ps.pattern_stabilizer(X, y, nRules=-1)
        
        # Test with invalid nAnts
        with pytest.raises((ValueError, TypeError)):
            ps.pattern_stabilizer(X, y, nAnts=0)
    
    def test_empty_data(self):
        """Test handling of empty data."""
        X = np.array([]).reshape(0, 2)
        y = np.array([])
        
        stabilizer = ps.pattern_stabilizer(X, y)
        assert stabilizer.X.shape == (0, 2)
        assert stabilizer.y.shape == (0,)
    
    def test_mismatched_data_dimensions(self):
        """Test handling of mismatched data dimensions."""
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1, 2])  # Mismatched length
        
        # Should still initialize but may cause issues later
        stabilizer = ps.pattern_stabilizer(X, y)
        assert stabilizer.X.shape[0] != stabilizer.y.shape[0]
    
    def test_rule_base_edge_cases(self, sample_data):
        """Test edge cases in rule base handling."""
        X, y = sample_data
        stabilizer = ps.pattern_stabilizer(X, y)
        
        # Test with None rule base
        with pytest.raises(AttributeError):
            stabilizer.count_unique_patterns(None)
        
        # Test with rule base that has no rules
        mock_rule_base = Mock()
        mock_rule_base.get_rulebase_matrix.return_value = []
        
        unique_patterns, patterns_ds, var_used = stabilizer.count_unique_patterns(mock_rule_base)
        assert unique_patterns == {}
        assert patterns_ds == {}
        assert var_used == {}
    
    def test_visualization_edge_cases(self, sample_data):
        """Test edge cases in visualization."""
        X, y = sample_data
        stabilizer = ps.pattern_stabilizer(X, y)
        
        # Mock empty class variables
        stabilizer.class_vars = {0: {0: {-1: 10}}}  # Only -1 values
        stabilizer.rule_bases = [[Mock()]]
        stabilizer.classes_names = ['Class0']
        stabilizer.n_linguist_variables = [2]
        
        # Should handle empty data gracefully
        with patch('matplotlib.pyplot.show'):
            stabilizer.pie_chart_basic(var_ix=0, class_ix=0)
    
    def test_string_conversion_edge_cases(self):
        """Test edge cases in string conversion."""
        # Test empty string
        result = ps.str_rule_as_list("")
        assert result == []
        
        # Test string with only separators
        result = ps.str_rule_as_list("[]().")
        assert result == []
        
        # Test string with mixed format
        result = ps.str_rule_as_list("[(1.0) 2.0] (3.0)")
        assert result == [10, 20, 30]


class TestPerformance:
    """Test performance characteristics."""
    
    def test_large_pattern_count(self):
        """Test handling of large number of patterns."""
        X = np.random.rand(1000, 10)
        y = np.random.randint(0, 3, 1000)
        
        stabilizer = ps.pattern_stabilizer(X, y)
        
        # Mock rule base with many patterns
        mock_rule_base = Mock()
        patterns = [[i, i+1, i+2] for i in range(1000)]
        mock_rule_base.get_rulebase_matrix.return_value = patterns
        
        # Mock rules with scores
        mock_rules = [Mock(score=0.5) for _ in range(1000)]
        mock_rule_base.__getitem__.side_effect = mock_rules
        
        # Should handle large number of patterns
        unique_patterns, patterns_ds, var_used = stabilizer.count_unique_patterns(mock_rule_base)
        
        assert len(unique_patterns) == 1000
        assert len(patterns_ds) == 1000
        assert len(var_used) == 3
    
    def test_memory_efficiency(self):
        """Test memory efficiency with repeated patterns."""
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        
        stabilizer = ps.pattern_stabilizer(X, y)
        
        # Mock rule base with repeated patterns
        mock_rule_base = Mock()
        patterns = [[1, 2, 3]] * 1000  # Same pattern repeated
        mock_rule_base.get_rulebase_matrix.return_value = patterns
        
        # Mock rules with scores
        mock_rules = [Mock(score=0.5) for _ in range(1000)]
        mock_rule_base.__getitem__.side_effect = mock_rules
        
        unique_patterns, patterns_ds, var_used = stabilizer.count_unique_patterns(mock_rule_base)
        
        # Should efficiently handle repeated patterns
        assert len(unique_patterns) == 1
        assert unique_patterns["[1, 2, 3]"] == 1000
        assert len(patterns_ds) == 1
        assert len(var_used) == 3


class TestDocumentation:
    """Test documentation and examples."""
    
    def test_module_docstring(self):
        """Test module docstring exists and is comprehensive."""
        assert ps.__doc__ is not None
        assert len(ps.__doc__) > 100
        assert "pattern stability" in ps.__doc__.lower()
        assert "fuzzy" in ps.__doc__.lower()
    
    def test_class_docstring(self):
        """Test class docstring exists and is comprehensive."""
        assert ps.pattern_stabilizer.__doc__ is not None
        assert len(ps.pattern_stabilizer.__doc__) > 50
    
    def test_function_docstrings(self):
        """Test function docstrings exist."""
        assert ps.add_dicts.__doc__ is None  # Simple utility function
        assert ps.concatenate_dicts.__doc__ is None  # Simple utility function
        assert ps.str_rule_as_list.__doc__ is None  # Simple utility function
        
        # Test method docstrings
        assert ps.pattern_stabilizer.generate_solutions.__doc__ is not None
        assert ps.pattern_stabilizer.count_unique_patterns.__doc__ is not None
        assert ps.pattern_stabilizer.stability_report.__doc__ is not None
    
    def test_parameter_documentation(self):
        """Test parameter documentation in docstrings."""
        # Test pattern_stabilizer.__init__ docstring
        init_doc = ps.pattern_stabilizer.__init__.__doc__
        assert init_doc is not None
        assert ":param" in init_doc
        assert "nRules" in init_doc
        assert "nAnts" in init_doc
        assert "fuzzy_type" in init_doc
    
    def test_return_documentation(self):
        """Test return value documentation."""
        # Test count_unique_patterns docstring
        method_doc = ps.pattern_stabilizer.count_unique_patterns.__doc__
        assert method_doc is not None
        assert ":return" in method_doc
        assert "unique_patterns" in method_doc
        assert "patterns_ds" in method_doc
        assert "var_used" in method_doc


if __name__ == '__main__':
    pytest.main([__file__])
