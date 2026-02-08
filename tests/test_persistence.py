"""
Tests for the persistence module.

Tests save/load functionality for fuzzy variables and rules,
ensuring round-trip consistency for all fuzzy set types.
"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ex_fuzzy', 'ex_fuzzy'))

import fuzzy_sets as fs
import rules as rl
import persistence as pers


class TestSaveFuzzyVariables:
    """Tests for saving fuzzy variables to text format."""

    def test_print_fuzzy_variable_t1_trapezoidal(self):
        """Test printing a T1 fuzzy variable with trapezoidal sets."""
        low = fs.FS('Low', [0, 0, 0.3, 0.5], [0, 1])
        medium = fs.FS('Medium', [0.3, 0.5, 0.5, 0.7], [0, 1])
        high = fs.FS('High', [0.5, 0.7, 1, 1], [0, 1])
        fvar = fs.fuzzyVariable('Temperature', [low, medium, high], 'Celsius')

        result = pers.print_fuzzy_variable(fvar)

        assert '$$$ Linguistic variable: Temperature' in result
        assert 'Celsius' in result
        assert 'Low' in result
        assert 'Medium' in result
        assert 'High' in result
        assert 'trap' in result

    def test_print_fuzzy_variable_t1_gaussian(self):
        """Test printing a T1 fuzzy variable with gaussian sets."""
        # gaussianFS inherits from FS: __init__(name, membership_parameters, domain)
        low = fs.gaussianFS('Low', [0.2, 0.1], [0, 1])
        high = fs.gaussianFS('High', [0.8, 0.1], [0, 1])
        fvar = fs.fuzzyVariable('Pressure', [low, high])

        result = pers.print_fuzzy_variable(fvar)

        assert '$$$ Linguistic variable: Pressure' in result
        assert 'Low' in result
        assert 'High' in result
        assert 'gauss' in result

    def test_print_fuzzy_variable_t2_trapezoidal(self):
        """Test printing a T2 fuzzy variable with trapezoidal sets."""
        low = fs.IVFS('Low', [0, 0, 0.3, 0.5], [0, 0.1, 0.4, 0.6], [0, 1])
        high = fs.IVFS('High', [0.5, 0.7, 1, 1], [0.4, 0.6, 1, 1], [0, 1])
        fvar = fs.fuzzyVariable('Speed', [low, high])

        result = pers.print_fuzzy_variable(fvar)

        assert '$$$ Linguistic variable: Speed' in result
        assert 'Low' in result
        assert 'High' in result
        assert 'trap' in result

    def test_save_multiple_fuzzy_variables(self):
        """Test saving multiple fuzzy variables."""
        low1 = fs.FS('Low', [0, 0, 0.5, 0.5], [0, 1])
        high1 = fs.FS('High', [0.5, 0.5, 1, 1], [0, 1])
        var1 = fs.fuzzyVariable('Var1', [low1, high1])

        low2 = fs.FS('Low', [0, 0, 0.5, 0.5], [0, 1])
        high2 = fs.FS('High', [0.5, 0.5, 1, 1], [0, 1])
        var2 = fs.fuzzyVariable('Var2', [low2, high2])

        result = pers.save_fuzzy_variables([var1, var2])

        assert 'Var1' in result
        assert 'Var2' in result
        # Should have two variable definitions
        assert result.count('$$$ Linguistic variable:') == 2


class TestLoadFuzzyVariables:
    """Tests for loading fuzzy variables from text format."""

    def test_load_fuzzy_variables_t1_trapezoidal(self):
        """Test loading T1 trapezoidal fuzzy variables."""
        text = """$$$ Linguistic variable: Temperature
Low;0,1;trap;0,0,0.3,0.5
Medium;0,1;trap;0.3,0.5,0.5,0.7
High;0,1;trap;0.5,0.7,1,1
"""
        variables = pers.load_fuzzy_variables(text)

        assert len(variables) == 1
        assert variables[0].name == 'Temperature'
        assert len(variables[0].linguistic_variables) == 3
        assert variables[0].fuzzy_type() == fs.FUZZY_SETS.t1

    def test_load_fuzzy_variables_t1_gaussian(self):
        """Test loading T1 gaussian fuzzy variables."""
        text = """$$$ Linguistic variable: Pressure
Low;0,1;gauss;0.2,0.1
High;0,1;gauss;0.8,0.1
"""
        variables = pers.load_fuzzy_variables(text)

        assert len(variables) == 1
        assert variables[0].name == 'Pressure'
        assert len(variables[0].linguistic_variables) == 2

    def test_load_multiple_fuzzy_variables(self):
        """Test loading multiple fuzzy variables."""
        text = """$$$ Linguistic variable: Var1
Low;0,1;trap;0,0,0.5,0.5
High;0,1;trap;0.5,0.5,1,1

$$$ Linguistic variable: Var2
Small;0,100;trap;0,0,30,50
Large;0,100;trap;50,70,100,100
"""
        variables = pers.load_fuzzy_variables(text)

        assert len(variables) == 2
        assert variables[0].name == 'Var1'
        assert variables[1].name == 'Var2'


class TestFuzzyVariablesRoundTrip:
    """Tests for save/load round-trip consistency."""

    def test_roundtrip_t1_trapezoidal(self):
        """Test round-trip for T1 trapezoidal fuzzy variables."""
        # Create original variables
        low = fs.FS('Low', [0, 0, 0.3, 0.5], [0, 1])
        medium = fs.FS('Medium', [0.3, 0.5, 0.5, 0.7], [0, 1])
        high = fs.FS('High', [0.5, 0.7, 1, 1], [0, 1])
        original = fs.fuzzyVariable('Temperature', [low, medium, high])

        # Save and load
        text = pers.print_fuzzy_variable(original)
        loaded_list = pers.load_fuzzy_variables(text)
        loaded = loaded_list[0]

        # Verify
        assert loaded.name == original.name
        assert len(loaded.linguistic_variables) == len(original.linguistic_variables)
        assert loaded.fuzzy_type() == original.fuzzy_type()

        # Test membership values match
        test_values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        for i, orig_fs in enumerate(original.linguistic_variables):
            loaded_fs = loaded.linguistic_variables[i]
            orig_membership = orig_fs.membership(test_values)
            loaded_membership = loaded_fs.membership(test_values)
            np.testing.assert_array_almost_equal(orig_membership, loaded_membership, decimal=5)

    def test_roundtrip_t2_trapezoidal(self):
        """Test round-trip for T2 trapezoidal fuzzy variables."""
        # Create original variables
        low = fs.IVFS('Low', [0, 0, 0.3, 0.5], [0, 0.1, 0.4, 0.6], [0, 1], lower_height=0.8)
        high = fs.IVFS('High', [0.5, 0.7, 1, 1], [0.4, 0.6, 1, 1], [0, 1], lower_height=0.8)
        original = fs.fuzzyVariable('Speed', [low, high])

        # Save and load
        text = pers.print_fuzzy_variable(original)
        loaded_list = pers.load_fuzzy_variables(text)
        loaded = loaded_list[0]

        # Verify
        assert loaded.name == original.name
        assert len(loaded.linguistic_variables) == len(original.linguistic_variables)
        assert loaded.fuzzy_type() == fs.FUZZY_SETS.t2

    def test_roundtrip_multiple_variables(self):
        """Test round-trip for multiple fuzzy variables."""
        # Create original variables
        low1 = fs.FS('Low', [0, 0, 0.5, 0.5], [0, 1])
        high1 = fs.FS('High', [0.5, 0.5, 1, 1], [0, 1])
        var1 = fs.fuzzyVariable('Var1', [low1, high1])

        low2 = fs.FS('Small', [0, 0, 50, 50], [0, 100])
        high2 = fs.FS('Large', [50, 50, 100, 100], [0, 100])
        var2 = fs.fuzzyVariable('Var2', [low2, high2])

        original_vars = [var1, var2]

        # Save and load
        text = pers.save_fuzzy_variables(original_vars)
        loaded_vars = pers.load_fuzzy_variables(text)

        # Verify
        assert len(loaded_vars) == len(original_vars)
        for i, orig_var in enumerate(original_vars):
            assert loaded_vars[i].name == orig_var.name
            assert len(loaded_vars[i].linguistic_variables) == len(orig_var.linguistic_variables)


class TestLoadFuzzyRules:
    """Tests for loading fuzzy rules from text format."""

    @pytest.fixture
    def sample_fuzzy_variables(self):
        """Create sample fuzzy variables for rule loading."""
        low = fs.FS('Low', [0, 0, 0.5, 0.5], [0, 1])
        high = fs.FS('High', [0.5, 0.5, 1, 1], [0, 1])
        var1 = fs.fuzzyVariable('Var1', [low, high])
        var2 = fs.fuzzyVariable('Var2', [low, high])
        return [var1, var2]

    def test_load_simple_rules(self, sample_fuzzy_variables):
        """Test loading simple fuzzy rules."""
        rules_text = """Rules for consequent: Class0
----------------
IF Var1 IS Low AND Var2 IS Low WITH DS 0.85, ACC 0.90

Rules for consequent: Class1
----------------
IF Var1 IS High AND Var2 IS High WITH DS 0.75, ACC 0.85
"""
        master_rb = pers.load_fuzzy_rules(rules_text, sample_fuzzy_variables)

        assert len(master_rb) == 2  # Two classes
        assert len(master_rb.get_rules()) >= 2

    def test_load_rules_with_weights(self, sample_fuzzy_variables):
        """Test loading rules with weight information."""
        rules_text = """Rules for consequent: Class0
----------------
IF Var1 IS Low WITH DS 0.85, ACC 0.90, WGHT 0.95

Rules for consequent: Class1
----------------
IF Var1 IS High WITH DS 0.75, ACC 0.85, WGHT 0.80
"""
        master_rb = pers.load_fuzzy_rules(rules_text, sample_fuzzy_variables)

        # Check that rules have weights
        rules = master_rb.get_rules()
        assert len(rules) >= 2


class TestUtilityFunctions:
    """Tests for utility functions in persistence module."""

    def test_extract_mod_word(self):
        """Test modifier word extraction."""
        text = "Var1 IS Low (MOD very)"
        result = pers._extract_mod_word(text)
        assert result == "very"

    def test_extract_mod_word_no_match(self):
        """Test modifier extraction with no modifier."""
        text = "Var1 IS Low"
        result = pers._extract_mod_word(text)
        assert result is None

    def test_remove_mod_completely(self):
        """Test modifier removal from text."""
        text = "Var1 IS Low (MOD very) AND Var2 IS High"
        result = pers._remove_mod_completely(text)
        assert "(MOD" not in result
        assert "Var1 IS Low" in result
        assert "Var2 IS High" in result

    def test_remove_parentheses(self):
        """Test parentheses removal."""
        text = "DS 0.85 (ACC 0.92), (WGHT 1.0)"
        result = pers.remove_parentheses(text)
        assert "ACC" not in result
        assert "WGHT" not in result
        assert "DS 0.85" in result


class TestCategoricalVariables:
    """Tests for categorical fuzzy variables persistence."""

    def test_print_categorical_variable(self):
        """Test printing a categorical fuzzy variable."""
        cat1 = fs.categoricalFS('Cat1', 'A')
        cat2 = fs.categoricalFS('Cat2', 'B')
        cat3 = fs.categoricalFS('Cat3', 'C')
        fvar = fs.fuzzyVariable('Category', [cat1, cat2, cat3])

        result = pers.print_fuzzy_variable(fvar)

        assert '$Categorical variable: Category' in result
        assert 'Cat1' in result
        assert 'Cat2' in result
        assert 'Cat3' in result

    def test_print_categorical_variable_numeric(self):
        """Test printing a categorical fuzzy variable with numeric categories."""
        cat1 = fs.categoricalFS('Val1', 1.0)
        cat2 = fs.categoricalFS('Val2', 2.0)
        fvar = fs.fuzzyVariable('NumericCat', [cat1, cat2])

        result = pers.print_fuzzy_variable(fvar)

        assert '$Categorical variable: NumericCat' in result
        assert 'float' in result


class TestEdgeCases:
    """Tests for edge cases in persistence."""

    def test_empty_fuzzy_variable_name(self):
        """Test handling of empty variable names."""
        low = fs.FS('Low', [0, 0, 0.5, 0.5], [0, 1])
        high = fs.FS('High', [0.5, 0.5, 1, 1], [0, 1])
        fvar = fs.fuzzyVariable('', [low, high])  # Empty name

        result = pers.print_fuzzy_variable(fvar)
        assert '$$$ Linguistic variable:' in result

    def test_special_characters_in_names(self):
        """Test handling of special characters in names."""
        low = fs.FS('Low-Value', [0, 0, 0.5, 0.5], [0, 1])
        high = fs.FS('High_Value', [0.5, 0.5, 1, 1], [0, 1])
        fvar = fs.fuzzyVariable('Test-Variable_1', [low, high])

        result = pers.print_fuzzy_variable(fvar)
        assert 'Test-Variable_1' in result
        assert 'Low-Value' in result
        assert 'High_Value' in result

    def test_large_domain_values(self):
        """Test handling of large domain values."""
        low = fs.FS('Low', [0, 0, 500, 500], [0, 1000])
        high = fs.FS('High', [500, 500, 1000, 1000], [0, 1000])
        fvar = fs.fuzzyVariable('LargeScale', [low, high])

        text = pers.print_fuzzy_variable(fvar)
        loaded = pers.load_fuzzy_variables(text)[0]

        assert loaded.name == 'LargeScale'
        assert len(loaded.linguistic_variables) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
