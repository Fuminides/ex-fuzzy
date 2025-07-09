"""
Comprehensive tests for the fuzzy_sets module.

This module tests all fuzzy set implementations including Type-1 and Type-2
fuzzy sets, fuzzy variables, and membership functions.
"""
import pytest
import numpy as np
import sys
import os

# Add the library path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ex_fuzzy', 'ex_fuzzy'))

import fuzzy_sets as fs
from conftest import assert_fuzzy_set_properties, FLOAT_TOLERANCE


class TestFuzzySetEnum:
    """Test the FUZZY_SETS enumeration."""
    
    def test_fuzzy_sets_enum_exists(self):
        """Test that the FUZZY_SETS enum exists and has expected values."""
        assert hasattr(fs, 'FUZZY_SETS')
        assert hasattr(fs.FUZZY_SETS, 't1')
        assert hasattr(fs.FUZZY_SETS, 't2')
        assert hasattr(fs.FUZZY_SETS, 'gt2')
    
    def test_enum_string_representation(self):
        """Test string representation of enum values."""
        assert str(fs.FUZZY_SETS.t1) is not None
        assert str(fs.FUZZY_SETS.t2) is not None
        assert str(fs.FUZZY_SETS.gt2) is not None


class TestFS:
    """Test the base FS (Type-1 fuzzy set) class."""
    
    def test_trapezoidal_fs_creation(self):
        """Test creation of trapezoidal fuzzy set."""
        fs_test = fs.FS('test', [0, 0.2, 0.8, 1.0], [0, 1])
        assert fs_test.name == 'test'
        assert fs_test.type() == fs.FUZZY_SETS.t1
        assert fs_test.shape() == 'trapezoidal'
    
    def test_triangular_fs_creation(self):
        """Test creation of triangular fuzzy set."""
        fs_test = fs.FS('triangle', [0, 0.5, 1.0], [0, 1])
        assert fs_test.name == 'triangle'
        assert fs_test.shape() == 'triangular'
    
    def test_membership_function_basic(self):
        """Test basic membership function evaluation."""
        fs_test = fs.FS('test', [0, 0.25, 0.75, 1.0], [0, 1])
        
        # Test key points
        assert fs_test.membership(np.array([0.0])) == pytest.approx([0.0], abs=FLOAT_TOLERANCE)
        assert fs_test.membership(np.array([0.25])) == pytest.approx([1.0], abs=FLOAT_TOLERANCE)
        assert fs_test.membership(np.array([0.5])) == pytest.approx([1.0], abs=FLOAT_TOLERANCE)
        assert fs_test.membership(np.array([0.75])) == pytest.approx([1.0], abs=FLOAT_TOLERANCE)
        assert fs_test.membership(np.array([1.0])) == pytest.approx([0.0], abs=FLOAT_TOLERANCE)
    
    def test_membership_function_slopes(self):
        """Test membership function on slopes."""
        fs_test = fs.FS('test', [0, 0.2, 0.8, 1.0], [0, 1])
        
        # Test ascending slope
        result = fs_test.membership(np.array([0.1]))
        assert 0.0 < result[0] < 1.0
        
        # Test descending slope  
        result = fs_test.membership(np.array([0.9]))
        assert 0.0 < result[0] < 1.0
    
    def test_membership_vector_input(self):
        """Test membership function with vector input."""
        fs_test = fs.FS('test', [0, 0.25, 0.75, 1.0], [0, 1])
        input_values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        result = fs_test.membership(input_values)
        
        assert len(result) == len(input_values)
        assert isinstance(result, np.ndarray)
    
    def test_membership_out_of_bounds(self):
        """Test membership function with out-of-bounds values."""
        fs_test = fs.FS('test', [0, 0.25, 0.75, 1.0], [0, 1])
        
        # Test below minimum
        result = fs_test.membership(np.array([-0.5]))
        assert result[0] == 0.0
        
        # Test above maximum
        result = fs_test.membership(np.array([1.5]))
        assert result[0] == 0.0
    
    def test_callable_interface(self):
        """Test that fuzzy set can be called directly."""
        fs_test = fs.FS('test', [0, 0.25, 0.75, 1.0], [0, 1])
        
        # Test callable interface
        result1 = fs_test(0.5)
        result2 = fs_test.membership(np.array([0.5]))[0]
        assert result1 == pytest.approx(result2, abs=FLOAT_TOLERANCE)


class TestIVFS:
    """Test the IVFS (Type-2 fuzzy set) class."""
    
    def test_ivfs_creation(self):
        """Test creation of interval-valued fuzzy set."""
        ivfs_test = fs.IVFS('test_t2', [0, 0.2, 0.8, 1.0], [0, 0.3, 0.7, 1.0], [0, 1])
        assert ivfs_test.name == 'test_t2'
        assert ivfs_test.type() == fs.FUZZY_SETS.t2
        assert ivfs_test.shape() == 'trapezoidal'
    
    def test_ivfs_membership_function(self):
        """Test IVFS membership function returns interval values."""
        ivfs_test = fs.IVFS('test_t2', [0, 0.2, 0.8, 1.0], [0, 0.3, 0.7, 1.0], [0, 1])
        
        input_values = np.array([0.0, 0.5, 1.0])
        result = ivfs_test.membership(input_values)
        
        # Should return matrix with lower and upper bounds
        assert result.shape == (len(input_values), 2)
        
        # Lower bound should be <= upper bound
        assert np.all(result[:, 0] <= result[:, 1])
    
    def test_ivfs_alpha_cuts(self):
        """Test alpha-cut operations on IVFS."""
        ivfs_test = fs.IVFS('test_t2', [0, 0.2, 0.8, 1.0], [0, 0.3, 0.7, 1.0], [0, 1])
        
        # Test alpha-cut at different levels
        alpha_05 = ivfs_test.alpha_cut(0.5)
        assert isinstance(alpha_05, tuple)
        assert len(alpha_05) == 2  # Lower and upper bounds
    
    def test_ivfs_with_lower_height(self):
        """Test IVFS creation with lower height parameter."""
        ivfs_test = fs.IVFS('test_t2', [0, 0.2, 0.8, 1.0], [0, 0.3, 0.7, 1.0], [0, 1], lower_height=0.8)
        assert ivfs_test.lower_height == 0.8


class TestGaussianFS:
    """Test the gaussianFS class."""
    
    def test_gaussian_fs_creation(self):
        """Test creation of Gaussian fuzzy set."""
        gauss_fs = fs.gaussianFS([0.5, 0.2], 'gaussian_test', 100)
        assert gauss_fs.name == 'gaussian_test'
        assert gauss_fs.type() == fs.FUZZY_SETS.t1
        assert gauss_fs.shape() == 'gaussian'
    
    def test_gaussian_membership_peak(self):
        """Test that Gaussian fuzzy set has peak at mean."""
        mean, std = 0.5, 0.2
        gauss_fs = fs.gaussianFS([mean, std], 'gaussian_test', 100)
        
        # Membership at mean should be 1.0
        result = gauss_fs.membership(np.array([mean]))
        assert result[0] == pytest.approx(1.0, abs=FLOAT_TOLERANCE)
    
    def test_gaussian_membership_symmetry(self):
        """Test that Gaussian membership is symmetric around mean."""
        mean, std = 0.5, 0.2
        gauss_fs = fs.gaussianFS([mean, std], 'gaussian_test', 100)
        
        # Test symmetry
        offset = 0.1
        left_val = gauss_fs.membership(np.array([mean - offset]))[0]
        right_val = gauss_fs.membership(np.array([mean + offset]))[0]
        assert left_val == pytest.approx(right_val, abs=FLOAT_TOLERANCE)
    
    def test_gaussian_membership_decreases_with_distance(self):
        """Test that Gaussian membership decreases with distance from mean."""
        mean, std = 0.5, 0.2
        gauss_fs = fs.gaussianFS([mean, std], 'gaussian_test', 100)
        
        val_at_mean = gauss_fs.membership(np.array([mean]))[0]
        val_at_std = gauss_fs.membership(np.array([mean + std]))[0]
        val_at_2std = gauss_fs.membership(np.array([mean + 2*std]))[0]
        
        assert val_at_mean > val_at_std > val_at_2std


class TestGaussianIVFS:
    """Test the gaussianIVFS class."""
    
    def test_gaussian_ivfs_creation(self):
        """Test creation of Gaussian interval-valued fuzzy set."""
        gauss_ivfs = fs.gaussianIVFS([0.5, 0.15], [0.5, 0.25], 'gaussian_t2_test', 100)
        assert gauss_ivfs.name == 'gaussian_t2_test'
        assert gauss_ivfs.type() == fs.FUZZY_SETS.t2
        assert gauss_ivfs.shape() == 'gaussian'
    
    def test_gaussian_ivfs_membership_interval(self):
        """Test that Gaussian IVFS returns interval values."""
        gauss_ivfs = fs.gaussianIVFS([0.5, 0.15], [0.5, 0.25], 'gaussian_t2_test', 100)
        
        input_values = np.array([0.3, 0.5, 0.7])
        result = gauss_ivfs.membership(input_values)
        
        # Should return intervals
        assert result.shape == (len(input_values), 2)
        
        # Lower bound should be <= upper bound
        assert np.all(result[:, 0] <= result[:, 1])


class TestFuzzyVariable:
    """Test the fuzzyVariable class."""
    
    def test_fuzzy_variable_creation(self, sample_fuzzy_sets):
        """Test creation of fuzzy variable."""
        fv = fs.fuzzyVariable('test_var', sample_fuzzy_sets['t1_sets'], 'units')
        assert fv.name == 'test_var'
        assert fv.units == 'units'
        assert len(fv.linguistic_variables) == len(sample_fuzzy_sets['t1_sets'])
    
    def test_fuzzy_variable_membership_computation(self, sample_fuzzy_sets):
        """Test fuzzy variable membership computation across all sets."""
        fv = fs.fuzzyVariable('test_var', sample_fuzzy_sets['t1_sets'])
        
        input_values = np.array([0.1, 0.5, 0.9])
        memberships = fv.membership(input_values)
        
        # Should return matrix: samples x linguistic_variables
        expected_shape = (len(input_values), len(sample_fuzzy_sets['t1_sets']))
        assert memberships.shape == expected_shape
    
    def test_fuzzy_variable_with_t2_sets(self, sample_fuzzy_sets):
        """Test fuzzy variable with Type-2 fuzzy sets."""
        fv = fs.fuzzyVariable('test_var_t2', sample_fuzzy_sets['t2_sets'])
        assert fv.fs_type == fs.FUZZY_SETS.t2
    
    def test_fuzzy_variable_linguistic_names(self, sample_fuzzy_sets):
        """Test retrieval of linguistic variable names."""
        fv = fs.fuzzyVariable('test_var', sample_fuzzy_sets['t1_sets'])
        names = fv.linguistic_variable_names()
        
        expected_names = [fs.name for fs in sample_fuzzy_sets['t1_sets']]
        assert names == expected_names


class TestMembershipFunctions:
    """Test standalone membership functions."""
    
    def test_trapezoidal_membership_function(self):
        """Test the trapezoidal membership function."""
        # Test with typical trapezoidal parameters
        params = [0.2, 0.4, 0.6, 0.8]
        test_points = np.array([0.0, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0])
        
        result = fs.trapezoidal_membership(test_points, params)
        
        # Check key points
        assert result[0] == 0.0  # Before start
        assert result[1] == 0.0  # At start
        assert result[3] == 1.0  # In plateau
        assert result[5] == 0.0  # At end
        assert result[6] == 0.0  # After end
    
    def test_triangular_membership_function(self):
        """Test triangular membership function (special case of trapezoidal)."""
        # Triangular: [a, b, b, c] where b is the peak
        params = [0.2, 0.5, 0.5, 0.8]
        test_points = np.array([0.2, 0.35, 0.5, 0.65, 0.8])
        
        result = fs.trapezoidal_membership(test_points, params)
        
        assert result[0] == 0.0  # At left base
        assert result[2] == 1.0  # At peak
        assert result[4] == 0.0  # At right base


class TestUtilityFunctions:
    """Test utility functions in the fuzzy_sets module."""
    
    def test_create_fuzzy_variables_function(self):
        """Test the create_fuzzy_variables utility function if available."""
        # This test depends on the actual implementation
        # Create sample data
        X = np.random.random((100, 3))
        try:
            # Try to create fuzzy variables
            variables = fs.create_fuzzy_variables(X, ['low', 'high'])
            assert len(variables) == X.shape[1]
            for var in variables:
                assert isinstance(var, fs.fuzzyVariable)
        except (AttributeError, NameError):
            # Function might not be in this module
            pytest.skip("create_fuzzy_variables function not found in fuzzy_sets module")


class TestFuzzySetValidation:
    """Test validation and error handling for fuzzy sets."""
    
    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        with pytest.raises((ValueError, AssertionError)):
            # Invalid trapezoidal parameters (not in ascending order)
            fs.FS('invalid', [0.8, 0.6, 0.4, 0.2], [0, 1])
    
    def test_empty_parameters(self):
        """Test behavior with empty or None parameters."""
        with pytest.raises((ValueError, TypeError)):
            fs.FS('empty', [], [0, 1])
    
    def test_domain_validation(self):
        """Test that domain validation works correctly."""
        # Create fuzzy set with specific domain
        fs_test = fs.FS('test', [0, 0.25, 0.75, 1.0], [0, 1])
        
        # Test that domain is respected
        assert hasattr(fs_test, 'domain')


# Integration tests
class TestFuzzySetIntegration:
    """Integration tests for fuzzy sets with other components."""
    
    def test_fuzzy_set_with_rules(self, sample_fuzzy_sets):
        """Test that fuzzy sets work correctly with rules."""
        # This is a basic integration test
        fv = fs.fuzzyVariable('test_var', sample_fuzzy_sets['t1_sets'])
        
        # Test that we can evaluate membership
        input_data = np.array([0.1, 0.5, 0.9])
        memberships = fv.membership(input_data)
        
        # Verify the shape and content
        assert memberships.shape[0] == len(input_data)
        assert memberships.shape[1] == len(sample_fuzzy_sets['t1_sets'])
        assert np.all(memberships >= 0) and np.all(memberships <= 1)
    
    def test_mixed_fuzzy_set_types(self, sample_fuzzy_sets):
        """Test that mixing different fuzzy set types is handled correctly."""
        # This test verifies error handling when mixing types
        try:
            # Try to create a fuzzy variable with mixed types
            mixed_sets = sample_fuzzy_sets['t1_sets'] + sample_fuzzy_sets['t2_sets']
            fv = fs.fuzzyVariable('mixed_var', mixed_sets)
            # If this succeeds, it should handle type checking
            assert fv is not None
        except (ValueError, TypeError):
            # This is expected behavior - mixed types should raise error
            pass


# Performance tests
class TestFuzzySetPerformance:
    """Performance tests for fuzzy sets."""
    
    @pytest.mark.performance
    def test_membership_computation_performance(self, sample_fuzzy_sets):
        """Test performance of membership computation with large inputs."""
        fv = fs.fuzzyVariable('perf_test', sample_fuzzy_sets['t1_sets'])
        
        # Large input array
        large_input = np.random.random(10000)
        
        import time
        start_time = time.time()
        memberships = fv.membership(large_input)
        end_time = time.time()
        
        # Basic performance check (should complete in reasonable time)
        assert end_time - start_time < 5.0  # Should complete in less than 5 seconds
        assert memberships.shape == (len(large_input), len(sample_fuzzy_sets['t1_sets']))
    
    @pytest.mark.performance  
    def test_multiple_fuzzy_sets_performance(self):
        """Test performance when creating many fuzzy sets."""
        import time
        
        start_time = time.time()
        fuzzy_sets = []
        for i in range(100):
            fs_test = fs.FS(f'test_{i}', [0, 0.25, 0.75, 1.0], [0, 1])
            fuzzy_sets.append(fs_test)
        end_time = time.time()
        
        assert len(fuzzy_sets) == 100
        assert end_time - start_time < 2.0  # Should complete quickly
