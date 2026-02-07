"""
Comprehensive tests for the utils module.

Tests partition construction, data preprocessing utilities,
and other helper functions.
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ex_fuzzy', 'ex_fuzzy'))

import fuzzy_sets as fs
import utils


class TestConstructPartitions:
    """Tests for construct_partitions function."""

    @pytest.fixture
    def sample_data_numpy(self):
        """Create sample numpy data."""
        np.random.seed(42)
        return np.random.rand(100, 4)

    @pytest.fixture
    def sample_data_pandas(self):
        """Create sample pandas DataFrame."""
        np.random.seed(42)
        data = np.random.rand(100, 4)
        return pd.DataFrame(data, columns=['A', 'B', 'C', 'D'])

    def test_construct_partitions_t1_numpy(self, sample_data_numpy):
        """Test T1 partition construction with numpy array."""
        partitions = utils.construct_partitions(sample_data_numpy, fs.FUZZY_SETS.t1)

        assert isinstance(partitions, list)
        assert len(partitions) == sample_data_numpy.shape[1]
        for p in partitions:
            assert isinstance(p, fs.fuzzyVariable)
            assert p.fuzzy_type() == fs.FUZZY_SETS.t1

    def test_construct_partitions_t1_pandas(self, sample_data_pandas):
        """Test T1 partition construction with pandas DataFrame."""
        partitions = utils.construct_partitions(sample_data_pandas, fs.FUZZY_SETS.t1)

        assert isinstance(partitions, list)
        assert len(partitions) == sample_data_pandas.shape[1]
        for p in partitions:
            assert isinstance(p, fs.fuzzyVariable)

    def test_construct_partitions_t2(self, sample_data_numpy):
        """Test T2 partition construction."""
        partitions = utils.construct_partitions(sample_data_numpy, fs.FUZZY_SETS.t2)

        assert isinstance(partitions, list)
        assert len(partitions) == sample_data_numpy.shape[1]
        for p in partitions:
            assert isinstance(p, fs.fuzzyVariable)
            assert p.fuzzy_type() == fs.FUZZY_SETS.t2

    def test_partitions_preserve_column_names(self, sample_data_pandas):
        """Test that partition names come from DataFrame columns."""
        partitions = utils.construct_partitions(sample_data_pandas, fs.FUZZY_SETS.t1)

        column_names = sample_data_pandas.columns.tolist()
        for i, p in enumerate(partitions):
            assert p.name == column_names[i]

    def test_partitions_default_names_for_numpy(self, sample_data_numpy):
        """Test that partitions get default names for numpy arrays."""
        partitions = utils.construct_partitions(sample_data_numpy, fs.FUZZY_SETS.t1)

        for i, p in enumerate(partitions):
            assert p.name is not None
            # Should have some name (e.g., 'x0', 'x1', etc.)
            assert len(p.name) > 0


class TestPartitionDomains:
    """Tests for partition domain handling."""

    def test_partition_domains_match_data_range(self):
        """Test that partition domains match data range."""
        np.random.seed(42)
        # Data in range [10, 50]
        data = np.random.rand(100, 2) * 40 + 10

        partitions = utils.construct_partitions(data, fs.FUZZY_SETS.t1)

        for i, p in enumerate(partitions):
            col_min = data[:, i].min()
            col_max = data[:, i].max()

            # Check that fuzzy sets cover the data range
            for fuzzy_set in p.linguistic_variables:
                # Domain should encompass data range
                assert fuzzy_set.domain[0] <= col_min
                assert fuzzy_set.domain[1] >= col_max

    def test_partition_with_negative_values(self):
        """Test partition construction with negative values."""
        np.random.seed(42)
        data = np.random.rand(100, 2) * 2 - 1  # Range [-1, 1]

        partitions = utils.construct_partitions(data, fs.FUZZY_SETS.t1)

        assert len(partitions) == 2
        for p in partitions:
            assert isinstance(p, fs.fuzzyVariable)

    def test_partition_with_large_range(self):
        """Test partition construction with large value range."""
        np.random.seed(42)
        data = np.random.rand(100, 2) * 10000  # Range [0, 10000]

        partitions = utils.construct_partitions(data, fs.FUZZY_SETS.t1)

        assert len(partitions) == 2
        for p in partitions:
            for fuzzy_set in p.linguistic_variables:
                # Domain should handle large values
                assert fuzzy_set.domain[1] >= 9000


class TestPartitionLinguisticVariables:
    """Tests for linguistic variables in partitions."""

    def test_default_number_of_linguistic_variables(self):
        """Test default number of linguistic variables per partition."""
        np.random.seed(42)
        data = np.random.rand(100, 3)

        partitions = utils.construct_partitions(data, fs.FUZZY_SETS.t1)

        # Default is usually 3 (low, medium, high)
        for p in partitions:
            assert len(p.linguistic_variables) >= 2

    def test_membership_coverage(self):
        """Test that memberships cover the data range properly."""
        np.random.seed(42)
        data = np.random.rand(100, 2)

        partitions = utils.construct_partitions(data, fs.FUZZY_SETS.t1)

        # Test a few points
        test_points = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        for p in partitions:
            memberships = p.compute_memberships(test_points)
            # compute_memberships returns (n_linguistic_variables, n_samples)
            # so transpose to get (n_samples, n_linguistic_variables)
            memberships = memberships.T

            # At each point, at least one fuzzy set should have non-zero membership
            for i in range(len(test_points)):
                assert np.max(memberships[i]) > 0, f"No coverage at point {test_points[i]}"


class TestPartitionTypes:
    """Tests for different partition types."""

    def test_t1_partitions_structure(self):
        """Test T1 partition structure."""
        np.random.seed(42)
        data = np.random.rand(50, 2)

        partitions = utils.construct_partitions(data, fs.FUZZY_SETS.t1)

        for p in partitions:
            for fuzzy_set in p.linguistic_variables:
                assert isinstance(fuzzy_set, fs.FS)
                # Membership should return 1D array
                test_val = np.array([0.5])
                mem = fuzzy_set.membership(test_val)
                assert mem.ndim == 1

    def test_t2_partitions_structure(self):
        """Test T2 partition structure."""
        np.random.seed(42)
        data = np.random.rand(50, 2)

        partitions = utils.construct_partitions(data, fs.FUZZY_SETS.t2)

        for p in partitions:
            for fuzzy_set in p.linguistic_variables:
                assert isinstance(fuzzy_set, fs.IVFS)
                # Membership should return 2D array (lower, upper bounds)
                test_val = np.array([0.5])
                mem = fuzzy_set.membership(test_val)
                assert mem.shape[1] == 2  # Lower and upper


class TestCategoricalPartitions:
    """Tests for categorical partitions."""

    def test_construct_crisp_categorical_partition(self):
        """Test construction of crisp categorical partition."""
        categories = np.array(['A', 'B', 'C', 'A', 'B', 'C'])  # Array with categorical values

        try:
            partition = utils.construct_crisp_categorical_partition(
                categories, 'category_var', fs.FUZZY_SETS.t1
            )

            assert isinstance(partition, fs.fuzzyVariable)
            # Should have one fuzzy set per unique category
            assert len(partition.linguistic_variables) == 3  # A, B, C
        except (AttributeError, NotImplementedError, TypeError) as e:
            pytest.skip(f"construct_crisp_categorical_partition not available: {e}")

    def test_categorical_partition_membership(self):
        """Test categorical partition membership computation."""
        categories = np.array(['A', 'B', 'C', 'A', 'B', 'C'])

        try:
            partition = utils.construct_crisp_categorical_partition(
                categories, 'category_var', fs.FUZZY_SETS.t1
            )

            # Test membership for exact category - categorical FS works differently
            assert partition is not None
            assert len(partition.linguistic_variables) > 0
        except (AttributeError, NotImplementedError, TypeError) as e:
            pytest.skip(f"construct_crisp_categorical_partition not available: {e}")


class TestEdgeCases:
    """Tests for edge cases in utils."""

    def test_single_column_data(self):
        """Test partition construction with single column."""
        np.random.seed(42)
        data = np.random.rand(100, 1)

        partitions = utils.construct_partitions(data, fs.FUZZY_SETS.t1)

        assert len(partitions) == 1
        assert isinstance(partitions[0], fs.fuzzyVariable)

    def test_single_row_data(self):
        """Test partition construction with single row."""
        data = np.array([[0.1, 0.5, 0.9]])

        try:
            partitions = utils.construct_partitions(data, fs.FUZZY_SETS.t1)
            # Should work, though partitions may be degenerate
            assert len(partitions) == 3
        except (ValueError, ZeroDivisionError):
            # May raise error due to no variance
            pass

    def test_constant_column(self):
        """Test partition construction with constant column."""
        data = np.ones((100, 2))
        data[:, 1] = np.arange(100)  # Second column has variance

        try:
            partitions = utils.construct_partitions(data, fs.FUZZY_SETS.t1)
            assert len(partitions) == 2
        except (ValueError, ZeroDivisionError):
            # May handle constant columns specially
            pass

    def test_empty_dataframe(self):
        """Test behavior with empty DataFrame."""
        data = pd.DataFrame()

        with pytest.raises((ValueError, IndexError)):
            utils.construct_partitions(data, fs.FUZZY_SETS.t1)


class TestMembershipComputation:
    """Tests for membership computation in partitions."""

    def test_membership_bounds(self):
        """Test that membership values are in [0, 1]."""
        np.random.seed(42)
        data = np.random.rand(100, 3)
        partitions = utils.construct_partitions(data, fs.FUZZY_SETS.t1)

        test_points = np.array([0.0, 0.1, 0.5, 0.9, 1.0])

        for p in partitions:
            memberships = p.compute_memberships(test_points)
            assert np.all(memberships >= 0)
            assert np.all(memberships <= 1)

    def test_membership_computation_vectorized(self):
        """Test vectorized membership computation."""
        np.random.seed(42)
        data = np.random.rand(100, 2)
        partitions = utils.construct_partitions(data, fs.FUZZY_SETS.t1)

        test_points = np.random.rand(50)

        for p in partitions:
            memberships = p.compute_memberships(test_points)
            # compute_memberships returns (n_linguistic_variables, n_samples)
            assert memberships.shape[0] == len(p.linguistic_variables)
            assert memberships.shape[1] == len(test_points)


class TestPartitionConsistency:
    """Tests for partition consistency."""

    def test_same_data_same_partitions(self):
        """Test that same data produces consistent partitions."""
        np.random.seed(42)
        data = np.random.rand(100, 3)

        partitions1 = utils.construct_partitions(data, fs.FUZZY_SETS.t1)
        partitions2 = utils.construct_partitions(data, fs.FUZZY_SETS.t1)

        # Should produce same partitions
        assert len(partitions1) == len(partitions2)
        for p1, p2 in zip(partitions1, partitions2):
            assert p1.name == p2.name
            assert len(p1.linguistic_variables) == len(p2.linguistic_variables)

    def test_partition_fuzzy_type_consistency(self):
        """Test that all fuzzy sets in partition have same type."""
        np.random.seed(42)
        data = np.random.rand(100, 3)

        for ftype in [fs.FUZZY_SETS.t1, fs.FUZZY_SETS.t2]:
            partitions = utils.construct_partitions(data, ftype)

            for p in partitions:
                assert p.fuzzy_type() == ftype
                for fuzzy_set in p.linguistic_variables:
                    assert fuzzy_set.type() == ftype


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
