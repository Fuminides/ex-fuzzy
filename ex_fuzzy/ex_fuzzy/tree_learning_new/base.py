"""
Base classes and shared utilities for FuzzyCART tree modules.

This module provides common functionality, interfaces, and utilities
that are shared across all tree modules to ensure consistency and
reduce code duplication.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple


class TreeComponent(ABC):
    """
    Abstract base class for all FuzzyCART tree components.
    
    Provides common interface and shared functionality for all modules
    that operate on the fuzzy decision tree.
    """
    
    def __init__(self, parent_classifier):
        """
        Initialize tree component with reference to parent classifier.
        
        Parameters
        ----------
        parent_classifier : FuzzyCART
            Reference to the main FuzzyCART classifier instance.
        """
        self.classifier = parent_classifier
    
    @property
    def fuzzy_partitions(self):
        """Access fuzzy partitions from parent classifier."""
        return self.classifier.fuzzy_partitions
    
    @property
    def classes_(self):
        """Access class labels from parent classifier."""
        return self.classifier.classes_
    
    @property
    def tree_rules(self):
        """Access tree rules count from parent classifier."""
        return self.classifier.tree_rules
    
    @property
    def max_rules(self):
        """Access max rules limit from parent classifier."""
        return self.classifier.max_rules
    
    @property
    def max_depth(self):
        """Access max depth limit from parent classifier."""
        return self.classifier.max_depth
    
    @property
    def coverage_threshold(self):
        """Access coverage threshold from parent classifier."""
        return self.classifier.coverage_threshold
    
    @property
    def root(self):
        """Access root node from parent classifier."""
        return self.classifier._root
    
    @property
    def node_dict_access(self):
        """Access node dictionary from parent classifier."""
        return self.classifier.node_dict_access


class CacheManager:
    """
    Manages caching mechanisms for tree components.
    
    Provides shared caching functionality to avoid redundant computations
    across different tree operations.
    """
    
    def __init__(self):
        """Initialize cache containers used across tree operations."""
        self._membership_cache = {}
        self._last_X_shape = None
        self._cached_leaves = None
        self._cached_all_nodes = None
    
    def get_cached_memberships(self, X: np.array, fuzzy_partitions) -> dict:
        """
        Get cached membership values or compute them if not cached.
        
        OPTIMIZATION: Cache membership computations to avoid redundant calculations
        across multiple split evaluations.
        
        Parameters
        ----------
        X : np.array
            Input data array.
        fuzzy_partitions : list
            List of fuzzy variables for each feature.
            
        Returns
        -------
        dict
            Cached membership values for all fuzzy sets.
        """
        # Check if we need to recompute cache
        if (self._last_X_shape != X.shape or 
            len(self._membership_cache) == 0):
            
            self._membership_cache = {}
            self._last_X_shape = X.shape
            
            # Pre-compute all memberships
            for feature_idx, fuzzy_var in enumerate(fuzzy_partitions):
                feature_memberships = np.zeros((len(fuzzy_var), X.shape[0]))
                for fz_idx, fuzzy_set in enumerate(fuzzy_var):
                    feature_memberships[fz_idx] = fuzzy_set.membership(X[:, feature_idx])
                self._membership_cache[feature_idx] = feature_memberships
        
        return self._membership_cache
    
    def clear_all_caches(self):
        """Clear all cached values."""
        self._membership_cache = {}
        self._last_X_shape = None
        self._cached_leaves = None
        self._cached_all_nodes = None
    
    def clear_split_caches(self):
        """Clear cached split evaluations from all nodes."""
        self._membership_cache = {}
    
    def invalidate_leaf_cache(self):
        """Invalidate cached leaf and node collections."""
        self._cached_leaves = None
        self._cached_all_nodes = None


class NodeValidator:
    """
    Utilities for validating and working with tree nodes.
    
    Provides common node validation and manipulation functions
    used across different tree modules.
    """
    
    @staticmethod
    def validate_node_structure(node: Dict[str, Any]) -> bool:
        """
        Validate that a node has required structure.
        
        Parameters
        ----------
        node : dict
            Node dictionary to validate.
            
        Returns
        -------
        bool
            True if node structure is valid.
        """
        required_fields = ['name', 'prediction', 'existing_membership', 'coverage']
        return all(field in node for field in required_fields)
    
    @staticmethod
    def is_leaf_node(node: Dict[str, Any]) -> bool:
        """
        Check if a node is a leaf (has no children).
        
        Parameters
        ----------
        node : dict
            Node to check.
            
        Returns
        -------
        bool
            True if node is a leaf.
        """
        return 'children' not in node or len(node.get('children', {})) == 0
    
    @staticmethod
    def get_node_depth(node: Dict[str, Any]) -> int:
        """
        Get the depth of a node in the tree.
        
        Parameters
        ----------
        node : dict
            Node to check.
            
        Returns
        -------
        int
            Depth of the node (root = 0).
        """
        return node.get('depth', 0)
    
    @staticmethod
    def create_node_name(parent_name: str, feature: int, fuzzy_set: int) -> str:
        """
        Create standardized node name.
        
        Parameters
        ----------
        parent_name : str
            Name of parent node.
        feature : int
            Feature index.
        fuzzy_set : int
            Fuzzy set index.
            
        Returns
        -------
        str
            Standardized node name.
        """
        return f"{parent_name}_F{feature}_L{fuzzy_set}"


class TreeMetrics:
    """
    Common tree evaluation and metrics utilities.
    
    Provides shared functionality for calculating tree-related
    metrics and statistics.
    """
    
    @staticmethod
    def calculate_node_coverage(membership: np.array, total_samples: int) -> float:
        """
        Calculate coverage ratio for a node.
        
        Parameters
        ----------
        membership : np.array
            Membership values for samples.
        total_samples : int
            Total number of samples.
            
        Returns
        -------
        float
            Coverage ratio (0.0 to 1.0).
        """
        return np.sum(membership) / total_samples if total_samples > 0 else 0.0
    
    @staticmethod
    def count_tree_nodes(root: Dict[str, Any]) -> int:
        """
        Count total number of nodes in tree.
        
        Parameters
        ----------
        root : dict
            Root node of the tree.
            
        Returns
        -------
        int
            Total number of nodes.
        """
        def count_recursive(node):
            """Recursively count nodes in a subtree."""
            count = 1  # Count this node
            if 'children' in node:
                for child in node['children'].values():
                    count += count_recursive(child)
            return count
        
        return count_recursive(root)
    
    @staticmethod
    def get_tree_depth(root: Dict[str, Any]) -> int:
        """
        Get maximum depth of the tree.
        
        Parameters
        ----------
        root : dict
            Root node of the tree.
            
        Returns
        -------
        int
            Maximum depth of the tree.
        """
        def depth_recursive(node, current_depth=0):
            """Recursively compute maximum depth from the current node."""
            max_depth = current_depth
            if 'children' in node:
                for child in node['children'].values():
                    child_depth = depth_recursive(child, current_depth + 1)
                    max_depth = max(max_depth, child_depth)
            return max_depth
        
        return depth_recursive(root)


# Shared constants and defaults
DEFAULT_EPSILON = 1e-6
DEFAULT_MIN_IMPROVEMENT = 0.01
DEFAULT_COVERAGE_THRESHOLD = 0.00
DEFAULT_CCP_ALPHA = 0.0
