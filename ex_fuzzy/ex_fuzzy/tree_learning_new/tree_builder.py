"""
Tree construction module for FuzzyCART algorithm.

This module contains all the logic for building fuzzy decision trees,
including node splitting, split evaluation, and tree growth algorithms.
"""

from platform import node
import numpy as np
from typing import Tuple, Dict, Any
from .base import TreeComponent, NodeValidator, TreeMetrics
from .metrics.fuzzy_metrics import (
    majority_class, class_probabilities, compute_fuzzy_purity as compute_fuzzy_gini_impurity, 
    compute_fuzzy_cci, _complete_classification_index, evaluate_fuzzy_feature_partition
)


class TreeBuilder(TreeComponent):
    """
    Handles tree construction and node splitting for FuzzyCART.
    
    This class implements the core fuzzy CART algorithm that builds the
    decision tree by iteratively finding and executing the best splits.
    """
    
    def __init__(self, parent_classifier):
        """Initialize tree builder with reference to parent classifier."""
        super().__init__(parent_classifier)
        self._coverage_cache = {}
        self._gini_cache = {}
        self._prediction_cache = None
    
    def _clear_node_caches_recursive(self, node: Dict[str, Any]):
        """
        Recursively clear all node-level caches that can persist between fits.
        
        This is critical for preventing cache-related issues when using metrics
        like CCI that store results directly on node objects.
        """
        if node is None:
            return
            
        # Clear CCI and impurity caches from this node
        if 'aux_cci_cache' in node:
            del node['aux_cci_cache']
        if 'aux_impurity_cache' in node:
            del node['aux_impurity_cache']
            
        # Recursively clear children
        if 'children' in node:
            for child_node in node['children'].values():
                self._clear_node_caches_recursive(child_node)
    
    def build_tree(self, X: np.array, y: np.array, bad_cuts_limit: int = 3, index: str = 'cci'):
        """
        Main tree construction algorithm using iterative CCI-based splitting.
        
        This method implements the core fuzzy CART algorithm that builds the
        decision tree by iteratively finding and executing the best splits.
        It continues until stopping criteria are met (max rules, low coverage,
        or no beneficial splits available).
        
        The algorithm:
        1. Initializes the root node
        2. Iteratively finds the best node to split using CCI
        3. Executes the split if it improves classification
        4. Stops when constraints are violated or no improvement is possible
        
        Parameters
        ----------
        X : np.array
            Training data features with shape (n_samples, n_features).
        y : np.array
            Training data labels with shape (n_samples,).
        bad_cuts_limit : int, default=3
            Number of consecutive bad cuts before stopping.
        index : str, default='cci'
            Split evaluation metric ('cci' or 'purity').
            Note: 'purity' mode uses Gini impurity for split evaluation.
        """
        # DEFENSIVE: Ensure completely clean state before building tree
        # Clear all caches including node-level caches that persist between fits
        self._coverage_cache = {}
        self._gini_cache = {}
        self._prediction_cache = None
        
        # Clear any existing node caches from previous fits
        if hasattr(self.classifier, '_root') and self.classifier._root is not None:
            self._clear_node_caches_recursive(self.classifier._root)
        
        # CRITICAL: Clear node_dict_access AFTER clearing node caches
        # This ensures no stale references persist when using CCI metric
        self.classifier.node_dict_access = {}
        
        # Stopping criteria
        self.build_root(X, y)
        best_coverage_achievable = 1.0
        bad_cuts = 0

        # OPTIMIZATION: Pre-warm membership cache once for entire training process
        self.classifier._get_cached_memberships(X)
        
        # OPTIMIZATION: Cache baseline prediction to avoid repeated computation
        baseline_prediction = None

        while self.tree_rules < self.max_rules and best_coverage_achievable >= self.coverage_threshold:
            # OPTIMIZATION: Clear split-specific caches but keep membership cache
            self.classifier._clear_all_split_caches()

            if index == 'purity':
                best_impurity_improvement, best_node = self.get_best_node_split(self.root, X, y)
                _best_cci = None
                best_result = best_impurity_improvement
            else:
                best_cci, best_node, _best_impurity = self.get_best_node_split_cci(self.root, X, y)
                best_result = best_cci

            # Find the node to split
            node_to_split = self.classifier._find_node_by_name(best_node)

            if best_result <= self.classifier.min_improvement:
                bad_cuts += 1
                if bad_cuts >= bad_cuts_limit:
                    break
                
            # Make sure that the best gain is actually a feature not a finish signal (-1)
            if 'aux_impurity_cache' in node_to_split and node_to_split['aux_impurity_cache']['feature'] == -1:
                break
            elif 'aux_cci_cache' in node_to_split and node_to_split['aux_cci_cache']['feature'] == -1:
                break
            else:
                # Split the atom! (now creates multiple children for all linguistic labels)
                self.split_node(node_to_split, X, y)
                best_coverage_achievable = self.get_best_possible_coverage(X, y)


        # Change the prediction in root node to majority class
        self.root['prediction'] = majority_class(y, classes=self.classes_)
        # Update root probabilities after tree construction
        self.root['class_probabilities'] = class_probabilities(y, classes=self.classes_)
        
        # OPTIMIZATION: Clear all caches after training to save memory
        self.classifier._membership_cache = {}
        self._coverage_cache = {}
        self._gini_cache = {}
        self._prediction_cache = None
        self.classifier._last_X_shape = None

    def build_root(self, X: np.array, y: np.array):
        """
        Initialize the root node of the fuzzy decision tree.
        
        Creates the root node with full membership for all samples and
        initializes the tree structure. The root represents the starting
        point before any fuzzy splits are applied, encompassing the entire
        dataset with uniform membership.
        
        Parameters
        ----------
        X : np.array
            Training data features used to determine tree structure.
        y : np.array
            Training data labels used for initial tree setup.
        """
        existing_membership = np.ones(X.shape[0])
        # Create path structure - boolean array, one per feature (not per fuzzy set)
        actual_path = np.ones(len(self.fuzzy_partitions), dtype=bool)

        self.classifier._root = {
            'depth': 0,
            'existing_membership': existing_membership,
            'father_path': actual_path.copy(),
            'child_splits': actual_path.copy(),  # Features available for splitting
            'name': 'root',
            'prediction': majority_class(y, existing_membership, self.classes_),
            'coverage': np.sum(existing_membership) / len(y),
            'class_probabilities': class_probabilities(y, existing_membership, self.classes_)
        }

        # Clear and initialize node_dict_access (don't overwrite, respect prior clearing)
        self.classifier.node_dict_access.clear()
        self.classifier.node_dict_access['root'] = self.classifier._root
        self.classifier.tree_rules = 1

    def node_impurity_checks(self, node: Dict[str, Any], X: np.array, y: np.array) -> float:
        """
        Evaluate all possible feature splits using Gini impurity metrics.
        
        This method evaluates each feature as a whole, computing the weighted Gini impurity
        improvement that would result from partitioning the node using ALL linguistic labels
        of that feature simultaneously. This is more aligned with traditional decision trees.
        
        For each feature, it creates partitions for all linguistic labels and computes
        the weighted average impurity across all partitions. The feature with the best
        overall impurity reduction is selected.
        
        Parameters
        ----------
        node : dict
            Node to evaluate for potential splits.
        X : np.array
            Training data features.
        y : np.array
            Training data labels.
            
        Returns
        -------
        float
            Best impurity improvement (reduction) achievable for this node.
        """
        # Check if this node already has cached results
        if 'aux_impurity_cache' in node:
            return node['aux_impurity_cache']['best_impurity_improvement']

        # Initialize tracking variables
        best_impurity_improvement = -1.0
        best_feature = -1
        
        # Get current node membership and labels
        existing_membership = node['existing_membership']
        
        # Calculate current node impurity (Gini)
        current_impurity = compute_fuzzy_gini_impurity(existing_membership, y)
        
        # CRITICAL: Check which features haven't been used in this path
        # Now we check features, not individual linguistic labels
        valid_features = node['father_path'] & node['child_splits']
        
        # Test each feature (not individual fuzzy sets)
        for feature_idx in range(len(self.fuzzy_partitions)):
            # Check if this feature is still available for splitting
            if valid_features[feature_idx]:
                # Use the extracted function to calculate weighted impurity for all linguistic labels
                total_weighted_impurity, total_membership, valid_partitions_data = evaluate_fuzzy_feature_partition(
                    existing_membership, y, X, self.fuzzy_partitions, feature_idx, 
                    compute_fuzzy_gini_impurity
                )
                
                # Only consider this feature if it has valid partitions and total membership > 0
                if valid_partitions_data > 0 and total_membership > 0:
                    # Calculate weighted average impurity for this feature split
                    weighted_avg_impurity = total_weighted_impurity / total_membership
                    
                    # Calculate impurity improvement
                    impurity_improvement = current_impurity - weighted_avg_impurity
                    
                    # Update best split if this feature is better
                    if impurity_improvement > best_impurity_improvement:
                        best_impurity_improvement = impurity_improvement
                        best_feature = feature_idx

        # Cache results (no fuzzy_set since we split on entire features)
        node['aux_impurity_cache'] = {
            'best_impurity_improvement': best_impurity_improvement,
            'feature': best_feature,
            'fuzzy_set': -1  # Not applicable for feature-based splits
        }
        
        return best_impurity_improvement

    def get_best_node_split(self, node_father: Dict[str, Any], X: np.array, y: np.array) -> Tuple[float, str]:
        """
        Find the best node to split in the tree using impurity improvement.
        
        This method recursively evaluates all nodes in the tree to find the one
        that would benefit most from splitting, based on Gini impurity reduction.
        It returns both the improvement value and the name of the best node.
        
        Parameters
        ----------
        node_father : dict
            Current node being evaluated (starts with root).
        X : np.array
            Training data features.
        y : np.array
            Training data labels.
            
        Returns
        -------
        tuple[float, str]
            Best impurity improvement and name of best node to split.
        """
        # Check if node has children to recurse into
        if 'children' not in node_father:
            # Leaf node - evaluate for splitting
            impurity_improvement = self.node_impurity_checks(node_father, X, y)
            return impurity_improvement, node_father['name']
        
        # Internal node - check both this node and its children
        best_impurity_improvement = self.node_impurity_checks(node_father, X, y)
        best_node_name = node_father['name']
        
        # Recursively check all children
        for child_name, child_node in node_father['children'].items():
            child_impurity_improvement, child_best_node = self.get_best_node_split(child_node, X, y)
            
            if child_impurity_improvement > best_impurity_improvement:
                best_impurity_improvement = child_impurity_improvement
                best_node_name = child_best_node

        return best_impurity_improvement, best_node_name


    def node_cci_checks(self, node: Dict[str, Any], X: np.array, y: np.array) -> float:
        """
        Evaluate all possible feature splits using Complete Classification Index (CCI).
        
        CCI measures the improvement in correct classification that would result
        from splitting on a feature (using all its linguistic labels). This approach
        evaluates each feature as a whole rather than individual fuzzy sets.
        
        For each feature, it computes the CCI improvement that would result from
        partitioning using all linguistic labels of that feature simultaneously.
        
        Parameters
        ----------
        node : dict
            Node to evaluate for potential splits.
        X : np.array
            Training data features.
        y : np.array
            Training data labels.
            
        Returns
        -------
        float
            Best CCI improvement achievable for this node.
        """
        # Check if this node already has cached results
        if 'aux_cci_cache' in node:
            return node['aux_cci_cache']['best_cci']

        # Initialize tracking variables
        best_cci = 0.0
        best_feature = -1
        
        # Get current node membership
        existing_membership = node['existing_membership']
        
        # Calculate baseline CCI for current node
        membership_father_incorrect = np.mean(existing_membership * (y != node['prediction']))
        membership_father_correct = np.mean(existing_membership * (y == node['prediction']))
        membership_father_validity = membership_father_correct - membership_father_incorrect

        # Check which features haven't been used in this path
        # Now we check features, not individual linguistic labels
        valid_features = node['father_path'] & node['child_splits']
        
        # Test each feature (not individual fuzzy sets)
        for feature_idx in range(len(self.fuzzy_partitions)):
            # Check if this feature is still available for splitting
            if valid_features[feature_idx]:
                # For CCI, we need a metric function that computes validity (correct - incorrect)
                def cci_validity_metric(child_membership, y_labels):
                    """Calculate CCI validity for a partition."""
                    child_prediction = majority_class(y_labels, child_membership, self.classes_)
                    membership_child_incorrect = np.mean(child_membership * (y_labels != child_prediction))
                    membership_child_correct = np.mean(child_membership * (y_labels == child_prediction))
                    return membership_child_correct - membership_child_incorrect
                
                # Use the extracted function to calculate weighted CCI for all linguistic labels
                total_weighted_validity, total_membership, valid_partitions_count = evaluate_fuzzy_feature_partition(
                    existing_membership, y, X, self.fuzzy_partitions, feature_idx, 
                    cci_validity_metric
                )
                
                # Only consider this feature if it has valid partitions and total membership > 0
                if valid_partitions_count > 0 and total_membership > 0:
                    # Calculate weighted average validity for this feature split
                    weighted_avg_validity = total_weighted_validity / total_membership
                    
                    # Calculate CCI improvement
                    total_improvement_ratio = (weighted_avg_validity - membership_father_validity) / membership_father_validity if membership_father_validity != 0 else weighted_avg_validity

                    # Handle sign differences
                    if (weighted_avg_validity >= 0 and membership_father_validity < 0) or (weighted_avg_validity < 0 and membership_father_validity >= 0):
                        total_improvement_ratio = weighted_avg_validity
                    
                    # Update best split if this feature is better
                    if total_improvement_ratio > best_cci:
                        best_cci = total_improvement_ratio
                        best_feature = feature_idx

        # Cache results (no fuzzy_set since we split on entire features)
        node['aux_cci_cache'] = {
            'best_cci': best_cci,
            'feature': best_feature,
            'fuzzy_set': -1  # Not applicable for feature-based splits
        }
        
        return best_cci

    def get_best_node_split_cci(self, node_father: Dict[str, Any], X: np.array, y: np.array) -> Tuple[float, str, float]:
        """
        Find the best node to split in the tree using CCI improvement.
        
        This method recursively evaluates all nodes in the tree to find the one
        that would benefit most from splitting, based on Complete Classification Index.
        
        Parameters
        ----------
        node_father : dict
            Current node being evaluated (starts with root).
        X : np.array
            Training data features.
        y : np.array
            Training data labels.
            
        Returns
        -------
        tuple[float, str, float]
            Best CCI improvement, name of best node to split, and purity improvement.
        """
        # Check if node has children to recurse into

        # Leaf node - evaluate for splitting
        if 'children' not in node_father:
            cci_improvement = self.node_cci_checks(node_father, X, y)
            impurity_improvement = self.node_impurity_checks(node_father, X, y)
            return cci_improvement, node_father['name'], impurity_improvement
        
        # Internal node - check both this node and its children
        best_cci = self.node_cci_checks(node_father, X, y)
        best_impurity = self.node_impurity_checks(node_father, X, y)
        best_node_name = node_father['name']
        
        # Recursively check all children
        for child_name, child_node in node_father['children'].items():
            child_cci, child_best_node, child_impurity = self.get_best_node_split_cci(child_node, X, y)
            
            if child_cci > best_cci:
                best_cci = child_cci
                best_node_name = child_best_node
                best_impurity = child_impurity
        
        return best_cci, best_node_name, best_impurity


    def split_node(self, node: Dict[str, Any], X: np.array, y: np.array):
        """
        Execute a feature split, creating child nodes for ALL linguistic labels.
        
        This method performs the actual tree split operation by creating child nodes
        for all linguistic labels of the selected feature simultaneously. This aligns
        with traditional decision tree semantics while maintaining fuzzy membership.
        
        Parameters
        ----------
        node : dict
            Node to split.
        X : np.array
            Training data features.
        y : np.array
            Training data labels.
        """
        # Get cached split information
        if 'aux_impurity_cache' in node:
            cache = node['aux_impurity_cache']
        elif 'aux_cci_cache' in node:
            cache = node['aux_cci_cache']
        else:
            raise ValueError("No cached split information found for node")
        
        feature = cache['feature']
        
        if feature == -1:
            return  # No valid split found
        
        # Create children dictionary if not exists
        if 'children' not in node:
            node['children'] = {}
        
        # Get parent membership
        existing_membership = node['existing_membership']
        
        # Mark this entire feature as used for future splits
        node['child_splits'][feature] = False
        
        # Create child nodes for ALL linguistic labels of the selected feature
        for fuzzy_set_idx in range(len(self.fuzzy_partitions[feature])):
            fuzzy_set_obj = self.fuzzy_partitions[feature][fuzzy_set_idx]
            child_existing_membership = existing_membership * fuzzy_set_obj.membership(X[:, feature])
            
            child_prediction = majority_class(y, child_existing_membership, self.classes_)
            
            # Child inherits parent's father_path with the current feature marked as used
            child_father_path = node['father_path'].copy()
            child_father_path[feature] = False  # Mark this feature as used
            
            # Child starts with all splits available in child_splits (except this feature)
            child_child_splits = node['child_splits'].copy()
            child_child_splits[feature] = False  # This feature is now used
            
            # Create child node
            new_node = {
                'depth': node['depth'] + 1,
                'existing_membership': child_existing_membership,
                'father_path': child_father_path,
                'child_splits': child_child_splits,
                'name': NodeValidator.create_node_name(node['name'], feature, fuzzy_set_idx),
                'prediction': child_prediction,
                'feature': feature,
                'fuzzy_set': fuzzy_set_idx,
                'coverage': TreeMetrics.calculate_node_coverage(child_existing_membership, len(y)),
                'class_probabilities': class_probabilities(y, child_existing_membership, self.classes_)
            }
            
            # Check for name conflicts
            if new_node['name'] in self.node_dict_access:
                raise ValueError(f"Node name {new_node['name']} already exists in node_dict_access. "
                               f"Current nodes: {list(self.node_dict_access.keys())}")
            
            if new_node['name'] in node['children'].keys():
                raise ValueError(f"Node name {new_node['name']} already exists in parent's children. "
                               f"Parent children: {list(node['children'].keys())}")
            
            # Add to tree structure
            node['children'][new_node['name']] = new_node
            self.node_dict_access[new_node['name']] = new_node
            
            # Increment tree rules counter for each child created
            self.classifier.tree_rules += 1
        
        # Note: Keep predictions in internal nodes for CCI calculations
        # Internal nodes may be evaluated for future splits and need predictions
        # for CCI metric computation (correct vs incorrect classification)
        
        # Invalidate leaf cache since tree structure changed
        self.classifier._invalidate_leaf_cache()

    def get_best_possible_coverage(self, X: np.array, y: np.array, sample_weight=None) -> float:
        """
        Calculate the best possible coverage that could be achieved by adding new nodes.
        
        This method evaluates all possible feature splits across all current leaf nodes
        to find the maximum coverage that any new child node could achieve. This is used
        for early termination - if no possible new node could meet the coverage threshold,
        tree growth can stop.
        
        Parameters
        ----------
        X : np.array
            Training data features.
        y : np.array
            Training data labels.
        sample_weight : array-like, optional
            Sample weights (not currently used).
            
        Returns
        -------
        float
            Maximum coverage achievable by any potential new node.
        """
        best_coverage = 0.0
        
        # Check all current leaf nodes for potential feature splits
        for node_name, node in self.node_dict_access.items():
            if NodeValidator.is_leaf_node(node):
                existing_membership = node['existing_membership']
                
                # Test all available features for this node
                valid_features = node['father_path'] & node['child_splits']
                
                for feature_idx in range(len(self.fuzzy_partitions)):
                    # Check if this feature is still available for splitting
                    if valid_features[feature_idx]:
                        # Check coverage for all linguistic labels of this feature
                        for fuzzy_set_idx in range(len(self.fuzzy_partitions[feature_idx])):
                            fuzzy_set = self.fuzzy_partitions[feature_idx][fuzzy_set_idx]
                            child_membership = existing_membership * fuzzy_set.membership(X[:, feature_idx])
                            coverage = TreeMetrics.calculate_node_coverage(child_membership, len(y))
                            
                            if coverage > best_coverage:
                                best_coverage = coverage
        
        return best_coverage