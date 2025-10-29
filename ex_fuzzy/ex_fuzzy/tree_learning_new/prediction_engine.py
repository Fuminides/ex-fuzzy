"""
Prediction engine module for FuzzyCART algorithm.

This module contains all prediction algorithms and fuzzy membership computation
for the FuzzyCART classifier, including recursive tree traversal and optimized
batch prediction methods.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional

from .base import TreeComponent, NodeValidator, TreeMetrics, DEFAULT_EPSILON
from .metrics.fuzzy_metrics import majority_class, class_probabilities


class PredictionEngine(TreeComponent):
    """
    Handles all prediction algorithms for FuzzyCART.
    
    This class implements the core fuzzy inference engine that evaluates 
    samples against the trained tree using fuzzy membership functions.
    """
    
    def __init__(self, parent_classifier):
        """Initialize prediction engine with reference to parent classifier."""
        super().__init__(parent_classifier)
        self._membership_cache = {}
        self._last_X_shape = None
    
    def predict_all_nodes(self, X: np.array, epsilon: float = DEFAULT_EPSILON) -> Tuple[np.array, np.array, np.array]:
        """
        Fuzzy prediction using ONLY leaf nodes (traditional decision tree semantics).
        
        With the new feature-based splitting approach, all linguistic labels are created
        simultaneously for each feature, so internal nodes are never needed for prediction.
        Only leaf nodes make predictions, following traditional decision tree semantics.
        
        Parameters
        ----------
        X : np.array
            Input data array with shape (n_samples, n_features).
        epsilon : float, default=1e-6
            Not used in new approach but kept for compatibility.
        
        Returns
        -------
        tuple[np.array, np.array, np.array]
            Predictions, membership values, and path names for all samples.
        """
        n_samples = X.shape[0]
        
        # Initialize output arrays
        predictions = np.full(n_samples, self.root['prediction'])  # Default to root
        best_memberships = np.zeros(n_samples)
        paths = np.full(n_samples, 'root', dtype=object)
        
        # Get all leaf nodes (only leaves predict in new approach)
        if not hasattr(self.classifier, '_cached_leaves'):
            self.classifier._cached_leaves = self.classifier.tree_structure.extract_leaves()
        
        leaves = self.classifier._cached_leaves
        
        # If no leaves (only root), return root predictions
        if not leaves:
            return predictions, np.ones(n_samples), paths
        
        # For each leaf, compute membership for all samples
        for leaf in leaves:
            # Compute path membership for all samples
            path_membership = np.ones(n_samples)
            
            # Multiply membership along the path to this leaf
            for feature_idx, fuzzy_set_idx in zip(leaf['path_features'], leaf['path_fuzzy_sets']):
                fuzzy_set = self.fuzzy_partitions[feature_idx][fuzzy_set_idx]
                feature_membership = fuzzy_set.membership(X[:, feature_idx])
                path_membership *= feature_membership
            
            # Update predictions where this leaf has better membership
            better_samples = path_membership > best_memberships
            
            predictions[better_samples] = leaf['prediction']
            best_memberships[better_samples] = path_membership[better_samples]
            paths[better_samples] = leaf['name']
        
        return predictions, best_memberships, paths

    def predict_proba_all_nodes(self, X: np.array, epsilon: float = DEFAULT_EPSILON) -> np.array:
        """
        Predict class probabilities using fuzzy membership weighting across leaf nodes only.
        
        With the new feature-based splitting approach, only leaf nodes make predictions.
        This method computes probability distributions by evaluating fuzzy membership
        to all leaf nodes in the tree.
        
        Parameters
        ----------
        X : np.array
            Input data array with shape (n_samples, n_features).
        epsilon : float, default=1e-6
            Not used in new approach but kept for compatibility.
        
        Returns
        -------
        np.array
            Probability matrix of shape (n_samples, n_classes).
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        # Initialize probability accumulation matrix
        probabilities = np.zeros((n_samples, n_classes))
        total_memberships = np.zeros(n_samples)
        
        # Get all leaf nodes (only leaves predict in new approach)
        if not hasattr(self.classifier, '_cached_leaves'):
            self.classifier._cached_leaves = self.classifier.tree_structure.extract_leaves()
        
        # Map class names to indices
        class_to_idx = {cls: i for i, cls in enumerate(self.classes_)}
        
        # Evaluate each leaf node
        for leaf in self.classifier._cached_leaves:
            # Compute path membership for all samples
            path_membership = np.ones(n_samples)
            
            # Multiply membership along the path
            for feature_idx, fuzzy_set_idx in zip(leaf['path_features'], leaf['path_fuzzy_sets']):
                fuzzy_set = self.fuzzy_partitions[feature_idx][fuzzy_set_idx]
                feature_membership = fuzzy_set.membership(X[:, feature_idx])
                path_membership *= feature_membership
            
            # Add weighted probabilities for this leaf
            if 'class_probabilities' in leaf:
                for class_idx, prob in enumerate(leaf['class_probabilities']):
                    probabilities[:, class_idx] += path_membership * prob
            else:
                # Fallback: use crisp prediction
                class_idx = class_to_idx.get(leaf['prediction'], 0)
                probabilities[:, class_idx] += path_membership
            
            total_memberships += path_membership
        
        # Normalize probabilities
        for i in range(n_samples):
            if total_memberships[i] > 0:
                probabilities[i] /= total_memberships[i]
            else:
                # Fallback to uniform distribution
                probabilities[i] = 1.0 / n_classes
        
        return probabilities

    def predict_recursive(self, x: np.array, node: Dict[str, Any], membership: Optional[np.array] = None, 
                         paths: Optional[np.array] = None, best_membership: Optional[np.array] = None, 
                         prediction: Optional[np.array] = None) -> Tuple[np.array, np.array, np.array]:
        """
        Core recursive prediction method using batch processing for efficient fuzzy tree traversal.
        
        This method implements the batch prediction algorithm that traverses the fuzzy
        decision tree for multiple samples simultaneously. It uses numpy vectorization
        to efficiently compute fuzzy memberships and update predictions for all samples
        in parallel.
        
        Parameters
        ----------
        x : np.array
            Input data array with shape (n_samples, n_features).
        node : dict
            Current tree node being processed.
        membership : np.array, optional
            Current membership values for each sample.
        paths : np.array, optional
            Current path names for each sample.
        best_membership : np.array, optional
            Best membership values found so far for each sample.
        prediction : np.array, optional
            Current predictions for each sample.
        
        Returns
        -------
        tuple[np.array, np.array, np.array]
            Final predictions, membership values, and paths for all samples.
        """
        n_samples = x.shape[0]
        
        # Initialize arrays if not provided
        if membership is None:
            membership = np.ones(n_samples)
            best_membership = np.zeros(n_samples) - 1.0
            prediction = np.full(n_samples, self.root['prediction'])
            paths = np.full(n_samples, 'root', dtype=object)
        
        # Update predictions where current membership is better
        better_samples = membership > best_membership
        prediction[better_samples] = node['prediction']
        best_membership[better_samples] = membership[better_samples]
        paths[better_samples] = node['name']
        
        # Process children if they exist
        if 'children' in node:
            for child_name, child in node['children'].items():
                # Calculate combined membership for all samples
                feature = child['feature']
                fuzzy_set_idx = child['fuzzy_set']
                fuzzy_set = self.fuzzy_partitions[feature][fuzzy_set_idx]
                
                # Compute fuzzy membership for this feature
                feature_membership = fuzzy_set.membership(x[:, feature])
                full_path_membership = membership * feature_membership
                
                # Recursively process child
                child_pred, path_membership, child_paths = self.predict_recursive(
                    x, child, membership=full_path_membership, paths=paths, 
                    best_membership=best_membership, prediction=prediction
                )
                
                # Update tracking arrays (passed by reference, so updates persist)
                prediction[:] = child_pred
                best_membership[:] = path_membership
                paths[:] = child_paths
        
        return prediction, best_membership, paths

    def predict_proba_recursive(self, x: np.array, node: Dict[str, Any], membership: Optional[np.array] = None, 
                               best_membership: Optional[np.array] = None, 
                               class_probabilities: Optional[np.array] = None) -> np.array:
        """
        Core recursive probability prediction method using batch processing with cached probabilities.
        
        This method traverses the fuzzy decision tree and uses pre-computed class probability
        distributions stored at each node during tree construction.
        
        Parameters
        ----------
        x : np.array
            Input data array with shape (n_samples, n_features).
        node : dict
            Current tree node being processed.
        membership : np.array, optional
            Current membership values for each sample.
        best_membership : np.array, optional
            Best membership values found so far for each sample.
        class_probabilities : np.array, optional
            Current probability accumulation matrix.
        
        Returns
        -------
        np.array
            Final probability matrix with shape (n_samples, n_classes).
        """
        n_samples = x.shape[0]
        n_classes = len(self.classes_)
        
        if membership is None:
            membership = np.ones(n_samples)
            best_membership = np.zeros(n_samples) - 1.0
            class_probabilities = np.zeros((n_samples, n_classes))
        
        # Update probabilities where current membership is better
        better_samples = membership > best_membership
        
        if np.any(better_samples) and 'class_probabilities' in node:
            # Update class probabilities for better samples
            class_probabilities[better_samples] = node['class_probabilities']
            best_membership[better_samples] = membership[better_samples]
        
        # Process children if they exist
        if 'children' in node:
            for child_name, child in node['children'].items():
                # Calculate combined membership for all samples
                feature = child['feature']
                fuzzy_set_idx = child['fuzzy_set']
                fuzzy_set = self.fuzzy_partitions[feature][fuzzy_set_idx]
                
                # Compute fuzzy membership for this feature
                feature_membership = fuzzy_set.membership(x[:, feature])
                full_path_membership = membership * feature_membership
                
                # Recursively process child
                class_probabilities = self.predict_proba_recursive(
                    x, child, membership=full_path_membership, 
                    best_membership=best_membership, 
                    class_probabilities=class_probabilities
                )
        
        return class_probabilities

    def calculate_membership_to_leaf(self, X: np.array, leaf_node: Dict[str, Any]) -> np.array:
        """
        Calculate fuzzy membership of samples to a specific leaf node.
        
        This method computes the membership degree of each sample to a given
        leaf node by multiplying memberships along the path from root to leaf.
        
        Parameters
        ----------
        X : np.array
            Input data array with shape (n_samples, n_features).
        leaf_node : dict
            Leaf node to calculate membership for.
        
        Returns
        -------
        np.array
            Membership values for each sample to the leaf node.
        """
        n_samples = X.shape[0]
        membership = np.ones(n_samples)
        
        # Get path to this leaf
        path = self.classifier.tree_structure.get_path_to_leaf(leaf_node)
        
        # Multiply membership along the path
        for step in path:
            feature_idx = step['feature']
            fuzzy_set_idx = step['fuzzy_set']
            fuzzy_set = self.fuzzy_partitions[feature_idx][fuzzy_set_idx]
            feature_membership = fuzzy_set.membership(X[:, feature_idx])
            membership *= feature_membership
        
        return membership

    def get_cached_memberships(self, X: np.array) -> dict:
        """
        Get cached membership values or compute them if not cached.
        
        OPTIMIZATION: Cache membership computations to avoid redundant calculations
        across multiple split evaluations.
        
        Parameters
        ----------
        X : np.array
            Input data array.
            
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
            for feature_idx, fuzzy_var in enumerate(self.fuzzy_partitions):
                feature_memberships = np.zeros((len(fuzzy_var), X.shape[0]))
                for fz_idx, fuzzy_set in enumerate(fuzzy_var):
                    feature_memberships[fz_idx] = fuzzy_set.membership(X[:, feature_idx])
                self._membership_cache[feature_idx] = feature_memberships
        
        return self._membership_cache

    def clear_all_split_caches(self):
        """Clear cached split evaluations from all nodes."""
        for node in self.node_dict_access.values():
            if 'aux_impurity_cache' in node:
                del node['aux_impurity_cache']
            if 'aux_cci_cache' in node:
                del node['aux_cci_cache']

    def _node_has_children(self, node_name: str) -> bool:
        """
        Check if a node has children.
        
        Parameters
        ----------
        node_name : str
            Name of the node to check.
            
        Returns
        -------
        bool
            True if node has children, False otherwise.
        """
        if node_name in self.node_dict_access:
            node = self.node_dict_access[node_name]
            return 'children' in node and len(node['children']) > 0
        return False

    def _get_node_children_names(self, node_name: str) -> list:
        """
        Get names of all children of a node.
        
        Parameters
        ----------
        node_name : str
            Name of the parent node.
            
        Returns
        -------
        list
            List of children node names.
        """
        if node_name in self.node_dict_access:
            node = self.node_dict_access[node_name]
            if 'children' in node:
                return list(node['children'].keys())
        return []