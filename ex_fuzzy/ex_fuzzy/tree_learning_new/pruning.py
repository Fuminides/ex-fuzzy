"""
Pruning module for FuzzyCART algorithm.

This module contains cost complexity pruning algorithms and tree optimization
methods for the FuzzyCART classifier.
"""

import numpy as np
import copy
from typing import Dict, Any, Tuple, Optional
from .base import TreeComponent, NodeValidator
from .metrics.fuzzy_metrics import majority_class, class_probabilities


class PruningEngine(TreeComponent):
    """
    Handles tree pruning and optimization for FuzzyCART.
    
    This class implements post-pruning techniques to reduce overfitting
    and improve generalization by removing less useful subtrees.
    """
    
    def __init__(self, parent_classifier):
        """Initialize pruning engine with reference to parent classifier."""
        super().__init__(parent_classifier)
    
    def cost_complexity_pruning(self, X: np.array, y: np.array, alpha: Optional[float] = None):
        """
        Perform cost complexity pruning on the tree.
        
        This method implements the CART pruning algorithm that removes
        subtrees to minimize a cost-complexity measure combining
        prediction error and tree complexity.
        
        Parameters
        ----------
        X : np.array
            Training data features.
        y : np.array
            Training data labels.
        alpha : float, optional
            Complexity parameter. If None, uses classifier's ccp_alpha.
        """
        if alpha is None:
            alpha = self.classifier.ccp_alpha
        
        if alpha <= 0:
            return  # No pruning needed
        
        while True:
            # Find the weakest link (subtree with minimum cost-complexity ratio)
            weakest_node, min_ratio = self._find_weakest_link(X, y)
            
            # If minimum ratio is greater than alpha, stop pruning
            if min_ratio > alpha:
                break
            
            # Prune the weakest subtree
            self._prune_subtree(weakest_node, X, y)
    
    def fit_with_pruning(self, X: np.array, y: np.array, 
                        X_val: Optional[np.array] = None, 
                        y_val: Optional[np.array] = None):
        """
        Fit the tree with validation-based pruning.
        
        This method builds the tree and then prunes it based on
        validation performance to find the optimal tree size.
        
        Parameters
        ----------
        X : np.array
            Training data features.
        y : np.array
            Training data labels.
        X_val : np.array, optional
            Validation data features.
        y_val : np.array, optional
            Validation data labels.
        """
        # First, build the full tree
        self.classifier.fit(X, y)
        
        if X_val is None or y_val is None:
            # No validation data - just apply fixed alpha pruning
            self.cost_complexity_pruning(X, y)
            return
        
        # Save the full tree
        best_tree = self.deep_copy_tree()
        best_score = self.classifier.score(X_val, y_val)
        
        # Try different pruning levels
        alphas = [0.001, 0.01, 0.1, 0.5, 1.0]
        
        for alpha in alphas:
            # Restore full tree
            self.restore_tree(best_tree)
            
            # Apply pruning with this alpha
            self.cost_complexity_pruning(X, y, alpha)
            
            # Evaluate on validation set
            val_score = self.classifier.score(X_val, y_val)
            
            # Keep track of best performing tree
            if val_score > best_score:
                best_score = val_score
                best_tree = self.deep_copy_tree()
        
        # Restore the best tree
        self.restore_tree(best_tree)
    
    def _find_weakest_link(self, X: np.array, y: np.array) -> Tuple[Dict[str, Any], float]:
        """
        Find the subtree with minimum cost-complexity ratio.
        
        Parameters
        ----------
        X : np.array
            Training data features.
        y : np.array
            Training data labels.
            
        Returns
        -------
        tuple
            Weakest node and its cost-complexity ratio.
        """
        min_ratio = float('inf')
        weakest_node = None
        
        def traverse(node):
            """Recursively search for the weakest pruning candidate."""
            nonlocal min_ratio, weakest_node
            
            # Only consider internal nodes for pruning
            if 'children' in node and len(node['children']) > 0:
                ratio = self._calculate_complexity_measure(node, X, y)
                
                if ratio < min_ratio:
                    min_ratio = ratio
                    weakest_node = node
                
                # Recursively check children
                for child in node['children'].values():
                    traverse(child)
        
        traverse(self.root)
        return weakest_node, min_ratio
    
    def _calculate_complexity_measure(self, node: Dict[str, Any], X: np.array, y: np.array) -> float:
        """
        Calculate the cost-complexity measure for a node.
        
        Parameters
        ----------
        node : dict
            Node to calculate measure for.
        X : np.array
            Training data features.
        y : np.array
            Training data labels.
            
        Returns
        -------
        float
            Cost-complexity measure.
        """
        # Calculate impurity of node if it were a leaf
        node_impurity = self._calculate_node_impurity(node, X, y)
        
        # Calculate weighted impurity of subtree
        subtree_impurity = self._calculate_subtree_impurity(node, X, y)
        
        # Calculate number of leaves in subtree
        num_leaves = self._count_leaves(node)
        
        # Cost-complexity measure
        if num_leaves > 1:
            return (node_impurity - subtree_impurity) / (num_leaves - 1)
        else:
            return float('inf')
    
    def _calculate_node_impurity(self, node: Dict[str, Any], X: np.array, y: np.array) -> float:
        """Calculate impurity if node were a leaf."""
        membership = node['existing_membership']
        if np.sum(membership) == 0:
            return 0.0
        
        # Use weighted misclassification rate
        correct = np.sum(membership * (y == node['prediction']))
        total = np.sum(membership)
        return 1.0 - (correct / total) if total > 0 else 0.0
    
    def _calculate_subtree_impurity(self, node: Dict[str, Any], X: np.array, y: np.array) -> float:
        """Calculate weighted impurity of entire subtree."""
        total_impurity = 0.0
        total_weight = 0.0
        
        def collect_leaf_impurity(current_node):
            """Accumulate weighted leaf impurities for a subtree."""
            nonlocal total_impurity, total_weight
            
            if NodeValidator.is_leaf_node(current_node):
                membership = current_node['existing_membership']
                weight = np.sum(membership)
                
                if weight > 0:
                    correct = np.sum(membership * (y == current_node['prediction']))
                    impurity = 1.0 - (correct / weight)
                    total_impurity += weight * impurity
                    total_weight += weight
            else:
                for child in current_node['children'].values():
                    collect_leaf_impurity(child)
        
        collect_leaf_impurity(node)
        return total_impurity / total_weight if total_weight > 0 else 0.0
    
    def _count_leaves(self, node: Dict[str, Any]) -> int:
        """Count number of leaves in subtree rooted at node."""
        if NodeValidator.is_leaf_node(node):
            return 1
        
        count = 0
        if 'children' in node:
            for child in node['children'].values():
                count += self._count_leaves(child)
        return count
    
    def _prune_subtree(self, node: Dict[str, Any], X: np.array, y: np.array):
        """
        Prune a subtree by converting internal node to leaf.
        
        Parameters
        ----------
        node : dict
            Node to convert to leaf.
        X : np.array
            Training data features.
        y : np.array
            Training data labels.
        """
        if 'children' in node:
            # Remove all children from node_dict_access
            self._remove_from_node_dict(node)
            
            # Remove children
            del node['children']
            
            # Recalculate prediction and probabilities for this new leaf
            membership = node['existing_membership']
            node['prediction'] = majority_class(y, membership, self.classes_)
            node['class_probabilities'] = class_probabilities(y, membership, self.classes_)
            
            # Update tree rules count
            self.classifier.tree_rules = len(self.node_dict_access)
            
            # Invalidate caches
            self.classifier.tree_structure.invalidate_leaf_cache()
    
    def _remove_from_node_dict(self, node: Dict[str, Any]):
        """Remove node and all its descendants from node_dict_access."""
        if 'children' in node:
            for child in node['children'].values():
                self._remove_from_node_dict(child)
                if child['name'] in self.node_dict_access:
                    del self.node_dict_access[child['name']]
    
    def deep_copy_tree(self) -> Dict[str, Any]:
        """
        Create a deep copy of the current tree structure.
        
        Returns
        -------
        dict
            Deep copy of tree structure and node dictionary.
        """
        return {
            'root': copy.deepcopy(self.root),
            'node_dict_access': copy.deepcopy(self.node_dict_access),
            'tree_rules': self.tree_rules
        }
    
    def restore_tree(self, tree_backup: Dict[str, Any]):
        """
        Restore tree from a backup.
        
        Parameters
        ----------
        tree_backup : dict
            Tree backup from deep_copy_tree().
        """
        self.classifier._root = tree_backup['root']
        self.classifier.node_dict_access = tree_backup['node_dict_access']
        self.classifier.tree_rules = tree_backup['tree_rules']
        
        # Invalidate caches
        self.classifier.tree_structure.invalidate_leaf_cache()
