"""
Main FuzzyCART classifier using modular architecture.

This module contains the main FuzzyCART class that integrates all tree modules
using composition pattern while maintaining backward compatibility.
"""

import numpy as np
from sklearn.base import ClassifierMixin
from typing import Tuple, Dict, Any, Optional

# Import tree modules
from .tree_builder import TreeBuilder
from .prediction_engine import PredictionEngine  
from .tree_structure import TreeStructure
from .pruning import PruningEngine

# Import for fuzzy sets and metrics
try:
    from .. import fuzzy_sets as fs
    from .metrics.fuzzy_metrics import majority_class, class_probabilities
except ImportError:
    # Handle import for different package structures
    try:
        import ex_fuzzy.fuzzy_sets as fs
        from ex_fuzzy.metrics.fuzzy_metrics import majority_class, class_probabilities
    except ImportError:
        pass


class FuzzyCART(ClassifierMixin):
    """
    Fuzzy CART (Classification and Regression Trees) classifier implementation.
    
    This class implements a fuzzy extension of the CART algorithm that uses
    fuzzy partitions instead of crisp splits. The algorithm builds a decision
    tree where each split is based on fuzzy membership functions, allowing
    for soft decisions and handling of uncertainty in the data.
    
    The classifier uses the Complete Classification Index (CCI) as the primary
    splitting criterion, which measures the improvement in classification
    accuracy rather than traditional impurity measures.
    
    Parameters
    ----------
    fuzzy_partitions : list[fs.fuzzyVariable]
        List of fuzzy variables, one for each feature. Each fuzzy variable
        contains multiple fuzzy sets that define the partitioning of the
        feature space.
    max_rules : int, default=10
        Maximum number of rules (leaf nodes) allowed in the tree. Controls
        tree complexity and helps prevent overfitting.
    max_depth : int, default=5
        Maximum depth of the tree. Limits how deep the tree can grow.
    coverage_threshold : float, default=0.00
        Minimum coverage ratio required for a split to be considered valid.
        Splits covering fewer samples than this threshold are rejected.
    min_improvement : float, default=0.01
        Minimum improvement required for a split to be executed.
    ccp_alpha : float, default=0.0
        Complexity parameter for cost complexity pruning.
    target_metric : str, default='cci'
        Target metric for split evaluation ('cci' or 'purity').
        Note: 'purity' mode uses Gini impurity for split evaluation.
    sample_for_splits : bool, optional
        Whether to sample data for split evaluation (performance optimization).
    sample_size : int, default=10000
        Size of sample for split evaluation if sampling is enabled.
    
    Attributes
    ----------
    classes_ : np.array
        Unique class labels found in the training data.
    tree_rules : int
        Current number of rules (nodes) in the tree.
    _root : dict
        Root node of the decision tree containing tree structure.
    node_dict_access : dict
        Dictionary for fast access to tree nodes by name.
    """

    def __init__(self, fuzzy_partitions, max_rules: int = 10, max_depth: int = 5, 
                 coverage_threshold: float = 0.00, min_improvement: float = 0.01, 
                 ccp_alpha: float = 0.0, target_metric: str = 'cci', 
                 sample_for_splits: Optional[bool] = None, sample_size: int = 10000):
        """Initialize FuzzyCART with parameters and create component modules."""
        
        # Store parameters
        self.fuzzy_partitions = fuzzy_partitions
        self.max_rules = max_rules
        self.max_depth = max_depth
        self.coverage_threshold = coverage_threshold
        self.min_improvement = min_improvement
        self.ccp_alpha = ccp_alpha
        self.target_metric = target_metric
        self.sample_for_splits = sample_for_splits
        self.sample_size = sample_size
        
        # Initialize tree state
        self.classes_ = None
        self.tree_rules = 0
        self._root = None
        self.node_dict_access = {}
        
        # Initialize caching
        self._membership_cache = {}
        self._last_X_shape = None
        
        # Create component modules using composition
        self.tree_builder = TreeBuilder(self)
        self.prediction_engine = PredictionEngine(self)
        self.tree_structure = TreeStructure(self)
        self.pruning_engine = PruningEngine(self)

    def fit(self, X: np.array, y: np.array, patience: int = 3):
        """
        Train the fuzzy decision tree on the provided data.
        
        This method builds the fuzzy decision tree by iteratively finding
        and executing the best splits until stopping criteria are met.
        
        Parameters
        ----------
        X : np.array
            Training data features with shape (n_samples, n_features).
        y : np.array
            Training data labels with shape (n_samples,).
        patience : int, default=3
            Number of consecutive bad cuts before stopping tree growth.
            
        Returns
        -------
        self
            Returns self for method chaining.
        """
        # Store unique classes
        self.classes_ = np.unique(y)
        
        # Reset tree state
        self.tree_rules = 0
        self._root = None
        self.node_dict_access = {}
        
        # Clear all caches to prevent stale data from previous fits
        self._membership_cache = {}
        self._last_X_shape = None
        
        # Clear TreeBuilder caches
        self.tree_builder._coverage_cache = {}
        self.tree_builder._gini_cache = {}
        self.tree_builder._prediction_cache = None
        
        # Clear any cached tree structure data
        if hasattr(self, '_cached_leaves'):
            delattr(self, '_cached_leaves')
        if hasattr(self, '_cached_all_nodes'):
            delattr(self, '_cached_all_nodes')
        
        # Build the tree using tree builder module
        self.tree_builder.build_tree(X, y, bad_cuts_limit=patience, index=self.target_metric)
        
        return self

    def predict(self, X: np.array) -> np.array:
        """
        Predicts the class for given samples using fuzzy membership evaluation across ALL nodes.

        In fuzzy decision trees, any node can provide the best prediction based on membership
        strength, not just leaf nodes. This method evaluates all nodes in the tree and selects
        the prediction from the node with highest membership for each sample.

        Parameters
        ----------
        X : np.array
            Data to predict. Each row is a sample.

        Returns
        -------
        np.array
            Predicted class for each sample.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Use fuzzy membership evaluation across all nodes
        prediction, _, _ = self.prediction_engine.predict_all_nodes(X)
        return prediction

    def predict_with_path(self, X: np.array) -> Tuple[np.array, np.array, np.array]:
        """
        Predicts the class for given samples along with membership and path information.

        In fuzzy decision trees, evaluates all nodes to find the one with highest membership
        for each sample, providing the prediction from the best-matching node.

        Parameters
        ----------
        X : np.array
            Data to predict. Each row is a sample.

        Returns
        -------
        tuple[np.array, np.array, np.array]
            Predicted classes, membership values, and paths for each sample.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Use fuzzy membership evaluation across all nodes with full output
        predictions, memberships, paths = self.prediction_engine.predict_all_nodes(X)
        return predictions, memberships, paths

    def predict_proba(self, X: np.array) -> np.array:
        """
        Predict class probabilities for given samples using fuzzy membership weighting across ALL nodes.
        
        This method computes probability distributions over all classes for each sample
        by evaluating fuzzy membership to all nodes in the tree, not just leaves. The probabilities
        are derived from the weighted voting mechanism across all nodes, providing soft predictions
        that reflect the true fuzzy nature of decision tree classification.
        
        Parameters
        ----------
        X : np.array
            Data to predict probabilities for. Each row is a sample.
        
        Returns
        -------
        np.array
            Array of shape (n_samples, n_classes) where each row contains the
            probability distribution over classes for the corresponding sample.
            Probabilities sum to 1.0 for each sample.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        return self.prediction_engine.predict_proba_all_nodes(X)

    def predict_all_leaves(self, X: np.array) -> Tuple[Dict, Dict]:
        """
        Get membership values and predictions for all leaf nodes for each sample.
        
        This method computes the fuzzy membership degree of each sample to every
        leaf node in the tree, along with each leaf's prediction. This provides
        a complete picture of how samples relate to all possible decision paths.
        
        Parameters
        ----------
        X : np.array
            Data to predict. Each row is a sample.
        
        Returns
        -------
        tuple[dict, dict]
            Two dictionaries: (memberships, predictions) where keys are leaf names
            and values are arrays of membership/prediction values for each sample.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Get all leaf nodes
        leaf_nodes = self.tree_structure.get_all_leaf_nodes()
        
        memberships = {}
        predictions = {}
        
        for leaf_name, leaf_node in leaf_nodes.items():
            # Calculate membership to this leaf
            membership = self.prediction_engine.calculate_membership_to_leaf(X, leaf_node)
            memberships[leaf_name] = membership
            predictions[leaf_name] = np.full(X.shape[0], leaf_node['prediction'])
        
        return memberships, predictions

    def predict_all_leaves_matrix(self, X: np.array) -> Tuple[np.array, np.array, list]:
        """
        Matrix version of predict_all_leaves for efficient computation.
        
        Parameters
        ----------
        X : np.array
            Data to predict. Each row is a sample.
        
        Returns
        -------
        tuple[np.array, np.array, list]
            Membership matrix, prediction matrix, and list of leaf names.
        """
        memberships_dict, predictions_dict = self.predict_all_leaves(X)
        
        # Convert to matrices
        leaf_names = list(memberships_dict.keys())
        n_samples = X.shape[0]
        n_leaves = len(leaf_names)
        
        membership_matrix = np.zeros((n_samples, n_leaves))
        prediction_matrix = np.zeros((n_samples, n_leaves))
        
        for i, leaf_name in enumerate(leaf_names):
            membership_matrix[:, i] = memberships_dict[leaf_name]
            prediction_matrix[:, i] = predictions_dict[leaf_name]
        
        return membership_matrix, prediction_matrix, leaf_names

    def print_tree(self, node: Optional[Dict[str, Any]] = None, prefix: str = "", is_last: bool = True):
        """
        Print a visual representation of the fuzzy decision tree.
        
        This method displays the tree structure in a readable format, showing
        the hierarchy of nodes, their splitting conditions, predictions, and
        coverage information.
        
        Parameters
        ----------
        node : dict, optional
            Node to start printing from. If None, starts from root.
        prefix : str, default=""
            Prefix for indentation (used internally for recursion).
        is_last : bool, default=True
            Whether this is the last child (used internally for formatting).
        """
        if node is None:
            node = self._root
        
        if node is None:
            print("Tree not yet trained.")
            return
        
        # Determine the connector
        connector = "└── " if is_last else "├── "
        current_prefix = prefix + connector
        
        # Print current node
        if node['name'] == 'root':
            print(f"{prefix}{current_prefix}Root: class={node['prediction']}, coverage={node['coverage']:.3f}")
        else:
            # Get feature and fuzzy set information
            feature_idx = node.get('feature', -1)
            fuzzy_set_idx = node.get('fuzzy_set', -1)
            
            if feature_idx >= 0 and fuzzy_set_idx >= 0:
                feature_name = f"Feature_{feature_idx}"
                fuzzy_set_name = f"FuzzySet_{fuzzy_set_idx}"
            else:
                feature_name = "Unknown"
                fuzzy_set_name = "Unknown"
            
            # Additional info
            cci_info = ""
            if 'aux_cci_cache' in node:
                cci_info = f", CCI={node['aux_cci_cache']['best_cci']:.4f}"
            
            print(f"{prefix}{current_prefix}{node['name']}: {feature_name} IS {fuzzy_set_name} → class={node['prediction']}, coverage={node['coverage']:.3f}{cci_info}")
        
        # Print children
        if 'children' in node and node['children']:
            children = list(node['children'].items())
            for i, (child_name, child_node) in enumerate(children):
                is_last_child = (i == len(children) - 1)
                child_prefix = prefix + ("    " if is_last else "│   ")
                self.print_tree(child_node, child_prefix, is_last_child)

    def get_tree_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the current tree.
        
        Returns
        -------
        dict
            Dictionary containing various tree statistics.
        """
        if self._root is None:
            return {"error": "Tree not yet trained"}
        
        stats = {
            'total_nodes': self.tree_structure.count_nodes(),
            'leaf_nodes': self.tree_structure.count_leaves(),
            'max_depth': self.tree_structure.get_tree_depth(),
            'tree_rules': self.tree_rules,
            'classes': self.classes_.tolist() if self.classes_ is not None else None
        }
        
        return stats

    # Delegation methods for backward compatibility and convenience
    def _find_node_by_name(self, name: str):
        """Find node by name (delegates to tree_structure)."""
        return self.tree_structure.find_node_by_name(name)

    def _get_cached_memberships(self, X: np.array) -> dict:
        """Get cached memberships (delegates to prediction_engine)."""
        return self.prediction_engine.get_cached_memberships(X)

    def _clear_all_split_caches(self):
        """Clear split caches (delegates to prediction_engine)."""
        return self.prediction_engine.clear_all_split_caches()

    def _invalidate_leaf_cache(self):
        """Invalidate leaf cache (delegates to tree_structure)."""
        return self.tree_structure.invalidate_leaf_cache()

    # Pruning methods (delegate to pruning_engine)
    def cost_complexity_pruning(self, X: np.array, y: np.array, alpha: Optional[float] = None):
        """Perform cost complexity pruning."""
        return self.pruning_engine.cost_complexity_pruning(X, y, alpha)

    def fit_with_pruning(self, X: np.array, y: np.array, 
                        X_val: Optional[np.array] = None, 
                        y_val: Optional[np.array] = None):
        """Fit with validation-based pruning."""
        return self.pruning_engine.fit_with_pruning(X, y, X_val, y_val)

    def clear_all_caches(self):
        """
        Clear all caches to prevent stale data issues.
        
        This method clears all internal caches including:
        - Membership caches
        - TreeBuilder caches
        - Tree structure caches
        - Split evaluation caches
        
        Useful for debugging cache-related issues or freeing memory.
        """
        # Clear main classifier caches
        self._membership_cache = {}
        self._last_X_shape = None
        
        # Clear TreeBuilder caches
        if hasattr(self, 'tree_builder'):
            self.tree_builder._coverage_cache = {}
            self.tree_builder._gini_cache = {}
            self.tree_builder._prediction_cache = None
        
        # Clear tree structure caches
        if hasattr(self, '_cached_leaves'):
            delattr(self, '_cached_leaves')
        if hasattr(self, '_cached_all_nodes'):
            delattr(self, '_cached_all_nodes')
        
        # Clear node-level caches
        if hasattr(self, 'node_dict_access'):
            for node in self.node_dict_access.values():
                if 'aux_impurity_cache' in node:
                    del node['aux_impurity_cache']
                if 'aux_cci_cache' in node:
                    del node['aux_cci_cache']