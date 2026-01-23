"""
Tree structure and navigation module for FuzzyCART algorithm.

This module handles tree traversal, node extraction, and navigation utilities
for the FuzzyCART classifier, including caching mechanisms for performance.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from .base import TreeComponent, NodeValidator


class TreeStructure(TreeComponent):
    """
    Handles tree structure, navigation, and node management for FuzzyCART.
    
    This class provides utilities for accessing and manipulating tree nodes,
    extracting paths, and managing cached tree structures.
    """
    
    def __init__(self, parent_classifier):
        """Initialize tree structure with reference to parent classifier."""
        super().__init__(parent_classifier)
    
    def extract_leaves(self) -> List[Dict[str, Any]]:
        """
        Extract all leaf nodes from the tree for fuzzy prediction evaluation.
        
        In fuzzy decision trees, leaf nodes represent the final decision paths
        where predictions are made. This method collects all leaves with their
        complete paths from root for direct evaluation during prediction.
        
        Returns
        -------
        list
            List of leaf node dictionaries with their paths and predictions.
        """
        leaves = []
        
        def collect_leaves(node, path_features=None, path_fuzzy_sets=None, path_name=""):
            """Recursively gather leaf nodes and their feature paths."""
            if path_features is None:
                path_features = []
                path_fuzzy_sets = []
            
            # Check if this is a leaf node
            if 'children' not in node or len(node['children']) == 0:
                # This is a leaf - add it to our collection
                leaf_info = {
                    'name': node['name'],
                    'prediction': node['prediction'],
                    'coverage': node['coverage'],
                    'path_features': path_features.copy(),
                    'path_fuzzy_sets': path_fuzzy_sets.copy(),
                    'path_length': len(path_features),
                    'depth': node.get('depth', 0)
                }
                if 'class_probabilities' in node:
                    leaf_info['class_probabilities'] = node['class_probabilities']
                
                leaves.append(leaf_info)
            else:
                # This is an internal node - recurse into children
                for child_name, child_node in node['children'].items():
                    # Build path for child
                    child_path_features = path_features + [child_node['feature']]
                    child_path_fuzzy_sets = path_fuzzy_sets + [child_node['fuzzy_set']]
                    
                    collect_leaves(child_node, child_path_features, child_path_fuzzy_sets, path_name)
        
        # Start collection from root
        collect_leaves(self.root)
        
        return leaves
    
    def extract_all_nodes(self) -> List[Dict[str, Any]]:
        """
        Extract all nodes from the tree for fuzzy prediction evaluation.
        
        In fuzzy decision trees, any node can provide the best prediction based
        on membership strength. This method collects all nodes with their paths
        and predictions for direct evaluation during prediction.
        
        Returns
        -------
        list
            List of all node dictionaries with their paths and predictions.
        """
        all_nodes = []
        
        def collect_nodes(node, path_features=None, path_fuzzy_sets=None):
            """Recursively gather all nodes and their feature paths."""
            if path_features is None:
                path_features = []
                path_fuzzy_sets = []
            
            # Add current node to collection
            node_info = {
                'name': node['name'],
                'prediction': node['prediction'],
                'coverage': node['coverage'],
                'path_features': path_features.copy(),
                'path_fuzzy_sets': path_fuzzy_sets.copy(),
                'path_length': len(path_features),
                'depth': node.get('depth', 0)
            }
            if 'class_probabilities' in node:
                node_info['class_probabilities'] = node['class_probabilities']
            
            all_nodes.append(node_info)
            
            # Recurse into children if they exist
            if 'children' in node:
                for child_name, child_node in node['children'].items():
                    # Build path for child
                    child_path_features = path_features + [child_node['feature']]
                    child_path_fuzzy_sets = path_fuzzy_sets + [child_node['fuzzy_set']]
                    
                    collect_nodes(child_node, child_path_features, child_path_fuzzy_sets)
        
        # Start collection from root
        collect_nodes(self.root)
        
        return all_nodes
    
    def invalidate_leaf_cache(self):
        """
        Invalidate the cached leaf nodes when the tree structure changes.
        
        This should be called whenever a new node is added to the tree
        to ensure the leaf cache is updated for the next prediction.
        """
        if hasattr(self.classifier, '_cached_leaves'):
            delattr(self.classifier, '_cached_leaves')
        if hasattr(self.classifier, '_cached_all_nodes'):
            delattr(self.classifier, '_cached_all_nodes')
    
    def find_node_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Find and return a node by its name.
        
        This method searches the tree structure for a node with the specified
        name and returns the node dictionary if found.
        
        Parameters
        ----------
        name : str
            Name of the node to find.
            
        Returns
        -------
        dict or None
            Node dictionary if found, None otherwise.
        """
        if name in self.node_dict_access:
            return self.node_dict_access[name]
        return None
    
    def get_node_path(self, target_node: Dict[str, Any]) -> List[str]:
        """
        Get the path from root to a target node.
        
        This method traces back from a node to the root to determine the
        complete path through the tree structure.
        
        Parameters
        ----------
        target_node : dict
            Target node to find path to.
            
        Returns
        -------
        list
            List of node names from root to target.
        """
        path = []
        
        def find_path(node, target_name, current_path):
            """Depth-first search for a node while tracking the current path."""
            current_path.append(node['name'])
            
            if node['name'] == target_name:
                path.extend(current_path)
                return True
            
            if 'children' in node:
                for child_name, child_node in node['children'].items():
                    if find_path(child_node, target_name, current_path.copy()):
                        return True
            
            return False
        
        find_path(self.root, target_node['name'], [])
        return path
    
    def get_all_leaf_nodes(self) -> Dict[str, Dict[str, Any]]:
        """
        Get dictionary of all leaf nodes in the tree.
        
        This method traverses the tree and collects all leaf nodes,
        returning them as a dictionary indexed by node name.
        
        Returns
        -------
        dict
            Dictionary mapping node names to leaf node dictionaries.
        """
        leaf_nodes = {}
        
        def traverse(node):
            """Traverse the tree to collect leaf nodes."""
            if NodeValidator.is_leaf_node(node):
                leaf_nodes[node['name']] = node
            elif 'children' in node:
                for child in node['children'].values():
                    traverse(child)
        
        traverse(self.root)
        return leaf_nodes
    
    def get_path_to_leaf(self, leaf_node: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get the fuzzy path from root to a specific leaf node.
        
        This method constructs the complete fuzzy path by tracing from
        root to leaf, collecting all feature-fuzzy_set pairs along the way.
        
        Parameters
        ----------
        leaf_node : dict
            Leaf node to trace path to.
            
        Returns
        -------
        list
            List of path steps, each containing feature and fuzzy_set indices.
        """
        path = []
        
        def trace_path(node, target_name, current_path):
            """Trace the path from the root to a target node name."""
            if node['name'] == target_name:
                path.extend(current_path)
                return True
            
            if 'children' in node:
                for child_name, child_node in node['children'].items():
                    step = {
                        'feature': child_node['feature'],
                        'fuzzy_set': child_node['fuzzy_set'],
                        'node_name': child_name
                    }
                    if trace_path(child_node, target_name, current_path + [step]):
                        return True
            
            return False
        
        trace_path(self.root, leaf_node['name'], [])
        return path
    
    def get_node_samples_mask(self, node: Dict[str, Any], X: np.array) -> np.array:
        """
        Get boolean mask for samples that belong to a specific node.
        
        This method computes which samples have significant membership
        to a given node based on the fuzzy path from root.
        
        Parameters
        ----------
        node : dict
            Node to compute mask for.
        X : np.array
            Input data array.
            
        Returns
        -------
        np.array
            Boolean mask indicating which samples belong to the node.
        """
        n_samples = X.shape[0]
        membership = np.ones(n_samples)
        
        # If this is not the root, compute path membership
        if node['name'] != 'root':
            path = self.get_path_to_leaf(node)
            for step in path:
                feature_idx = step['feature']
                fuzzy_set_idx = step['fuzzy_set']
                fuzzy_set = self.fuzzy_partitions[feature_idx][fuzzy_set_idx]
                feature_membership = fuzzy_set.membership(X[:, feature_idx])
                membership *= feature_membership
        
        # Return mask for samples with significant membership
        return membership > 1e-6
    
    def calculate_membership_to_node(self, X: np.array, node_name: str) -> np.array:
        """
        Calculate fuzzy membership of samples to a specific node.
        
        Parameters
        ----------
        X : np.array
            Input data array.
        node_name : str
            Name of the node to calculate membership for.
            
        Returns
        -------
        np.array
            Membership values for each sample to the node.
        """
        node = self.find_node_by_name(node_name)
        if node is None:
            return np.zeros(X.shape[0])
        
        n_samples = X.shape[0]
        membership = np.ones(n_samples)
        
        # If this is not the root, compute path membership
        if node_name != 'root':
            path = self.get_path_to_leaf(node)
            for step in path:
                feature_idx = step['feature']
                fuzzy_set_idx = step['fuzzy_set']
                fuzzy_set = self.fuzzy_partitions[feature_idx][fuzzy_set_idx]
                feature_membership = fuzzy_set.membership(X[:, feature_idx])
                membership *= feature_membership
        
        return membership
    
    def get_tree_depth(self) -> int:
        """
        Calculate the maximum depth of the tree.
        
        Returns
        -------
        int
            Maximum depth of the tree (root is depth 0).
        """
        def calculate_depth(node, current_depth=0):
            """Recursively compute maximum depth for the subtree."""
            max_depth = current_depth
            if 'children' in node:
                for child in node['children'].values():
                    child_depth = calculate_depth(child, current_depth + 1)
                    max_depth = max(max_depth, child_depth)
            return max_depth
        
        return calculate_depth(self.root)
    
    def count_nodes(self) -> int:
        """
        Count total number of nodes in the tree.
        
        Returns
        -------
        int
            Total number of nodes.
        """
        return len(self.node_dict_access)
    
    def count_leaves(self) -> int:
        """
        Count number of leaf nodes in the tree.
        
        Returns
        -------
        int
            Number of leaf nodes.
        """
        def count_recursive(node):
            """Recursively count leaf nodes in a subtree."""
            if NodeValidator.is_leaf_node(node):
                return 1
            else:
                count = 0
                if 'children' in node:
                    for child in node['children'].values():
                        count += count_recursive(child)
                return count
        
        return count_recursive(self.root)
