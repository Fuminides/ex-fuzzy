'''
Fuzzy CART-like learning algorithm for classification problems.

Main components:


'''
from xxlimited import new
import numpy as np

from sklearn.base import ClassifierMixin
try:
    from . import fuzzy_sets as fs
    from . import utils
except ImportError:
    import fuzzy_sets as fs
    import utils


def _calculate_coverage(truth_values: np.array, total_samples: int) -> float:
    """Calculate the proportion of samples covered by the given truth values."""
    return np.sum(truth_values) / total_samples


def _weighted_gini_index(truth_values: np.array, y: np.array) -> float:
    """
    Compute the weighted Gini index for multiclass classification using fuzzy membership values.
    
    Args:
        truth_values (np.array): The membership degrees/weights for each sample (0-1).
        y (np.array): The class labels for all samples.
    
    Returns:
        float: The weighted Gini index (0 = pure, higher = more impure).
    """
    if len(truth_values) == 0 or np.sum(truth_values) == 0:
        return float('inf')
    
    unique_classes = np.unique(y)
    total_weight = np.sum(truth_values)
    
    # Calculate weighted class proportions
    weighted_proportions = []
    for cls in unique_classes:
        cls_mask = (y == cls)
        cls_weight = np.sum(truth_values[cls_mask])
        proportion = cls_weight / total_weight if total_weight > 0 else 0.0
        weighted_proportions.append(proportion)
    
    # Compute weighted gini index (multiclass)
    weighted_proportions = np.array(weighted_proportions)
    weighted_gini = 1.0 - np.sum(weighted_proportions ** 2)
    
    return weighted_gini


def _gini_index(y: np.array) -> float:
    """
    Compute the gini index of a set of labels.

    Args:
        y (np.array): The labels.

    Returns:
        float: The gini index.
    """
    classes, counts = np.unique(y, return_counts=True)
    total = np.sum(counts)
    if total == 0:
        return 0.0
    gini = 1.0 - np.sum((counts / total) ** 2)
    return gini


def _complete_classification_index(y: np.array, pre_yhat: np.array, new_yhat: np.array) -> float:
    """
    Compute the complete classification index (CCI) to evaluate the improvement
    in classification accuracy when moving from pre_yhat to new_yhat.
    It uses MCC as a metric of the metaclassification improvement.

    Args:
        y (np.array): The true class labels.
        pre_yhat (np.array): The previous predicted class labels.
        new_yhat (np.array): The new predicted class labels.

    Returns:
        float: The CCI value, where higher values indicate better improvement.
    """
    if len(y) == 0:
        return 0.0

    correct_pre = (y == pre_yhat)
    correct_new = (y == new_yhat)

    TP = np.sum(correct_pre & correct_new)  # True Positives: Correctly classified in both
    TN = np.sum(~correct_pre & ~correct_new)  # True Negatives: Incorrectly classified in both
    FP = np.sum(~correct_pre & correct_new)  # False Positives: Improved classification
    FN = np.sum(correct_pre & ~correct_new)  # False Negatives: Worsened classification

    numerator = (TP * TN) - (FP * FN)
    denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    if denominator == 0:
        return 0.0

    cci = numerator / denominator

    return cci


def compute_purity(thresholded_truth_values: np.array, y: np.array) -> float:
    """
        Compute the purity of a split in the dataset with the gini index (crisp case).

        Args:
            thresholded_truth_values (np.array): Boolean array indicating which samples 
                                            are included in this split.
            y (np.array): The class labels for all samples.

        Returns:
            float: The gini impurity score (lower is better, 0 = pure).
    """
    # Filter labels to only include samples in this split
    y_split = y[thresholded_truth_values]
    
    # If no samples in split, return 0 (pure by definition)
    if len(y_split) == 0:
        return 0.0
    
    # Calculate gini index for the filtered labels
    return _gini_index(y_split)


def compute_fuzzy_purity(truth_values: np.array, y: np.array, minimum_coverage_threshold: float = 0.0) -> float:
    """
    Compute the fuzzy purity of a split using weighted gini index for multiclass.

    Args:
        truth_values (np.array): The membership degrees/weights for each sample (0-1).
        y (np.array): The class labels for all samples.

    Returns:
        float: The weighted gini impurity score (lower is better, 0 = pure).
    """
    if len(truth_values) == 0 or np.sum(truth_values) == 0:
        return float('inf')
    
    # Use the extracted weighted Gini function
    weighted_gini = _weighted_gini_index(truth_values, y)
    coverage = _calculate_coverage(truth_values, len(y))

    if coverage < minimum_coverage_threshold: # Minimum coverage threshold
        return float('inf')
    else:
        return weighted_gini


def compute_fuzzy_cci(y: np.array, truth_values: np.array, pre_yhat: np.array, new_yhat: np.array, minimum_coverage_threshold: float = 0.0) -> float:
    """
    Compute the complete classification index (CCI) to evaluate the improvement
    """
    if len(truth_values) == 0 or np.sum(truth_values) == 0:
        return float(0.0)

    cci_index =  _complete_classification_index(y, pre_yhat, new_yhat)
    coverage = _calculate_coverage(truth_values, len(y))

    if coverage < minimum_coverage_threshold: # Minimum coverage threshold
        return float('inf')
    else:
        return cci_index


class FuzzyCART(ClassifierMixin):

    def __init__(self, fuzzy_partitions: list[fs.fuzzyVariable], max_rules: int = 10, max_depth: int = 5, coverage_threshold: float = 0.01):
        self.fuzzy_partitions = fuzzy_partitions
        self.max_depth = max_depth
        self.tree = None
        self.tree_rules = 1 # Start with 1 (so that the first split creates the first rule)
        self.max_rules = max_rules
        self.coverage_threshold = coverage_threshold


    def fit(self, X: np.array, y: np.array):
        self.classes_ = np.unique(y)
        self._build_tree(X, y)


    def _build_root(self, X: np.array, y: np.array):
        existing_membership = np.ones(X.shape[0])
        actual_path = np.ones([len(self.fuzzy_partitions), len(self.fuzzy_partitions[0])], dtype=bool)

        self._root  = {
            'depth': 0,
            'existing_membership': existing_membership,
            'actual_path': actual_path,
            'name': 'root',
            'prediction': self._majority_class(y, existing_membership),
            'coverage': 1.0
        }

        self.node_dict_access = {'root': self._root}


    def _node_purity_checks(self, node, X: np.array, y: np.array) -> float:
        existing_membership = node['existing_membership']
        actual_path = node['actual_path']

        features, fuzzy_ll = actual_path.shape
        best_purity_improvement = float('-inf')
        best_feature = -1
        best_fuzzy_set = -1
        best_coverage = 0.0
        father_purity = compute_fuzzy_purity(existing_membership, y, self.coverage_threshold)
        # For debugging, save the purity of each possible split
        debug_cache = np.zeros((features, fuzzy_ll))
        coverage_cache = np.zeros((features, fuzzy_ll))

        for feature in range(features):
            fuzzy_var = self.fuzzy_partitions[feature]

            for fz_index in range(len(fuzzy_var)):
                if actual_path[feature, fz_index]:
                    fz = fuzzy_var[fz_index]
                    memberships = fz.membership(X[:, feature])
                    full_path_membership = memberships * existing_membership
                    
                    purity = compute_fuzzy_purity(full_path_membership, y, self.coverage_threshold)
                    debug_cache[feature, fz_index] = purity
                    coverage = _calculate_coverage(full_path_membership, len(y))
                    coverage_cache[feature, fz_index] = coverage
                    purity_improvement = father_purity - purity

                    if purity_improvement > best_purity_improvement:
                        best_purity_improvement = purity_improvement
                        best_feature = feature
                        best_fuzzy_set = fz_index
                        best_coverage = coverage

        node['aux_purity_cache'] = {
            'purity_improvement': best_purity_improvement,
            'feature': best_feature,
            'fuzzy_set': best_fuzzy_set,
            'coverage': best_coverage
        }

        return best_purity_improvement


    def _node_cci_checks(self, node, X: np.array, y: np.array, skeleton_memberships, skeleton_yhat, skeleton_predictors) -> float:
        existing_membership = node['existing_membership']
        actual_path = node['actual_path']

        features, fuzzy_ll = actual_path.shape
        best_cci = float('-inf')
        best_feature = -1
        best_fuzzy_set = -1
        best_coverage = 0.0
        my_predictions = skeleton_predictors == node['name']
        # For debugging, save the cci of each possible split
        debug_cache = np.zeros((features, fuzzy_ll))
        coverage_cache = np.zeros((features, fuzzy_ll))

        for feature in range(features):
            fuzzy_var = self.fuzzy_partitions[feature]
            for fz_index in range(len(fuzzy_var)):
                if actual_path[feature, fz_index]:
                    fz = fuzzy_var[fz_index]
                    memberships = fz.membership(X[:, feature])
                    full_path_membership = memberships * existing_membership

                    relevant_cases = skeleton_memberships[my_predictions] < full_path_membership[my_predictions]
                    modified_predictions = skeleton_yhat.copy()
                    node_prediction = self._majority_class(y, full_path_membership)
                    modified_predictions[my_predictions] = np.where(relevant_cases, node_prediction, skeleton_yhat[my_predictions])

                    cci = compute_fuzzy_cci(full_path_membership, y, self.coverage_threshold)
                    debug_cache[feature, fz_index] = cci
                    coverage = _calculate_coverage(full_path_membership, len(y))
                    coverage_cache[feature, fz_index] = coverage

                    if cci > best_cci:
                        best_cci = cci
                        best_feature = feature
                        best_fuzzy_set = fz_index
                        best_coverage = coverage

        node['aux_cci_cache'] = {
            'cci': best_cci,
            'feature': best_feature,
            'fuzzy_set': best_fuzzy_set,
            'coverage': best_coverage
        }

        return best_cci


    def _get_best_node_split(self, node_father, X: np.array, y: np.array) -> tuple[float, str]:
        best_purity_improvement = self._node_purity_checks(node_father, X, y)
        best_node = node_father['name']

        if 'children' in node_father:
            for child_name, child in node_father['children'].items():
                child_purity_improvement, _split_name = self._get_best_node_split(child, X, y)

                if child_purity_improvement > best_purity_improvement:
                    best_purity_improvement = child_purity_improvement
                    best_node = child_name

        return best_purity_improvement, best_node
    

    def _get_best_node_split_cci(self, node_father, X: np.array, y: np.array, skeleton_yhat, skeleton_predictors) -> tuple[float, str]:
        best_cci = self._node_cci_checks(node_father, X, y, skeleton_yhat, skeleton_predictors)
        best_node = node_father['name']

        if 'children' in node_father:
            for child_name, child in node_father['children'].items():
                child_cci, _split_name = self._get_best_node_split_cci(child, X, y, skeleton_yhat, skeleton_predictors)

                if child_cci > best_cci:
                    best_child_cci = child_cci
                    best_node = child_name

        return best_child_cci, best_node


    def _find_node_by_name(self, name: str):
        return self.node_dict_access[name]
    

    def _split_node(self, node, X: np.array, y: np.array):
        cache = node['aux_purity_cache']
        best_purity_improvement = cache['purity_improvement']
        best_feature = cache['feature']
        best_fuzzy_set = cache['fuzzy_set']
        best_coverage = cache['coverage']

        # Update existing membership and actual path
        existing_membership = node['existing_membership']
        child_existing_membership = existing_membership * self.fuzzy_partitions[best_feature][best_fuzzy_set].membership(X[:, best_feature])
        node['actual_path'][best_feature, best_fuzzy_set] = False
        child_actual_path = node['actual_path'].copy()
        child_prediction = self._majority_class(y, child_existing_membership)


        # Create children dictionary if not exists
        if 'children' not in node:
            node['children'] = {}
        else:
            self.tree_rules += 1

        # Create children node
        new_node = {
            'depth': node['depth'] + 1,
            'existing_membership': child_existing_membership,
            'actual_path': child_actual_path,           
            'name': node['name'] + f"_F{best_feature}_L{best_fuzzy_set}",
            'prediction': child_prediction,
            'feature': best_feature,
            'fuzzy_set': best_fuzzy_set,
            'coverage': np.sum(child_existing_membership) / len(y)         
        }

        # Raise an error if the node name already exists
        if new_node['name'] in self.node_dict_access or new_node['name'] in node['children'].keys():
            raise ValueError(f"Node name {new_node['name']} already exists in the tree.")
        else:
            node['children'][new_node['name']] = new_node
        self.node_dict_access[new_node['name']] = new_node


    def _build_tree(self, X: np.array, y: np.array):
        # Stopping criteria
    
        self._build_root(X, y)
        best_coverage_achieable = 1.0
        while self.tree_rules < self.max_rules and best_coverage_achieable >= self.coverage_threshold:
            skeleton_prediction, paths = self.predict_with_path(X)

            best_purity_improvement, best_node = self._get_best_node_split(self._root, X, y)
            
            # Split the best node
            node_to_split = self._find_node_by_name(best_node)
            
            # Make sure that the best gain is actually a feature not a finish signal (-1)
            if node_to_split['aux_purity_cache']['feature'] == -1:
                break
            else:
                self._split_node(node_to_split, X, y)
                best_coverage_achieable = self._get_best_possible_coverage()


    def _majority_class(self, y: np.array, membership: np.array = None):
        if len(y) == 0:
            return self.classes_[0]  # Return first class if no samples
        classes, counts = np.unique(y, return_counts=True)

        # Weight counts by membership
        counts = np.array([np.sum(membership[y == cls]) for cls in classes])
        if np.sum(counts) != 0.0:
            counts = counts / np.sum(counts)  # Normalize to get probabilities

        return classes[np.argmax(counts)]


    def predict(self, X: np.array) -> np.array:
        return np.array([self._predict_single(x[:], self._root)[0] for x in X])
    

    def predict_with_path(self, X: np.array) -> tuple[np.array, list[str]]:
        predictions = []
        paths = []
        for x in X:
            pred, _, path = self._predict_single(x[:], self._root)
            predictions.append(pred)
            paths.append(path)
        return np.array(predictions), paths


    def _predict_single(self, x: np.array, node, membership=None):
        if membership is None:
            if len(x.shape) > 1:
                membership = np.ones(x.shape[0])
            else:
                membership = 1.0

       # If there are no children, return prediction
        if not node.get('children', False) or len(node['children']) == 0:
            if node['name'] == 'root':
                return node['prediction'], membership * 0.0, node['name']
            else:
                return node['prediction'], membership, node['name']
        else:
            best_membership = -1
            for child_name, child in node['children'].items():
                relevant_feature = child['feature']
                relevant_fuzzy_set = child['fuzzy_set']
                child_path_membership = self.fuzzy_partitions[relevant_feature][relevant_fuzzy_set].membership(x[relevant_feature])

                child_pred, path_membership, predictor_name = self._predict_single(x, child, child_path_membership * membership)

                if path_membership > best_membership:
                    best_membership = path_membership
                    prediction = child_pred
                    best_predictor_name = predictor_name

            return prediction, best_membership, best_predictor_name


    def _get_best_possible_coverage(self):
        best_coverage = 0.0

        for node_name, node in self.node_dict_access.items():
            if 'aux_purity_cache' in node:
                cache = node['aux_purity_cache']
                coverage = cache['coverage']
                if coverage >= self.coverage_threshold:
                    best_coverage = max(best_coverage, coverage)
        
        return best_coverage

    def print_tree(self, node=None, prefix="", is_last=True):
        """
        Print the tree structure in a hierarchical format showing coverage information.
        
        Args:
            node: The node to start printing from (default: root)
            prefix: String prefix for indentation
            is_last: Whether this is the last child at this level
        """
        if node is None:
            node = self._root
            print(f"Fuzzy CART Tree (max_rules={self.max_rules}, coverage_threshold={self.coverage_threshold})")
            print("=" * 60)
        
        # Print current node
        current_prefix = "└── " if is_last else "├── "
        
        if node['name'] == 'root':
            print(f"{prefix}{current_prefix}Root: class={node['prediction']}, coverage={node['coverage']:.3f}")
        else:
            feature_name = f"feature_{node['feature']}"
            fuzzy_set_name = f"fuzzy_set_{node['fuzzy_set']}"
            print(f"{prefix}{current_prefix}{node['name']}: {feature_name}[{fuzzy_set_name}] → class={node['prediction']}, coverage={node['coverage']:.3f}")
        
        # Print children
        if 'children' in node and node['children']:
            # Determine the prefix for children
            child_prefix = prefix + ("    " if is_last else "│   ")
            
            children_list = list(node['children'].items())
            for i, (child_name, child_node) in enumerate(children_list):
                is_last_child = (i == len(children_list) - 1)
                self.print_tree(child_node, child_prefix, is_last_child)
    

    def get_tree_stats(self):
        """
        Get statistics about the tree structure.
        
        Returns:
            dict: Dictionary containing tree statistics
        """
        def traverse_tree(node):
            stats = {'total_nodes': 1, 'leaves': 0, 'internal': 0, 'depth': node['depth']}
            
            if 'children' not in node or not node['children']:
                stats['leaves'] = 1
            else:
                stats['internal'] = 1
                for child in node['children'].values():
                    child_stats = traverse_tree(child)
                    stats['total_nodes'] += child_stats['total_nodes']
                    stats['leaves'] += child_stats['leaves']
                    stats['internal'] += child_stats['internal']
                    stats['depth'] = max(stats['depth'], child_stats['depth'])
            
            return stats
        
        return traverse_tree(self._root)

if __name__ == "__main__":
    # Load iris dataset
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    print("Testing with Iris dataset:")
    iris = datasets.load_wine()
    X = iris.data
    y = iris.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create fuzzy partitions
    fuzzy_partitions = utils.construct_partitions(X_train, fs.FUZZY_SETS.t1, n_partitions=3, shape='trapezoid')
    
    # Train classifier
    classifier = FuzzyCART(fuzzy_partitions, max_depth=3, max_rules=20)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Iris Accuracy: {accuracy:.3f}")
    
    print("\nTree Structure:")
    classifier.print_tree()
    
    print("\nTree Statistics:")
    stats = classifier.get_tree_stats()
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"Leaf nodes: {stats['leaves']}")
    print(f"Internal nodes: {stats['internal']}")
    print(f"Tree depth: {stats['depth']}")