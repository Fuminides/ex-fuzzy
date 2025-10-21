'''
Fuzzy CART-like learning algorithm for classification problems.

Main components:


'''
import numpy as np

from sklearn.base import ClassifierMixin
try:
    from . import fuzzy_sets as fs
    from . import utils
except ImportError:
    import fuzzy_sets as fs
    import utils


def _calculate_coverage(truth_values: np.array, total_samples: int) -> float:
    """
    Calculate the proportion of samples covered by the given truth values.
    
    This function determines what fraction of the total dataset is covered
    by the current fuzzy membership values, providing insight into how
    much of the data space the current rule or node affects.
    
    Parameters
    ----------
    truth_values : np.array
        Array of membership degrees/weights for each sample (values between 0-1).
        Higher values indicate stronger membership in the current fuzzy set.
    total_samples : int
        Total number of samples in the dataset used for normalization.
    
    Returns
    -------
    float
        Coverage ratio between 0 and 1, where 1 means all samples are fully
        covered and 0 means no samples are covered.
    """
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
    Compute the Gini impurity index of a set of class labels.
    
    The Gini index measures the impurity or disorder in a classification
    dataset. It ranges from 0 (pure, all samples belong to one class) 
    to approximately 0.5 (maximum impurity for binary classification).
    Lower values indicate more homogeneous class distributions.
    
    Parameters
    ----------
    y : np.array
        Array of class labels for the samples.
    
    Returns
    -------
    float
        Gini impurity index. 0 indicates perfect purity (all samples same class),
        higher values indicate more mixed class distributions.
    """
    classes, counts = np.unique(y, return_counts=True)
    total = np.sum(counts)
    if total == 0:
        return 0.0
    gini = 1.0 - np.sum((counts / total) ** 2)
    return gini


def _complete_classification_index(y: np.array, pre_yhat: np.array, new_yhat: np.array) -> float:
    """
    Compute the Complete Classification Index (CCI) to evaluate classification improvement.
    
    The CCI measures how much a new prediction strategy improves over a previous one
    by analyzing the change in classification accuracy. It provides a metric for
    evaluating whether a tree split or rule addition actually improves the overall
    classification performance. The metric focuses on the improvement in accuracy
    rather than absolute accuracy values.
    
    Parameters
    ----------
    y : np.array
        Array of true class labels for all samples.
    pre_yhat : np.array
        Array of predicted class labels from the previous/baseline classifier.
    new_yhat : np.array
        Array of predicted class labels from the new/improved classifier.
    
    Returns
    -------
    float
        CCI value representing the improvement in classification accuracy.
        Positive values indicate improvement, negative values indicate degradation,
        and zero indicates no change in performance.
    """
    if len(y) == 0:
        return 0.0

    correct_pre = (y == pre_yhat)
    correct_new = (y == new_yhat)

    TP = np.mean(correct_pre & correct_new)  # True Positives: Correctly classified in both
    TN = np.mean(~correct_pre & ~correct_new)  # True Negatives: Incorrectly classified in both
    FP = np.mean(~correct_pre & correct_new)  # False Positives: Improved classification
    FN = np.mean(correct_pre & ~correct_new)  # False Negatives: Worsened classification

    improvement = np.mean(correct_new) - np.mean(correct_pre)

    if np.mean(correct_pre) == 0.0:
        improvement_percentage = np.mean(correct_new)
    else:
        improvement_percentage = improvement / np.mean(correct_pre)

    return improvement_percentage


def compute_purity(thresholded_truth_values: np.array, y: np.array) -> float:
    """
    Compute the purity of a dataset split using the Gini index (crisp/hard split case).
    
    This function evaluates how pure (homogeneous) a subset of data is after
    applying a crisp (binary) split criterion. It uses the traditional Gini
    impurity measure for discrete splits where samples either belong to
    the split or they don't (no fuzzy membership).
    
    Parameters
    ----------
    thresholded_truth_values : np.array
        Boolean array indicating which samples are included in this split.
        True means the sample belongs to this partition, False means it doesn't.
    y : np.array
        Array of class labels for all samples in the original dataset.
    
    Returns
    -------
    float
        Gini impurity score for the split subset. Lower values (closer to 0) 
        indicate higher purity (more homogeneous class distribution).
        0 means perfect purity (all samples in the split have the same class).
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
    Compute the fuzzy purity of a dataset split using weighted Gini index for multiclass problems.
    
    This function extends the traditional Gini impurity to handle fuzzy (soft) splits
    where samples can have partial membership in different partitions. It weights
    the class distribution by the fuzzy membership values, allowing for more
    nuanced evaluation of split quality in fuzzy decision trees.
    
    Parameters
    ----------
    truth_values : np.array
        Array of fuzzy membership degrees/weights for each sample (values 0-1).
        Higher values indicate stronger membership in the current fuzzy partition.
    y : np.array
        Array of class labels for all samples in the dataset.
    minimum_coverage_threshold : float, default=0.0
        Minimum coverage ratio required for the split to be considered valid.
        Splits with coverage below this threshold return infinite impurity.
    
    Returns
    -------
    float
        Weighted Gini impurity score for the fuzzy split. Lower values indicate
        higher purity. Returns float('inf') if coverage is below threshold or
        if no samples have positive membership.
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
    Compute the fuzzy Complete Classification Index (CCI) for evaluating split improvement.
    
    This function extends the CCI to handle fuzzy partitions by incorporating
    coverage requirements. It evaluates whether a fuzzy split provides sufficient
    improvement in classification accuracy while meeting minimum coverage constraints.
    This is crucial for fuzzy decision trees where splits with very low coverage
    might overfit to small subsets of data.
    
    Parameters
    ----------
    y : np.array
        Array of true class labels for all samples.
    truth_values : np.array
        Array of fuzzy membership degrees for samples in the current partition.
    pre_yhat : np.array
        Predicted class labels from the baseline/previous classifier.
    new_yhat : np.array
        Predicted class labels from the improved/new classifier.
    minimum_coverage_threshold : float, default=0.0
        Minimum coverage ratio required for the split to be considered valid.
    
    Returns
    -------
    float
        Fuzzy CCI value. Returns the improvement score if coverage meets threshold,
        -1.0 if coverage is insufficient, or 0.0 if no samples have membership.
    """
    if len(truth_values) == 0 or np.sum(truth_values) == 0:
        return float(0.0)

    cci_index =  _complete_classification_index(y, pre_yhat, new_yhat)
    coverage = _calculate_coverage(truth_values, len(y))

    if coverage < minimum_coverage_threshold: # Minimum coverage threshold
        return float(-1.0)
    else:
        return cci_index


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

    def __init__(self, fuzzy_partitions: list[fs.fuzzyVariable], max_rules: int = 10, max_depth: int = 5, coverage_threshold: float = 0.00, min_improvement=0.01, ccp_alpha: float = 0.0, target_metric: str = 'cci'):
        """
        Initialize the Fuzzy CART classifier.
        """
        self.fuzzy_partitions = fuzzy_partitions
        self.max_depth = max_depth
        self.tree = None
        self.tree_rules = 1 # Start with 1 (so that the first split creates the first rule)
        self.max_rules = max_rules
        self.coverage_threshold = coverage_threshold
        self.min_improvement = min_improvement
        self.ccp_alpha = ccp_alpha
        self.target_metric = target_metric


    def fit(self, X: np.array, y: np.array, patience:int = 3):
        """
        Train the Fuzzy CART classifier on the provided dataset.
        
        This method builds the fuzzy decision tree by identifying the unique
        classes in the target variable and then constructing the tree structure
        using the fuzzy partitions and splitting criteria.
        
        Parameters
        ----------
        X : np.array
            Training data features with shape (n_samples, n_features).
            Each row represents a sample and each column a feature.
        y : np.array
            Target class labels with shape (n_samples,).
            Contains the class labels for each training sample.
        """
        self.classes_ = np.unique(y)
        self._build_tree(X, y, bad_cuts_limit=patience,index=self.target_metric)


    def _build_root(self, X: np.array, y: np.array):
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
        actual_path = np.ones([len(self.fuzzy_partitions), len(self.fuzzy_partitions[0])], dtype=bool)

        self._root  = {
            'depth': 0,
            'existing_membership': existing_membership,
            'father_path': actual_path,
            'child_splits': actual_path.copy(),
            'name': 'root',
            'prediction': -1, # No prediction at root
            'coverage': 1.0,
            'class_probabilities': self._class_probabilities(y, existing_membership)
        }

        self.node_dict_access = {'root': self._root}


    def _node_purity_checks(self, node, X: np.array, y: np.array) -> float:
        """
        Evaluate all possible fuzzy splits for a given node using purity improvement.
        
        This method examines every available fuzzy set in each feature dimension
        to find the split that provides the maximum improvement in node purity
        (reduction in weighted Gini impurity). It considers the node's current
        membership and path constraints to ensure valid splits.
        
        Parameters
        ----------
        node : dict
            Tree node dictionary containing membership, path, and other node information.
        X : np.array
            Training data features for evaluating splits.
        y : np.array
            Training data labels for computing purity measures.
        
        Returns
        -------
        float
            Maximum purity improvement achievable from this node.
            Higher values indicate better potential splits.
        """
        existing_membership = node['existing_membership']
        father_path = node['father_path']
        child_splits = node['child_splits']
        actual_path = np.logical_and(father_path, child_splits)


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
                    node_prediction = self._split_node_dummy(node, feature, fz_index, X, y)

                    if purity_improvement > best_purity_improvement:
                        best_purity_improvement = purity_improvement
                        best_feature = feature
                        best_fuzzy_set = fz_index
                        best_coverage = coverage
                        child_decision = node_prediction

                    self._delete_node_dummy(node, feature, fz_index)

   
        node['aux_purity_cache'] = {
            'feature': best_feature,
            'fuzzy_set': best_fuzzy_set,
            'coverage': best_coverage,
            'split_criterion': best_purity_improvement,
            'child_decision': child_decision,
            'purity': best_purity_improvement
        }

        return best_purity_improvement


    def _get_best_node_split(self, node_father, X: np.array, y: np.array) -> tuple[float, str]:
        """
        Recursively find the best node in the tree for splitting based on purity improvement.
        
        This method traverses the entire tree to identify which node would benefit
        most from being split. It compares the purity improvement potential of
        the current node with all its descendants to find the globally optimal
        split location.
        
        Parameters
        ----------
        node_father : dict
            Root node to start the search from (typically the tree root).
        X : np.array
            Training data features for evaluating split quality.
        y : np.array
            Training data labels for computing purity improvements.
        
        Returns
        -------
        tuple[float, str]
            Tuple containing the best purity improvement value and the name
            of the node that should be split to achieve this improvement.
        """
        best_purity_improvement = self._node_purity_checks(node_father, X, y)
        best_node = node_father['name']

        if 'children' in node_father:
            for child_name, child in node_father['children'].items():
                child_purity_improvement, _split_name = self._get_best_node_split(child, X, y)

                if child_purity_improvement > best_purity_improvement:
                    best_purity_improvement = child_purity_improvement
                    best_node = child_name

        return best_purity_improvement, best_node


    def _node_cci_checks(self, node, X: np.array, y: np.array) -> float:
        """
        Evaluate all possible fuzzy splits for a node using Complete Classification Index (CCI).
        
        This method is the core splitting criterion evaluator that examines every
        available fuzzy partition to find the split that maximizes classification
        improvement. Unlike purity-based methods, CCI focuses on actual classification
        accuracy improvement, making it more directly relevant to predictive performance.
        
        The method creates temporary child nodes to evaluate how each potential split
        would affect the overall tree's classification accuracy, using the skeleton
        prediction as a baseline for comparison.
        
        Parameters
        ----------
        node : dict
            Tree node to evaluate for potential splits.
        X : np.array
            Training data features for split evaluation.
        y : np.array
            Training data labels for computing CCI values.
        
        Returns
        -------
        tuple[float, float]
            Best CCI improvement value and corresponding purity for the optimal split.
        """
        existing_membership = node['existing_membership']
        child_decision = node['prediction']
        features, fuzzy_ll = node['father_path'].shape

        if self.tree_rules <= 3:
            best_cci = float('-inf')
        else:
            best_cci = float(0.0)
        best_purity = float('inf')
        best_feature = -1
        best_fuzzy_set = -1
        best_coverage = 0.0

        # For debugging, save the cci of each possible split
        debug_cache_purity = np.zeros((features, fuzzy_ll))
        debug_cache_cci = np.zeros((features, fuzzy_ll))
        coverage_cache = np.zeros((features, fuzzy_ll))

        legal_paths = np.logical_and(node['father_path'], node['child_splits'])
        skeleton_yhat = self.predict(X)

        for feature in range(features):
            fuzzy_var = self.fuzzy_partitions[feature]
            for fz_index in range(len(fuzzy_var)):
                if legal_paths[feature, fz_index]:
                    fz = fuzzy_var[fz_index]
                    memberships = fz.membership(X[:, feature])
                    full_path_membership = memberships * existing_membership

                    node_prediction = self._split_node_dummy(node, feature, fz_index, X, y)
                    skeleton_yhat_child = self.predict(X)
                    
                    cci = compute_fuzzy_cci(y, full_path_membership, skeleton_yhat, skeleton_yhat_child, self.coverage_threshold)
                    purity = compute_fuzzy_purity(full_path_membership, y, self.coverage_threshold)

                    debug_cache_cci[feature, fz_index] = cci
                    debug_cache_purity[feature, fz_index] = purity

                    coverage = _calculate_coverage(full_path_membership, len(y))
                    coverage_cache[feature, fz_index] = coverage

                    if cci > best_cci:
                        best_cci = cci
                        best_feature = feature
                        best_fuzzy_set = fz_index
                        best_coverage = coverage
                        child_decision = node_prediction
                        best_purity = purity
                    elif cci == best_cci and purity < best_purity:
                        best_cci = cci
                        best_feature = feature
                        best_fuzzy_set = fz_index
                        best_coverage = coverage
                        child_decision = node_prediction
                        best_purity = purity

                    self._delete_node_dummy(node, feature, fz_index)


        node['aux_purity_cache'] = {
            'cci': best_cci,
            'feature': best_feature,
            'fuzzy_set': best_fuzzy_set,
            'coverage': best_coverage,
            'split_criterion': best_cci,
            'child_decision': child_decision,
            'purity': best_purity
        }

        return best_cci, best_purity
    

    def _get_best_node_split_cci(self, node_father, X: np.array, y: np.array) -> tuple[float, str]:
        """
        Recursively find the best node for splitting using CCI-based evaluation.
        
        This method traverses the tree to find the node that would provide the
        maximum improvement in classification accuracy when split. It uses the
        Complete Classification Index as the primary criterion, with purity
        as a tiebreaker when CCI values are equal.
        
        Parameters
        ----------
        node_father : dict
            Root node to start the recursive search from.
        X : np.array
            Training data features for evaluating split quality.
        y : np.array
            Training data labels for computing CCI improvements.
        
        Returns
        -------
        tuple[float, str, float]
            Best CCI value, name of the node to split, and corresponding purity.
        """
        best_cci, best_purity = self._node_cci_checks(node_father, X, y)
        best_node = node_father['name']

        if 'children' in node_father:
            for child_name, child in node_father['children'].items():
                child_cci, _child_name_bis, child_purity = self._get_best_node_split_cci(child, X, y)

                if child_cci > best_cci:
                    best_cci = child_cci
                    best_node = _child_name_bis
                elif child_cci == best_cci and child_purity < best_purity:
                    best_cci = child_cci
                    best_node = _child_name_bis
                    best_purity = child_purity

        return best_cci, best_node, best_purity


    def _find_node_by_name(self, name: str):
        """
        Retrieve a tree node by its unique name identifier.
        
        This is a simple lookup method that provides fast access to any node
        in the tree using the node dictionary. Essential for tree navigation
        and node manipulation operations.
        
        Parameters
        ----------
        name : str
            Unique identifier of the node to retrieve.
        
        Returns
        -------
        dict
            Node dictionary containing all node information and structure.
        """
        return self.node_dict_access[name]
    

    def _split_node(self, node, X: np.array, y: np.array):
        """
        Perform the actual split of a tree node based on its cached split information.
        
        This method executes the split that was determined to be optimal by the
        CCI evaluation methods. It creates a new child node with updated membership,
        path constraints, and prediction, then integrates it into the tree structure.
        
        The split updates both the node's path constraints (to prevent reusing
        the same fuzzy set) and creates the child with appropriate membership
        based on the selected fuzzy set.
        
        Parameters
        ----------
        node : dict
            Parent node to split, must have 'aux_purity_cache' with split information.
        X : np.array
            Training data features for computing child membership.
        y : np.array
            Training data labels for child node prediction.
        """

        # Debug message
        # print(f"Splitting node {node['name']} at depth {node['depth']} using feature {node['aux_purity_cache']['feature']} and fuzzy set {node['aux_purity_cache']['fuzzy_set']} with improvement {node['aux_purity_cache']['split_criterion']}")
        cache = node['aux_purity_cache']
        best_purity_improvement = cache['split_criterion']
        best_feature = cache['feature']
        best_fuzzy_set = cache['fuzzy_set']
        best_coverage = cache['coverage']

        # Update existing membership and actual path
        existing_membership = node['existing_membership']
        child_existing_membership = existing_membership * self.fuzzy_partitions[best_feature][best_fuzzy_set].membership(X[:, best_feature])
        node['child_splits'][best_feature, best_fuzzy_set] = False
        child_actual_path = node['father_path'].copy()
        child_actual_path[best_feature, best_fuzzy_set] = False
        child_prediction = cache['child_decision']
        child_new_child_splits = np.ones_like(node['child_splits'], dtype=bool)


        # Create children dictionary if not exists
        if 'children' not in node:
            node['children'] = {}
        else:
            self.tree_rules += 1

        # Create children node
        new_node = {
            'depth': node['depth'] + 1,
            'existing_membership': child_existing_membership,
            'father_path': child_actual_path,
            'child_splits': child_new_child_splits,
            'name': node['name'] + f"_F{best_feature}_L{best_fuzzy_set}",
            'prediction': child_prediction,
            'feature': best_feature,
            'fuzzy_set': best_fuzzy_set,
            'coverage': np.sum(child_existing_membership) / len(y),
            'quality_improvement': best_purity_improvement,
            'class_probabilities': self._class_probabilities(y, child_existing_membership)
        }

        
        # Raise an error if the node name already exists
        if new_node['name'] in self.node_dict_access or new_node['name'] in node['children'].keys():
            raise ValueError(f"Node name {new_node['name']} already exists in the tree.")
        else:
            node['children'][new_node['name']] = new_node

        self.node_dict_access[new_node['name']] = new_node


    def _delete_node_dummy(self, node, feature, fuzzy_set):
        """
        Remove a temporary dummy node created during CCI evaluation.
        
        During CCI computation, temporary child nodes are created to evaluate
        the impact of potential splits. This method cleans up these temporary
        nodes after evaluation to prevent them from polluting the tree structure.
        
        Parameters
        ----------
        node : dict
            Parent node that contains the dummy child to be removed.
        feature : int
            Feature index used in the dummy node name.
        fuzzy_set : int
            Fuzzy set index used in the dummy node name.
        """
        if 'children' in node:
            node_name = node['name'] + f"_F{feature}_L{fuzzy_set}_dummy"
            if node_name in node['children']:
                del node['children'][node_name]
            if node_name in self.node_dict_access:
                del self.node_dict_access[node_name]


    def _split_node_dummy(self, node, feature, fuzzy_set, X: np.array, y: np.array):
        """
        Create a temporary dummy child node for CCI evaluation purposes.
        
        This method creates a temporary child node to evaluate how a potential
        split would affect the tree's prediction performance. The dummy node
        is used only for computing CCI values and is removed after evaluation.
        This allows the algorithm to assess split quality without permanently
        modifying the tree structure.
        
        Parameters
        ----------
        node : dict
            Parent node to create the dummy child for.
        feature : int
            Feature index for the potential split.
        fuzzy_set : int
            Fuzzy set index for the potential split.
        X : np.array
            Training data features for computing membership.
        y : np.array
            Training data labels for determining child prediction.
        
        Returns
        -------
        int
            Predicted class for the dummy child node.
        """
        # Update existing membership and actual path
        existing_membership = node['existing_membership']
        child_existing_membership = existing_membership * self.fuzzy_partitions[feature][fuzzy_set].membership(X[:, feature])

        child_prediction = self._majority_class(y, child_existing_membership)


        # Create children dictionary if not exists
        if 'children' not in node:
            node['children'] = {}

        # Create children node
        new_node = {
            'depth': node['depth'] + 1,
            'existing_membership': child_existing_membership,
            'name': node['name'] + f"_F{feature}_L{fuzzy_set}_dummy",
            'prediction': child_prediction,
            'feature': feature,
            'fuzzy_set': fuzzy_set,
            'coverage': np.sum(child_existing_membership) / len(y),
            'class_probabilities': self._class_probabilities(y, child_existing_membership)
        }

        
        # Raise an error if the node name already exists
        if new_node['name'] in self.node_dict_access or new_node['name'] in node['children'].keys():
            raise ValueError(f"Node name {new_node['name']} already exists in the tree.")
        else:
            node['children'][new_node['name']] = new_node


        return child_prediction


    def _build_tree(self, X: np.array, y: np.array, bad_cuts_limit: int = 3, index: str='cci'):
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
        """
        # Stopping criteria
    
        self._build_root(X, y)
        best_coverage_achievable = 1.0
        bad_cuts = 0

        while self.tree_rules < self.max_rules and best_coverage_achievable >= self.coverage_threshold:
            skeleton_prediction, skeleton_memberships, paths = self.predict_with_path(X)
            # print('Accuracy:', np.mean(skeleton_prediction == y), 'Rules:', self.tree_rules, 'Best achievable coverage:', best_coverage_achievable)
            if index == 'purity':
                best_purity, best_node = self._get_best_node_split(self._root, X, y)
                _best_cci = None
                best_result = best_purity
            else:
                best_cci, best_node, _best_purity = self._get_best_node_split_cci(self._root, X, y)
                best_result = best_cci

            
            # Split the best node
            node_to_split = self._find_node_by_name(best_node)

            if best_result <= self.min_improvement:
                bad_cuts += 1
                if bad_cuts >= bad_cuts_limit:
                    # print("No more beneficial splits found after several attempts. Stopping.")
                    break
                
            # Make sure that the best gain is actually a feature not a finish signal (-1)
            if node_to_split['aux_purity_cache']['feature'] == -1:
                # print("No valid splits found. Stopping.")
                break
            else:
                self._split_node(node_to_split, X, y)
                best_coverage_achievable = self._get_best_possible_coverage()
            
        # Change the prediction in root node to majority class
        #          <self._majority_class(y, skeleton_memberships)>
        self._root['prediction'] = self._majority_class(y)
        # Update root probabilities after tree construction
        self._root['class_probabilities'] = self._class_probabilities(y)
        # print("Final tree built.")


    def _majority_class(self, y: np.array, membership: np.array = None):
        """
        Determine the majority class using weighted voting based on fuzzy membership.
        
        This method computes the predominant class in a dataset subset, optionally
        weighting each sample's contribution by its fuzzy membership value.
        This is essential for determining node predictions in fuzzy decision trees
        where samples may have partial membership in different nodes.
        
        Parameters
        ----------
        y : np.array
            Array of class labels for all samples.
        membership : np.array, optional
            Array of fuzzy membership weights for each sample. If None,
            uniform weights (crisp majority vote) are used.
        
        Returns
        -------
        int or class_type
            The majority class after weighting by membership values.
            Returns the first class if no samples are provided.
        """
        if membership is None:
            membership = np.ones(len(y))

        if len(y) == 0:
            return self.classes_[0]  # Return first class if no samples
        classes, counts = np.unique(y, return_counts=True)

        # Weight counts by membership
        counts = np.array([np.sum(membership[y == cls]) for cls in classes])
        if np.sum(counts) != 0.0:
            counts = counts / np.sum(counts)  # Normalize to get probabilities

        return classes[np.argmax(counts)]

    def _class_probabilities(self, y: np.array, membership: np.array = None):
        """
        Calculate class probabilities using weighted voting based on fuzzy membership.
        
        This method computes the probability distribution over all classes for a
        dataset subset, weighting each sample's contribution by its fuzzy membership
        value. This provides the probabilistic foundation for predict_proba.
        
        Parameters
        ----------
        y : np.array
            Array of class labels for all samples.
        membership : np.array, optional
            Array of fuzzy membership weights for each sample. If None,
            uniform weights are used.
        
        Returns
        -------
        np.array
            Probability vector with length equal to number of classes, where
            probabilities sum to 1.0. Each element represents the probability
            of the corresponding class in self.classes_.
        """
        if membership is None:
            membership = np.ones(len(y))

        # Initialize probability vector for all classes
        class_probs = np.zeros(len(self.classes_))
        
        if len(y) == 0:
            # If no samples, return uniform distribution
            class_probs.fill(1.0 / len(self.classes_))
            return class_probs
        
        # Calculate weighted counts for each class
        for i, cls in enumerate(self.classes_):
            class_mask = (y == cls)
            class_probs[i] = np.sum(membership[class_mask])
        
        # Normalize to get probabilities
        total_weight = np.sum(class_probs)
        if total_weight > 0:
            class_probs = class_probs / total_weight
        else:
            # If no membership weight, return uniform distribution
            class_probs.fill(1.0 / len(self.classes_))
        
        return class_probs


    def predict(self, X: np.array) -> np.array:
        """
        Predicts the class for given samples using batch processing.

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
        
        prediction, _, _ = self._predict(X, self._root)
        return prediction
    

    def predict_with_path(self, X: np.array) -> tuple[np.array, np.array, np.array]:
        """
        Predicts the class for given samples along with membership and path information using batch processing.

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
        
        predictions, memberships, paths = self._predict(X, self._root)
        return predictions, memberships, paths

    def predict_proba(self, X: np.array) -> np.array:
        """
        Predict class probabilities for given samples using fuzzy membership weighting.
        
        This method computes probability distributions over all classes for each sample
        by leveraging the fuzzy membership values from tree traversal. The probabilities
        are derived from the weighted voting mechanism used in the fuzzy decision tree,
        providing soft predictions that reflect the uncertainty in the classification.
        
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
        
        return self._predict_proba(X, self._root)


    def _predict_proba(self, x: np.array, node, membership=None, best_membership=None, 
                      class_probabilities=None) -> np.array:
        """
        Core recursive probability prediction method using batch processing with cached probabilities.
        
        This method traverses the fuzzy decision tree and uses pre-computed class probability
        distributions stored at each node during tree construction. This approach is much more
        efficient than recomputing probabilities on-the-fly and ensures consistency between
        training and prediction phases.
        
        Parameters
        ----------
        x : np.array
            Input data array with shape (n_samples, n_features).
        node : dict
            Current tree node being processed (contains 'class_probabilities').
        membership : np.array, optional
            Current membership values for each sample.
        best_membership : np.array, optional
            Best membership values found so far for each sample.
        class_probabilities : np.array, optional
            Current probability accumulation matrix (n_samples, n_classes).
        
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

        # If this is a leaf node or root with no children
        if not node.get('children', False) or len(node['children']) == 0:
            # Use cached probabilities from the node
            if 'class_probabilities' in node:
                node_probs = node['class_probabilities']
            else:
                # Fallback: uniform distribution if no cached probabilities
                node_probs = np.ones(n_classes) / n_classes
            
            if node['name'] == 'root':
                # For root node, apply to all samples
                for i in range(n_samples):
                    class_probabilities[i] = node_probs
            else:
                # For leaf nodes, update probabilities for samples with higher membership
                improved_samples = membership > best_membership
                if np.any(improved_samples):
                    class_probabilities[improved_samples] = node_probs
                    best_membership[improved_samples] = membership[improved_samples]
            
            return class_probabilities
        
        # For internal nodes, process all children
        for child_name, child in node['children'].items():
            relevant_feature = child['feature']
            relevant_fuzzy_set = child['fuzzy_set']
            child_path_membership = self.fuzzy_partitions[relevant_feature][relevant_fuzzy_set].membership(x[:, relevant_feature])
            full_path_membership = child_path_membership * membership
            
            # Recursively get probabilities from child
            class_probabilities = self._predict_proba(
                x, child, 
                membership=full_path_membership,
                best_membership=best_membership,
                class_probabilities=class_probabilities
            )
        
        # Handle samples that didn't reach any leaf (inactive samples)
        inactive_samples = best_membership <= 0.0
        if inactive_samples.any():
            # Use root node cached probabilities for inactive samples
            if 'class_probabilities' in self._root:
                root_probs = self._root['class_probabilities']
            else:
                root_probs = np.ones(n_classes) / n_classes
            class_probabilities[inactive_samples] = root_probs
        
        return class_probabilities


    def _predict(self, x: np.array, node, membership=None, paths=None, best_membership=None, prediction=None) -> tuple[np.array, np.array, np.array]:
        """
        Core recursive prediction method using batch processing for efficient fuzzy tree traversal.
        
        This method implements the batch prediction algorithm that traverses the fuzzy
        decision tree for multiple samples simultaneously. It uses numpy vectorization
        to efficiently compute fuzzy memberships and update predictions for all samples
        in parallel, significantly improving performance over sample-by-sample prediction.
        
        The algorithm maintains tracking arrays for each sample's best membership path,
        current prediction, and the path taken through the tree. For each child node,
        it computes the combined membership (parent membership × fuzzy set membership)
        and recursively processes all children, updating the best prediction whenever
        a higher membership path is found.
        
        Parameters
        ----------
        x : np.array
            Input data array with shape (n_samples, n_features).
        node : dict
            Current tree node being processed.
        membership : np.array, optional
            Current membership values for each sample. Initialized to all 1.0 if None.
        paths : np.array, optional
            Current path names for each sample. Initialized to 'root' if None.
        best_membership : np.array, optional
            Best membership values found so far for each sample. Initialized to -1.0 if None.
        prediction : np.array, optional
            Current predictions for each sample. Initialized to -1 if None.
        
        Returns
        -------
        tuple[np.array, np.array, np.array]
            Final predictions, best membership values, and path names for all samples.
        """

        if membership is None:
            membership = np.ones(x.shape[0])
            paths = np.full(x.shape[0], 'root', dtype=object)
            best_membership = np.zeros(x.shape[0]) - 1.0
            prediction = np.zeros(x.shape[0]) - 1

       # If there are no children, return prediction
        if not node.get('children', False) or len(node['children']) == 0:
            if node['name'] == 'root':
                return np.ones(x.shape[0]) * node['prediction'], membership * 0.0, paths
            else:
                # Update paths for all samples that reach this leaf
                improved_samples = membership > best_membership
                paths[improved_samples] = node['name']
                return np.ones(x.shape[0]) * node['prediction'], membership, paths
        else:

            for child_name, child in node['children'].items():
                relevant_feature = child['feature']
                relevant_fuzzy_set = child['fuzzy_set']
                child_path_membership = self.fuzzy_partitions[relevant_feature][relevant_fuzzy_set].membership(x[:, relevant_feature])
                full_path_membership = child_path_membership * membership
                
                child_pred, path_membership, child_paths = self._predict(x, child, membership=full_path_membership, paths=paths, best_membership=best_membership, prediction=prediction)

                changed_predictions = path_membership > best_membership

                best_membership[changed_predictions] = path_membership[changed_predictions]
                prediction[changed_predictions] = child_pred[changed_predictions]
                paths[changed_predictions] = child_paths[changed_predictions]

            inactive_samples = best_membership <= 0.0
            if inactive_samples.any():
                prediction[inactive_samples] = self.node_dict_access['root']['prediction']
                best_membership[inactive_samples] = 0.0
                paths[inactive_samples] = 'root'

            return prediction, best_membership, paths


    def _get_best_possible_coverage(self):
        """
        Calculate the maximum coverage achievable by any remaining valid split.
        
        This method examines all nodes in the tree to find the highest coverage
        value among all cached potential splits that meet the coverage threshold.
        It's used as a stopping criterion to determine when no more beneficial
        splits are possible.
        
        Returns
        -------
        float
            Maximum coverage ratio achievable by any valid split, or 0.0 if
            no valid splits exist.
        """
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
            feature_name = fuzzy_partitions[node['feature']].name
            fuzzy_set_name = fuzzy_partitions[node['feature']][node['fuzzy_set']].name
            
            # Get CCI/split criterion if available
            cci_info = ""
            if 'aux_purity_cache' in node:
                cache = node['aux_purity_cache']
                if 'quality_improvement' in node:
                    cci_info = f", split_criterion={node['quality_improvement']:.3f}"
            
            print(f"{prefix}{current_prefix}{node['name']}: {feature_name} IS {fuzzy_set_name} → class={node['prediction']}, coverage={node['coverage']:.3f}{cci_info}")
        
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
        Calculate comprehensive statistics about the tree structure.
        
        This method provides detailed information about the tree's structural
        properties including the total number of nodes, leaf/internal node counts,
        and maximum depth. Useful for understanding tree complexity and
        for debugging purposes.
        
        Returns
        -------
        dict
            Dictionary containing tree statistics:
            - 'total_nodes': Total number of nodes in the tree
            - 'leaves': Number of leaf nodes (terminal nodes)
            - 'internal': Number of internal nodes (non-terminal nodes)
            - 'depth': Maximum depth of the tree
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

    def _calculate_node_impurity(self, node: dict, X: np.array, y: np.array) -> float:
        """Calculate the weighted impurity of a node based on its membership."""
        membership = node['existing_membership']
        if np.sum(membership) == 0:
            return 0.0
        return compute_fuzzy_purity(membership, y, 0.0)

    def _calculate_subtree_impurity(self, node: dict, X: np.array, y: np.array) -> float:
        """Calculate the total weighted impurity of a subtree."""
        membership = node['existing_membership']
        node_weight = np.sum(membership)
        
        if 'children' not in node or not node['children']:
            # Leaf node
            return node_weight * self._calculate_node_impurity(node, X, y)
        
        # Internal node - sum of children impurities
        total_impurity = 0.0
        for child in node['children'].values():
            total_impurity += self._calculate_subtree_impurity(child, X, y)
        
        return total_impurity

    def _count_leaves(self, node: dict) -> int:
        """Count the number of leaf nodes in a subtree."""
        if 'children' not in node or not node['children']:
            return 1
        
        total_leaves = 0
        for child in node['children'].values():
            total_leaves += self._count_leaves(child)
        
        return total_leaves

    def _calculate_complexity_measure(self, node: dict, X: np.array, y: np.array) -> float:
        """Calculate the complexity measure (alpha) for pruning a subtree at this node."""
        if 'children' not in node or not node['children']:
            return float('inf')  # Can't prune a leaf
        
        # Impurity if we prune this subtree (make it a leaf)
        node_impurity = self._calculate_node_impurity(node, X, y) * np.sum(node['existing_membership'])
        
        # Impurity of the current subtree
        subtree_impurity = self._calculate_subtree_impurity(node, X, y)
        
        # Number of leaves that would be removed
        leaves_removed = self._count_leaves(node) - 1
        
        if leaves_removed <= 0:
            return float('inf')
        
        # Complexity measure (alpha)
        alpha = (node_impurity - subtree_impurity) / leaves_removed
        return alpha

    def _find_weakest_link(self, X: np.array, y: np.array) -> tuple[dict, float]:
        """Find the node with the smallest complexity measure (weakest link)."""
        min_alpha = float('inf')
        weakest_node = None
        
        def traverse(node):
            nonlocal min_alpha, weakest_node
            
            if 'children' in node and node['children']:
                alpha = self._calculate_complexity_measure(node, X, y)
                if alpha < min_alpha:
                    min_alpha = alpha
                    weakest_node = node
                
                # Recursively check children
                for child in node['children'].values():
                    traverse(child)
        
        traverse(self._root)
        return weakest_node, min_alpha

    def _prune_subtree(self, node: dict, X: np.array, y: np.array):
        """Convert an internal node to a leaf by removing its children."""
        if 'children' in node:
            # Remove children from node dictionary
            for child_name in list(node['children'].keys()):
                self._remove_from_node_dict(node['children'][child_name])
            
            # Remove children reference
            del node['children']
            
            # Update tree rules count
            leaves_removed = self._count_leaves(node)
            self.tree_rules -= (leaves_removed - 1)
            
            # Recalculate prediction and probabilities for this new leaf
            membership = node['existing_membership']
            node['prediction'] = self._majority_class(y, membership)
            node['class_probabilities'] = self._class_probabilities(y, membership)

    def _remove_from_node_dict(self, node: dict):
        """Recursively remove a node and its children from node_dict_access."""
        if node['name'] in self.node_dict_access:
            del self.node_dict_access[node['name']]
        
        if 'children' in node:
            for child in node['children'].values():
                self._remove_from_node_dict(child)


    def cost_complexity_pruning(self, X: np.array, y: np.array, alpha: float = None):
        """
        Perform cost-complexity pruning on the tree.
        
        Parameters
        ----------
        X : np.array
            Training data features used for pruning decisions.
        y : np.array
            Training data labels used for impurity calculations.
        alpha : float, optional
            Complexity parameter. If None, uses self.ccp_alpha.
        
        Returns
        -------
        list[float]
            Sequence of alpha values used for pruning.
        """
        if alpha is None:
            alpha = self.ccp_alpha
        
        alpha_sequence = [0.0]  # Start with no pruning
        
        while True:
            # Find the weakest link
            weakest_node, min_alpha = self._find_weakest_link(X, y)
            
            if weakest_node is None or min_alpha >= alpha:
                break
            
            # Prune the weakest link
            self._prune_subtree(weakest_node, X, y)
            alpha_sequence.append(min_alpha)
            
            # Stop if tree becomes just the root
            if 'children' not in self._root or not self._root['children']:
                break
        
        return alpha_sequence

    def fit_with_pruning(self, X: np.array, y: np.array, X_val: np.array = None, y_val: np.array = None):
        """
        Fit the tree and apply cost-complexity pruning.
        
        Parameters
        ----------
        X : np.array
            Training data features.
        y : np.array
            Training data labels.
        X_val : np.array, optional
            Validation data for selecting optimal alpha. If None, uses training data.
        y_val : np.array, optional
            Validation labels for selecting optimal alpha. If None, uses training labels.
        """
        # First, build the full tree
        self.fit(X, y)
        
        # If no validation data provided, use training data
        if X_val is None:
            X_val, y_val = X, y
        
        # Store the original tree
        original_tree = self._deep_copy_tree()
        
        # Get sequence of alpha values
        alpha_sequence = self.cost_complexity_pruning(X, y, float('inf'))
        
        best_score = -float('inf')
        best_alpha = 0.0
        
        # Test each alpha value
        for alpha in alpha_sequence:
            # Restore original tree
            self._restore_tree(original_tree)
            
            # Prune with this alpha
            self.cost_complexity_pruning(X, y, alpha)
            
            # Evaluate on validation data
            score = self.score(X_val, y_val)
            
            if score > best_score:
                best_score = score
                best_alpha = alpha
        
        # Final pruning with best alpha
        self._restore_tree(original_tree)
        self.cost_complexity_pruning(X, y, best_alpha)
        
        return best_alpha, best_score

    def _deep_copy_tree(self) -> dict:
        """Create a deep copy of the current tree structure."""
        import copy
        return {
            'root': copy.deepcopy(self._root),
            'node_dict': copy.deepcopy(self.node_dict_access),
            'tree_rules': self.tree_rules
        }

    def _restore_tree(self, tree_backup: dict):
        """Restore tree from backup."""
        self._root = tree_backup['root']
        self.node_dict_access = tree_backup['node_dict']
        self.tree_rules = tree_backup['tree_rules']

if __name__ == "__main__":
    # Load iris dataset
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    print("Testing with iris dataset:")
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create fuzzy partitions
    fuzzy_partitions = utils.construct_partitions(X_train, fs.FUZZY_SETS.t1, n_partitions=3, shape='trapezoid')
    
    # Train classifier
    classifier = FuzzyCART(fuzzy_partitions, max_depth=4, max_rules=15, coverage_threshold=0.0, min_improvement=0.0, target_metric='cci')
    classifier.fit(X_train, y_train, patience=20)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    y_train_pred = classifier.predict(X_train)
    print(f"Iris Test Accuracy: {accuracy:.3f}")
    print(f"Iris Train Accuracy: {accuracy_score(y_train, y_train_pred):.3f}")

    # Test probability predictions
    y_proba = classifier.predict_proba(X_test)
    print(f"\nProbability predictions shape: {y_proba.shape}")
    print(f"Sample probabilities for first 5 test samples:")
    for i in range(min(5, len(X_test))):
        pred_class = y_pred[i]
        true_class = y_test[i]
        probs = y_proba[i]
        print(f"  Sample {i}: True={true_class}, Pred={pred_class}, Probs={probs}")
    
    # Verify probabilities sum to 1
    prob_sums = np.sum(y_proba, axis=1)
    print(f"Probability sums (should be ~1.0): min={prob_sums.min():.6f}, max={prob_sums.max():.6f}")
    
    # Test that cached probabilities are consistent
    y_proba2 = classifier.predict_proba(X_test)
    consistency_check = np.allclose(y_proba, y_proba2)
    print(f"Probability consistency check: {consistency_check}")
    
    # Show some node probabilities for verification
    print(f"\nExample node probabilities (cached during tree construction):")
    for node_name, node in list(classifier.node_dict_access.items())[:3]:
        if 'class_probabilities' in node:
            print(f"  {node_name}: {node['class_probabilities']}")

    print("\nTree Structure:")
    classifier.print_tree()
    
    print("\nTree Statistics:")
    stats = classifier.get_tree_stats()
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"Leaf nodes: {stats['leaves']}")
    print(f"Internal nodes: {stats['internal']}")
    print(f"Tree depth: {stats['depth']}")

    # Create and fit the classifier with pruning
    fuzzy_cart = FuzzyCART(
        fuzzy_partitions=fuzzy_partitions,
        max_rules=20,
        ccp_alpha=0.01,  # Set desired complexity parameter
        coverage_threshold=0.03,
        min_improvement=0.01,
        target_metric='cci'
    )
    print()
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=33)
    
    # Fit with automatic pruning parameter selection
    '''best_alpha, best_score = fuzzy_cart.fit_with_pruning(X_train, y_train, X_val, y_val)
    final_score = fuzzy_cart.score(X_test, y_test)
    print(f"Best alpha: {best_alpha}, Final score: {final_score}")
    print("\nTree Statistics:")
    stats = classifier.get_tree_stats()
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"Leaf nodes: {stats['leaves']}")
    print(f"Internal nodes: {stats['internal']}")
    print(f"Tree depth: {stats['depth']}")'''

    
    