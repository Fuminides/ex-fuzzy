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


def compute_fuzzy_purity(truth_values: np.array, y: np.array) -> float:
    """
    Compute the fuzzy purity of a split using weighted gini index for multiclass.

    Args:
        truth_values (np.array): The membership degrees/weights for each sample (0-1).
        y (np.array): The class labels for all samples.

    Returns:
        float: The weighted gini impurity score (lower is better, 0 = pure).
    """
    if len(truth_values) == 0 or np.sum(truth_values) == 0:
        return 0.0
    
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


def best_split(X: np.array, y: np.array, feature_indices: list[int], fuzzy_partitions:list[fs.fuzzyVariable], previous_membership=None) -> tuple[int, int, float]: 
    """
    Find the best feature and fuzzy set to split the data.

    Args:
        X (np.array): The input features.
        y (np.array): The class labels.
        feature_indices (list[int]): Indices of features to consider for splitting.
        fuzzy_partitions (list[fs.fuzzyVariable]): List of fuzzy variables for each feature.

    Returns:
        tuple: (best_feature_index, best_fuzzy_set_index, best_purity_score)
    """
    best_purity = float('inf')
    best_feature_index = -1
    best_fuzzy_set_index = -1

    for feature_index in feature_indices:
        fuzzy_var = fuzzy_partitions[feature_index]
        for fz_index, fz in enumerate(fuzzy_var):
            # Compute truth values for the current fuzzy set
            truth_values = fz.membership(X[:, feature_index])
            
            if previous_membership is not None:
                truth_values = truth_values * previous_membership
            # Compute fuzzy purity
            purity = compute_fuzzy_purity(truth_values, y)
            
            if purity < best_purity:
                best_purity = purity
                best_feature_index = feature_index
                best_fuzzy_set_index = fz_index

    return best_feature_index, best_fuzzy_set_index, best_purity



class FuzzyCART(ClassifierMixin):

    def __init__(self, fuzzy_partitions: list[fs.fuzzyVariable], max_depth: int = 5):
        self.fuzzy_partitions = fuzzy_partitions
        self.max_depth = max_depth
        self.tree = None


    def fit(self, X: np.array, y: np.array):
        self.classes_ = np.unique(y)
        self.tree = self._build_tree(X, y, depth=0)
        return self


    def _build_tree(self, X: np.array, y: np.array, depth: int, existing_membership=None):
        # Stopping criteria
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < 2:
            return {'prediction': self._majority_class(y)}
        if existing_membership is None:
            existing_membership = np.ones(X.shape[0])

        # Find best split
        feature_indices = list(range(X.shape[1]))
        best_feature, best_fuzzy_set, best_purity = best_split(X, y, feature_indices, self.fuzzy_partitions)
        
        # If the best feature is this already, we return a leaf
        if best_feature == -1: 
            return {'prediction': self._majority_class(y)}
        
        # Split data
        fuzzy_set = self.fuzzy_partitions[best_feature][best_fuzzy_set]
        memberships = fuzzy_set.membership(X[:, best_feature])
        threshold = 0.0  # You can adjust this threshold as needed
        left_membership = memberships * existing_membership
        right_membership = (1 - memberships) * existing_membership

        left_mask = left_membership > threshold
        right_mask = right_membership > threshold

        # Check if split results in empty partitions
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return {'prediction': self._majority_class(y)}
        
        # Build subtrees
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1, left_membership[left_mask])
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1, right_membership[right_mask])

        return {
            'feature': best_feature,
            'fuzzy_set': best_fuzzy_set,
            'threshold': threshold,
            'left': left_tree,
            'right': right_tree
        }


    def _majority_class(self, y: np.array):
        if len(y) == 0:
            return self.classes_[0]  # Return first class if no samples
        classes, counts = np.unique(y, return_counts=True)
        return classes[np.argmax(counts)]


    def predict(self, X: np.array) -> np.array:
        return np.array([self._predict_single(x, self.tree) for x in X])
    

    def _predict_single(self, x: np.array, node):
        if 'prediction' in node:
            return node['prediction']
        else:
            fuzzy_set = self.fuzzy_partitions[node['feature']][node['fuzzy_set']]
            membership = fuzzy_set.membership(x[node['feature']])
            
            if membership >= 1 - membership:
                return self._predict_single(x, node['left'])
            else:
                return self._predict_single(x, node['right'])


if __name__ == "__main__":
    # Load iris dataset
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create fuzzy partitions
    fuzzy_partitions = utils.construct_partitions(X_train, fs.FUZZY_SETS.t1, n_partitions=3, shape='trapezoid')
    
    # Train classifier
    classifier = FuzzyCART(fuzzy_partitions, max_depth=3)
    classifier.fit(X_train, y_train)
    
    # Make predictions
    y_pred = classifier.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.3f}")

