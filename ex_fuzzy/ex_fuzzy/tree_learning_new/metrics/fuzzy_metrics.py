"""
Fuzzy classification metrics and evaluation functions.

This module provides core mathematical functions for evaluating fuzzy classification
performance, including Complete Classification Index (CCI), fuzzy purity measures,
and related utility functions.

These functions are designed to be reusable across different fuzzy classifiers
and evaluation scenarios.
"""

import numpy as np


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
    if len(truth_values) == 0:
        return float('inf')
    
    total_weight = np.sum(truth_values)
    if total_weight == 0:
        return float('inf')
    
    unique_classes = np.unique(y)
    
    # OPTIMIZATION: Vectorized class proportion calculation
    weighted_proportions = np.zeros(len(unique_classes))
    for i, cls in enumerate(unique_classes):
        cls_weight = np.sum(truth_values[y == cls])
        weighted_proportions[i] = cls_weight / total_weight
    
    # Compute weighted gini index (multiclass)
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
    weighted_gini_node = _weighted_gini_index(truth_values, y)
    coverage = _calculate_coverage(truth_values, len(y))

    if coverage < minimum_coverage_threshold:  # Minimum coverage threshold
        return float('inf')
    else:
        return weighted_gini_node


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

    cci_index = _complete_classification_index(y, pre_yhat, new_yhat)
    coverage = _calculate_coverage(truth_values, len(y))

    if coverage < minimum_coverage_threshold:  # Minimum coverage threshold
        return float(-1.0)
    else:
        return cci_index


def majority_class(y: np.array, membership: np.array = None, classes: np.array = None):
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
    classes : np.array, optional
        Array of unique classes. If None, derived from y.
    
    Returns
    -------
    int or class_type
        The majority class after weighting by membership values.
        Returns the first class if no samples are provided.
    """
    if membership is None:
        membership = np.ones(len(y))

    if len(y) == 0:
        if classes is not None:
            return classes[0]
        else:
            return np.unique(y)[0] if len(y) > 0 else 0  # Fallback

    if classes is None:
        classes, counts = np.unique(y, return_counts=True)
    else:
        classes = np.array(classes)
        counts = np.array([np.sum(y == cls) for cls in classes])

    # Weight counts by membership
    weighted_counts = np.array([np.sum(membership[y == cls]) for cls in classes])
    if np.sum(weighted_counts) != 0.0:
        weighted_counts = weighted_counts / np.sum(weighted_counts)  # Normalize to get probabilities

    return classes[np.argmax(weighted_counts)]


def class_probabilities(y: np.array, membership: np.array = None, classes: np.array = None):
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
    classes : np.array, optional
        Array of unique classes. If None, derived from y.
    
    Returns
    -------
    np.array
        Probability vector with length equal to number of classes, where
        probabilities sum to 1.0. Each element represents the probability
        of the corresponding class in classes.
    """
    if membership is None:
        membership = np.ones(len(y))

    if classes is None:
        classes = np.unique(y)
    
    # Initialize probability vector for all classes
    class_probs = np.zeros(len(classes))
    
    if len(y) == 0:
        # If no samples, return uniform distribution
        class_probs.fill(1.0 / len(classes))
        return class_probs
    
    # Calculate weighted counts for each class
    for i, cls in enumerate(classes):
        class_mask = (y == cls)
        class_probs[i] = np.sum(membership[class_mask])
    
    # Normalize to get probabilities
    total_weight = np.sum(class_probs)
    if total_weight > 0:
        class_probs = class_probs / total_weight
    else:
        # If no membership weight, return uniform distribution
        class_probs.fill(1.0 / len(classes))
    
    return class_probs


def evaluate_fuzzy_feature_partition(existing_membership: np.array, y: np.array, 
                                   X: np.array, fuzzy_partitions: list, feature_idx: int, 
                                   metric_func) -> tuple:
    """
    Evaluate the quality of partitioning a feature using all its linguistic labels.
    
    This function computes the weighted metric (impurity or CCI) improvement that would 
    result from splitting on a feature using ALL its linguistic labels simultaneously.
    It's used by both purity and CCI-based splitting algorithms.
    
    Parameters
    ----------
    existing_membership : np.array
        Current membership values for samples in the node being evaluated.
    y : np.array
        Class labels for all samples.
    X : np.array
        Training data features.
    fuzzy_partitions : list
        List of fuzzy variables, each containing linguistic labels.
    feature_idx : int
        Index of the feature to evaluate.
    metric_func : callable
        Function to compute the metric for each partition (e.g., compute_fuzzy_gini_impurity).
        Must accept (membership, y) parameters and return a numeric score.
    
    Returns
    -------
    tuple
        (total_weighted_metric, total_membership, valid_partitions)
        - total_weighted_metric: Sum of metric * weight for all partitions
        - total_membership: Sum of membership weights across all partitions  
        - valid_partitions: Number of partitions evaluated
    """
    total_weighted_metric = 0.0
    total_membership = 0.0
    valid_partitions = 0
    
    # Evaluate all linguistic labels for this feature
    for fuzzy_set_idx in range(len(fuzzy_partitions[feature_idx])):
        fuzzy_set = fuzzy_partitions[feature_idx][fuzzy_set_idx]
        child_membership = existing_membership * fuzzy_set.membership(X[:, feature_idx])
        
        # Calculate partition metric
        partition_metric = metric_func(child_membership, y)
        child_weight = np.sum(child_membership)
        
        total_weighted_metric += partition_metric * child_weight
        total_membership += child_weight
        valid_partitions += 1
    
    return total_weighted_metric, total_membership, valid_partitions