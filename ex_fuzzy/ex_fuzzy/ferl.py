"""Fast Evidential Rule Learning (FERL) for interpretable classification.

This module is a native Ex-Fuzzy port of the fuzzy greedy rule-tree learner.
FERL grows fuzzy rules greedily and turns their firing strengths into
Dempster--Shafer evidence for point, interval, and set-valued predictions.
"""
from __future__ import annotations
import copy

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
try:
    from . import fuzzy_sets as fs
    from . import utils
    from .ferl_partitions import learn_partitions_mdlp
except ImportError:
    import fuzzy_sets as fs
    import utils
    from ferl_partitions import learn_partitions_mdlp




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


class LearnedRampSet:
    """A fuzzy set whose boundary location is learned from the data (the
    performance-mode / ``split_mode='learned'`` alternative to the fixed
    per-feature partition). Linguistically "x is (softly) below/above ``center``",
    with a linear ramp of half-width ``h``: membership is 1 at ``center - h`` and
    0 at ``center + h`` for the 'below' set (mirrored for 'above'). ``h`` encodes
    boundary uncertainty — narrow ramp = confident cut, wide ramp = fuzzy cut.

    Quacks like an ex_fuzzy fuzzy set (exposes ``.membership`` and ``.name``), so
    once created it is an ordinary global set: all prediction / caching / credal
    paths index it through ``fuzzy_partitions[f][fz].membership(x)`` unchanged.
    """
    __slots__ = ("center", "h", "direction", "name")

    def __init__(self, center, h, direction, name=""):
        self.center = float(center)
        self.h = float(max(h, 1e-9))
        self.direction = direction          # 'below' or 'above'
        self.name = name

    def membership(self, x):
        below = np.clip((self.center + self.h - x) / (2.0 * self.h), 0.0, 1.0)
        return below if self.direction == "below" else 1.0 - below

    # ex_fuzzy fuzzy sets are callable (set(x) == set.membership(x)); match that
    # so code paths that call the set directly work on learned sets too.
    def __call__(self, x):
        return self.membership(x)


def _learned_best_cut(x, y_oh, w, parent, W):
    """Weighted-Gini optimal crisp threshold on one feature (the location a
    LearnedRampSet is centered on). ``y_oh`` is the (n, C) one-hot label matrix,
    ``w`` the per-sample node membership weights. Returns (gain, threshold).

    Vectorized: all candidate split gains are evaluated at once from the class
    cumulative sums (no per-cut-point Python loop), then the best is taken."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    cum = np.cumsum(y_oh[order] * w[order, None], axis=0)      # (n, C)
    Wl_arr = np.cumsum(w[order])                              # (n,)
    tot = cum[-1]                                             # (C,)
    bnd = np.flatnonzero(xs[:-1] != xs[1:])                   # candidate split rows
    if bnd.size == 0:
        return 0.0, None
    Wl = Wl_arr[bnd]
    Wr = W - Wl
    ok = (Wl > 1e-12) & (Wr > 1e-12)
    if not ok.any():
        return 0.0, None
    bnd, Wl, Wr = bnd[ok], Wl[ok], Wr[ok]
    l = cum[bnd]                                              # (m, C)
    r = tot[None, :] - l
    gini_l = 1.0 - ((l / Wl[:, None]) ** 2).sum(1)
    gini_r = 1.0 - ((r / Wr[:, None]) ** 2).sum(1)
    gain = parent - (Wl / W * gini_l + Wr / W * gini_r)       # (m,)
    k = int(np.argmax(gain))
    if gain[k] <= 0.0:
        return 0.0, None
    i = bnd[k]
    return float(gain[k]), 0.5 * (xs[i] + xs[i + 1])


class FERL(BaseEstimator, ClassifierMixin):
    """Fast Evidential Rule Learning classifier.

    FERL greedily grows a fuzzy rule tree and aggregates activated rules for
    ordinary probabilities or Dempster--Shafer evidential predictions. It can
    use a compact, human-readable fixed partition or learn soft split locations
    and widths from the data for a deeper model.

    Parameters
    ----------
    fuzzy_partitions : list[fs.fuzzyVariable], optional
        One fuzzy variable per feature. When omitted, FERL constructs the
        partitions during :meth:`fit`.
    max_rules : int, default=15
        Maximum number of rules (leaf nodes) allowed in the tree. Controls
        tree complexity and helps prevent overfitting.
    max_depth : int, default=5
        Maximum depth of the tree. Limits how deep the tree can grow.
    coverage_threshold : float, default=0.00
        Minimum coverage ratio required for a split to be considered valid.
        Splits covering fewer samples than this threshold are rejected.
    min_improvement : float, default=0.01
        Minimum split-quality improvement. FERL stops after ``patience``
        consecutive candidates do not exceed it.
    ccp_alpha : float, default=0.0
        Cost-complexity parameter used by the explicit pruning methods.
    target_metric : {"cci", "purity"}, default="cci"
        Greedy split criterion. CCI aligns compact-tree growth with the
        classifier decision; purity uses weighted Gini impurity.
    sample_for_splits : bool, optional
        Whether to evaluate split candidates on a sample of the training data.
        ``None`` enables sampling automatically above 50,000 rows.
    sample_size : int, default=10000
        Maximum rows used by sampled split evaluation.
    reliability_k : float, optional
        Support pseudo-count for reliability discounting in evidential output.
    partition : {"quantile", "mdlp"}, default="quantile"
        Automatic partition construction used when ``fuzzy_partitions`` is
        omitted.
    n_partitions : int, default=3
        Number of quantile terms per numerical feature.
    overlap_frac : float, default=0.8
        Overlap fraction for supervised MDLP trapezoids.
    split_mode : {"fixed", "learned"}, default="fixed"
        Search the fitted linguistic terms or learn soft binary split ramps.
    learned_width : {"bootstrap"} or float, default="bootstrap"
        How learned ramps obtain their half-width. A float multiplies the
        weighted feature standard deviation.
    learned_n_boot : int, default=25
        Bootstrap replicates used to estimate learned ramp widths.
    prediction_mode : {"soft", "soft_gate", "hard_gate", "winner"}, default="soft"
        Rule aggregation used by point prediction.
    consistent_cci : bool, default=True
        Score CCI candidates in the same additive vote space as soft inference.
    coverage_weight : float, default=0.0
        Optional reward for covering samples not activated by current rules.
    multiway_splits : bool, default=False
        Add all remaining terms of a selected feature as sibling rules.
    random_state : int, optional
        Seed for split sampling and learned-width bootstrapping.

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
    fuzzy_partitions_ : list[fs.fuzzyVariable]
        Fitted partitions, including learned ramp sets when applicable.
    """

    def _as_array(self, X):
        """Validate prediction input and return a numeric array."""
        check_is_fitted(self, attributes=["classes_", "fuzzy_partitions_"])
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError("X must be a one- or two-dimensional array.")
        if hasattr(self, "n_features_in_") and X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but FERL was fitted with "
                f"{self.n_features_in_} features."
            )
        return X


    def _clear_all_split_caches(self):
        """Clear cached split evaluations from all nodes."""
        for node in self.node_dict_access.values():
            if 'aux_purity_cache' in node:
                del node['aux_purity_cache']


    def _get_cached_memberships(self, X: np.array) -> dict:
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
            for feature_idx, fuzzy_var in enumerate(self.fuzzy_partitions_):
                feature_memberships = np.zeros((len(fuzzy_var), X.shape[0]))
                for fz_idx, fuzzy_set in enumerate(fuzzy_var):
                    feature_memberships[fz_idx] = fuzzy_set.membership(X[:, feature_idx])
                self._membership_cache[feature_idx] = feature_memberships

        return self._membership_cache


    def __init__(self, fuzzy_partitions=None, max_rules: int = 15, max_depth: int = 5,
                 coverage_threshold: float = 0.0, min_improvement: float = 0.01,
                 ccp_alpha: float = 0.0, target_metric: str = 'cci',
                 sample_for_splits: bool = None, sample_size: int = 10000,
                 reliability_k: float = None, partition: str = 'quantile',
                 n_partitions: int = 3, overlap_frac: float = 0.8,
                 split_mode: str = 'fixed', learned_width='bootstrap',
                 learned_n_boot: int = 25, prediction_mode: str = 'soft',
                 consistent_cci: bool = True, coverage_weight: float = 0.0,
                 multiway_splits: bool = False, random_state=None):
        """
        Initialize FERL.

        Parameters
        ----------
        sample_for_splits : bool, optional
            If True, use sampling for split evaluation on large datasets.
            If None, automatically enabled for datasets > 50,000 samples.
        sample_size : int, default=10000
            Number of samples to use for split evaluation when sampling is enabled.
        reliability_k : float, optional
            Pseudo-count for the Shafer reliability discount used in predict_ds.
            A node with fuzzy training support N is trusted by r = N / (N + k),
            so low-support (thin, deep) leaves send mass to ignorance instead of
            their noisy class estimate. None (default) disables discounting.
        """
        self.fuzzy_partitions = fuzzy_partitions
        self.fuzzy_partitions_ = None
        self.partition = partition
        self.n_partitions = n_partitions
        self.overlap_frac = overlap_frac
        self.random_state = random_state
        self.max_depth = max_depth
        self.tree = None
        self.tree_rules = 1 # Start with 1 (so that the first split creates the first rule)
        self.max_rules = max_rules
        self.coverage_threshold = coverage_threshold
        self.min_improvement = min_improvement
        self.ccp_alpha = ccp_alpha
        self.target_metric = target_metric
        self.sample_for_splits = sample_for_splits
        self.sample_size = sample_size
        self.reliability_k = reliability_k

        # Fuzzification mode. 'fixed' (default) = split on the pre-built fuzzy
        # partition (fast / interpretable). 'learned' = performance mode: each
        # split places a data-chosen threshold (CART-like) rendered as a soft
        # LearnedRampSet, with the ramp half-width taken from the bootstrap
        # uncertainty of the cut location ('bootstrap') or c * feature-std (a
        # float). Learned sets are appended to fuzzy_partitions as they are
        # discovered, so all prediction / credal paths are unchanged.
        self.split_mode = split_mode
        self.learned_width = learned_width
        self.learned_n_boot = learned_n_boot

        # Inference mode for predict()/predict_proba(). Controls how node
        # evidence is aggregated into a probability distribution:
        #   'soft_gate' : membership-weighted node class_probabilities, with
        #                 internal nodes gated to where all children are ~0.
        #   'hard_gate' : membership-weighted hard node class (one-hot), gated.
        #   'soft'      : membership-weighted class_probabilities, no gating
        #                 (every firing node contributes; root excluded).
        #   'winner'    : legacy hard winner-take-all over the single
        #                 highest-membership node.
        # 'soft' is the standard fuzzy rule-base aggregation and the best
        # performing on the benchmark suite; it is the default.
        self.prediction_mode = prediction_mode

        # When True, the CCI splitting criterion scores a candidate split in the
        # same additive soft-vote space that 'soft' inference uses (adding the
        # candidate node's membership-weighted class distribution and re-taking
        # the argmax), instead of the legacy hard membership>0.01 override.
        # Enabled by default: aligns the split criterion with 'soft' inference
        # and gives the best accuracy on the benchmark suite.
        self.consistent_cci = consistent_cci

        # Coverage-aware growth: when > 0, the split criterion is rewarded for
        # firing on currently-uncovered (zero total-firing) samples, biasing the
        # greedy search toward filling rule-coverage gaps. 0 = standard CCI.
        self.coverage_weight = coverage_weight

        # Multiway-split intervention: after a greedy split, also add the chosen
        # feature's *remaining* fuzzy sets as sibling children, so each split is a
        # full (mutually-exclusive) partition rather than a one-child chain. For
        # the credal-calibration study (does less nesting reduce the Dempster
        # double-counting?). Off = standard one-child greedy growth.
        self.multiway_splits = multiway_splits


        # OPTIMIZATION: Add membership cache to avoid recomputing
        self._membership_cache = {}
        self._last_X_shape = None

        # OPTIMIZATION: Add computation caches for large datasets
        self._coverage_cache = {}
        self._gini_cache = {}
        self._prediction_cache = None


    def fit(self, X: np.array, y: np.array, patience:int = 3):
        """
        Train FERL on the provided dataset.

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
        feature_names = None
        if hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        if hasattr(X, "columns"):
            feature_names = np.asarray(X.columns, dtype=object)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")
        if y.ndim != 1:
            y = np.ravel(y)
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples.")
        if len(X) == 0:
            raise ValueError("FERL requires at least one training sample.")
        if not np.all(np.isfinite(X)):
            raise ValueError("FERL does not support NaN or infinite feature values.")
        if self.partition not in ("quantile", "mdlp"):
            raise ValueError("partition must be either 'quantile' or 'mdlp'.")
        if self.split_mode not in ("fixed", "learned"):
            raise ValueError("split_mode must be either 'fixed' or 'learned'.")
        if self.prediction_mode not in ("soft", "soft_gate", "hard_gate", "winner"):
            raise ValueError(
                "prediction_mode must be 'soft', 'soft_gate', 'hard_gate', or 'winner'."
            )

        if self.fuzzy_partitions is None:
            if self.partition == "mdlp":
                self.fuzzy_partitions_ = learn_partitions_mdlp(
                    X, y, overlap_frac=self.overlap_frac
                )
            else:
                self.fuzzy_partitions_ = utils.construct_partitions(
                    X, fs.FUZZY_SETS.t1, n_partitions=self.n_partitions
                )
        else:
            self.fuzzy_partitions_ = copy.deepcopy(self.fuzzy_partitions)

        if len(self.fuzzy_partitions_) != X.shape[1]:
            raise ValueError(
                "fuzzy_partitions must contain one fuzzy variable per feature."
            )
        if feature_names is not None:
            for name, fuzzy_variable in zip(feature_names, self.fuzzy_partitions_):
                fuzzy_variable.name = str(name)
            self.feature_names_in_ = feature_names

        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("FERL requires at least two target classes.")
        self.tree_rules = 1
        self._rng = np.random.default_rng(self.random_state)
        if hasattr(self, "_fixed_partitions"):
            del self._fixed_partitions
        self._invalidate_leaf_cache()
        self._build_tree(X, y, bad_cuts_limit=patience, index=self.target_metric)
        return self


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
        # Number of training samples; used to turn each node's coverage back into
        # a fuzzy support count for the reliability discount in predict_ds.
        self._n_train = X.shape[0]
        # Create flexible path structure - list of boolean arrays, one per feature
        actual_path = [np.ones(len(fuzzy_var), dtype=bool) for fuzzy_var in self.fuzzy_partitions_]

        self._root  = {
            'depth': 0,
            'existing_membership': existing_membership,
            'father_path': actual_path,
            'child_splits': [path.copy() for path in actual_path],  # Deep copy of the list structure
            'name': 'root',
            'prediction': -1, # No prediction at root
            'coverage': 1.0,
            'class_probabilities': self._class_probabilities(y, existing_membership),
            '_cached_path': []  # OPTIMIZATION: Empty path for root
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
        # Enforce max_depth: a node at the depth limit cannot be split.
        if node['depth'] >= self.max_depth:
            node['aux_purity_cache'] = {
                'feature': -1, 'fuzzy_set': -1, 'coverage': 0.0,
                'split_criterion': 0.0, 'child_decision': None, 'purity': 0.0
            }
            return float('-inf')

        # OPTIMIZATION: Use sampling for very large datasets
        use_sampling = self.sample_for_splits
        if use_sampling is None:
            use_sampling = X.shape[0] > 50000

        if use_sampling and X.shape[0] > self.sample_size:
            # Sample for split evaluation
            n_samples = min(self.sample_size, X.shape[0])
            sample_indices = self._rng.choice(X.shape[0], n_samples, replace=False)
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]
            existing_membership_sample = node['existing_membership'][sample_indices]
        else:
            X_sample = X
            y_sample = y
            existing_membership_sample = node['existing_membership']

        existing_membership = existing_membership_sample
        father_path = node['father_path']
        child_splits = node['child_splits']
        # Combine paths: element-wise AND for each feature
        actual_path = [np.logical_and(father_path[i], child_splits[i]) for i in range(len(father_path))]

        n_features = len(self.fuzzy_partitions_)
        best_purity_improvement = float('-inf')
        best_feature = -1
        best_fuzzy_set = -1
        best_coverage = 0.0
        father_purity = compute_fuzzy_purity(existing_membership, y_sample, self.coverage_threshold)

        # For debugging, create cache structures that accommodate variable fuzzy set counts
        debug_cache = [np.zeros(len(self.fuzzy_partitions_[i])) for i in range(n_features)]
        coverage_cache = [np.zeros(len(self.fuzzy_partitions_[i])) for i in range(n_features)]

        # OPTIMIZATION: Use cached memberships instead of recomputing
        if use_sampling and X.shape[0] > self.sample_size:
            # Compute memberships for sample
            cached_memberships = {}
            for feature_idx, fuzzy_var in enumerate(self.fuzzy_partitions_):
                feature_memberships = np.zeros((len(fuzzy_var), X_sample.shape[0]))
                for fz_idx, fuzzy_set in enumerate(fuzzy_var):
                    feature_memberships[fz_idx] = fuzzy_set.membership(X_sample[:, feature_idx])
                cached_memberships[feature_idx] = feature_memberships
        else:
            cached_memberships = self._get_cached_memberships(X_sample)

        for feature in range(n_features):
            for fz_index in range(len(self.fuzzy_partitions_[feature])):
                if actual_path[feature][fz_index]:
                    # Use cached membership
                    memberships = cached_memberships[feature][fz_index]
                    full_path_membership = memberships * existing_membership

                    purity = compute_fuzzy_purity(full_path_membership, y_sample, self.coverage_threshold)
                    debug_cache[feature][fz_index] = purity
                    coverage = _calculate_coverage(full_path_membership, len(y_sample))
                    coverage_cache[feature][fz_index] = coverage
                    purity_improvement = father_purity - purity

                    # OPTIMIZATION: Compute child prediction directly without dummy nodes
                    node_prediction = self._majority_class(y_sample, full_path_membership)

                    if purity_improvement > best_purity_improvement:
                        best_purity_improvement = purity_improvement
                        best_feature = feature
                        best_fuzzy_set = fz_index
                        best_coverage = coverage
                        child_decision = node_prediction

        if best_feature != -1:
            node['aux_purity_cache'] = {
                'feature': best_feature,
                'fuzzy_set': best_fuzzy_set,
                'coverage': best_coverage,
                'split_criterion': best_purity_improvement,
                'child_decision': child_decision,
                'purity': best_purity_improvement
            }
        else:
            node['aux_purity_cache'] = {
                'feature': -1,
                'fuzzy_set': -1,
                'coverage': 0.0,
                'split_criterion': 0.0,
                'child_decision': None,
                'purity': 0.0
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


    def _build_cci_context(self, X: np.array, y: np.array) -> dict:
        """
        Build the per-iteration scoring context shared by all nodes in one tree scan.

        The baseline tree prediction (and, for consistent scoring, the unnormalized
        soft-vote sums) is identical for every node within a single split search
        because the tree is frozen during the scan. Computing it once here and
        passing it to every ``_node_cci_checks`` call turns the previous
        O(nodes^2) full re-predictions per iteration into O(nodes).

        Returns
        -------
        dict
            Context with the (optionally sampled) data, cached memberships, the
            baseline hard prediction, and the baseline soft-vote sums/prediction.
        """
        use_sampling = self.sample_for_splits
        if use_sampling is None:
            use_sampling = X.shape[0] > 50000

        if use_sampling and X.shape[0] > self.sample_size:
            n_samples = min(self.sample_size, X.shape[0])
            sample_indices = self._rng.choice(X.shape[0], n_samples, replace=False)
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]
            cached_memberships = {}
            for feature_idx, fuzzy_var in enumerate(self.fuzzy_partitions_):
                feature_memberships = np.zeros((len(fuzzy_var), X_sample.shape[0]))
                for fz_idx, fuzzy_set in enumerate(fuzzy_var):
                    feature_memberships[fz_idx] = fuzzy_set.membership(X_sample[:, feature_idx])
                cached_memberships[feature_idx] = feature_memberships
        else:
            sample_indices = None
            X_sample = X
            y_sample = y
            cached_memberships = self._get_cached_memberships(X_sample)

        consistent = getattr(self, 'consistent_cci', False) and getattr(self, 'prediction_mode', 'soft') == 'soft'
        skeleton_yhat = self.predict(X_sample)
        ones_mask = np.ones_like(X_sample, dtype=bool)
        if consistent:
            base_votes, base_total = self._predict_proba_all_nodes(
                X_sample, ones_mask, return_votes=True)
            base_pred = self.classes_[np.argmax(base_votes, axis=1)]
        else:
            base_votes = None
            base_pred = None
            base_total = None

        # Coverage-aware growth: mark samples the current tree does not cover
        # (zero total firing) so candidate splits can be rewarded for reaching them.
        if getattr(self, 'coverage_weight', 0.0) > 0.0:
            if base_total is None:
                _, base_total = self._predict_proba_all_nodes(
                    X_sample, ones_mask, return_votes=True)
            uncovered_mask = base_total <= 1e-8
        else:
            uncovered_mask = None

        return {
            'sample_indices': sample_indices,
            'X_sample': X_sample,
            'y_sample': y_sample,
            'cached_memberships': cached_memberships,
            'skeleton_yhat': skeleton_yhat,
            'consistent': consistent,
            'base_votes': base_votes,
            'base_pred': base_pred,
            'uncovered_mask': uncovered_mask,
        }


    def _node_cci_checks(self, node, X: np.array, y: np.array, ctx: dict = None) -> float:
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
        # Shared per-iteration context (baseline prediction / soft-vote sums)
        # computed once per tree scan instead of once per node.
        if ctx is None:
            ctx = self._build_cci_context(X, y)

        n_features = len(self.fuzzy_partitions_)

        # Enforce max_depth: a node at the depth limit cannot be split. Emit a
        # "no split" cache and a sentinel score so it is never selected.
        if node['depth'] >= self.max_depth:
            node['aux_purity_cache'] = {
                'cci': 0.0, 'feature': -1, 'fuzzy_set': -1, 'coverage': 0.0,
                'split_criterion': 0.0, 'child_decision': None, 'purity': 0.0
            }
            return float('-inf'), float('inf')

        # Performance mode: place a data-chosen threshold instead of scanning the
        # fixed partition. Fills aux_purity_cache the same way and returns.
        if self.split_mode == 'learned':
            return self._learned_cci_candidates(node, ctx)

        sample_indices = ctx['sample_indices']
        y_sample = ctx['y_sample']
        cached_memberships = ctx['cached_memberships']
        skeleton_yhat = ctx['skeleton_yhat']
        consistent = ctx['consistent']
        base_votes = ctx['base_votes']
        base_pred = ctx['base_pred']
        uncovered_mask = ctx.get('uncovered_mask')
        coverage_weight = getattr(self, 'coverage_weight', 0.0)
        n_uncovered = int(uncovered_mask.sum()) if uncovered_mask is not None else 0

        if sample_indices is not None:
            existing_membership = node['existing_membership'][sample_indices]
        else:
            existing_membership = node['existing_membership']

        child_decision = node['prediction']

        if self.tree_rules <= 3:
            best_cci = float('-inf')
        else:
            best_cci = float(0.0)
        best_purity = float('inf')
        best_feature = -1
        best_fuzzy_set = -1
        best_coverage = 0.0

        # For debugging, create cache structures that accommodate variable fuzzy set counts
        debug_cache_purity = [np.zeros(len(self.fuzzy_partitions_[i])) for i in range(n_features)]
        debug_cache_cci = [np.zeros(len(self.fuzzy_partitions_[i])) for i in range(n_features)]
        coverage_cache = [np.zeros(len(self.fuzzy_partitions_[i])) for i in range(n_features)]

        # Combine paths: element-wise AND for each feature
        legal_paths = [np.logical_and(node['father_path'][i], node['child_splits'][i]) for i in range(len(node['father_path']))]

        for feature in range(n_features):
            for fz_index in range(len(self.fuzzy_partitions_[feature])):
                if legal_paths[feature][fz_index]:
                    # Use cached membership
                    memberships = cached_memberships[feature][fz_index]
                    full_path_membership = memberships * existing_membership

                    # OPTIMIZATION: Early skip for very low coverage splits (large datasets)
                    coverage = np.sum(full_path_membership) / len(y_sample)
                    if coverage < self.coverage_threshold:
                        continue

                    # OPTIMIZATION: Compute child prediction directly without dummy nodes
                    child_prediction = self._majority_class(y_sample, full_path_membership)

                    if consistent:
                        # Simulate adding this candidate node in the soft-vote space:
                        # add its membership-weighted class distribution and re-argmax.
                        child_probs = self._class_probabilities(y_sample, full_path_membership)
                        new_votes = base_votes + full_path_membership[:, np.newaxis] * child_probs[np.newaxis, :]
                        new_pred = self.classes_[np.argmax(new_votes, axis=1)]
                        cci = compute_fuzzy_cci(y_sample, full_path_membership, base_pred, new_pred, self.coverage_threshold)
                    else:
                        # Legacy: hard override of predictions for samples with
                        # significant membership in the candidate node.
                        skeleton_yhat_child = skeleton_yhat.copy()
                        significant_membership = full_path_membership > 0.01
                        skeleton_yhat_child[significant_membership] = child_prediction
                        cci = compute_fuzzy_cci(y_sample, full_path_membership, skeleton_yhat, skeleton_yhat_child, self.coverage_threshold)
                    purity = compute_fuzzy_purity(full_path_membership, y_sample, self.coverage_threshold)

                    # Coverage-aware growth: reward reaching currently-uncovered
                    # samples (fraction of them this candidate would start firing).
                    if coverage_weight > 0.0 and n_uncovered > 0:
                        coverage_gain = np.mean(full_path_membership[uncovered_mask] > 1e-3)
                        cci = cci + coverage_weight * coverage_gain

                    debug_cache_cci[feature][fz_index] = cci
                    debug_cache_purity[feature][fz_index] = purity
                    coverage_cache[feature][fz_index] = coverage

                    if cci > best_cci:
                        best_cci = cci
                        best_feature = feature
                        best_fuzzy_set = fz_index
                        best_coverage = coverage
                        child_decision = child_prediction
                        best_purity = purity
                    elif cci == best_cci and purity < best_purity:
                        best_cci = cci
                        best_feature = feature
                        best_fuzzy_set = fz_index
                        best_coverage = coverage
                        child_decision = child_prediction
                        best_purity = purity

        if best_feature != -1:
            node['aux_purity_cache'] = {
                'cci': best_cci,
                'feature': best_feature,
                'fuzzy_set': best_fuzzy_set,
                'coverage': best_coverage,
                'split_criterion': best_cci,
                'child_decision': child_decision,
                'purity': best_purity
            }
        else:
            node['aux_purity_cache'] = {
                'cci': 0.0,
                'feature': -1,
                'fuzzy_set': -1,
                'coverage': 0.0,
                'split_criterion': 0.0,
                'child_decision': None,
                'purity': 0.0
            }

        return best_cci, best_purity


    def _learned_cci_candidates(self, node, ctx):
        """Performance-mode split search: for each feature, place a data-chosen
        threshold (weighted-Gini optimal) and score both ramp directions in the
        same soft-vote CCI space the fixed path uses. The ramp half-width is the
        bootstrap std of the cut location (``learned_width='bootstrap'``) or
        ``c * feature-std``. Fills ``node['aux_purity_cache']`` (with the winning
        cut center/half-width) and returns (best_cci, best_purity). Cached per
        node -- a node's data/membership is fixed once created, so its best cut
        never changes; the node is retired after it is split."""
        if node.get('_learned_exhausted'):
            node['aux_purity_cache'] = {'cci': 0.0, 'feature': -1, 'fuzzy_set': -1,
                'coverage': 0.0, 'split_criterion': 0.0, 'child_decision': None, 'purity': 0.0}
            return float('-inf'), float('inf')
        if node.get('_learned_aux') is not None:
            node['aux_purity_cache'] = node['_learned_aux']
            return node['_learned_aux']['cci'], node['_learned_aux']['purity']
        X_sample = ctx['X_sample']
        y_sample = ctx['y_sample']
        sample_indices = ctx['sample_indices']
        base_votes = ctx['base_votes']
        base_pred = ctx['base_pred']
        uncovered_mask = ctx.get('uncovered_mask')
        coverage_weight = getattr(self, 'coverage_weight', 0.0)
        n_uncovered = int(uncovered_mask.sum()) if uncovered_mask is not None else 0

        if sample_indices is not None:
            existing = node['existing_membership'][sample_indices]
        else:
            existing = node['existing_membership']

        n = len(y_sample)
        y_idx = np.array([self._class_to_idx[c] for c in y_sample])
        y_oh = np.eye(len(self.classes_))[y_idx]

        best_cci = float('-inf') if self.tree_rules <= 3 else 0.0
        best_purity = float('inf')
        best_feature, best_center, best_h, best_coverage = -1, 0.0, 0.0, 0.0
        child_decision = node['prediction']

        W = existing.sum()
        # Cut-finding uses only the node's effective region (membership > floor);
        # zero-membership samples cannot change the weighted cut and sorting them
        # is the dominant cost on deep nodes. CCI scoring below stays on all rows,
        # keeping performance mode consistent with the fixed path's criterion (and
        # so with the soft-vote inference and the credal read-out).
        region = existing > 1e-6
        n_eff = int(region.sum())
        if W > 1e-9 and n_eff >= 4:
            xm_all = X_sample[region]
            ym = y_oh[region]
            wm = existing[region]
            Wm = wm.sum()
            pwm = wm / Wm
            p = (ym * wm[:, None]).sum(0) / Wm
            parent = 1.0 - (p ** 2).sum()
            for feature in range(len(self.fuzzy_partitions_)):
                xf = X_sample[:, feature]
                xfm = xm_all[:, feature]
                gain, thr = _learned_best_cut(xfm, ym, wm, parent, Wm)
                if thr is None:
                    continue
                # Ramp half-width = bootstrap std of the cut location (data-driven
                # fuzziness) or c * feature-std.
                if self.learned_width == 'bootstrap':
                    thetas = []
                    for _ in range(self.learned_n_boot):
                        idx = self._rng.choice(n_eff, n_eff, p=pwm)
                        _g, tb = _learned_best_cut(xfm[idx], ym[idx], np.ones(n_eff), 1.0, float(n_eff))
                        if tb is not None:
                            thetas.append(tb)
                    if len(thetas) < 2:
                        continue
                    center, h = float(np.mean(thetas)), float(np.std(thetas))
                else:
                    mean = (wm * xfm).sum() / Wm
                    std = float(np.sqrt((wm * (xfm - mean) ** 2).sum() / Wm))
                    center, h = thr, float(self.learned_width) * std
                h = max(h, 1e-3 * (float(xfm.max() - xfm.min()) + 1e-9))

                # Score both ramp directions in the soft-vote CCI space.
                for direction in ('below', 'above'):
                    full = LearnedRampSet(center, h, direction).membership(xf) * existing
                    coverage = float(np.sum(full)) / n
                    if coverage < self.coverage_threshold:
                        continue
                    child_probs = self._class_probabilities(y_sample, full)
                    new_votes = base_votes + full[:, np.newaxis] * child_probs[np.newaxis, :]
                    new_pred = self.classes_[np.argmax(new_votes, axis=1)]
                    cci = compute_fuzzy_cci(y_sample, full, base_pred, new_pred, self.coverage_threshold)
                    if coverage_weight > 0.0 and n_uncovered > 0:
                        cci = cci + coverage_weight * np.mean(full[uncovered_mask] > 1e-3)
                    purity = compute_fuzzy_purity(full, y_sample, self.coverage_threshold)
                    if cci > best_cci or (cci == best_cci and purity < best_purity):
                        best_cci, best_purity = cci, purity
                        best_feature, best_center, best_h, best_coverage = feature, center, h, coverage
                        child_decision = self._majority_class(y_sample, full)

        if best_feature != -1:
            aux = {
                'cci': best_cci, 'feature': best_feature, 'fuzzy_set': -1,
                'learned': True, 'learned_center': best_center, 'learned_h': best_h,
                'coverage': best_coverage, 'split_criterion': best_cci,
                'child_decision': child_decision, 'purity': best_purity,
            }
        else:
            aux = {
                'cci': 0.0, 'feature': -1, 'fuzzy_set': -1, 'coverage': 0.0,
                'split_criterion': 0.0, 'child_decision': None, 'purity': 0.0,
            }
        node['aux_purity_cache'] = aux
        node['_learned_aux'] = aux
        return best_cci, best_purity


    def _get_best_node_split_cci(self, node_father, X: np.array, y: np.array, ctx: dict = None) -> tuple[float, str]:
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
        if ctx is None:
            ctx = self._build_cci_context(X, y)

        best_cci, best_purity = self._node_cci_checks(node_father, X, y, ctx)
        best_node = node_father['name']

        if 'children' in node_father:
            for child_name, child in node_father['children'].items():
                child_cci, _child_name_bis, child_purity = self._get_best_node_split_cci(child, X, y, ctx)

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

        # Learned (performance) mode: binary split -- add BOTH ramp children
        # (below & above the data-chosen cut) at once and retire the parent.
        if cache.get('learned'):
            self._split_node_learned(node, X, y, cache)
            return

        # Update existing membership and actual path
        existing_membership = node['existing_membership']
        child_existing_membership = existing_membership * self.fuzzy_partitions_[best_feature][best_fuzzy_set].membership(X[:, best_feature])

        # Update parent's child_splits to mark this fuzzy set as used
        node['child_splits'][best_feature][best_fuzzy_set] = False

        # Create child's father_path (copy parent's father_path and mark this fuzzy set as unavailable)
        child_actual_path = [path.copy() for path in node['father_path']]
        child_actual_path[best_feature][best_fuzzy_set] = False

        child_prediction = cache['child_decision']
        # Create fresh child_splits for the new child (all available initially)
        child_new_child_splits = [np.ones(len(fuzzy_var), dtype=bool) for fuzzy_var in self.fuzzy_partitions_]


        # Create children dictionary if not exists
        if 'children' not in node:
            node['children'] = {}
        else:
            self.tree_rules += 1

        # OPTIMIZATION: Cache the path to avoid repeated string parsing
        parent_path = node.get('_cached_path', [])
        cached_path = parent_path + [(best_feature, best_fuzzy_set)]

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
            'class_probabilities': self._class_probabilities(y, child_existing_membership),
            '_cached_path': cached_path  # Cache for fast path retrieval
        }


        # Raise an error if the node name already exists
        if new_node['name'] in self.node_dict_access or new_node['name'] in node['children'].keys():
            raise ValueError(f"Node name {new_node['name']} already exists in the tree.")
        else:
            node['children'][new_node['name']] = new_node

        self.node_dict_access[new_node['name']] = new_node

        # Invalidate leaf cache since tree structure changed
        self._invalidate_leaf_cache()


    def _split_node_learned(self, node, X: np.array, y: np.array, cache):
        """Binary learned split: append the below+above LearnedRampSets at the
        data-chosen cut, create both child nodes, and retire the parent (an
        exhausted node is never re-selected). Each learned set becomes an
        ordinary fuzzy_partitions index resolved by every prediction path."""
        feature = cache['feature']
        center, h = cache['learned_center'], cache['learned_h']
        existing = node['existing_membership']
        if 'children' not in node:
            node['children'] = {}
        parent_path = node.get('_cached_path', [])
        for direction in ('below', 'above'):
            s = LearnedRampSet(center, h, direction)
            self.fuzzy_partitions_[feature].append(s)
            fz = len(self.fuzzy_partitions_[feature]) - 1
            cem = existing * s.membership(X[:, feature])
            name = node['name'] + f"_F{feature}_L{fz}"
            child = {
                'depth': node['depth'] + 1,
                'existing_membership': cem,
                'father_path': [p.copy() for p in node['father_path']],
                'child_splits': [np.ones(len(fv), dtype=bool) for fv in self.fuzzy_partitions_],
                'name': name,
                'prediction': self._majority_class(y, cem),
                'feature': feature,
                'fuzzy_set': fz,
                'coverage': float(np.sum(cem)) / len(y),
                'quality_improvement': cache['split_criterion'],
                'class_probabilities': self._class_probabilities(y, cem),
                '_cached_path': parent_path + [(feature, fz)],
            }
            node['children'][name] = child
            self.node_dict_access[name] = child
        # Two children replace this leaf -> net +1 leaf. Retire the parent.
        self.tree_rules += 1
        node['_learned_exhausted'] = True
        self._invalidate_leaf_cache()


    def _expand_multiway(self, node, X: np.array, y: np.array):
        """Add the just-split feature's remaining fuzzy sets as sibling children,
        turning a one-child split into a full mutually-exclusive partition. Used
        only when ``multiway_splits`` is set (the credal-calibration intervention).
        """
        feat = node['aux_purity_cache']['feature']
        if feat == -1:
            return
        existing = node['existing_membership']
        for fs in range(len(self.fuzzy_partitions_[feat])):
            if not node['child_splits'][feat][fs]:        # already used (best child or prior)
                continue
            name = node['name'] + f"_F{feat}_L{fs}"
            if name in self.node_dict_access:
                continue
            cem = existing * self.fuzzy_partitions_[feat][fs].membership(X[:, feat])
            child_path = [p.copy() for p in node['father_path']]
            child_path[feat][fs] = False
            node['child_splits'][feat][fs] = False
            new_node = {
                'depth': node['depth'] + 1,
                'existing_membership': cem,
                'father_path': child_path,
                'child_splits': [np.ones(len(fv), dtype=bool) for fv in self.fuzzy_partitions_],
                'name': name,
                'prediction': self._majority_class(y, cem),
                'feature': feat,
                'fuzzy_set': fs,
                'coverage': np.sum(cem) / len(y),
                'quality_improvement': 0.0,
                'class_probabilities': self._class_probabilities(y, cem),
                '_cached_path': node.get('_cached_path', []) + [(feat, fs)],
            }
            node['children'][name] = new_node
            self.node_dict_access[name] = new_node
            self.tree_rules += 1
        self._invalidate_leaf_cache()


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
        child_existing_membership = existing_membership * self.fuzzy_partitions_[feature][fuzzy_set].membership(X[:, feature])

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

        This method implements the core FERL algorithm that builds the
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

        # Learned (performance) mode: reset partitions to the fixed seed sets as
        # plain appendable lists; discovered LearnedRampSets are appended here as
        # the tree grows. Resetting each fit drops stale learned sets from a
        # previous fit.
        if self.split_mode == 'learned':
            self._fixed_partitions = copy.deepcopy(self.fuzzy_partitions_)
            self._class_to_idx = {c: i for i, c in enumerate(self.classes_)}

        self._build_root(X, y)
        best_coverage_achievable = 1.0
        bad_cuts = 0

        # OPTIMIZATION: Cache baseline prediction to avoid repeated computation
        baseline_prediction = None

        while self.tree_rules < self.max_rules and best_coverage_achievable >= self.coverage_threshold:
            # OPTIMIZATION: Clear caches and update cached memberships
            self._clear_all_split_caches()
            # Pre-warm the membership cache for this iteration
            self._get_cached_memberships(X)

            # OPTIMIZATION: Only compute predictions when tree structure changes
            if baseline_prediction is None:
                skeleton_prediction, skeleton_memberships, paths = self.predict_with_path(X)
                baseline_prediction = skeleton_prediction.copy()
            else:
                skeleton_prediction = baseline_prediction.copy()

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
                if self.multiway_splits:
                    self._expand_multiway(node_to_split, X, y)
                best_coverage_achievable = self._get_best_possible_coverage(X, y)
                # Invalidate cached prediction since tree structure changed
                baseline_prediction = None

        # Change the prediction in root node to majority class
        #          <self._majority_class(y, skeleton_memberships)>
        self._root['prediction'] = self._majority_class(y)
        # Update root probabilities after tree construction
        self._root['class_probabilities'] = self._class_probabilities(y)
        # print("Final tree built.")

        # OPTIMIZATION: Clear all caches after training to save memory
        self._membership_cache = {}
        self._coverage_cache = {}
        self._gini_cache = {}
        self._prediction_cache = None
        self._last_X_shape = None


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
        if len(y) == 0:
            return self.classes_[0]  # Return first class if no samples

        if membership is None:
            membership = np.ones(len(y))

        # OPTIMIZATION: Vectorized weighted voting using bincount (10-100x faster)
        # Map classes to indices for bincount
        class_to_idx = {cls: idx for idx, cls in enumerate(self.classes_)}
        y_indices = np.array([class_to_idx.get(cls, 0) for cls in y], dtype=np.int32)

        # Use bincount with weights for fast accumulation
        weighted_counts = np.bincount(y_indices, weights=membership, minlength=len(self.classes_))

        return self.classes_[np.argmax(weighted_counts)]

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


    def _get_best_possible_coverage(self, X, y, sample_weight=None):
        """
        Calculate the best possible coverage that could be achieved by adding a new node.

        This method evaluates all possible splits across all features and fuzzy sets
        to find the maximum coverage that any new child node could achieve. This is used
        for early termination - if no possible new node could meet the coverage threshold,
        we can stop splitting.

        Parameters
        ----------
        X : np.array
            Input samples
        y : np.array
            Target values
        sample_weight : np.array, optional
            Sample weights

        Returns
        -------
        float
            Best possible coverage value achievable by any new node
        """
        if len(X) == 0:
            return 0.0

        if sample_weight is None:
            sample_weight = np.ones(len(X))

        max_coverage = 0.0

        # Check all leaf nodes that could potentially be split
        for node_name, node in self.node_dict_access.items():
            if len(node.get('children', {})) == 0:  # This is a leaf node
                # Get samples that reach this node
                node_samples_mask = self._get_node_samples_mask(node, X)
                if not np.any(node_samples_mask):
                    continue

                node_X = X[node_samples_mask]
                node_y = y[node_samples_mask]
                node_weights = sample_weight[node_samples_mask]

                if len(node_X) == 0:
                    continue

                # Check all possible splits for this node
                for feature_idx in range(len(self.fuzzy_partitions_)):
                    fuzzy_sets = self.fuzzy_partitions_[feature_idx]

                    for fuzzy_set_idx in range(len(fuzzy_sets)):
                        # Calculate potential membership for this split
                        feature_values = node_X[:, feature_idx]
                        memberships = fuzzy_sets[fuzzy_set_idx](feature_values)

                        # Calculate coverage as weighted membership sum normalized by total weight
                        weighted_memberships = memberships * node_weights
                        total_weight = np.sum(node_weights)

                        if total_weight > 0:
                            coverage = np.sum(weighted_memberships) / total_weight
                            max_coverage = max(max_coverage, coverage)

        return max_coverage

    def _get_node_samples_mask(self, node, X):
        """
        Get a boolean mask indicating which samples reach a specific node.

        Parameters
        ----------
        node : dict
            The node to check
        X : np.array
            Input samples

        Returns
        -------
        np.array
            Boolean mask indicating which samples reach this node
        """
        if node == self._root:
            return np.ones(len(X), dtype=bool)

        # Get the path from root to this node
        path = self._get_node_path(node)

        # Calculate membership along the path
        membership = np.ones(len(X))
        for feature_idx, fuzzy_set_idx in path:
            if feature_idx < len(self.fuzzy_partitions_):
                fuzzy_sets = self.fuzzy_partitions_[feature_idx]
                if fuzzy_set_idx < len(fuzzy_sets):
                    feature_values = X[:, feature_idx]
                    node_membership = fuzzy_sets[fuzzy_set_idx](feature_values)
                    membership *= node_membership

        # Return samples with non-zero membership (considering floating point precision)
        return membership > 1e-10

    def _get_node_path(self, target_node):
        """
        Get the path from root to a target node.

        Parameters
        ----------
        target_node : dict
            The target node

        Returns
        -------
        list
            List of (feature, fuzzy_set) tuples representing the path from root to target
        """
        if target_node == self._root:
            return []

        # OPTIMIZATION: Use cached path if available (avoids string parsing)
        if '_cached_path' in target_node:
            return target_node['_cached_path']

        # Fallback: Parse the node name (for backward compatibility)
        path = []
        node_name = target_node['name']

        if node_name == 'root':
            return []

        # Split by '_' and process pairs of F{feature}_L{fuzzy_set}
        name_parts = node_name.split('_')

        # Skip 'root' and process remaining parts in pairs
        i = 1
        while i < len(name_parts) - 1:
            if name_parts[i].startswith('F') and name_parts[i+1].startswith('L'):
                feature_idx = int(name_parts[i][1:])  # Remove 'F' prefix
                fuzzy_set_idx = int(name_parts[i+1][1:])  # Remove 'L' prefix
                path.append((feature_idx, fuzzy_set_idx))
                i += 2
            else:
                i += 1

        return path

    def predict(self, X: np.array, observed_mask: np.array=None) -> np.array:
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
        X = self._as_array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if observed_mask is None:
            observed_mask = np.ones_like(X, dtype=bool)

        # Legacy hard winner-take-all over a single node.
        if getattr(self, 'prediction_mode', 'soft') == 'winner':
            prediction, _, _ = self._predict_all_nodes(X, observed_mask)
            return prediction

        # Predict via the aggregated soft fuzzy vote so that predict() is consistent
        # with predict_proba() (argmax of the same probabilities) instead of a
        # hard winner-take-all over a single node.
        proba = self._predict_proba_all_nodes(X, observed_mask)
        return self.classes_[np.argmax(proba, axis=1)]


    def predict_with_path(self, X: np.array, observed_mask: np.array=None) -> tuple[np.array, np.array, np.array]:
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
        X = self._as_array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if observed_mask is None:
            observed_mask = np.ones_like(X, dtype=bool)

        # Use fuzzy membership evaluation across all nodes with full output
        predictions, memberships, paths = self._predict_all_nodes(X, observed_mask)
        return predictions, memberships, paths

    def predict_proba(self, X: np.array, observed_mask: np.array=None) -> np.array:
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
        observed_mask : np.array, optional
            Boolean mask indicating which features are observed (True) vs unobserved (False).
            Shape should be (n_samples, n_features). If None, assumes all features are observed.

        Returns
        -------
        np.array
            Array of shape (n_samples, n_classes) where each row contains the
            probability distribution over classes for the corresponding sample.
            Probabilities sum to 1.0 for each sample.
        """
        X = self._as_array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if observed_mask is None:
            observed_mask = np.ones_like(X, dtype=bool)

        return self._predict_proba_all_nodes(X, observed_mask)

    def firing_strength(self, X: np.array, observed_mask: np.array = None) -> np.array:
        """
        Total rule-firing strength Phi(x) = sum over non-root nodes of path membership.

        This is the (unnormalized) total fuzzy activation a sample receives from the
        rule base. Low values mean the sample lies far from every rule (the model is
        extrapolating) and is a natural fuzzy coverage / epistemic-uncertainty signal,
        e.g. as a conformal nonconformity ingredient.

        Returns
        -------
        np.array
            Per-sample total firing strength, shape (n_samples,).
        """
        X = self._as_array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if observed_mask is None:
            observed_mask = np.ones_like(X, dtype=bool)
        _, total_memberships = self._predict_proba_all_nodes(X, observed_mask, return_votes=True)
        return total_memberships

    def node_activation_matrix(self, X: np.array, observed_mask: np.array = None,
                               membership_floor: float = 0.0):
        """
        Expose the per-node firing matrix used by 'soft' inference.

        Returns (M, consequents, names) where M[i, k] is the path membership of
        non-root node k for sample i, consequents[k] is that node's current
        (MLE) class-probability vector, and names[k] its identifier. Soft
        inference is exactly  P(c|x) = (M @ consequents) / M.sum(1)  (root
        excluded). This lets the node consequents be treated as differentiable
        parameters for post-hoc recalibration while the tree stays frozen.

        ``membership_floor`` (epsilon leak) clamps each per-feature membership to
        [floor, 1] before the product, so bounded-support trapezoids never zero
        out a rule entirely -- a fix for samples that otherwise get zero total
        firing and fall back to the prior.
        """
        X = self._as_array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if observed_mask is None:
            observed_mask = np.ones_like(X, dtype=bool)
        if not hasattr(self, '_cached_all_nodes'):
            self._cached_all_nodes = self._extract_all_nodes()
        nodes = [nd for nd in self._cached_all_nodes if nd['path_length'] > 0]

        N, K, C = X.shape[0], len(nodes), len(self.classes_)
        M = np.zeros((N, K))
        consequents = np.zeros((K, C))
        names = []
        for k, nd in enumerate(nodes):
            pm = np.ones(N)
            for f, fz in zip(nd['path_features'], nd['path_fuzzy_sets']):
                fmem = self.fuzzy_partitions_[f][fz].membership(X[:, f])
                fmem = np.where(observed_mask[:, f], fmem, 1.0 / len(self.fuzzy_partitions_[f]))
                if membership_floor > 0.0:
                    fmem = np.maximum(fmem, membership_floor)
                pm *= fmem
            M[:, k] = pm
            cp = self.node_dict_access[nd['name']].get('class_probabilities')
            consequents[k] = cp if cp is not None and len(cp) == C else np.ones(C) / C
            names.append(nd['name'])
        return M, consequents, names

    def predict_ds(self, X: np.array, observed_mask: np.array = None, leaves_only: bool = False,
                   rule: str = "dempster", reliability_k: float = None, prior_strength: float = None,
                   reliability_vec: np.array = None, top_p: float = None):
        """Combine activated rule nodes as Dempster--Shafer evidence.

        A rule firing with strength ``mu`` commits ``mu * p(c)`` to its class
        consequent and assigns the residual ``1 - mu`` to ignorance. Optional
        support reliability, Dirichlet prior smoothing, and top-p routing move
        additional mass to ignorance.

        Args:
            X: Samples of shape ``(n_samples, n_features)``.
            observed_mask: Boolean feature-observation mask with the same shape
                as ``X``.
            leaves_only: Combine leaf rules only.
            rule: Combination rule. Supported values are ``"dempster"``,
                ``"cautious"``, ``"hybrid"``, ``"incremental"``, and
                ``"incremental_local"``.
            reliability_k: Support pseudo-count used to discount thin rules.
            prior_strength: Dirichlet prior strength for consequent smoothing.
            reliability_vec: Explicit per-node reliability values.
            top_p: Optional nucleus threshold applied to each consequent.

        Returns:
            A tuple ``(betp, belief, plausibility, ignorance)``. The first three
            arrays have shape ``(n_samples, n_classes)`` and ignorance has shape
            ``(n_samples,)``.
        """
        X = self._as_array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        M, cons, names = self.node_activation_matrix(X, observed_mask)
        if leaves_only and M.shape[1] > 0:
            keep = np.array([not self._node_has_children(n) for n in names])
            M, cons = M[:, keep], cons[keep]
            names = [n for n, k in zip(names, keep) if k]

        if top_p is not None and M.shape[1] > 0:
            # top-p (nucleus) routing: keep each node's top classes until the
            # cumulative consequent reaches top_p, route the tail to Theta. The
            # renormalized top-p consequent + a firing discount by the kept mass.
            order = np.argsort(-cons, axis=1)
            sc = np.take_along_axis(cons, order, axis=1)
            before = np.cumsum(sc, axis=1) - sc            # cumulative strictly before each
            kept = np.zeros_like(cons)
            np.put_along_axis(kept, order, np.where(before < top_p, sc, 0.0), axis=1)
            rtp = kept.sum(1)
            cons = kept / np.clip(rtp[:, None], 1e-12, None)
            M = M * rtp[None, :]

        k = self.reliability_k if reliability_k is None else reliability_k
        M_raw = M.copy()                                       # firing only (pre-reliability)
        r_vec = np.ones(M.shape[1])                            # per-node reliability rho_n
        if reliability_vec is not None and M.shape[1] > 0:
            # learned per-node reliability (from finetune_reliability), aligned to
            # node_activation_matrix order; discount firing toward Theta.
            r_vec = np.asarray(reliability_vec, dtype=float)
            M = M * r_vec[None, :]
        elif prior_strength is not None and M.shape[1] > 0:
            # Dirichlet leaf posteriors (DUM): smooth consequents to the posterior
            # mean and discount by support (k = C*a0). Teacher-free epistemic.
            a0, Cc = prior_strength, cons.shape[1]
            support = np.array([self.node_dict_access[n]['coverage'] for n in names]) * self._n_train
            counts = cons * support[:, None]                    # recover c_{n,k}
            cons = (a0 + counts) / (Cc * a0 + support[:, None])  # posterior mean p_bar
            r_vec = support / (Cc * a0 + support)               # leaf reliability = 1 - iota_n
            M = M * r_vec[None, :]
        elif k is not None and M.shape[1] > 0:
            support = np.array([self.node_dict_access[n]['coverage'] for n in names]) * self._n_train
            r_vec = support / (support + k)                     # (K,) reliability per node
            M = M * r_vec[None, :]                              # discount firing toward Theta

        C = len(self.classes_)
        if M.shape[1] == 0:                       # no rules -> total ignorance
            ign = np.ones(X.shape[0])
            betp = np.full((X.shape[0], C), 1.0 / C)
            return betp, np.zeros((X.shape[0], C)), np.ones((X.shape[0], C)), ign

        N = M.shape[0]
        one_minus = 1.0 - M                                    # (N, K)
        term = M[:, :, None] * cons[None, :, :] + one_minus[:, :, None]  # (N, K, C)

        def _cautious_mass(cols):
            """Cautious-combined (m_c, m_theta) over a subset of node columns."""
            w_nc = one_minus[:, cols, None] / np.clip(term[:, cols, :], 1e-12, None)
            w_c = np.clip(w_nc.min(axis=1), 1e-12, 1.0)        # (N, C)
            inv = 1.0 / w_c
            S = (1.0 - C) + inv.sum(axis=1)                    # (N,), >= 1
            return (inv - 1.0) / S[:, None], 1.0 / S

        if rule in ("incremental", "incremental_local"):
            # Incremental (residual) evidence: along each root->leaf chain, keep the
            # root's full belief and replace every descendant by its *residual* over
            # the immediate parent, so the shared ancestor evidence is counted once
            # (no nesting double-count) while the refinement's independent increment
            # is preserved. Residuals are extracted from the unfired, reliability-
            # discounted base masses; firing is applied after.
            #
            # Base singleton+Theta mass per node n: a_n(c) = rho_n p_n(c),
            # t_n = 1 - rho_n; base commonality g_n(c) = a_n(c) + t_n, g_n(Theta)=t_n.
            # Residual of child k over parent p (commonality division):
            #   q_res(c) = g_k(c)/g_p(c),  q_res(Theta) = clip(t_k/t_p, 0, 1)
            #   a_res(c) = q_res(c) - q_res(Theta).
            # Specialization here *raises* ignorance (t_k >= t_p), so the Theta
            # residual is clamped; the validity test is on the class residual:
            # a_res(c) >= 0 for all c means the child reinforces (additive
            # refinement). A negative entry means the child *contradicts* the parent
            # on some class -- an exception, not an increment -- and the node keeps
            # its own full base mass (that link degrades toward Dempster; the parent
            # is not cancelled). So the rule interpolates: clean refinements cancel
            # their ancestor (toward specificity), exceptions double-count (Dempster).
            #
            # Firing variants. 'incremental' (global): each node's mass is discounted
            # by its own full path firing mu_n. 'incremental_local': a residual node
            # is discounted by the *conditional* firing mu_k/mu_parent -- the
            # membership of the newly-added condition alone -- so the parent's firing
            # is not re-counted in the increment (path firing factorizes:
            # mu_{A^B} = mu_A * mu_{B|A}, so the chain's firing telescopes to the
            # leaf's instead of compounding).
            local = (rule == "incremental_local")
            r_inc = r_vec
            if np.allclose(r_inc, 1.0):                         # need t_n>0 to divide
                supp = np.array([self.node_dict_access[n]['coverage'] for n in names]) * self._n_train
                r_inc = supp / (supp + 10.0)                   # default beta=10
            t = np.clip(1.0 - r_inc, 1e-9, 1.0)                # (K,) Theta mass
            a = r_inc[:, None] * cons                          # (K,C) singleton mass
            g = a + t[:, None]                                 # (K,C) base commonality

            # immediate active parent of each node (longest name that is a prefix)
            parent = np.full(M.shape[1], -1, dtype=int)
            for kk, nk in enumerate(names):
                blen = -1
                for j, nj in enumerate(names):
                    if j != kk and nk.startswith(nj + "_") and len(nj) > blen:
                        parent[kk], blen = j, len(nj)

            a_eff = a.copy()                                   # (K,C) effective singleton mass
            t_eff = t.copy()                                   # (K,)  effective Theta mass
            is_residual = np.zeros(M.shape[1], dtype=bool)     # node uses a residual (valid edge)
            n_nonroot = int((parent >= 0).sum())
            n_invalid = 0
            for kk in range(M.shape[1]):
                p = parent[kk]
                if p < 0:                                      # root component: full mass
                    continue
                q_res = g[kk] / np.clip(g[p], 1e-12, None)     # (C,)
                q_res_th = float(np.clip(t[kk] / t[p], 0.0, 1.0))
                a_res = q_res - q_res_th
                if np.all(a_res >= -1e-9):                      # additive refinement
                    a_eff[kk] = np.clip(a_res, 0.0, None)
                    t_eff[kk] = q_res_th
                    is_residual[kk] = True
                else:                                          # exception -> keep full mass
                    n_invalid += 1
            self._last_incremental_diag = {
                'n_nonroot': n_nonroot, 'n_invalid': n_invalid,
                'residual_invalid_rate': (n_invalid / n_nonroot) if n_nonroot else 0.0,
            }
            # per-sample firing per node: full path firing, or (local) the conditional
            # firing mu_k/mu_parent on residual edges.
            mu = M_raw.copy()                                  # (N,K)
            if local and is_residual.any():
                pr = parent[is_residual]
                mu[:, is_residual] = M_raw[:, is_residual] / np.clip(M_raw[:, pr], 1e-12, None)
                mu = np.clip(mu, 0.0, 1.0)
            f_th = 1.0 - mu * (1.0 - t_eff[None, :])           # (N,K) discounted Theta
            f_c = mu[:, :, None] * a_eff[None, :, :] + f_th[:, :, None]  # (N,K,C) commonality
            Qc = np.prod(f_c, axis=1)
            Qt = np.prod(f_th, axis=1)
            m_c_un = np.clip(Qc - Qt[:, None], 0.0, None)
            total = np.where(m_c_un.sum(1) + Qt <= 0, 1.0, m_c_un.sum(1) + Qt)
            m_c, m_theta = m_c_un / total[:, None], Qt / total
        elif rule == "cautious":
            m_c, m_theta = _cautious_mass(np.arange(M.shape[1]))
        elif rule == "hybrid":
            # Structure-aware: cautious within each root->leaf chain (dependent,
            # nested rules), then Dempster across chains (independent branches).
            leaves = [i for i, n in enumerate(names)
                      if not any(o != n and o.startswith(n + "_") for o in names)]
            Qc, Qt = np.ones((N, C)), np.ones(N)
            for li in leaves:
                ln = names[li]
                anc = [j for j, n in enumerate(names) if ln == n or ln.startswith(n + "_")]
                mc, mt = _cautious_mass(anc)                   # per-chain cautious mass
                Qc *= (mc + mt[:, None])                       # Dempster across chains
                Qt *= mt
            m_c_un = np.clip(Qc - Qt[:, None], 0.0, None)
            total = np.where(m_c_un.sum(1) + Qt <= 0, 1.0, m_c_un.sum(1) + Qt)
            m_c, m_theta = m_c_un / total[:, None], Qt / total
        else:                                                  # dempster
            Qc = np.prod(term, axis=1)                         # (N, C)
            Qtheta = np.prod(one_minus, axis=1)                # (N,)
            m_c_un = np.clip(Qc - Qtheta[:, None], 0.0, None)
            total = m_c_un.sum(axis=1) + Qtheta
            total = np.where(total <= 0, 1.0, total)
            m_c = m_c_un / total[:, None]
            m_theta = Qtheta / total

        bel = m_c
        pl = m_c + m_theta[:, None]
        betp = m_c + m_theta[:, None] / C
        return betp, bel, pl, m_theta

    def predict_dirichlet(self, X: np.array, observed_mask: np.array = None,
                          u_floor: float = 1e-3, **ds_kwargs):
        """
        Dirichlet (second-order) distribution per sample, via the Subjective-Logic
        isomorphism of the DS singleton+Theta mass:

            alpha_c = K * Bel(c) / m(Theta) + 1,   S = sum_c alpha_c = K / m(Theta)

        (uniform base rate, prior weight W=K). The Dirichlet mean alpha/S equals the
        pignistic ``betp``; the concentration S is set by the ignorance, so total
        ignorance -> Dir(1,...,1) (uniform). Per-class marginals are
        Beta(alpha_c, S - alpha_c), giving a full distribution -- not just the
        [Bel, Pl] range -- for each class probability.

        ``u_floor`` clamps m(Theta) away from 0 so confident samples give a finite
        (very peaked) Dirichlet instead of an infinite concentration. Extra keyword
        args (e.g. ``reliability_k``, ``prior_strength``, ``rule``) pass through to
        ``predict_ds``. Returns the (n_samples, n_classes) Dirichlet parameters.
        """
        _, bel, _, m_theta = self.predict_ds(X, observed_mask=observed_mask, **ds_kwargs)
        K = bel.shape[1]
        u = np.clip(m_theta, u_floor, 1.0)
        return K * bel / u[:, None] + 1.0

    def predict_credal(self, X: np.array, observed_mask: np.array = None,
                       leaves_only: bool = None, rule: str = "dempster", **kwargs):
        """Return pignistic probabilities, belief, plausibility, and ignorance.

        leaves_only defaults to True for learned-split (deep) FERL and False
        for compact fixed-partition FERL.
        """
        if leaves_only is None:
            leaves_only = self.split_mode == "learned"
        return self.predict_ds(
            X,
            observed_mask=observed_mask,
            leaves_only=leaves_only,
            rule=rule,
            **kwargs,
        )

    def predict_set(self, X: np.array, observed_mask: np.array = None,
                    leaves_only: bool = None, rule: str = "dempster", **kwargs):
        """Return native credal prediction sets as a boolean class mask."""
        _, belief, plausibility, _ = self.predict_credal(
            X,
            observed_mask=observed_mask,
            leaves_only=leaves_only,
            rule=rule,
            **kwargs,
        )
        return plausibility >= belief.max(axis=1, keepdims=True) - 1e-12

    def n_rules(self) -> int:
        """Return the current number of FERL rules."""
        return int(self.tree_rules)


    def _predict_proba_direct_leaves(self, X: np.array, observed_mask: np.array=None) -> np.array:
        """
        Fast probability prediction using direct leaf iteration.

        Computes class probabilities by evaluating membership to all leaves
        and weighting predictions by membership strength.

        Parameters
        ----------
        X : np.array
            Input data array with shape (n_samples, n_features).
        observed_mask : np.array, optional
            Boolean mask indicating which features are observed (True) vs unobserved (False).
            Shape should be (n_samples, n_features). If None, assumes all features are observed.

        Returns
        -------
        np.array
            Probability matrix of shape (n_samples, n_classes).
        """
        n_samples = X.shape[0]

        if observed_mask is None:
            observed_mask = np.ones_like(X, dtype=bool)

        # Get all unique classes from training data
        unique_classes = np.unique([leaf['prediction'] for leaf in self._get_leaves()])
        n_classes = len(unique_classes)
        class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}

        # Initialize probability matrix
        probabilities = np.zeros((n_samples, n_classes))
        total_memberships = np.zeros(n_samples)

        # Get cached leaves
        leaves = self._get_leaves()

        # For each leaf, compute membership and accumulate weighted votes
        for leaf in leaves:
            # Compute path membership for all samples
            path_membership = np.ones(n_samples)

            # Multiply membership along the path
            for feature_idx, fuzzy_set_idx in zip(leaf['path_features'], leaf['path_fuzzy_sets']):
                # Check if feature is observed for each sample
                feature_observed = observed_mask[:, feature_idx]

                fuzzy_set = self.fuzzy_partitions_[feature_idx][fuzzy_set_idx]
                feature_membership = fuzzy_set.membership(X[:, feature_idx])

                # For unobserved features, use uniform membership
                n_partitions = len(self.fuzzy_partitions_[feature_idx])
                uniform_membership = 1.0 / n_partitions
                feature_membership = np.where(feature_observed, feature_membership, uniform_membership)

                path_membership *= feature_membership

            # Add weighted vote for this leaf's prediction
            class_idx = class_to_idx[leaf['prediction']]
            probabilities[:, class_idx] += path_membership
            total_memberships += path_membership

        # Normalize probabilities (handle division by zero)
        for i in range(n_samples):
            if total_memberships[i] > 0:
                probabilities[i] /= total_memberships[i]
            else:
                # Uniform distribution if no membership
                probabilities[i] = 1.0 / n_classes

        return probabilities

    def _get_leaves(self):
        """Get cached leaves, creating cache if necessary."""
        if not hasattr(self, '_cached_leaves'):
            self._cached_leaves = self._extract_leaves()
        return self._cached_leaves

    def predict_all_leaves(self, X: np.array, observed_mask: np.array=None) -> tuple[dict, dict]:
        """
        Get membership values and predictions for all leaf nodes for each sample.

        This method computes the fuzzy membership degree of each sample to every
        leaf node in the tree, along with each leaf's prediction. This provides
        a complete picture of how samples relate to all possible decision paths.

        Parameters
        ----------
        X : np.array
            Data to predict. Each row is a sample.
        observed_mask : np.array, optional
            Boolean mask indicating which features are observed (True) vs unobserved (False).
            Shape should be (n_samples, n_features). If None, assumes all features are observed.

        Returns
        -------
        tuple[dict, dict]
            Two dictionaries:
            - memberships_dict: {leaf_name: np.array of memberships for each sample}
            - predictions_dict: {leaf_name: prediction_class}
        """
        X = self._as_array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if observed_mask is None:
            observed_mask = np.ones_like(X, dtype=bool)

        # Get all leaf nodes
        leaf_nodes = self._get_all_leaf_nodes()

        # Initialize results
        memberships_dict = {}
        predictions_dict = {}

        # Calculate membership to each leaf for each sample
        for leaf_name, leaf_node in leaf_nodes.items():
            memberships = self._calculate_membership_to_leaf(X, leaf_node, observed_mask)
            memberships_dict[leaf_name] = memberships
            predictions_dict[leaf_name] = leaf_node['prediction']

        return memberships_dict, predictions_dict

    def predict_all_leaves_matrix(self, X: np.array, observed_mask: np.array=None) -> tuple[np.array, np.array, list]:
        """
        Get memberships and predictions for all leaves in matrix format.

        This is a convenience method that returns the same information as
        predict_all_leaves but in matrix format for easier analysis.

        Parameters
        ----------
        X : np.array
            Data to predict. Each row is a sample.
        observed_mask : np.array, optional
            Boolean mask indicating which features are observed (True) vs unobserved (False).
            Shape should be (n_samples, n_features). If None, assumes all features are observed.

        Returns
        -------
        tuple[np.array, np.array, list]
            - membership_matrix: (n_samples, n_leaves) matrix of memberships
            - predictions_array: (n_leaves,) array of leaf predictions
            - leaf_names: list of leaf node names in same order as columns
        """
        memberships_dict, predictions_dict = self.predict_all_leaves(X, observed_mask)

        leaf_names = list(memberships_dict.keys())
        n_samples = X.shape[0] if X.ndim > 1 else 1
        n_leaves = len(leaf_names)

        # Create membership matrix
        membership_matrix = np.zeros((n_samples, n_leaves))
        predictions_array = np.zeros(n_leaves, dtype=int)

        for i, leaf_name in enumerate(leaf_names):
            membership_matrix[:, i] = memberships_dict[leaf_name]
            predictions_array[i] = predictions_dict[leaf_name]

        return membership_matrix, predictions_array, leaf_names


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
            child_path_membership = self.fuzzy_partitions_[relevant_feature][relevant_fuzzy_set].membership(x[:, relevant_feature])
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


    def _extract_leaves(self) -> list:
        """
        Extract all leaf nodes from the tree for direct iteration.

        This method traverses the tree once to collect all leaf nodes,
        allowing for direct iteration instead of recursive traversal
        during prediction, which significantly improves prediction speed.

        Returns
        -------
        list
            List of leaf node dictionaries with their paths and predictions.
        """
        leaves = []

        def collect_leaves(node, path_features=None, path_fuzzy_sets=None, path_name=""):
            if path_features is None:
                path_features = []
                path_fuzzy_sets = []

            # If no children, this is a leaf
            if not node.get('children', False) or len(node['children']) == 0:
                leaves.append({
                    'prediction': node['prediction'],
                    'name': node['name'] if node['name'] != 'root' else path_name,
                    'path_features': path_features.copy(),
                    'path_fuzzy_sets': path_fuzzy_sets.copy(),
                    'path_length': len(path_features)
                })
            else:
                # Recursively collect from children
                for child_name, child in node['children'].items():
                    new_path_features = path_features + [child['feature']]
                    new_path_fuzzy_sets = path_fuzzy_sets + [child['fuzzy_set']]
                    new_path_name = child_name if path_name == "" else f"{path_name}->{child_name}"

                    collect_leaves(child, new_path_features, new_path_fuzzy_sets, new_path_name)

        collect_leaves(self._root)

        # Sort leaves by path length (shorter paths first for efficiency)
        leaves.sort(key=lambda x: x['path_length'])

        return leaves

    def _predict_direct_leaves(self, X: np.array) -> tuple[np.array, np.array, np.array]:
        """
        Fast prediction using direct leaf iteration instead of recursion.

        This method iterates through all leaf nodes directly and computes
        membership for each sample to each leaf, selecting the leaf with
        the highest membership. This approach is significantly faster than
        recursive tree traversal, especially for deep trees.

        Parameters
        ----------
        X : np.array
            Input data array with shape (n_samples, n_features).

        Returns
        -------
        tuple[np.array, np.array, np.array]
            Predictions, membership values, and path names for all samples.
        """
        n_samples = X.shape[0]

        # Initialize output arrays
        predictions = np.full(n_samples, self._root['prediction'])
        best_memberships = np.zeros(n_samples)
        paths = np.full(n_samples, 'root', dtype=object)

        # Get all leaves if not cached
        if not hasattr(self, '_cached_leaves'):
            self._cached_leaves = self._extract_leaves()

        # Handle root-only case
        if not self._cached_leaves:
            return predictions, best_memberships, paths

        # For each leaf, compute membership for all samples
        for leaf in self._cached_leaves:
            # Compute path membership for all samples
            path_membership = np.ones(n_samples)

            # Multiply membership along the path
            for feature_idx, fuzzy_set_idx in zip(leaf['path_features'], leaf['path_fuzzy_sets']):
                fuzzy_set = self.fuzzy_partitions_[feature_idx][fuzzy_set_idx]
                feature_membership = fuzzy_set.membership(X[:, feature_idx])
                path_membership *= feature_membership

            # Update best predictions where this leaf has higher membership
            better_samples = path_membership > best_memberships

            predictions[better_samples] = leaf['prediction']
            best_memberships[better_samples] = path_membership[better_samples]
            paths[better_samples] = leaf['name']

        return predictions, best_memberships, paths

    def _invalidate_leaf_cache(self):
        """
        Invalidate the cached leaf nodes when the tree structure changes.

        This should be called whenever a new node is added to the tree
        to ensure the leaf cache is updated for the next prediction.
        """
        if hasattr(self, '_cached_leaves'):
            delattr(self, '_cached_leaves')
        if hasattr(self, '_cached_all_nodes'):
            delattr(self, '_cached_all_nodes')

    def _extract_all_nodes(self) -> list:
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
            if path_features is None:
                path_features = []
                path_fuzzy_sets = []

            # Add current node (use actual node name for lookups)
            actual_node_name = node['name']

            all_nodes.append({
                'prediction': node['prediction'],
                'name': actual_node_name,  # Use actual node name, not display name
                'path_features': path_features.copy(),
                'path_fuzzy_sets': path_fuzzy_sets.copy(),
                'path_length': len(path_features)
            })

            # Recursively collect from children
            if node.get('children', False) and len(node['children']) > 0:
                for child_name, child in node['children'].items():
                    new_path_features = path_features + [child['feature']]
                    new_path_fuzzy_sets = path_fuzzy_sets + [child['fuzzy_set']]

                    collect_nodes(child, new_path_features, new_path_fuzzy_sets)

        collect_nodes(self._root)

        # Sort nodes by path length (shorter paths first for efficiency)
        all_nodes.sort(key=lambda x: x['path_length'])

        return all_nodes


    def _predict_all_nodes(self, X: np.array, observed_mask: np.array, epsilon: float = 1e-6) -> tuple[np.array, np.array, np.array]:
        """
        Fuzzy prediction using ALL nodes in the tree, with proper internal node constraints.

        This method implements the correct fuzzy decision tree semantics: internal nodes
        can only be used for prediction when ALL their children have membership ≤ epsilon.
        This ensures children are preferred when they have meaningful membership, while
        internal nodes serve as fallback predictions.

        Parameters
        ----------
        X : np.array
            Input data array with shape (n_samples, n_features).
        observed_mask : np.array
            Boolean mask indicating which features are observed for each sample.
        epsilon : float, default=1e-6
            Threshold below which child membership is considered zero.

        Returns
        -------
        tuple[np.array, np.array, np.array]
            Predictions, membership values, and path names for all samples.
        """
        n_samples = X.shape[0]

        # Initialize output arrays - start with invalid values, not root defaults
        predictions = np.full(n_samples, -1)  # Invalid prediction initially
        best_memberships = np.full(n_samples, -1.0)  # Invalid membership initially
        paths = np.full(n_samples, '', dtype=object)  # Empty path initially

        # Get all nodes if not cached
        if not hasattr(self, '_cached_all_nodes'):
            self._cached_all_nodes = self._extract_all_nodes()

        # First pass: collect all node memberships
        node_memberships = {}

        for node in self._cached_all_nodes:
            # Compute path membership for all samples
            if node['path_length'] == 0:
                # Root node
                path_membership = np.ones(n_samples)
            else:
                path_membership = np.ones(n_samples)
                # Multiply membership along the path
                for feature_idx, fuzzy_set_idx in zip(node['path_features'], node['path_fuzzy_sets']):
                    fuzzy_set = self.fuzzy_partitions_[feature_idx][fuzzy_set_idx]
                    feature_membership = fuzzy_set.membership(X[:, feature_idx])
                    feature_membership = np.where(observed_mask[:, feature_idx], feature_membership, 1.0)
                    path_membership *= feature_membership

            node_memberships[node['name']] = {
                'membership': path_membership,
                'prediction': node['prediction'],
                'path_length': node['path_length']
            }

        # Second pass: apply predictions with internal node constraints
        # Process leaf nodes first (deepest first), then internal nodes (shallowest last)
        node_items = list(node_memberships.items())

        # Separate leaf and internal nodes
        leaf_nodes = []
        internal_nodes = []

        for node_name, node_data in node_items:
            if self._node_has_children(node_name):
                internal_nodes.append((node_name, node_data))
            else:
                leaf_nodes.append((node_name, node_data))

        # Sort internal nodes by path length (deepest first, then root last)
        internal_nodes.sort(key=lambda x: x[1]['path_length'], reverse=True)

        # Process leaf nodes first
        for node_name, node_data in leaf_nodes:
            current_membership = node_data['membership']
            current_prediction = node_data['prediction']

            # Leaf nodes can always be considered - update where membership is better
            better_samples = current_membership > best_memberships

            predictions[better_samples] = current_prediction
            best_memberships[better_samples] = current_membership[better_samples]
            paths[better_samples] = node_name

        # Then process internal nodes with constraints
        for node_name, node_data in internal_nodes:
            current_membership = node_data['membership']
            current_prediction = node_data['prediction']

            # For internal nodes, check if all children have membership ≤ epsilon
            children_names = self._get_node_children_names(node_name)

            # For each sample, check if ALL children have low membership
            can_use_internal = np.ones(n_samples, dtype=bool)

            for child_name in children_names:
                if child_name in node_memberships:
                    child_membership = node_memberships[child_name]['membership']
                    # If any child has membership > epsilon, can't use internal node for those samples
                    high_child_membership = child_membership > epsilon
                    can_use_internal = can_use_internal & (~high_child_membership)

            # Only consider internal node for samples where all children have low membership
            valid_membership = np.where(can_use_internal, current_membership, 0.0)

            # Update predictions where this internal node has higher valid membership
            better_samples = valid_membership > best_memberships

            predictions[better_samples] = current_prediction
            best_memberships[better_samples] = valid_membership[better_samples]
            paths[better_samples] = node_name

        # Handle any samples that still have no prediction (fallback to root)
        no_prediction = best_memberships < 0
        if np.any(no_prediction):
            predictions[no_prediction] = self._root['prediction']
            best_memberships[no_prediction] = 1.0
            paths[no_prediction] = 'root'

        return predictions, best_memberships, paths


    def _node_has_children(self, node_name: str) -> bool:
        """
        Check if a node has children (is an internal node).

        Parameters
        ----------
        node_name : str
            Name of the node to check.

        Returns
        -------
        bool
            True if the node has children, False otherwise.
        """
        if node_name not in self.node_dict_access:
            return False

        node = self.node_dict_access[node_name]
        return 'children' in node and len(node['children']) > 0

    def _get_node_children_names(self, node_name: str) -> list:
        """
        Get the names of all direct children of a node.

        Parameters
        ----------
        node_name : str
            Name of the parent node.

        Returns
        -------
        list
            List of child node names.
        """
        if node_name not in self.node_dict_access:
            return []

        node = self.node_dict_access[node_name]
        if 'children' not in node:
            return []

        return list(node['children'].keys())

    def _predict_proba_all_nodes(self, X: np.array, observed_mask: np.array, epsilon: float = 1e-6,
                                 return_votes: bool = False) -> np.array:
        """
        Predict class probabilities using fuzzy membership across ALL nodes with internal node constraints.

        This method computes probability distributions by evaluating membership
        to all nodes in the tree. Internal nodes can only contribute when ALL their
        children have membership ≤ epsilon, ensuring proper fuzzy tree semantics.

        Parameters
        ----------
        X : np.array
            Input data array with shape (n_samples, n_features).
        observed_mask : np.array
            Boolean mask indicating which features are observed (True) vs unobserved (False).
            Shape should be (n_samples, n_features).
        epsilon : float, default=1e-6
            Threshold below which child membership is considered zero.

        Returns
        -------
        np.array
            Array of shape (n_samples, n_classes) with probability distributions.
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        # Get all nodes if not cached
        if not hasattr(self, '_cached_all_nodes'):
            self._cached_all_nodes = self._extract_all_nodes()

        # Initialize probability accumulator and total membership
        class_memberships = np.zeros((n_samples, n_classes))
        total_memberships = np.zeros(n_samples)

        # First pass: collect all node memberships
        node_memberships = {}

        for node in self._cached_all_nodes:
            # Compute path membership for all samples
            if node['path_length'] == 0:
                # Root node has membership 1.0 for all samples
                path_membership = np.ones(n_samples)
            else:
                path_membership = np.ones(n_samples)
                # Multiply membership along the path
                for feature_idx, fuzzy_set_idx in zip(node['path_features'], node['path_fuzzy_sets']):
                    # Check if feature is observed for each sample
                    feature_observed = observed_mask[:, feature_idx]

                    fuzzy_set = self.fuzzy_partitions_[feature_idx][fuzzy_set_idx]
                    feature_membership = fuzzy_set.membership(X[:, feature_idx])

                    # For unobserved features, use uniform membership (0.5 for binary, 1.0/n_partitions for multi)
                    # This represents maximum uncertainty
                    n_partitions = len(self.fuzzy_partitions_[feature_idx])
                    uniform_membership = 1.0 / n_partitions
                    feature_membership = np.where(feature_observed, feature_membership, uniform_membership)

                    path_membership *= feature_membership

            node_memberships[node['name']] = {
                'membership': path_membership,
                'prediction': node['prediction'],
                'path_length': node['path_length']
            }

        # Second pass: apply memberships with internal node constraints
        # Process nodes in REVERSE order (longest paths first) so children are processed before parents
        node_items = list(node_memberships.items())
        node_items.sort(key=lambda x: x[1]['prediction'], reverse=False)  # Just to have consistent ordering

        mode = getattr(self, 'prediction_mode', 'soft_gate')
        gate_internal = mode in ('soft_gate', 'hard_gate')
        use_soft = mode in ('soft_gate', 'soft')

        for node_name, node_data in node_items:
            current_membership = node_data['membership']
            current_prediction = node_data['prediction']

            # Check if this is an internal node (has children)
            is_internal_node = self._node_has_children(node_name)

            if is_internal_node and gate_internal:
                # For internal nodes, check if all children have membership ≤ epsilon
                children_names = self._get_node_children_names(node_name)

                # For each sample, check if ALL children have low membership
                can_use_internal = np.ones(n_samples, dtype=bool)

                for child_name in children_names:
                    if child_name in node_memberships:
                        child_membership = node_memberships[child_name]['membership']
                        # If any child has membership > epsilon, can't use internal node for those samples
                        high_child_membership = child_membership > epsilon
                        can_use_internal = can_use_internal & (~high_child_membership)

                # Only consider internal node for samples where all children have low membership
                valid_membership = np.where(can_use_internal, current_membership, 0.0)
            else:
                # Leaf nodes (or, in non-gated modes, every node) always considered
                valid_membership = current_membership

            # In non-gated 'soft' mode the root spans all samples and would wash
            # everything toward the class prior, so it is excluded.
            if not gate_internal and node_data.get('path_length', 1) == 0:
                continue

            # Soft modes weight each node's full class-probability vector by
            # membership; hard modes cast a one-hot vote for the node's class.
            if use_soft:
                node = self.node_dict_access.get(node_name)
                node_probs = node.get('class_probabilities') if node is not None else None
                if node_probs is not None and len(node_probs) == n_classes:
                    class_memberships += valid_membership[:, np.newaxis] * node_probs[np.newaxis, :]
                    total_memberships += valid_membership
                continue

            # Accumulate membership for this node's prediction class.
            # Nodes without a valid class (e.g. the root during construction,
            # whose prediction is -1) contribute no vote.
            class_match = np.where(self.classes_ == current_prediction)[0]
            if len(class_match) == 0:
                continue
            class_idx = class_match[0]
            class_memberships[:, class_idx] += valid_membership
            total_memberships += valid_membership

        # Return the unnormalized vote accumulators (used by consistent CCI
        # scoring, which simulates adding a candidate node in the same additive
        # vote space that inference uses).
        if return_votes:
            return class_memberships, total_memberships

        # Normalize to get probabilities
        # Avoid division by zero
        nonzero_total = total_memberships > 0
        probabilities = np.zeros((n_samples, n_classes))
        probabilities[nonzero_total] = class_memberships[nonzero_total] / total_memberships[nonzero_total, np.newaxis]

        # For samples with zero total membership, use uniform distribution
        zero_total = total_memberships == 0
        probabilities[zero_total] = 1.0 / n_classes

        return probabilities
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

    def _get_all_leaf_nodes(self) -> dict:
        """
        Get all leaf nodes in the tree.

        Returns
        -------
        dict
            Dictionary of {leaf_name: leaf_node} for all leaf nodes.
        """
        leaf_nodes = {}

        def traverse(node):
            if 'children' not in node or not node['children']:
                # This is a leaf node
                leaf_nodes[node['name']] = node
            else:
                # Traverse children
                for child in node['children'].values():
                    traverse(child)

        traverse(self._root)
        return leaf_nodes

    def _calculate_membership_to_leaf(self, X: np.array, leaf_node: dict, observed_mask: np.array) -> np.array:
        """
        Calculate fuzzy membership of samples to a specific leaf node.

        This method traces the path from root to the specified leaf and computes
        the combined membership by multiplying memberships at each step.

        Parameters
        ----------
        X : np.array
            Input samples with shape (n_samples, n_features).
        leaf_node : dict
            The target leaf node.
        observed_mask : np.array
            Boolean mask indicating which features are observed (True) vs unobserved (False).
            Shape should be (n_samples, n_features).

        Returns
        -------
        np.array
            Membership values for each sample to this leaf node.
        """
        # Get the path from root to leaf
        path_to_leaf = self._get_path_to_leaf(leaf_node)

        # Start with full membership at root
        membership = np.ones(X.shape[0])

        # OPTIMIZATION: Use cached memberships when available
        try:
            cached_memberships = self._get_cached_memberships(X)
            use_cache = True
        except:
            use_cache = False

        # Apply each step in the path
        for step in path_to_leaf:
            if step['type'] == 'split':
                feature_idx = step['feature']
                fuzzy_set_idx = step['fuzzy_set']

                # Check if feature is observed for each sample
                feature_observed = observed_mask[:, feature_idx]

                if use_cache:
                    step_membership = cached_memberships[feature_idx][fuzzy_set_idx]
                else:
                    fuzzy_set = self.fuzzy_partitions_[feature_idx][fuzzy_set_idx]
                    step_membership = fuzzy_set.membership(X[:, feature_idx])

                # For unobserved features, use uniform membership
                n_partitions = len(self.fuzzy_partitions_[feature_idx])
                uniform_membership = 1.0 / n_partitions
                step_membership = np.where(feature_observed, step_membership, uniform_membership)

                membership = membership * step_membership

        return membership

    def _get_path_to_leaf(self, leaf_node: dict) -> list:
        """
        Get the sequence of splits from root to a leaf node.

        Parameters
        ----------
        leaf_node : dict
            The target leaf node.

        Returns
        -------
        list
            List of dictionaries describing each split step.
        """
        # Reconstruct path by analyzing node name
        node_name = leaf_node['name']

        if node_name == 'root':
            return []  # Root has no path

        # Parse node name to extract path
        # Format: root_F0_L1_F2_L0 means: feature 0, fuzzy set 1, then feature 2, fuzzy set 0
        path_steps = []
        parts = node_name.split('_')

        i = 1  # Skip 'root'
        while i < len(parts):
            if parts[i].startswith('F') and i + 1 < len(parts) and parts[i + 1].startswith('L'):
                feature_idx = int(parts[i][1:])  # Remove 'F' prefix
                fuzzy_set_idx = int(parts[i + 1][1:])  # Remove 'L' prefix

                path_steps.append({
                    'type': 'split',
                    'feature': feature_idx,
                    'fuzzy_set': fuzzy_set_idx
                })

                i += 2  # Skip both F and L parts
            else:
                i += 1

        return path_steps

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
            print(f"FERL tree (max_rules={self.max_rules}, coverage_threshold={self.coverage_threshold})")
            print("=" * 60)

        # Print current node
        current_prefix = "└── " if is_last else "├── "

        if node['name'] == 'root':
            print(f"{prefix}{current_prefix}Root: class={node['prediction']}, coverage={node['coverage']:.3f}")
        else:
            feature_name = self.fuzzy_partitions_[node['feature']].name
            fuzzy_set_name = self.fuzzy_partitions_[node['feature']][node['fuzzy_set']].name

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



__all__ = ["FERL", "LearnedRampSet", "compute_fuzzy_cci", "compute_fuzzy_purity"]
