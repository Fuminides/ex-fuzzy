"""
Conformal Prediction for Fuzzy Rule-Based Classifiers

Provides prediction sets with coverage guarantees using split conformal prediction.
Supports both class-level and rule-level conformal prediction for explainable uncertainty.

Main Components:
    - ConformalFuzzyClassifier: Main classifier with conformal prediction
    - evaluate_conformal_coverage: Evaluation utility for coverage metrics

Key Features:
    - Wrap existing trained classifiers or create from scratch
    - Class-level conformal prediction (standard)
    - Rule-wise conformal prediction (explainable uncertainty)
    - Multiple nonconformity score strategies

Example:
    >>> from ex_fuzzy.conformal import ConformalFuzzyClassifier
    >>> # Wrap existing classifier
    >>> conf_clf = ConformalFuzzyClassifier(trained_clf)
    >>> conf_clf.calibrate(X_cal, y_cal)
    >>> pred_sets = conf_clf.predict_set(X_test, alpha=0.1)

    >>> # Or create from scratch
    >>> conf_clf = ConformalFuzzyClassifier(nRules=20, nAnts=4)
    >>> conf_clf.fit(X_train, y_train, X_cal, y_cal, n_gen=50)
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Union, Optional, List, Dict

try:
    from . import fuzzy_sets as fs
    from . import evolutionary_fit as evf
    from . import rules
except ImportError:
    import fuzzy_sets as fs
    import evolutionary_fit as evf
    import rules


class ConformalFuzzyClassifier(ClassifierMixin, BaseEstimator):
    """
    Conformal prediction wrapper for fuzzy rule-based classifiers.

    Can wrap an existing BaseFuzzyRulesClassifier or create one internally
    using the same parameter signature. Provides prediction sets with
    statistical coverage guarantees.

    Parameters
    ----------
    clf_or_nRules : BaseFuzzyRulesClassifier or int, optional
        Either a trained classifier to wrap, or nRules parameter to create new one.
        If None, uses default nRules=30.
    score_type : str, default='membership'
        Nonconformity score type:
        - 'membership': 1 - membership degree of true class (default)
        - 'association': 1 - max association degree for rules of true class
        - 'entropy': Entropy of class probability distribution
    **kwargs : dict
        Additional parameters passed to BaseFuzzyRulesClassifier if creating new.
        Common parameters include: nAnts, fuzzy_type, n_linguistic_variables,
        tolerance, verbose, backend, etc.

    Attributes
    ----------
    clf : BaseFuzzyRulesClassifier
        The underlying fuzzy classifier
    score_type : str
        The type of nonconformity score being used
    rule_base : MasterRuleBase
        The rule base from the underlying classifier (property)
    nclasses_ : int
        Number of classes (property)

    Examples
    --------
    Wrap an existing trained classifier:

    >>> from ex_fuzzy.evolutionary_fit import BaseFuzzyRulesClassifier
    >>> from ex_fuzzy.conformal import ConformalFuzzyClassifier
    >>>
    >>> clf = BaseFuzzyRulesClassifier(nRules=20)
    >>> clf.fit(X_train, y_train, n_gen=50)
    >>>
    >>> conf_clf = ConformalFuzzyClassifier(clf)
    >>> conf_clf.calibrate(X_cal, y_cal)
    >>> pred_sets = conf_clf.predict_set(X_test, alpha=0.1)

    Create classifier from scratch:

    >>> conf_clf = ConformalFuzzyClassifier(nRules=20, nAnts=4)
    >>> conf_clf.fit(X_train, y_train, X_cal, y_cal, n_gen=50)
    >>> pred_sets = conf_clf.predict_set(X_test, alpha=0.1)

    Get rule-wise explanations:

    >>> results = conf_clf.predict_set_with_rules(X_test, alpha=0.1)
    >>> for r in results:
    ...     print(f"Prediction set: {r['prediction_set']}")
    ...     print(f"Rule contributions: {r['rule_contributions']}")
    """

    def __init__(self, clf_or_nRules: Union[evf.BaseFuzzyRulesClassifier, int, None] = None,
                 score_type: str = 'membership',
                 **kwargs):

        if isinstance(clf_or_nRules, evf.BaseFuzzyRulesClassifier):
            # Wrapper mode - use existing classifier
            self.clf = clf_or_nRules
            self._owns_clf = False
        else:
            # Creation mode - create new classifier
            nRules = clf_or_nRules if clf_or_nRules is not None else kwargs.pop('nRules', 30)
            self.clf = evf.BaseFuzzyRulesClassifier(nRules=nRules, **kwargs)
            self._owns_clf = True

        if score_type not in ('membership', 'association', 'entropy'):
            raise ValueError(f"Unknown score_type: {score_type}. "
                           f"Expected one of: 'membership', 'association', 'entropy'")
        self.score_type = score_type
        self._calibrated = False
        self._calibration_scores = None
        self._thresholds = None  # Per-class thresholds
        self._rule_calibration = None  # Per-rule thresholds (for rule-wise)

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_cal: np.ndarray = None, y_cal: np.ndarray = None,
            cal_size: float = 0.2,
            **fit_kwargs) -> 'ConformalFuzzyClassifier':
        """
        Fit the classifier and calibrate for conformal prediction.

        If X_cal/y_cal not provided, automatically splits X/y using cal_size.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Training labels
        X_cal : array-like of shape (n_cal_samples, n_features), optional
            Calibration data. If None, split from X using cal_size.
        y_cal : array-like of shape (n_cal_samples,), optional
            Calibration labels. If None, split from y using cal_size.
        cal_size : float, default=0.2
            Fraction of data to use for calibration if X_cal not provided
        **fit_kwargs : dict
            Parameters passed to clf.fit() (n_gen, pop_size, etc.)

        Returns
        -------
        self : ConformalFuzzyClassifier
            The fitted conformal classifier

        Examples
        --------
        >>> conf_clf = ConformalFuzzyClassifier(nRules=20)
        >>> # Auto-split for calibration
        >>> conf_clf.fit(X, y, cal_size=0.2, n_gen=50)

        >>> # Or provide explicit calibration set
        >>> conf_clf.fit(X_train, y_train, X_cal, y_cal, n_gen=50)
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X_cal is None:
            # Split data for calibration
            X, X_cal, y, y_cal = self._split_calibration(X, y, cal_size)
        else:
            X_cal = np.asarray(X_cal)
            y_cal = np.asarray(y_cal)

        if self._owns_clf:
            self.clf.fit(X, y, **fit_kwargs)

        self.calibrate(X_cal, y_cal)
        return self

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> 'ConformalFuzzyClassifier':
        """
        Calibrate conformal prediction thresholds using calibration set.

        Computes nonconformity scores on the calibration set and stores
        them for computing p-values during prediction.

        Parameters
        ----------
        X_cal : array-like of shape (n_cal_samples, n_features)
            Calibration samples
        y_cal : array-like of shape (n_cal_samples,)
            Calibration labels

        Returns
        -------
        self : ConformalFuzzyClassifier
            The calibrated conformal classifier

        Raises
        ------
        ValueError
            If the underlying classifier has not been fitted yet

        Examples
        --------
        >>> # After fitting the base classifier separately
        >>> conf_clf = ConformalFuzzyClassifier(trained_clf)
        >>> conf_clf.calibrate(X_cal, y_cal)
        """
        X_cal = np.asarray(X_cal)
        y_cal = np.asarray(y_cal)

        # Check that the classifier is fitted
        if not hasattr(self.clf, 'rule_base') or self.clf.rule_base is None:
            raise ValueError("Classifier not fitted. Either wrap a trained classifier "
                           "or use fit() instead of calibrate().")

        # Compute nonconformity scores for calibration samples
        scores = self._compute_nonconformity_scores(X_cal, y_cal)

        # Store scores grouped by true class
        self._calibration_scores = {}
        unique_classes = np.unique(y_cal)
        for class_idx in unique_classes:
            mask = y_cal == class_idx
            self._calibration_scores[int(class_idx)] = scores[mask]

        # Compute rule-wise calibration if applicable
        self._calibrate_rules(X_cal, y_cal)

        self._calibrated = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Standard point prediction (delegates to wrapped classifier).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Predicted class labels
        """
        return self.clf.predict(X)

    def predict_set(self, X: np.ndarray, alpha: float = 0.1) -> List[set]:
        """
        Predict conformal sets with coverage guarantee 1-alpha.

        Returns prediction sets that, under exchangeability assumptions,
        contain the true label with probability at least 1-alpha.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        alpha : float, default=0.1
            Significance level. The coverage guarantee is 1-alpha.
            For example, alpha=0.1 gives 90% coverage.

        Returns
        -------
        prediction_sets : list of sets
            For each sample, a set of class indices in the prediction set.
            Empty sets indicate high uncertainty for all classes.
            Sets with multiple classes indicate ambiguous predictions.

        Raises
        ------
        ValueError
            If the classifier is not calibrated

        Examples
        --------
        >>> pred_sets = conf_clf.predict_set(X_test, alpha=0.1)
        >>> for i, pred_set in enumerate(pred_sets):
        ...     if len(pred_set) == 1:
        ...         print(f"Sample {i}: confident prediction {pred_set}")
        ...     elif len(pred_set) > 1:
        ...         print(f"Sample {i}: ambiguous between {pred_set}")
        ...     else:
        ...         print(f"Sample {i}: high uncertainty (empty set)")
        """
        if not self._calibrated:
            raise ValueError("Classifier not calibrated. Call calibrate() or fit() first.")

        X = np.asarray(X)
        n_samples = X.shape[0]
        n_classes = self.clf.nclasses_

        # Get soft scores for all classes
        class_scores = self.clf.predict_membership_class(X)  # (n_samples, n_classes)

        prediction_sets = []
        for i in range(n_samples):
            pred_set = set()
            for c in range(n_classes):
                # Compute p-value for class c
                score = 1 - class_scores[i, c]  # Nonconformity: 1 - membership
                p_value = self._compute_p_value(score, c)
                if p_value > alpha:
                    pred_set.add(c)
            prediction_sets.append(pred_set)

        return prediction_sets

    def predict_set_with_rules(self, X: np.ndarray, alpha: float = 0.1) -> List[Dict]:
        """
        Predict conformal sets with rule-level explanations.

        Returns prediction sets along with which rules contribute
        and their individual confidence levels. This provides
        explainable uncertainty quantification.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        alpha : float, default=0.1
            Significance level (0.1 = 90% coverage)

        Returns
        -------
        results : list of dicts
            Each dict contains:
            - 'prediction_set': set of class indices in the prediction set
            - 'rule_contributions': list of dicts with rule info, each containing:
                - 'rule_index': index of the rule
                - 'class': consequent class of the rule
                - 'firing_strength': how strongly the rule fired
                - 'rule_confidence': calibrated confidence for this rule
            - 'class_p_values': dict mapping class index to p-value

        Raises
        ------
        ValueError
            If the classifier is not calibrated

        Examples
        --------
        >>> results = conf_clf.predict_set_with_rules(X_test[:5], alpha=0.1)
        >>> for i, r in enumerate(results):
        ...     print(f"Sample {i}:")
        ...     print(f"  Prediction set: {r['prediction_set']}")
        ...     print(f"  P-values: {r['class_p_values']}")
        ...     for contrib in r['rule_contributions'][:3]:
        ...         print(f"  Rule {contrib['rule_index']}: "
        ...               f"class={contrib['class']}, "
        ...               f"strength={contrib['firing_strength']:.3f}")
        """
        if not self._calibrated:
            raise ValueError("Classifier not calibrated. Call calibrate() or fit() first.")

        X = np.asarray(X)

        # Get per-rule association degrees
        rule_scores = self.clf.predict_proba_rules(X, truth_degrees=False)
        rule_consequents = self.clf.rule_base.get_consequents()

        n_samples = X.shape[0]
        results = []

        for i in range(n_samples):
            result = {
                'prediction_set': set(),
                'rule_contributions': [],
                'class_p_values': {}
            }

            # Compute p-values per class
            class_scores = self.clf.predict_membership_class(X[i:i+1])[0]
            for c in range(self.clf.nclasses_):
                score = 1 - class_scores[c]
                p_value = self._compute_p_value(score, c)
                result['class_p_values'][c] = p_value
                if p_value > alpha:
                    result['prediction_set'].add(c)

            # Get rule contributions for classes in prediction set
            for rule_idx, consequent in enumerate(rule_consequents):
                if consequent in result['prediction_set']:
                    rule_score = rule_scores[i, rule_idx]
                    if rule_score > 0:
                        result['rule_contributions'].append({
                            'rule_index': rule_idx,
                            'class': consequent,
                            'firing_strength': float(rule_score),
                            'rule_confidence': self._get_rule_confidence(rule_idx, rule_score)
                        })

            # Sort by firing strength (highest first)
            result['rule_contributions'].sort(key=lambda x: x['firing_strength'], reverse=True)
            results.append(result)

        return results

    def _compute_nonconformity_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute nonconformity scores based on score_type."""
        if self.score_type == 'membership':
            # 1 - membership degree of true class
            class_memberships = self.clf.predict_membership_class(X)
            scores = 1 - class_memberships[np.arange(len(y)), y.astype(int)]

        elif self.score_type == 'association':
            # 1 - max association degree for rules of true class
            rule_scores = self.clf.predict_proba_rules(X, truth_degrees=False)
            consequents = self.clf.rule_base.get_consequents()
            scores = np.zeros(len(y))
            for i, true_class in enumerate(y):
                class_rules = [j for j, c in enumerate(consequents) if c == int(true_class)]
                if class_rules:
                    scores[i] = 1 - np.max(rule_scores[i, class_rules])
                else:
                    scores[i] = 1.0

        elif self.score_type == 'entropy':
            # Entropy of class probability distribution
            probs = self.clf.predict_proba(X)
            probs = np.clip(probs, 1e-10, 1)  # Avoid log(0)
            scores = -np.sum(probs * np.log(probs), axis=1)
            # Normalize by max entropy for comparability
            max_entropy = np.log(self.clf.nclasses_)
            scores = scores / max_entropy if max_entropy > 0 else scores
        else:
            raise ValueError(f"Unknown score_type: {self.score_type}")

        return scores

    def _compute_p_value(self, score: float, class_idx: int) -> float:
        """
        Compute conformal p-value for a given score and class.

        Uses the formula: (#{s_i >= s} + 1) / (n + 1)
        where s_i are calibration scores and s is the test score.
        """
        if class_idx not in self._calibration_scores:
            return 0.0

        cal_scores = self._calibration_scores[class_idx]
        if len(cal_scores) == 0:
            return 0.0

        # P-value = proportion of calibration scores >= test score
        # Adding +1 to numerator and denominator for finite sample correction
        return (np.sum(cal_scores >= score) + 1) / (len(cal_scores) + 1)

    def _calibrate_rules(self, X_cal: np.ndarray, y_cal: np.ndarray):
        """Calibrate per-rule thresholds for rule-wise conformal prediction."""
        rule_scores = self.clf.predict_proba_rules(X_cal, truth_degrees=False)
        consequents = self.clf.rule_base.get_consequents()
        n_rules = len(consequents)

        self._rule_calibration = {}
        for rule_idx in range(n_rules):
            rule_class = consequents[rule_idx]
            # Get scores for samples where this rule's class is correct
            correct_mask = y_cal == rule_class
            if np.sum(correct_mask) > 0:
                self._rule_calibration[rule_idx] = rule_scores[correct_mask, rule_idx]

    def _get_rule_confidence(self, rule_idx: int, score: float) -> float:
        """Get confidence for a rule firing at given strength."""
        if self._rule_calibration is None or rule_idx not in self._rule_calibration:
            return 0.0
        cal_scores = self._rule_calibration[rule_idx]
        if len(cal_scores) == 0:
            return 0.0
        # Confidence = proportion of calibration scores <= current score
        return float(np.sum(cal_scores <= score) / len(cal_scores))

    def _split_calibration(self, X, y, cal_size):
        """Split data into training and calibration sets (stratified)."""
        from sklearn.model_selection import train_test_split
        return train_test_split(X, y, test_size=cal_size, stratify=y, random_state=42)

    def get_calibration_info(self) -> Dict:
        """
        Return calibration statistics for inspection.

        Returns
        -------
        info : dict
            Dictionary containing:
            - 'n_calibration_samples': total calibration samples
            - 'samples_per_class': dict of samples per class
            - 'score_type': the nonconformity score type used
            - 'n_rules_calibrated': number of rules with calibration data

        Examples
        --------
        >>> info = conf_clf.get_calibration_info()
        >>> print(f"Calibrated with {info['n_calibration_samples']} samples")
        >>> print(f"Score type: {info['score_type']}")
        """
        if not self._calibrated:
            return {}

        return {
            'n_calibration_samples': sum(len(v) for v in self._calibration_scores.values()),
            'samples_per_class': {k: len(v) for k, v in self._calibration_scores.items()},
            'score_type': self.score_type,
            'n_rules_calibrated': len(self._rule_calibration) if self._rule_calibration else 0
        }

    # Delegate common properties to wrapped classifier
    @property
    def rule_base(self):
        """The underlying rule base from the fuzzy classifier."""
        return self.clf.rule_base

    @property
    def nclasses_(self):
        """Number of classes."""
        return self.clf.nclasses_

    def print_rules(self, return_rules: bool = False, bootstrap_results: bool = False):
        """Print or return the rules from the underlying classifier."""
        return self.clf.print_rules(return_rules=return_rules,
                                    bootstrap_results=bootstrap_results)


def evaluate_conformal_coverage(conf_clf: ConformalFuzzyClassifier,
                                 X_test: np.ndarray, y_test: np.ndarray,
                                 alpha: float = 0.1) -> Dict:
    """
    Evaluate conformal prediction coverage and efficiency.

    Computes metrics to assess the quality of conformal predictions,
    including whether the coverage guarantee is satisfied and how
    informative the prediction sets are.

    Parameters
    ----------
    conf_clf : ConformalFuzzyClassifier
        A calibrated conformal classifier
    X_test : array-like of shape (n_samples, n_features)
        Test samples
    y_test : array-like of shape (n_samples,)
        True labels for test samples
    alpha : float, default=0.1
        Significance level used for prediction sets

    Returns
    -------
    metrics : dict
        Dictionary containing:
        - 'coverage': Empirical coverage (should be >= 1-alpha)
        - 'expected_coverage': The target coverage (1-alpha)
        - 'avg_set_size': Average prediction set size
        - 'efficiency': 1 / avg_set_size (higher is better)
        - 'empty_sets': Proportion of empty prediction sets
        - 'singleton_sets': Proportion of single-class sets
        - 'coverage_by_class': Per-class coverage rates

    Examples
    --------
    >>> metrics = evaluate_conformal_coverage(conf_clf, X_test, y_test, alpha=0.1)
    >>> print(f"Coverage: {metrics['coverage']:.3f} (expected: {metrics['expected_coverage']:.3f})")
    >>> print(f"Average set size: {metrics['avg_set_size']:.2f}")
    >>> print(f"Singleton rate: {metrics['singleton_sets']:.2%}")

    >>> # Check coverage guarantee
    >>> if metrics['coverage'] >= metrics['expected_coverage'] - 0.05:
    ...     print("Coverage guarantee approximately satisfied")
    """
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    pred_sets = conf_clf.predict_set(X_test, alpha=alpha)

    # Coverage: proportion where true class is in prediction set
    coverage = np.mean([y_test[i] in pred_sets[i] for i in range(len(y_test))])

    # Average set size
    set_sizes = [len(s) for s in pred_sets]
    avg_size = np.mean(set_sizes)

    # Efficiency metrics
    empty = np.mean([len(s) == 0 for s in pred_sets])
    singleton = np.mean([len(s) == 1 for s in pred_sets])

    # Per-class coverage
    coverage_by_class = {}
    for c in np.unique(y_test):
        mask = y_test == c
        class_coverage = np.mean([y_test[i] in pred_sets[i] for i in np.where(mask)[0]])
        coverage_by_class[int(c)] = class_coverage

    return {
        'coverage': coverage,
        'expected_coverage': 1 - alpha,
        'avg_set_size': avg_size,
        'efficiency': 1 / avg_size if avg_size > 0 else 0,
        'empty_sets': empty,
        'singleton_sets': singleton,
        'coverage_by_class': coverage_by_class
    }
