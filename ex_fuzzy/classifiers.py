"""
Fuzzy Classification Algorithms for Ex-Fuzzy Library

This module provides high-level classification algorithms that combine rule mining,
genetic optimization, and fuzzy inference for pattern classification tasks. The
classifiers implement sophisticated two-stage optimization approaches that first
discover candidate rules through data mining and then optimize rule combinations
using evolutionary algorithms.

Main Components:
    - RuleMineClassifier: Two-stage classifier combining rule mining and genetic optimization
    - DoubleGo classifier: Advanced multi-objective genetic optimization
    - Integrated preprocessing: Automatic linguistic variable generation
    - Performance optimization: Efficient rule evaluation and selection
    - Scikit-learn compatibility: Standard fit/predict interface

Key Features:
    - Automatic feature fuzzification with optimal partitioning
    - Rule mining with support, confidence, and lift thresholds
    - Multi-objective optimization balancing accuracy and interpretability
    - Support for imbalanced datasets with specialized fitness functions
    - Cross-validation based fitness evaluation for robust models
    - Integration with various fuzzy set types (Type-1, Type-2, GT2)

The classifiers are designed to be both highly accurate and interpretable,
making them suitable for applications where understanding the decision process
is as important as predictive performance.
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from . import fuzzy_sets as fs
from . import evolutionary_fit as evf
from . import rule_mining as rm
from . import rules
from . import utils
from . import _fuzzy_association as association


class RuleMineClassifier(ClassifierMixin, BaseEstimator):
    """
    A classifier that works by mining a set of candidate rules with a minimum support, confidence and lift, and then using a genetic algorithm that chooses
    the optimal combination of those rules.
    """

    def __init__(self, nRules: int = 30, nAnts: int = 4, fuzzy_type: fs.FUZZY_SETS = fs.FUZZY_SETS.t1, tolerance: float = 0.0,
                verbose=False, n_class: int=None, runner: int=1, linguistic_variables: list[fs.fuzzyVariable]=None,
                n_gen: int = 30, pop_size: int = 50, patience: Optional[int] = 10, random_state: int = 33) -> None:
        """
        Inits the optimizer with the corresponding parameters.

        Args:
            nRules: number of rules to optimize.
            nAnts: max number of antecedents to use.
            fuzzy_type: FUZZY_SET enum type in fuzzy_sets module. The kind of fuzzy set used.
            tolerance: tolerance for the support/dominance score of the rules.
            verbose: if True, prints the progress of the optimization.
            n_class: number of classes in the problem. If None (default) the classifier will compute it empirically.
            runner: number of threads to use.
            linguistic_variables: linguistic variables per antecedent.
            n_gen: number of generations of the genetic search. fit can override it for one call.
            pop_size: population size of the genetic search. fit can override it for one call.
            patience: generations without improvement before the search stops early; None runs every generation.
            random_state: random seed of the genetic search.
        """
        self.n_gen = n_gen
        self.pop_size = pop_size
        self.patience = patience
        self.random_state = random_state
        self.nRules = nRules
        self.nAnts = nAnts
        self.fuzzy_type = fuzzy_type
        self.tolerance = tolerance
        self.verbose = verbose
        self.n_class = n_class
        self.runner = runner
        self.linguistic_variables = linguistic_variables


    def _new_classifier(self, nRules: int, nAnts: int = 4) -> evf.BaseFuzzyRulesClassifier:
        """
        Builds the genetic classifier that selects the final rules.
        """
        return evf.BaseFuzzyRulesClassifier(nRules=nRules, nAnts=nAnts, linguistic_variables=self.linguistic_variables,
                                            fuzzy_type=self.fuzzy_type, verbose=self.verbose, tolerance=self.tolerance,
                                            runner=self.runner, n_class=self.n_class,
                                            patience=self.patience, random_state=self.random_state)


    @property
    def fl_classifier(self) -> evf.BaseFuzzyRulesClassifier:
        """
        The classifier that performs the final predictions. Built when fit is called.
        """
        check_is_fitted(self, attributes=['fl_classifier_'])
        return self.fl_classifier_


    def fit(self, X: np.array, y: np.array, n_gen: int = evf.CONSTRUCTOR, pop_size: int = evf.CONSTRUCTOR, **kwargs):
        """
        Trains the model with the given data.

        Args:
            X: samples to train.
            y: labels for each sample.
            n_gen: number of generations to compute in the genetic optimization. Defaults to the constructor's value.
            pop_size: number of subjects per generation. Defaults to the constructor's value.
            kwargs: additional parameters for the genetic optimization, including early stopping with patience=10 and min_delta=1e-4 by default. See fit method in BaseRuleBaseClassifier.

        Returns:
            the fitted classifier.
        """
        n_gen = self.n_gen if n_gen is evf.CONSTRUCTOR else n_gen
        pop_size = self.pop_size if pop_size is evf.CONSTRUCTOR else pop_size
        # Mine on the linguistic variables the classifier will use.
        fuzzy_vars = self.linguistic_variables
        if fuzzy_vars is None:
            fuzzy_vars = utils.construct_partitions(X, self.fuzzy_type)
        candidate_rules = rm.multiclass_mine_rulebase(X, y, fuzzy_vars, self.tolerance, max_depth=self.nAnts)
        self.fl_classifier_ = self._new_classifier(self.nRules, self.nAnts)
        self.fl_classifier_.fit(X, y, checkpoints=0, candidate_rules=candidate_rules, n_gen=n_gen, pop_size=pop_size, **kwargs)
        self.classes_ = self.fl_classifier_.classes_
        self.n_features_in_ = self.fl_classifier_.n_features_in_
        return self


    def predict(self, X: np.array) -> np.array:
        """
        Predict for each sample the corresponding class.

        Args:
            X: samples to predict.

        Returns:
            a class for each sample.
        """
        return self.fl_classifier.predict(X)


    def internal_classifier(self) -> evf.BaseFuzzyRulesClassifier:
        # Returns the classifier that performs the final predictions
        return self.fl_classifier
    


class FuzzyRulesClassifier(ClassifierMixin, BaseEstimator):
    """
    Fuzzy association rule classifier with feature capping and genetic rule selection.

    The classifier follows the FARC-HD family of fuzzy association rule
    classifiers (Alcalá-Fdez, Alcalá and Herrera, IEEE Transactions on Fuzzy
    Systems 19(5), 2011), which learns a compact linguistic rule base:

    1. The input space is capped to the ``max_features`` most informative
       features, which keeps rule generation tractable and removes noise.
    2. Each kept feature gets a fixed fuzzy partition (quantile-based terms for
       numerical features, one term per category for categorical ones).
    3. Candidate rules with up to ``nAnts`` conditions are mined per class and
       filtered by support, confidence and penalized certainty factor.
    4. A covering-based subgroup-discovery prescreen keeps a diverse pool of up
       to ``candidates_per_class`` rules per class.
    5. A genetic algorithm selects a compact subset of rules that maximizes
       training accuracy with a small penalty per rule, capped by ``nRules``
       or ``rules_per_class`` when set.

    Rules are weighted by their penalized certainty factor and combined as in
    Ex-Fuzzy's regression: with ``rule_mode="additive"`` every matching rule
    votes with its weighted firing, and with ``rule_mode="sufficient"`` only each
    sample's strongest weighted rule decides. Samples that fire no rule take the
    training majority class. :class:`ex_fuzzy.BaseFuzzyRulesClassifier`
    is not modified.

    Args:
        nRules (int or None, default=None):
            Maximum number of rules in the final rule base. ``None`` removes the
            absolute cap.
        nAnts (int or "auto", default=3):
            Maximum number of conditions per rule. ``"auto"`` allows four
            conditions when a class's rules are mined from at most six features,
            where interactions must carry the model, and three otherwise.
            Four or five conditions are supported; the candidate search is then
            pruned by ``min_support``, so raising it also speeds up long rules.
        fuzzy_type (FUZZY_SETS, default=FUZZY_SETS.t1):
            Only Type-1 fuzzy sets are supported.
        tolerance (float, default=0.0):
            Minimum penalized certainty factor a candidate rule must exceed.
        verbose (bool, default=False):
            Print a summary of the fitting stages.
        n_class (int, optional):
            Ignored; the classes are read from ``y``. Kept for compatibility.
        runner (int, default=1):
            Ignored; the rule selection is vectorized. Kept for compatibility.
        expansion_factor (int, default=1):
            Multiplies ``candidates_per_class``, enlarging the pool the genetic
            search chooses from.
        linguistic_variables (list of fuzzyVariable, optional):
            One fuzzy variable per input feature. When omitted, partitions are
            built from the data.
        max_features (int, "auto" or None, default=8):
            Maximum number of input features to keep. ``None`` keeps them all.
            ``"auto"`` keeps 8 for problems with up to five classes and 16 for
            problems with more, which need more features to separate.
        feature_selector ({"mutual_info", "f_classif"} or callable, default="mutual_info"):
            Feature relevance score. A callable receives ``(X, y)`` and returns one
            score per feature.
        feature_selection ({"global", "per_class"}, default="per_class"):
            ``"global"`` keeps the ``max_features`` most relevant features for all
            classes. ``"per_class"`` scores features one class against the rest and
            mines each class's rules on its own ``max_features`` features, so the
            rule base can use more features in total while each rule search stays
            small.
        n_linguistic_variables (int or "auto", default="auto"):
            Number of fuzzy terms per numerical feature. ``"auto"`` uses three
            terms for binary problems and five otherwise.
        min_support (float, default=0.05):
            Minimum class support of a candidate rule.
        min_confidence (float, default=0.5):
            Minimum confidence of a candidate rule.
        candidates_per_class (int, default=50):
            Maximum rules per class kept by the prescreen.
        rule_mode ({"additive", "sufficient"}, default="additive"):
            How rules are combined into a class decision. ``"additive"`` sums each
            class's weighted rule firing; ``"sufficient"`` keeps only each sample's
            strongest weighted rule, as in :class:`ex_fuzzy.BaseFuzzyRulesRegressor`.
        n_gen (int, default=100):
            Maximum generations of the rule-selection search.
        pop_size (int, default=60):
            Population size of the rule-selection search.
        patience (int, default=20):
            Generations without improvement before the search stops.
        rule_penalty (float, default=1e-3):
            Accuracy traded for each selected rule per class.
        rules_per_class (int, optional):
            Cap the rule base at ``rules_per_class`` times the number of classes.
            When ``nRules`` is also set, the smaller cap applies.
        random_state (int, optional):
            Seed for feature scoring and the genetic search.

    Attributes:
        classes_ (np.ndarray):
            Class labels.
        selected_features_ (np.ndarray):
            Indices of the input features the rules use.
        class_features_ (list of np.ndarray):
            Input feature indices each class's rules were mined from.
        max_features_ (int or None):
            Resolved feature cap.
        n_conditions_ (list of int):
            Resolved maximum number of conditions per class.
        linguistic_variables_ (list of fuzzyVariable):
            Partitions of the selected features.
        rule_base_ (MasterRuleBase or None):
            Selected rules, one rule base per class, weighted by certainty factor.
        n_rules_ (int):
            Number of selected rules.
        majority_class_ (int):
            Index of the class predicted when no rule fires.
    """

    def __init__(self, nRules: int = None, nAnts: int = 3, fuzzy_type: fs.FUZZY_SETS = fs.FUZZY_SETS.t1,
                 tolerance: float = 0.0, verbose=False, n_class: int = None, runner: int = 1,
                 expansion_factor: int = 1, linguistic_variables: list[fs.fuzzyVariable] = None,
                 max_features=8, feature_selector='mutual_info', feature_selection: str = 'per_class',
                 n_linguistic_variables='auto',
                 min_support: float = 0.05, min_confidence: float = 0.5, candidates_per_class: int = 50,
                 rule_mode: str = 'additive', n_gen: int = 100, pop_size: int = 60, patience: int = 20,
                 rule_penalty: float = 1e-3, rules_per_class: int = None, random_state=None) -> None:
        self.nRules = nRules
        self.nAnts = nAnts
        self.fuzzy_type = fuzzy_type
        self.tolerance = tolerance
        self.verbose = verbose
        self.n_class = n_class
        self.runner = runner
        self.expansion_factor = expansion_factor
        self.linguistic_variables = linguistic_variables
        self.max_features = max_features
        self.feature_selector = feature_selector
        self.feature_selection = feature_selection
        self.n_linguistic_variables = n_linguistic_variables
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.candidates_per_class = candidates_per_class
        self.rule_mode = rule_mode
        self.n_gen = n_gen
        self.pop_size = pop_size
        self.patience = patience
        self.rule_penalty = rule_penalty
        self.rules_per_class = rules_per_class
        self.random_state = random_state

    def _validate(self, n_gen, pop_size, patience):
        if self.fuzzy_type != fs.FUZZY_SETS.t1:
            raise ValueError("FuzzyRulesClassifier supports only Type-1 fuzzy sets (FUZZY_SETS.t1).")
        if self.rule_mode not in association.RULE_MODES:
            raise ValueError("rule_mode must be either 'additive' or 'sufficient'.")
        if self.nAnts != 'auto' and (isinstance(self.nAnts, str) or self.nAnts < 1):
            raise ValueError("nAnts must be at least 1 or 'auto'.")
        if self.nRules is not None and self.nRules < 1:
            raise ValueError("nRules must be at least 1.")
        if self.max_features not in (None, 'auto') and (isinstance(self.max_features, str)
                                                         or self.max_features < 1):
            raise ValueError("max_features must be at least 1, 'auto' or None.")
        if self.n_linguistic_variables != 'auto' and (isinstance(self.n_linguistic_variables, str)
                                                     or self.n_linguistic_variables < 2):
            raise ValueError("n_linguistic_variables must be at least 2 or 'auto'.")
        if self.rules_per_class is not None and self.rules_per_class < 1:
            raise ValueError("rules_per_class must be at least 1.")
        if self.candidates_per_class < 1:
            raise ValueError("candidates_per_class must be at least 1.")
        if self.feature_selection not in ('global', 'per_class'):
            raise ValueError("feature_selection must be either 'global' or 'per_class'.")
        if not (callable(self.feature_selector) or self.feature_selector in ('mutual_info', 'f_classif')):
            raise ValueError("feature_selector must be 'mutual_info', 'f_classif' or a callable.")
        if n_gen < 0 or pop_size < 2 or patience < 1:
            raise ValueError("n_gen must be non-negative, pop_size at least 2 and patience at least 1.")

    def _feature_scores(self, X, y_index, seed):
        if callable(self.feature_selector):
            return np.asarray(self.feature_selector(X, y_index), dtype=float)
        if self.feature_selector == 'f_classif':
            from sklearn.feature_selection import f_classif
            scores = f_classif(X, y_index)[0]
            return np.nan_to_num(scores, nan=0.0, posinf=np.finfo(float).max)
        from sklearn.feature_selection import mutual_info_classif
        discrete = utils.detect_categorical_mask(X) > 0
        return mutual_info_classif(X, y_index, discrete_features=discrete, random_state=seed)

    def fit(self, X, y, n_gen: int = None, pop_size: int = None, checkpoints: int = 0, **kwargs):
        """
        Mine, prescreen and select the rule base.

        Args:
            X (array-like of shape (n_samples, n_features)):
                Training features.
            y (array-like of shape (n_samples,)):
                Class labels.
            n_gen, pop_size : int, optional:
                Override the constructor's search budget for this fit.
            checkpoints (int, default=0):
                Ignored; kept for compatibility.
            **kwargs:
                ``random_state`` and ``patience`` override the constructor values.

        Returns:
            FuzzyRulesClassifier
                The fitted estimator.
        """
        unknown = set(kwargs) - {'random_state', 'patience'}
        if unknown:
            raise TypeError(f"Unexpected fit arguments: {sorted(unknown)}")
        n_gen = self.n_gen if n_gen is None else n_gen
        pop_size = self.pop_size if pop_size is None else pop_size
        patience = kwargs.get('patience', self.patience)
        seed = kwargs.get('random_state', self.random_state)
        self._validate(n_gen, pop_size, patience)

        if hasattr(X, 'columns'):
            columns = [str(column) for column in X.columns]
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        else:
            columns = None
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")
        if y.ndim != 1:
            y = np.ravel(y)
        if len(X) != len(y) or len(X) == 0:
            raise ValueError("X and y must contain the same, non-zero number of samples.")
        if not np.all(np.isfinite(X)):
            raise ValueError("FuzzyRulesClassifier does not support NaN or infinite feature values.")
        self.classes_, y_index = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError("FuzzyRulesClassifier needs at least two classes.")
        self.n_features_in_ = X.shape[1]
        if self.linguistic_variables is not None and len(self.linguistic_variables) != X.shape[1]:
            raise ValueError("linguistic_variables must contain one fuzzy variable per feature.")
        rng = np.random.default_rng(seed)
        score_seed = int(rng.integers(2 ** 31 - 1))

        if self.max_features == 'auto':
            self.max_features_ = 8 if n_classes <= 5 else 16
        else:
            self.max_features_ = self.max_features
        capped = self.max_features_ is not None and X.shape[1] > self.max_features_
        if capped and self.feature_selection == 'per_class':
            self.class_features_ = []
            for c in range(n_classes):
                scores = self._feature_scores(X, (y_index == c).astype(int), score_seed)
                order = np.argsort(-scores, kind='stable')[:self.max_features_]
                self.class_features_.append(np.sort(order))
            self.selected_features_ = np.unique(np.concatenate(self.class_features_))
        elif capped:
            scores = self._feature_scores(X, y_index, score_seed)
            order = np.argsort(-scores, kind='stable')[:self.max_features_]
            self.selected_features_ = np.sort(order)
            self.class_features_ = [self.selected_features_] * n_classes
        else:
            self.selected_features_ = np.arange(X.shape[1])
            self.class_features_ = [self.selected_features_] * n_classes
        X_selected = X[:, self.selected_features_]
        names = [columns[i] if columns else str(i) for i in self.selected_features_]

        if self.linguistic_variables is not None:
            self.linguistic_variables_ = [self.linguistic_variables[i] for i in self.selected_features_]
        else:
            n_terms = self.n_linguistic_variables
            if n_terms == 'auto':
                n_terms = 3 if n_classes == 2 else 5
            self.linguistic_variables_ = utils.construct_partitions(
                pd.DataFrame(X_selected, columns=names), fs.FUZZY_SETS.t1, n_partitions=n_terms)

        memberships = association.membership_matrices(self.linguistic_variables_, X_selected)
        self.n_conditions_ = [self._conditions(len(features)) for features in self.class_features_]
        if capped and self.feature_selection == 'per_class':
            candidates = self._mine_per_class(memberships, y_index, n_classes)
        else:
            candidates = association.mine_candidates(
                memberships, y_index, n_classes, max_conditions=self._conditions(len(memberships)),
                min_support=self.min_support,
                min_confidence=self.min_confidence, min_weight=self.tolerance)
        pool = association.prescreen(candidates, y_index, n_classes,
                                     per_class=self.candidates_per_class * max(1, self.expansion_factor))
        counts = np.bincount(y_index, minlength=n_classes)
        self.majority_class_ = int(np.argmax(counts))
        caps = [cap for cap in (self.nRules, None if self.rules_per_class is None
                                else self.rules_per_class * n_classes) if cap is not None]
        mask = association.select_rules(
            pool, y_index, n_classes, self.majority_class_, rule_mode=self.rule_mode,
            max_rules=min(caps) if caps else None, rule_penalty=self.rule_penalty, n_gen=n_gen,
            pop_size=pop_size, patience=patience, rng=rng)
        self._rules = association._take(pool, np.flatnonzero(mask)) if mask.size else association._empty_pool()
        self._rules['firing'] = None                            # training firing is not needed after fit
        self.n_rules_ = int(len(self._rules['consequent']))
        self.class_prior_ = counts / counts.sum()
        self.rule_base_ = self._build_rule_base()
        if self.verbose:
            print(f"FuzzyRulesClassifier: {len(self.selected_features_)} features, "
                  f"{len(candidates['consequent'])} candidates, {len(pool['consequent'])} prescreened, "
                  f"{self.n_rules_} rules selected.")
        return self

    def _conditions(self, n_available: int) -> int:
        """Resolved maximum rule length for mining over ``n_available`` features."""
        if self.nAnts == 'auto':
            return 4 if n_available <= 6 else 3
        return self.nAnts

    def _mine_per_class(self, memberships, y_index, n_classes):
        """Mine each class's rules on its own features, then merge the pools."""
        position = {feature: index for index, feature in enumerate(self.selected_features_)}
        pools = []
        for c in range(n_classes):
            local = [position[feature] for feature in self.class_features_[c]]
            local_memberships = [memberships[index] for index in local]
            mined = association.mine_candidates(
                local_memberships, y_index, n_classes, max_conditions=self._conditions(len(local)),
                min_support=self.min_support, min_confidence=self.min_confidence,
                min_weight=self.tolerance)
            own = np.flatnonzero(mined['consequent'] == c)
            if own.size:
                mined = association._take(mined, own)
            else:
                mined = association.best_rules_for_class(local_memberships, y_index, n_classes, c,
                                                         min(2, self._conditions(len(local))))
            mined['features'] = [tuple(local[f] for f in features) for features in mined['features']]
            pools.append(mined)
        return association._merge(pools)

    def _build_rule_base(self):
        """Export the selected rules as Ex-Fuzzy rule bases, one per class."""
        if self.n_rules_ == 0:
            return None
        n_features = len(self.selected_features_)
        rule_bases = []
        for c in range(len(self.classes_)):
            class_rules = []
            for row in np.flatnonzero(self._rules['consequent'] == c):
                antecedents = [-1] * n_features
                for feature, term in zip(self._rules['features'][row], self._rules['terms'][row]):
                    antecedents[feature] = int(term)
                rule = rules.RuleSimple(antecedents, consequent=0)
                rule.weight = float(self._rules['weight'][row])
                rule.score = float(self._rules['weight'][row])
                rule.accuracy = float(self._rules['confidence'][row])
                class_rules.append(rule)
            rule_bases.append(rules.RuleBaseT1(self.linguistic_variables_, class_rules))
        return rules.MasterRuleBase(rule_bases, [str(label) for label in self.classes_],
                                    ds_mode=2, allow_unknown=False)

    def _scores(self, X):
        check_is_fitted(self, attributes=['classes_', 'selected_features_'])
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but FuzzyRulesClassifier was fitted with "
                             f"{self.n_features_in_} features.")
        n_classes = len(self.classes_)
        if self.n_rules_ == 0:
            return np.zeros((len(X), n_classes))
        X_selected = X[:, self.selected_features_]
        memberships = association.membership_matrices(self.linguistic_variables_, X_selected)
        firing = association.rule_firing(memberships, self._rules['features'], self._rules['terms'], len(X))
        return association.class_scores(firing, self._rules['weight'], self._rules['consequent'],
                                        n_classes, self.rule_mode)

    def predict(self, X) -> np.ndarray:
        """Predict the class of each sample."""
        prediction = association.predict_from_scores(self._scores(X), self.majority_class_)
        return self.classes_[prediction]

    def predict_proba(self, X) -> np.ndarray:
        """
        Class probabilities from normalized rule scores.

        Samples that fire no rule receive the training class distribution.
        """
        scores = self._scores(X)
        totals = scores.sum(1, keepdims=True)
        return np.divide(scores, totals, out=np.tile(self.class_prior_, (len(scores), 1)),
                         where=totals > 0)

    def print_rules(self, return_rules: bool = False):
        """Print the selected rules, one block per class."""
        check_is_fitted(self, attributes=['rule_base_'])
        if self.rule_base_ is None:
            text = "No rules were selected; every sample takes the majority class."
            if return_rules:
                return text
            print(text)
            return None
        return self.rule_base_.print_rules(return_rules)

    def internal_classifier(self) -> evf.BaseFuzzyRulesClassifier:
        """
        The selected rule base wrapped as a :class:`BaseFuzzyRulesClassifier`.

        The wrapped model expects only the selected features
        (``X[:, selected_features_]``) and uses winning-rule inference without
        the majority-class fallback; use this estimator's ``predict`` for the
        classifier's own decisions.
        """
        check_is_fitted(self, attributes=['rule_base_'])
        if self.rule_base_ is None:
            raise ValueError("No rules were selected, so there is no internal rule-base classifier.")
        return evf.BaseFuzzyRulesClassifier(precomputed_rules=self.rule_base_)


class RuleFineTuneClassifier(ClassifierMixin, BaseEstimator):
    """
    A classifier that works by mining a set of candidate rules with a minimum support and then uses a two step genetic optimization that chooses
    the optimal combination of those rules and fine tunes them.
    """

    def __init__(self, nRules: int = 30, nAnts: int = 4, fuzzy_type: fs.FUZZY_SETS = fs.FUZZY_SETS.t1, tolerance: float = 0.0,
                 verbose=False, n_class: int=None, runner: int=1, expansion_factor:int=1, linguistic_variables: list[fs.fuzzyVariable]=None,
                 n_gen: int = 30, pop_size: int = 50, patience: Optional[int] = 10, random_state: int = 33) -> None:
        """
        Inits the optimizer with the corresponding parameters.

        Args:
            nRules: number of rules to optimize.
            nAnts: max number of antecedents to use.
            fuzzy_type: FUZZY_SET enum type in fuzzy_sets module. The kind of fuzzy set used.
            tolerance: tolerance for the dominance score of the rules.
            verbose: if True, prints the progress of the optimization.
            n_class: number of classes in the problem. If None (default) the classifier will compute it empirically.
            linguistic_variables: linguistic variables per antecedent.
            n_gen: number of generations of the genetic search. fit can override it for one call.
            pop_size: population size of the genetic search. fit can override it for one call.
            patience: generations without improvement before the search stops early; None runs every generation.
            random_state: random seed of the genetic search.
        """
        self.n_gen = n_gen
        self.pop_size = pop_size
        self.patience = patience
        self.random_state = random_state
        self.nRules = nRules
        self.nAnts = nAnts
        self.fuzzy_type = fuzzy_type
        self.tolerance = tolerance
        self.verbose = verbose
        self.n_class = n_class
        self.runner = runner
        self.expansion_factor = expansion_factor
        self.linguistic_variables = linguistic_variables


    _new_classifier = RuleMineClassifier._new_classifier


    @property
    def fl_classifier1(self) -> evf.BaseFuzzyRulesClassifier:
        """
        The classifier of the first phase, which selects among the mined rules. Built when fit is called.
        """
        check_is_fitted(self, attributes=['fl_classifier1_'])
        return self.fl_classifier1_


    @property
    def fl_classifier2(self) -> evf.BaseFuzzyRulesClassifier:
        """
        The classifier of the second phase, which fine tunes the selected rules. Built when fit is called.
        """
        check_is_fitted(self, attributes=['fl_classifier2_'])
        return self.fl_classifier2_


    def fit(self, X: np.array, y: np.array, n_gen: int = evf.CONSTRUCTOR, pop_size: int = evf.CONSTRUCTOR, checkpoints:int=0, **kwargs):
        """
        Trains the model with the given data.

        Args:
            X: samples to train.
            y: labels for each sample.
            n_gen: number of generations to compute in the genetic optimization. Defaults to the constructor's value.
            pop_size: number of subjects per generation. Defaults to the constructor's value.
            checkpoints: if bigger than 0, will save the best subject per x generations in a text file.
            kwargs: additional parameters for the genetic optimization, including early stopping with patience=10 and min_delta=1e-4 by default. See fit method in BaseRuleBaseClassifier.

        Returns:
            the fitted classifier.
        """
        n_gen = self.n_gen if n_gen is evf.CONSTRUCTOR else n_gen
        pop_size = self.pop_size if pop_size is evf.CONSTRUCTOR else pop_size
        partitions = self.linguistic_variables
        if partitions is None:
            partitions = utils.construct_partitions(X, self.fuzzy_type)
        candidate_rules = rm.multiclass_mine_rulebase(X, y, partitions, self.tolerance)

        # The first phase gets more rules so that the second one has a bigger search space.
        self.fl_classifier1_ = self._new_classifier(self.nRules * self.expansion_factor, self.nAnts)
        self.fl_classifier1_.fit(X, y, n_gen, pop_size, checkpoints, candidate_rules=candidate_rules, **kwargs)
        self.phase1_rules = self.fl_classifier1_.rule_base
        self.fl_classifier2_ = self._new_classifier(self.nRules, self.nAnts)
        self.fl_classifier2_.fit(X, y, n_gen, pop_size, checkpoints, initial_rules=self.phase1_rules, **kwargs)
        self.classes_ = self.fl_classifier2_.classes_
        self.n_features_in_ = self.fl_classifier2_.n_features_in_
        return self


    def predict(self, X: np.array) -> np.array:
        """
        Predict for each sample the corresponding class.

        Args:
            X: samples to predict.

        Returns:
            a class for each sample.
        """
        return self.fl_classifier2.predict(X)


    def internal_classifier(self) -> evf.BaseFuzzyRulesClassifier:
        # Returns the classifier that performs the final predictions
        return self.fl_classifier2
