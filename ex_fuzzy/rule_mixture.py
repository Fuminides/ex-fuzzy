"""
Mixture-of-rules inference: a learned weighted aggregation of every rule's firing strength.

Winning-rule inference lets one rule decide each sample. A mixture of rules lets all of them speak:
the score of class ``c`` for a sample ``x`` is

    score_c(x) = b_c + sum_r W[r, c] * mu_r(x)

where ``mu_r(x)`` is the firing strength of rule ``r`` (the t-norm of its antecedent memberships), and
the prediction is the class with the largest score. ``W`` (one weight per rule and class) and ``b``
(one bias per class) are fitted by L2-regularised multinomial logistic regression on the matrix of
firing strengths of the training samples. The rule antecedents come from the genetic search; their
consequents only matter for the search and for pruning, not for the mixture's predictions.

The weights are those of a fitted linear model, not rule confidences. ``softmax(score)`` is that
model's estimate of the class probabilities on data like the training data, not a calibrated
probability, and a firing strength is still a degree of match, not a probability.

Adding the same constant to every class weight of a rule changes no prediction and no softmax score
(the constant times the firing is added to every class). Unconstrained weights are therefore
relative: with the L2 penalty the weights of each rule add up to zero over the classes, so a negative
weight means "less evidence for this class than for the average class", not evidence against it in an
absolute sense. ``non_negative=True`` fits the same family of models with every weight >= 0, so that
a rule reads "votes for class c with strength w"; at the optimum each rule has weight 0 for at least
one class. Both describe the same family of decision functions; only the penalty differs, so their fitted
predictions differ slightly.
"""
from __future__ import annotations

import hashlib
import warnings
from typing import Optional

import numpy as np

from . import fuzzy_sets as fs
from . import rules as rl

#: Inference modes of ``BaseFuzzyRulesClassifier(inference=...)``.
INFERENCE_MODES = ('winning_rule', 'mixture')

#: How ``mixture_search`` prunes a candidate before its mixture is scored (``mixture_pruning``).
#: 'mixture': by the rule's role in the mixture fitted on the candidate (``prune_by_mixture``);
#: 'winning_rule': as a winning-rule model, deleting the rules that never win a training sample correctly.
MIXTURE_PRUNINGS = ('mixture', 'winning_rule')

#: Default ``mixture_prune_threshold``: a rule whose largest centred contribution, in log-odds, over the
#: training samples (``mixture_effects``) is at most this is deleted.
MIXTURE_PRUNE_THRESHOLD = 0.02


def check_inference(inference: str, mixture_C: float, mixture_search: bool, mixture_pruning: str = 'mixture',
                    mixture_prune_threshold: float = MIXTURE_PRUNE_THRESHOLD) -> None:
    """Rejects an unknown inference mode, a non-positive ``mixture_C``, a mixture search without a mixture,
    an unknown ``mixture_pruning`` or a negative ``mixture_prune_threshold``."""
    if inference not in INFERENCE_MODES:
        raise ValueError(f'Unknown inference {inference!r}; use one of {list(INFERENCE_MODES)}.')
    if not (np.isscalar(mixture_C) and float(mixture_C) > 0):
        raise ValueError(f'mixture_C must be a positive number, got {mixture_C!r}.')
    if mixture_search and inference != 'mixture':
        raise ValueError("mixture_search scores the search under the mixture; it needs inference='mixture'.")
    if mixture_pruning not in MIXTURE_PRUNINGS:
        raise ValueError(f'Unknown mixture_pruning {mixture_pruning!r}; use one of {list(MIXTURE_PRUNINGS)}.')
    if not (np.isscalar(mixture_prune_threshold) and float(mixture_prune_threshold) >= 0):
        raise ValueError(f'mixture_prune_threshold must be a number >= 0, got {mixture_prune_threshold!r}.')


def rule_firing(rule_base: rl.MasterRuleBase, X: np.ndarray, precomputed_truth=None) -> np.ndarray:
    """
    The firing strength of every rule for every sample, as a ``(samples, rules)`` array.

    Rules are in ``rule_base.get_rules()`` order. Interval (and alpha-reduced general) Type-2
    firing strengths are averaged over their last axis, as ``compute_association_degrees`` does.

    Args:
        rule_base: the rule base.
        X: samples x features.
        precomputed_truth: antecedent memberships of ``X``, if already computed.

    Returns:
        the firing strengths.
    """
    X = np.asarray(X)
    if not rule_base.get_rules():
        return np.zeros((X.shape[0], 0))
    firing = np.asarray(rule_base.compute_firing_strengths(X, precomputed_truth=precomputed_truth), dtype=float)
    if firing.ndim > 2:
        firing = firing.reshape(firing.shape[0], firing.shape[1], -1).mean(axis=2)
    return firing


def antecedent_keys(rule_base: rl.MasterRuleBase) -> list:
    """What makes two rules fire alike: their antecedents and modifiers (not their consequent)."""
    keys = []
    for rule in rule_base.get_rules():
        antecedents, _consequent, modifiers = rule._identity()
        keys.append((antecedents, modifiers))
    return keys


def rule_keys(rule_base: rl.MasterRuleBase) -> list:
    """The identity of every rule (antecedents, consequent and modifiers), in rule order."""
    return [rule._identity() for rule in rule_base.get_rules()]


class RuleMixture:
    """
    Per-rule, per-class weights and per-class biases fitted on a matrix of rule firing strengths.

    The fit minimises the multinomial logistic loss summed over the samples plus
    ``||W||^2 / (2 C)`` (the convention of scikit-learn's ``LogisticRegression``; the bias is not
    penalised) with L-BFGS-B, from zero weights and the log class frequencies as biases. Samples with
    identical firing rows are added up before the fit, which leaves the objective unchanged.

    Rules with the same antecedents (and modifiers) fire identically; only the first of them gets
    weights, and the others keep zero weights and are listed in ``duplicate_of_``.

    Attributes:
        coef_: ``(rules, classes)`` weights ``W``.
        intercept_: ``(classes,)`` biases ``b``, centred to mean 0 (which changes no score difference).
        duplicate_of_: for each rule, the index of the earlier rule with the same antecedents, or -1.
        rule_keys_: the identity of each rule the weights were fitted for (see ``rule_keys``).
        n_iter_: L-BFGS-B iterations.
        converged_: whether L-BFGS-B reported convergence.
    """

    def __init__(self, C: float = 1.0, non_negative: bool = False, max_iter: int = 1000, tol: float = 1e-6) -> None:
        """
        Args:
            C: inverse strength of the L2 penalty on the weights, as in scikit-learn.
            non_negative: if True, every weight is constrained to be >= 0 (the biases are free).
            max_iter: maximum number of L-BFGS-B iterations.
            tol: L-BFGS-B tolerance on the projected gradient of the per-sample loss.
        """
        self.C = C
        self.non_negative = non_negative
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, firing: np.ndarray, y: np.ndarray, n_classes: int, keys: Optional[list] = None,
            identities: Optional[list] = None, warn: bool = True) -> 'RuleMixture':
        """
        Fits the weights.

        Args:
            firing: ``(samples, rules)`` firing strengths (see ``rule_firing``).
            y: consequent index of each sample, in ``0 .. n_classes - 1``.
            n_classes: number of classes.
            keys: antecedent key of each rule (see ``antecedent_keys``), to merge rules that fire alike.
            identities: identity of each rule (see ``rule_keys``), kept to detect a changed rule base.
            warn: if True, warn when L-BFGS-B does not converge.

        Returns:
            the fitted mixture.

        Raises:
            ValueError: for labels outside the classes, or a class without any sample (its bias would
                diverge).
        """
        from scipy.optimize import minimize

        firing = np.asarray(firing, dtype=float)
        y = np.asarray(y)
        n_samples, n_rules = firing.shape
        if y.shape != (n_samples,):
            raise ValueError(f'y has shape {y.shape}; expected ({n_samples},).')
        if n_samples == 0:
            raise ValueError('No samples to fit the mixture weights on.')
        if y.dtype.kind not in 'iu' or y.min() < 0 or y.max() >= n_classes:
            raise ValueError(f'y must hold consequent indexes 0 .. {n_classes - 1}.')
        frequencies = np.bincount(y, minlength=n_classes).astype(float)
        if np.any(frequencies == 0):
            missing = np.flatnonzero(frequencies == 0).tolist()
            raise ValueError(f'Classes {missing} have no training sample; their mixture bias would diverge.')

        duplicate_of = np.full(n_rules, -1, dtype=int)
        if keys is not None:
            if len(keys) != n_rules:
                raise ValueError(f'{len(keys)} rule keys for {n_rules} rules.')
            first = {}
            for index, key in enumerate(keys):
                duplicate_of[index] = first.setdefault(key, index)
            duplicate_of[duplicate_of == np.arange(n_rules)] = -1
        columns = np.flatnonzero(duplicate_of == -1)

        # Identical firing rows (frequent with crisp term cores) are fitted once, with their class counts.
        features = firing[:, columns]
        if features.shape[1]:
            unique, inverse = np.unique(features, axis=0, return_inverse=True)
            inverse = np.asarray(inverse).reshape(-1)
        else:
            unique, inverse = np.zeros((1, 0)), np.zeros(n_samples, dtype=int)
        counts = np.zeros((unique.shape[0], n_classes))
        np.add.at(counts, (inverse, y), 1.0)
        totals = counts.sum(axis=1)
        n_features = unique.shape[1]
        n_weights = n_features * n_classes
        penalty = 1.0 / float(self.C)

        def objective(theta):
            weights = theta[:n_weights].reshape(n_features, n_classes)
            scores = unique @ weights + theta[n_weights:]
            top = scores.max(axis=1, keepdims=True)
            exponentials = np.exp(scores - top)
            sums = exponentials.sum(axis=1, keepdims=True)
            log_partition = (top + np.log(sums)).ravel()
            loss = totals @ log_partition - np.sum(counts * scores) + 0.5 * penalty * np.sum(weights * weights)
            residual = totals[:, None] * (exponentials / sums) - counts
            gradient = np.concatenate([(unique.T @ residual + penalty * weights).ravel(), residual.sum(axis=0)])
            return loss / n_samples, gradient / n_samples

        start = np.zeros(n_weights + n_classes)
        priors = np.log(frequencies / n_samples)
        start[n_weights:] = priors - priors.mean()
        bounds = None
        if self.non_negative:
            bounds = [(0.0, None)] * n_weights + [(None, None)] * n_classes
        result = minimize(objective, start, jac=True, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': int(self.max_iter), 'gtol': float(self.tol), 'ftol': 1e-12})

        coef = np.zeros((n_rules, n_classes))
        coef[columns] = result.x[:n_weights].reshape(n_features, n_classes)
        intercept = result.x[n_weights:]
        self.coef_ = coef
        self.intercept_ = intercept - intercept.mean()
        self.duplicate_of_ = duplicate_of
        self.rule_keys_ = None if identities is None else list(identities)
        self.n_iter_ = int(result.nit)
        self.converged_ = bool(result.success)
        if warn and not self.converged_:
            warnings.warn(f'The mixture weights did not converge in {self.max_iter} iterations '
                          f'({result.message}); raise max_iter or lower C.', stacklevel=2)
        return self

    def scores(self, firing: np.ndarray) -> np.ndarray:
        """``(samples, classes)`` scores ``b + firing @ W``."""
        return np.asarray(firing, dtype=float) @ self.coef_ + self.intercept_

    def contributions(self, firing: np.ndarray) -> np.ndarray:
        """``(samples, rules, classes)`` contributions ``W[r, c] * mu_r``; they add up to ``scores - b``."""
        firing = np.asarray(firing, dtype=float)
        return firing[:, :, None] * self.coef_[None, :, :]

    def softmax(self, firing: np.ndarray) -> np.ndarray:
        """``softmax(scores)``: the fitted model's class probabilities (not calibrated probabilities)."""
        scores = self.scores(firing)
        exponentials = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exponentials / exponentials.sum(axis=1, keepdims=True)

    def predict(self, firing: np.ndarray, allow_unknown: bool = False) -> np.ndarray:
        """
        The consequent index of the largest score (the first on ties).

        A sample that fires no rule has score ``b``; with ``allow_unknown`` it is predicted as -1,
        as winning-rule inference does, and otherwise as the class of the largest bias.
        """
        firing = np.asarray(firing, dtype=float)
        prediction = np.argmax(self.scores(firing), axis=1)
        if allow_unknown:
            prediction[~np.any(firing > 0, axis=1)] = -1
        return prediction

    def weight_spread(self) -> np.ndarray:
        """``max_c W[r, c] - min_c W[r, c]`` per rule: how much a fully firing rule moves the score of one
        class against another. It does not change when a constant is added to a rule's weights."""
        if self.coef_.shape[0] == 0:
            return np.zeros(0)
        return self.coef_.max(axis=1) - self.coef_.min(axis=1)

    def describe(self, rule_base: rl.MasterRuleBase, class_names: Optional[list] = None, decimals: int = 2) -> str:
        """
        The rules as text, each with its weight for every class, its strongest class, and the biases.

        Args:
            rule_base: the rule base the weights were fitted for.
            class_names: class names, in consequent order; defaults to the rule base's consequent names.
            decimals: decimals of the weights.

        Returns:
            the text.
        """
        names = [str(name) for name in (class_names if class_names is not None
                                        else rule_base.get_consequents_names())]
        kind = 'non-negative' if self.non_negative else 'unconstrained (relative)'
        lines = ['Mixture of rules: score(class) = bias(class) + sum over rules of weight(rule, class) x '
                 f'firing(rule); the prediction is the class with the largest score. Weights: {kind}, C = {self.C}.',
                 'Bias: ' + ' | '.join(f'{name} {value:+.{decimals}f}' for name, value in zip(names, self.intercept_))]
        for index, rule in enumerate(rule_base.get_rules()):
            text = antecedent_text(rule, rule_base.antecedents)
            weights = self.coef_[index]
            if self.duplicate_of_[index] >= 0:
                lines.append(f'Rule {index}: {text}; same antecedents as rule {self.duplicate_of_[index]}, '
                             'which carries the weights')
                continue
            if np.all(weights == weights[0]):
                strongest = 'none (equal weights: it changes no prediction)'
            else:
                strongest = names[int(np.argmax(weights))]
            lines.append(f'Rule {index}: {text}; strongest class: {strongest}; weights: '
                         + ' | '.join(f'{name} {value:+.{decimals}f}' for name, value in zip(names, weights)))
        return '\n'.join(lines)


def mixture_effects(coef: np.ndarray, firing: np.ndarray) -> np.ndarray:
    """
    The largest centred contribution of each rule to any class score over the samples of ``firing``.

    The centred weights ``W[r, c] - mean_c W[r, c]`` describe the same decision function as ``W`` (a constant added to
    every class weight of a rule changes no score difference), so a rule's effect on the class scores of a sample,
    relative to one another, is ``firing[n, r] * centred[r, c]``. Firing strengths are non-negative, so its largest
    absolute value over the samples and classes is ``max_n firing[n, r] * max_c |centred[r, c]|``. A rule that
    never fires, or whose weights are all equal (a duplicate, or a rule the fit left without effect), has effect 0;
    a rule that only lowers some classes has a positive effect, even if it never wins a sample under winning-rule
    inference.

    Args:
        coef: ``(rules, classes)`` mixture weights.
        firing: ``(samples, rules)`` firing strengths.

    Returns:
        ``(rules,)`` effects, in log-odds.
    """
    coef = np.asarray(coef, dtype=float)
    if coef.shape[0] == 0:
        return np.zeros(0)
    centred = coef - coef.mean(axis=1, keepdims=True)
    return np.max(np.asarray(firing, dtype=float), axis=0) * np.max(np.abs(centred), axis=1)


def remove_rules_at(rule_base: rl.MasterRuleBase, positions) -> None:
    """Deletes the rules at ``positions`` of ``rule_base.get_rules()`` (class-major order), in place."""
    positions = set(int(p) for p in positions)
    offset = 0
    for class_base in rule_base.get_rulebases():
        size = len(class_base.get_rules())
        local = [index - offset for index in range(offset, offset + size) if index in positions]
        if local:
            class_base.remove_rules(local)
        offset += size


def prune_by_mixture(rule_base: rl.MasterRuleBase, X: np.ndarray, y: np.ndarray, n_classes: int, C: float,
                     non_negative: bool, threshold: float = MIXTURE_PRUNE_THRESHOLD, precomputed_truth=None):
    """
    Prunes ``rule_base`` by the rules' role in the mixture fitted on it, in place, and fits the survivors' mixture.

    The weights are fitted on every rule of ``rule_base``; the rules whose effect (``mixture_effects``) is at most
    ``threshold`` are deleted; the weights are fitted again, from scratch, on the survivors. Deleting a rule never
    depends on whether it wins a sample, so a rule that only suppresses some classes stays. The second fit is
    the one ``BaseFuzzyRulesClassifier.fit`` gives the final model, so a candidate is scored with the mixture it
    would get.

    Args:
        rule_base: the candidate; rules are deleted from it.
        X: training samples.
        y: consequent index of each sample.
        n_classes: number of classes.
        C: inverse strength of the L2 penalty.
        non_negative: whether the weights are constrained to be >= 0.
        threshold: rules with an effect at most this are deleted; 0 deletes only the rules without any effect.
        precomputed_truth: antecedent memberships of ``X``, if already computed.

    Returns:
        ``(mixture, firing)`` of the survivors, or ``(None, None)`` when no rule survives.
    """
    if not rule_base.get_rules():
        return None, None
    firing = rule_firing(rule_base, X, precomputed_truth)
    mixture = RuleMixture(C=C, non_negative=non_negative).fit(
        firing, y, n_classes, keys=antecedent_keys(rule_base), warn=False)
    keep = mixture_effects(mixture.coef_, firing) > threshold
    if keep.all():
        return mixture, firing
    remove_rules_at(rule_base, np.flatnonzero(~keep))
    if not keep.any():
        return None, None
    firing = np.ascontiguousarray(firing[:, keep])
    mixture = RuleMixture(C=C, non_negative=non_negative).fit(
        firing, y, n_classes, keys=antecedent_keys(rule_base), warn=False)
    return mixture, firing


def antecedent_text(rule: rl.RuleSimple, antecedents: list) -> str:
    """``IF <variable> IS <term> [(MOD <hedge>)] AND ...``, or ``IF (always)`` for a rule without antecedents."""
    parts = []
    modifiers = getattr(rule, 'modifiers', None)
    for index, variable in enumerate(antecedents):
        term = rule[index]
        if term == -1:
            continue
        part = f'{variable.name} IS {fs.condition_label(variable, term)}'
        if modifiers is not None and modifiers[index] != 1:
            part += f' (MOD {rl.modifiers_names.get(modifiers[index], modifiers[index])})'
        parts.append(part)
    return 'IF ' + (' AND '.join(parts) if parts else '(always)')


class MixtureObjective:
    """
    Genetic objective of ``mixture_search``: a candidate is scored as the mixture fitted on it.

    The search problem (``evolutionary_fit._MixtureSearchProblem``) calls ``score`` for each candidate
    instead of the built-in evaluators, so their caches and kernels are not involved. ``score`` prunes the
    candidate, fits the mixture weights on the surviving rules and the training samples, and returns their
    macro F1 minus the same class-coverage and compactness penalties as the built-in objective. With
    ``pruning='mixture'`` (the default) the rules are pruned by their role in the mixture (``prune_by_mixture``);
    with ``pruning='winning_rule'`` as a winning-rule model is finalised (dominance scores, winning-rule
    accuracies, ``purge_rules(tolerance)``), which deletes every rule that never wins a training sample
    correctly, also one that only suppresses some classes in the mixture. Scores are memoised per decoded rule
    base for the fit (the population repeats many candidates), up to ``capacity`` of them.
    """

    def __init__(self, C: float, non_negative: bool, max_conditions: int, capacity: int = 20000,
                 pruning: str = 'mixture', threshold: float = MIXTURE_PRUNE_THRESHOLD) -> None:
        self.C = C
        self.non_negative = non_negative
        self.max_conditions = max_conditions
        self.capacity = capacity
        self.pruning = pruning
        self.threshold = threshold
        self.cache = {}
        self.fits = 0

    def score(self, rule_base, X, y, tolerance, alpha=0.0, beta=0.0, precomputed_truth=None) -> float:
        """The objective of ``rule_base`` (which is pruned in place) on the training samples ``X``, ``y``."""
        from . import eval_rules
        from . import fuzzy_sets as fs
        from ._fitness import score_rulebase

        # A short digest of the decoded rules keeps the memo small (a rule base of 60 rules is kilobytes).
        key = hashlib.blake2b(repr(rule_keys(rule_base)).encode(), digest_size=16).digest()
        if key in self.cache:
            return self.cache[key]
        n_classes = len(rule_base.get_rulebases())
        y = np.asarray(y)
        if self.pruning == 'mixture':
            mixture, firing = prune_by_mixture(rule_base, X, y, n_classes, self.C, self.non_negative,
                                               self.threshold, precomputed_truth)
        else:
            if rule_base.get_rules():
                if y.dtype.kind in 'biu' and rule_base.fuzzy_type() in (fs.FUZZY_SETS.t1, fs.FUZZY_SETS.t2):
                    # The built-in scorer prunes exactly as the fitted model is finalised; its score is not used.
                    score_rulebase(rule_base, X, y, tolerance, alpha, beta, precomputed_truth,
                                   max_conditions=self.max_conditions)
                else:
                    with rule_base._firing_cache_scope():
                        evaluation = eval_rules.evalRuleBase(rule_base, X, y, precomputed_truth=precomputed_truth)
                        evaluation.add_rule_weights()
                        evaluation.add_classification_metrics()
                        rule_base.purge_rules(tolerance)
            mixture = firing = None
            if rule_base.get_rules():
                firing = rule_firing(rule_base, X, precomputed_truth)
                mixture = RuleMixture(C=self.C, non_negative=self.non_negative)
                mixture.fit(firing, y, n_classes, keys=antecedent_keys(rule_base), warn=False)
        score = self._penalized(rule_base, y, n_classes, mixture, firing, alpha, beta)
        if len(self.cache) < self.capacity:
            self.cache[key] = score
        return score

    def _penalized(self, rule_base, y, n_classes, mixture, firing, alpha, beta) -> float:
        """Macro F1 of ``mixture`` on ``firing`` minus the penalties; the empty objective without survivors."""
        from ._fitness import _class_counts, _macro_f1, _penalized_objective

        survivors = rule_base.get_rules()
        if mixture is None or not survivors:
            return float(_penalized_objective(0.0, 0, 0, n_classes, self.max_conditions, alpha, beta))
        self.fits += 1
        prediction = mixture.predict(firing, allow_unknown=rule_base.allow_unknown)
        primary = _macro_f1(*_class_counts(y, prediction, n_classes))
        covered = np.count_nonzero(np.bincount(rule_base.get_consequents(), minlength=n_classes))
        conditions = sum(np.count_nonzero(np.asarray(rule.antecedents) != -1) for rule in survivors)
        return float(_penalized_objective(primary, covered, conditions, n_classes, self.max_conditions,
                                          alpha, beta))
