"""
Private fitness evaluation for decoded T1/T2 classification rule bases.

Decoding remains in FitRuleBase: this module never interprets chromosomes or
caches chromosome-dependent memberships. The full evaluator is the oracle.
"""
import numpy as np
from collections import OrderedDict
from contextlib import contextmanager
from typing import Iterator, Optional

from . import rules, eval_rules

#: Default weight of the compactness penalty: the share of the possible antecedent
#: conditions a rule base uses (``alpha`` of ``reparametrize_loss``).
DEFAULT_COMPACTNESS_WEIGHT = 0.03
#: Default weight of the class-coverage penalty: the share of classes left
#: without rules (``beta`` of ``reparametrize_loss``).
DEFAULT_COVERAGE_WEIGHT = 0.05


class _FitnessCache:
    """Bounded exact-genotype memoization for one immutable fit context."""

    def __init__(self, capacity: int = 256, max_key_bytes: int = 1024 * 1024) -> None:
        self.capacity = capacity
        self.max_key_bytes = max_key_bytes
        self.entries = OrderedDict()
        self.key_bytes = 0

    def key(self, gene: np.ndarray) -> Optional[bytes]:
        values = np.asarray(gene)
        if values.dtype.kind not in 'biuf' or values.ndim != 1:
            return None
        if values.nbytes > self.max_key_bytes:
            return None
        return values.dtype.str.encode() + b':' + values.tobytes()

    def get(self, key: Optional[bytes]) -> Optional[float]:
        if key is None or key not in self.entries:
            return None
        self.entries.move_to_end(key)
        return self.entries[key]

    def put(self, key: Optional[bytes], value: float) -> None:
        if key is None or len(key) > self.max_key_bytes:
            return
        if key in self.entries:
            self.entries.move_to_end(key)
            self.entries[key] = value
            return
        self.entries[key] = value
        self.key_bytes += len(key)
        while len(self.entries) > self.capacity or self.key_bytes > self.max_key_bytes:
            removed, _ = self.entries.popitem(last=False)
            self.key_bytes -= len(removed)


class _ClassMaskCache:
    """Bounded, evaluation-local cache of consequent-match masks."""

    def __init__(self, y: np.ndarray, max_bytes: int = 1024 * 1024) -> None:
        self.y = y
        self.max_bytes = max_bytes
        self.entries = {}
        self.bytes = 0

    def get(self, consequent) -> np.ndarray:
        mask = self.entries.get(consequent)
        if mask is not None:
            return mask
        mask = self.y == consequent
        if mask.nbytes <= self.max_bytes - self.bytes:
            self.entries[consequent] = mask
            self.bytes += mask.nbytes
        return mask


class _FiringCache:
    """
    Bounded fit-local cache of one rule's firing column.

    A rule's firing strengths depend only on its effective antecedents and on
    the antecedent memberships, so the entries stay valid exactly as long as the
    memberships do: fixed partitions only.  With optimized partitions the same
    antecedent indexes denote different fuzzy sets on every candidate, and reuse
    is measurably wrong, so callers must not install this cache there.

    ``max_bytes`` bounds the retained firing columns alone.  It is not a bound
    on process memory, dictionary overhead or the caller's own arrays.
    """

    def __init__(self, max_bytes: int = 8 * 1024 * 1024) -> None:
        self.max_bytes = max_bytes
        self.entries = OrderedDict()
        self.bytes = 0

    def get(self, key: bytes) -> Optional[np.ndarray]:
        column = self.entries.get(key)
        if column is not None:
            self.entries.move_to_end(key)
        return column

    def put(self, key: bytes, column: np.ndarray) -> None:
        if column.nbytes > self.max_bytes:
            return
        # Rules of different classes can share antecedents, so the same key can
        # be stored twice within one candidate: drop the old size first.
        replaced = self.entries.pop(key, None)
        if replaced is not None:
            self.bytes -= replaced.nbytes
        self.entries[key] = column
        self.bytes += column.nbytes
        while self.bytes > self.max_bytes:
            _, removed = self.entries.popitem(last=False)
            self.bytes -= removed.nbytes


#: Populations of genotypes the fitness cache holds when the fit's size is known.
_FITNESS_CACHE_POPULATIONS = 4


@contextmanager
def _fitness_cache_scope(problem, enabled: bool,
                         population: Optional[int] = None) -> Iterator[None]:
    """
    Discard all memoized fitness on optimizer return or exception.

    ``population`` sizes the genotype cache to span several generations: the
    default 256 entries hold less than one large EvoX population. The key
    payload limit grows with it, so retained keys stay proportional to the
    chromosomes the optimizer itself keeps.
    """
    if not enabled:
        yield
        return
    capacity, key_bytes = 256, 1024 * 1024
    if population is not None:
        capacity = max(capacity, _FITNESS_CACHE_POPULATIONS * int(population))
        n_var = getattr(problem, 'n_var', None)
        if n_var:
            key_bytes = max(key_bytes, capacity * (8 * int(n_var) + 8))
    problem._fitness_cache = _FitnessCache(capacity, key_bytes)
    # Firing reuse needs memberships that do not change between candidates.
    fixed_partitions = getattr(problem, 'lvs', None) is not None
    if fixed_partitions:
        problem._firing_cache = _FiringCache()
    # Populated on first use by the population evaluator, which owns the class.
    # Creating the slot here keeps the route choice fit-local, like the caches.
    problem._route_probe = None
    # Likewise the exact device objective and its verification state, which
    # also releases the device copies of the training data with the fit.
    problem._torch_route = None
    try:
        yield
    finally:
        del problem._fitness_cache
        del problem._route_probe
        del problem._torch_route
        if fixed_partitions:
            del problem._firing_cache


def _dominance_reference(firing: np.ndarray, y: np.ndarray, consequents: np.ndarray,
                        mask_cache: Optional[_ClassMaskCache] = None) -> np.ndarray:
    """
    Match evalRuleBase's support/confidence reductions, including T2 pooling.

    Retained as the parity oracle for the grouped implementation below and as
    the benchmark's ``--legacy-dominance`` path.
    """
    scores = np.empty(len(consequents))
    for index, consequent in enumerate(consequents):
        values = firing[:, index]
        match = mask_cache.get(consequent) if mask_cache is not None else y == consequent
        if firing.ndim == 3:
            support = np.mean([
                np.mean(values[:, 0] * match),
                np.mean(values[:, 1] * match),
            ])
        else:
            support = np.mean(values * match)
        denominator = np.sum(values)
        confidence = np.sum(values[match]) / denominator if denominator != 0 else 0.0
        scores[index] = support * confidence
    return scores


def _dominance(firing: np.ndarray, y: np.ndarray, consequents: np.ndarray,
               mask_cache: Optional[_ClassMaskCache] = None) -> np.ndarray:
    """
    Grouped equivalent of :func:`_dominance_reference`.

    Rules that share a consequent are reduced together, but every reduced axis
    is made contiguous first so NumPy applies the same pairwise summation as
    the per-rule reference.  The T2 denominator is the one reduction whose
    reference operand is a strided ``(samples, 2)`` view; no vectorised form
    of it reproduced that summation order, so it keeps its per-rule call.
    """
    n_rules = len(consequents)
    scores = np.empty(n_rules)
    if n_rules == 0:
        return scores
    interval = firing.ndim == 3
    if interval:
        lower = np.ascontiguousarray(firing[:, :, 0].T)
        upper = np.ascontiguousarray(firing[:, :, 1].T)
        denominator = np.fromiter(
            (np.sum(firing[:, index]) for index in range(n_rules)),
            dtype=np.float64, count=n_rules)
    else:
        table = np.ascontiguousarray(firing.T)
        denominator = np.sum(table, axis=1)
    consequents = np.asarray(consequents)
    # Rules arrive grouped by consequent, so each group is a contiguous run and
    # can be sliced as a view. Runs are handled independently, so a consequent
    # that appears in several runs, or in no particular order, stays correct.
    for start, stop in _consequent_runs(consequents):
        rows = slice(start, stop)
        width = stop - start
        consequent = consequents[start]
        match = mask_cache.get(consequent) if mask_cache is not None else y == consequent
        if interval:
            support = (np.mean(lower[rows] * match, axis=1)
                       + np.mean(upper[rows] * match, axis=1)) / 2
            # Select samples first: the pooled copy then covers only this
            # class's samples and rules instead of the whole firing array.
            selected = np.ascontiguousarray(
                np.moveaxis(firing[match][:, rows], 1, 0))
            numerator = np.sum(selected.reshape(width, -1), axis=1)
        else:
            support = np.mean(table[rows] * match, axis=1)
            numerator = np.sum(np.ascontiguousarray(table[rows][:, match]), axis=1)
        divisor = denominator[rows]
        confidence = np.divide(numerator, divisor, out=np.zeros(width),
                               where=divisor != 0)
        scores[rows] = support * confidence
    return scores


def _consequent_runs(consequents: np.ndarray) -> Iterator[tuple]:
    """Yield the (start, stop) bounds of each maximal equal-consequent run."""
    start = 0
    for index in range(1, len(consequents)):
        if consequents[index] != consequents[index - 1]:
            yield start, index
            start = index
    yield start, len(consequents)


def _association(firing: np.ndarray, scores: np.ndarray,
                 weights: np.ndarray, ds_mode: int) -> np.ndarray:
    if ds_mode == 1:
        result = firing
    else:
        multiplier = scores if ds_mode == 0 else weights
        if firing.ndim == 3:
            multiplier = multiplier[None, :, None]
        result = firing * multiplier
    return np.mean(result, axis=2) if firing.ndim == 3 else result


class _LabelDomain:
    """
    Immutable label layout for one problem's integer training labels.

    Classes are the consequent indexes ``0 .. n_classes - 1``; a candidate can
    also predict the unknown ``-1``.  Shifting every label by one gives each
    class and the unknown a counting slot, the unknown at index 0, so per-class
    counts need no sorting.  Created once per problem; ``y`` must not change
    while it is in use, exactly like the precomputed memberships.
    """

    __slots__ = ('size', 'shifted_y', 'class_sizes')

    def __init__(self, y: np.ndarray, n_classes: int) -> None:
        self.size = n_classes + 1  # one slot per class, plus unknown at index 0
        self.shifted_y = np.asarray(y) + 1
        self.class_sizes = np.bincount(self.shifted_y, minlength=self.size)[1:]

    @classmethod
    def build(cls, y: np.ndarray, n_classes: int) -> Optional['_LabelDomain']:
        """Return None for labels the counting layout cannot represent."""
        values = np.asarray(y)
        if values.dtype.kind not in 'biu' or values.size == 0:
            return None
        if values.min() < -1 or values.max() >= n_classes:
            return None
        return cls(values, n_classes)


def _class_counts(y: np.ndarray, prediction: np.ndarray, n_classes: int) -> tuple:
    """
    Per-class true positives, predictions and samples, as exact integers.

    Classes are the consequent indexes ``0 .. n_classes - 1``.  An unknown
    prediction (-1) counts for no class, so it is a miss for its sample's class.
    """
    domain = _LabelDomain.build(y, n_classes)
    if domain is not None:
        return _class_counts_encoded(prediction, domain)
    classes = range(n_classes)
    predicted = np.array([np.count_nonzero(prediction == k) for k in classes])
    actual = np.array([np.count_nonzero(y == k) for k in classes])
    true_positive = np.array([np.count_nonzero((prediction == k) & (y == k)) for k in classes])
    return true_positive, predicted, actual


def _class_counts_encoded(prediction: np.ndarray, domain: _LabelDomain) -> tuple:
    """:func:`_class_counts` for labels with a counting layout."""
    shifted = prediction + 1
    hits = domain.shifted_y[shifted == domain.shifted_y]
    true_positive = np.bincount(hits, minlength=domain.size)[1:]
    predicted = np.bincount(shifted, minlength=domain.size)[1:]
    return true_positive, predicted, domain.class_sizes


def _macro_f1(true_positive, predicted, actual) -> np.ndarray:
    """
    Macro F1 from per-class counts shaped ``(..., classes)``.

    A class's F1 is ``2 TP / (predicted + actual)``, and 0 when the class is
    neither present nor predicted.  The classes are added one at a time, left
    to right, so the result does not depend on the array layout: every
    evaluator, scalar or batched, rounds the same way.
    """
    true_positive = np.asarray(true_positive)
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    n_classes = true_positive.shape[-1]
    total = np.zeros(true_positive.shape[:-1])
    for klass in range(n_classes):
        denominator = predicted[..., klass] + actual[..., klass]
        total = total + np.divide(2 * true_positive[..., klass], denominator,
                                  out=np.zeros(total.shape), where=denominator != 0)
    return total / n_classes


def _penalized_objective(primary, covered, conditions, n_classes: int,
                         max_conditions: int, alpha: float, beta: float):
    """
    Subtract the class-coverage and compactness penalties from ``primary``.

    ``covered`` counts the classes that keep at least one rule and
    ``conditions`` the antecedent conditions of the kept rules; ``beta``
    weighs the share of classes without rules and ``alpha`` the share of
    ``max_conditions`` used.  Works on scalars and arrays alike.  Every
    evaluator finishes here, so all of them round the same way.
    """
    uncovered = (n_classes - covered) / n_classes
    return primary - beta * uncovered - alpha * (conditions / max_conditions)


def score_rulebase(rulebase, X: np.ndarray, y: np.ndarray, tolerance: float,
                   alpha: float, beta: float, precomputed_truth=None, *,
                   max_conditions: int) -> float:
    """
    Compute the standard objective with one firing-strength evaluation.

    The objective is the macro F1 of the pruned rule base on the training data,
    minus the penalties of :func:`_penalized_objective`; ``y`` holds consequent
    indexes.  Pruning uses winners *before* removal. Final predictions select
    winners among surviving rules, preserving class/rule order and first-index
    ties.
    """
    n_classes = len(rulebase.get_rulebases())
    all_rules = rulebase.get_rules()
    if not all_rules:
        return float(_penalized_objective(0.0, 0, 0, n_classes, max_conditions, alpha, beta))
    if precomputed_truth is None:
        precomputed_truth = rules.compute_antecedents_memberships(rulebase.antecedents, X)
    firing = rulebase.compute_firing_strengths(X, precomputed_truth=precomputed_truth)
    y = np.asarray(y)
    consequents = np.asarray(rulebase.get_consequents(), dtype=int)
    weights = rulebase.get_weights() if rulebase.ds_mode == 2 else np.ones(len(all_rules))
    mask_cache = _ClassMaskCache(y)
    scores = _dominance(firing, y, consequents, mask_cache)
    association = _association(firing, scores, weights, rulebase.ds_mode)
    winners = np.argmax(association, axis=1)
    eligible = (np.max(association, axis=1) != 0.0 if rulebase.allow_unknown
                else np.ones(len(y), dtype=bool))
    correct = (consequents[winners] == y) & eligible
    correct_wins = np.bincount(winners[correct], minlength=len(all_rules))
    wins = np.bincount(winners[eligible], minlength=len(all_rules))

    # Keep the reference's strict '< tolerance' and zero-accuracy removal.
    for index, rule in enumerate(all_rules):
        rule.score = scores[index]
        rule.accuracy = correct_wins[index] / wins[index] if wins[index] else 0.0
    rulebase.purge_rules(tolerance)
    survivors = rulebase.get_rules()
    if not survivors:
        return float(_penalized_objective(0.0, 0, 0, n_classes, max_conditions, alpha, beta))
    retained = {id(rule) for rule in survivors}
    keep = np.asarray([id(rule) in retained for rule in all_rules])
    # Match the contiguous layout of a fresh reference firing calculation.
    firing = np.ascontiguousarray(firing[:, keep])
    consequents = consequents[keep]
    scores = _dominance(firing, y, consequents, mask_cache)
    for rule, score in zip(survivors, scores):
        rule.score = score
    association = _association(firing, scores, weights[keep], rulebase.ds_mode)
    prediction = consequents[np.argmax(association, axis=1)]
    if rulebase.allow_unknown:
        prediction[np.max(association, axis=1) == 0.0] = -1
    primary = _macro_f1(*_class_counts(y, prediction, n_classes))
    covered = np.count_nonzero(np.bincount(consequents, minlength=n_classes))
    conditions = sum(np.count_nonzero(np.asarray(rule.antecedents) != -1) for rule in survivors)
    return float(_penalized_objective(primary, covered, conditions, n_classes,
                                      max_conditions, alpha, beta))


def score_rulebase_objects(rulebase, X: np.ndarray, y: np.ndarray, tolerance: float,
                           alpha: float, beta: float, precomputed_truth=None, *,
                           max_conditions: int) -> float:
    """
    The objective of :func:`score_rulebase` through ``evalRuleBase``.

    For rule bases ``score_rulebase`` does not reproduce, such as general
    Type-2 ones, and for labels that are not consequent indexes.
    """
    n_classes = len(rulebase.get_rulebases())
    if not rulebase.get_rules():
        return float(_penalized_objective(0.0, 0, 0, n_classes, max_conditions, alpha, beta))
    if precomputed_truth is None:
        precomputed_truth = rules.compute_antecedents_memberships(rulebase.antecedents, X)
    evaluator = eval_rules.evalRuleBase(rulebase, X, y, precomputed_truth=precomputed_truth)
    evaluator.add_full_evaluation()
    rulebase.purge_rules(tolerance)
    survivors = rulebase.get_rules()
    if not survivors:
        return float(_penalized_objective(0.0, 0, 0, n_classes, max_conditions, alpha, beta))
    evaluator.add_rule_weights()
    prediction = np.asarray(rulebase.winning_rule_predict(X, precomputed_truth=precomputed_truth))
    primary = _macro_f1(*_class_counts(np.asarray(y), prediction, n_classes))
    covered = sum(1 for rule_base in rulebase.get_rulebases() if len(rule_base) > 0)
    conditions = sum(np.count_nonzero(np.asarray(rule.antecedents) != -1) for rule in survivors)
    return float(_penalized_objective(primary, covered, conditions, n_classes,
                                      max_conditions, alpha, beta))
