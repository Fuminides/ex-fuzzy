"""Object-free evaluation of the built-in T1/T2 classification objective.

This is the A04/A08/A09 path: it decodes a chromosome straight into integer
antecedent/consequent/weight arrays and scores it without building any
``RuleSimple``, ``RuleBase`` or ``MasterRuleBase`` object.  Fuzzy variables are
still decoded by ``FitRuleBase._decode_antecedents`` so partition normalization
keeps exactly one implementation.

``FitRuleBase._construct_ruleBase`` plus ``_fitness.score_rulebase`` remains the
oracle *and* the fallback: every function here returns ``None`` (or the caller
refuses the candidate) as soon as an input leaves the narrow supported case, so
custom losses, modifiers, temporal partitions, GT2 and custom t-norms keep their
previous behavior.
"""
import numpy as np
from typing import Optional

try:
    from . import rules
    from ._fitness import (_ClassMaskCache, _FiringCache, _LabelDomain,
                           _association, _dominance, _mcc, _mcc_encoded)
except ImportError:  # pragma: no cover - direct module execution
    import rules
    from _fitness import (_ClassMaskCache, _FiringCache, _LabelDomain,
                          _association, _dominance, _mcc, _mcc_encoded)

class _WeightedRuleKey:
    """Reproduce ``RuleSimple`` dictionary behavior for weighted rules.

    ``RuleSimple.__hash__`` hashes ``str(rule)``, which includes the weight,
    while ``__eq__`` compares only antecedents and consequent.  Duplicate
    removal therefore depends on hash *collisions* between rules with equal
    antecedents and different weights.  Keying a dictionary on this class
    reproduces that exactly, because it hashes the identical string and
    compares with the identical rule.
    """

    __slots__ = ('antecedents', 'text')

    def __init__(self, antecedents: tuple, text: str) -> None:
        self.antecedents = antecedents
        self.text = text

    def __hash__(self) -> int:
        return hash(self.text)

    def __eq__(self, other) -> bool:
        return self.antecedents == other.antecedents


class _DecodedCandidate:
    """Effective phenotype of one chromosome, in class-major rule order."""

    __slots__ = ('antecedents', 'consequents', 'weights')

    def __init__(self, antecedents: np.ndarray, consequents: np.ndarray,
                 weights: np.ndarray) -> None:
        self.antecedents = antecedents
        self.consequents = consequents
        self.weights = weights


def decode_rule_arrays(x: np.ndarray, n_rules: int, n_ants: int, n_features: int,
                       n_classes: int, term_counts: np.ndarray, ds_mode: int,
                       fourth_pointer: int) -> Optional[_DecodedCandidate]:
    """Decode the rule slices of a chromosome exactly as ``_construct_ruleBase``.

    Returns None when a consequent leaves the decoder's valid range, so the
    object decoder runs and raises its own assertion.
    """
    features = x[:n_rules * n_ants].reshape(n_rules, n_ants)
    terms = x[n_rules * n_ants:2 * n_rules * n_ants].reshape(n_rules, n_ants)
    consequents = x[fourth_pointer:fourth_pointer + n_rules]
    if np.any(consequents >= n_classes):
        return None
    if np.any(features < 0) or np.any(features >= n_features):
        return None
    if ds_mode == 2:
        raw_weights = x[fourth_pointer + n_rules:fourth_pointer + 2 * n_rules] / 100
    else:
        raw_weights = None

    effective = np.full((n_rules, n_features), -1, dtype=np.int64)
    clipped = np.minimum(terms, term_counts[features] - 1)
    # The reference writes slots left to right, so a feature selected twice in
    # one rule keeps its last slot.  Advanced assignment has the same rule.
    rows = np.repeat(np.arange(n_rules), n_ants)
    effective[rows, features.reshape(-1)] = clipped.reshape(-1)
    active = (consequents != -1) & np.any(effective != -1, axis=1)

    order, kept_consequents, kept_weights = [], [], []
    for klass in range(n_classes):
        seen = {}
        for source in np.flatnonzero(active & (consequents == klass)):
            row = effective[source]
            if ds_mode == 2:
                antecedents = tuple(row.tolist())
                weight = float(raw_weights[source])
                key = _WeightedRuleKey(
                    antecedents,
                    'Rule: antecedents: %s consequent: 0 weight: %s'
                    % (list(antecedents), weight))
            else:
                # Without weights the rule string, and therefore the reference's
                # hash, depends only on the antecedents: raw bytes key the same
                # equivalence classes as the tuple, for less work per rule.
                key = row.tobytes()
                weight = 1.0
            if seen.setdefault(key, source) != source:
                continue
            order.append(source)
            kept_consequents.append(klass)
            kept_weights.append(weight)
    if not order:
        return _DecodedCandidate(np.empty((0, n_features), dtype=np.int64),
                                 np.empty(0, dtype=int), np.empty(0))
    selected = np.asarray(order)
    return _DecodedCandidate(effective[selected],
                             np.asarray(kept_consequents, dtype=int),
                             np.asarray(kept_weights, dtype=float))


def firing_strengths(antecedents: np.ndarray, truth, n_samples: int, interval: bool,
                     cache: Optional[_FiringCache] = None,
                     packed: Optional[tuple] = None) -> Optional[np.ndarray]:
    """Firing strengths for decoded antecedents, matching the object path.

    T2 reuses the shared gathered kernel.  T1 repeats the reference's per-rule
    ``np.prod`` over the full feature axis, including don't-care ones, so the
    reduction order and rounding are unchanged.

    With a ``cache`` the already known rules are copied instead of recomputed and
    only the remaining ones are gathered.  A rule's column does not depend on
    which other rules share its gather, so cached and freshly computed columns
    are identical values.
    """
    n_rules, n_features = antecedents.shape
    tail = (2,) if interval else ()
    if cache is not None:
        return _cached_firing(antecedents, truth, n_samples, tail, cache, packed)
    gathered = rules._gather_firing_from_arrays(antecedents, truth, n_samples, tail,
                                                packed)
    if gathered is not None or interval or truth is None:
        return gathered
    if not n_features:
        return None
    lengths = np.asarray([len(feature) for feature in truth])
    if len(truth) != n_features or np.any(antecedents >= lengths):
        return None
    result = np.empty((n_samples, n_rules))
    scratch = np.empty((n_samples, n_features))
    for index in range(n_rules):
        row = antecedents[index]
        used = 0
        for feature in range(n_features):
            term = row[feature]
            if term >= 0:
                values = truth[feature]
                if not isinstance(values, (list, tuple, np.ndarray)):
                    return None
                scratch[:, feature] = values[term]
                used += 1
            else:
                scratch[:, feature] = 1.0
        if used == 0:
            scratch[:, n_features - 1] = 0.0
        result[:, index] = np.prod(scratch, axis=1)
    return result


def _cached_firing(antecedents: np.ndarray, truth, n_samples: int, tail: tuple,
                   cache: _FiringCache,
                   packed: Optional[tuple] = None) -> Optional[np.ndarray]:
    """Assemble firing columns, gathering only the rules the cache is missing."""
    keys = [row.tobytes() for row in antecedents]
    columns = [cache.get(key) for key in keys]
    missing = [index for index, column in enumerate(columns) if column is None]
    if missing:
        computed = rules._gather_firing_from_arrays(
            antecedents[missing], truth, n_samples, tail, packed)
        if computed is None:
            return None
        for slot, index in enumerate(missing):
            column = np.ascontiguousarray(computed[:, slot])
            columns[index] = column
            cache.put(keys[index], column)
    result = np.empty((n_samples, len(columns)) + tail)
    for index, column in enumerate(columns):
        result[:, index] = column
    return result


def _complexity(antecedents: np.ndarray, consequents: np.ndarray,
                scores: np.ndarray, n_classes: int,
                tolerance: float) -> tuple[float, float]:
    """Array form of size_antecedents_eval and effective_rulesize_eval."""
    if np.any(np.bincount(consequents, minlength=n_classes)[:n_classes] == 0):
        return 0.0, 0.0  # evalRuleBase returns 0.0 as soon as one class is empty.
    n_features = antecedents.shape[1]
    selected = scores > tolerance  # evalRuleBase uses a strict comparison here.
    widths = np.count_nonzero(antecedents != -1, axis=1)
    possible = int(np.count_nonzero(selected)) * n_features
    chosen = widths[selected]
    effective = int(np.sum(np.where(chosen == 0, n_features, chosen)))
    size = 1 - effective / possible if possible else 0.0
    effective_rules = int(np.count_nonzero(selected & (widths != 0)))
    rulesize = effective_rules / len(consequents) if len(consequents) else 0.0
    return size, rulesize


def score_candidate(decoded: _DecodedCandidate, truth, X: np.ndarray, y: np.ndarray,
                    n_classes: int, ds_mode: int, allow_unknown: bool,
                    tolerance: float, alpha: float, beta: float,
                    interval: bool, labels: Optional[_LabelDomain] = None,
                    firing_cache: Optional[_FiringCache] = None,
                    packed: Optional[tuple] = None) -> Optional[float]:
    """Objective value for a decoded candidate, or None if unsupported.

    Mirrors ``_fitness.score_rulebase`` step by step, including pruning on the
    pre-removal winners and the second dominance pass over the survivors.
    """
    n_rules = len(decoded.consequents)
    if n_rules == 0:
        return 0.0
    firing = firing_strengths(decoded.antecedents, truth, len(X), interval,
                              firing_cache, packed)
    if firing is None:
        return None
    consequents = decoded.consequents
    weights = decoded.weights if ds_mode == 2 else np.ones(n_rules)
    mask_cache = _ClassMaskCache(y)
    scores = _dominance(firing, y, consequents, mask_cache)
    association = _association(firing, scores, weights, ds_mode)
    winners = np.argmax(association, axis=1)
    eligible = (np.max(association, axis=1) != 0.0 if allow_unknown
                else np.ones(len(y), dtype=bool))
    correct = (consequents[winners] == y) & eligible
    correct_wins = np.bincount(winners[correct], minlength=n_rules)
    wins = np.bincount(winners[eligible], minlength=n_rules)
    accuracy = np.divide(correct_wins, wins, out=np.zeros(n_rules),
                         where=wins != 0)
    # prune_bad_rules drops 'score < tolerance or accuracy == 0.0'.  A NaN score
    # is retained because 'NaN < tolerance' is False.
    keep = ~(scores < tolerance) & (accuracy != 0.0)
    if not keep.any():
        return 0.0
    firing = np.ascontiguousarray(firing[:, keep])
    consequents = consequents[keep]
    scores = _dominance(firing, y, consequents, mask_cache)
    association = _association(firing, scores, weights[keep], ds_mode)
    prediction = consequents[np.argmax(association, axis=1)]
    if allow_unknown:
        prediction[np.max(association, axis=1) == 0.0] = -1
    result = _mcc(y, prediction) if labels is None else _mcc_encoded(prediction, labels)
    if alpha != 0.0 or beta != 0.0:
        # Accumulate exactly as score_rulebase does: one addition per penalty.
        size, rulesize = _complexity(decoded.antecedents[keep], consequents,
                                     scores, n_classes, tolerance)
        if alpha != 0.0:
            result += alpha * size
        if beta != 0.0:
            result += beta * rulesize
    return result
