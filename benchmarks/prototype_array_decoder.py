"""Correctness-only prototype for the fixed-partition array evaluator path.

This is deliberately not imported by the package or training code.  It records
the effective phenotype which ``FitRuleBase._construct_ruleBase`` creates for
fixed T1/T2 partitions: slot overwrite order, term clipping, disabled rules,
class ordering, and weights.  Tuple duplicate keys match chromosomes decoded
by FitRuleBase (plain RuleSimple objects with no modifiers), but do *not* claim
parity with arbitrary RuleSimple subclasses whose hash/equality contract differs.

Run ``python benchmarks/prototype_array_decoder.py`` for structural notes; this
does not time anything.
"""
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INNER = ROOT / "ex_fuzzy" / "ex_fuzzy"
if str(INNER) not in sys.path:
    sys.path.insert(0, str(INNER))


@dataclass(frozen=True)
class DecodedRuleArrays:
    """Object-free effective fixed-partition chromosome representation."""

    antecedents: np.ndarray  # (rules, features), class-major and deduplicated
    consequents: np.ndarray  # (rules,)
    weights: np.ndarray  # (rules,); 1 unless ds_mode == 2
    class_counts: np.ndarray  # includes empty classes
    source_rows: np.ndarray  # original chromosome rule slots retained


def decode_fixed_partitions(x: np.ndarray, *, n_rules: int, n_ants: int,
                            n_features: int, n_classes: int,
                            n_lv_possible: Iterable[int], ds_mode: int = 0) -> DecodedRuleArrays:
    """Decode fixed-partition T1/T2 chromosomes without fuzzy/rule objects.

    This intentionally follows the current decoder's sequential writes: if a
    chromosome selects a feature more than once in a rule, its final slot wins.
    It is defined only for valid fixed-partition chromosomes produced within
    their pymoo bounds.
    """
    values = np.asarray(x).astype(int, copy=True).reshape(-1)
    n_lv = np.asarray(tuple(n_lv_possible), dtype=int)
    if n_lv.shape != (n_features,):
        raise ValueError("n_lv_possible must contain one count per feature")
    fourth = 2 * n_rules * n_ants
    required = fourth + n_rules + (n_rules if ds_mode == 2 else 0)
    if values.size < required:
        raise ValueError("fixed-partition chromosome is shorter than its required slices")

    features = values[:n_rules * n_ants].reshape(n_rules, n_ants)
    terms = values[n_rules * n_ants:fourth].reshape(n_rules, n_ants)
    raw_consequents = values[fourth:fourth + n_rules]
    raw_weights = (values[fourth + n_rules:fourth + 2 * n_rules] / 100.0
                   if ds_mode == 2 else np.ones(n_rules))
    if np.any(features < 0) or np.any(features >= n_features):
        raise ValueError("feature indices must be in the fixed decoder bounds")
    if np.any(raw_consequents < -1) or np.any(raw_consequents >= n_classes):
        raise ValueError("consequents must be in the fixed decoder bounds")

    effective = np.full((n_rules, n_features), -1, dtype=int)
    # This scalar loop is intentional: it is the reference's overwrite order.
    for row in range(n_rules):
        for slot in range(n_ants):
            feature = features[row, slot]
            effective[row, feature] = min(terms[row, slot], n_lv[feature] - 1)

    active = (raw_consequents != -1) & np.any(effective != -1, axis=1)
    rows, cons, weights = [], [], []
    # RuleBase construction is class-major; duplicate removal is per class and
    # keeps the first value for ordinary generated RuleSimple instances.
    for klass in range(n_classes):
        seen = set()
        for source in np.flatnonzero(active & (raw_consequents == klass)):
            # The generated RuleSimple objects have consequent=0.  In mode 2
            # their weight is assigned before RuleBase duplicate lookup and is
            # included by RuleSimple.__str__/__hash__, despite __eq__ ignoring
            # it.  Preserve that observable, hash-inconsistent behavior.
            key = (tuple(effective[source].tolist()), float(raw_weights[source])) if ds_mode == 2 else tuple(effective[source].tolist())
            if key in seen:
                continue
            seen.add(key)
            rows.append(source)
            cons.append(klass)
            weights.append(raw_weights[source])
    source_rows = np.asarray(rows, dtype=int)
    antecedents = (effective[source_rows] if len(source_rows)
                   else np.empty((0, n_features), dtype=int))
    consequents = np.asarray(cons, dtype=int)
    result_weights = np.asarray(weights, dtype=float)
    class_counts = np.bincount(consequents, minlength=n_classes)
    return DecodedRuleArrays(antecedents, consequents, result_weights,
                             class_counts, source_rows)


def prune_mask(scores: np.ndarray, accuracies: np.ndarray, tolerance: float) -> np.ndarray:
    """Array equivalent of RuleBase.prune_bad_rules for supplied rule metrics."""
    score_values = np.asarray(scores)
    if score_values.ndim > 1:
        score_values = np.mean(score_values, axis=tuple(range(1, score_values.ndim)))
    # RuleBase.prune_bad_rules removes only ``score < tolerance``.  In
    # particular, a NaN score is retained because ``NaN < tolerance`` is False.
    return ~(score_values < tolerance) & (np.asarray(accuracies) != 0.0)


def complexity_metrics(decoded: DecodedRuleArrays, scores: np.ndarray,
                       tolerance: float) -> tuple[float, float]:
    """Match evalRuleBase size_antecedents/effective_rulesize formulas.

    The caller supplies dominance scores in the decoder's class-major order.
    """
    scores = np.asarray(scores)
    score_values = (np.mean(scores, axis=tuple(range(1, scores.ndim)))
                    if scores.ndim > 1 else scores)
    if score_values.shape != decoded.consequents.shape:
        raise ValueError("scores must align with decoded rules")
    if np.any(decoded.class_counts == 0):
        return 0.0, 0.0
    selected = score_values > tolerance  # evalRuleBase uses strict > here.
    widths = np.sum(decoded.antecedents != -1, axis=1)
    feature_count = decoded.antecedents.shape[1]
    possible_size = int(np.sum(selected) * feature_count)
    effective_size = int(np.sum(np.where(widths[selected] == 0,
                                         feature_count, widths[selected])))
    size = 0.0 if possible_size == 0 else 1.0 - effective_size / possible_size
    effective_rules = int(np.sum(selected & (widths != 0)))
    rules = len(decoded.consequents)
    rule_size = 0.0 if rules == 0 else effective_rules / rules
    return size, rule_size


if __name__ == "__main__":
    print("A04 prototype: fixed T1/T2 array decoder only; no production integration or timing.")
    print("A05: direct membership still needs endpoint/NaN parity against FS/IVFS.")
    print("A06: vector reductions must preserve T2 lower/upper reduction order.")
    print("A07: integer-label pre-encoding needs an unknown (-1) bucket.")
    print("Duplicate caveat: tuple keys do not model arbitrary custom hash collisions.")
