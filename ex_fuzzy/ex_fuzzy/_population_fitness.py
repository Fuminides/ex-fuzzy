"""Exact population scoring for small fixed-partition T1 problems.

The scalar array evaluator remains the oracle and handles every unsupported
case. This module only adds a population axis around the same chromosome
decode, firing, dominance, pruning and MCC operations. Reductions keep each
rule's sample or feature axis contiguous so NumPy uses the scalar evaluator's
summation order.
"""
from typing import Optional

import numpy as np

try:
    from . import rules
    from ._fitness import _FiringCache, _LabelDomain
except ImportError:  # pragma: no cover - direct module execution
    import rules
    from _fitness import _FiringCache, _LabelDomain


_MAX_SAMPLES = 512
_GATHER_BUDGET = 32 * 1024 * 1024


def supports_shape(population: int, n_samples: int, n_rules: int,
                   n_features: int,
                   gather_budget: int = _GATHER_BUDGET) -> bool:
    """Whether the bounded population intermediates fit this workload."""
    bytes_per_candidate = 8 * n_samples * n_rules * n_features
    return (population >= 2 and 0 < n_samples <= _MAX_SAMPLES
            and bytes_per_candidate > 0
            and population * bytes_per_candidate <= gather_budget)


def _decode_population(genes: np.ndarray, n_rules: int, n_ants: int,
                       n_features: int, n_classes: int,
                       term_counts: np.ndarray, fourth_pointer: int):
    """Decode and pad effective rules in class-major order."""
    if (genes.ndim != 2 or genes.dtype.kind not in 'biu'
            or genes.shape[1] != fourth_pointer + n_rules
            or term_counts.shape != (n_features,)):
        return None
    population = len(genes)
    features = genes[:, :n_rules * n_ants].reshape(population, n_rules, n_ants)
    terms = genes[:, n_rules * n_ants:2 * n_rules * n_ants].reshape(
        population, n_rules, n_ants)
    consequents = genes[:, fourth_pointer:fourth_pointer + n_rules]
    if (np.any(consequents < -1) or np.any(consequents >= n_classes)
            or np.any(features < 0) or np.any(features >= n_features)
            or np.any(terms < -1)):
        return None

    effective = np.full((population, n_rules, n_features), -1, dtype=np.int64)
    batch_rows = np.arange(population)[:, None]
    rule_rows = np.arange(n_rules)[None, :]
    for slot in range(n_ants):
        selected_features = features[:, :, slot]
        selected_terms = np.minimum(
            terms[:, :, slot], term_counts[selected_features] - 1)
        effective[batch_rows, rule_rows, selected_features] = selected_terms

    active = (consequents != -1) & np.any(effective != -1, axis=2)
    for source in range(1, n_rules):
        same = np.all(effective[:, :source] == effective[:, source, None], axis=2)
        duplicate = (active[:, :source]
                     & (consequents[:, :source] == consequents[:, source, None])
                     & same)
        active[:, source] &= ~np.any(duplicate, axis=1)

    key = np.where(active, consequents, n_classes)
    order = np.argsort(key, axis=1, kind='stable')
    effective = np.take_along_axis(effective, order[:, :, None], axis=1)
    consequents = np.take_along_axis(consequents, order, axis=1)
    active = np.take_along_axis(active, order, axis=1)
    return effective, consequents, active


def _firing_population(antecedents: np.ndarray, active: np.ndarray,
                       packed: tuple, n_samples: int,
                       cache: Optional[_FiringCache]) -> Optional[np.ndarray]:
    """Gather each distinct effective rule once and scatter it to candidates."""
    population, n_rules, n_features = antecedents.shape
    inverse = np.full(population * n_rules, -1, dtype=np.intp)
    unique_rows, unique_keys, positions = [], [], {}
    flat_rows = antecedents.reshape(-1, n_features)
    for position in np.flatnonzero(active.reshape(-1)):
        row = flat_rows[position]
        key = row.tobytes()
        index = positions.get(key)
        if index is None:
            index = len(unique_rows)
            positions[key] = index
            unique_rows.append(row)
            unique_keys.append(key)
        inverse[position] = index

    if not unique_rows:
        return np.zeros((population, n_samples, n_rules))
    unique_rows = np.asarray(unique_rows)
    columns = [cache.get(key) if cache is not None else None for key in unique_keys]
    missing = [index for index, column in enumerate(columns) if column is None]
    if missing:
        computed = rules._gather_firing_from_arrays(
            unique_rows[missing], None, n_samples, (), packed)
        if computed is None:
            return None
        for slot, index in enumerate(missing):
            column = np.ascontiguousarray(computed[:, slot])
            columns[index] = column
            if cache is not None:
                cache.put(unique_keys[index], column)
    unique_firing = np.column_stack(columns)
    unique_firing = np.column_stack((unique_firing, np.zeros(n_samples)))
    inverse[inverse < 0] = len(columns)
    firing = unique_firing[:, inverse].reshape(
        n_samples, population, n_rules).transpose(1, 0, 2)
    return np.ascontiguousarray(firing)


def _dominance_population(firing: np.ndarray, y: np.ndarray,
                          consequents: np.ndarray, active: np.ndarray,
                          n_classes: int) -> np.ndarray:
    """Exact T1 dominance with candidates and rules batched together."""
    table = np.ascontiguousarray(firing.transpose(0, 2, 1))
    denominator = np.sum(table, axis=2)
    scores = np.zeros(consequents.shape)
    for consequent in range(n_classes):
        selected_rules = active & (consequents == consequent)
        if not np.any(selected_rules):
            continue
        rows = table[selected_rules]
        match = y == consequent
        support = np.mean(rows * match, axis=1)
        numerator = np.sum(np.ascontiguousarray(rows[:, match]), axis=1)
        divisor = denominator[selected_rules]
        confidence = np.divide(numerator, divisor, out=np.zeros(len(rows)),
                               where=divisor != 0)
        scores[selected_rules] = support * confidence
    return scores


def _winner_counts(winners: np.ndarray, selected: np.ndarray,
                   n_rules: int) -> np.ndarray:
    population = len(winners)
    offsets = np.arange(population)[:, None] * n_rules
    encoded = (offsets + winners)[selected]
    return np.bincount(encoded, minlength=population * n_rules).reshape(
        population, n_rules)


def _mcc_population(prediction: np.ndarray, domain: _LabelDomain) -> np.ndarray:
    """MCC for a population using a fixed superset of zero rows/columns."""
    population, _ = prediction.shape
    size = domain.size
    shifted = prediction + 1
    offsets = np.arange(population)[:, None] * size * size
    encoded = offsets + domain.shifted_y[None, :] * size + shifted
    matrix = np.bincount(encoded.reshape(-1),
                         minlength=population * size * size).reshape(
                             population, size, size)
    true_sum = matrix.sum(axis=2, dtype=np.float64)
    pred_sum = matrix.sum(axis=1, dtype=np.float64)
    correct = np.trace(matrix, axis1=1, axis2=2, dtype=np.float64)
    samples = pred_sum.sum(axis=1)
    covariance = correct * samples - np.sum(true_sum * pred_sum, axis=1)
    pred_variance = samples ** 2 - np.sum(pred_sum * pred_sum, axis=1)
    true_variance = samples ** 2 - np.sum(true_sum * true_sum, axis=1)
    denominator = pred_variance * true_variance
    result = np.zeros(population)
    valid = denominator != 0
    result[valid] = covariance[valid] / np.sqrt(denominator[valid])
    return result


def score_population(genes: np.ndarray, packed: tuple, y: np.ndarray,
                     n_rules: int, n_ants: int, n_features: int,
                     n_classes: int, term_counts: np.ndarray,
                     fourth_pointer: int, ds_mode: int, allow_unknown: bool,
                     tolerance: float, alpha: float, beta: float,
                     labels: _LabelDomain,
                     firing_cache: Optional[_FiringCache] = None,
                     gather_budget: int = _GATHER_BUDGET) -> Optional[np.ndarray]:
    """Return exact built-in scores, or None when this narrow route declines."""
    genes = np.asarray(genes)
    population = len(genes)
    n_samples = len(y)
    if (not supports_shape(population, n_samples, n_rules, n_features,
                           gather_budget)
            or ds_mode not in (0, 1) or labels is None or packed is None):
        return None
    decoded = _decode_population(
        genes, n_rules, n_ants, n_features, n_classes, term_counts,
        fourth_pointer)
    if decoded is None:
        return None
    antecedents, consequents, active = decoded
    firing = _firing_population(
        antecedents, active, packed, n_samples, firing_cache)
    if firing is None:
        return None

    scores = _dominance_population(firing, y, consequents, active, n_classes)
    association = firing if ds_mode == 1 else firing * scores[:, None, :]
    association = np.where(active[:, None, :], association, -np.inf)
    winners = np.argmax(association, axis=2)
    maxima = np.max(association, axis=2)
    has_rules = np.any(active, axis=1)
    eligible = ((maxima != 0.0) if allow_unknown
                else np.ones((population, n_samples), dtype=bool))
    eligible &= has_rules[:, None]
    predicted = np.take_along_axis(consequents, winners, axis=1)
    correct = (predicted == y[None, :]) & eligible
    wins = _winner_counts(winners, eligible, n_rules)
    correct_wins = _winner_counts(winners, correct, n_rules)
    accuracy = np.divide(correct_wins, wins, out=np.zeros_like(scores),
                         where=wins != 0)
    keep = active & ~(scores < tolerance) & (accuracy != 0.0)

    order = np.argsort(~keep, axis=1, kind='stable')
    firing = np.take_along_axis(firing, order[:, None, :], axis=2)
    antecedents = np.take_along_axis(antecedents, order[:, :, None], axis=1)
    consequents = np.take_along_axis(consequents, order, axis=1)
    survivor_count = np.sum(keep, axis=1)
    active = np.arange(n_rules)[None, :] < survivor_count[:, None]

    scores = _dominance_population(firing, y, consequents, active, n_classes)
    association = firing if ds_mode == 1 else firing * scores[:, None, :]
    association = np.where(active[:, None, :], association, -np.inf)
    winners = np.argmax(association, axis=2)
    prediction = np.take_along_axis(consequents, winners, axis=1)
    if allow_unknown:
        prediction[np.max(association, axis=2) == 0.0] = -1
    result = _mcc_population(prediction, labels)
    result[survivor_count == 0] = 0.0

    if alpha != 0.0 or beta != 0.0:
        class_ids = np.arange(n_classes)
        class_present = np.any(
            active[:, :, None] & (consequents[:, :, None] == class_ids), axis=1)
        complete = np.all(class_present, axis=1)
        widths = np.count_nonzero(antecedents != -1, axis=2)
        selected = active & (scores > tolerance)
        possible = np.count_nonzero(selected, axis=1) * n_features
        effective = np.sum(np.where(selected, np.where(
            widths == 0, n_features, widths), 0), axis=1)
        size = np.divide(effective, possible, out=np.zeros(population),
                         where=possible != 0)
        size = np.where(complete, 1 - size, 0.0)
        effective_rules = np.count_nonzero(selected & (widths != 0), axis=1)
        rulesize = np.divide(effective_rules, survivor_count,
                             out=np.zeros(population), where=survivor_count != 0)
        rulesize = np.where(complete, rulesize, 0.0)
        if alpha != 0.0:
            result += alpha * size
        if beta != 0.0:
            result += beta * rulesize
    return result
