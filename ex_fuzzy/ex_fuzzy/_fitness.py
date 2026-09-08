"""Private fitness evaluation for decoded T1/T2 classification rule bases.

Decoding remains in FitRuleBase: this module never interprets chromosomes or
caches chromosome-dependent memberships. The full evaluator is the oracle.
"""
import numpy as np
from sklearn.metrics import matthews_corrcoef

try:
    from . import rules, eval_rules
except ImportError:
    import rules
    import eval_rules


def _dominance(firing: np.ndarray, y: np.ndarray, consequents: np.ndarray) -> np.ndarray:
    """Match evalRuleBase's support/confidence reductions, including T2 pooling."""
    scores = np.empty(len(consequents))
    for index, consequent in enumerate(consequents):
        values = firing[:, index]
        match = y == consequent
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


def _mcc(y: np.ndarray, prediction: np.ndarray) -> float:
    """MCC for internal integer labels, retaining sklearn for other dtypes.

    The union includes unknown (-1) and non-contiguous labels. Zero rows and
    columns are allowed, and a zero denominator yields the reference's 0.0.
    """
    if y.dtype.kind not in 'biu':
        return matthews_corrcoef(y, prediction)
    labels, encoded = np.unique(np.concatenate((y, prediction)), return_inverse=True)
    n = y.size
    count = labels.size
    matrix = np.bincount(
        encoded[:n] * count + encoded[n:], minlength=count * count,
    ).reshape(count, count)
    true_sum = matrix.sum(axis=1, dtype=np.float64)
    pred_sum = matrix.sum(axis=0, dtype=np.float64)
    correct = np.trace(matrix, dtype=np.float64)
    samples = pred_sum.sum()
    covariance = correct * samples - np.dot(true_sum, pred_sum)
    pred_variance = samples ** 2 - np.dot(pred_sum, pred_sum)
    true_variance = samples ** 2 - np.dot(true_sum, true_sum)
    if pred_variance * true_variance == 0:
        return 0.0
    return covariance / np.sqrt(pred_variance * true_variance)


def score_rulebase(rulebase, X: np.ndarray, y: np.ndarray, tolerance: float,
                   alpha: float, beta: float, precomputed_truth=None) -> float:
    """Compute the standard objective with one firing-strength evaluation.

    Pruning uses winners *before* removal. Final predictions select winners
    among surviving rules, preserving class/rule order and first-index ties.
    """
    all_rules = rulebase.get_rules()
    if not all_rules:
        return 0.0
    if precomputed_truth is None:
        precomputed_truth = rules.compute_antecedents_memberships(rulebase.antecedents, X)
    firing = rulebase.compute_firing_strenghts(X, precomputed_truth=precomputed_truth)
    y = np.asarray(y)
    consequents = np.asarray(rulebase.get_consequents(), dtype=int)
    weights = rulebase.get_weights() if rulebase.ds_mode == 2 else np.ones(len(all_rules))
    scores = _dominance(firing, y, consequents)
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
        return 0.0
    retained = {id(rule) for rule in survivors}
    keep = np.asarray([id(rule) in retained for rule in all_rules])
    # Match the contiguous layout of a fresh reference firing calculation.
    firing = np.ascontiguousarray(firing[:, keep])
    consequents = consequents[keep]
    scores = _dominance(firing, y, consequents)
    for rule, score in zip(survivors, scores):
        rule.score = score
    association = _association(firing, scores, weights[keep], rulebase.ds_mode)
    prediction = consequents[np.argmax(association, axis=1)]
    if rulebase.allow_unknown:
        prediction[np.max(association, axis=1) == 0.0] = -1
    result = _mcc(y, prediction)
    if alpha != 0.0 or beta != 0.0:
        evaluator = eval_rules.evalRuleBase(rulebase, X, y, precomputed_truth=precomputed_truth)
        if alpha != 0.0:
            result += alpha * evaluator.size_antecedents_eval(tolerance)
        if beta != 0.0:
            result += beta * evaluator.effective_rulesize_eval(tolerance)
    return result
