"""
Fuzzy association rule classification on fixed fuzzy partitions.

This module implements the rule-generation and rule-selection stages of the
FARC-HD family of fuzzy association rule classifiers (Alcalá-Fdez, Alcalá and
Herrera, IEEE Transactions on Fuzzy Systems 19(5), 2011):

1. Candidate rules are enumerated per class up to a maximum number of
   conditions and filtered by class support, confidence and penalized
   certainty factor.
2. A covering-based subgroup-discovery prescreen keeps a diverse pool of rules
   per class, ranking them by weighted relative accuracy and down-weighting
   samples that are already covered. Every class keeps some candidates.
3. Rules carry their penalized certainty factor as a weight. As in Ex-Fuzzy's
   regression, rules combine additively (every matching rule votes) or
   sufficiently (only each sample's strongest rule decides).
4. A genetic algorithm selects a compact subset of the pool. Because candidate
   firing strengths are fixed, whole populations are scored from one
   precomputed firing matrix without building rule-base objects.

Rule firing uses the product t-norm, matching Ex-Fuzzy's Type-1 rule bases.
"""
from __future__ import annotations

import itertools

import numpy as np

#: Float64 entries of firing kept per mining or scoring call.
MEMORY_BUDGET = 32_000_000
RULE_MODES = ("additive", "sufficient")


def membership_matrices(partitions, X: np.ndarray) -> list[np.ndarray]:
    """Membership of every sample to every term: one ``(n_terms, n)`` array per feature."""
    memberships = []
    for index, variable in enumerate(partitions):
        values = np.asarray(variable(X[:, index]), dtype=float)
        if values.ndim == 1:
            values = values[None, :]
        memberships.append(values)
    return memberships


def rule_firing(memberships: list[np.ndarray], features, terms, n_samples: int) -> np.ndarray:
    """Product-t-norm firing of each rule, shape ``(n_rules, n_samples)``."""
    firing = np.empty((len(features), n_samples))
    for row, (rule_features, rule_terms) in enumerate(zip(features, terms)):
        current = memberships[rule_features[0]][rule_terms[0]]
        for feature, term in zip(rule_features[1:], rule_terms[1:]):
            current = current * memberships[feature][term]
        firing[row] = current
    return firing


def _empty_pool() -> dict:
    return dict(features=[], terms=[], consequent=np.zeros(0, int), weight=np.zeros(0),
                confidence=np.zeros(0), support=np.zeros(0), quality=np.zeros(0), firing=None)


def _concat(blocks: dict) -> dict:
    pool = dict(features=sum(blocks['features'], []), terms=sum(blocks['terms'], []))
    for key in ('consequent', 'weight', 'confidence', 'support', 'quality'):
        pool[key] = np.concatenate(blocks[key]) if blocks[key] else np.zeros(0)
    pool['consequent'] = pool['consequent'].astype(int)
    pool['firing'] = np.vstack(blocks['firing']) if blocks['firing'] else None
    return pool


def _take(pool: dict, rows) -> dict:
    rows = np.asarray(rows, dtype=int)
    out = dict(features=[pool['features'][r] for r in rows], terms=[pool['terms'][r] for r in rows])
    for key in ('consequent', 'weight', 'confidence', 'support', 'quality'):
        out[key] = pool[key][rows]
    out['firing'] = pool['firing'][rows] if pool['firing'] is not None else None
    return out


def _merge(pools: list[dict]) -> dict:
    blocks = {key: [] for key in ('features', 'terms', 'consequent', 'weight', 'confidence',
                                  'support', 'quality', 'firing')}
    for pool in pools:
        if len(pool['consequent']) == 0:
            continue
        for key in blocks:
            blocks[key].append(pool[key])
    return _concat(blocks) if blocks['consequent'] else _empty_pool()


def _combinations(memberships, max_conditions: int):
    """Yield ``(feature_tuple, term_shape, firing)`` with firing of shape ``(n_terms_product, n)``."""
    n_features = len(memberships)
    n = memberships[0].shape[1]
    for depth in range(1, min(max_conditions, n_features) + 1):
        for feature_tuple in itertools.combinations(range(n_features), depth):
            firing = memberships[feature_tuple[0]]
            shape = [firing.shape[0]]
            for feature in feature_tuple[1:]:
                other = memberships[feature]
                firing = (firing[:, None, :] * other[None, :, :]).reshape(-1, n)
                shape.append(other.shape[0])
            yield feature_tuple, shape, firing


def _row_keys(terms: np.ndarray, base: int) -> np.ndarray:
    """Encode each row of small term indices as one integer, for set membership tests."""
    weights = base ** np.arange(terms.shape[1] - 1, -1, -1, dtype=np.int64)
    return terms.astype(np.int64) @ weights


def mine_candidates(memberships, y_index: np.ndarray, n_classes: int, max_conditions: int = 3,
                    min_support: float = 0.05, min_confidence: float = 0.5, min_weight: float = 0.0,
                    per_class_cap: int = None, memory_budget: int = MEMORY_BUDGET,
                    prune: bool = None) -> dict:
    """
    Enumerate fuzzy association rules and keep the promising ones per class.

    A rule's consequent is its most confident class. It is kept when its class
    support (covered class mass over class size) reaches ``min_support``, its
    confidence reaches ``min_confidence`` and beats the class prior, and its
    penalized certainty factor ``2 * confidence - 1`` exceeds ``min_weight``.
    At most ``per_class_cap`` rules per class are kept, ranked by weighted
    relative accuracy, so memory stays bounded on large problems.

    With ``prune=True`` the search is pruned Apriori-style. Adding a condition
    multiplies firing by a membership of at most one, so a rule's support for
    every class can only shrink. A rule is therefore only extended when its
    best class support reaches ``min_support`` and all of its shorter sub-rules
    were extended too. This keeps the same candidates as the exhaustive search
    (``prune=False``), in the same order and with the same firing values, while
    making rules with four or five conditions affordable. Statistics can differ
    by floating-point round-off, which may break near-ties differently, so the
    default ``prune=None`` enumerates exhaustively up to three conditions (the
    published benchmark configuration) and prunes only deeper searches.
    """
    if prune is None:
        prune = max_conditions > 3
    n = len(y_index)
    onehot = np.eye(n_classes)[y_index]
    class_mass = np.maximum(onehot.sum(0), 1e-12)
    prior = class_mass / n
    if per_class_cap is None:
        per_class_cap = int(max(50, min(2000, memory_budget // max(1, n * n_classes))))
    blocks = {c: {key: [] for key in ('features', 'terms', 'consequent', 'weight', 'confidence',
                                      'support', 'quality', 'firing')} for c in range(n_classes)}
    stored = np.zeros(n_classes, dtype=int)

    def score(feature_tuple, terms, firing):
        """Keep the promising rules of one feature combination; return their class mass."""
        total = firing.sum(1)
        covered = total > 1e-12
        by_class = firing @ onehot
        confidence = np.zeros_like(by_class)
        np.divide(by_class, total[:, None], out=confidence, where=covered[:, None])
        consequent = confidence.argmax(1)
        rows = np.arange(len(consequent))
        conf = confidence[rows, consequent]
        support = by_class[rows, consequent] / class_mass[consequent]
        weight = 2.0 * conf - 1.0
        keep = (covered & (support >= min_support) & (conf >= min_confidence)
                & (conf > prior[consequent]) & (weight > min_weight))
        if not keep.any():
            return by_class
        quality = (total / n) * (conf - prior[consequent])
        for c in np.unique(consequent[keep]):
            chosen = np.flatnonzero(keep & (consequent == c))
            block = blocks[c]
            block['features'].append([feature_tuple] * len(chosen))
            block['terms'].append([tuple(int(t) for t in terms[r]) for r in chosen])
            block['consequent'].append(np.full(len(chosen), c))
            block['weight'].append(weight[chosen])
            block['confidence'].append(conf[chosen])
            block['support'].append(support[chosen])
            block['quality'].append(quality[chosen])
            block['firing'].append(firing[chosen])
            stored[c] += len(chosen)
            if stored[c] > 2 * per_class_cap:            # bound memory as mining proceeds
                pool = _concat(block)
                best = np.argsort(-pool['quality'], kind='stable')[:per_class_cap]
                trimmed = _take(pool, best)
                blocks[c] = {key: [trimmed[key]] for key in trimmed}
                blocks[c]['features'] = [trimmed['features']]
                blocks[c]['terms'] = [trimmed['terms']]
                stored[c] = per_class_cap
        return by_class

    if not prune:
        for feature_tuple, shape, firing in _combinations(memberships, max_conditions):
            terms = np.stack(np.unravel_index(np.arange(firing.shape[0]), shape), axis=1)
            score(feature_tuple, terms, firing)
    else:
        n_features = len(memberships)
        depth_limit = min(max_conditions, n_features)
        base = max(m.shape[0] for m in memberships) + 1
        # A tiny tolerance keeps borderline rules extendable, so rounding can never
        # prune a branch the exhaustive search would keep.
        threshold = min_support * (1.0 - 1e-9)
        previous = {}
        for depth in range(1, depth_limit + 1):
            current = {}
            for feature_tuple in itertools.combinations(range(n_features), depth):
                if depth == 1:
                    feature = feature_tuple[0]
                    terms = np.arange(memberships[feature].shape[0])[:, None]
                    firing = memberships[feature]
                else:
                    parent = previous.get(feature_tuple[:-1])
                    if parent is None:
                        continue
                    parent_terms, parent_firing, _ = parent
                    last = feature_tuple[-1]
                    n_last = memberships[last].shape[0]
                    parent_rows = np.repeat(np.arange(len(parent_terms)), n_last)
                    terms = np.hstack([parent_terms[parent_rows],
                                       np.tile(np.arange(n_last), len(parent_terms))[:, None]])
                    alive = np.ones(len(terms), dtype=bool)
                    for drop in range(depth - 1):                 # every other shorter sub-rule
                        sibling = previous.get(feature_tuple[:drop] + feature_tuple[drop + 1:])
                        if sibling is None:
                            alive[:] = False
                            break
                        alive &= np.isin(_row_keys(np.delete(terms, drop, axis=1), base), sibling[2])
                    if not alive.any():
                        continue
                    terms, parent_rows = terms[alive], parent_rows[alive]
                    firing = parent_firing[parent_rows] * memberships[last][terms[:, -1]]
                by_class = score(feature_tuple, terms, firing)
                if depth < depth_limit:
                    extendable = (by_class / class_mass).max(1) >= threshold
                    if extendable.any():
                        survivors = terms[extendable]
                        current[feature_tuple] = (survivors, firing[extendable], _row_keys(survivors, base))
            previous = current

    pools = []
    for c in range(n_classes):
        if stored[c] == 0:
            pools.append(best_rules_for_class(memberships, y_index, n_classes, c, min(2, max_conditions)))
            continue
        pool = _concat(blocks[c])
        best = np.argsort(-pool['quality'], kind='stable')[:per_class_cap]
        pools.append(_take(pool, best))
    return _merge(pools)


def best_rules_for_class(memberships, y_index: np.ndarray, n_classes: int, target: int,
                         max_conditions: int = 2, count: int = 5) -> dict:
    """
    Relaxed fallback: the best short rules for a class that passed no threshold.

    Rules are forced to predict ``target`` and ranked by weighted relative
    accuracy. Their weight is floored at a small positive value so they can
    still win where no other rule fires.
    """
    n = len(y_index)
    is_target = (y_index == target).astype(float)
    target_mass = max(is_target.sum(), 1e-12)
    prior = target_mass / n
    best = []
    for feature_tuple, shape, firing in _combinations(memberships, max_conditions):
        total = firing.sum(1)
        covered = total > 1e-12
        in_target = firing @ is_target
        conf = np.divide(in_target, total, out=np.zeros_like(total), where=covered)
        quality = np.where(covered, (total / n) * (conf - prior), -np.inf)
        term_grid = np.stack(np.unravel_index(np.arange(len(total)), shape), axis=1)
        for r in np.argsort(-quality, kind='stable')[:count]:
            if not np.isfinite(quality[r]):
                break
            best.append((quality[r], feature_tuple, tuple(int(t) for t in term_grid[r]), conf[r],
                         in_target[r] / target_mass, firing[r]))
    best.sort(key=lambda item: -item[0])
    best = best[:count]
    if not best:
        return _empty_pool()
    return dict(features=[b[1] for b in best], terms=[b[2] for b in best],
                consequent=np.full(len(best), target, dtype=int),
                weight=np.array([max(2.0 * b[3] - 1.0, 1e-3) for b in best]),
                confidence=np.array([b[3] for b in best]), support=np.array([b[4] for b in best]),
                quality=np.array([b[0] for b in best]), firing=np.vstack([b[5] for b in best]))


def prescreen(pool: dict, y_index: np.ndarray, n_classes: int, per_class: int,
              cover_times: float = 2.0, max_scan: int = 500) -> dict:
    """
    Covering-based subgroup discovery over each class's candidates.

    Repeatedly picks the rule with the best weighted relative accuracy, then
    lowers the weight of the class samples it covers (weight ``1 / (1 + k)``
    after fuzzy coverage ``k``), until ``per_class`` rules are chosen, no rule
    adds positive accuracy, or every class sample is covered ``cover_times``.
    Only the ``max_scan`` best rules of a class by unweighted quality are
    scanned, which bounds the cost on large problems.
    """
    n = len(y_index)
    chosen_rows = []
    for c in range(n_classes):
        rows = np.flatnonzero(pool['consequent'] == c)
        if rows.size == 0:
            continue
        rows = rows[np.argsort(-pool['quality'][rows], kind='stable')[:max_scan]]
        F = pool['firing'][rows]
        positives = y_index == c
        prior = max(positives.mean(), 1e-12)
        weights = np.ones(n)
        coverage = np.zeros(n)
        available = np.ones(len(rows), dtype=bool)
        picked = []
        for _ in range(min(per_class, len(rows))):
            cover = F @ weights
            positive = F[:, positives] @ weights[positives]
            conf = np.divide(positive, cover, out=np.zeros_like(cover), where=cover > 0)
            wracc = (cover / weights.sum()) * (conf - prior)
            wracc[~available | (cover <= 0)] = -np.inf
            best = int(np.argmax(wracc))
            if not np.isfinite(wracc[best]) or wracc[best] <= 0:
                break
            picked.append(best)
            available[best] = False
            coverage[positives] += F[best, positives]
            weights[positives] = 1.0 / (1.0 + coverage[positives])
            if np.all(coverage[positives] >= cover_times):
                break
        if not picked:
            picked = [0]                                     # best unweighted rule of the class
        chosen_rows.extend(rows[picked].tolist())
    return _take(pool, sorted(chosen_rows, key=lambda r: (pool['consequent'][r], r)))


def class_scores(firing: np.ndarray, weights: np.ndarray, consequents: np.ndarray, n_classes: int,
                 rule_mode: str) -> np.ndarray:
    """
    Class scores ``(n_samples, n_classes)`` of one rule set.

    ``firing`` has shape ``(n_rules, n_samples)``. ``additive`` sums each
    class's weighted firing; ``sufficient`` keeps each class's strongest
    weighted rule, so each sample's single strongest rule decides.
    """
    association = firing * weights[:, None]
    scores = np.zeros((firing.shape[1], n_classes))
    for c in range(n_classes):
        rows = consequents == c
        if rows.any():
            scores[:, c] = association[rows].max(0) if rule_mode == "sufficient" else association[rows].sum(0)
    return scores


def predict_from_scores(scores: np.ndarray, default_class: int) -> np.ndarray:
    """Arg-max class, or ``default_class`` where no rule fires."""
    prediction = scores.argmax(1)
    prediction[scores.max(1) <= 0.0] = default_class
    return prediction


def _population_predictions(association: np.ndarray, masks: np.ndarray, run_starts: np.ndarray,
                            run_classes: np.ndarray, n_classes: int, default_class: int,
                            rule_mode: str) -> np.ndarray:
    """Predictions ``(n_individuals, n_samples)`` for rule masks over a class-sorted pool."""
    selected = masks[:, None, :] * association[None, :, :]    # (p, n, N)
    if rule_mode == "sufficient":
        grouped = np.maximum.reduceat(selected, run_starts, axis=2)
    else:
        grouped = np.add.reduceat(selected, run_starts, axis=2)
    scores = np.zeros(selected.shape[:2] + (n_classes,))
    scores[:, :, run_classes] = grouped
    prediction = scores.argmax(2)
    prediction[scores.max(2) <= 0.0] = default_class
    return prediction


def select_rules(pool: dict, y_index: np.ndarray, n_classes: int, default_class: int,
                 rule_mode: str = "additive", max_rules: int = None, rule_penalty: float = 1e-3,
                 n_gen: int = 100, pop_size: int = 60, patience: int = 20, rng=None,
                 memory_budget: int = MEMORY_BUDGET) -> np.ndarray:
    """
    Genetic selection of a rule subset maximizing training accuracy.

    Fitness is accuracy minus ``rule_penalty`` times the number of rules per
    class. Masks larger than ``max_rules`` are repaired by keeping their
    highest-quality rules. The algorithm keeps the best ``pop_size`` of parents
    and children each generation, uses binary tournaments, uniform crossover
    and bit-flip mutation, and stops after ``patience`` generations without
    improvement. The pool must be sorted by consequent.
    """
    rng = np.random.default_rng(rng)
    n_rules = len(pool['consequent'])
    if n_rules == 0:
        return np.zeros(0, dtype=bool)
    firing = pool['firing']                                  # (N, n)
    association = (firing * pool['weight'][:, None]).T       # (n, N)
    consequents = pool['consequent']
    if np.any(np.diff(consequents) < 0):
        raise ValueError("The rule pool must be sorted by consequent.")
    run_starts = np.flatnonzero(np.r_[True, np.diff(consequents) != 0])
    run_classes = consequents[run_starts]
    n = len(y_index)
    chunk = int(max(1, memory_budget // max(1, n * n_rules)))
    priority = np.argsort(-pool['quality'], kind='stable')

    def repair(masks):
        if max_rules is None:
            return masks
        for row in np.flatnonzero(masks.sum(1) > max_rules):
            keep = priority[masks[row, priority]][:max_rules]
            masks[row] = False
            masks[row, keep] = True
        return masks

    def evaluate(masks):
        accuracy = np.empty(len(masks))
        for start in range(0, len(masks), chunk):
            part = masks[start:start + chunk].astype(float)
            prediction = _population_predictions(association, part, run_starts, run_classes,
                                                 n_classes, default_class, rule_mode)
            accuracy[start:start + chunk] = (prediction == y_index[None, :]).mean(1)
        return accuracy - rule_penalty * masks.sum(1) / n_classes

    target = n_rules if max_rules is None else min(n_rules, max_rules)
    seeds = [np.ones(n_rules, dtype=bool)]
    per_class_best = np.zeros(n_rules, dtype=bool)
    for start, end in zip(run_starts, np.r_[run_starts[1:], n_rules]):
        per_class_best[start + int(np.argmax(pool['quality'][start:end]))] = True
    seeds.append(per_class_best)
    population = [*seeds]
    densities = np.clip(np.array([0.5, 2.0 * target / n_rules, target / n_rules]), 1.0 / n_rules, 1.0)
    while len(population) < pop_size:
        density = densities[len(population) % len(densities)]
        population.append(rng.random(n_rules) < density)
    population = repair(np.array(population[:pop_size]))
    fitness = evaluate(population)
    best, stale = fitness.max(), 0
    mutation = 1.0 / n_rules
    for _ in range(n_gen):
        first = rng.integers(pop_size, size=(pop_size, 2))
        second = rng.integers(pop_size, size=(pop_size, 2))
        parents_a = population[np.where(fitness[first[:, 0]] >= fitness[first[:, 1]], first[:, 0], first[:, 1])]
        parents_b = population[np.where(fitness[second[:, 0]] >= fitness[second[:, 1]], second[:, 0], second[:, 1])]
        children = np.where(rng.random((pop_size, n_rules)) < 0.5, parents_a, parents_b)
        children ^= rng.random((pop_size, n_rules)) < mutation
        children = repair(children)
        child_fitness = evaluate(children)
        merged = np.vstack([population, children])
        merged_fitness = np.concatenate([fitness, child_fitness])
        order = np.argsort(-merged_fitness, kind='stable')[:pop_size]
        population, fitness = merged[order], merged_fitness[order]
        if fitness[0] > best + 1e-12:
            best, stale = fitness[0], 0
        else:
            stale += 1
            if stale >= patience:
                break
    return population[int(np.argmax(fitness))].copy()
