"""
Exact batched PyTorch scoring of the built-in Type-1 classification objective.

EvoX keeps its population in a tensor, possibly on a GPU.  This module scores a
population on that device while reproducing the CPU objective bit for bit, so
a fit follows the same search whichever device scored it.

Elementwise IEEE operations give identical results on any conforming device;
reductions are what differ between libraries.  Products over the feature axis
are written as the left-to-right loop ``np.prod`` performs, and every float sum
reproduces NumPy's pairwise summation tree (:class:`_PairwisePlan`).  Label
counts and the MCC's integer-valued float sums are exact in any order.  The
remaining MCC arithmetic, including its square root, and the complexity
penalties run in NumPy on the host: PyTorch's CPU ``sqrt`` was measured not to
be correctly rounded.

``_population_fitness.score_population`` is the template and, with the scalar
array evaluator, the oracle.  :meth:`TorchObjective.build` declines problems it
cannot represent and :meth:`TorchObjective.score` flags candidates it cannot
score exactly, which the caller scores on the CPU instead.  Parity still rests
on the installed NumPy's summation tree and on the device's arithmetic, so
:class:`DeviceRoute` trusts a device only after it has reproduced a whole
generation of CPU scores.
"""
from typing import Optional

import numpy as np

from ._population_fitness import _RouteProbe


#: NumPy's pairwise summation block size (``PW_BLOCKSIZE``).
_PAIRWISE_BLOCK = 128

#: ``fuzzy_sets.trapezoidal_membership``'s default shoulder adjustment.
_TRAPEZOID_EPSILON = 10E-5

#: Scratch budget for one scoring chunk on the CPU.
_CPU_BUDGET = 256 * 1024 * 1024

#: Share of the currently free CUDA memory one scoring chunk may use.
_CUDA_FREE_FRACTION = 0.35

#: Label counts square to at most ``samples ** 2``, which must stay exact.
_MAX_SAMPLES = 2 ** 26


def _torch():
    import torch
    return torch


class _PairwisePlan:
    """
    Index layout of NumPy's pairwise float64 sum for one length.

    NumPy sums a contiguous run of fewer than eight values left to right from
    0.0.  A run of at most 128 values is summed into eight accumulators,
    combined as ``((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))``, and its
    remainder is then added left to right.  Longer runs split at half their
    length rounded down to a multiple of eight.  The tree depends only on the
    length, so it is laid out once and every leaf is evaluated together.
    Padding uses -0.0, the exact identity of IEEE addition, so a padded leaf
    adds up to the same value as the unpadded one.
    """

    def __init__(self, length: int, device) -> None:
        torch = _torch()
        leaves, internal = [], []

        def build(start: int, count: int, depth: int) -> tuple:
            if count <= _PAIRWISE_BLOCK:
                leaves.append((start, count))
                return False, len(leaves) - 1
            half = count // 2
            half -= half % 8
            left = build(start, half, depth + 1)
            right = build(start + half, count - half, depth + 1)
            internal.append((left, right, depth))
            return True, len(internal) - 1

        root = build(0, length, 0)
        n_leaves = len(leaves)

        def node(reference: tuple) -> int:
            is_internal, index = reference
            return n_leaves + index if is_internal else index

        blocks = np.full((n_leaves, _PAIRWISE_BLOCK // 8, 8), length, dtype=np.int64)
        tails = np.full((n_leaves, 7), length, dtype=np.int64)
        small = np.zeros(n_leaves, dtype=bool)
        block_steps = tail_steps = 0
        for leaf, (start, count) in enumerate(leaves):
            full = count // 8 if count >= 8 else 0
            small[leaf] = full == 0
            blocks[leaf, :full] = (start + np.arange(8 * full)).reshape(full, 8)
            rest = count - 8 * full
            tails[leaf, :rest] = start + 8 * full + np.arange(rest)
            block_steps = max(block_steps, full)
            tail_steps = max(tail_steps, rest)

        by_depth = {}
        for index, (left, right, depth) in enumerate(internal):
            by_depth.setdefault(depth, []).append((n_leaves + index, node(left), node(right)))
        self.length = length
        self.n_leaves = n_leaves
        self.n_nodes = n_leaves + len(internal)
        self.root = node(root)
        self.blocks = torch.as_tensor(blocks[:, :max(1, block_steps)], device=device)
        self.tails = torch.as_tensor(tails[:, :tail_steps], device=device)
        self.small = torch.as_tensor(small, device=device)
        # Children sit one level deeper than their parent, so evaluating the
        # deepest level first always finds both operands complete.
        self.levels = [
            tuple(torch.as_tensor(column, device=device) for column in zip(*by_depth[depth]))
            for depth in sorted(by_depth, reverse=True)]

    def __call__(self, values):
        torch = _torch()
        shape = values.shape[:-1]
        if self.length == 0:
            return values.new_zeros(shape)
        padded = torch.cat((values, values.new_full(shape + (1,), -0.0)), dim=-1)
        blocks = padded[..., self.blocks]
        accumulators = blocks[..., 0, :]
        for step in range(1, blocks.shape[-2]):
            accumulators = accumulators + blocks[..., step, :]
        a = accumulators
        total = ((a[..., 0] + a[..., 1]) + (a[..., 2] + a[..., 3])) + (
            (a[..., 4] + a[..., 5]) + (a[..., 6] + a[..., 7]))
        total = torch.where(self.small, values.new_zeros(()), total)
        if self.tails.shape[1]:
            tails = padded[..., self.tails]
            for step in range(tails.shape[-1]):
                total = total + tails[..., step]
        if not self.levels:
            return total[..., 0]
        nodes = values.new_empty(shape + (self.n_nodes,))
        nodes[..., :self.n_leaves] = total
        for ids, lefts, rights in self.levels:
            nodes[..., ids] = nodes[..., lefts] + nodes[..., rights]
        return nodes[..., self.root]


def pairwise_sum(values, plans: Optional[dict] = None):
    """
    Sum the last axis exactly as NumPy sums a contiguous float64 run.

    ``plans`` optionally memoizes the index layouts by length and device; the
    caller owns it and decides how long the layouts live.
    """
    key = (values.shape[-1], str(values.device))
    plan = None if plans is None else plans.get(key)
    if plan is None:
        plan = _PairwisePlan(values.shape[-1], values.device)
        if plans is not None:
            plans[key] = plan
    return plan(values)


def _membership_rows(truth, variables, n_samples: int) -> Optional[np.ndarray]:
    """Fixed memberships as ``(terms + 1, samples)`` with a final row of ones."""
    if truth is None or not isinstance(variables, (list, tuple)) or len(truth) != len(variables):
        return None
    rows = []
    for feature, variable in zip(truth, variables):
        if not isinstance(feature, (list, tuple, np.ndarray)):
            return None
        values = np.asarray(feature)
        if values.dtype.kind not in 'biuf' or values.shape != (len(variable), n_samples):
            return None
        rows.append(values)
    table = np.empty((sum(len(values) for values in rows) + 1, n_samples))
    position = 0
    for values in rows:
        table[position:position + len(values)] = values
        position += len(values)
    table[-1] = 1.
    return table if np.all(np.isfinite(table)) else None


class TorchObjective:
    """
    Device copy of one problem's immutable scoring context.

    Created by :meth:`build` once per fit and discarded with the fit's other
    caches.  The training data, labels and partitions must not change while it
    is in use, exactly like the CPU evaluators' packed memberships.
    """

    @classmethod
    def build(cls, problem, device) -> Optional['TorchObjective']:
        """
        Upload a problem's scoring context, or return None if unsupported.

        The caller has already established that the problem uses the built-in
        Type-1 objective with ``ds_mode`` 0 or 1 and numeric labels.
        """
        torch = _torch()
        device = torch.device(device)
        labels = problem._label_domain()
        X = np.asarray(problem.X)
        if (labels is None or X.ndim != 2 or 0 in X.shape or X.dtype.kind not in 'biuf'
                or X.shape[0] > _MAX_SAMPLES):
            return None
        n_samples, n_features = X.shape
        if problem.lvs is None:
            counts = np.asarray(problem.n_lv_possible, dtype=np.int64)
            categorical = problem.categorical_boolean_mask
            # Categorical variables keep their crisp partitions, and a
            # single-term partition decodes genes beyond its own; both, and
            # non-finite data, stay on the CPU.
            if ((categorical is not None and np.any(categorical)) or len(counts) != n_features
                    or np.any(counts < 2) or not np.all(np.isfinite(X))):
                return None
            table = None
            minimum, maximum, _ = problem._normalization_domain()
            columns = np.empty((n_features, n_samples))
            for feature in range(n_features):
                columns[feature] = np.clip(X[:, feature], minimum[feature], maximum[feature])
        else:
            table = _membership_rows(problem._precomputed_truth, problem.lvs, n_samples)
            if table is None:
                return None
            counts = np.asarray([len(variable) for variable in problem.lvs], dtype=np.int64)
            columns = None

        n_rules = problem.nRules
        # Firing, association and pairwise-sum scratch grow with the rules. An
        # optimized partition adds its membership table and the trapezoid
        # temporaries of one feature at a time.
        floats = 16 * n_rules
        if table is None:
            floats += int(counts.sum()) + 1 + 6 * int(counts.max())
        per_candidate = 8 * n_samples * floats + 2 * n_rules * n_rules * n_features
        if device.type == 'cuda':
            free, _ = torch.cuda.mem_get_info(device)
            budget = int(free * _CUDA_FREE_FRACTION)
        else:
            budget = _CPU_BUDGET
        chunk = budget // per_candidate
        if chunk < 1:
            return None
        return cls(problem, device, counts, table, columns, int(chunk))

    def __init__(self, problem, device, counts: np.ndarray, table: Optional[np.ndarray],
                 columns: Optional[np.ndarray], chunk: int) -> None:
        torch = _torch()
        self.device = device
        self.chunk = chunk
        self.n_rules = problem.nRules
        self.n_ants = problem.nAnts
        self.n_samples, self.n_features = np.asarray(problem.X).shape
        self.n_classes = problem.n_classes
        self.fourth_pointer = problem._consequent_pointer(problem.fuzzy_type)
        self.ds_mode = problem.ds_mode
        self.allow_unknown = problem.allow_unknown
        self.tolerance = problem.tolerance
        self.alpha = problem.alpha_
        self.beta = problem.beta_
        self.n_terms = int(counts.sum())
        self._counts = counts.tolist()
        self.term_counts = torch.as_tensor(counts, device=device)
        self.offsets = torch.as_tensor(np.concatenate(([0], np.cumsum(counts)[:-1])),
                                       dtype=torch.int64, device=device)
        self.table = None if table is None else torch.as_tensor(table, device=device)
        self.columns = None if columns is None else torch.as_tensor(columns, device=device)
        if table is None:
            minimum, _, span = problem._normalization_domain()
            self._minimum = [float(value) for value in minimum]
            self._span = [float(value) for value in span]
            self.partition_pointer = 2 * self.n_ants * self.n_rules
        y = np.asarray(problem.y)
        labels = problem._label_domain()
        self.y = torch.as_tensor(y, dtype=torch.int64, device=device)
        self.label_size = labels.size
        self.shifted_y = torch.as_tensor(labels.shifted_y, dtype=torch.int64, device=device)
        self.class_masks = [torch.as_tensor(y == klass, dtype=torch.float64, device=device)
                            for klass in range(self.n_classes)]
        self.class_samples = [torch.as_tensor(np.flatnonzero(y == klass), dtype=torch.int64,
                                              device=device)
                              for klass in range(self.n_classes)]
        self._plans = {}

    def score(self, genes: np.ndarray) -> tuple:
        """
        Return ``(scores, declined)`` for a population of chromosomes.

        ``scores`` holds the values ``FitRuleBase._array_score`` returns.
        Entries flagged in ``declined`` are placeholders the caller must score
        on the CPU.
        """
        torch = _torch()
        values = np.asarray(genes)
        population = len(values)
        if (values.ndim != 2 or values.dtype.kind not in 'biu' or population == 0
                or values.shape[1] != self.fourth_pointer + self.n_rules):
            return np.zeros(population), np.ones(population, dtype=bool)
        tensor = torch.as_tensor(np.ascontiguousarray(values, dtype=np.int64), device=self.device)
        chunks = [self._score_chunk(tensor[start:start + self.chunk])
                  for start in range(0, population, self.chunk)]
        return self._finish(*(torch.cat(pieces).cpu().numpy() for pieces in zip(*chunks)))

    def _finish(self, covariance, pred_variance, true_variance, survivors, declined,
                complete=None, width=None, possible=None, rules=None) -> tuple:
        """
        Complete the objective on the host exactly as ``score_population`` does.

        Every input holds exact integer values; the square root, the divisions
        and the penalty additions are left to NumPy.
        """
        population = len(covariance)
        result = np.zeros(population)
        denominator = pred_variance * true_variance
        valid = denominator != 0
        result[valid] = covariance[valid] / np.sqrt(denominator[valid])
        result[survivors == 0] = 0.0
        if complete is not None:
            size = np.divide(width, possible, out=np.zeros(population), where=possible != 0)
            size = np.where(complete & (possible != 0), 1 - size, 0.0)
            rulesize = np.divide(rules, survivors, out=np.zeros(population),
                                 where=survivors != 0)
            rulesize = np.where(complete, rulesize, 0.0)
            # Accumulate as the reference does: one addition per penalty.
            if self.alpha != 0.0:
                result += self.alpha * size
            if self.beta != 0.0:
                result += self.beta * rulesize
        return result, declined

    def _score_chunk(self, genes) -> tuple:
        torch = _torch()
        n_rules, n_ants = self.n_rules, self.n_ants
        n_features, n_classes, n_samples = self.n_features, self.n_classes, self.n_samples
        population = genes.shape[0]
        zero = torch.zeros((), dtype=torch.float64, device=self.device)
        features = genes[:, :n_rules * n_ants].reshape(population, n_rules, n_ants)
        terms = genes[:, n_rules * n_ants:2 * n_rules * n_ants].reshape(
            population, n_rules, n_ants)
        consequents = genes[:, self.fourth_pointer:self.fourth_pointer + n_rules]
        # The CPU decoders refuse these chromosomes; clamping only keeps their
        # placeholder rows computable.
        declined = (((consequents < -1) | (consequents >= n_classes)).any(dim=1)
                    | ((features < 0) | (features >= n_features)).flatten(1).any(dim=1)
                    | (terms < -1).flatten(1).any(dim=1))
        features = features.clamp(0, n_features - 1)
        consequents = consequents.clamp(-1, n_classes - 1)

        effective = torch.full((population, n_rules, n_features), -1, dtype=torch.int64,
                               device=self.device)
        for slot in range(n_ants):
            # Later slots overwrite earlier ones, as in the reference decoder.
            chosen = features[:, :, slot]
            chosen_terms = torch.minimum(terms[:, :, slot], self.term_counts[chosen] - 1)
            effective.scatter_(2, chosen.unsqueeze(2), chosen_terms.unsqueeze(2))
        active = (consequents != -1) & (effective != -1).any(dim=2)
        for source in range(1, n_rules):
            same = (effective[:, :source] == effective[:, source:source + 1]).all(dim=2)
            duplicate = (active[:, :source]
                         & (consequents[:, :source] == consequents[:, source:source + 1])
                         & same)
            active[:, source] &= ~duplicate.any(dim=1)

        rule_ids = torch.arange(n_rules, device=self.device)
        # A unique composite key gives the stable class-major order.
        order = torch.argsort(torch.where(active, consequents, n_classes) * n_rules + rule_ids,
                              dim=1)
        effective = effective.gather(1, order.unsqueeze(2).expand(-1, -1, n_features))
        consequents = consequents.gather(1, order)
        active = active.gather(1, order)

        if self.table is None:
            table, degenerate = self._partition_table(genes)
            declined |= degenerate
        index = torch.where(effective >= 0, effective + self.offsets, self.n_terms)
        firing = None
        for feature in range(n_features):
            column = index[:, :, feature]
            if self.table is None:
                values = table.gather(1, column.unsqueeze(2).expand(-1, -1, n_samples))
            else:
                values = self.table[column]
            # np.prod multiplies the feature axis from left to right.
            firing = values if firing is None else firing * values
        firing = torch.where(active.unsqueeze(2), firing, zero)
        # argmax treats NaN differently from NumPy, so NaN firing stays on the CPU.
        declined |= torch.isnan(firing).flatten(1).any(dim=1)
        # Release the per-candidate memberships before the scoring scratch.
        table = values = None

        scores = self._dominance(firing, consequents, active)
        association = firing if self.ds_mode == 1 else firing * scores.unsqueeze(2)
        association = torch.where(active.unsqueeze(2), association, -torch.inf)
        winners = association.argmax(dim=1)
        maxima = association.gather(1, winners.unsqueeze(1)).squeeze(1)
        if self.allow_unknown:
            eligible = maxima != 0.0
        else:
            eligible = torch.ones((population, n_samples), dtype=torch.bool, device=self.device)
        eligible &= active.any(dim=1).unsqueeze(1)
        correct = (consequents.gather(1, winners) == self.y) & eligible
        correct_wins = self._winner_counts(winners, correct)
        # prune_bad_rules drops 'score < tolerance or accuracy == 0.0', and a
        # rule's accuracy is zero exactly when it wins no sample correctly.
        keep = active & ~(scores < self.tolerance) & (correct_wins != 0)

        order = torch.argsort((~keep).long() * n_rules + rule_ids, dim=1)
        firing = firing.gather(1, order.unsqueeze(2).expand(-1, -1, n_samples))
        effective = effective.gather(1, order.unsqueeze(2).expand(-1, -1, n_features))
        consequents = consequents.gather(1, order)
        survivors = keep.sum(dim=1)
        active = rule_ids.unsqueeze(0) < survivors.unsqueeze(1)

        scores = self._dominance(firing, consequents, active)
        association = firing if self.ds_mode == 1 else firing * scores.unsqueeze(2)
        association = torch.where(active.unsqueeze(2), association, -torch.inf)
        winners = association.argmax(dim=1)
        prediction = consequents.gather(1, winners)
        if self.allow_unknown:
            maxima = association.gather(1, winners.unsqueeze(1)).squeeze(1)
            prediction = torch.where(maxima == 0.0, -1, prediction)
        parts = list(self._mcc_terms(prediction)) + [survivors, declined]
        if self.alpha != 0.0 or self.beta != 0.0:
            parts.extend(self._complexity_counts(effective, consequents, active, scores))
        return parts

    def _partition_table(self, genes) -> tuple:
        """
        Decode, normalize and evaluate every candidate's optimized partitions.

        Mirrors ``FitRuleBase._decode_membership_functions`` followed by the
        normalization in ``_decode_antecedents`` and the trapezoids of
        ``fuzzy_sets.trapezoidal_membership``.  A feature whose parameters all
        coincide normalizes by zero; those candidates are declined.
        """
        torch = _torch()
        population = genes.shape[0]
        table = torch.empty((population, self.n_terms + 1, self.n_samples),
                            dtype=torch.float64, device=self.device)
        table[:, self.n_terms] = 1.
        degenerate = torch.zeros(population, dtype=torch.bool, device=self.device)
        pointer, row = self.partition_pointer, 0
        for feature, count in enumerate(self._counts):
            parameters = self._trapezoids(genes, pointer, count)
            pointer += (count - 1) * 4 + 3
            low = parameters.flatten(1).amin(dim=1)
            spread = parameters.flatten(1).amax(dim=1) - low
            degenerate |= spread == 0
            spread = torch.where(spread == 0, 1, spread)
            scaled = ((parameters - low[:, None, None]).double()
                      / spread.double()[:, None, None]
                      * self._span[feature] + self._minimum[feature])
            table[:, row:row + count] = self._trapezoid_membership(scaled, self.columns[feature])
            row += count
        return table, degenerate

    @staticmethod
    def _trapezoids(genes, pointer: int, count: int):
        """Integer trapezoid vertices of one feature's partition, per candidate."""
        torch = _torch()
        first = genes[:, pointer:pointer + 4]
        start = first[:, 0]
        shoulder = start + first[:, 1]
        following = shoulder + first[:, 2]
        end = following + first[:, 3]
        sets = [torch.stack((start, start, shoulder, end), dim=1)]
        for term in range(1, count - 1):
            middle = genes[:, pointer + 4 * term:pointer + 4 * term + 4]
            rise = end + middle[:, 0]
            top = rise + middle[:, 1]
            next_start = top + middle[:, 2]
            next_end = next_start + middle[:, 3]
            sets.append(torch.stack((following, rise, top, next_end), dim=1))
            following, end = next_start, next_end
        last = genes[:, pointer + 4 * (count - 1):pointer + 4 * (count - 1) + 2]
        rise = end + last[:, 0]
        top = rise + last[:, 1]
        sets.append(torch.stack((following, rise, top, top), dim=1))
        return torch.stack(sets, dim=1)

    @staticmethod
    def _trapezoid_membership(parameters, x):
        """``trapezoidal_membership`` for ``(population, terms, 4)`` vertices."""
        torch = _torch()
        a, b, c, d = parameters.unbind(dim=2)
        left = torch.where(b == a, a - _TRAPEZOID_EPSILON, a)
        right = torch.where(c == d, d + _TRAPEZOID_EPSILON, d)
        rising = (x - left.unsqueeze(2)) / (b - left).unsqueeze(2)
        falling = (right.unsqueeze(2) - x) / (right - c).unsqueeze(2)
        values = torch.clamp(torch.minimum(rising, falling), 0.0, 1.0)
        singleton = (x == a.unsqueeze(2)).double()
        return torch.where((a == d).unsqueeze(2), singleton, values)

    def _dominance(self, firing, consequents, active):
        """Exact T1 dominance scores for ``(population, rules, samples)`` firing."""
        torch = _torch()
        denominator = pairwise_sum(firing, self._plans)
        scores = torch.zeros(consequents.shape, dtype=torch.float64, device=self.device)
        zero = scores.new_zeros(())
        for klass in range(self.n_classes):
            selected = active & (consequents == klass)
            rows = firing[selected]
            if rows.shape[0] == 0:
                continue
            support = pairwise_sum(rows * self.class_masks[klass], self._plans) / self.n_samples
            numerator = pairwise_sum(rows.index_select(1, self.class_samples[klass]),
                                     self._plans)
            divisor = denominator[selected]
            confidence = torch.where(divisor != 0, numerator / divisor, zero)
            scores[selected] = support * confidence
        return scores

    def _winner_counts(self, winners, selected):
        torch = _torch()
        population = winners.shape[0]
        offsets = torch.arange(population, device=self.device).unsqueeze(1) * self.n_rules
        encoded = (offsets + winners)[selected]
        return torch.bincount(encoded, minlength=population * self.n_rules).reshape(
            population, self.n_rules)

    def _mcc_terms(self, prediction) -> tuple:
        """Integer-valued MCC covariance and variances over the fixed label layout."""
        torch = _torch()
        population = prediction.shape[0]
        size = self.label_size
        offsets = torch.arange(population, device=self.device).unsqueeze(1) * size * size
        encoded = offsets + self.shifted_y.unsqueeze(0) * size + (prediction + 1)
        matrix = torch.bincount(encoded.reshape(-1), minlength=population * size * size)
        matrix = matrix.reshape(population, size, size).double()
        true_sum = matrix.sum(dim=2)
        pred_sum = matrix.sum(dim=1)
        correct = matrix.diagonal(dim1=1, dim2=2).sum(dim=1)
        samples = pred_sum.sum(dim=1)
        covariance = correct * samples - (true_sum * pred_sum).sum(dim=1)
        pred_variance = samples * samples - (pred_sum * pred_sum).sum(dim=1)
        true_variance = samples * samples - (true_sum * true_sum).sum(dim=1)
        return covariance, pred_variance, true_variance

    def _complexity_counts(self, effective, consequents, active, scores) -> list:
        """Integer inputs of ``_array_fitness._complexity`` for every candidate."""
        torch = _torch()
        class_ids = torch.arange(self.n_classes, device=self.device)
        complete = (active.unsqueeze(2) & (consequents.unsqueeze(2) == class_ids)).any(
            dim=1).all(dim=1)
        widths = (effective != -1).sum(dim=2)
        # evalRuleBase uses a strict comparison here.
        selected = active & (scores > self.tolerance)
        possible = selected.sum(dim=1) * self.n_features
        width = torch.where(selected, torch.where(widths == 0, self.n_features, widths), 0)
        rules = (selected & (widths != 0)).sum(dim=1)
        return [complete, width.sum(dim=1), possible, rules]


class _DeviceProbe(_RouteProbe):
    """
    Choose between CPU and device scoring by measurement.

    The verification generation already runs the device on a whole generation,
    so no further unrecorded warm-up generation is needed.
    """

    WARMUP = ()


class DeviceRoute:
    """
    Fit-local trust and speed decision for one :class:`TorchObjective`.

    The device objective is trusted only after it reproduces the CPU scores of a
    sample of the first generation exactly; a single difference rejects it for
    the rest of the fit.  Scoring a whole generation on the CPU just to verify
    the device was the dominant fixed cost of expensive fits, while the failures
    verification guards against -- another NumPy summation order, a device's
    arithmetic -- are systematic, so a few candidates reveal them.  Exactness
    itself rests on the parity tests.  Once verified, which route runs is a
    speed question, settled by a counterbalanced probe like the scalar/batched
    choice.
    """

    CPU = _RouteProbe.SCALAR
    DEVICE = _RouteProbe.BATCH

    #: Fresh candidates a generation needs before it can verify or time a route.
    MIN_CANDIDATES = 4

    #: Candidates of the verification generation that are also scored on the CPU.
    VERIFY_CANDIDATES = 4

    #: Per-candidate speed ratio that settles on the device during verification.
    #: The sample is timed on the scalar CPU route, which the batched route beat
    #: by up to 2.09x in the C01 calibration, so the 1.25x margin of matched
    #: whole generations is widened accordingly.
    SAMPLE_DECISIVE = 3.0

    def __init__(self, objective: TorchObjective) -> None:
        self.objective = objective
        self.verified = False
        self.rejected = False
        self.probe = _DeviceProbe(self.MIN_CANDIDATES)

    def verify(self, expected: np.ndarray, actual: np.ndarray,
               cpu_seconds: Optional[float] = None,
               device_seconds: Optional[float] = None) -> bool:
        """
        Trust the device if ``actual`` equals the CPU's ``expected`` exactly.

        ``cpu_seconds`` and ``device_seconds`` are per-candidate costs: the CPU
        scored the sample on its scalar route, the device the whole generation
        while paying its first-use cost.  A device faster than that by more
        than ``SAMPLE_DECISIVE`` is chosen at once, which spares large problems
        the slow CPU generations of further probing.
        """
        self.verified = bool(np.array_equal(expected, actual, equal_nan=True))
        self.rejected = not self.verified
        if (self.verified and cpu_seconds is not None and device_seconds is not None
                and cpu_seconds > device_seconds * self.SAMPLE_DECISIVE):
            self.probe.decision = self.DEVICE
        return self.verified
