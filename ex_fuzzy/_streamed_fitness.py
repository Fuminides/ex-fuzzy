"""
The built-in objective on all the training samples, in batches of samples.

:func:`_array_fitness.score_candidate` holds a candidate's firing strengths for
every sample at once, so its memory grows with samples x rules.
:func:`score_candidate` scores the same candidate on the same samples, to the
same bit, while holding the firing strengths of one batch of samples at a time.
It reads the samples in up to three passes:

1. the per-rule sums behind the dominance scores (support and confidence);
2. the winning rule of every sample, counted per rule for pruning, and the
   predictions in case pruning removes nothing;
3. the winners among the surviving rules, counted per class for the macro F1.
   Skipped when no rule was pruned: the predictions of pass 2 are then final.

The counts of passes 2 and 3 add up exactly across batches. The sums of pass 1
would not: NumPy sums a row in an order fixed by its length, as a pairwise
tree whose leaves are runs of at most 128 values, in some layouts first cut
into buffers of ``np.getbufsize()`` values whose sums are then added left to
right. Subtotals per batch would round differently. :class:`_StreamingSum`
instead holds the values of a leaf that a batch boundary cuts until the next
batch completes it, sums every leaf with NumPy, and combines the leaves in
NumPy's order when the pass ends. Which layouts NumPy buffers changed between
releases (2.3 stopped buffering contiguous rows), so :meth:`Summation.measure`
checks the order on the running NumPy, and streaming is refused when a layout
matches neither known order.

Besides one batch of firing strengths, a candidate holds the leaf sums: a few
floats per rule for every hundred or so samples.
"""
from typing import Callable, Optional

import numpy as np

from ._fitness import (_as_score, _association, _class_counts, _macro_f1, _majority_class,
                       _penalized_objective)

#: NumPy's pairwise sum adds runs of at most this many values without splitting them.
_PAIRWISE_BLOCK = 128


def _split(count: int) -> int:
    """Length of the first half NumPy's pairwise sum splits ``count`` values into."""
    half = count // 2
    return half - half % 8


class _SumLayout:
    """
    The order in which NumPy sums ``length`` contiguous float64 values.

    The values are cut into buffers of ``buffer`` values (a single buffer when
    ``buffer`` is None), each buffer is summed as a pairwise tree, and the
    buffer sums are added left to right from 0. A tree's shape depends only on
    its length, so the subtrees of one depth that have the same length are
    built and combined together, a few array operations per depth.

    Attributes:
        leaf_starts, leaf_counts, leaf_stops: the leaves in stream order.
    """

    def __init__(self, length: int, buffer: Optional[int]) -> None:
        self.length = length
        self._leaves_in = {}
        size = length if buffer is None else min(buffer, length)
        full, rest = divmod(length, size) if size else (0, 0)
        level = {}
        if full:
            level[size] = (np.arange(full) * self._count(size), np.arange(full) * size)
        if rest:
            level[rest] = (np.array([full * self._count(size)]), np.array([full * size]))
        n_leaves = sum(len(offsets) * self._count(count) for count, (offsets, _) in level.items())
        self.leaf_starts = np.empty(n_leaves, dtype=np.int64)
        self.leaf_counts = np.empty(n_leaves, dtype=np.int64)
        # Each level maps a subtree length to the index of each such subtree's
        # first leaf (sorted) and its first stream position, depth by depth.
        self._levels = []
        while level:
            self._levels.append(level)
            children = {}
            for count, (offsets, starts) in level.items():
                if count <= _PAIRWISE_BLOCK:
                    self.leaf_starts[offsets] = starts
                    self.leaf_counts[offsets] = count
                    continue
                half = _split(count)
                for child, child_offsets, child_starts in (
                        (half, offsets, starts),
                        (count - half, offsets + self._count(half), starts + half)):
                    known = children.get(child)
                    if known is not None:
                        child_offsets = np.concatenate((known[0], child_offsets))
                        child_starts = np.concatenate((known[1], child_starts))
                    children[child] = (child_offsets, child_starts)
            for child, (offsets, starts) in children.items():
                order = np.argsort(offsets)
                children[child] = (offsets[order], starts[order])
            level = children
        self.leaf_stops = self.leaf_starts + self.leaf_counts

    def _count(self, count: int) -> int:
        """Number of leaves in the pairwise tree of ``count`` values."""
        known = self._leaves_in.get(count)
        if known is None:
            half = _split(count)
            known = 1 if count <= _PAIRWISE_BLOCK else self._count(half) + self._count(count - half)
            self._leaves_in[count] = known
        return known

    def combine(self, leaves: np.ndarray) -> np.ndarray:
        """Total of each row of ``leaves``, the leaf sums in stream order."""
        total = np.zeros(leaves.shape[0])
        if not self._levels:
            return total
        below = {}
        for level in reversed(self._levels):
            current = {}
            for count, (offsets, _) in level.items():
                if count <= _PAIRWISE_BLOCK:
                    values = leaves[:, offsets]
                else:
                    half = _split(count)
                    left_offsets, left = below[half]
                    right_offsets, right = below[count - half]
                    values = (left[:, np.searchsorted(left_offsets, offsets)]
                              + right[:, np.searchsorted(right_offsets, offsets + self._count(half))])
                current[count] = (offsets, values)
            below = current
        offsets = np.concatenate([offsets for offsets, _ in below.values()])
        buffers = np.concatenate([values for _, values in below.values()], axis=1)
        for column in np.argsort(offsets):
            total = total + buffers[:, column]
        return total


class _StreamingSum:
    """
    Sums of equally long rows whose values arrive a batch of columns at a time.

    Every row is summed as NumPy sums it whole (``layout``). A leaf cut by a
    batch boundary is held back until the next batch completes it, so every
    leaf is summed by NumPy from the same values, in the same layout, as it is
    within the whole row.
    """

    def __init__(self, layout: _SumLayout, width: int) -> None:
        self.layout = layout
        self.leaves = np.empty((width, len(layout.leaf_counts)))
        self.pending = np.empty((width, 0))
        self.position = 0  # stream position of the first pending column
        self.summed = 0  # number of leaves already summed
        self._total = None

    def feed(self, values: np.ndarray) -> None:
        """Append the next columns of every row."""
        data = np.concatenate((self.pending, values), axis=1) if self.pending.shape[1] else values
        layout = self.layout
        ready = int(np.searchsorted(layout.leaf_stops, self.position + data.shape[1], side='right'))
        if ready > self.summed:
            first = self.summed
            starts = layout.leaf_starts[first:ready] - self.position
            counts = layout.leaf_counts[first:ready]
            for count in np.unique(counts):
                which = np.flatnonzero(counts == count)
                # Contiguous runs on the last axis, each summed like that leaf of the whole row.
                runs = np.ascontiguousarray(data[:, starts[which, None] + np.arange(count)])
                self.leaves[:, first + which] = np.sum(runs, axis=2)
            used = int(layout.leaf_stops[ready - 1]) - self.position
            self.position += used
            self.summed = ready
            data = data[:, used:]
        # What is left lies within one unfinished leaf: fewer than 128 columns.
        self.pending = np.array(data)

    def total(self) -> np.ndarray:
        """The sum of every row; all its values must have been fed."""
        if self._total is None:
            if self.summed != len(self.layout.leaf_counts):
                raise RuntimeError('The streamed sum is missing values.')
            self._total = self.layout.combine(self.leaves)
        return self._total


class Summation:
    """
    How the running NumPy sums each layout the objective reduces.

    ``buffers`` maps each layout to the buffer that splits it (None: one
    pairwise tree), and :meth:`layout` builds and keeps the layouts of one
    problem. The layouts are those of :func:`_fitness._dominance`:

    - ``'rows'``: a contiguous row, as in every Type-1 sum and the Type-2
      supports and numerators;
    - ``'interval'``: one rule's Type-2 denominator among several rules, the
      sum of a strided ``(samples, 2)`` slice of the firing array;
    - ``'single_interval'``: the same for a firing array with a single rule,
      where that slice is contiguous.
    """

    LAYOUTS = ('rows', 'interval', 'single_interval')

    def __init__(self, buffers: dict) -> None:
        self.buffers = buffers
        self._layouts = {}

    @classmethod
    def measure(cls) -> Optional['Summation']:
        """
        Match each layout's NumPy sum against the two known orders.

        Returns None when a layout matches neither or both of them: batched
        sums could then differ from whole ones. The probe streams its values
        in batches, so it also checks :class:`_StreamingSum` itself.
        """
        bufsize, width = np.getbufsize(), 8
        length = 3 * bufsize + 37  # several buffers and a partial one
        # Uniform values: the two orders round differently on most rows, while
        # values of very different magnitudes would let the largest decide.
        values = np.random.default_rng(0).random((length, width, 2))
        rows = np.ascontiguousarray(values[:, :, 0].T)
        pairs = np.ascontiguousarray(values.transpose(1, 0, 2).reshape(width, -1))
        cases = {
            'rows': (np.sum(rows, axis=1), rows),
            'interval': (np.array([np.sum(values[:, index]) for index in range(width)]), pairs),
            'single_interval': (np.array([np.sum(np.ascontiguousarray(values[:, index]))
                                          for index in range(width)]), pairs),
        }
        buffers = {}
        for name, (expected, stream) in cases.items():
            matches = []
            for buffer in (None, bufsize):
                streamed = _StreamingSum(_SumLayout(stream.shape[1], buffer), len(stream))
                for start in range(0, stream.shape[1], 1000):
                    streamed.feed(stream[:, start:start + 1000])
                if np.array_equal(streamed.total(), expected):
                    matches.append(buffer)
            if len(matches) != 1:
                return None
            buffers[name] = matches[0]
        return cls(buffers)

    def layout(self, length: int, name: str) -> _SumLayout:
        """The summation order of ``length`` values in layout ``name``."""
        key = (length, self.buffers[name])
        layout = self._layouts.get(key)
        if layout is None:
            layout = self._layouts[key] = _SumLayout(*key)
        return layout


class _DominanceSums:
    """
    The sums behind :func:`_fitness._dominance`, fed one batch of samples at a time.

    Per rule: its firing strength summed over all samples (the confidence's
    denominator), the same restricted to its class but still taken over all
    samples (the support), and over its class's samples alone (the
    confidence's numerator). Each sum has the layout of the reduction it
    replaces.
    """

    def __init__(self, consequents: np.ndarray, y: np.ndarray, interval: bool,
                 summation: Summation) -> None:
        n_rules, n_samples = len(consequents), len(y)
        self.consequents = consequents
        self.interval = interval
        self.n_samples = n_samples

        def stream(length, width, name='rows'):
            return _StreamingSum(summation.layout(length, name), width)

        if interval:
            self.supports = (stream(n_samples, n_rules), stream(n_samples, n_rules))
            # Pruning may leave a single rule, whose denominator has the
            # contiguous layout, so both are kept unless they coincide.
            self.denominators = {}
            for name in ('interval', 'single_interval') if n_rules > 1 else ('single_interval',):
                twin = next((kept for other, kept in self.denominators.items()
                             if summation.buffers[other] == summation.buffers[name]), None)
                self.denominators[name] = twin or stream(2 * n_samples, n_rules, name)
        else:
            self.supports = (stream(n_samples, n_rules),)
            self.denominators = {'rows': stream(n_samples, n_rules)}
        values_per_sample = 2 if interval else 1
        self.classes = []
        for klass in np.unique(consequents):
            members = np.flatnonzero(consequents == klass)
            count = int(np.count_nonzero(y == klass))
            self.classes.append((klass, members, stream(values_per_sample * count, len(members))))

    def feed(self, firing: np.ndarray, y: np.ndarray) -> None:
        """Add the firing strengths of a batch of samples and their labels."""
        n_rules = len(self.consequents)
        match = y == self.consequents[:, None]  # each rule's class, per sample
        if self.interval:
            self.supports[0].feed(firing[:, :, 0].T * match)
            self.supports[1].feed(firing[:, :, 1].T * match)
            pairs = firing.transpose(1, 0, 2).reshape(n_rules, -1)
            for denominator in {id(kept): kept for kept in self.denominators.values()}.values():
                denominator.feed(pairs)
        else:
            self.supports[0].feed(firing.T * match)
            self.denominators['rows'].feed(firing.T)
        for klass, members, numerator in self.classes:
            selected = firing[y == klass][:, members]
            numerator.feed(np.moveaxis(selected, 1, 0).reshape(len(members), -1))

    def scores(self, keep: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Dominance scores of the rules ``keep`` selects (all by default).

        Equal to ``_dominance`` on the firing strengths of those rules alone:
        every sum is per rule, and only the Type-2 denominator's layout depends
        on how many rules remain.
        """
        n_rules = len(self.consequents)
        keep = np.ones(n_rules, dtype=bool) if keep is None else keep
        numerator = np.empty(n_rules)
        for _, members, stream in self.classes:
            numerator[members] = stream.total()
        if self.interval:
            support = (self.supports[0].total() / self.n_samples
                       + self.supports[1].total() / self.n_samples) / 2
            denominator = self.denominators[
                'single_interval' if np.count_nonzero(keep) == 1 else 'interval'].total()
        else:
            support = self.supports[0].total() / self.n_samples
            denominator = self.denominators['rows'].total()
        confidence = np.divide(numerator, denominator, out=np.zeros(n_rules),
                               where=denominator != 0)
        return (support * confidence)[keep]


def _add_counts(total: Optional[tuple], part: tuple) -> tuple:
    """Add per-class count arrays element by element."""
    return part if total is None else tuple(left + right for left, right in zip(total, part))


def score_candidate(decoded, firing_rows: Callable, y: np.ndarray, n_classes: int, ds_mode: int,
                    allow_unknown: bool, tolerance: float, alpha: float, beta: float,
                    interval: bool, labels, batch_size: int, summation: Summation, *,
                    max_conditions: int, pareto: bool = False):
    """
    The objective of :func:`_array_fitness.score_candidate`, in batches of samples.

    Args:
        decoded: the candidate's ``_DecodedCandidate``.
        firing_rows: ``firing_rows(antecedents, start, stop)`` returns the firing
            strengths of the decoded rules ``antecedents`` on samples
            ``start:stop``, or None if it cannot.
        batch_size: samples per batch; it changes memory and time, never the result.
        summation: NumPy's summation order, from :meth:`Summation.measure`.

    Returns:
        the objective, the ``(accuracy, compactness)`` pair with ``pareto``, or
        None when ``firing_rows`` returns None.
    """
    n_rules = len(decoded.consequents)
    if n_rules == 0:
        return _as_score(_penalized_objective(0.0, 0, 0, n_classes, max_conditions, alpha, beta,
                                              pareto))
    y = np.asarray(y)
    batches = [(start, min(start + batch_size, len(y))) for start in range(0, len(y), batch_size)]
    antecedents = decoded.antecedents
    consequents = decoded.consequents
    weights = decoded.weights if ds_mode == 2 else np.ones(n_rules)
    if allow_unknown:
        uncovered = -1
    else:
        uncovered = _majority_class(y, n_classes) if labels is None else labels.majority

    sums = _DominanceSums(consequents, y, interval, summation)
    for start, stop in batches:
        firing = firing_rows(antecedents, start, stop)
        if firing is None:
            return None
        sums.feed(firing, y[start:stop])
    scores = sums.scores()

    wins = np.zeros(n_rules, dtype=np.int64)
    correct_wins = np.zeros(n_rules, dtype=np.int64)
    counts = None
    for start, stop in batches:
        firing = firing_rows(antecedents, start, stop)
        if firing is None:
            return None
        association = _association(firing, scores, weights, ds_mode)
        winners = np.argmax(association, axis=1)
        eligible = np.max(association, axis=1) != 0.0
        correct = (consequents[winners] == y[start:stop]) & eligible
        correct_wins += np.bincount(winners[correct], minlength=n_rules)
        wins += np.bincount(winners[eligible], minlength=n_rules)
        prediction = consequents[winners]
        prediction[~eligible] = uncovered
        counts = _add_counts(counts, _class_counts(y[start:stop], prediction, n_classes))
    accuracy = np.divide(correct_wins, wins, out=np.zeros(n_rules), where=wins != 0)
    keep = ~(scores < tolerance) & (accuracy != 0.0)
    if not keep.any():
        return _as_score(_penalized_objective(0.0, 0, 0, n_classes, max_conditions, alpha, beta,
                                              pareto))

    if not keep.all():
        antecedents, consequents, weights = antecedents[keep], consequents[keep], weights[keep]
        scores = sums.scores(keep)
        counts = None
        for start, stop in batches:
            firing = firing_rows(antecedents, start, stop)
            if firing is None:
                return None
            association = _association(firing, scores, weights, ds_mode)
            prediction = consequents[np.argmax(association, axis=1)]
            prediction[np.max(association, axis=1) == 0.0] = uncovered
            counts = _add_counts(counts, _class_counts(y[start:stop], prediction, n_classes))
    covered = np.count_nonzero(np.bincount(consequents, minlength=n_classes))
    conditions = np.count_nonzero(antecedents != -1)
    return _as_score(_penalized_objective(_macro_f1(*counts), covered, conditions, n_classes,
                                          max_conditions, alpha, beta, pareto))
