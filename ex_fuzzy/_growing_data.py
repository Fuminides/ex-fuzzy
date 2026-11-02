"""
A training subset that grows during the genetic search.

The first generations hold nearly random rule bases, and ranking those needs
few samples. :class:`GrowingData` scores generation ``g`` on a stratified
random subset holding ``x + (1 - x) * g / R`` of the training samples, where
``x`` is the initial fraction and ``R`` is ``full_data_at`` of the
generations, and every generation from ``R`` on on all the samples. Each
subset contains the previous one.

Parents keep the scores they got on the smaller subsets: re-scoring them at
every step would cost more than the subset saves. So does a chromosome the
search proposes again, through the fitness cache, which in a converging
population serves most candidates. When the data becomes complete the cache
is emptied, the backend re-scores the population once and restarts its
best-so-far and its patience, so the result and early stopping compare
scores on all the samples, computed exactly as without the schedule.
"""
from typing import Optional

import numpy as np

#: Mixed into the seed so the subsets do not follow the optimizer's random stream.
_SUBSET_STREAM = 0x67726f77


def _check_fraction(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f'{name} must be a number in (0, 1], got {value!r}.')
    if not 0 < value <= 1:
        raise ValueError(f'{name} must be in (0, 1], got {value!r}.')
    return float(value)


class GrowingData:
    """
    Sets the training subset of each generation on a ``FitRuleBase``.

    Args:
        problem: the problem whose rows change (through ``_activate_rows`` and
            ``_restore_rows``).
        initial_fraction: share of the samples of the first generation, in (0, 1].
        full_data_at: share of ``n_gen`` after which every sample is used, in (0, 1].
            It is reached by generation ``n_gen - 1`` at the latest.
        n_gen: generations of the search.
        seed: seed of the subsets, or None.
    """

    def __init__(self, problem, initial_fraction: float, full_data_at: float, n_gen: int,
                 seed: Optional[int]) -> None:
        self.initial_fraction = _check_fraction(initial_fraction, 'initial_data_fraction')
        _check_fraction(full_data_at, 'full_data_at')
        #: The first generation scored on every sample; 0 turns the schedule off.
        self.ramp = min(n_gen - 1, round(full_data_at * n_gen))
        y = np.asarray(problem.y)
        self.problem = problem
        self.n_samples = len(y)
        rng = np.random.default_rng(
            None if seed is None else np.random.SeedSequence([int(seed), _SUBSET_STREAM]))
        # A random rank within its class, as a share of the class size: any
        # prefix of this order holds every class in proportion, within one sample.
        keys = np.empty(self.n_samples)
        classes = np.unique(y)
        for klass in classes:
            members = np.flatnonzero(y == klass)
            keys[members] = (rng.permutation(len(members)) + rng.random(len(members))) / len(members)
        self.order = np.argsort(keys, kind='stable')
        position = np.empty(self.n_samples, dtype=np.int64)
        position[self.order] = np.arange(self.n_samples)
        #: The shortest prefix with a sample of every class.
        self.minimum = max(int(position[y == klass].min()) for klass in classes) + 1
        self.size = self.n_samples  # samples the problem holds now

    def size_at(self, gen: int) -> int:
        """Number of samples generation ``gen`` is scored on."""
        if gen >= self.ramp:
            return self.n_samples
        fraction = self.initial_fraction + (1 - self.initial_fraction) * gen / self.ramp
        return min(self.n_samples, max(self.minimum, int(np.ceil(fraction * self.n_samples))))

    @property
    def complete(self) -> bool:
        """Whether the problem holds every training sample."""
        return self.size == self.n_samples

    def advance(self, gen: int) -> bool:
        """
        Give the problem the subset of generation ``gen``.

        Returns:
            True when the data has just become complete: the scores measured
            before are on fewer samples and must not be compared with new ones.
        """
        size = self.size_at(gen)
        if size == self.size:
            return False
        if size == self.n_samples:
            self.problem._restore_rows()
        else:
            self.problem._activate_rows(np.sort(self.order[:size]))
        became_complete = size == self.n_samples
        self.size = size
        return became_complete
