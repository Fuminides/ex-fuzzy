# cython: language_level=3
"""Optional native FERL vote scoring.

Adapted from the additive-vote loop in ``fgrt_fast/_kernels.pyx``
(``score_node_cci``) in Fuminides/fuzzy_greedy_tree. Cython compiles this
module to C. Python retains membership normalization, coverage checks and
split tie-breaking so both backends train with the same rule semantics.
"""
import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport isnan


@cython.boundscheck(False)
@cython.wraparound(False)
def added_vote_argmax(const double[:, ::1] votes,
                      const double[::1] membership,
                      const double[::1] probabilities):
    """Argmax(votes + membership[:, None] * probabilities), without a matrix.

    Strict comparisons retain NumPy's first-index tie break. Multiplication
    and addition must stay separate (build with floating-point contraction off).
    """
    cdef Py_ssize_t n = votes.shape[0]
    cdef Py_ssize_t classes = votes.shape[1]
    if membership.shape[0] != n or probabilities.shape[0] != classes:
        raise ValueError("Vote, membership and probability shapes do not match.")
    if classes == 0:
        raise ValueError("At least one class is required.")
    cdef cnp.ndarray[cnp.intp_t, ndim=1] result = np.empty(n, dtype=np.intp)
    cdef Py_ssize_t i, k, best
    cdef double value, maximum, contribution
    with nogil:
        for i in range(n):
            best = 0
            contribution = membership[i] * probabilities[0]
            maximum = votes[i, 0] + contribution
            for k in range(1, classes):
                contribution = membership[i] * probabilities[k]
                value = votes[i, k] + contribution
                if value > maximum or (isnan(value) and not isnan(maximum)):
                    maximum = value
                    best = k
            result[i] = best
    return result
