"""
Supervised fuzzification via MDLP (Fayyad-Irani) entropy discretization.

For each feature, recursive minimal-description-length discretization places cut
points exactly where the class distribution changes, using the MDL stopping
criterion to decide how many cuts a feature deserves (0 for uninformative
features). The resulting cut points are turned into overlapping *trapezoidal*
fuzzy sets — same shape ex_fuzzy's quantile partitioner uses, so there is no
"Gaussian shape tax", but with supervised, class-aware boundaries.

Drop-in replacement for ``ex_fuzzy.utils.construct_partitions``:

    parts = learn_partitions_mdlp(X_train, y_train)
    clf = FERL(parts, ...)

Reference: Fayyad & Irani (1993), "Multi-interval discretization of
continuous-valued attributes for classification learning".
"""
from __future__ import annotations

import numpy as np

try:
    from . import fuzzy_sets as fs
except ImportError:
    import fuzzy_sets as fs

_TERM_NAMES = ["low", "lowmed", "med", "medhigh", "high"]


def _term_name(k: int, K: int) -> str:
    if K == 1:
        return "all"
    if K <= len(_TERM_NAMES):
        idx = int(round(k * (len(_TERM_NAMES) - 1) / max(K - 1, 1)))
        return f"{_TERM_NAMES[idx]}_{k}"
    return f"term_{k}"


def _entropy(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def _entropy_from_counts(counts: np.ndarray, total: int) -> float:
    if total == 0:
        return 0.0
    nz = counts[counts > 0]
    p = nz / total
    return float(-np.sum(p * np.log2(p)))


def _entropy_rows(counts: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Row-wise entropy of a (m, C) count matrix with per-row totals n (m,)."""
    p = counts / n[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(counts > 0, p * np.log2(p), 0.0)
    return -terms.sum(axis=1)


def mdlp_cuts(x: np.ndarray, y: np.ndarray) -> list[float]:
    """Return sorted MDLP cut points for one feature.

    Vectorized implementation: sort once, and at each recursion level evaluate
    *all* candidate splits at once via cumulative class counts (no Python loop).
    Typical cost O(N log N * C); the cut selection is identical to the naive
    O(N^2) version (validated to match exactly).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    classes, ys = np.unique(y[order], return_inverse=True)  # labels -> 0..C-1, in x order
    C = len(classes)
    onehot = np.eye(C, dtype=float)[ys]
    cuts: list[float] = []

    def recurse(lo: int, hi: int):
        m = hi - lo
        if m < 2:
            return
        seg = onehot[lo:hi]
        total = seg.sum(axis=0)
        k = int(np.count_nonzero(total))
        if k < 2:
            return
        base_ent = _entropy_from_counts(total, m)

        # All candidate splits i = 1..m-1: left = seg[0:i], right = seg[i:].
        left = np.cumsum(seg, axis=0)[:-1]          # (m-1, C), left[j] = counts of seg[0:j+1]
        right = total - left
        nl = np.arange(1, m, dtype=float)
        nr = m - nl
        el = _entropy_rows(left, nl)
        er = _entropy_rows(right, nr)
        gain = base_ent - (nl / m) * el - (nr / m) * er

        # Cannot cut between equal feature values.
        valid = xs[lo + 1:hi] != xs[lo:hi - 1]
        gain = np.where(valid, gain, -np.inf)

        j = int(np.argmax(gain))                    # leftmost max (matches naive strict >)
        if gain[j] == -np.inf:
            return
        best_i = lo + j + 1

        k1 = int(np.count_nonzero(left[j]))
        k2 = int(np.count_nonzero(right[j]))
        delta = np.log2(3 ** k - 2) - (k * base_ent - k1 * el[j] - k2 * er[j])
        threshold = (np.log2(m - 1) + delta) / m
        if gain[j] <= threshold:
            return

        cuts.append((xs[best_i - 1] + xs[best_i]) / 2.0)
        recurse(lo, best_i)
        recurse(best_i, hi)

    recurse(0, len(xs))
    return sorted(cuts)


def cuts_to_trapezoids(cuts: list[float], lo: float, hi: float,
                       overlap_frac: float = 0.8) -> list:
    """Build overlapping trapezoidal fuzzy sets from cut points.

    Adjacent sets cross at 0.5 over each interior cut; the first/last sets are
    shouldered (full membership out to the feature min/max).
    """
    if hi - lo < 1e-9:
        hi = lo + 1e-6
    bounds = [lo] + list(cuts) + [hi]
    K = len(bounds) - 1

    sets = []
    for i in range(K):
        L, R = bounds[i], bounds[i + 1]
        # Symmetric half-overlap at each interior boundary (matched across the
        # shared cut so neighbors cross at 0.5).
        left_w = 0.0 if i == 0 else 0.5 * overlap_frac * min(bounds[i] - bounds[i - 1], R - L)
        right_w = 0.0 if i == K - 1 else 0.5 * overlap_frac * min(R - L, bounds[i + 2] - R)
        a, b, c, d = L - left_w, L + left_w, R - right_w, R + right_w
        if i == 0:
            a = b = lo
        if i == K - 1:
            c = d = hi
        # Keep monotone a <= b <= c <= d.
        b = min(b, c)
        a = min(a, b)
        d = max(d, c)
        sets.append(fs.FS(name=_term_name(i, K),
                          membership_parameters=[float(a), float(b), float(c), float(d)],
                          domain=[float(lo), float(hi)]))
    return sets


def learn_partitions_mdlp(X: np.ndarray, y: np.ndarray, overlap_frac: float = 0.8,
                          fallback_median: bool = True) -> list:
    """Supervised trapezoidal partitions via MDLP, as a list[fuzzyVariable].

    ``fallback_median`` controls uninformative features (MDLP returns no cut):
    if True they get a single median split (2 terms) so the feature stays
    usable by the tree; if False they get a single all-covering term.
    """
    X = np.asarray(X, dtype=float)
    variables = []
    for j in range(X.shape[1]):
        col = X[:, j]
        lo, hi = float(col.min()), float(col.max())
        cuts = mdlp_cuts(col, y)
        if not cuts and fallback_median and hi > lo:
            cuts = [float(np.median(col))]
        sets = cuts_to_trapezoids(cuts, lo, hi, overlap_frac=overlap_frac)
        variables.append(fs.fuzzyVariable(name=f"feature_{j}", fuzzy_sets=sets))
    return variables


__all__ = ["cuts_to_trapezoids", "learn_partitions_mdlp", "mdlp_cuts"]
