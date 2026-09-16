"""
Deep fuzzy evidential rule trees with learned soft splits (FERL-deep).

:class:`DeepFERL` is the high-accuracy member of the FERL family. :class:`FERL`
grows a compact tree over a fixed or learned partition under a rule budget.
DeepFERL instead grows a deep binary tree recursively. Each split sits at the
weighted-Gini optimal threshold for the samples reaching the node, drawn as a
linear soft ramp whose half-width is the bootstrap spread of that threshold: a
stable cut becomes nearly crisp and an unstable one stays fuzzy. Point
predictions are a soft vote over the leaves, and the evidential read-out uses
the same Dempster--Shafer combination rules as FERL.

The estimator is a port of ``LearnedFuzzyTree`` from the ``fuzzy_greedy_tree``
repository, the model its paper reports as FERL-deep. For the same data and
seed it grows the same tree and makes the same predictions.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from . import _evidence

EPS = 1e-9


def _best_cut(x, y_oh, w, parent, W, criterion="gini"):
    """
    Return ``(gain, threshold)`` for the best crisp cut on one feature.

    ``criterion='gini'`` maximizes the weighted-Gini impurity reduction.
    ``criterion='cci'`` maximizes the Complete Classification Index: the
    accuracy gain of the two children's majority classes over the parent's
    single majority, with a tiny Gini term to break ties. All candidate cuts are
    evaluated at once from cumulative class weights.
    """
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    cum = np.cumsum(y_oh[order] * w[order, None], axis=0)      # (n, C)
    Wl_arr = np.cumsum(w[order])                              # (n,)
    tot = cum[-1]                                             # (C,)
    bnd = np.flatnonzero(xs[:-1] != xs[1:])                   # candidate split rows
    if bnd.size == 0:
        return 0.0, None
    Wl = Wl_arr[bnd]
    Wr = W - Wl
    ok = (Wl > EPS) & (Wr > EPS)
    if not ok.any():
        return 0.0, None
    bnd, Wl, Wr = bnd[ok], Wl[ok], Wr[ok]
    l = cum[bnd]                                              # (m, C)
    r = tot[None, :] - l
    gini_l = 1.0 - ((l / Wl[:, None]) ** 2).sum(1)
    gini_r = 1.0 - ((r / Wr[:, None]) ** 2).sum(1)
    gini_gain = parent - (Wl / W * gini_l + Wr / W * gini_r)  # (m,)
    if criterion == "cci":
        cci = (l.max(1) + r.max(1) - tot.max()) / W          # >= 0 always
        gain = cci + 1e-6 * np.clip(gini_gain, 0.0, None)
    else:
        gain = gini_gain
    k = int(np.argmax(gain))
    if gain[k] <= 0.0:
        return 0.0, None
    i = bnd[k]
    return float(gain[k]), 0.5 * (xs[i] + xs[i + 1])


def _ramp(x, center, h):
    """Membership of the 'below' branch: 1 at ``center - h``, 0 at ``center + h``."""
    return np.clip((center + h - x) / (2.0 * h), 0.0, 1.0)


class DeepFERL(BaseEstimator, ClassifierMixin):
    """
    Deep FERL: a recursively grown fuzzy rule tree with learned soft splits.

    Args:
        max_depth (int, default=12):
            Maximum depth of the tree.
        min_leaf_w (float, default=2.0):
            Minimum fuzzy weight (sum of memberships) each child of a split must
            keep. Nodes with less than twice this weight become leaves.
        n_boot (int, default=25):
            Bootstrap replicates used to estimate each split's location and width
            when ``width="bootstrap"``.
        width ({"bootstrap"} or float, default="bootstrap"):
            ``"bootstrap"`` centers each ramp on the mean of the bootstrap
            thresholds, with half-width equal to their standard deviation. A
            positive float ``c`` centers the ramp on the optimal threshold, with
            half-width ``c`` times the node's weighted feature standard deviation.
        criterion ({"gini", "cci"}, default="gini"):
            Split criterion. Gini is the stronger choice once split locations are
            learned; CCI favors changes to the majority class.
        bounded_support (bool, default=True):
            Gate each split to the node's training range on its feature, widened on
            both sides by ``oob_margin`` times that range. Far out-of-distribution
            samples then reach no leaf and surface as ignorance; in-distribution
            predictions are unaffected.
        oob_margin (float, default=1.0):
            Relative margin of the bounded support gate.
        random_state (int, optional):
            Seed for the bootstrap resampling.

    Attributes:
        classes_ (np.ndarray):
            Class labels seen during ``fit``.
        n_features_in_ (int):
            Number of features seen during ``fit``.
        feature_names_in_ (np.ndarray):
            Feature names, when ``X`` was a DataFrame with string column names.
        root_ (dict):
            Root of the fitted tree. Internal nodes hold the split feature ``f``,
            ramp ``center`` and half-width ``h``, the node's training range
            ``lo``/``hi``, and children ``L`` (below) and ``R`` (above). Every node
            holds its Laplace-smoothed class distribution ``dist`` and fuzzy
            training ``support``.
        n_leaves_ (int):
            Number of leaves, which are the model's rules.
    """

    def __init__(self, max_depth: int = 12, min_leaf_w: float = 2.0, n_boot: int = 25,
                 width="bootstrap", criterion: str = "gini", bounded_support: bool = True,
                 oob_margin: float = 1.0, random_state=None):
        self.max_depth = max_depth
        self.min_leaf_w = min_leaf_w
        self.n_boot = n_boot
        self.width = width
        self.criterion = criterion
        self.bounded_support = bounded_support
        self.oob_margin = oob_margin
        self.random_state = random_state

    # ------------------------------------------------------------------ fitting
    def _validate_params(self):
        if isinstance(self.max_depth, bool) or not isinstance(self.max_depth, (int, np.integer)) \
                or self.max_depth < 0:
            raise ValueError("max_depth must be a non-negative integer.")
        if not self.min_leaf_w > 0:
            raise ValueError("min_leaf_w must be positive.")
        if self.criterion not in ("gini", "cci"):
            raise ValueError("criterion must be either 'gini' or 'cci'.")
        if isinstance(self.width, str):
            if self.width != "bootstrap":
                raise ValueError("width must be 'bootstrap' or a positive number.")
            if isinstance(self.n_boot, bool) or not isinstance(self.n_boot, (int, np.integer)) \
                    or self.n_boot < 2:
                raise ValueError("n_boot must be an integer of at least 2 when width='bootstrap'.")
        elif isinstance(self.width, bool) or not isinstance(self.width, (int, float, np.number)) \
                or not self.width > 0:
            raise ValueError("width must be 'bootstrap' or a positive number.")
        if not self.oob_margin >= 0:
            raise ValueError("oob_margin must be non-negative.")

    def fit(self, X, y):
        """
        Grow the tree.

        Args:
            X (array-like of shape (n_samples, n_features)):
                Training features. Must be finite.
            y (array-like of shape (n_samples,)):
                Class labels.

        Returns:
            DeepFERL
                The fitted estimator.
        """
        self._validate_params()
        if hasattr(X, "columns"):
            columns = np.asarray(X.columns, dtype=object)
            if all(isinstance(name, str) for name in columns):
                self.feature_names_in_ = columns
            elif hasattr(self, "feature_names_in_"):
                del self.feature_names_in_
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")
        if y.ndim != 1:
            y = np.ravel(y)
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples.")
        if len(X) == 0:
            raise ValueError("DeepFERL requires at least one training sample.")
        if not np.all(np.isfinite(X)):
            raise ValueError("DeepFERL does not support NaN or infinite feature values.")

        self.classes_, y_index = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]
        self._y_oh = np.eye(len(self.classes_))[y_index]
        self._X = X
        self._rng = np.random.default_rng(self.random_state)
        self.n_leaves_ = 0
        try:
            self.root_ = self._build(np.ones(len(X)), 0)
        finally:                                               # release training references
            self._X = self._y_oh = self._rng = None
        return self

    def _leaf(self, w):
        self.n_leaves_ += 1
        cnt = (self._y_oh * w[:, None]).sum(0) + 1.0          # Laplace
        return {"leaf": True, "dist": cnt / cnt.sum(), "support": float(w.sum())}

    def _build(self, w, depth):
        W = w.sum()
        if depth >= self.max_depth or W < 2 * self.min_leaf_w:
            return self._leaf(w)
        p = (self._y_oh * w[:, None]).sum(0) / W
        parent = 1.0 - (p ** 2).sum()
        if parent < 1e-9:
            return self._leaf(w)
        region = w > 1e-6
        n_eff = int(region.sum())
        if n_eff < 4:
            return self._leaf(w)
        Xr, yr, wr = self._X[region], self._y_oh[region], w[region]
        Wr = wr.sum()
        pr = 1.0 - (((yr * wr[:, None]).sum(0) / Wr) ** 2).sum()
        bf, bg, bthr = -1, 0.0, None                           # best feature by criterion gain
        for f in range(self._X.shape[1]):
            g, thr = _best_cut(Xr[:, f], yr, wr, pr, Wr, self.criterion)
            if thr is not None and g > bg:
                bf, bg, bthr = f, g, thr
        if bf < 0:
            return self._leaf(w)
        xr = Xr[:, bf]
        if self.width == "bootstrap":
            pw = wr / Wr
            thetas = []
            for _ in range(self.n_boot):
                idx = self._rng.choice(n_eff, n_eff, p=pw)
                _g, tb = _best_cut(xr[idx], yr[idx], np.ones(n_eff), 1.0, float(n_eff), self.criterion)
                if tb is not None:
                    thetas.append(tb)
            if len(thetas) < 2:
                return self._leaf(w)
            center, h = float(np.mean(thetas)), float(np.std(thetas))
        else:
            mean = (wr * xr).sum() / Wr
            std = float(np.sqrt((wr * (xr - mean) ** 2).sum() / Wr))
            center, h = bthr, float(self.width) * std
        h = max(h, 1e-3 * (float(xr.max() - xr.min()) + EPS))
        ml = _ramp(self._X[:, bf], center, h)
        wl, wR = w * ml, w * (1.0 - ml)
        if wl.sum() < self.min_leaf_w or wR.sum() < self.min_leaf_w:
            return self._leaf(w)
        cnt = (self._y_oh * w[:, None]).sum(0) + 1.0          # internal-node consequent
        return {"leaf": False, "f": bf, "center": center, "h": h,
                "lo": float(xr.min()), "hi": float(xr.max()),
                "dist": cnt / cnt.sum(), "support": float(W),
                "L": self._build(wl, depth + 1), "R": self._build(wR, depth + 1)}

    # --------------------------------------------------------------- inference
    def _prepare(self, X, observed_mask):
        check_is_fitted(self, attributes=["classes_", "root_"])
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError("X must be a one- or two-dimensional array.")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but DeepFERL was fitted with "
                f"{self.n_features_in_} features."
            )
        if observed_mask is not None:
            observed_mask = np.asarray(observed_mask, dtype=bool)
            if observed_mask.ndim == 1:
                observed_mask = observed_mask.reshape(1, -1)
            if observed_mask.shape != X.shape:
                raise ValueError("observed_mask must have the same shape as X.")
        return X, observed_mask

    def _split(self, node, X, observed):
        """
        Routing weights ``(below, above)`` of a node, gated to its support.

        A sample whose split feature is unobserved sends half of its weight
        down each branch, so leaf weights still sum to its total routing mass.
        """
        xf = X[:, node["f"]]
        ml = _ramp(xf, node["center"], node["h"])
        if self.bounded_support:
            lo, hi = node["lo"], node["hi"]
            g = self.oob_margin * (hi - lo) + EPS
            gate = np.clip((xf - (lo - g)) / g, 0.0, 1.0) * np.clip(((hi + g) - xf) / g, 0.0, 1.0)
            left, right = ml * gate, (1.0 - ml) * gate
        else:
            left, right = ml, 1.0 - ml
        if observed is not None:
            missing = ~observed[:, node["f"]]
            if missing.any():
                left = np.where(missing, 0.5, left)
                right = np.where(missing, 0.5, right)
        return left, right

    def _accumulate(self, node, X, m, out, observed):
        if node["leaf"]:
            out += m[:, None] * node["dist"]
            return
        left, right = self._split(node, X, observed)
        self._accumulate(node["L"], X, m * left, out, observed)
        self._accumulate(node["R"], X, m * right, out, observed)

    def predict_proba(self, X, observed_mask=None) -> np.ndarray:
        """
        Class probabilities from the soft vote of the leaves.

        Samples that reach no leaf, such as far out-of-distribution inputs under
        ``bounded_support``, receive uniform probabilities.

        Args:
            X (array-like of shape (n_samples, n_features)):
                Samples to predict.
            observed_mask (array-like of bool, optional):
                Same shape as ``X``; ``False`` marks an unobserved feature value.

        Returns:
            np.ndarray
                Probabilities of shape (n_samples, n_classes).
        """
        X, observed = self._prepare(X, observed_mask)
        C = len(self.classes_)
        out = np.zeros((len(X), C))
        self._accumulate(self.root_, X, np.ones(len(X)), out, observed)
        s = out.sum(1)
        out[s <= EPS] = 1.0 / C
        return out / out.sum(1, keepdims=True)

    def predict(self, X, observed_mask=None) -> np.ndarray:
        """Predict the most probable class for each sample."""
        return self.classes_[self.predict_proba(X, observed_mask).argmax(1)]

    def _collect(self, node, X, m, name, acc, observed):
        if name != "r":
            acc.append((name, m, node["dist"], node["support"]))
        if not node["leaf"]:
            left, right = self._split(node, X, observed)
            self._collect(node["L"], X, m * left, name + "_0", acc, observed)
            self._collect(node["R"], X, m * right, name + "_1", acc, observed)

    def node_activation_matrix(self, X, observed_mask=None):
        """
        Path membership of every non-root node.

        Returns:
            tuple
                ``(M, cons, names, support)``: firing of shape (n_samples, n_nodes),
                consequents of shape (n_nodes, n_classes), hierarchical node names
                (a child extends its parent's name with ``"_0"`` or ``"_1"``), and
                each node's fuzzy training support.
        """
        X, observed = self._prepare(X, observed_mask)
        acc = []
        self._collect(self.root_, X, np.ones(len(X)), "r", acc, observed)
        if not acc:
            return (np.zeros((len(X), 0)), np.zeros((0, len(self.classes_))), [],
                    np.zeros(0))
        names = [entry[0] for entry in acc]
        M = np.stack([entry[1] for entry in acc], axis=1)
        cons = np.stack([entry[2] for entry in acc], axis=0)
        support = np.array([entry[3] for entry in acc], float)
        return M, cons, names, support

    @staticmethod
    def leaf_mask(names) -> np.ndarray:
        """Boolean mask of the nodes in ``names`` that have no descendant."""
        return np.array([not any(o != n and o.startswith(n + "_") for o in names)
                         for n in names], dtype=bool)

    @staticmethod
    def node_depths(names) -> np.ndarray:
        """Number of splits on the path to each node in ``names``."""
        return np.array([n.count("_") for n in names], float)

    def firing_strength(self, X, observed_mask=None) -> np.ndarray:
        """
        Total leaf firing per sample.

        In-distribution samples route all of their weight to the leaves (total
        1). Under ``bounded_support`` the total falls toward 0 outside the
        training range, which makes it an out-of-distribution signal.
        """
        M, _, names, _ = self.node_activation_matrix(X, observed_mask)
        if M.shape[1] == 0:
            return np.ones(M.shape[0])
        return M[:, self.leaf_mask(names)].sum(1)

    def predict_ds(self, X, observed_mask=None, leaves_only: bool = False, rule: str = "dempster",
                   top_p: float = None, reliability_vec=None, beta: float = 10.0):
        """
        Combine activated rule nodes as Dempster--Shafer evidence.

        Args:
            X (array-like of shape (n_samples, n_features)):
                Samples to predict.
            observed_mask (array-like of bool, optional):
                Same shape as ``X``; ``False`` marks an unobserved feature value.
            leaves_only (bool, default=False):
                Combine leaf rules only.
            rule (str, default="dempster"):
                One of ``"dempster"``, ``"cautious"``, ``"hybrid"``,
                ``"incremental"``, ``"incremental_local"`` or ``"mixture"``.
            top_p (float, optional):
                Top-p routing threshold applied to each consequent.
            reliability_vec (array-like, optional):
                Per-node reliability in [0, 1], aligned with the combined nodes,
                that discounts firing toward ignorance.
            beta (float, default=10.0):
                Support pseudo-count for the incremental and mixture reliabilities.

        Returns:
            tuple
                ``(betp, belief, plausibility, ignorance)``; the first three have
                shape (n_samples, n_classes) and ignorance has shape (n_samples,).
        """
        _evidence.check_rule(rule)
        M, cons, names, support = self.node_activation_matrix(X, observed_mask)
        if leaves_only and M.shape[1] > 0:
            keep = np.flatnonzero(self.leaf_mask(names))
            M, cons, support = M[:, keep], cons[keep], support[keep]
            names = [names[i] for i in keep]
        if M.shape[1] > 0:
            if top_p is not None:
                M, cons = _evidence.route_top_p(M, cons, top_p)
            if reliability_vec is not None:
                M = M * np.clip(np.asarray(reliability_vec, float), 0.0, 1.0)[None, :]
        betp, belief, plausibility, ignorance, _ = _evidence.combine_evidence(
            M, cons, names, len(self.classes_), rule=rule, support=support, beta=beta)
        return betp, belief, plausibility, ignorance

    def predict_credal(self, X, observed_mask=None, leaves_only: bool = True,
                       rule: str = "dempster", **kwargs):
        """
        Pignistic probabilities, belief, plausibility and ignorance.

        Leaves are combined by default, which avoids repeating the evidence of
        nested internal rules on a deep tree.
        """
        return self.predict_ds(X, observed_mask=observed_mask, leaves_only=leaves_only,
                               rule=rule, **kwargs)

    def predict_set(self, X, observed_mask=None, leaves_only: bool = True,
                    rule: str = "dempster", **kwargs) -> np.ndarray:
        """
        Native credal prediction sets as a boolean class mask.

        A class is kept when its plausibility reaches the largest belief. A
        singleton is a confident prediction; a larger set abstains between its
        classes.
        """
        _, belief, plausibility, _ = self.predict_credal(
            X, observed_mask=observed_mask, leaves_only=leaves_only, rule=rule, **kwargs)
        return plausibility >= belief.max(axis=1, keepdims=True) - 1e-12

    def predict_dirichlet(self, X, observed_mask=None, u_floor: float = 1e-3, **ds_kwargs):
        """
        Dirichlet parameters ``alpha_c = K * Bel(c) / m(Theta) + 1`` per sample.

        ``u_floor`` keeps ignorance away from zero so confident samples give a
        finite concentration. Extra keyword arguments go to :meth:`predict_ds`.
        """
        _, belief, _, ignorance = self.predict_ds(X, observed_mask=observed_mask, **ds_kwargs)
        u = np.clip(ignorance, u_floor, 1.0)
        return len(self.classes_) * belief / u[:, None] + 1.0

    # ------------------------------------------------------------- inspection
    def n_rules(self) -> int:
        """Number of rules, i.e. leaves."""
        check_is_fitted(self, attributes=["root_"])
        return int(self.n_leaves_)

    def get_tree_stats(self) -> dict:
        """Structural statistics: leaves, internal nodes and leaf depths."""
        check_is_fitted(self, attributes=["root_"])
        leaf_depths, internal = [], 0
        stack = [(self.root_, 0)]
        while stack:
            node, depth = stack.pop()
            if node["leaf"]:
                leaf_depths.append(depth)
            else:
                internal += 1
                stack.append((node["R"], depth + 1))
                stack.append((node["L"], depth + 1))
        return {"leaves": len(leaf_depths), "internal_nodes": internal,
                "max_depth": int(max(leaf_depths)), "mean_leaf_depth": float(np.mean(leaf_depths)),
                "total_leaf_depth": int(sum(leaf_depths))}

    def print_tree(self, feature_names=None, decimals: int = 3) -> None:
        """
        Print the tree as soft threshold rules.

        Each split reads "feature is below/above center (± half-width)", and
        each leaf shows its most probable class, that probability and its
        support.
        """
        check_is_fitted(self, attributes=["root_"])
        if feature_names is None:
            feature_names = getattr(self, "feature_names_in_", None)
        if feature_names is None:
            feature_names = [f"x{index}" for index in range(self.n_features_in_)]

        def describe(node):
            best = int(np.argmax(node["dist"]))
            return (f"{self.classes_[best]} (p={node['dist'][best]:.{decimals}f}, "
                    f"support={node['support']:.1f})")

        print(f"root: {describe(self.root_)}")

        def walk(node, prefix):
            if node["leaf"]:
                return
            name = feature_names[node["f"]]
            spread = f"{node['center']:.{decimals}g} (±{node['h']:.{decimals}g})"
            for index, (branch, word) in enumerate(((node["L"], "below"), (node["R"], "above"))):
                last = index == 1
                marker = "└── " if last else "├── "
                label = "leaf: " + describe(branch) if branch["leaf"] else describe(branch)
                print(f"{prefix}{marker}{name} {word} {spread} -> {label}")
                walk(branch, prefix + ("    " if last else "│   "))

        walk(self.root_, "")
