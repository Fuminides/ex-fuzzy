"""
Dempster--Shafer evidence combination shared by the FERL estimators.

A rule node ``n`` firing with strength ``mu`` is a simple mass function: it
commits ``mu * p_n(c)`` to each class ``c`` and ``1 - mu`` to ignorance
(Theta). This module combines those masses across nodes. Estimators decide
which nodes take part, how firing is discounted and which support counts to
use before calling it, so each keeps its own semantics while the combination
rules exist once.
"""
import numpy as np

#: Combination rules understood by :func:`combine_evidence`.
RULES = ("dempster", "cautious", "hybrid", "incremental", "incremental_local", "mixture")


def check_rule(rule: str) -> None:
    """Raise ``ValueError`` for an unknown combination rule."""
    if rule not in RULES:
        raise ValueError(
            "rule must be one of " + ", ".join(repr(name) for name in RULES) + f"; got {rule!r}."
        )


def route_top_p(M: np.ndarray, cons: np.ndarray, top_p: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply top-p (nucleus) routing to rule consequents.

    Each node keeps its most probable classes until their cumulative consequent
    reaches ``top_p``. The kept consequent is renormalized and the node's firing
    is discounted by the kept mass, so the dropped tail moves to ignorance.

    Args:
        M: Firing strengths of shape ``(n_samples, n_nodes)``.
        cons: Class consequents of shape ``(n_nodes, n_classes)``.
        top_p: Cumulative probability threshold.

    Returns:
        The discounted firing matrix and the routed consequents.
    """
    order = np.argsort(-cons, axis=1)
    sorted_cons = np.take_along_axis(cons, order, axis=1)
    before = np.cumsum(sorted_cons, axis=1) - sorted_cons      # cumulative strictly before each
    kept = np.zeros_like(cons)
    np.put_along_axis(kept, order, np.where(before < top_p, sorted_cons, 0.0), axis=1)
    kept_mass = kept.sum(1)
    return M * kept_mass[None, :], kept / np.clip(kept_mass[:, None], 1e-12, None)


def combine_evidence(M: np.ndarray, cons: np.ndarray, names: list[str], n_classes: int,
                     rule: str = "dempster", firing: np.ndarray = None,
                     incremental_reliability: np.ndarray = None, support: np.ndarray = None,
                     beta: float = 10.0):
    """
    Combine activated rule nodes into belief, plausibility and ignorance.

    Args:
        M: Discounted firing strengths of shape ``(n_samples, n_nodes)``.
        cons: Class consequents of shape ``(n_nodes, n_classes)``.
        names: Hierarchical node names, where a child's name extends its
            parent's name with ``"_"``. The ``hybrid`` and incremental rules
            read the tree structure from them.
        n_classes: Number of classes.
        rule: One of :data:`RULES`.
        firing: Firing used for the incremental rules' per-sample discount.
            Defaults to ``M``.
        incremental_reliability: Per-node reliability for the incremental
            rules' base masses. Defaults to ``support / (support + beta)``.
        support: Per-node training support, used by ``mixture`` and by the
            incremental default above. Defaults to one for every node.
        beta: Support pseudo-count for those reliabilities.

    Returns:
        ``(betp, belief, plausibility, ignorance, diagnostics)``. The first
        three have shape ``(n_samples, n_classes)``, ignorance has shape
        ``(n_samples,)``, and ``diagnostics`` describes incremental residuals
        (``None`` for other rules).
    """
    check_rule(rule)
    N, K = M.shape
    C = n_classes
    if K == 0:                                                 # no rules -> total ignorance
        return np.full((N, C), 1.0 / C), np.zeros((N, C)), np.ones((N, C)), np.ones(N), None

    firing = M if firing is None else firing
    one_minus = 1.0 - M                                        # (N, K)
    term = M[:, :, None] * cons[None, :, :] + one_minus[:, :, None]  # (N, K, C)
    diagnostics = None

    def _cautious_mass(cols):
        """Cautious-combined (m_c, m_theta) over a subset of node columns."""
        w_nc = one_minus[:, cols, None] / np.clip(term[:, cols, :], 1e-12, None)
        w_c = np.clip(w_nc.min(axis=1), 1e-12, 1.0)            # (N, C)
        inv = 1.0 / w_c
        S = (1.0 - C) + inv.sum(axis=1)                        # (N,), >= 1
        return (inv - 1.0) / S[:, None], 1.0 / S

    if rule in ("incremental", "incremental_local"):
        # Incremental (residual) evidence: along each root->leaf chain keep the
        # root's full belief and replace every descendant by its residual over the
        # immediate parent, so shared ancestor evidence is counted once while the
        # refinement's independent increment is preserved. A child that contradicts
        # its parent on some class keeps its own full mass instead, so the rule
        # interpolates between specificity and Dempster. 'incremental_local'
        # discounts a residual node by its conditional firing mu_k / mu_parent so
        # the parent's firing is not re-counted in the increment.
        local = rule == "incremental_local"
        if incremental_reliability is None:
            node_support = np.ones(K) if support is None else np.asarray(support, float)
            incremental_reliability = node_support / (node_support + beta)
        r_inc = np.asarray(incremental_reliability, float)
        t = np.clip(1.0 - r_inc, 1e-9, 1.0)                    # (K,) Theta mass
        a = r_inc[:, None] * cons                              # (K, C) singleton mass
        g = a + t[:, None]                                     # (K, C) base commonality

        # Immediate active parent of each node: the longest name that is a prefix.
        parent = np.full(K, -1, dtype=int)
        for kk, nk in enumerate(names):
            best_length = -1
            for j, nj in enumerate(names):
                if j != kk and nk.startswith(nj + "_") and len(nj) > best_length:
                    parent[kk], best_length = j, len(nj)

        a_eff = a.copy()                                       # effective singleton mass
        t_eff = t.copy()                                       # effective Theta mass
        is_residual = np.zeros(K, dtype=bool)
        n_nonroot = int((parent >= 0).sum())
        n_invalid = 0
        for kk in range(K):
            p = parent[kk]
            if p < 0:                                          # root component: full mass
                continue
            q_res = g[kk] / np.clip(g[p], 1e-12, None)         # (C,)
            q_res_theta = float(np.clip(t[kk] / t[p], 0.0, 1.0))
            a_res = q_res - q_res_theta
            if np.all(a_res >= -1e-9):                          # additive refinement
                a_eff[kk] = np.clip(a_res, 0.0, None)
                t_eff[kk] = q_res_theta
                is_residual[kk] = True
            else:                                              # exception -> keep full mass
                n_invalid += 1
        diagnostics = {
            'n_nonroot': n_nonroot, 'n_invalid': n_invalid,
            'residual_invalid_rate': (n_invalid / n_nonroot) if n_nonroot else 0.0,
        }
        mu = firing.copy()                                     # (N, K)
        if local and is_residual.any():
            parents = parent[is_residual]
            mu[:, is_residual] = firing[:, is_residual] / np.clip(firing[:, parents], 1e-12, None)
            mu = np.clip(mu, 0.0, 1.0)
        f_theta = 1.0 - mu * (1.0 - t_eff[None, :])            # (N, K) discounted Theta
        f_c = mu[:, :, None] * a_eff[None, :, :] + f_theta[:, :, None]  # (N, K, C) commonality
        Qc = np.prod(f_c, axis=1)
        Qt = np.prod(f_theta, axis=1)
        m_c_un = np.clip(Qc - Qt[:, None], 0.0, None)
        total = np.where(m_c_un.sum(1) + Qt <= 0, 1.0, m_c_un.sum(1) + Qt)
        m_c, m_theta = m_c_un / total[:, None], Qt / total
    elif rule == "mixture":
        # Convex (additive) combination; valid when the combined nodes form a
        # partition of unity, e.g. the leaves of a tree. It never multiplies one
        # node's evidence against another's, so shared ancestors are not
        # double-counted.
        node_support = np.ones(K) if support is None else np.asarray(support, float)
        rho = node_support / (node_support + beta)
        m_c = M @ (rho[:, None] * cons)                        # (N, C)
        m_theta = np.clip(1.0 - m_c.sum(1), 0.0, 1.0)          # unassigned -> Theta
    elif rule == "cautious":
        m_c, m_theta = _cautious_mass(np.arange(K))
    elif rule == "hybrid":
        # Structure-aware: cautious within each root->leaf chain (dependent, nested
        # rules), then Dempster across chains (independent branches).
        leaves = [i for i, n in enumerate(names)
                  if not any(o != n and o.startswith(n + "_") for o in names)]
        Qc, Qt = np.ones((N, C)), np.ones(N)
        for leaf in leaves:
            leaf_name = names[leaf]
            chain = [j for j, n in enumerate(names) if leaf_name == n or leaf_name.startswith(n + "_")]
            mc, mt = _cautious_mass(chain)
            Qc *= (mc + mt[:, None])
            Qt *= mt
        m_c_un = np.clip(Qc - Qt[:, None], 0.0, None)
        total = np.where(m_c_un.sum(1) + Qt <= 0, 1.0, m_c_un.sum(1) + Qt)
        m_c, m_theta = m_c_un / total[:, None], Qt / total
    else:                                                      # dempster
        Qc = np.prod(term, axis=1)                             # (N, C)
        Qtheta = np.prod(one_minus, axis=1)                    # (N,)
        m_c_un = np.clip(Qc - Qtheta[:, None], 0.0, None)
        total = m_c_un.sum(axis=1) + Qtheta
        total = np.where(total <= 0, 1.0, total)
        m_c = m_c_un / total[:, None]
        m_theta = Qtheta / total

    belief = m_c
    plausibility = m_c + m_theta[:, None]
    betp = m_c + m_theta[:, None] / C
    return betp, belief, plausibility, m_theta, diagnostics
