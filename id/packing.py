"""
Packing Number Intrinsic Dimension estimator.

Reference
---------
Kégl, B. (2002). Intrinsic dimension estimation using packing numbers.
    Advances in Neural Information Processing Systems 15 (NIPS 2002).
"""

import numpy as np
from ._utils import knn


class PackingDim:
    """Intrinsic dimension via greedy packing numbers (Kégl, NIPS 2002).

    Algorithm
    ---------
    The capacity dimension satisfies M(r) ∝ r^{-D}, where M(r) is the
    r-packing number (maximum size of a subset whose points are mutually
    at distance ≥ r apart). A greedy approximation M̂(r) is computed by
    scanning a randomly permuted copy of the data and retaining each point
    that lies at distance ≥ r from every already-retained point.

    To reduce ordering-induced variance, the procedure is repeated on
    independent random permutations. The stopping criterion from Figure 2
    of the paper halts once the 95%-CI half-width on D̂ falls below
    D̂*(1-ε)/2 (ε = 0.01 gives 99% accuracy nine times in ten).

    The dimension estimate is:

        D̂ = -(E[log M̂(r2)] − E[log M̂(r1)]) / (log r2 − log r1)

    Parameters
    ----------
    k1, k2 : int
        Radii r1 and r2 are the median k1-th and k2-th nearest-neighbour
        distances across the dataset (same convention as CorrInt).
    epsilon : float
        Accuracy parameter ε for the stopping criterion. Paper uses 0.01.
    max_iter : int
        Hard upper limit on the number of permutation repetitions.
    random_state : int or None
    """

    def __init__(self, k1: int = 10, k2: int = 20, epsilon: float = 0.01,
                 max_iter: int = 1000, random_state=None):
        self.k1 = k1
        self.k2 = k2
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.random_state = random_state

    # ── Greedy packing ────────────────────────────────────────────────────────

    @staticmethod
    def _greedy_pack(X: np.ndarray, r: float) -> int:
        """Return the size of a greedy r-packing of X (in its given order).

        Scans X row-by-row; a point is added to the packing set C only if
        its squared L2 distance to every current member of C is ≥ r².
        """
        n, D = X.shape
        centers = np.empty((n, D))
        centers[0] = X[0]
        n_c = 1
        r_sq = r * r
        for i in range(1, n):
            diff = centers[:n_c] - X[i]            # (n_c, D)
            sq_dists = (diff * diff).sum(axis=1)   # (n_c,)
            if sq_dists.min() >= r_sq:
                centers[n_c] = X[i]
                n_c += 1
        return n_c

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n = len(X)
        rng = np.random.default_rng(self.random_state)

        # ── Step 1: choose r1, r2 from median kNN distances ───────────────────
        k2 = min(self.k2, n - 1)
        k1 = min(self.k1, k2 - 1)
        dists, _ = knn(X, k2)
        r1 = float(np.median(dists[:, k1 - 1]))
        r2 = float(np.median(dists[:, k2 - 1]))

        if r1 <= 0 or r2 <= r1:
            raise ValueError(
                f"Degenerate radii r1={r1:.4g}, r2={r2:.4g}. "
                "Try increasing k1/k2 or using a larger dataset."
            )

        log_r_diff = np.log(r2) - np.log(r1)   # > 0

        # ── Step 2: repeat packing on random permutations ─────────────────────
        # (Figure 2, Kégl 2002)
        logs1: list[float] = []
        logs2: list[float] = []

        for _ in range(self.max_iter):
            X_perm = X[rng.permutation(n)]

            m1 = max(1, self._greedy_pack(X_perm, r1))
            m2 = max(1, self._greedy_pack(X_perm, r2))

            logs1.append(np.log(m1))
            logs2.append(np.log(m2))

            L = len(logs1)
            if L > 10:
                L1 = np.array(logs1)
                L2 = np.array(logs2)
                D_hat = -(L2.mean() - L1.mean()) / log_r_diff
                # Stopping criterion from paper (eq. line 13, Fig 2)
                var_sum = np.var(L1, ddof=1) + np.var(L2, ddof=1)
                ci_hw = 1.65 * np.sqrt(var_sum) / (np.sqrt(L) * log_r_diff)
                if D_hat > 0 and ci_hw < D_hat * (1 - self.epsilon) / 2:
                    break

        L1 = np.array(logs1)
        L2 = np.array(logs2)
        self.dimension_ = float(-(L2.mean() - L1.mean()) / log_r_diff)
        self.n_iter_    = len(logs1)
        self.r1_        = r1
        self.r2_        = r2
        return self
