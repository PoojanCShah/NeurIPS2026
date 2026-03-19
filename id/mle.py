"""
Maximum Likelihood Estimator (MLE) for intrinsic dimension.

References
----------
Levina & Bickel (2005). Maximum likelihood estimation of intrinsic dimension.
    NeurIPS 17.
Hill (1975). A simple general approach to inference about the tail of a distribution.
    Annals of Statistics, 3(5), 1163–1174.
"""

import numpy as np
from ._utils import knn


class MLE:
    """Intrinsic dimension via Maximum Likelihood (Levina & Bickel, 2005).

    Algorithm
    ---------
    For each point, given its k nearest-neighbor distances R1 ≤ R2 ≤ … ≤ Rk:

        d̂_i = (k − 1) / Σ_{j=1}^{k} log(Rk / Rj)

    The sum has k terms, but the last (j=k) contributes log(1) = 0, so it is
    effectively a sum of k−1 nonzero terms (Hill estimator).

    The global estimate combines pointwise estimates via one of:
      - 'mle'    : harmonic mean  1 / mean(1 / d̂_i)   [default, matches MLE theory]
      - 'mean'   : arithmetic mean
      - 'median' : median

    Parameters
    ----------
    n_neighbors : int
        Number of nearest neighbors used per local estimate.
    comb : str
        Aggregation strategy: 'mle' | 'mean' | 'median'.
    unbiased : bool
        If True, use k−2 in the numerator instead of k−1 (bias correction).
    """

    def __init__(self, n_neighbors: int = 20, comb: str = "mle", unbiased: bool = False):
        self.n_neighbors = n_neighbors
        self.comb = comb
        self.unbiased = unbiased

    # ── Core per-neighborhood estimator ──────────────────────────────────────

    def _local_dim(self, Rs: np.ndarray) -> float:
        """Hill / MLE estimate from a sorted array of NN distances Rs."""
        k = len(Rs)
        kfac = k - 2 if self.unbiased else k - 1
        Rk = Rs[-1]
        if Rk == 0:
            return np.nan
        total = np.sum(np.log(Rk / Rs[Rs > 0]))   # skip zero distances
        if total == 0:
            return np.nan
        return kfac / total

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n = len(X)

        k = min(self.n_neighbors, n - 1)
        dists, _ = knn(X, k)          # shape (n, k), sorted ascending

        # ── Step 1: pointwise estimates ───────────────────────────────────────
        self.dimension_pw_ = np.array([self._local_dim(dists[i]) for i in range(n)])

        # ── Step 2: combine ───────────────────────────────────────────────────
        pw = self.dimension_pw_[np.isfinite(self.dimension_pw_)]
        if self.comb == "mle":
            self.dimension_ = 1.0 / np.mean(1.0 / pw)
        elif self.comb == "mean":
            self.dimension_ = np.mean(pw)
        elif self.comb == "median":
            self.dimension_ = np.median(pw)
        else:
            raise ValueError(f"Unknown comb='{self.comb}'. Use 'mle', 'mean', or 'median'.")

        return self
