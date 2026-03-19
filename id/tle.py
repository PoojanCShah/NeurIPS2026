"""
Tight Local intrinsic dimensionality Estimator (TLE).

Reference
---------
Amsaleg et al. (2019). Intrinsic dimensionality estimation within tight
    localities. Proceedings of the 2019 SIAM International Conference on
    Data Mining (SDM), pp. 181–189.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from ._utils import knn


class TLE:
    """Intrinsic dimension via the Tight Local ID Estimator (Amsaleg et al., 2019).

    Algorithm
    ---------
    For each point and its k nearest neighbors:

    Let Di = dist from the query to the i-th neighbor, r = D_k (farthest NN),
    and V = pairwise distance matrix between the k neighbors themselves.

    Two families of log-ratio measurements are derived from the geometry:

        S_ij — "direct" projection measurement
        T_ij — "reflected" projection measurement

    Four degenerate boundary cases (Di=0, Dj=0, Vij=0, Di=r) are handled
    explicitly. Near-zero measurements (< epsilon) are dropped for stability.

    The local ID estimate is:

        ID = −2 (k² − n_dropped) / (Σ log(S/r) + Σ log(T/r) + 2·Σ log(d/r))

    where the last sum is over the k NN distances.

    Parameters
    ----------
    epsilon : float
        Measurements S_ij or T_ij below this threshold are discarded.
    n_neighbors : int
        Neighbourhood size k.
    """

    def __init__(self, epsilon: float = 1e-4, n_neighbors: int = 20):
        self.epsilon = epsilon
        self.n_neighbors = n_neighbors

    # ── Per-neighborhood TLE ──────────────────────────────────────────────────

    def _idtle(self, nn, dists_row):
        """
        Parameters
        ----------
        nn        : (k, n_features) nearest-neighbor coordinates
        dists_row : (1, k) NN distances sorted ascending
        """
        r = dists_row[0, -1]
        if r == 0:
            raise ValueError("All k-NN distances are zero — degenerate neighborhood.")

        k = dists_row.shape[1]
        V = squareform(pdist(nn))

        Di = np.tile(dists_row.T, (1, k))   # Di[i,j] = dist to i-th NN
        Dj = Di.T                            # Dj[i,j] = dist to j-th NN

        # ── S and T measurements ──────────────────────────────────────────────
        Z2 = 2 * Di ** 2 + 2 * Dj ** 2 - V ** 2
        disc_S = (Di ** 2 + V ** 2 - Dj ** 2) ** 2 + 4 * V ** 2 * (r ** 2 - Di ** 2)
        disc_T = (Di ** 2 + Z2 - Dj ** 2) ** 2 + 4 * Z2 * (r ** 2 - Di ** 2)
        denom = 2 * (r ** 2 - Di ** 2)

        # denom is zero when Di==r; the boundary block below overwrites those
        # cells, so divide-by-zero here produces intermediate inf/nan that are
        # harmless.  Suppress the numpy warnings explicitly.
        with np.errstate(divide="ignore", invalid="ignore"):
            S = r * (np.sqrt(disc_S) - (Di ** 2 + V ** 2 - Dj ** 2)) / denom
            T = r * (np.sqrt(disc_T) - (Di ** 2 + Z2 - Dj ** 2)) / denom

        # ── Boundary: Di == r (repeated distances) ────────────────────────────
        Dr = (dists_row == r).squeeze()
        with np.errstate(divide="ignore", invalid="ignore"):
            S[Dr, :] = r * V[Dr, :] ** 2 / (r ** 2 + V[Dr, :] ** 2 - Dj[Dr, :] ** 2)
            T[Dr, :] = r * Z2[Dr, :] / (r ** 2 + Z2[Dr, :] - Dj[Dr, :] ** 2)

        # ── Boundary: Di == 0 ─────────────────────────────────────────────────
        Di0 = (Di == 0).squeeze()
        S[Di0] = Dj[Di0]
        T[Di0] = Dj[Di0]

        # ── Boundary: Dj == 0 ─────────────────────────────────────────────────
        Dj0 = (Dj == 0).squeeze()
        S[Dj0] = r * V[Dj0] / (r + V[Dj0])
        T[Dj0] = r * V[Dj0] / (r + V[Dj0])

        # ── Boundary: Vij == 0 (identical neighbors) ──────────────────────────
        V0 = (V == 0).copy()
        np.fill_diagonal(V0, False)
        S[V0] = r
        T[V0] = r
        nV0 = int(np.sum(V0))

        # ── Drop near-zero or non-positive S or T ────────────────────────────
        TSeps = (T < self.epsilon) | (S < self.epsilon) | ~np.isfinite(T) | ~np.isfinite(S)
        np.fill_diagonal(TSeps, False)
        nTSeps = int(np.sum(TSeps))
        S[TSeps] = r
        T[TSeps] = r

        # ── Log-ratio sums ────────────────────────────────────────────────────
        # Take log first (invalid entries already set to r → log(r/r)=0),
        # then zero the diagonal so self-pairs don't contribute.
        with np.errstate(divide="ignore", invalid="ignore"):
            S = np.log(S / r)
            T = np.log(T / r)
        np.fill_diagonal(S, 0)
        np.fill_diagonal(T, 0)
        s1s = np.sum(S)
        s1t = np.sum(T)

        d_flat = dists_row.flatten()
        nDeps = int(np.sum(d_flat < self.epsilon))
        s2 = np.sum(np.log(d_flat[nDeps:] / r))

        n_valid = k ** 2 - nTSeps - nDeps - nV0
        return -2 * n_valid / (s1t + s1s + 2 * s2)

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n = len(X)

        k = min(self.n_neighbors, n - 1)
        dists, idx = knn(X, k)

        dims = np.zeros(n)
        for i in range(n):
            dims[i] = self._idtle(X[idx[i]], dists[[i], :])

        self.dimension_pw_ = dims
        self.dimension_ = float(np.mean(dims))
        return self
