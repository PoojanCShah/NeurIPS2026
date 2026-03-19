"""
Expected Simplex Skewness estimator (ESS).

Reference
---------
Johnsson et al. (2015). Low bias local intrinsic dimension estimation from
    expected simplex skewness. IEEE Transactions on Pattern Analysis and
    Machine Intelligence, 37(1), 196–202.
"""

import bisect
import numpy as np
from scipy.special import gamma
from ._utils import knn, lens, efficient_indnComb


class ESS:
    """Intrinsic dimension via Expected Simplex Skewness (Johnsson et al., 2015).

    Algorithm
    ---------
    The ESS statistic is computed inside each point's k-NN neighborhood
    (treated as a locally uniform ball):

    **Version 'a'** (default, any d):
        Sample random (d+1)-simplices from the centered neighborhood vectors.
        Compute each simplex's volume (via Gram determinant) and normalise by
        the product of its edge lengths:

            ESS = Σ vol(simplex) / Σ (product of edge lengths)

    **Version 'b'** (d=1 only):
        Sample random pairs of centered vectors and compute the absolute dot
        product normalised by the product of norms:

            ESS = Σ |v_i · v_j| / Σ (‖v_i‖ · ‖v_j‖)

    The observed ESS is then mapped to a dimension by comparing it to
    precomputed theoretical reference values (closed-form via Gamma functions)
    and linearly interpolating between adjacent integers.

    Parameters
    ----------
    ver : str
        Version 'a' or 'b'.
    d : int
        Simplex order; ver='b' only supports d=1.
    n_neighbors : int
        Neighbourhood size.
    random_state : int or None
        Seed for combination sampling.
    """

    def __init__(self, ver: str = "a", d: int = 1, n_neighbors: int = 20,
                 random_state=None):
        self.ver = ver
        self.d = d
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    # ── Theoretical reference values ─────────────────────────────────────────

    def _ess_reference(self, maxdim, mindim=1):
        """Theoretical ESS values for integer dimensions mindim … maxdim."""
        if self.ver == "a":
            # factor1(n) = Γ(n/2) / Γ((n+1)/2)
            J1 = np.arange(1, maxdim + 1, 2)
            J2 = np.arange(2, maxdim + 1, 2)
            factor1 = np.full(maxdim, np.nan)
            if len(J1):
                factor1[J1 - 1] = (gamma(0.5) / gamma(1.0)
                                   * np.concatenate(([1], np.cumprod(J1[:-1] / (J1[:-1] + 1)))))
            if len(J2):
                factor1[J2 - 1] = (gamma(1.0) / gamma(1.5)
                                   * np.concatenate(([1], np.cumprod(J2[:-1] / (J2[:-1] + 1)))))

            # factor2(n) = Γ(n/2) / Γ((n−d)/2)
            K1 = np.arange(self.d + 1, maxdim + 1, 2)
            K2 = np.arange(self.d + 2, maxdim + 1, 2)
            factor2 = np.zeros(maxdim)
            if len(K1):
                factor2[K1 - 1] = (gamma((self.d + 1) / 2) / gamma(0.5)
                                   * np.concatenate(([1], np.cumprod(K1[:-1] / (K1[:-1] - self.d)))))
            if len(K2):
                factor2[K2 - 1] = (gamma((self.d + 2) / 2) / gamma(1.0)
                                   * np.concatenate(([1], np.cumprod(K2[:-1] / (K2[:-1] - self.d)))))

            return (factor1 ** self.d * factor2)[mindim - 1: maxdim]

        if self.ver == "b":
            if self.d != 1:
                raise ValueError("ver='b' only supports d=1.")
            # ID(n) = 2π^(−1/2) / n · Γ((n+1)/2) / Γ((n+2)/2)
            J1 = np.arange(1, maxdim + 1, 2)
            J2 = np.arange(2, maxdim + 1, 2)
            ID = np.full(maxdim, np.nan)
            if len(J1):
                ID[J1 - 1] = (gamma(1.5) / gamma(1.0)
                              * np.concatenate(([1], np.cumprod((J1[:-1] + 2) / (J1[:-1] + 1)))))
            if len(J2):
                ID[J2 - 1] = (gamma(2.0) / gamma(1.5)
                              * np.concatenate(([1], np.cumprod((J2[:-1] + 2) / (J2[:-1] + 1)))))
            ns = np.arange(mindim, maxdim + 1)
            return ID[mindim - 1: maxdim] * 2 / np.sqrt(np.pi) / ns

        raise ValueError(f"Unknown ver='{self.ver}'. Use 'a' or 'b'.")

    # ── ESS statistic from a single neighborhood ──────────────────────────────

    def _compute_ess(self, X_local):
        """Compute the ESS statistic from a local neighborhood patch."""
        p = self.d + 1
        if p > X_local.shape[1]:
            return 0.0 if self.ver == "a" else 1.0

        vecs = X_local - X_local.mean(axis=0)
        groups = efficient_indnComb(len(vecs), p, self.rng_)
        Alist = [vecs[g] for g in groups]
        weight = np.prod([lens(A) for A in Alist], axis=1)

        if self.ver == "a":
            vol = np.sqrt(np.abs([np.linalg.det(A @ A.T) for A in Alist]))
            return np.sum(vol) / np.sum(weight)

        if self.ver == "b":
            proj = [abs(A[0] @ A[1]) for A in Alist]
            return np.sum(proj) / np.sum(weight)

        raise ValueError(f"Unknown ver='{self.ver}'.")

    # ── Map ESS value → fractional dimension ─────────────────────────────────

    def _ess_to_dim(self, essval):
        if np.isnan(essval):
            return np.nan

        mindim, maxdim = 1, 20
        refs = self._ess_reference(maxdim, mindim)

        while ((self.ver == "a" and essval > refs[-1])
               or (self.ver == "b" and essval < refs[-1])):
            new_mindim, new_maxdim = maxdim + 1, 2 * maxdim
            refs = np.append(refs, self._ess_reference(new_maxdim, new_mindim))
            mindim, maxdim = 1, new_maxdim

        if self.ver == "a":
            i = bisect.bisect(refs[mindim - 1:], essval)
        else:
            i = len(refs[mindim - 1:]) - bisect.bisect(refs[mindim - 1:][::-1], essval)

        de_int = mindim + i - 1
        frac = (essval - refs[de_int - 1]) / (refs[de_int] - refs[de_int - 1])
        return de_int + frac

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n = len(X)
        self.rng_ = np.random.default_rng(self.random_state)

        k = min(self.n_neighbors, n - 1)
        _, idx = knn(X, k)

        ess_vals = np.zeros(n)
        dims = np.zeros(n)
        for i in range(n):
            ess_vals[i] = self._compute_ess(X[idx[i]])
            dims[i] = self._ess_to_dim(ess_vals[i])

        self.dimension_pw_ = dims
        self.essval_ = ess_vals
        self.dimension_ = float(np.nanmean(dims))
        return self
