"""
Two Nearest Neighbors estimator (TwoNN).

Reference
---------
Facco et al. (2017). Estimating the intrinsic dimension of datasets by a minimal
    neighborhood information. Scientific Reports, 7, 12140.
    https://doi.org/10.1038/s41598-017-11873-y
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from ._utils import knn


class TwoNN:
    """Intrinsic dimension via the TwoNN algorithm (Facco et al., 2017).

    Algorithm
    ---------
    For each point compute the ratio of its 2nd to 1st nearest-neighbor distance:

        μ_i = r2_i / r1_i

    Under a locally uniform d-dimensional distribution, μ follows:

        P(μ > u) = u^(−d)   ⟺   −log(1 − F(μ)) = d · log(μ)

    A forced-through-origin linear regression of −log(1 − F_emp(μ)) vs log(μ)
    over the sorted (and partially discarded) μ values gives the slope d.

    Parameters
    ----------
    discard_fraction : float
        Fraction (0–1) of points with the largest μ to discard before fitting.
        The paper recommends 0.1 as a heuristic.
    """

    def __init__(self, discard_fraction: float = 0.1):
        self.discard_fraction = discard_fraction

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        N = len(X)

        # ── Step 1: ratio μ = r2 / r1 for every point ────────────────────────
        dists, _ = knn(X, k=2)          # shape (N, 2): columns are r1, r2
        r1, r2 = dists[:, 0], dists[:, 1]
        mu = r2 / r1

        # Drop points where r1 = 0 (exact duplicates) — μ is undefined
        mu = mu[np.isfinite(mu) & (mu > 0)]

        # ── Step 2: sort and discard the largest discard_fraction ─────────────
        n_keep = int(N * (1 - self.discard_fraction))
        mu_sorted = np.sort(mu)[:n_keep]

        # ── Step 3: empirical complementary CDF ──────────────────────────────
        # F_emp(i) = i / N  (denominator is N, not n_keep, per the paper)
        F_emp = np.arange(len(mu_sorted)) / N

        # ── Step 4: forced-origin linear regression ───────────────────────────
        # −log(1 − F_emp) = d · log(μ)
        x = np.log(mu_sorted).reshape(-1, 1)
        y_reg = -np.log(1 - F_emp).reshape(-1, 1)

        # Keep only finite regression inputs
        valid = np.isfinite(x.ravel()) & np.isfinite(y_reg.ravel())
        lr = LinearRegression(fit_intercept=False)
        lr.fit(x[valid], y_reg[valid])

        self.dimension_ = float(lr.coef_[0, 0])
        self.mu_ = mu_sorted
        self.x_ = x
        self.y_ = y_reg
        return self
