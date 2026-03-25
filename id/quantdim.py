"""
Quantization Dimension estimator (r = 2 case).

Reference
---------
Raginsky, M. & Lazebnik, S. (2005). Estimation of intrinsic dimensionality using
    high-rate vector quantization. Advances in Neural Information Processing
    Systems 18 (NIPS 2005).
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression


class QuantDim:
    """Intrinsic dimension via quantization dimension (Raginsky & Lazebnik, 2005).

    Algorithm
    ---------
    For data on a d-dimensional manifold the optimal r=2 quantization error
    satisfies e*(k|μ) = Θ(k^{−1/d}), so plotting log(k) vs −log(e_test)
    over a geometric grid of codebook sizes yields a line of slope 1/d.

    Procedure (§2.1 and §3 of the paper):
    1. Split X randomly into training and test halves.
    2. For each k in a geometric grid [k_min, k_max], run k-means on the
       training half and compute the RMS nearest-centroid error on the test
       half.
    3. Fit OLS to (log k, −log e_test); the slope is 1/d.

    Using a held-out test set is essential: training error optimistically
    biases the slope downward (overestimates d), as shown in Table 1 of the
    paper.

    Parameters
    ----------
    k_min : int or None
        Minimum codebook size. Defaults to max(2, n_train // 50).
    k_max : int or None
        Maximum codebook size. Defaults to n_train // 4.
    n_codebooks : int
        Number of k values sampled in [k_min, k_max] on a log scale.
    test_fraction : float
        Fraction of X held out for computing test error.
    random_state : int or None
    """

    def __init__(self, k_min=None, k_max=None, n_codebooks: int = 15,
                 test_fraction: float = 0.5, random_state=None):
        self.k_min = k_min
        self.k_max = k_max
        self.n_codebooks = n_codebooks
        self.test_fraction = test_fraction
        self.random_state = random_state

    # ── Quantization error ────────────────────────────────────────────────────

    @staticmethod
    def _rms_error(X_test: np.ndarray, centroids: np.ndarray) -> float:
        """RMS nearest-centroid quantization error (r = 2 distortion^{1/2})."""
        nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(centroids)
        dists, _ = nn.kneighbors(X_test)      # (n_test, 1) — L2 distances
        return float(np.sqrt((dists[:, 0] ** 2).mean()))

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n = len(X)
        rng = np.random.default_rng(self.random_state)

        # ── Step 1: train / test split ────────────────────────────────────────
        idx     = rng.permutation(n)
        n_test  = max(1, int(n * self.test_fraction))
        n_train = n - n_test
        X_train = X[idx[:n_train]]
        X_test  = X[idx[n_train:]]

        # ── Step 2: geometric grid of codebook sizes ──────────────────────────
        k_min = self.k_min if self.k_min is not None else max(2, n_train // 50)
        k_max = self.k_max if self.k_max is not None else max(k_min + 1, n_train // 4)
        k_max = min(k_max, n_train - 1)
        k_min = min(k_min, k_max - 1)

        ks = np.unique(
            np.round(np.geomspace(k_min, k_max, self.n_codebooks)).astype(int)
        )

        # ── Step 3: k-means + test error for each k ───────────────────────────
        log_k: list[float]     = []
        neg_log_e: list[float] = []
        seed = int(rng.integers(0, 2 ** 31))
        for k in ks:
            km = KMeans(n_clusters=int(k), n_init=3, random_state=seed)
            km.fit(X_train)
            e = self._rms_error(X_test, km.cluster_centers_)
            if e > 0:
                log_k.append(np.log(float(k)))
                neg_log_e.append(-np.log(e))

        if len(log_k) < 2:
            raise ValueError("Too few valid codebook sizes for regression.")

        # ── Step 4: OLS slope → d = 1 / slope ────────────────────────────────
        # Model: −log e = (1/d) · log k + const
        lk  = np.array(log_k).reshape(-1, 1)
        nle = np.array(neg_log_e)
        lr  = LinearRegression(fit_intercept=True).fit(lk, nle)
        slope = float(lr.coef_[0])

        self.dimension_  = 1.0 / slope if slope > 0 else np.nan
        self.log_k_      = np.array(log_k)
        self.neg_log_e_  = np.array(neg_log_e)
        self.slope_      = slope
        return self
