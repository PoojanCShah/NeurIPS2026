"""
Correlation Dimension estimator (CorrInt).

Reference
---------
Grassberger & Procaccia (1983). Characterization of strange attractors.
Physical Review Letters, 50(5), 346.
"""

import numpy as np
from sklearn.metrics import pairwise_distances_chunked
from ._utils import knn


class CorrInt:
    """Intrinsic dimension via the Correlation Dimension (Grassberger & Procaccia, 1983).

    Algorithm
    ---------
    1. Use the median k1-th and k2-th NN distances as reference radii r1, r2.
    2. Compute correlation integrals C(r1), C(r2): fraction of all point pairs
       whose distance is < r.
    3. Estimate d from the log-log slope:

           d ≈ [log C(r2) − log C(r1)] / log(r2 / r1)

    Parameters
    ----------
    k1 : int
        Index of the first reference neighbour (sets radius r1).
    k2 : int
        Index of the second reference neighbour (sets radius r2).
    """

    def __init__(self, k1: int = 10, k2: int = 20):
        self.k1 = k1
        self.k2 = k2

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n = len(X)

        k1, k2 = self.k1, self.k2
        if k2 >= n:
            k2 = n - 1
        if k1 >= k2:
            k1 = k2 - 1

        # ── Step 1: reference radii from median kNN distances ─────────────────
        dists, _ = knn(X, k2)
        r1 = np.median(dists[:, k1 - 1])
        r2 = np.median(dists[:, k2 - 1])

        # ── Step 2: correlation integrals (streamed to avoid O(n²) memory) ────
        n_pairs = n ** 2
        s1 = -n   # subtract n diagonal zeros (self-distances)
        s2 = -n
        for chunk in pairwise_distances_chunked(X):
            s1 += (chunk < r1).sum()
            s2 += (chunk < r2).sum()

        C1 = s1 / n_pairs
        C2 = s2 / n_pairs

        # ── Step 3: log-log slope ─────────────────────────────────────────────
        self.dimension_ = (np.log(C2) - np.log(C1)) / np.log(r2 / r1)
        self.r1_ = r1
        self.r2_ = r2
        self.C1_ = C1
        self.C2_ = C2
        return self
