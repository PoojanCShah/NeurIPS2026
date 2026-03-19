"""Shared utilities used across estimator modules."""

import itertools
import numpy as np
from sklearn.neighbors import NearestNeighbors


def knn(X, k):
    """Return (distances, indices) of k nearest neighbors, excluding self."""
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    return nn.kneighbors()   # no argument → training points, self excluded


def lens(vectors):
    """Row-wise L2 norms of a 2-D array."""
    return np.sqrt(np.sum(vectors ** 2, axis=1))


def binom_coeff(n, k):
    """C(n, k) computed iteratively (no floating-point conversion)."""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    total = 1
    for i in range(min(k, n - k)):
        total = total * (n - i) // (i + 1)
    return total


def indnComb(n, k):
    """All C(n, k) combinations as an (ncomb, k) integer array."""
    return np.array(list(itertools.combinations(range(n), k)))


def efficient_indnComb(n, k, rng):
    """Sample up to 5000 combinations uniformly from C(n, k)."""
    ncomb = binom_coeff(n, k)
    pop = itertools.combinations(range(n), k)
    targets = set(rng.choice(ncomb, min(ncomb, 5000), replace=False))
    return np.array(
        list(itertools.compress(pop, map(targets.__contains__, itertools.count())))
    )
