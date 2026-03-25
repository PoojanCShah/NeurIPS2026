"""
Entry point. Imports all estimators from the `id` package and exposes run_all().
"""

import numpy as np
from id import CorrInt, MLE, TwoNN, DANCo, ESS, TLE, PackingDim, QuantDim


def run_all(X, n_neighbors: int = 20, random_state=None) -> dict:
    """Fit all eight estimators on X and return estimated intrinsic dimensions.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    n_neighbors : int
        Neighbourhood size for MLE, ESS, TLE.
    random_state : int or None
        Seed for DANCo, ESS, PackingDim, and QuantDim.

    Returns
    -------
    dict[str, float]
    """
    return {
        "corrint":    CorrInt().fit(X).dimension_,
        "mle":        MLE(n_neighbors=n_neighbors).fit(X).dimension_,
        "twonn":      TwoNN().fit(X).dimension_,
        "danco":      DANCo(random_state=random_state).fit(X).dimension_,
        "ess":        ESS(n_neighbors=n_neighbors, random_state=random_state).fit(X).dimension_,
        "tle":        TLE(n_neighbors=n_neighbors).fit(X).dimension_,
        "packingdim": PackingDim(random_state=random_state).fit(X).dimension_,
        "quantdim":   QuantDim(random_state=random_state).fit(X).dimension_,
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # 500 points on a 2D manifold embedded in 10D
    true_id, n, ambient = 2, 500, 10
    t = rng.uniform(0, 1, (n, true_id))
    X = t @ rng.standard_normal((true_id, ambient))

    print(f"Dataset: {n} points, ambient dim = {ambient}, true ID = {true_id}\n")
    for name, dim in run_all(X, n_neighbors=20, random_state=0).items():
        print(f"  {name:<8s}  ID = {dim:.2f}")
