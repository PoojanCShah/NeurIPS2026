"""
datasets.py
-----------
Generate, store, and load the synthetic benchmark datasets used in experiments.

Dataset families
----------------
1. Hypercube   — uniform on [0,1]^d,  d in DIMS_INTRINSIC
2. Gaussian    — isotropic N(0,I_d),  d in DIMS_INTRINSIC
3. Linear      — d-dim linear subspace embedded in D=1000 ambient dims,
                 d in DIMS_INTRINSIC

Noise model
-----------
After generation, optional i.i.d. Gaussian noise with std = eta * avg_dist is
added pointwise, where avg_dist is the mean pairwise distance of the clean data
estimated from a subsample of 500 points.

Each dataset is 25 000 points by default.  A subsampled version can be obtained
with `load_dataset(..., n_samples=k)`.

Storage layout
--------------
data/
  hypercube_d{d}_eta{eta:.3f}.npz
  gaussian_d{d}_eta{eta:.3f}.npz
  linear_d{d}_D{D}_eta{eta:.3f}.npz

Each .npz contains:
  X        : (N, ambient_dim) float32 array
  d        : true intrinsic dimension (scalar)
  D        : ambient dimension (scalar)
  eta      : noise level (scalar)
  avg_dist : estimated average pairwise distance of the clean data (scalar)
  seed     : global seed used (scalar)

CLI
---
  python datasets.py              # generate all datasets with eta=0
  python datasets.py --eta 0.1   # also generate noisy versions
  python datasets.py --help
"""

import argparse
import os
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────

DIMS_INTRINSIC = [2, 3, 5, 10, 20, 50, 75, 100, 250, 500, 1000]
AMBIENT_DIM    = 1000   # fixed ambient dimension for the linear-subspace family
N_TOTAL        = 25_000
SEED           = 42
DATA_DIR       = os.path.join(os.path.dirname(__file__), "data")

# Number of points used to estimate avg pairwise distance (kept small for speed)
_AVG_DIST_SUBSAMPLE = 500


# ── Internal helpers ──────────────────────────────────────────────────────────

def _estimate_avg_dist(X: np.ndarray, rng: np.random.Generator) -> float:
    """Estimate mean pairwise distance from a random subsample of X."""
    n = len(X)
    m = min(_AVG_DIST_SUBSAMPLE, n)
    idx = rng.choice(n, m, replace=False)
    sub = X[idx].astype(np.float64)
    # pairwise distances via broadcasting (m is small so memory is fine)
    diff = sub[:, None, :] - sub[None, :, :]          # (m, m, d)
    dists = np.sqrt((diff ** 2).sum(axis=-1))          # (m, m)
    # mean of upper triangle (exclude diagonal)
    i_upper = np.triu_indices(m, k=1)
    return float(dists[i_upper].mean())


def _add_noise(X: np.ndarray, eta: float, avg_dist: float,
               rng: np.random.Generator) -> np.ndarray:
    """Add i.i.d. Gaussian noise with std = eta * avg_dist."""
    if eta == 0.0:
        return X
    noise = rng.standard_normal(X.shape).astype(X.dtype)
    return X + (eta * avg_dist * noise)


def _save(path: str, X: np.ndarray, d: int, D: int, eta: float,
          avg_dist: float, seed: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        X=X,
        d=np.int32(d),
        D=np.int32(D),
        eta=np.float32(eta),
        avg_dist=np.float32(avg_dist),
        seed=np.int32(seed),
    )
    print(f"  saved  {os.path.relpath(path)}")


# ── Generators ────────────────────────────────────────────────────────────────

def make_hypercube(d: int, n: int, eta: float, seed: int) -> dict:
    """Uniform distribution on [0,1]^d."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, d), dtype=np.float32)  # uniform [0,1)
    avg_dist = _estimate_avg_dist(X, rng)
    X_noisy = _add_noise(X, eta, avg_dist, rng)
    return dict(X=X_noisy, d=d, D=d, eta=eta, avg_dist=avg_dist, seed=seed)


def make_gaussian(d: int, n: int, eta: float, seed: int) -> dict:
    """Isotropic Gaussian N(0, I_d)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    avg_dist = _estimate_avg_dist(X, rng)
    X_noisy = _add_noise(X, eta, avg_dist, rng)
    return dict(X=X_noisy, d=d, D=d, eta=eta, avg_dist=avg_dist, seed=seed)


def make_linear(d: int, D: int, n: int, eta: float, seed: int) -> dict:
    """d-dim linear subspace embedded in R^D via a random orthonormal basis."""
    rng = np.random.default_rng(seed)

    # Random orthonormal basis: draw D×d Gaussian matrix, then QR-decompose
    G = rng.standard_normal((D, d))
    basis, _ = np.linalg.qr(G)          # shape (D, d), columns orthonormal
    basis = basis.astype(np.float32)

    # Latent coords ~ N(0, I_d), then embed
    Z = rng.standard_normal((n, d)).astype(np.float32)
    X = Z @ basis.T                     # shape (n, D)

    avg_dist = _estimate_avg_dist(X, rng)
    X_noisy = _add_noise(X, eta, avg_dist, rng)
    return dict(X=X_noisy, d=d, D=D, eta=eta, avg_dist=avg_dist, seed=seed)


# ── Public API ────────────────────────────────────────────────────────────────

def dataset_path(family: str, d: int, eta: float, D: int = None) -> str:
    """Return the canonical .npz path for a dataset (no I/O performed)."""
    eta_str = f"{eta:.3f}"
    if family == "hypercube":
        fname = f"hypercube_d{d}_eta{eta_str}.npz"
    elif family == "gaussian":
        fname = f"gaussian_d{d}_eta{eta_str}.npz"
    elif family == "linear":
        if D is None:
            D = AMBIENT_DIM
        fname = f"linear_d{d}_D{D}_eta{eta_str}.npz"
    else:
        raise ValueError(f"Unknown family '{family}'")
    return os.path.join(DATA_DIR, fname)


def load_dataset(family: str, d: int, eta: float = 0.0,
                 D: int = None, n_samples: int = None,
                 seed: int = SEED) -> dict:
    """Load a previously generated dataset.

    Parameters
    ----------
    family    : 'hypercube' | 'gaussian' | 'linear'
    d         : intrinsic dimension
    eta       : noise level (must match a generated file)
    D         : ambient dimension (only relevant for 'linear')
    n_samples : if given, return a random subsample of this size
    seed      : RNG seed used for subsampling (does not affect stored data)

    Returns
    -------
    dict with keys: X, d, D, eta, avg_dist, seed
    """
    path = dataset_path(family, d, eta, D)
    data = np.load(path)
    result = {k: data[k] for k in data.files}

    if n_samples is not None and n_samples < len(result["X"]):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(result["X"]), n_samples, replace=False)
        result["X"] = result["X"][idx]

    return result


def generate_all(eta_values: list[float], n: int = N_TOTAL,
                 seed: int = SEED, dims: list[int] = None,
                 D: int = AMBIENT_DIM,
                 ambient_factor: int = None) -> None:
    """Generate and save all datasets for every (family, d, eta) combination.

    Parameters
    ----------
    eta_values     : list of noise levels to generate (0.0 = clean)
    n              : number of points per dataset
    seed           : global RNG seed
    dims           : intrinsic dimensions to generate; defaults to DIMS_INTRINSIC
    D              : fixed ambient dimension for the linear family
    ambient_factor : if set, use D = ambient_factor * d per dimension instead
                     of the fixed D (overrides D for the linear family)
    """
    if dims is None:
        dims = DIMS_INTRINSIC

    for eta in eta_values:
        print(f"\n=== eta = {eta:.3f} ===")

        print("\n-- Hypercube --")
        for d in dims:
            data = make_hypercube(d=d, n=n, eta=eta, seed=seed)
            _save(dataset_path("hypercube", d, eta), **data)

        print("\n-- Gaussian --")
        for d in dims:
            data = make_gaussian(d=d, n=n, eta=eta, seed=seed)
            _save(dataset_path("gaussian", d, eta), **data)

        D_label = f"factor×d" if ambient_factor else str(D)
        print(f"\n-- Linear (D={D_label}) --")
        for d in dims:
            D_d = ambient_factor * d if ambient_factor else D
            if d > D_d:
                print(f"  skip  d={d} > D={D_d}")
                continue
            data = make_linear(d=d, D=D_d, n=n, eta=eta, seed=seed)
            _save(dataset_path("linear", d, eta, D_d), **data)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic benchmark datasets."
    )
    parser.add_argument(
        "--eta", type=float, nargs="+", default=[0.0],
        metavar="ETA",
        help="Noise level(s) eta (default: 0.0). Can pass multiple, e.g. --eta 0 0.1 0.5",
    )
    parser.add_argument(
        "--n", type=int, default=N_TOTAL,
        help=f"Points per dataset (default: {N_TOTAL})",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help=f"Global RNG seed (default: {SEED})",
    )
    parser.add_argument(
        "--dims", type=int, nargs="+", default=None,
        metavar="D",
        help="Intrinsic dimensions to generate (default: all in DIMS_INTRINSIC)",
    )
    parser.add_argument(
        "--ambient", type=int, default=AMBIENT_DIM,
        help=f"Ambient dimension for the linear family (default: {AMBIENT_DIM})",
    )
    args = parser.parse_args()

    generate_all(
        eta_values=args.eta,
        n=args.n,
        seed=args.seed,
        dims=args.dims,
        D=args.ambient,
    )
