# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

NeurIPS 2026 research project on selection of distance-based estimators for intrinsic dimensionality (ID) estimation. Six estimators are implemented from scratch in the `id/` package, each wrapping the scikit-learn API pattern (`.fit(X)` sets `.dimension_`).

## Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running Experiments

```bash
# Quick smoke test — all six estimators on a 2D manifold in 10D ambient space
python estimators.py

# exp0: sanity check on hypercube [0,1]^d for d ∈ {2,3,5,7,10}
python experiments/exp0_check.py
```

## Code Architecture

### `id/` package — estimator implementations

Each estimator lives in its own module and follows the same interface:

```python
est = EstimatorClass(**params)
est.fit(X)          # X: (n_samples, n_features) ndarray
est.dimension_      # float — estimated intrinsic dimension
```

| Module | Estimator | Key parameters |
|---|---|---|
| `corrint.py` | CorrInt | — |
| `mle.py` | MLE | `n_neighbors`, `comb` (`'mle'`/`'mean'`/`'median'`), `unbiased` |
| `twonn.py` | TwoNN | — |
| `danco.py` | DANCo | `random_state` |
| `ess.py` | ESS | `n_neighbors`, `random_state` |
| `tle.py` | TLE | `n_neighbors` |

`_utils.py` provides shared helpers used across all estimators:
- `knn(X, k)` — k-NN distances/indices via sklearn (self excluded)
- `lens(vectors)` — row-wise L2 norms
- `binom_coeff(n, k)`, `indnComb(n, k)`, `efficient_indnComb(n, k, rng)` — combinatorics for DANCo/ESS

### `estimators.py` — entry point

`run_all(X, n_neighbors=20, random_state=None) -> dict[str, float]` fits all six estimators and returns a name→dimension mapping. The `__main__` block demonstrates usage on a synthetic low-rank manifold.

### `experiments/` — reproducible experiment scripts

Each script does `sys.path.insert(0, "..")` to import from `id/` without installation. New experiments should follow the same pattern as `exp0_check.py`: define settings as module-level constants, use `tqdm` for progress, and print results as a formatted table.
