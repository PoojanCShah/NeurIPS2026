# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

NeurIPS 2026 research project on selection of distance-based estimators for intrinsic dimensionality (ID) estimation. Eight estimators are implemented from scratch in the `id/` package, each wrapping the scikit-learn API pattern (`.fit(X)` sets `.dimension_`).

## Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the Dashboard

```bash
python app.py
python app.py --port 8051 --debug
```

## Running Experiments

```bash
# Quick smoke test — all estimators on a 2D manifold in 10D ambient space
python estimators.py

# Generate datasets before running experiments (required first time)
python datasets.py                          # clean only (eta=0)
python datasets.py --eta 0 0.1 0.5         # also noisy variants

# exp0: sanity check on hypercube [0,1]^d for d ∈ {2,3,5,7,10}
python experiments/exp0_check.py

# exp1: noise sensitivity — sweeps η, records ID estimate + runtime per estimator
python experiments/exp1_noise.py
python experiments/exp1_noise.py --families hypercube --estimators MLE TwoNN CorrInt --dims 2 5 10 --eta 0 0.1 0.5 1.0

# exp2: sample size scaling — sweeps n, fixed noise level
python experiments/exp2_sample.py
python experiments/exp2_sample.py --eta 0.1 --no-generate

# exp3: Johnson–Lindenstrauss projection — compares id_jl vs id_original as k varies
python experiments/exp3_jl.py
python experiments/exp3_jl.py --families linear --estimators MLE TwoNN --dims 2 5 10 --n-samples 2000
```

All experiment scripts share these flags: `--dims`, `--families`, `--estimators`, `--no-generate` (skip dataset generation). Outputs go to `results/{exp_name}/` as PDFs.

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
| `packing.py` | PackingDim | `k1`, `k2`, `epsilon`, `random_state` |
| `quantdim.py` | QuantDim | `k_min`, `k_max`, `n_codebooks`, `random_state` |

`_utils.py` provides shared helpers used across all estimators:
- `knn(X, k)` — k-NN distances/indices via sklearn (self excluded)
- `lens(vectors)` — row-wise L2 norms
- `binom_coeff(n, k)`, `indnComb(n, k)`, `efficient_indnComb(n, k, rng)` — combinatorics for DANCo/ESS

### `datasets.py` — dataset generation and loading

Three synthetic families: `hypercube` (uniform on `[0,1]^d`), `gaussian` (isotropic `N(0,I_d)`), `linear` (d-dim subspace in `R^1000` via random orthonormal basis). Each stored as `data/{family}_d{d}[_D{D}]_eta{eta:.3f}.npz` with arrays `X, d, D, eta, avg_dist, seed`. Noise model: `std = η × avg_dist`.

Key constants: `DIMS_INTRINSIC = [2,3,5,10,20,50,75,100,250,500,1000]`, `AMBIENT_DIM = 1000`, `N_TOTAL = 25_000`.

`load_dataset(family, d, eta, D, n_samples, seed)` handles subsampling at load time without modifying stored files.

### `estimators.py` — entry point

`run_all(X, n_neighbors=20, random_state=None) -> dict[str, float]` fits all estimators and returns a name→dimension mapping. The `__main__` block demonstrates usage on a synthetic low-rank manifold.

### `experiments/` — reproducible experiment scripts

Each script does `sys.path.insert(0, "..")` to import from `id/` without installation. New experiments should follow the same pattern as `exp0_check.py`: define settings as module-level constants, use `tqdm` for progress, and print results as a formatted table.

### Dashboard (`app.py`, `layout.py`, `components/`, `tabs/`, `registry/`)

A Dash/Bootstrap app for interactive exploration of experiment results.

- `registry/estimators.py` — single source of truth for `ESTIMATOR_CLASSES` and `PARAM_DEFS` (hyperparameter descriptors). Add new estimators here to automatically expose them in the sidebar and experiment runners.
- `registry/datasets.py` — analogous registry for dataset families and their parameters.
- `components/sidebar.py` — global controls (estimator checkboxes, hyperparameter sliders); `components/dataset_panel.py` — dataset selection UI.
- `tabs/exp1.py`, `tabs/exp2.py`, `tabs/exp3.py` — tab layouts and Dash callbacks for each experiment; import from `registry/` to stay in sync with the estimator list.
- `layout.py` — assembles components and tabs into the top-level `app.layout`.

To add a new estimator to the dashboard: implement it in `id/`, export it from `id/__init__.py`, then register it in `registry/estimators.py` (class + param descriptors). No other files need editing.
