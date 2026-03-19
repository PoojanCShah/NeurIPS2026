# Code for NeurIPS 2026 Estimation

### 1. Selection of Distance Based Estimators

| Name | Citation |
|------|----------|
| **CorrInt** | Grassberger & Procaccia, *Phys. Rev. Lett.* 1983 |
| **MLE** | Levina & Bickel, *NeurIPS* 2005 |
| **TwoNN** | Facco et al., *Scientific Reports* 2017 |
| **DANCo** | Ceruti et al., *Pattern Recognition* 2012 |
| **ESS** | Johnsson et al., *Statistics and Computing* 2015 |
| **TLE** | Amsaleg et al., *Statistical Analysis and Data Mining* 2019 |

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

## Datasets

Synthetic benchmark datasets are generated and stored under `data/` by `datasets.py`.  Three families are supported, each at 25 000 points:

| Family | Description | Ambient dim |
|--------|-------------|-------------|
| `hypercube` | Uniform on `[0,1]^d` | `d` |
| `gaussian` | Isotropic `N(0, I_d)` | `d` |
| `linear` | `d`-dim linear subspace embedded via random orthonormal basis | 1000 |

Intrinsic dimensions: `d ∈ {2, 3, 5, 10, 20, 50, 75, 100, 250, 500, 1000}`.

A Gaussian noise model is available: pointwise perturbation with std `η × avg_dist`, where `avg_dist` is the mean pairwise distance of the clean data.  Datasets are generated for each `(family, d, η)` combination and stored as `.npz` files (arrays `X`, `d`, `D`, `eta`, `avg_dist`, `seed`).

```bash
# generate all datasets, clean only
python datasets.py

# also generate noisy variants
python datasets.py --eta 0 0.1 0.5

# subset of dims
python datasets.py --dims 2 5 10 --eta 0 0.1
```

Subsampling is handled at load time and does not modify stored files:

```python
from datasets import load_dataset
ds = load_dataset("linear", d=10, eta=0.0, n_samples=2000, seed=42)
X  = ds["X"]   # (2000, 1000)
```

## Experiments

All experiments support `--families`, `--estimators`, `--dims`, and `--no-generate` flags for targeted runs.  Outputs go to `results/{exp_name}/`.

---

### exp0 — Sanity check

Estimates the ID of a uniform distribution on `[0,1]^d` for `d ∈ {2, 3, 5, 7, 10}` using all six estimators.  Verifies each estimator is functioning correctly.

```bash
python experiments/exp0_check.py
```

Results (`N=2000`, `n_neighbors=20`):

```
true d     CorrInt       MLE     TwoNN     DANCo       ESS       TLE
----------------------------------------------------------------------
     2        1.98      1.97      1.99      2.00      2.03      2.12
     3        2.90      2.89      2.92      2.69      2.99      3.16
     5        4.53      4.48      4.51      4.98      4.76      4.94
     7        5.90      6.00      6.48      7.00      6.52      6.63
    10        7.96      8.10      8.91     10.00      9.19      8.84
```

---

### exp1 — Noise sensitivity

Sweeps noise levels `η ∈ {0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0}` and records each estimator's ID estimate.  Repeated over `N_RUNS=5` independent subsamples; plots show mean ± 95 % CI with individual runs in the background.  A runtime-vs-d figure is also produced per family.

```bash
# full run
python experiments/exp1_noise.py

# targeted run (recommended for initial results)
python experiments/exp1_noise.py --families hypercube --estimators MLE TwoNN CorrInt --dims 2 5 10 25 50 --n-samples 5000
```

Outputs: `results/exp1_noise/{family}_d{d}.pdf`, `results/exp1_noise/runtime_{family}.pdf`

---





### exp2 — Sample size scaling

Sweeps sample sizes `n ∈ {50, 100, 250, 500, 750, 1k, 2.5k, 5k}` at a fixed noise level and records each estimator's ID estimate and runtime.  Reveals how much data each estimator needs to converge toward the true ID.

```bash
# full run
python experiments/exp2_sample.py

# targeted run
python experiments/exp2_sample.py --families hypercube --estimators MLE TwoNN CorrInt --dims 2 5 10 25 50

# run on noisy data (eta must have been generated first)
python experiments/exp2_sample.py --eta 0.1 --no-generate
```

Outputs: `results/exp2_sample/{family}_d{d}.pdf`, `results/exp2_sample/runtime_{family}.pdf`

---





### exp3 — Johnson–Lindenstrauss projection

Projects data to `k ∈ {2, 5, 10, 20, 50, 100, 200, 500, 750, 1000}` dimensions via a scaled Gaussian JL map and compares `id_jl` against `id_original` as a function of `k`.  Each run uses an independent subsample and an independent projection matrix.  The linear family (ambient D=1000) is the most informative: `id_jl` should recover `id_original` once `k` crosses the intrinsic dimension `d`.

```bash
# full run
python experiments/exp3_jl.py

# most informative targeted run
python experiments/exp3_jl.py --families linear --estimators MLE TwoNN CorrInt --dims 2 5 10 25 50 --n-samples 2000
```

Outputs: `results/exp3_jl/{family}_d{d}.pdf`, `results/exp3_jl/runtime_{family}.pdf`


