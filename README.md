---
title: ID Estimation Dashboard
emoji: 📐
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: apache-2.0
short_description: Experiment Dashboard for Intrinsic Dimension Estimators
---

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

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Running Experiments

### Step 1 — Generate datasets (required before any experiment)

All experiments load data from `data/`. Run this once to generate clean datasets for all families and dimensions (`DIMS_INTRINSIC`, `N_TOTAL` points each, defined in `datasets.py`):

```bash
python datasets.py
```

To also generate noisy variants (needed for exp1 with non-zero η, or exp2 with `--eta`):

```bash
python datasets.py --eta 0 0.1 0.5
```

To generate only a subset of dimensions (faster for quick tests):

```bash
python datasets.py --dims 2 5 10 --eta 0 0.1
```

> **Note:** If a dataset file already exists for a given `(family, d, η)` combination it will be overwritten. Use `--no-generate` on any experiment script to skip this step and reuse existing files.

---

### Step 2 — Run an experiment

All experiment scripts are in `experiments/` and must be run from the **project root**:

```bash
python experiments/expN_name.py [options]
```

Every script generates its own datasets by default unless you pass `--no-generate`.

#### Shared flags (all experiments)

| Flag | Description |
|------|-------------|
| `--dims D [D ...]` | Intrinsic dimensions to run (default: `DIMS_INTRINSIC` from `datasets.py`) |
| `--families F [F ...]` | Dataset families: `hypercube`, `gaussian`, `linear` (default: `FAMILIES`) |
| `--estimators E [E ...]` | Estimators: `CorrInt`, `MLE`, `TwoNN`, `DANCo`, `ESS`, `TLE` (default: all) |
| `--no-generate` | Skip dataset generation; use existing files in `data/` |

Outputs are saved to `results/{exp_name}/` as PDFs.

---

### exp0 — Sanity check

Verifies all estimators work correctly. Fits each on `[0,1]^d` for `d ∈ DIMS` and prints a results table. Settings are defined as `DIMS`, `N_SAMPLES`, and `N_NEIGHBORS` at the top of `experiments/exp0_check.py`.

```bash
python experiments/exp0_check.py
```

---

### exp1 — Noise sensitivity

Sweeps noise levels `η ∈ ETA_VALUES` (defined in `experiments/exp1_noise.py`).
For each `(family, d, η)`, runs `N_RUNS` independent subsamples of `N_SAMPLES` points and records the ID estimate and wall-clock runtime.

```bash
# Full run (slow — all families, dims, estimators)
python experiments/exp1_noise.py

# Recommended targeted run
python experiments/exp1_noise.py --families hypercube --estimators MLE TwoNN CorrInt --dims 2 5 10

# Subset of noise levels
python experiments/exp1_noise.py --eta 0 0.1 0.5 1.0

# Skip regenerating datasets
python experiments/exp1_noise.py --no-generate
```

**Outputs:** `results/exp1_noise/{family}_d{d}.pdf` (ID vs η), `results/exp1_noise/runtime_{family}.pdf` (runtime vs d)

---

### exp2 — Sample size scaling

Sweeps sample sizes `n ∈ SAMPLE_SIZES` at a fixed noise level `ETA` (both defined in `experiments/exp2_sample.py`).
For each `(family, d, n)`, runs `N_RUNS` independent subsamples and records the ID estimate and runtime.

```bash
# Full run
python experiments/exp2_sample.py

# Targeted run (subset of estimators/dims)
python experiments/exp2_sample.py --estimators MLE TwoNN CorrInt TLE --dims 2 5 10

# Run on noisy data (datasets must already exist for that η)
python experiments/exp2_sample.py --eta 0.1 --no-generate
```

**Outputs:** `results/exp2_sample/{family}_d{d}.pdf` (ID vs n), `results/exp2_sample/runtime_{family}.pdf` (runtime vs n)

---

### exp3 — Johnson–Lindenstrauss projection

Projects data to `k ∈ JL_DIMS` dimensions via a scaled Gaussian JL map (defined in `experiments/exp3_jl.py`), then estimates ID on the projected data. Compares `id_jl` against `id_original` as a function of `k`. Each run uses an independent subsample of `N_SAMPLES` points and an independent projection matrix (`N_RUNS` total).

The linear family (ambient `AMBIENT_DIM`) is most informative: `id_jl` should recover `id_original` once `k` exceeds the intrinsic dimension `d`.

```bash
# Full run
python experiments/exp3_jl.py

# Most informative targeted run
python experiments/exp3_jl.py --families linear --estimators MLE TwoNN CorrInt --dims 2 5 10

# Custom noise level or sample size
python experiments/exp3_jl.py --eta 0.1 --n-samples 2000 --no-generate
```

**Outputs:** `results/exp3_jl/{family}_d{d}.pdf` (ID vs k), `results/exp3_jl/runtime_{family}.pdf` (runtime vs k)

---

## Datasets

Three families, each with `N_TOTAL` points (defined in `datasets.py`):

| Family | Description | Ambient dim |
|--------|-------------|-------------|
| `hypercube` | Uniform on `[0,1]^d` | `d` |
| `gaussian` | Isotropic `N(0, I_d)` | `d` |
| `linear` | `d`-dim linear subspace embedded via random orthonormal basis | `AMBIENT_DIM` |

Intrinsic dimensions: `d ∈ DIMS_INTRINSIC` (defined in `datasets.py`).

Files are stored as `data/{family}_d{d}[_D{D}]_eta{eta:.3f}.npz` and contain arrays `X`, `d`, `D`, `eta`, `avg_dist`, `seed`.

To load a dataset directly in Python:

```python
from datasets import load_dataset
ds = load_dataset("linear", d=10, eta=0.0, n_samples=2000, seed=42)
X  = ds["X"]   # shape (n_samples, AMBIENT_DIM)
```
