"""
exp0_check.py
-------------
Sanity check: estimate the intrinsic dimension of a d-dimensional uniform
distribution on the hypercube [0,1]^d for d in {2, 3, 5, 7, 10}.

Each estimator should return a value close to the true d.
Results are printed as a table (true d vs. each estimator's estimate).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from tqdm import tqdm
from id import CorrInt, MLE, TwoNN, DANCo, ESS, TLE

# ── Experiment settings ───────────────────────────────────────────────────────

DIMS        = [2, 3, 5, 7, 10]
N_SAMPLES   = 2000
N_NEIGHBORS = 20
RANDOM_STATE = 42

ESTIMATORS = {
    "CorrInt": CorrInt(),
    "MLE":     MLE(n_neighbors=N_NEIGHBORS),
    "TwoNN":   TwoNN(),
    "DANCo":   DANCo(random_state=RANDOM_STATE),
    "ESS":     ESS(n_neighbors=N_NEIGHBORS, random_state=RANDOM_STATE),
    "TLE":     TLE(n_neighbors=N_NEIGHBORS),
}

# ── Run ───────────────────────────────────────────────────────────────────────

rng = np.random.default_rng(RANDOM_STATE)

col_w = 10
header = f"{'true d':>8}  " + "".join(f"{name:>{col_w}}" for name in ESTIMATORS)
print(header)
print("-" * len(header))

for d in tqdm(DIMS, desc="dimensions"):
    X = rng.uniform(0, 1, size=(N_SAMPLES, d))

    estimates = {}
    for name, est in tqdm(ESTIMATORS.items(), desc=f"  d={d}", leave=False):
        estimates[name] = est.fit(X).dimension_

    row = f"{d:>8}  " + "".join(f"{estimates[name]:>{col_w}.2f}" for name in ESTIMATORS)
    print(row)
