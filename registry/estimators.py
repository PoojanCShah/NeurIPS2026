"""
Estimator registry.

To add a new estimator:
  1. Import its class below.
  2. Add it to ESTIMATOR_CLASSES.
  3. Add its hyperparameter descriptors to PARAM_DEFS.

Everything else (sidebar controls, stores, experiment runners) derives
from these two dicts automatically.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from id import CorrInt, MLE, TwoNN, DANCo, ESS, TLE, PackingDim, QuantDim

# ── Ordered display list ───────────────────────────────────────────────────────

ESTIMATOR_NAMES = [
    "CorrInt", "MLE", "TwoNN", "DANCo", "ESS", "TLE", "PackingDim", "QuantDim",
]

# ── Class map ──────────────────────────────────────────────────────────────────

ESTIMATOR_CLASSES = {
    "CorrInt":    CorrInt,
    "MLE":        MLE,
    "TwoNN":      TwoNN,
    "DANCo":      DANCo,
    "ESS":        ESS,
    "TLE":        TLE,
    "PackingDim": PackingDim,
    "QuantDim":   QuantDim,
}

# ── Hyperparameter descriptors ─────────────────────────────────────────────────
# Each descriptor: {id, label, type, default, [options], [min], [max], [step]}
# type: "int" | "float" | "select" | "bool"

PARAM_DEFS = {
    "CorrInt": [
        {"id": "k1", "label": "k1", "type": "int", "default": 10, "min": 1, "max": 200},
        {"id": "k2", "label": "k2", "type": "int", "default": 20, "min": 2, "max": 500},
    ],
    "MLE": [
        {"id": "n_neighbors", "label": "n_neighbors", "type": "int",    "default": 20,    "min": 2, "max": 200},
        {"id": "comb",        "label": "comb",        "type": "select", "default": "mle", "options": ["mle", "mean", "median"]},
        {"id": "unbiased",    "label": "unbiased",    "type": "bool",   "default": False},
    ],
    "TwoNN": [
        {"id": "discard_fraction", "label": "discard_fraction", "type": "float", "default": 0.1, "min": 0.0, "max": 0.5, "step": 0.01},
    ],
    "DANCo": [
        {"id": "random_state", "label": "random_state", "type": "int", "default": 42},
    ],
    "ESS": [
        {"id": "n_neighbors",  "label": "n_neighbors",  "type": "int", "default": 20, "min": 2, "max": 200},
        {"id": "random_state", "label": "random_state", "type": "int", "default": 42},
    ],
    "TLE": [
        {"id": "n_neighbors", "label": "n_neighbors", "type": "int",   "default": 20,   "min": 2, "max": 200},
        {"id": "epsilon",     "label": "epsilon",     "type": "float", "default": 1e-4, "min": 0, "max": 1, "step": 1e-5},
    ],
    "PackingDim": [
        {"id": "k1",           "label": "k1",           "type": "int",   "default": 10,   "min": 1,  "max": 200},
        {"id": "k2",           "label": "k2",           "type": "int",   "default": 20,   "min": 2,  "max": 500},
        {"id": "epsilon",      "label": "epsilon",      "type": "float", "default": 0.01, "min": 0,  "max": 0.5, "step": 0.001},
        {"id": "max_iter",     "label": "max_iter",     "type": "int",   "default": 1000, "min": 10, "max": 5000},
        {"id": "random_state", "label": "random_state", "type": "int",   "default": 42},
    ],
    "QuantDim": [
        {"id": "k_min",         "label": "k_min (None=auto)",  "type": "int",   "default": None},
        {"id": "k_max",         "label": "k_max (None=auto)",  "type": "int",   "default": None},
        {"id": "n_codebooks",   "label": "n_codebooks",        "type": "int",   "default": 15,  "min": 3,  "max": 50},
        {"id": "test_fraction", "label": "test_fraction",      "type": "float", "default": 0.5, "min": 0.1, "max": 0.9, "step": 0.05},
        {"id": "random_state",  "label": "random_state",       "type": "int",   "default": 42},
    ],
}


def build_estimator(name: str, params: dict):
    """Instantiate an estimator by name with params from the Dash store.

    Handles JSON → Python type coercion:
      - bool params stored as 0/1 → Python bool
      - int/float params coerced from JSON numbers
      - None passes through (e.g. k_min=None → auto)
    """
    cls  = ESTIMATOR_CLASSES[name]
    defs = {p["id"]: p for p in PARAM_DEFS.get(name, [])}
    kwargs = {}
    for k, v in (params or {}).items():
        if k not in defs:
            continue
        pdef = defs[k]
        if v is None:
            kwargs[k] = None
        elif pdef["type"] == "bool":
            kwargs[k] = bool(v)
        elif pdef["type"] == "int":
            kwargs[k] = int(v)
        elif pdef["type"] == "float":
            kwargs[k] = float(v)
        else:
            kwargs[k] = v
    return cls(**kwargs)
