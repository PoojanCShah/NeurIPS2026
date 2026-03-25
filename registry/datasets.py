"""
Dataset registry.

To add a new dataset family:
  1. Implement generate / load logic in datasets.py.
  2. Add the family name to FAMILIES below.

All UI dropdowns and experiment runners derive from these constants.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets import (
    DIMS_INTRINSIC, AMBIENT_DIM, N_TOTAL,
    load_dataset, generate_all,
)

# ── Registry constants ─────────────────────────────────────────────────────────

FAMILIES = ["hypercube", "gaussian", "linear"]

# η values offered in dropdowns
ETA_OPTIONS = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]

# Ambient dimension choices for the linear family (must be > d)
AMBIENT_DIMS = [10, 50, 100, 500, 1000]

# Per-family parameter definitions (drives dataset_panel UI)
FAMILY_PARAMS = {
    "hypercube": [
        {"id": "d", "type": "int", "label": "Intrinsic dim d",
         "default": 2, "min": 1},
    ],
    "gaussian": [
        {"id": "d", "type": "int", "label": "Intrinsic dim d",
         "default": 2, "min": 1},
    ],
    "linear": [
        {"id": "d", "type": "int", "label": "Intrinsic dim d",
         "default": 2, "min": 1},
        {"id": "D", "type": "int", "label": "Ambient dim D",
         "default": 1000, "min": 1},
    ],
}

# Default sweep values for each experiment tab
DEFAULT_ETA_SWEEP    = [0.0, 0.05, 0.1, 0.2, 0.5]
DEFAULT_SAMPLE_SIZES = [250, 500, 1000, 2000, 5000]
DEFAULT_JL_DIMS      = [5, 10, 20, 50, 100, 200]

# Re-export for convenience
__all__ = [
    "FAMILIES", "ETA_OPTIONS", "AMBIENT_DIMS", "FAMILY_PARAMS",
    "DEFAULT_ETA_SWEEP", "DEFAULT_SAMPLE_SIZES", "DEFAULT_JL_DIMS",
    "DIMS_INTRINSIC", "AMBIENT_DIM", "N_TOTAL",
    "load_dataset", "generate_all",
    "ensure_dataset",
]


def ensure_dataset(family: str, d: int, eta: float, D: int = None) -> None:
    """Generate the .npz file for (family, d, η[, D]) only if it does not exist."""
    if family == "linear":
        D = D if D is not None else AMBIENT_DIM
    else:
        D = d
    try:
        load_dataset(family, d, eta=eta, D=D if family == "linear" else None)
    except FileNotFoundError:
        generate_all(eta_values=[eta], dims=[d], D=D)
