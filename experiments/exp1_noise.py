"""
exp1_noise.py
-------------
Study how each estimator responds to added noise.

For every (family, intrinsic dimension d) we sweep ETA_VALUES and record
each estimator's ID estimate and wall-clock runtime.  Each (eta, estimator)
cell is repeated N_RUNS=5 times using independent random subsamples of the
stored 25k dataset.  Results are saved to results/exp1_noise/.

Output
------
results/exp1_noise/
  {family}_d{d}.pdf        — ID estimate vs eta, one subplot per estimator
                              mean (line) + 95 % CI (band) + individual runs
  runtime_{family}.pdf     — mean runtime (s) vs intrinsic dim d,
                              one line per estimator (averaged over eta & runs)

Usage
-----
  python experiments/exp1_noise.py                             # all
  python experiments/exp1_noise.py --dims 2 5 10               # subset of dims
  python experiments/exp1_noise.py --families hypercube        # one family
  python experiments/exp1_noise.py --estimators MLE TwoNN      # one or more estimators
  python experiments/exp1_noise.py --no-generate               # skip dataset generation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from datasets import (
    generate_all, load_dataset,
    DIMS_INTRINSIC, AMBIENT_DIM, N_TOTAL,
)
from id import CorrInt, MLE, TwoNN, DANCo, ESS, TLE

# ── Experiment settings ───────────────────────────────────────────────────────

ETA_VALUES  = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
N_RUNS      = 5
N_SAMPLES   = 5_000   # points drawn per run from the 25k pool
N_NEIGHBORS = 20
BASE_SEED   = 42
FAMILIES    = ["hypercube", "gaussian", "linear"]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "exp1_noise")

Z95 = 1.96   # 95 % CI multiplier


def _make_estimators(seed: int) -> dict:
    return {
        "CorrInt": CorrInt(),
        "MLE":     MLE(n_neighbors=N_NEIGHBORS),
        "TwoNN":   TwoNN(),
        "DANCo":   DANCo(random_state=seed),
        "ESS":     ESS(n_neighbors=N_NEIGHBORS, random_state=seed),
        "TLE":     TLE(n_neighbors=N_NEIGHBORS),
    }


ALL_EST_NAMES = list(_make_estimators(0).keys())
_PALETTE      = plt.rcParams["axes.prop_cycle"].by_key()["color"]
# Colours are fixed per estimator regardless of which subset is selected
_COLOR        = {name: _PALETTE[i % len(_PALETTE)]
                 for i, name in enumerate(ALL_EST_NAMES)}


# ── Core computation ──────────────────────────────────────────────────────────

def run_experiment(dims: list[int],
                   families: list[str] | None = None,
                   est_names: list[str] | None = None,
                   n_samples: int = N_SAMPLES,
                   eta_values: list[float] = None,
                   ambient_factor: int = None,
                   ambient_dim: int = None) -> tuple[dict, dict]:
    """
    Parameters
    ----------
    dims      : intrinsic dimensions to sweep
    families  : subset of FAMILIES to run (default: all)
    est_names : subset of estimator names to run (default: all)

    Returns
    -------
    scores   : scores[family][d][est_name]   -> ndarray (n_eta, N_RUNS)
    runtimes : runtimes[family][d][est_name] -> ndarray (n_eta, N_RUNS)  [seconds]
    """
    if families is None:
        families = FAMILIES
    if est_names is None:
        est_names = ALL_EST_NAMES
    if eta_values is None:
        eta_values = ETA_VALUES

    scores   = {f: {} for f in families}
    runtimes = {f: {} for f in families}

    pbar_family = tqdm(families, desc="family", position=0)
    for family in pbar_family:
        pbar_family.set_description(f"family={family}")

        valid_dims = [d for d in dims
                      if not (family == "linear" and d > AMBIENT_DIM)]

        pbar_d = tqdm(valid_dims, desc="  d", position=1, leave=False)
        for d in pbar_d:
            pbar_d.set_description(f"  d={d}")

            sc = {name: np.full((len(eta_values), N_RUNS), np.nan)
                  for name in est_names}
            rt = {name: np.full((len(eta_values), N_RUNS), np.nan)
                  for name in est_names}

            pbar_eta = tqdm(enumerate(eta_values), total=len(eta_values),
                            desc="    eta", position=2, leave=False)
            for eta_idx, eta in pbar_eta:
                pbar_eta.set_description(f"    eta={eta:.3f}")

                D_d = (ambient_factor * d if ambient_factor else (ambient_dim if ambient_dim else AMBIENT_DIM))
                ds     = load_dataset(family, d, eta=eta,
                                      D=D_d if family == "linear" else None)
                X_full = ds["X"]

                pbar_run = tqdm(range(N_RUNS), desc="      run",
                                position=3, leave=False)
                for run in pbar_run:
                    pbar_run.set_description(f"      run={run+1}/{N_RUNS}")

                    sub_seed = BASE_SEED + eta_idx * 1000 + run
                    rng = np.random.default_rng(sub_seed)
                    idx = rng.choice(len(X_full), n_samples, replace=False)
                    X   = X_full[idx]

                    all_estimators = _make_estimators(sub_seed)
                    selected = {k: v for k, v in all_estimators.items()
                                if k in est_names}
                    pbar_est = tqdm(selected.items(), desc="        est",
                                    position=4, leave=False)
                    for name, est in pbar_est:
                        pbar_est.set_description(f"        {name}")
                        try:
                            t0 = time.perf_counter()
                            est.fit(X)
                            rt[name][eta_idx, run] = time.perf_counter() - t0
                            sc[name][eta_idx, run] = est.dimension_
                        except Exception:
                            pass   # leave as nan

            scores[family][d]   = sc
            runtimes[family][d] = rt

    return scores, runtimes


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _mean_ci(data: np.ndarray):
    """Given (n_eta, N_RUNS), return (mean, ci_halfwidth) each shape (n_eta,)."""
    n_valid = np.sum(~np.isnan(data), axis=1).clip(1)
    mean    = np.nanmean(data, axis=1)
    sem     = np.nanstd(data, axis=1, ddof=1) / np.sqrt(n_valid)
    return mean, Z95 * sem


# ── ID estimate plots ─────────────────────────────────────────────────────────

def plot_id(family: str, d: int, scores: dict, out_path: str,
            n_samples: int = N_SAMPLES,
            eta_values: list = None) -> None:
    """Subplots = estimators (only those present in scores); x = eta; y = estimated ID."""
    if eta_values is None:
        eta_values = ETA_VALUES
    present = [n for n in ALL_EST_NAMES if n in scores]   # preserve canonical order
    ncols = min(3, len(present))
    nrows = (len(present) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows),
                             squeeze=False, sharey=True)
    eta_arr = np.asarray(eta_values)

    for ax_idx, name in enumerate(present):
        ax    = axes[ax_idx // ncols][ax_idx % ncols]
        color = _COLOR[name]
        data  = scores[name]                  # (n_eta, N_RUNS)
        mean, ci = _mean_ci(data)

        ax.axhline(d, color="grey", linestyle="--", linewidth=1.2,
                   label=f"true d = {d}", zorder=1)
        for run in range(N_RUNS):
            ax.plot(eta_arr, data[:, run], color=color,
                    alpha=0.15, linewidth=0.8, zorder=1)
        ax.fill_between(eta_arr, mean - ci, mean + ci,
                        color=color, alpha=0.25, zorder=2)
        ax.plot(eta_arr, mean, color=color, linewidth=2,
                marker="o", markersize=5, zorder=3, label="mean ± 95% CI")

        ax.set_title(name, fontsize=12)
        ax.set_xlabel("η (noise level)")
        ax.set_ylabel("estimated ID")
        ax.legend(fontsize=8)
        ax.set_xticks(eta_arr)
        ax.set_xticklabels([str(e) for e in eta_values], rotation=45, ha="right")

    for ax_idx in range(len(present), nrows * ncols):
        axes[ax_idx // ncols][ax_idx % ncols].set_visible(False)

    _FAMILY_LABEL = {
        "hypercube": "Hypercube [0,1]^d",
        "gaussian":  "Isotropic Gaussian",
        "linear":    f"Linear subspace in R^{AMBIENT_DIM}",
    }
    fig.suptitle(
        f"{_FAMILY_LABEL[family]} — true d = {d}\n"
        f"({N_RUNS} runs × {n_samples} points each, η up to {max(eta_values)})",
        fontsize=14, y=1.01,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    tqdm.write(f"  saved  {os.path.relpath(out_path)}")


# ── Runtime plots ─────────────────────────────────────────────────────────────

def plot_runtime(family: str, dims: list[int], runtimes: dict,
                 out_path: str, n_samples: int = N_SAMPLES,
                 eta_values: list = None) -> None:
    """Mean runtime (s) vs intrinsic dimension d, one line per estimator."""
    if eta_values is None:
        eta_values = ETA_VALUES
    # Determine which estimators are actually present
    first_d = next(iter(runtimes), None)
    if first_d is None:
        return
    present = [n for n in ALL_EST_NAMES if n in runtimes[first_d]]

    fig, ax = plt.subplots(figsize=(8, 5))

    for name in present:
        means, cis = [], []
        valid_dims = []
        for d in dims:
            if d not in runtimes:
                continue
            data = runtimes[d][name]           # (n_eta, N_RUNS)
            m, ci = _mean_ci(data)
            # average over eta axis to get a single (runtime, ci) per d
            means.append(float(np.nanmean(m)))
            cis.append(float(np.nanmean(ci)))
            valid_dims.append(d)

        if not valid_dims:
            continue
        d_arr = np.asarray(valid_dims)
        m_arr = np.asarray(means)
        ci_arr = np.asarray(cis)

        ax.fill_between(d_arr, m_arr - ci_arr, m_arr + ci_arr,
                        color=_COLOR[name], alpha=0.20)
        ax.plot(d_arr, m_arr, color=_COLOR[name], linewidth=2,
                marker="o", markersize=5, label=name)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("intrinsic dimension d (log scale)")
    ax.set_ylabel("mean runtime per fit (s, log scale)")
    _FAMILY_LABEL = {
        "hypercube": "Hypercube [0,1]^d",
        "gaussian":  "Isotropic Gaussian",
        "linear":    f"Linear subspace in R^{AMBIENT_DIM}",
    }
    ax.set_title(
        f"Runtime — {_FAMILY_LABEL[family]}\n"
        f"({n_samples} points, mean ± 95% CI over {N_RUNS} runs × {len(ETA_VALUES)} η values)",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    tqdm.write(f"  saved  {os.path.relpath(out_path)}")


# ── Save all plots ────────────────────────────────────────────────────────────

def save_all_plots(scores: dict, runtimes: dict, dims: list[int],
                   families: list[str], n_samples: int = N_SAMPLES,
                   eta_values: list = None) -> None:
    if eta_values is None:
        eta_values = ETA_VALUES
    pbar = tqdm(families, desc="saving figures", position=0)
    for family in pbar:
        for d in scores[family]:
            out = os.path.join(RESULTS_DIR, f"{family}_d{d}.pdf")
            plot_id(family, d, scores[family][d], out, n_samples=n_samples,
                    eta_values=eta_values)

        out_rt = os.path.join(RESULTS_DIR, f"runtime_{family}.pdf")
        plot_runtime(family, dims, runtimes[family], out_rt, n_samples=n_samples,
                     eta_values=eta_values)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="exp1: noise sensitivity study")
    parser.add_argument("--dims", type=int, nargs="+", default=None,
                        metavar="D",
                        help="Intrinsic dimensions to run (default: all)")
    parser.add_argument("--families", type=str, nargs="+", default=None,
                        choices=FAMILIES, metavar="FAMILY",
                        help="Dataset families to run (default: all). "
                             f"Choices: {FAMILIES}")
    parser.add_argument("--estimators", type=str, nargs="+", default=None,
                        choices=ALL_EST_NAMES, metavar="EST",
                        help="Estimators to run (default: all). "
                             f"Choices: {ALL_EST_NAMES}")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES,
                        help=f"Points to subsample per run (default: {N_SAMPLES})")
    parser.add_argument("--no-generate", action="store_true",
                        help="Skip dataset generation (datasets must already exist)")
    parser.add_argument("--eta", type=float, nargs="+", default=None,
                        metavar="ETA",
                        help="Noise levels to sweep (default: ETA_VALUES). "
                             "Example: --eta 0 0.1 0.5 1.0")
    parser.add_argument("--ambient-factor", type=int, default=None,
                        metavar="K",
                        help="For linear family: use D = K * d instead of fixed "
                             f"AMBIENT_DIM={AMBIENT_DIM}. Example: --ambient-factor 2")
    parser.add_argument("--ambient-dim", type=int, default=None,
                        metavar="D",
                        help="For linear family: use this fixed ambient dimension D "
                             f"(default: {AMBIENT_DIM}). Example: --ambient-dim 50")
    args = parser.parse_args()

    dims           = args.dims           if args.dims           is not None else DIMS_INTRINSIC
    families       = args.families       if args.families       is not None else FAMILIES
    est_names      = args.estimators     if args.estimators     is not None else ALL_EST_NAMES
    n_samples      = args.n_samples
    eta_values     = args.eta            if args.eta            is not None else ETA_VALUES
    ambient_factor = args.ambient_factor
    ambient_dim    = args.ambient_dim

    if not args.no_generate:
        print("Generating datasets …")
        generate_all(eta_values=eta_values, n=N_TOTAL, seed=BASE_SEED,
                     dims=dims, ambient_factor=ambient_factor,
                     D=ambient_dim if ambient_dim else AMBIENT_DIM)

    print("\nRunning estimators …")
    scores, runtimes = run_experiment(dims, families=families, est_names=est_names,
                                      n_samples=n_samples, eta_values=eta_values,
                                      ambient_factor=ambient_factor,
                                      ambient_dim=ambient_dim)

    print("\nSaving figures …")
    save_all_plots(scores, runtimes, dims, families, n_samples=n_samples,
                   eta_values=eta_values)
    print("Done.")


if __name__ == "__main__":
    main()
