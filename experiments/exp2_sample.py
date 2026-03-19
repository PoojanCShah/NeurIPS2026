"""
exp2_sample.py
--------------
Study how each estimator's accuracy and runtime scale with sample size.

For every (family, intrinsic dimension d) we sweep SAMPLE_SIZES and record
each estimator's ID estimate and wall-clock runtime at a fixed noise level eta.
Each (n, estimator) cell is repeated N_RUNS=5 times using independent random
subsamples of the stored 25k dataset.  Results are saved to results/exp2_sample/.

Output
------
results/exp2_sample/
  {family}_d{d}.pdf        — ID estimate vs n, one subplot per estimator
                              mean (line) + 95 % CI (band) + individual runs
                              x-axis is log-scaled
  runtime_{family}.pdf     — mean runtime (s) vs n (log-log),
                              one line per estimator (averaged over dims)

Usage
-----
  python experiments/exp2_sample.py                             # all
  python experiments/exp2_sample.py --dims 2 5 10              # subset of dims
  python experiments/exp2_sample.py --families hypercube       # one family
  python experiments/exp2_sample.py --estimators MLE TwoNN     # subset of estimators
  python experiments/exp2_sample.py --eta 0.1                  # run on noisy data
  python experiments/exp2_sample.py --no-generate              # skip dataset generation
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

SAMPLE_SIZES = [50, 100, 250, 500, 750, 1_000, 2_500, 5_000]
N_RUNS       = 5
N_NEIGHBORS  = 20
ETA          = 0.0   # default noise level; override with --eta
BASE_SEED    = 42
FAMILIES     = ["hypercube", "gaussian", "linear"]

RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "..", "results", "exp2_sample")

Z95 = 1.96


def _make_estimators(seed: int, n: int) -> dict:
    # Clamp n_neighbors to n-1 so small-n runs don't crash
    k = min(N_NEIGHBORS, n - 1)
    return {
        "CorrInt": CorrInt(),
        "MLE":     MLE(n_neighbors=k),
        "TwoNN":   TwoNN(),
        "DANCo":   DANCo(random_state=seed),
        "ESS":     ESS(n_neighbors=k, random_state=seed),
        "TLE":     TLE(n_neighbors=k),
    }


ALL_EST_NAMES = list(_make_estimators(0, 100).keys())
_PALETTE      = plt.rcParams["axes.prop_cycle"].by_key()["color"]
_COLOR        = {name: _PALETTE[i % len(_PALETTE)]
                 for i, name in enumerate(ALL_EST_NAMES)}


# ── Core computation ──────────────────────────────────────────────────────────

def run_experiment(dims: list[int],
                   families: list[str] | None = None,
                   est_names: list[str] | None = None,
                   eta: float = ETA) -> tuple[dict, dict]:
    """
    Parameters
    ----------
    dims      : intrinsic dimensions to sweep
    families  : subset of FAMILIES to run (default: all)
    est_names : subset of estimator names to run (default: all)
    eta       : fixed noise level used for all datasets

    Returns
    -------
    scores   : scores[family][d][est_name]   -> ndarray (n_sizes, N_RUNS)
    runtimes : runtimes[family][d][est_name] -> ndarray (n_sizes, N_RUNS)  [seconds]
    """
    if families is None:
        families = FAMILIES
    if est_names is None:
        est_names = ALL_EST_NAMES

    scores   = {f: {} for f in families}
    runtimes = {f: {} for f in families}

    pbar_family = tqdm(families, desc="family", position=0)
    for family in pbar_family:
        pbar_family.set_description(f"family={family}")

        valid_dims = [d for d in dims
                      if not (family == "linear" and d > AMBIENT_DIM)]

        # Load each dataset once per (family, d) — subsampling varies below
        pbar_d = tqdm(valid_dims, desc="  d", position=1, leave=False)
        for d in pbar_d:
            pbar_d.set_description(f"  d={d}")

            ds     = load_dataset(family, d, eta=eta,
                                  D=AMBIENT_DIM if family == "linear" else None)
            X_full = ds["X"]

            sc = {name: np.full((len(SAMPLE_SIZES), N_RUNS), np.nan)
                  for name in est_names}
            rt = {name: np.full((len(SAMPLE_SIZES), N_RUNS), np.nan)
                  for name in est_names}

            pbar_n = tqdm(enumerate(SAMPLE_SIZES), total=len(SAMPLE_SIZES),
                          desc="    n", position=2, leave=False)
            for n_idx, n in pbar_n:
                pbar_n.set_description(f"    n={n}")

                if n > len(X_full):
                    tqdm.write(f"  skip  n={n} > dataset size {len(X_full)}")
                    continue

                pbar_run = tqdm(range(N_RUNS), desc="      run",
                                position=3, leave=False)
                for run in pbar_run:
                    pbar_run.set_description(f"      run={run+1}/{N_RUNS}")

                    sub_seed = BASE_SEED + n_idx * 1000 + run
                    rng = np.random.default_rng(sub_seed)
                    idx = rng.choice(len(X_full), n, replace=False)
                    X   = X_full[idx]

                    all_estimators = _make_estimators(sub_seed, n)
                    selected = {k: v for k, v in all_estimators.items()
                                if k in est_names}
                    pbar_est = tqdm(selected.items(), desc="        est",
                                    position=4, leave=False)
                    for name, est in pbar_est:
                        pbar_est.set_description(f"        {name}")
                        try:
                            t0 = time.perf_counter()
                            est.fit(X)
                            rt[name][n_idx, run] = time.perf_counter() - t0
                            sc[name][n_idx, run] = est.dimension_
                        except Exception:
                            pass   # leave as nan

            scores[family][d]   = sc
            runtimes[family][d] = rt

    return scores, runtimes


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _mean_ci(data: np.ndarray):
    """Given (n_sizes, N_RUNS), return (mean, ci_halfwidth) each shape (n_sizes,)."""
    n_valid = np.sum(~np.isnan(data), axis=1).clip(1)
    mean    = np.nanmean(data, axis=1)
    sem     = np.nanstd(data, axis=1, ddof=1) / np.sqrt(n_valid)
    return mean, Z95 * sem


# ── ID estimate plots ─────────────────────────────────────────────────────────

def plot_id(family: str, d: int, scores: dict, out_path: str,
            eta: float = ETA) -> None:
    """Subplots = estimators; x = n_samples (log); y = estimated ID."""
    present = [n for n in ALL_EST_NAMES if n in scores]
    ncols = min(3, len(present))
    nrows = (len(present) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows),
                             squeeze=False, sharey=True)
    n_arr = np.asarray(SAMPLE_SIZES)

    for ax_idx, name in enumerate(present):
        ax    = axes[ax_idx // ncols][ax_idx % ncols]
        color = _COLOR[name]
        data  = scores[name]               # (n_sizes, N_RUNS)
        mean, ci = _mean_ci(data)

        ax.axhline(d, color="grey", linestyle="--", linewidth=1.2,
                   label=f"true d = {d}", zorder=1)
        for run in range(N_RUNS):
            ax.plot(n_arr, data[:, run], color=color,
                    alpha=0.15, linewidth=0.8, zorder=1)
        ax.fill_between(n_arr, mean - ci, mean + ci,
                        color=color, alpha=0.25, zorder=2)
        ax.plot(n_arr, mean, color=color, linewidth=2,
                marker="o", markersize=5, zorder=3, label="mean ± 95% CI")

        ax.set_xscale("log")
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("sample size n (log scale)")
        ax.set_ylabel("estimated ID")
        ax.legend(fontsize=8)
        ax.set_xticks(n_arr)
        ax.set_xticklabels([str(n) for n in n_arr], rotation=45, ha="right")
        ax.xaxis.set_minor_locator(plt.NullLocator())

    for ax_idx in range(len(present), nrows * ncols):
        axes[ax_idx // ncols][ax_idx % ncols].set_visible(False)

    _FAMILY_LABEL = {
        "hypercube": "Hypercube [0,1]^d",
        "gaussian":  "Isotropic Gaussian",
        "linear":    f"Linear subspace in R^{AMBIENT_DIM}",
    }
    fig.suptitle(
        f"{_FAMILY_LABEL[family]} — true d = {d}  (η = {eta:.3f})\n"
        f"({N_RUNS} runs per sample size)",
        fontsize=14, y=1.01,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    tqdm.write(f"  saved  {os.path.relpath(out_path)}")


# ── Runtime plots ─────────────────────────────────────────────────────────────

def plot_runtime(family: str, dims: list[int], runtimes: dict,
                 out_path: str, eta: float = ETA) -> None:
    """Mean runtime (s) vs n (log-log), one line per estimator, averaged over dims."""
    first_d = next(iter(runtimes), None)
    if first_d is None:
        return
    present = [n for n in ALL_EST_NAMES if n in runtimes[first_d]]

    fig, ax = plt.subplots(figsize=(8, 5))
    n_arr = np.asarray(SAMPLE_SIZES)

    for name in present:
        # Collect (n_sizes, N_RUNS) arrays across all dims and stack
        all_rt = []
        for d in dims:
            if d not in runtimes:
                continue
            all_rt.append(runtimes[d][name])   # (n_sizes, N_RUNS)

        if not all_rt:
            continue

        # Average across dims first, then compute mean/CI over runs
        stacked = np.nanmean(np.stack(all_rt, axis=0), axis=0)  # (n_sizes, N_RUNS)
        mean, ci = _mean_ci(stacked)

        ax.fill_between(n_arr, mean - ci, mean + ci,
                        color=_COLOR[name], alpha=0.20)
        ax.plot(n_arr, mean, color=_COLOR[name], linewidth=2,
                marker="o", markersize=5, label=name)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sample size n (log scale)")
    ax.set_ylabel("mean runtime per fit (s, log scale)")
    _FAMILY_LABEL = {
        "hypercube": "Hypercube [0,1]^d",
        "gaussian":  "Isotropic Gaussian",
        "linear":    f"Linear subspace in R^{AMBIENT_DIM}",
    }
    ax.set_title(
        f"Runtime — {_FAMILY_LABEL[family]}  (η = {eta:.3f})\n"
        f"(mean ± 95% CI over {N_RUNS} runs, averaged across dims)",
        fontsize=12,
    )
    ax.set_xticks(n_arr)
    ax.set_xticklabels([str(n) for n in n_arr], rotation=45, ha="right")
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.legend(fontsize=9)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    tqdm.write(f"  saved  {os.path.relpath(out_path)}")


# ── Save all plots ────────────────────────────────────────────────────────────

def save_all_plots(scores: dict, runtimes: dict, dims: list[int],
                   families: list[str], eta: float = ETA) -> None:
    pbar = tqdm(families, desc="saving figures", position=0)
    for family in pbar:
        for d in scores[family]:
            out = os.path.join(RESULTS_DIR, f"{family}_d{d}.pdf")
            plot_id(family, d, scores[family][d], out, eta=eta)

        out_rt = os.path.join(RESULTS_DIR, f"runtime_{family}.pdf")
        plot_runtime(family, dims, runtimes[family], out_rt, eta=eta)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="exp2: sample size study")
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
    parser.add_argument("--eta", type=float, default=ETA,
                        help=f"Noise level for datasets (default: {ETA}). "
                             "Must match a generated dataset.")
    parser.add_argument("--no-generate", action="store_true",
                        help="Skip dataset generation (datasets must already exist)")
    args = parser.parse_args()

    dims      = args.dims       if args.dims       is not None else DIMS_INTRINSIC
    families  = args.families   if args.families   is not None else FAMILIES
    est_names = args.estimators if args.estimators is not None else ALL_EST_NAMES
    eta       = args.eta

    if not args.no_generate:
        print("Generating datasets …")
        generate_all(eta_values=[eta], n=N_TOTAL, seed=BASE_SEED, dims=dims)

    print("\nRunning estimators …")
    scores, runtimes = run_experiment(dims, families=families,
                                      est_names=est_names, eta=eta)

    print("\nSaving figures …")
    save_all_plots(scores, runtimes, dims, families, eta=eta)
    print("Done.")


if __name__ == "__main__":
    main()
