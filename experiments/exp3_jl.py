"""
exp3_jl.py
----------
Study the effect of Johnson-Lindenstrauss random projections on ID estimation.

For every (family, intrinsic dimension d) we project the data to a sweep of
projection dimensions k using a Gaussian JL map, then estimate the ID on both
the original and the projected data.  Each (k, estimator) cell is repeated
N_RUNS=5 times; each run draws an independent subsample AND an independent
projection matrix, so CIs capture both sources of randomness.

The key question:  at what projection dimension k does id_jl recover id_original?

Output
------
results/exp3_jl/
  {family}_d{d}.pdf     — one subplot per estimator
                           x: projection dimension k (log scale)
                           y: estimated ID
                           grey dashed  = true intrinsic d
                           dotted band  = id_original mean ± 95% CI
                           solid line   = id_jl mean ± 95% CI vs k
  runtime_{family}.pdf  — mean runtime vs k (log-log), one line per estimator

Usage
-----
  python experiments/exp3_jl.py                              # all
  python experiments/exp3_jl.py --dims 2 5 10               # subset of dims
  python experiments/exp3_jl.py --families linear            # most informative
  python experiments/exp3_jl.py --estimators MLE TwoNN       # subset
  python experiments/exp3_jl.py --eta 0.0                    # noise level
  python experiments/exp3_jl.py --n-samples 2000             # points per run
  python experiments/exp3_jl.py --no-generate                # skip data gen
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

# Projection dimensions to sweep (will be capped at each dataset's ambient dim)
JL_DIMS     = [2, 5, 10, 20, 50, 100, 200, 500, 750, 1_000]
N_RUNS      = 5
N_SAMPLES   = 2_000
N_NEIGHBORS = 20
ETA         = 0.0
BASE_SEED   = 42
FAMILIES    = ["hypercube", "gaussian", "linear"]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "exp3_jl")

Z95 = 1.96


def _make_estimators(seed: int, n: int) -> dict:
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


# ── JL projection ─────────────────────────────────────────────────────────────

def jl_project(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Project X ∈ R^{n×D} to R^{n×k} via a scaled Gaussian JL map.

    Each entry of the projection matrix A is drawn from N(0, 1/k), so
    E[||Ax||²] = ||x||² (distance-preserving in expectation).
    Returns X unchanged if k >= D.
    """
    D = X.shape[1]
    if k >= D:
        return X
    A = rng.standard_normal((k, D)).astype(X.dtype)
    A /= np.sqrt(k)
    return X @ A.T   # (n, k)


# ── Core computation ──────────────────────────────────────────────────────────

def run_experiment(dims: list[int],
                   families: list[str] | None = None,
                   est_names: list[str] | None = None,
                   n_samples: int = N_SAMPLES,
                   eta: float = ETA) -> tuple[dict, dict, dict, dict]:
    """
    Returns
    -------
    orig_scores   : [family][d][est_name] -> ndarray (N_RUNS,)
    orig_runtimes : [family][d][est_name] -> ndarray (N_RUNS,)
    jl_scores     : [family][d][est_name] -> ndarray (n_jl_dims, N_RUNS)
    jl_runtimes   : [family][d][est_name] -> ndarray (n_jl_dims, N_RUNS)
    """
    if families is None:
        families = FAMILIES
    if est_names is None:
        est_names = ALL_EST_NAMES

    orig_scores    = {f: {} for f in families}
    orig_runtimes  = {f: {} for f in families}
    jl_scores      = {f: {} for f in families}
    jl_runtimes    = {f: {} for f in families}
    family_ambient = {}   # family -> ambient dim (same for all d in family)

    pbar_family = tqdm(families, desc="family", position=0)
    for family in pbar_family:
        pbar_family.set_description(f"family={family}")

        valid_dims = [d for d in dims
                      if not (family == "linear" and d > AMBIENT_DIM)]

        pbar_d = tqdm(valid_dims, desc="  d", position=1, leave=False)
        for d in pbar_d:
            pbar_d.set_description(f"  d={d}")

            ds     = load_dataset(family, d, eta=eta,
                                  D=AMBIENT_DIM if family == "linear" else None)
            X_full = ds["X"]
            ambient = X_full.shape[1]
            family_ambient[family] = ambient   # constant within a family

            # JL dims valid for this dataset
            valid_jl = [k for k in JL_DIMS if k <= ambient]

            o_sc = {name: np.full(N_RUNS, np.nan) for name in est_names}
            o_rt = {name: np.full(N_RUNS, np.nan) for name in est_names}
            j_sc = {name: np.full((len(valid_jl), N_RUNS), np.nan)
                    for name in est_names}
            j_rt = {name: np.full((len(valid_jl), N_RUNS), np.nan)
                    for name in est_names}

            pbar_run = tqdm(range(N_RUNS), desc="    run",
                            position=2, leave=False)
            for run in pbar_run:
                pbar_run.set_description(f"    run={run+1}/{N_RUNS}")

                sub_seed  = BASE_SEED + run
                proj_seed = BASE_SEED + run + 10_000   # independent stream
                rng_sub   = np.random.default_rng(sub_seed)
                rng_proj  = np.random.default_rng(proj_seed)

                idx = rng_sub.choice(len(X_full), n_samples, replace=False)
                X   = X_full[idx]

                # ── Original (no projection) ──────────────────────────────
                ests = _make_estimators(sub_seed, n_samples)
                pbar_est = tqdm(est_names, desc="      est (orig)",
                                position=3, leave=False)
                for name in pbar_est:
                    pbar_est.set_description(f"      {name} (orig)")
                    est = {k: v for k, v in ests.items() if k == name}[name]
                    try:
                        t0 = time.perf_counter()
                        est.fit(X)
                        o_rt[name][run] = time.perf_counter() - t0
                        o_sc[name][run] = est.dimension_
                    except Exception:
                        pass

                # ── JL projections ────────────────────────────────────────
                pbar_k = tqdm(enumerate(valid_jl), total=len(valid_jl),
                              desc="      k", position=3, leave=False)
                for k_idx, k in pbar_k:
                    pbar_k.set_description(f"      k={k}")
                    X_proj = jl_project(X, k, rng_proj)

                    ests_proj = _make_estimators(sub_seed, n_samples)
                    pbar_est2 = tqdm(est_names, desc="        est (jl)",
                                     position=4, leave=False)
                    for name in pbar_est2:
                        pbar_est2.set_description(f"        {name} k={k}")
                        est = {kk: v for kk, v in ests_proj.items()
                               if kk == name}[name]
                        try:
                            t0 = time.perf_counter()
                            est.fit(X_proj)
                            j_rt[name][k_idx, run] = time.perf_counter() - t0
                            j_sc[name][k_idx, run] = est.dimension_
                        except Exception:
                            pass

            orig_scores[family][d]   = o_sc
            orig_runtimes[family][d] = o_rt
            jl_scores[family][d]     = j_sc
            jl_runtimes[family][d]   = j_rt

    return orig_scores, orig_runtimes, jl_scores, jl_runtimes, family_ambient


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _mean_ci_1d(data: np.ndarray):
    """(N_RUNS,) -> (mean, ci_halfwidth) scalars."""
    n = np.sum(~np.isnan(data)).clip(1)
    m = np.nanmean(data)
    s = np.nanstd(data, ddof=1) / np.sqrt(n)
    return float(m), float(Z95 * s)


def _mean_ci_2d(data: np.ndarray):
    """(n_k, N_RUNS) -> (mean, ci_halfwidth) each shape (n_k,)."""
    n_valid = np.sum(~np.isnan(data), axis=1).clip(1)
    mean    = np.nanmean(data, axis=1)
    sem     = np.nanstd(data, axis=1, ddof=1) / np.sqrt(n_valid)
    return mean, Z95 * sem


# ── ID plots ──────────────────────────────────────────────────────────────────

def plot_id(family: str, d: int, ambient: int,
            orig_scores: dict, jl_scores: dict,
            out_path: str, n_samples: int, eta: float) -> None:
    """One subplot per estimator; x = k; y = estimated ID."""
    present = [n for n in ALL_EST_NAMES if n in jl_scores]
    # Infer valid_jl from actual data shape — avoids ambient mismatch for
    # hypercube/gaussian where ambient = d and varies per dataset
    n_k      = jl_scores[present[0]].shape[0]
    valid_jl = JL_DIMS[:n_k]
    k_arr    = np.asarray(valid_jl)

    ncols = min(3, len(present))
    nrows = (len(present) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows), squeeze=False,
                             sharey=True)

    for ax_idx, name in enumerate(present):
        ax    = axes[ax_idx // ncols][ax_idx % ncols]
        color = _COLOR[name]

        # true d
        ax.axhline(d, color="grey", linestyle="--", linewidth=1.2,
                   label=f"true d = {d}", zorder=1)

        # id_original — horizontal dotted band
        o_mean, o_ci = _mean_ci_1d(orig_scores[name])
        ax.axhspan(o_mean - o_ci, o_mean + o_ci,
                   color=color, alpha=0.12, zorder=2)
        ax.axhline(o_mean, color=color, linestyle=":",
                   linewidth=1.8, label="id_original", zorder=3)

        # id_jl vs k
        data = jl_scores[name]             # (n_k, N_RUNS)
        mean, ci = _mean_ci_2d(data)

        for run in range(N_RUNS):
            ax.plot(k_arr, data[:, run], color=color,
                    alpha=0.12, linewidth=0.8, zorder=2)
        ax.fill_between(k_arr, mean - ci, mean + ci,
                        color=color, alpha=0.25, zorder=3)
        ax.plot(k_arr, mean, color=color, linewidth=2,
                marker="o", markersize=5, zorder=4,
                label="id_jl ± 95% CI")

        # mark where k == d (the "critical" projection dim)
        if d in valid_jl:
            ax.axvline(d, color="black", linestyle="-.",
                       linewidth=0.9, alpha=0.5, label=f"k = d = {d}")

        ax.set_xscale("log")
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("projection dim k (log scale)")
        ax.set_ylabel("estimated ID")
        ax.legend(fontsize=7)
        ax.set_xticks(k_arr)
        ax.set_xticklabels([str(k) for k in k_arr], rotation=45, ha="right")
        ax.xaxis.set_minor_locator(plt.NullLocator())

    for ax_idx in range(len(present), nrows * ncols):
        axes[ax_idx // ncols][ax_idx % ncols].set_visible(False)

    _FAMILY_LABEL = {
        "hypercube": f"Hypercube [0,1]^d  (ambient = d = {d})",
        "gaussian":  f"Isotropic Gaussian  (ambient = d = {d})",
        "linear":    f"Linear subspace  (intrinsic d={d}, ambient D={AMBIENT_DIM})",
    }
    fig.suptitle(
        f"{_FAMILY_LABEL[family]}  η={eta:.3f}\n"
        f"({N_RUNS} runs × {n_samples} points, fresh JL matrix per run)",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    tqdm.write(f"  saved  {os.path.relpath(out_path)}")


# ── Runtime plot ──────────────────────────────────────────────────────────────

def plot_runtime(family: str, dims: list[int], jl_runtimes: dict,
                 ambient: int, out_path: str, eta: float) -> None:
    """Mean runtime vs k (log-log), one line per estimator, averaged over dims."""
    first_d = next(iter(jl_runtimes), None)
    if first_d is None:
        return
    present  = [n for n in ALL_EST_NAMES if n in jl_runtimes[first_d]]
    # Infer from data shape — each d may have different valid_jl length;
    # use the maximum (linear family) so the runtime plot is as complete as possible
    max_n_k  = max(jl_runtimes[d][present[0]].shape[0]
                   for d in jl_runtimes if present[0] in jl_runtimes[d])
    valid_jl = JL_DIMS[:max_n_k]
    k_arr    = np.asarray(valid_jl)

    fig, ax = plt.subplots(figsize=(8, 5))

    for name in present:
        all_rt = []
        for d in dims:
            if d not in jl_runtimes:
                continue
            rt = jl_runtimes[d][name]   # (n_k_d, N_RUNS)
            # Pad shorter arrays with NaN so all dims can be stacked
            padded = np.full((len(valid_jl), rt.shape[1]), np.nan)
            padded[:rt.shape[0]] = rt
            all_rt.append(padded)

        if not all_rt:
            continue

        stacked = np.nanmean(np.stack(all_rt, axis=0), axis=0)  # (max_n_k, N_RUNS)
        mean, ci = _mean_ci_2d(stacked)

        ax.fill_between(k_arr, mean - ci, mean + ci,
                        color=_COLOR[name], alpha=0.20)
        ax.plot(k_arr, mean, color=_COLOR[name], linewidth=2,
                marker="o", markersize=5, label=name)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("projection dim k (log scale)")
    ax.set_ylabel("mean runtime per fit (s, log scale)")
    _FAMILY_LABEL = {
        "hypercube": "Hypercube [0,1]^d",
        "gaussian":  "Isotropic Gaussian",
        "linear":    f"Linear subspace in R^{AMBIENT_DIM}",
    }
    ax.set_title(
        f"Runtime — {_FAMILY_LABEL[family]}  (η={eta:.3f})\n"
        f"(mean ± 95% CI over {N_RUNS} runs, averaged across dims)",
        fontsize=12,
    )
    ax.set_xticks(k_arr)
    ax.set_xticklabels([str(k) for k in k_arr], rotation=45, ha="right")
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.legend(fontsize=9)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    tqdm.write(f"  saved  {os.path.relpath(out_path)}")


# ── Save all ──────────────────────────────────────────────────────────────────

def save_all_plots(orig_scores: dict, jl_scores: dict,
                   jl_runtimes: dict, dims: list[int],
                   families: list[str], n_samples: int, eta: float,
                   family_ambient: dict) -> None:
    pbar = tqdm(families, desc="saving figures", position=0)
    for family in pbar:
        ambient = family_ambient[family]

        for d in jl_scores[family]:
            out = os.path.join(RESULTS_DIR, f"{family}_d{d}.pdf")
            plot_id(family, d, ambient,
                    orig_scores[family][d], jl_scores[family][d],
                    out, n_samples, eta)

        out_rt = os.path.join(RESULTS_DIR, f"runtime_{family}.pdf")
        plot_runtime(family, dims, jl_runtimes[family],
                     ambient, out_rt, eta)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="exp3: JL projection effect on ID estimation")
    parser.add_argument("--dims", type=int, nargs="+", default=None,
                        metavar="D",
                        help="Intrinsic dimensions to run (default: all)")
    parser.add_argument("--families", type=str, nargs="+", default=None,
                        choices=FAMILIES, metavar="FAMILY",
                        help=f"Families to run. Choices: {FAMILIES}")
    parser.add_argument("--estimators", type=str, nargs="+", default=None,
                        choices=ALL_EST_NAMES, metavar="EST",
                        help=f"Estimators to run. Choices: {ALL_EST_NAMES}")
    parser.add_argument("--eta", type=float, default=ETA,
                        help=f"Noise level (default: {ETA})")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES,
                        help=f"Points per run (default: {N_SAMPLES})")
    parser.add_argument("--no-generate", action="store_true",
                        help="Skip dataset generation")
    args = parser.parse_args()

    dims      = args.dims       if args.dims       is not None else DIMS_INTRINSIC
    families  = args.families   if args.families   is not None else FAMILIES
    est_names = args.estimators if args.estimators is not None else ALL_EST_NAMES
    eta       = args.eta
    n_samples = args.n_samples

    if not args.no_generate:
        print("Generating datasets …")
        generate_all(eta_values=[eta], n=N_TOTAL, seed=BASE_SEED, dims=dims)

    print("\nRunning estimators …")
    orig_sc, orig_rt, jl_sc, jl_rt, fam_ambient = run_experiment(
        dims, families=families, est_names=est_names,
        n_samples=n_samples, eta=eta,
    )

    print("\nSaving figures …")
    save_all_plots(orig_sc, jl_sc, jl_rt, dims, families,
                   n_samples, eta, fam_ambient)
    print("Done.")


if __name__ == "__main__":
    main()
