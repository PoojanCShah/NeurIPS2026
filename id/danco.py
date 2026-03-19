"""
Dimensionality from Angle and Norm Concentration (DANCo).

References
----------
Ceruti et al. (2012). DANCo: An intrinsic dimensionality estimator exploiting
    angle and norm concentration. Pattern Recognition, 46(8), 2158–2171.
"""

import sys
import numpy as np
from scipy.optimize import minimize
from scipy.special import i0, i1, digamma, gammainc
from scipy.interpolate import interp1d
from ._utils import knn, lens, binom_coeff, indnComb


# ── Module-level helpers ──────────────────────────────────────────────────────

def _hyperBall(n, d, radius=1.0, rng=None):
    """Sample n points uniformly from a d-dimensional ball (Muller 1959)."""
    if rng is None:
        rng = np.random.default_rng()
    x = rng.standard_normal((n, d))
    ssq = np.sum(x ** 2, axis=1)
    fr = radius * gammainc(d / 2, ssq / 2) ** (1 / d) / np.sqrt(ssq)
    return x * fr[:, None]


def _Ainv(eta):
    """Inverse of A(τ) = I₁(τ)/I₀(τ) for the von Mises distribution."""
    if eta < 0.53:
        return 2 * eta + eta ** 3 + 5 * (eta ** 5) / 6
    elif eta < 0.85:
        return -0.4 + 1.39 * eta + 0.43 / (1 - eta)
    else:
        return 1 / (eta ** 3 - 4 * eta ** 2 + 3 * eta)


# ── Estimator ─────────────────────────────────────────────────────────────────

class DANCo:
    """Intrinsic dimension via DANCo (Ceruti et al., 2012).

    Algorithm
    ---------
    DANCo jointly uses two sources of information from the k-NN graph:

    1. **Norm** (MIND_ML): For each point i, the ratio
           ρ_i = dist(i, 1st NN) / dist(i, k-th NN)
       follows a known distribution on [0,1] parameterised by d.
       The log-likelihood is maximised first over integers (MIND_MLi),
       then refined continuously (MIND_MLk):

           l(d) = N·log(k·d) + (d−1)·Σ log(ρ) + (k−1)·Σ log(1−ρ^d)

    2. **Angle**: Pairwise angles between k neighbour vectors at each point are
       modelled with a von Mises distribution; parameters (ν, τ) are estimated
       by ML.

    For each candidate dimension d = 1, …, D, calibration statistics are
    generated from a uniform d-ball and the KL divergence to the data
    statistics is computed. The d with minimum KL divergence is selected.

    If ``fractal=True``, the KL curve is spline-smoothed and the continuous
    minimum is returned.

    Parameters
    ----------
    k : int
        Neighbourhood size.
    D : int or None
        Maximum candidate dimension (defaults to ambient dimension).
    fractal : bool
        Return continuous (spline-interpolated) dimension if True.
    random_state : int or None
        Seed for calibration data generation.
    """

    def __init__(self, k: int = 10, D: int = None, fractal: bool = True,
                 random_state=None):
        self.k = k
        self.D = D
        self.fractal = fractal
        self.random_state = random_state

    # ── Log-likelihood (norm component) ──────────────────────────────────────

    def _lld(self, d, rhos, N):
        if d <= 0:
            return -1e30
        return (N * np.log(self.k * d)
                + (d - 1) * np.sum(np.log(rhos))
                + (self.k - 1) * np.sum(np.log(1 - rhos ** d)))

    def _lld_grad(self, d, rhos, N):
        if d <= 0:
            return -1e30
        return -(N / d
                 + np.sum(np.log(rhos)
                          - (self.k - 1) * rhos ** d * np.log(rhos) / (1 - rhos ** d)))

    def _MIND_MLi(self, rhos, D):
        """Integer MLE: grid search over d = 1 … D."""
        N = len(rhos)
        liks = [self._lld(d + 1, rhos, N) for d in range(D)]
        return int(np.argmax(liks)) + 1

    def _MIND_MLk(self, rhos, D, d_init):
        """Continuous MLE: L-BFGS-B starting from d_init."""
        res = minimize(
            fun=lambda d: -self._lld(d[0], rhos, len(rhos)),
            x0=[d_init],
            jac=lambda d: self._lld_grad(d[0], rhos, len(rhos)),
            method="L-BFGS-B",
            bounds=[(1e-6, D)],
        )
        return float(res.x[0])

    # ── Angle component ───────────────────────────────────────────────────────

    def _local_angles(self, pt, neighbors):
        """Pairwise angles between vectors from pt to each neighbor."""
        vecs = neighbors - pt
        norms = lens(vecs)
        pairs = indnComb(len(neighbors), 2)
        i, j = pairs[:, 0], pairs[:, 1]
        cos_th = np.clip(
            np.sum(vecs[i] * vecs[j], axis=1) / (norms[i] * norms[j]), -1, 1
        )
        return np.arccos(cos_th)

    def _fit_vonMises(self, thetas):
        """ML estimate of von Mises (ν, τ) from angles."""
        nu = np.arctan2(np.sum(np.sin(thetas)), np.sum(np.cos(thetas)))
        eta = np.sqrt(np.mean(np.cos(thetas)) ** 2 + np.mean(np.sin(thetas)) ** 2)
        tau = _Ainv(eta)
        return nu, tau

    def _data_statistics(self, X, dists, idx):
        """Compute summary statistics (d_hat, mu_nu, mu_tau) from the dataset."""
        N = len(X)
        rhos = dists[:, 0] / dists[:, -1]

        d_i = self._MIND_MLi(rhos, self.D_)
        d_hat = self._MIND_MLk(rhos, self.D_, d_i)

        thetas_all = np.concatenate([
            self._local_angles(X[i], X[idx[i, :self.k]])
            for i in range(N)
        ])
        mu_nu, mu_tau = self._fit_vonMises(thetas_all)
        return dict(dhat=d_hat, mu_nu=mu_nu, mu_tau=mu_tau)

    # ── KL divergences ────────────────────────────────────────────────────────

    def _KL_norm(self, dhat, dcal):
        """KL divergence between two MIND_ML distributions (norm component)."""
        H_k = np.sum(1.0 / np.arange(1, self.k + 1))
        quo = dcal / dhat
        a = sum(
            (-1) ** i * binom_coeff(self.k, i) * digamma(1 + i / quo)
            for i in range(self.k + 1)
        )
        return H_k * quo - np.log(quo) - (self.k - 1) * a

    @staticmethod
    def _KL_vonMises(nu1, nu2, tau1, tau2):
        """KL divergence between two von Mises distributions."""
        _safe = lambda x: min(sys.float_info.max, x)
        return (np.log(_safe(i0(tau2)) / _safe(i0(tau1)))
                + _safe(i1(tau1)) / _safe(i0(tau1)) * (tau1 - tau2 * np.cos(nu1 - nu2)))

    def _KL(self, data_stats, cal_stats):
        return (self._KL_norm(data_stats["dhat"], cal_stats["dhat"])
                + self._KL_vonMises(data_stats["mu_nu"], cal_stats["mu_nu"],
                                    data_stats["mu_tau"], cal_stats["mu_tau"]))

    # ── Calibration data ──────────────────────────────────────────────────────

    def _calibration_stats(self, N, d_cal):
        """Statistics from a fresh uniform d_cal-ball sample."""
        cal_data = _hyperBall(N, d_cal, rng=self.rng_)
        cal_dists, cal_idx = knn(cal_data, self.k + 1)
        return self._data_statistics(cal_data, cal_dists, cal_idx)

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        N, ambient = X.shape

        self.rng_ = np.random.default_rng(self.random_state)
        self.k = min(self.k, N - 2)
        self.D_ = ambient if self.D is None else self.D

        # ── Step 1: data statistics ───────────────────────────────────────────
        dists, idx = knn(X, self.k + 1)
        data_stats = self._data_statistics(X, dists, idx)

        if any(np.isnan(v) for v in data_stats.values()):
            self.dimension_ = np.nan
            return self

        # ── Step 2: calibration + KL for each candidate dimension ────────────
        kl = np.full(self.D_, np.nan)
        self.calibration_data_ = []
        for d in range(1, self.D_ + 1):
            cal_stats = self._calibration_stats(N, d)
            self.calibration_data_.append(cal_stats)
            kl[d - 1] = self._KL(data_stats, cal_stats)

        # ── Step 3: dimension with minimum KL ────────────────────────────────
        de = int(np.argmin(kl)) + 1
        self.kl_divergence_ = kl[de - 1]

        if self.fractal and self.D_ >= 3:
            kind = {2: "linear", 3: "quadratic"}.get(ambient, "cubic")
            f = interp1d(np.arange(1, self.D_ + 1), kl, kind=kind,
                         bounds_error=False, fill_value=(kl[0], kl[-1]))
            self.dimension_ = float(minimize(f, de, bounds=[(1, self.D_)], tol=1e-3).x[0])
        else:
            self.dimension_ = float(de)

        return self
