"""Fit Chinchilla scaling law L(N,D) = E + A/N^{alpha} + B/D^{beta}"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScalingLawFit:
    E: float
    A: float
    B: float
    alpha: float
    beta: float
    rmse: float

    def predict(self, n_params: np.ndarray, tokens: np.ndarray) -> np.ndarray:
        return self.E + self.A / n_params**self.alpha + self.B / tokens**self.beta
    
    def iso_loss_contour_n(self, tokens: float, target_loss: float) -> float:
        """Solve for N at fixed D and target loss (single root, positive)."""
        residual = target_loss - self.E - self.B / tokens**self.beta
        if residual <= 0:
            return float("inf")
        return (self.A / residual) ** (1.0 / self.alpha)
    
    def iso_loss_contour_d(self, n_params: float, target_loss: float) -> float:
        residual = target_loss - self.E - self.A / n_params**self.alpha
        if residual <= 0:
            return float("inf")
        return (self.B / residual) ** (1.0 / self.beta)
    

@dataclass
class FreeScalingLawFit:
    """L(N, D) = a * (N/N0)^-alpha + b * (D/D0)^-beta, exponents fitted.

    Centered at the data's geometric means (N0, D0) so coefficients and
    exponents stay decorrelated when N and D span few orders of magnitude.
    """

    a: float
    b: float
    alpha: float
    beta: float
    n0: float
    d0: float
    rel_rmse: float

    def predict(self, n_params: np.ndarray, tokens: np.ndarray) -> np.ndarray:
        return (
            self.a * (n_params / self.n0) ** -self.alpha
            + self.b * (tokens / self.d0) ** -self.beta
        )

    def to_chinchilla(self) -> ScalingLawFit:
        """Convert to the uncentered E + A/N^a + B/D^b form (E = 0)."""
        return ScalingLawFit(
            E=0.0,
            A=self.a * self.n0**self.alpha,
            B=self.b * self.d0**self.beta,
            alpha=self.alpha,
            beta=self.beta,
            rmse=self.rel_rmse,
        )


def fit_scaling_law_free(
    runs: list[dict],
    *,
    use_active_params: bool = True,
) -> FreeScalingLawFit:
    """Fit a, b, alpha, beta jointly (E pinned to 0; needs scipy).

    Uses relative error weighting (sigma = loss) so small late-training losses
    count as much as large early ones.
    """
    n, d, loss = _arrays(runs, use_active_params)
    n0 = float(np.exp(np.mean(np.log(n))))
    d0 = float(np.exp(np.mean(np.log(d))))
    return _fit_free(n, d, loss, n0, d0)


def bootstrap_free_fit(
    runs: list[dict],
    *,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
    use_active_params: bool = True,
) -> dict:
    """Percentile bootstrap CIs for the free fit (case resampling over runs).

    Centering (N0, D0) is pinned to the full data so a, b keep one meaning
    across resamples. Resamples with fewer than 5 distinct (N, D) points
    cannot pin 4 parameters and are skipped, as are fits that fail to
    converge. Also reports the compute-optimal allocation exponent
    N_opt ~ C^(beta/(alpha+beta)).

    Returns {"a": (lo, hi), ..., "alloc_exp": (lo, hi), "n_ok": int}.
    """
    n, d, loss = _arrays(runs, use_active_params)
    n0 = float(np.exp(np.mean(np.log(n))))
    d0 = float(np.exp(np.mean(np.log(d))))
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        i = rng.integers(0, len(runs), len(runs))
        if len(set(zip(n[i], d[i]))) < 5:
            continue
        try:
            f = _fit_free(n[i], d[i], loss[i], n0, d0)
        except RuntimeError:  # curve_fit: no convergence within maxfev
            continue
        samples.append((f.a, f.b, f.alpha, f.beta, f.beta / (f.alpha + f.beta)))
    if not samples:
        raise ValueError("no bootstrap resample produced a fit")
    s = np.array(samples)
    q = 100 * np.array([(1 - ci) / 2, (1 + ci) / 2])
    lo, hi = np.percentile(s, q, axis=0)
    names = ("a", "b", "alpha", "beta", "alloc_exp")
    return {k: (float(l), float(h)) for k, l, h in zip(names, lo, hi)} | {"n_ok": len(s)}


def _arrays(runs: list[dict], use_active_params: bool) -> tuple[np.ndarray, ...]:
    key = "n_active_params" if use_active_params else "n_params"
    n = np.array([r[key] for r in runs], dtype=np.float64)
    d = np.array([r["tokens"] for r in runs], dtype=np.float64)
    loss = np.array([r["final_loss"] for r in runs], dtype=np.float64)
    return n, d, loss


def _fit_free(n, d, loss, n0: float, d0: float) -> FreeScalingLawFit:
    from scipy.optimize import curve_fit

    def model(x, a, b, alpha, beta):
        nn, dd = x
        return a * (nn / n0) ** -alpha + b * (dd / d0) ** -beta

    popt, _ = curve_fit(
        model,
        (n, d),
        loss,
        p0=[0.1, 0.1, 0.5, 0.5],
        bounds=([1e-6, 1e-6, 0.05, 0.05], [10.0, 10.0, 4.0, 4.0]),
        sigma=np.maximum(loss, 1e-3),
        maxfev=50_000,
    )
    a, b, alpha, beta = (float(v) for v in popt)
    pred = model((n, d), *popt)
    rel_rmse = float(np.sqrt(np.mean(((loss - pred) / np.maximum(loss, 1e-3)) ** 2)))
    return FreeScalingLawFit(a, b, alpha, beta, n0, d0, rel_rmse)


def fit_scaling_law(
    runs: list[dict],
    *,
    alpha: float = 0.34,
    beta: float = 0.28,
    use_active_params: bool = True,
) -> ScalingLawFit:
    """Fit E, A, B with fixed exponents (Chinchilla defaults)."""
    key = "n_active_params" if use_active_params else "n_params"
    n = np.array([r[key] for r in runs], dtype=np.float64)
    d = np.array([r["tokens"] for r in runs], dtype=np.float64)
    loss = np.array([r["final_loss"] for r in runs], dtype=np.float64)

    x_n = 1.0 / n**alpha
    x_d = 1.0 / d**beta
    design = np.stack([x_n, x_d], axis=1)
    coeffs, _, _, _ = np.linalg.lstsq(design, loss, rcond=None)
    A, B = coeffs
    E = 0.0
    pred = design @ coeffs
    rmse = float(np.sqrt(np.mean((loss - pred) ** 2)))
    return ScalingLawFit(float(E), float(A), float(B), alpha, beta, rmse)


def compute_optimal_allotment(
    fit: ScalingLawFit,
    flops_budget: float,
    *,
    use_active_params: bool = True,
) -> tuple[float, float, float]:
    """Chinchilla compute-optimal N, D for budget C = 6ND (active params)."""
    # N_opt = G * (C/6)^(0.5) style - from minimizing L s.t C=6ND
    # Standard result: N = C^a, D \prop C^b with a=b=0.5 for symmetric 6ND
    # More precisely with L = E + A/N^{alpha} + B/D^{beta} and C=6ND:
    # N_opt = ((alpha A)/beta B)^(1/(alpha+beta)) * (flops_budget / 6.0) ** (beta / (alpha + beta))
    # D_opt = C / (6 * N_opt)
    alpha, beta = fit.alpha, fit.beta
    ratio = (alpha * fit.A) / (beta * fit.B)
    n_opt = ratio ** (1.0 / (alpha + beta)) * (flops_budget / 6.0) ** (beta /(alpha + beta))
    d_opt = flops_budget / (6.0 * n_opt)
    loss = float(fit.predict(np.array([n_opt]), np.array([d_opt]))[0])
    return n_opt, d_opt, loss


def tokens_for_target_acc(runs: list[dict], target_acc: float = 0.99) -> dict | None:
    """Find smallest run meeting target accuracy; return its stats."""
    ok = [r for r in runs if r["addition_acc"] >= target_acc]
    if not ok:
        return None
    return min(ok, key=lambda r: r["tokens"])
