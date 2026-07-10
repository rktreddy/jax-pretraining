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
    """Chinhilla compute-optimal N, D for budget C = 6ND (active params)."""
    # N_opt = G * (C/6)^(0.5) style - from minimizing L s.t C=6ND
    # Standard result: N = C^a, D \prop C^b with a=b=0.5 for symmetric 6ND
    # More precisely with L = E + A/N^{alpha} + B/D^{beta} and C=6ND:
    # N_opt = ((alpha A)/beta B)^(1/(alpha+beta)) * (flops_budget / 6.0) ** (beta / (alpha + beta))
    # D_opt = C / (6 + N_opt)
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
