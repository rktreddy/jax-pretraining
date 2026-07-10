"""Chinchilla scaling-law experiments for the addition transformer."""

from chinchilla.fit import ScalingLawFit, compute_optimal_allotment, fit_scaling_law
from chinchilla.report import generate_report
from chinchilla.sweep import quick_sweep, run_sweep

__all__ = [
    "ScalingLawFit",
    "compute_optimal_allotment",
    "fit_scaling_law",
    "generate_report",
    "quick_sweep",
    "run_sweep",
]