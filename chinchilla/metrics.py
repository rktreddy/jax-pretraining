"""Compute and training metrics for scaling-law analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class RunMetrics:
    architecture: str
    preset: str
    n_params: int
    n_active_params: int
    tokens: int
    steps: int
    batch_size: int
    final_loss: float
    addition_acc: float
    flops: float

    def to_dict(self) -> dict:
        return asdict(self)
    

def estimate_flops(n_active_params: int, tokens: int) -> float:
    """Approximate training FLOPS: 6 x active_params x tokens (forward + backward)."""
    return 6.0 * n_active_params * tokens


def count_active_params(
    n_params: int,
    *,
    architecture: str,
    n_layers: int,
    d_model: int,
    d_ff: int,
    n_experts: int = 1,
) -> int:
    """MoE active params = total - inactive expert FFN weights."""
    if architecture == "dense":
        return n_params
    inactive = n_layers * (n_experts - 1) * 2 * d_model * d_ff
    return n_params - inactive
    