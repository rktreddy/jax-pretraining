"""Model presets and sweep grids for Chinchilla experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPreset:
    name: str
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int


DENSE_PRESETS: tuple[ModelPreset, ...] = (
    ModelPreset("tiny", 64, 2, 4, 256),
    ModelPreset("small", 128, 4, 4, 512),
    ModelPreset("medium", 224, 6, 8, 896),
    ModelPreset("base", 320, 8, 8, 1280),
)

# Token budgests = max_steps * batch_size * ~4 answer tokens / example
TOKEN_BUDGETS: tuple[int, ...] = (
    250_000,
    500_000,
    1_000_000,
    2_000_000,
)

MOE_EXPERTS = 4


def step_for_token_budget(tokens: int, batch_size: int = 128) -> int:
    """Approximate steps to reach token budget (amswer tokens ony, ~4/example)."""
    tokens_per_step = batch_size * 4
    return max(200, tokens // tokens_per_step)


def curriculum_for_steps(max_steps: int) -> tuple[int, int]:
    """Scale curriculum stages to total training length."""
    stage1 = max(200, max_steps // 8)
    stage2 = max(stage1 + 200, int(max_steps * 0.6))
    return stage1, stage2