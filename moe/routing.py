"""Token routing utilities for ragged MoE"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def route_tokens(
    x: jnp.ndarray,
    expert_idx: jnp.ndarray,
    n_experts: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sort tokens by expert for ragged_dot.
    
    Args:
        x: (..., D) activations
        expert_idx: (...,) int expert id per token
        n_experts: number of experts

    Returns:
        sorted_x: (M, D) tokens grouped by expert
        group_sizes: (n_experts,) token count per expert
        sort_idx: permutation used to sort
        inv_sort_idx: inverse permutation to restore order
    """
    orig_shape = x.shape
    d = orig_shape[-1]
    flat_x = x.reshape(-1, d)
    flat_idx = expert_idx.reshape(-1).astype(jnp.int32)

    sort_idx = jnp.argsort(flat_idx)
    sorted_x = flat_x[sort_idx]
    group_sizes = jnp.bincount(flat_idx, length=n_experts).astype(jnp.int32)
    inv_sort_idx = jnp.empty_like(sort_idx)
    inv_sort_idx = inv_sort_idx.at[sort_idx].set(jnp.arange(sort_idx.shape[0]))
    return sorted_x, group_sizes, sort_idx, inv_sort_idx


def unroute_tokens(
    sorted_y: jnp.ndarray,
    inv_sort_idx: jnp.ndarray,
    orig_shape: tuple[int, ...],
) -> jnp.ndarray:
    """Restore original token order after ragged MoE."""
    flat = sorted_y[inv_sort_idx]
    return flat.reshape(orig_shape)

