"""MoE FFN using jax.lax.ragged_dot (reference implementation)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from moe.routing import route_tokens, unroute_tokens

def moe_ffn_ragged(
    x: jnp.ndarray,
    w_up: jnp.ndarray,
    w_down: jnp.ndarray,
    expert_idx: jnp.ndarray,
    *,
    activation=jax.nn.gelu,
) -> jnp.ndarray:
    """Two ragged_dot MoE FFN: up (D->F) then down (F->D).
    
    Args:
        x: (..., D)
        w_up: (G, D, F) expert up-projection weights
        w_down: (G, F, D) expert down-projection weights
        expert_idx: (...,) routed expert per token
    """
    n_experts = w_up.shape[0]
    orig_shape = x.shape
    sorted_x, group_sizes, _, inv_idx = route_tokens(x, expert_idx, n_experts)

    h = activation(lax.ragged_dot(sorted_x, w_up, group_sizes))
    y = lax.ragged_dot(h, w_down, group_sizes)
    return unroute_tokens(y, inv_idx, orig_shape)


def moe_ffn_loop(
    x: jnp.ndarray,
    w_up: jnp.ndarray,
    w_down: jnp.ndarray,
    expert_idx: jnp.ndarray,
    *,
    activation=jax.nn.gelu,
) -> jnp.ndarray:
    """Naive pre-expert loop baseline (matches original chinchilla/moe_model.py)."""
    out = jnp.zeros_like(x)
    for e in range(w_up.shape[0]):
        mask = (expert_idx == e)[..., None]
        h = activation(x @ w_up[e])
        out = out + mask.astype(x.dtype) * (h @ w_down[e])
    return out