"""Fused MoE up/down FFN - avoids materializing full (M, F) when F > D."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from moe.routing import route_tokens, unroute_tokens


def fused_ffn_single_expert(
    x: jnp.ndarray,
    w_up: jnp.ndarray,
    w_down: jnp.ndarray,
    f_tile: int = 256,
) -> jnp.ndarray:
    """Fused up-gelu-down for one expert with F-axis tiling."""
    f = w_up.shape[1]
    y = jnp.zeros((x.shape[0], w_down.shape[1]), dtype=x.dtype)
    n_tiles = (f + f_tile - 1) // f_tile
    for i in range(n_tiles):
        f0 = i * f_tile
        f1 = min(f0 + f_tile, f)
        h = jax.nn.gelu(x @ w_up[:, f0:f1])
        y = y + h @ w_down[f0:f1, :]
    return y


def moe_ffn_fused_sorted(
    sorted_x: jnp.ndarray,
    w_up: jnp.ndarray,
    w_down: jnp.ndarray,
    group_sizes: jnp.ndarray,
    *,
    f_tile: int = 256,
) -> jnp.ndarray:
    """Fused MoE on pre-sorted tokens. Requires equal group sizes (JIT-friendly)."""
    g = w_up.shape[0]
    m, d = sorted_x.shape
    per = m // g
    xg = sorted_x.reshape(g, per, d)
    yg = jax.vmap(
        lambda xe, wu, wd: fused_ffn_single_expert(xe, wu, wd, f_tile=f_tile)
    )(xg, w_up, w_down)
    return yg.reshape(m, d)


def moe_ffn_fused_tiled(
    x: jnp.ndarray,
    w_up: jnp.ndarray,
    w_down: jnp.ndarray,
    expert_idx: jnp.ndarray,
    *,
    f_tile: int = 256,
) -> jnp.ndarray:
    """Route tokens, apply fused FFN, restore order."""
    n_experts = w_up.shape[0]
    orig_shape = x.shape
    sorted_x, group_sizes, _, inv_idx = route_tokens(x, expert_idx, n_experts)
    out = moe_ffn_fused_sorted(sorted_x, w_up, w_down, group_sizes, f_tile=f_tile)
    return unroute_tokens(out, inv_idx, orig_shape)


