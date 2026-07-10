"""Pallas kernel fusing MoE up/down projections when F > D."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental.pallas import pallas_call

from moe.routing import route_tokens, unroute_tokens


def fused_moe_pallas_sorted(
    sorted_x: jnp.ndarray,
    w_up: jnp.ndarray,
    w_down: jnp.ndarray,
    *,
    f_tile: int = 256,
    interpret: bool = True,
) -> jnp.ndarray:
    """Pallas fused MoE on equal-sized expert groups."""
    n_experts = w_up.shape[0]
    per = sorted_x.shape[0] // n_experts
    d = sorted_x.shape[1]
    xg = sorted_x.reshape(n_experts, per, d)

    def kernel(x_ref, wu_ref, wd_ref, y_ref):
        x = x_ref[:]
        wu = wu_ref[:]
        wd = wd_ref[:]
        f = wu.shape[1]
        out_d = wd.shape[1]
        y_acc = jnp.zeros((x.shape[0], out_d), dtype=x.dtype)
        n_tiles = (f + f_tile - 1) // f_tile
        for i in range(n_tiles):
            f0 = i * f_tile
            f1 = min(f0 + f_tile, f)
            h = jax.nn.gelu(x @ wu[:, f0:f1])
            y_acc = y_acc + h @ wd[f0:f1, :]
        y_ref[:] = y_acc

    vmapped = jax.vmap(
        lambda xe, wu, wd: pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((per, d), sorted_x.dtype),
            grid=(1,),
            interpret=interpret,
        )(xe, wu, wd),
        in_axes=(0, 0, 0),
    )
    yg = vmapped(xg, w_up, w_down)
    return yg.reshape(-1, d)


def moe_ffn_fused_pallas(
    x: jnp.ndarray,
    w_up: jnp.ndarray,
    w_down: jnp.ndarray,
    expert_idx: jnp.ndarray,
    *,
    f_tile: int = 256,
    interpret: bool = True,
) -> jnp.ndarray:
    """Full MoE path: route -> Pallas fused FFN -> unroute."""
    n_experts = w_up.shape[0]
    orig_shape = x.shape
    sorted_x, group_sizes, _, inv_idx = route_tokens(x, expert_idx, n_experts)
    out = fused_moe_pallas_sorted(
        sorted_x, w_up, w_down, f_tile=f_tile, interpret=interpret
    )
    return unroute_tokens(out, inv_idx, orig_shape)


# Fallback: use JAX tiled fusion when pallas unavailable
def moe_ffn_fused(
    x: jnp.ndarray,
    w_up: jnp.ndarray,
    w_down: jnp.ndarray,
    expert_idx: jnp.ndarray,
    *,
    f_tile: int = 256,
    use_pallas: bool = True,
    interpret: bool = True,
) -> jnp.ndarray:
    if use_pallas:
        return moe_ffn_fused_pallas(
            x, w_up, w_down, expert_idx, f_tile=f_tile, interpret=interpret
        )
    from kernels.fused_moe_reference import moe_ffn_fused_tiled

    return moe_ffn_fused_tiled(x, w_up, w_down, expert_idx, f_tile=f_tile)


