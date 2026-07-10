"""Tiled Pallas kernel: fused MoE up/down projections (GPU-ready).

Design (per program, grid = (G, per // bm)):
  - X block (bm, D) and Y block (bm, D) auto-staged via BlockSpec.
  - Wu (G, D, F) and Wd (G, F, D) stay in HBM (memory_space=ANY);
    (D, bf) / (bf, D) tiles are loaded manually inside an F loop, so the
    (bm, F) intermediate never exists anywhere but SRAM.
  - fp32 accumulation over fp16/bf16 inputs (preferred_element_type).

Requires balanced expert groups: sorted_x.shape[0] % n_experts == 0.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from moe.routing import route_tokens, unroute_tokens


def _fused_kernel(x_ref, wu_ref, wd_ref, y_ref, *, f_tile: int, d_ff: int):
    """One (expert, token-block) program.

    Refs:
        x_ref:  (1, bm, D) block in SRAM (BlockSpec-staged)
        wu_ref: (G, D, F) full array in HBM - load tiles manually
        wd_ref: (G, F, D) full array in HBM - load tiles manually
        y_ref:  (1, bm, D) output block in SRAM
    """
    e = pl.program_id(0)  # which expert this program serves

    x = x_ref[0]  # (bm, D) - staged into SRAM by the BlockSpec
    acc = jnp.zeros((x.shape[0], x.shape[1]), dtype=jnp.float32)

    for j in range(d_ff // f_tile):  # static bound -> unrolled at trace time
        # (D, f_tile) slice of this expert's up-projection, HBM -> SRAM
        wu = pl.load(wu_ref, (e, slice(None), pl.dslice(j * f_tile, f_tile)))
        # (bm, f_tile) sliver of the hidden activation. This is the fusion:
        # h lives only in SRAM/registers and dies at the end of the iteration -
        # the full (bm, F) intermediate never exists.
        h = jax.nn.gelu(jnp.dot(x, wu, preferred_element_type=jnp.float32))
        h = h.astype(x.dtype)  # back to input dtype so the MMA units engage
        # (f_tile, D) slice of the down-projection
        wd = pl.load(wd_ref, (e, pl.dslice(j * f_tile, f_tile), slice(None)))
        # partial contraction over this F tile; fp32 accumulation
        acc = acc + jnp.dot(h, wd, preferred_element_type=jnp.float32)

    y_ref[0] = acc.astype(y_ref.dtype)


def fused_moe_pallas_v2_sorted(
    sorted_x: jnp.ndarray,
    w_up: jnp.ndarray,
    w_down: jnp.ndarray,
    *,
    block_m: int = 128,
    f_tile: int = 128,
    interpret: bool = False,
) -> jnp.ndarray:
    """Fused MoE FFN on pre-sorted, balanced expert groups.

    Args:
        sorted_x: (M, D) tokens grouped by expert, M % G == 0
        w_up: (G, D, F)
        w_down: (G, F, D)
    """
    n_experts, d_model, d_ff = w_up.shape
    m = sorted_x.shape[0]
    per = m // n_experts
    assert m % n_experts == 0, "balanced groups required"
    assert per % block_m == 0, f"per-expert tokens {per} % block_m {block_m} != 0"
    assert d_ff % f_tile == 0, f"d_ff {d_ff} % f_tile {f_tile} != 0"

    xg = sorted_x.reshape(n_experts, per, d_model)

    kernel = functools.partial(_fused_kernel, f_tile=f_tile, d_ff=d_ff)
    yg = pl.pallas_call(
        kernel,
        grid=(n_experts, per // block_m),
        in_specs=[
            pl.BlockSpec((1, block_m, d_model), lambda e, i: (e, i, 0)),
            pl.BlockSpec(memory_space=pl.ANY),  # w_up stays in HBM
            pl.BlockSpec(memory_space=pl.ANY),  # w_down stays in HBM
        ],
        out_specs=pl.BlockSpec((1, block_m, d_model), lambda e, i: (e, i, 0)),
        out_shape=jax.ShapeDtypeStruct((n_experts, per, d_model), sorted_x.dtype),
        interpret=interpret,
    )(xg, w_up, w_down)
    return yg.reshape(m, d_model)


def moe_ffn_fused_pallas_v2(
    x: jnp.ndarray,
    w_up: jnp.ndarray,
    w_down: jnp.ndarray,
    expert_idx: jnp.ndarray,
    *,
    block_m: int = 128,
    f_tile: int = 128,
    interpret: bool = False,
) -> jnp.ndarray:
    """Full path: route -> fused Pallas FFN -> unroute."""
    n_experts = w_up.shape[0]
    orig_shape = x.shape
    sorted_x, _, _, inv_idx = route_tokens(x, expert_idx, n_experts)
    out = fused_moe_pallas_v2_sorted(
        sorted_x, w_up, w_down,
        block_m=block_m, f_tile=f_tile, interpret=interpret,
    )
    return unroute_tokens(out, inv_idx, orig_shape)
