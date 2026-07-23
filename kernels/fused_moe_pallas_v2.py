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

from moe.routing import route_tokens, unroute_tokens, route_tokens_block_padded


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



def _fused_kernel_ragged(x_ref, bte_ref, wu_ref, wd_ref, y_ref, *, f_tile: int, d_ff: int):
    """One (expert, token-block) program.

    Refs:
        x_ref:  (1, bm, D) block in SRAM (BlockSpec-staged)
        wu_ref: (G, D, F) full array in HBM - load tiles manually
        wd_ref: (G, F, D) full array in HBM - load tiles manually
        y_ref:  (1, bm, D) output block in SRAM
    """
    e = pl.load(bte_ref, (pl.program_id(0),))  # which expert this program serves

    x = x_ref[...]  # (bm, D) - staged into SRAM by the BlockSpec
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

    y_ref[...] = acc.astype(y_ref.dtype)



def moe_ffn_fused_ragged_v2(x, w_up, w_down, expert_idx, *, block_m=128, f_tile=128, interpret=False):
    """Fused MoE FFN for unbalanced (ragged) routing GPU-style."""
    n_experts, d_model, d_ff = w_up.shape
    orig_shape = x.shape
    padded_x, block_to_expert, dest = route_tokens_block_padded(
        x.reshape(-1, x.shape[-1]), expert_idx, n_experts, block_m
    )
    n_blocks = padded_x.shape[0] // block_m
    kernel = functools.partial(_fused_kernel_ragged, f_tile=f_tile,d_ff=d_ff)
    padded_y = pl.pallas_call(
        kernel,
        grid=(n_blocks,),
        in_specs=[
            pl.BlockSpec((block_m, d_model), lambda i: (i, 0)),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=pl.BlockSpec((block_m, d_model), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct(padded_x.shape, x.dtype),
        interpret=interpret,
    )(padded_x, block_to_expert, w_up, w_down)
    
    return padded_y[dest].reshape(orig_shape)


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


# ---------------------------------------------------------------------------
# TPU variant.
#
# On GPU (Triton), grid programs are parallel CTAs, so the F loop must live
# inside the kernel. On TPU (Mosaic), grid steps run *sequentially*, so F can
# be a third grid axis: the same output block is visited F//f_tile times and
# a scratch accumulator carries the partial sum between visits. Same math,
# opposite orchestration - the hardware execution model shapes the kernel.
# ---------------------------------------------------------------------------


def _fused_kernel_tpu(x_ref, wu_ref, wd_ref, y_ref, acc_ref):
    """One (expert, token-block, F-tile) grid step.

    Refs (all BlockSpec-staged into VMEM):
        x_ref:   (1, bm, D)   - same block for every j
        wu_ref:  (1, D, bf)   - j-th F tile of this expert's up projection
        wd_ref:  (1, bf, D)   - j-th F tile of the down projection
        y_ref:   (1, bm, D)   - same output block for every j
        acc_ref: (bm, D) fp32 scratch, persists across sequential grid steps
    """
    j = pl.program_id(2)

    @pl.when(j == 0)
    def _init():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    x = x_ref[0]
    wu = wu_ref[0]
    wd = wd_ref[0]
    h = jax.nn.gelu(jnp.dot(x, wu, preferred_element_type=jnp.float32))
    h = h.astype(x.dtype)
    acc_ref[...] += jnp.dot(h, wd, preferred_element_type=jnp.float32)

    @pl.when(j == pl.num_programs(2) - 1)
    def _store():
        y_ref[0] = acc_ref[...].astype(y_ref.dtype)


def fused_moe_pallas_tpu_sorted(
    sorted_x: jnp.ndarray,
    w_up: jnp.ndarray,
    w_down: jnp.ndarray,
    *,
    block_m: int = 128,
    f_tile: int = 128,
    interpret: bool = False,
) -> jnp.ndarray:
    """TPU (Mosaic) fused MoE FFN on pre-sorted, balanced expert groups."""
    from jax.experimental.pallas import tpu as pltpu

    n_experts, d_model, d_ff = w_up.shape
    m = sorted_x.shape[0]
    per = m // n_experts
    assert m % n_experts == 0, "balanced groups required"
    assert per % block_m == 0, f"per-expert tokens {per} % block_m {block_m} != 0"
    assert d_ff % f_tile == 0, f"d_ff {d_ff} % f_tile {f_tile} != 0"

    xg = sorted_x.reshape(n_experts, per, d_model)

    yg = pl.pallas_call(
        _fused_kernel_tpu,
        grid=(n_experts, per // block_m, d_ff // f_tile),
        in_specs=[
            pl.BlockSpec((1, block_m, d_model), lambda e, i, j: (e, i, 0)),
            pl.BlockSpec((1, d_model, f_tile), lambda e, i, j: (e, 0, j)),
            pl.BlockSpec((1, f_tile, d_model), lambda e, i, j: (e, j, 0)),
        ],
        out_specs=pl.BlockSpec((1, block_m, d_model), lambda e, i, j: (e, i, 0)),
        out_shape=jax.ShapeDtypeStruct((n_experts, per, d_model), sorted_x.dtype),
        scratch_shapes=[pltpu.VMEM((block_m, d_model), jnp.float32)],
        interpret=interpret,
    )(xg, w_up, w_down)
    return yg.reshape(m, d_model)


def moe_ffn_fused_pallas_tpu(
    x: jnp.ndarray,
    w_up: jnp.ndarray,
    w_down: jnp.ndarray,
    expert_idx: jnp.ndarray,
    *,
    block_m: int = 128,
    f_tile: int = 128,
    interpret: bool = False,
) -> jnp.ndarray:
    """Full TPU path: route -> fused Pallas FFN -> unroute."""
    n_experts = w_up.shape[0]
    orig_shape = x.shape
    sorted_x, _, _, inv_idx = route_tokens(x, expert_idx, n_experts)
    out = fused_moe_pallas_tpu_sorted(
        sorted_x, w_up, w_down,
        block_m=block_m, f_tile=f_tile, interpret=interpret,
    )
    return unroute_tokens(out, inv_idx, orig_shape)
