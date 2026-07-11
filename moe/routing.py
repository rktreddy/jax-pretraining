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


def route_tokens_block_padded(
    x: jnp.ndarray,
    expert_idx: jnp.ndarray,
    n_experts: int,
    block_m: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sort tokens by expert, padding each group to a block_m multiple.

    In the returned layout, every contiguous block_m-sized block of rows
    belongs to exactly ONE expert (the MegaBlocks block-alignment trick),
    so a kernel can process fixed-size token blocks against a single
    expert's weights even when routing is unbalanced.

    Args:
        x: (..., D) activations
        expert_idx: (...,) int expert id per token
        n_experts: number of experts
        block_m: kernel token-block size

    Returns:
        padded_x: (capacity, D) block-aligned layout; pad rows are zero.
            capacity is static: ceil(M / block_m) * block_m + n_experts * block_m.
        block_to_expert: (capacity // block_m,) int32; expert owning each
            block. Blocks past the last real group are clamped to
            n_experts - 1 and compute garbage that is never read back.
        dest: (M,) int32; padded_x row of each *original* (unsorted) token.
            Invert with y = padded_y[dest].reshape(orig_shape).
    """
    orig_shape = x.shape
    d = orig_shape[-1]
    flat_x = x.reshape(-1, d)
    flat_idx = expert_idx.reshape(-1).astype(jnp.int32)
    m = flat_x.shape[0]

    group_sizes = jnp.bincount(flat_idx, length=n_experts).astype(jnp.int32)
    padded_sizes = ((group_sizes + block_m - 1) // block_m) * block_m
    group_starts = jnp.concatenate(
        [jnp.zeros(1, jnp.int32), jnp.cumsum(group_sizes)[:-1]]
    )
    padded_starts = jnp.concatenate(
        [jnp.zeros(1, jnp.int32), jnp.cumsum(padded_sizes)[:-1]]
    )

    # Where each original token lands in the padded layout: its group's
    # padded start plus its rank within the group (stable sort order).
    sort_idx = jnp.argsort(flat_idx, stable=True)
    rank_of_sorted = jnp.arange(m, dtype=jnp.int32) - group_starts[flat_idx[sort_idx]]
    dest_of_sorted = padded_starts[flat_idx[sort_idx]] + rank_of_sorted
    dest = jnp.zeros(m, jnp.int32).at[sort_idx].set(dest_of_sorted)

    capacity = ((m + block_m - 1) // block_m) * block_m + n_experts * block_m
    padded_x = jnp.zeros((capacity, d), flat_x.dtype).at[dest].set(flat_x)

    # Expert owning each block: count how many padded groups end at or
    # before the block's start row.
    block_starts = jnp.arange(capacity // block_m, dtype=jnp.int32) * block_m
    ends = jnp.cumsum(padded_sizes)
    block_to_expert = jnp.searchsorted(ends, block_starts, side="right").astype(
        jnp.int32
    )
    block_to_expert = jnp.minimum(block_to_expert, n_experts - 1)
    return padded_x, block_to_expert, dest

