"""Tests for block-padded routing and the ragged (unbalanced) fused kernel.

Run: uv run pytest tests/test_ragged_fused.py -q

The routing tests pass already. The kernel tests are RED until
moe_ffn_fused_ragged_{v2,tpu} exist in kernels/fused_moe_pallas_v2.py -
that implementation is the exercise.
"""

import jax
import jax.numpy as jnp
import pytest

from moe.ragged_moe import moe_ffn_loop
from moe.routing import route_tokens_block_padded


def _random_routing(m, g, seed=0, skew=False):
    key = jax.random.key(seed)
    if skew:
        # expert 0 gets ~70% of tokens - stress imbalance
        p = jnp.array([0.7] + [0.3 / (g - 1)] * (g - 1))
        return jax.random.choice(key, g, (m,), p=p)
    return jax.random.randint(key, (m,), 0, g)


@pytest.mark.parametrize("skew", [False, True])
@pytest.mark.parametrize("m,g,block_m", [(500, 4, 64), (1024, 8, 128), (37, 3, 16)])
def test_block_padded_layout_invariants(m, g, block_m, skew):
    key = jax.random.key(1)
    x = jax.random.normal(key, (m, 8))
    idx = _random_routing(m, g, skew=skew)
    padded_x, block_to_expert, dest = route_tokens_block_padded(x, idx, g, block_m)

    # round trip: gathering dest rows recovers the original tokens
    assert jnp.allclose(padded_x[dest], x)
    # every real token's block is owned by that token's expert
    token_block = dest // block_m
    assert jnp.all(block_to_expert[token_block] == idx)
    # capacity is a block multiple and map covers it
    assert padded_x.shape[0] % block_m == 0
    assert block_to_expert.shape[0] == padded_x.shape[0] // block_m
    # pad rows are zero (all rows not hit by dest)
    mask = jnp.zeros(padded_x.shape[0], bool).at[dest].set(True)
    assert jnp.all(padded_x[~mask] == 0)


def _kernel_inputs(m, d, f, g, dtype, skew, seed=0):
    k1, k2, k3 = jax.random.split(jax.random.key(seed), 3)
    x = jax.random.normal(k1, (m, d), dtype=jnp.float32).astype(dtype)
    w_up = (jax.random.normal(k2, (g, d, f)) * 0.05).astype(dtype)
    w_down = (jax.random.normal(k3, (g, f, d)) * 0.05).astype(dtype)
    idx = _random_routing(m, g, skew=skew)
    return x, w_up, w_down, idx


@pytest.mark.parametrize("variant", ["v2", "tpu"])
@pytest.mark.parametrize("skew", [False, True])
@pytest.mark.parametrize("dtype,atol", [(jnp.float32, 1e-4), (jnp.float16, 2e-2)])
def test_ragged_fused_matches_loop(variant, skew, dtype, atol):
    from kernels import fused_moe_pallas_v2 as k

    fn = getattr(k, f"moe_ffn_fused_ragged_{variant}", None)
    assert fn is not None, f"moe_ffn_fused_ragged_{variant} not implemented yet"

    m, d, f, g = 777, 64, 256, 4  # deliberately not a block multiple
    x, w_up, w_down, idx = _kernel_inputs(m, d, f, g, dtype, skew)
    y_ref = moe_ffn_loop(
        x.astype(jnp.float32), w_up.astype(jnp.float32), w_down.astype(jnp.float32), idx
    )
    y = fn(x, w_up, w_down, idx, block_m=64, f_tile=64, interpret=True)
    err = float(jnp.max(jnp.abs(y.astype(jnp.float32) - y_ref)))
    assert err < atol, f"max abs err {err} >= {atol}"
