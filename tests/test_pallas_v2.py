"""Correctness tests for the tiled Pallas fused-MoE kernel (interpret mode).

Run: uv run pytest tests/test_pallas_v2.py -q
Fast on CPU; the same code compiles for GPU with interpret=False.
"""

import jax
import jax.numpy as jnp
import pytest

from kernels.fused_moe_pallas_v2 import (
    moe_ffn_fused_pallas_tpu,
    moe_ffn_fused_pallas_v2,
)
from moe.ragged_moe import moe_ffn_ragged


def _make_inputs(m, d, f, g, dtype, seed=0):
    k1, k2, k3 = jax.random.split(jax.random.key(seed), 3)
    x = jax.random.normal(k1, (m, d), dtype=jnp.float32).astype(dtype)
    w_up = (jax.random.normal(k2, (g, d, f), dtype=jnp.float32) * 0.05).astype(dtype)
    w_down = (jax.random.normal(k3, (g, f, d), dtype=jnp.float32) * 0.05).astype(dtype)
    idx = jnp.tile(jnp.arange(g), m // g)  # balanced round-robin routing
    return x, w_up, w_down, idx


@pytest.mark.parametrize("kernel", [moe_ffn_fused_pallas_v2, moe_ffn_fused_pallas_tpu])
@pytest.mark.parametrize("dtype,atol", [(jnp.float32, 1e-4), (jnp.float16, 2e-2)])
@pytest.mark.parametrize("m,d,f,g,block_m,f_tile", [
    (512, 64, 256, 4, 64, 64),     # small
    (1024, 64, 1024, 8, 128, 128), # target-ish shape, F/D = 16
    (512, 128, 512, 4, 128, 256),  # f_tile > block_m
])
def test_matches_ragged_dot(m, d, f, g, block_m, f_tile, dtype, atol, kernel):
    x, w_up, w_down, idx = _make_inputs(m, d, f, g, dtype)
    y_ref = moe_ffn_ragged(
        x.astype(jnp.float32), w_up.astype(jnp.float32), w_down.astype(jnp.float32), idx
    )
    y = kernel(x, w_up, w_down, idx, block_m=block_m, f_tile=f_tile, interpret=True)
    err = float(jnp.max(jnp.abs(y.astype(jnp.float32) - y_ref)))
    assert err < atol, f"max abs err {err} >= {atol}"


def test_rejects_unbalanced_block():
    x, w_up, w_down, idx = _make_inputs(512, 64, 256, 4, jnp.float32)
    with pytest.raises(AssertionError):
        moe_ffn_fused_pallas_v2(
            x, w_up, w_down, idx, block_m=100, f_tile=64, interpret=True
        )
