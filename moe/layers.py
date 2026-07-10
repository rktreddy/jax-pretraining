"""Flax MoE FFN layers (loop, ragged_dot, fused)."""

from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn

from kernels.fused_moe_updown import moe_ffn_fused
from moe.ragged_moe import moe_ffn_ragged


class RaggedMoEFFN(nn.Module):
    """MoE FFN using jax.lax.ragged_dot."""

    d_model: int
    d_ff: int
    n_experts: int

    @nn.compact
    def __call__(self, x):
        gate = nn.Dense(self.n_experts, name="gate")(x)
        expert_idx = jnp.argmax(gate, axis=-1)
        w_up = jnp.stack(
            [self.param(f"w_up_{e}", nn.initializers.lecun_normal(), (self.d_model, self.d_ff))
             for e in range(self.n_experts)]
        )
        w_down = jnp.stack(
            [self.param(f"w_down_{e}", nn.initializers.lecun_normal(), (self.d_ff, self.d_model))
             for e in range(self.n_experts)]
        )
        return moe_ffn_ragged(x, w_up, w_down, expert_idx)
    

class FusedMoEFFN(nn.Module):
    """MoE FFN with fused up/down projections (Pallas or JAX tiled)."""

    d_model: int
    d_ff: int
    n_experts: int
    f_tile: int = 512
    use_pallas: bool = True

    @nn.compact
    def __call__(self, x):
        gate = nn.Dense(self.n_experts, name="gate")(x)
        expert_idx = jnp.argmax(gate, axis=-1)
        w_up = jnp.stack(
            [self.param(f"w_up_{e}", nn.initializers.lecun_normal(), (self.d_model, self.d_ff))
             for e in range(self.n_experts)]
        )
        w_down = jnp.stack(
            [self.param(f"w_down_{e}", nn.initializers.lecun_normal(), (self.d_ff, self.d_model))
             for e in range(self.n_experts)]
        )
        return moe_ffn_fused(
            x, w_up, w_down, expert_idx,
            f_tile=self.f_tile,
            use_pallas=self.use_pallas,
            )




