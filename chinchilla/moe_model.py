"""MoE decode-only transformer with same active FFN width as dense baseline"""

from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn

from addition_transformer.model import CausalSelfAttension
from addition_transformer.vocab import MAX_SEQ_LEN, VOCAB_SIZE


class MoEFFN(nn.Module):
    """Top-1 routed MoE feed-forward block"""

    d_model: int
    d_ff: int
    n_experts: int

    @nn.compact
    def __call__(self, x):
        gate = nn.Dense(self.n_experts, name="gate")(x)
        idx = jnp.argmax(gate, axis=-1)
        out = jnp.zeros_like(x)
        for e in range(self.n_experts):
            mask = (idx == e)[..., None]
            h = nn.Dense(self.d_ff, name=f"fc1_{e}")(x)
            h = nn.gelu(h)
            h = nn.Dense(self.d_model, name=f"gc2_{e}")(h)
            out = out + mask.astype(x.dtype) * h
        return out
    

class MoETransformerBlock(nn.Module):
    n_heads: int
    d_model: int
    d_ff: int
    n_experts: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, *, train: bool, attn_mask):
        h = nn.LayerNorm(name="ln1")(x)
        h = CausalSelfAttension(self.n_heads, self.d_model, self.dropout_rate)(
            h, train=train, attn_mask=attn_mask
        )
        x = x + h
        h = nn.LayerNorm(name="ln2")(x)
        h = MoEFFN(self.d_model, self.d_ff, self.n_experts)(h)
        h = nn.Dropout(self.dropout_rate, deterministic=not train)(h)
        return x + h
    

class MoEAdditionTransformer(nn.Module):
    """MoE variant: `n_experts` FFNs, top-1 routing; active params ~ one expert FFN."""

    d_model: int = 320
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 1280
    n_experts: int = 4
    max_seq_len: int = MAX_SEQ_LEN
    vocab_size: int = VOCAB_SIZE
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, input_ids, *, train: bool = True):
        B, T = input_ids.shape
        tok_emb = nn.Embed(self.vocab_size, self.d_model, name="tok_embed")(input_ids)
        pos = jnp.arange(T)
        pos_emb = nn.Embed(self.max_seq_len, self.d_model, name="pos_embed")(pos)
        x = tok_emb + pos_emb
        x = nn.Dropout(self.dropout_rate, deterministic=not train)(x)

        causal = jnp.tril(jnp.ones((T, T), dtype=bool))[None, None, :, :]

        for i in range(self.n_layers):
            x = MoETransformerBlock(
                self.n_heads, 
                self.d_model, 
                self.d_ff, 
                self.n_experts,
                self.dropout_rate, 
                name=f"block_{i}"
            )(x, train=train, attn_mask=causal)

        x = nn.LayerNorm(name="ln_f")(x)
        return nn.Dense(self.vocab_size, use_bias=False, name='lm_head')(x)
