"""!10M-parameter decoder only transformer for character-level addition."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn

from addition_transformer.vocab import MAX_SEQ_LEN, VOCAB_SIZE

def count_params(params) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(params))

class CausalSelfAttension(nn.Module):
    nn_heads: int
    d_model: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, *, train: bool, attn_mask):
        B, T, C = x.shape
        head_dim = C // self.nn_heads
        qkv = nn.Dense(3 * C, use_bias=False, name="qkv")(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(B, T, self.nn_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.nn_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.nn_heads, head_dim).transpose(0, 2, 1, 3)

        scale = head_dim**-0.5
        attn = (q @ k.swapaxes(-2, -1)) * scale
        attn = jnp.where(attn_mask, attn, jnp.finfo(attn.dtype).min)
        attn = nn.softmax(attn, axis=-1)
        attn = nn.Dropout(self.dropout_rate, deterministic=not train)(attn)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return nn.Dense(C, use_bias=False, name="out")(out)
    

class TransformerBlock(nn.Module):
    n_heads: int
    d_model: int
    d_ff: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, *, train: bool, attn_mask):
        h = nn.LayerNorm(name="ln1")(x)
        h = CausalSelfAttension(self.n_heads, self.d_model, self.dropout_rate)(
            h, train=train, attn_mask=attn_mask
        )
        x = x + h
        h = nn.LayerNorm(name="ln2")(x)
        h = nn.Dense(self.d_ff, name="fc1")(h)
        h = nn.gelu(h)
        h = nn.Dropout(self.dropout_rate, deterministic=not train)(h)
        h = nn.Dense(self.d_model, name="fc2")(h)
        h = nn.Dropout(self.dropout_rate, deterministic=not train)(h)
        return x + h
    

class AdditionTransformer(nn.Module):
    """Decoder-only GPT-style model (~10M params with default config)"""

    d_model: int = 320
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 1280
    max_seq_len: int = MAX_SEQ_LEN
    vocab_size: int = VOCAB_SIZE
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, input_ids, *, train: bool = True):
        B, T = input_ids.shape
        tok_emb = nn.Embed(self.vocab_size, self.d_model, name="tok_embed")(input_ids)
        pos = jnp.arange(T)
        pos_emb = nn.Embed(self.max_seq_len, self.d_model, name="pos_embed")(pos)
        x = tok_emb + pos_emb
        x = nn.Dropout(self.dropout_rate, deterministic=not train)(x)

        causal = jnp.tril(jnp.ones((T, T), dtype=bool))
        causal = causal[None, None, :, :]

        for i in range(self.n_layers):
            x = TransformerBlock(
                self.n_heads, self.d_model, self.d_ff, self.dropout_rate, name=f"block_{i}"
            )(x, train=train, attn_mask=causal)

        x = nn.LayerNorm(name="ln_f")(x)
        logits = nn.Dense(self.vocab_size, use_bias=False, name='lm_head')(x)
        return logits