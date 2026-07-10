"""Sythetic addition dataset: up to 3-digit operands, fixed-length padding."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from addition_transformer.vocab import MAX_SEQ_LEN, PAD_ID, encode

def format_example(a: int, b: int) -> str:
    return f"{a} + {b} = {a + b}"


def answer_start_index(text: str) -> int:
    """Index in full sequence of the first answer character."""
    return text.index("= ") + 2


def answer_label_mask(text: str) -> int:
    """True for label positions that predict answer digits and terminating PAD."""
    start = answer_start_index(text)
    ids = encode(text)
    first_pad = len(text)
    mask = []
    for i in range(MAX_SEQ_LEN):
        target_idx = i + 1
        if target_idx >= MAX_SEQ_LEN:
            mask.append(False)
            continue
        if target_idx < start:
            mask.append(False)
        elif target_idx < first_pad:
            mask.append(True)
        elif target_idx == first_pad:
            mask.append(True)  # learn to emit PAD after the answer
        else:
            mask.append(False)
    return mask


def shift_for_causal_lm(ids: list[int]) -> tuple[list[int], list[int]]:
    """Build (input, label) pairs for next-token prediction."""
    inp = ids[:-1] + [PAD_ID]
    lab = ids[1:] + [PAD_ID]
    return inp, lab


def generate_batch(
        key: jax.Array,
        batch_size: int,
        max_operand: int = 999,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sample a batch of addition examples.
    
    Returns:
        input_ids: (batch, seq_len)
        labels: (batch, seq_len) next-token targets
        answer_mask: (batch, seq_len) True where answer-digit loss applies
    """
    key_a, key_b = jax.random.split(key)
    a = jax.random.randint(key_a, (batch_size), 0, max_operand + 1)
    b = jax.random.randint(key_b, (batch_size), 0, max_operand + 1)

    input_ids = []
    labels = []
    answer_masks = []
    for ai, bi in zip(np.asarray(a), np.asarray(b)):
        text = format_example(int(ai), int(bi))
        ids = encode(text)
        inp, lab = shift_for_causal_lm(ids)
        input_ids.append(inp)
        labels.append(lab)
        answer_masks.append(answer_label_mask(text))

    return(
        jnp.array(input_ids, dtype=jnp.int32), jnp.array(labels, dtype=jnp.int32), jnp.array(answer_masks, dtype=bool)
    )


def make_data_iterator(
    key: jax.Array,
    batch_size: int, 
    max_operand: int = 999,
):
    """Infinite iterator of training batches."""
    while True:
        key, subkey = jax.random.split(key)
        yield generate_batch(subkey, batch_size, max_operand)


def masked_accuracy(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    preds = jnp.argmax(logits, axis=-1)
    active = mask & (labels != PAD_ID)
    correct = (preds == labels) & active
    return jnp.sum(correct) / jnp.maximum(jnp.sum(active), 1)


def token_accuracy(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Fraction of correct next-token predictions on non-PAD positions."""
    mask = labels != PAD_ID
    return masked_accuracy(logits, labels, mask)


def answer_accuracy(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    answer_mask: jnp.ndarray,
) -> jnp.ndarray:
    return masked_accuracy(logits, labels, answer_mask)


def eval_batch(
    key: jax.Array,
    n_examples: int = 512,
    max_operand: int = 999,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Fixed evaluation batch for loss/accuracy metrics"""
    return generate_batch(key, n_examples, max_operand)


def fixed_eval_pairs(key: jax.Array, n: int = 128) -> list[tuple[int, int]]:
    """Reproducible (a, b) pairs for greedy addition evaluation"""
    key_a, key_b = jax.random.split(key)
    a = jax.random.randint(key_a, (n,), 0, 1000)
    b = jax.random.randint(key_b, (n,), 0, 1000)
    return [(int(ai), int(bi)) for ai, bi in zip(np.asarray(a), np.asarray(b))]