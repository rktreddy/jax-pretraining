"""Charactor-level addition transformer (JAX + Flax + Optax)."""

from addition_transformer.data import (
    answer_accuracy,
    eval_batch,
    fixed_eval_pairs,
    format_example,
    generate_batch,
)
from addition_transformer.model import AdditionTransformer, count_params
from addition_transformer.checkpoint import (
    DEFAULT_CHECKPOINT_DIR,
    load_checkpoint,
    load_config,
    save_checkpoint,
)
from addition_transformer.train import TrainConfig, demo_prompts, generate_fixed, train
from addition_transformer.vocab import CHARS, MAX_SEQ_LEN, VOCAB_SIZE, decode, encode

__all__ = [
    "AdditionTransformer",
    "CHARS",
    "DEFAULT_CHECKPOINT_DIR",
    "MAX_SEQ_LEN",
    "TrainConfig",
    "VOCAB_SIZE",
    "count_params",
    "decode",
    "demo_prompts",
    "encode",
    "format_example",
    "generate_batch",
    "generate_fixed",
    "load_checkpoint",
    "load_config",
    "save_checkpoint",
    "train",
]