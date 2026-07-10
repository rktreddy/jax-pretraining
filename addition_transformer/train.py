"""Training loop, loss, and generation for the addition transformer."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from addition_transformer.data import (
    answer_accuracy,
    eval_batch,
    fixed_eval_pairs,
    format_example,
    generate_batch,
    make_data_iterator,
    token_accuracy
)
from addition_transformer.model import AdditionTransformer, count_params
from addition_transformer.vocab import MAX_ANSWER_DIGITS, MAX_SEQ_LEN, PAD_ID, VOCAB_SIZE, decode, encode


@dataclass
class TrainConfig:
    d_model: int = 320
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 1280
    dropout_rate: float = 0.0
    learnin_rate: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 512
    max_steps: int = 15000
    eval_every: int = 500
    log_every: int = 100
    seed: int = 42
    max_operand: int = 999
    curriculum_stage1_steps: int = 5000 # 1-digit 0-9
    curriculum_stage2_steps: int = 12000 # 2-digit 0-99
    checkpoint_dir: str = "checkpoints/addition_transformer"
    save_every: int | None = None # save every N steps during eval; always save at end
    architecture: str = "dense" # "dense | moe"
    n_experts: int = 4
    save_checkpoint_at_end: bool = True


class TrainState(train_state.TrainState):
    pass


def max_operand_for_step(step: int, config: TrainConfig) -> int:
    if step <= config.curriculum_stage1_steps:
        return 9
    if step <= config.curriculum_stage2_steps:
        return 99
    return config.max_operand


def cross_entropy_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    loss_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Masked cross-entropy; only positions where loss_mask is True"""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(labels, VOCAB_SIZE)
    mask = loss_mask.astype(jnp.float32)
    loss = -jnp.sum(log_probs * one_hot, axis=-1) * mask
    return jnp.sum(loss) / jnp.maximum(jnp.sum(mask), 1.0)


def _build_model(config: TrainConfig):
    if config.architecture == 'moe':
        from chinchilla.moe_model import MoEAdditionTransformer

        return MoEAdditionTransformer(
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            n_experts=config.n_experts,
            dropout_rate=config.dropout_rate,
        )
    return AdditionTransformer(
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        dropout_rate=config.dropout_rate,
    )


def make_train_state(key: jax.Array, config: TrainConfig, *, verbose: bool = True) -> TrainState:
    model = _build_model(config)
    dummy = jnp.zeros((1, MAX_SEQ_LEN), dtype=jnp.int32)
    init_key, dropout_key = jax.random.split(key)
    variables = model.init({"params": init_key, "dropout": dropout_key}, dummy, train=True)
    params = variables["params"]
    n_params = count_params(params)
    if verbose:
        print(f"Model parameters: {n_params,}({n_params / 1e6:.2f}M)")

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(config.learnin_rate, weight_decay=config.weight_decay)
    )
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(
    state: TrainState,
    input_ids: jnp.ndarray,
    labels: jnp.ndarray,
    answer_mask: jnp.ndarray,
    rng: jax.Array,
):
    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params}, input_ids, train=True, rngs={"dropout": rng}
        )
        loss = cross_entropy_loss(logits, labels, answer_mask)
        return loss, logits
    
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    ans_acc = answer_accuracy(logits, labels, answer_mask)
    return state, loss, ans_acc


@jax.jit
def eval_step(
    state: TrainState,
    input_ids: jnp.ndarray,
    labels: jnp.ndarray,
    answer_mask: jnp.ndarray,
):
    logits = state.apply_fn({"params": state.params}, input_ids, train=False)
    loss = cross_entropy_loss(logits, labels, answer_mask)
    ans_acc = answer_accuracy(logits, labels, answer_mask)
    tok_acc = token_accuracy(logits, labels)
    return loss, ans_acc, tok_acc


def generate_fixed(
        state: TrainState,
        prompt: str,
        max_new_tokens: int = MAX_ANSWER_DIGITS,
) -> str:
    """Greedy decode: extend 'prompt' with up to 'max_new_tokens' answer characters."""
    ids = encode(prompt)
    start = len(prompt)
    for _ in range(max_new_tokens):
        pos = start + _
        if pos >= MAX_SEQ_LEN:
            break
        x = jnp.array([ids], dtype=jnp.int32)
        logits = state.apply_fn({"params": state.params}, x, train=False)
        next_id = int(jnp.argmax(logits[0, pos - 1]))
        ids[pos] = next_id
        if next_id == PAD_ID:
            break
    return decode(ids)


def evaluate_addition(state: TrainState, pairs: list[tuple[int, int]]) -> float:
    """Greede-decode from 'a + b = ' and check the full expression."""
    if not pairs:
        return 0.0
    correct = 0.0
    for a, b in pairs:
        expected = format_example(a, b)
        prompt = f"{a} + {b} = "
        pred = generate_fixed(state, prompt)
        if pred == expected:
            correct += 1
    return correct / len(pairs)


def train(config: TrainConfig | None = None, *, quiet: bool = False) -> TrainState:
    state, _ = run_training(config, quiet=quiet)
    return state


def run_training(
    config: TrainConfig | None = None,
    *,
    quiet: bool = False,
) -> tuple[TrainState, dict]:
    """Train and return (state, metrics dict)."""
    config = config or TrainConfig()
    key = jax.random.key(config.seed)
    key, init_key, eval_key, pairs_key = jax.random.split(key, 4)

    state = make_train_state(init_key, config, verbose=not quiet)
    eval_ids, eval_labels, eval_answer_mask = eval_batch(eval_key, max_operand=999)
    eval_pairs = fixed_eval_pairs(pairs_key, n=64)
    current_max_op = -1
    token_count = 0.0
    final_loss = 0.0

    for step in range(1, config.max_steps + 1):
        max_op = max_operand_for_step(step, config)
        if max_op != current_max_op:
            if not quiet:
                print(f" curriculum | max_operans={max_op}")
            current_max_op = max_op

        key, batch_key, step_key = jax.random.split(key, 3)
        mix_key, batch_key = jax.random.split(batch_key)
        sample_op = 9 if float(jax.random.uniform(mix_key)) < 0.25 else max_op
        input_ids, labels, answer_mask = generate_batch(
            batch_key, config.batch_size, sample_op
        )
        state, loss, ans_acc = train_step(state, input_ids, labels, answer_mask, step_key)
        token_count += float(jnp.sum(answer_mask))
        final_loss = float(loss)

        if not quiet and step % config.log_every == 0:
            print(
                f"step {step:5d} | loss {float(loss):.4f}"
                f" | answer_acc {float(ans_acc):.3f} | max_op {max_op}"
            )

        if step % config.eval_every == 0 or step == config.max_steps:
            eval_loss, eval_ans_acc, _ = eval_step(
                state, eval_ids, eval_labels, eval_answer_mask
            )
            add_acc = evaluate_addition(state, eval_pairs)
            if step == config.max_steps:
                final_loss = float(eval_loss)
            if not quiet:
                print(
                    f" eval | loss {float(eval_loss):.4f}"
                    f" | answer_tok_acc {float(eval_ans_acc):.3f}"
                    f" | addition_acc {add_acc:.3f}"
                )
                for a, b in [(7, 8), (48, 58), (123, 456), (999, 999)]:
                    p = f"{a} + {b} = "
                    out = generate_fixed(state, p)
                    exp = format_example(a, b)
                    mark = "OK" if out == exp else "FAIL"
                    print(f"  {mark}  {out!r}  (expected {exp!r})")
            
            if config.save_every and step % config.save_every == 0:
                from addition_transformer.checkpoint import save_checkpoint

                save_checkpoint(state, config, config.checkpoint_dir, step=step)


    if config.save_checkpoint_at_end:
        from addition_transformer.checkpoint import save_checkpoint

        save_checkpoint(state, config, config.checkpoint_dir, step=config.max_steps)


    from addition_transformer.model import count_params as _count
    from chinchilla.metrics import count_active_params, estimate_flops

    n_params = _count(state.params)
    n_active = count_active_params(
        n_params,
        architecture=config.architecture,
        n_layers=config.n_layers,
        d_model=config.d_model,
        d_ff=config.d_ff,
        n_experts=config.n_experts,
    )
    tokens = int(token_count)
    add_acc = evaluate_addition(state, eval_pairs)
    metrics = {
        "architecture": config.architecture,
        "n_params": n_params,
        "n_active_params": n_active,
        "tokens": tokens,
        "steps": config.max_steps,
        "batch_size": config.batch_size,
        "final_loss": final_loss,
        "addition_acc": add_acc,
        "flops": estimate_flops(n_active, tokens),
        "d_model": config.d_model,
        "n_layers": config.n_layers,
        "n_experts": config.n_experts,
    }
    return state, metrics


def demo_prompts(state: TrainState) -> None:
    """Print greedy predictions on a few fixed addition prompts."""
    for a, b in [(7, 8), (42, 58), (123, 456), (999, 999)]:
        p = f"{a} + {b} = "
        out = generate_fixed(state, p)
        exp = format_example(a, b)
        mark = "OK" if out == exp else "FAIL"
        print(f"{mark} {out!r} (expected {exp!r})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or run the addition transformer.")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("train", help="Train from scratch (default)")

    infer_p = sub.add_parser("infer", help="Load checkpoint and run demo prompts")
    infer_p.add_argument(
        "--checkpoint",
        default="checkpoints/addition_transformer",
        help='Checkpoint directory',
    )

    args = parser.parse_args()
    if args.command == "infer":
        from addition_transformer.checkpoint import load_checkpoint

        demo_prompts(load_checkpoint(args.checkpoint))
    else:
        train()