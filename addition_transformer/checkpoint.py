"""Save and load trained addition transformer checkpoints."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import jax
import orbax.checkpoint as ocp

from addition_transformer.train import TrainConfig, TrainState, make_train_state

DEFAULT_CHECKPOINT_DIR = Path("checkpoints/addition_transformer")


def _config_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "config.json"


def _params_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "params"


def _resolve(checkpoint_dir: str | Path) -> Path:
    return Path(checkpoint_dir).expanduser().resolve()


def save_checkpoint(
    state: TrainState,
    config: TrainConfig,
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
    step: int | None = None,
) -> Path:
    """Save model params and training config to `checkpoint_dir`."""
    checkpoint_dir = _resolve(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    _config_path(checkpoint_dir).write_text(
        json.dumps(asdict(config), indent=2) + "\n"
    )
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(_params_path(checkpoint_dir), state.params)
    checkpointer.wait_until_finished()
    if step is not None:
        (checkpoint_dir / "step.txt").write_text(f"{step}\n")

    print(f"Saved checkpoint to {checkpoint_dir}")
    return checkpoint_dir


def load_config(checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR) -> TrainConfig:
    checkpoint_dir = _resolve(checkpoint_dir)
    data = json.loads(_config_path(checkpoint_dir).read_text())
    return TrainConfig(**data)


def load_checkpoint(
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
) -> TrainState:
    """Restore a TrainState from disk (ready for inference or continued training)."""
    checkpoint_dir = _resolve(checkpoint_dir)
    config = load_config(checkpoint_dir)
    state = make_train_state(jax.random.key(0), config)
    # Restore against abstract shapes so checkpoints saved on one device type
    # (e.g. cuda) load on another (e.g. cpu).
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    abstract_params = jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=sharding),
        state.params,
    )
    params = ocp.StandardCheckpointer().restore(
        _params_path(checkpoint_dir), abstract_params
    )
    return state.replace(params=params)

