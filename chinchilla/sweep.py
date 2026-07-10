"""Run Chinchilla scaling sweeps over model size and token budget."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from addition_transformer.train import TrainConfig, run_training
from chinchilla.metrics import RunMetrics
from chinchilla.presets import (
    DENSE_PRESETS,
    MOE_EXPERTS,
    ModelPreset,
    TOKEN_BUDGETS,
    curriculum_for_steps,
    step_for_token_budget,
)


RESULTS_PATH = Path("reports/chinchilla_results.json")


@dataclass
class SweepSpec:
    preset: ModelPreset
    token_budget: int
    architecture: str = "dense"
    batch_size: int = 128
    seed: int = 0


def build_runs(
        presets: tuple[ModelPreset, ...] = DENSE_PRESETS,
        token_budgets: tuple[int, ...] = TOKEN_BUDGETS,
        architectures: tuple[str, ...] = ("dense", "moe")
) -> list[SweepSpec]:
    specs = []
    for arch in architectures:
        for preset in presets:
            for budget in token_budgets:
                specs.append(
                    SweepSpec(preset=preset, token_budget=budget, architecture=arch)
                )
    return specs


def run_single(spec: SweepSpec, *, quiet: bool = True) -> RunMetrics:
    max_steps = step_for_token_budget(spec.token_budget, spec.batch_size)
    s1, s2 = 0, 0
    cfg = TrainConfig(
        d_model=spec.preset.d_model,
        n_layers=spec.preset.n_layers,
        n_heads=spec.preset.n_heads,
        d_ff=spec.preset.d_ff,
        batch_size=spec.batch_size,
        max_steps=max_steps,
        curriculum_stage1_steps=s1,
        curriculum_stage2_steps=s2,
        eval_every=max_steps,
        log_every=max_steps,
        seed=spec.seed,
        architecture=spec.architecture,
        n_experts=MOE_EXPERTS,
        save_checkpoint_at_end=False,
    )
    print(
        f"\n>>> {spec.architecture} | {spec.preset.name} | "
        f"budget={spec.token_budget:,} tokens | steps={max_steps}"
    )
    t0 = time.time()
    _, metrics = run_training(cfg, quiet=quiet)
    elapsed = time.time() - t0
    print(
        f".    done in {elapsed:.0f}s | params={metrics['n_params']:,} "
        f"| active={metrics['n_active_params']:,} | loss={metrics['final_loss']:.4f} "
        f"| acc={metrics['addition_acc']:.3f}"
    )
    return RunMetrics(
        architecture=spec.architecture,
        preset=spec.preset.name,
        n_params=metrics["n_params"],
        n_active_params=metrics["n_active_params"],
        tokens=metrics["tokens"],
        steps=metrics["steps"],
        batch_size=metrics["batch_size"],
        final_loss=metrics["final_loss"],
        addition_acc=metrics["addition_acc"],
        flops=metrics["flops"],
    )


def run_sweep(
    specs: list[SweepSpec] | None = None,
    *, 
    output: Path = RESULTS_PATH,
    resume: bool = True,
) -> list[dict]:
    specs = specs or build_runs()
    output.parent.mkdir(parents=True, exist_ok=True)

    existing: list[dict] = []
    done_keys: set[str] = set[str]()
    if resume and output.exists():
        existing = json.loads(output.read_text())
        done_keys = {r["key"] for r in existing}

    results = list[dict](existing)
    for spec in specs:
        key = f"{spec.architecture}:{spec.preset.name}:{spec.token_budget}"
        if key in done_keys:
            print("skip {key} (already done)")
            continue
        run = run_single(spec)
        row = run.to_dict()
        row["key"] = key
        row["token_budget"] = spec.token_budget
        results.append(row)
        output.write_text(json.dumps(results, indent=2) + "\n")

    return results


def quick_sweep() -> list[dict]:
    """Reduced grid for local CPU (~1-2 hours)."""
    presets = (DENSE_PRESETS[0], DENSE_PRESETS[1], DENSE_PRESETS[2])
    budgets = (500_000, 1_000_000, 2_000_000)
    specs = []
    for arch in ("dense", "moe"):
        for preset in presets:
            for budget in budgets:
                specs.append(SweepSpec(preset=preset, token_budget=budget, architecture=arch))
    return run_sweep(specs)