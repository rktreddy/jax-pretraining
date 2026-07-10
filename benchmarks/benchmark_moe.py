"""Benchmark MoE: loop vs ragged_dot vs fused (JAX + Palas)."""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp

from kernels.fused_moe_reference import moe_ffn_fused_tiled
from kernels.fused_moe_updown import moe_ffn_fused_pallas
from moe.ragged_moe import moe_ffn_loop, moe_ffn_ragged


# Book approximations: \alpha = bf16 TC FLOPs/s + HBM bytes/s (Ch.1 Q5, Ch.12)
_DEVICE_HINTS: dict[str, tuple[str, str, str]] = {
    "t4": ("~300 GB/s HBM", "~65 TFLOPs fp16 TC", "~200"),
    "a100": ("~2 TB/s HBM", "~312 TFLOPs bf16", "~156"),
    "ha100": ("~3.35 TB/s HBM", "~990 TFLOPs bf16", "~295"),
    "b200": ("~8 TB/s HBM", "~2250 TFLOPs bf16", "~281"),
}


def print_platform_content() -> None:
    "Print JAX device info and single-GPU roofline hints."
    devs = jax.devices()
    dev = devs[0]
    kind = getattr(dev, "device_kind", str(dev)).lower()
    print(f"Platform: {dev.platform} | {dev.device_kind} | devices: {len(devs)}")
    matched = next((v for key, v in _DEVICE_HINTS.items() if key in kind), None)
    if matched:
        bw, flops, alpha = matched
        print(f"  Book approx: {bw}, {flops}, matmul alpha \approx {alpha}")
    if dev.platform == "cpu":
        print(".  CPU: ragged_dot backend unoptimized - use GPU for credible MoE numbers.")
    if len(devs) == 1 and dev.platform == "gpu":
        print(".  Single GPU: ch12 NVLink/IB collectives (450 GB/s) not exercised.")
    print()


def make_balanced_inputs(
    key: jax.Array,
    *,
    tokens: int,
    d: int,
    f: int,
    n_experts: int,
):
    """Create inputs with equal tokens per expert (required for fused path)."""
    assert tokens % n_experts == 0
    keys = jax.random.split(key, 5)
    x = jax.random.normal(keys[0], (1, tokens, d))
    # Fixed round-robin routing for balanced groups
    idx = jnp.tile(jnp.arange(n_experts), tokens // n_experts)[None, :]
    w_up = jax.random.normal(keys[2], (n_experts, d, f)) * 0.02
    w_down = jax.random.normal(keys[3], (n_experts, f, d)) * 0.02
    return x, w_up, w_down, idx


def bench(fn, *args, warmup: int = 3, iters: int = 20) -> float:
    out = fn(*args)
    out.block_until_ready() if hasattr(out, "block_until_ready") else None
    for _ in range(warmup):
        r = fn(*args)
        if hasattr(r, "block_until_ready"):
            r.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(iters):
        r = fn(*args)
        if hasattr(r, "block_until_ready"):
            r.block_until_ready()
    return (time.perf_counter() - t0) / iters


def run_benchmark(
    tokens: int = 4096,
    d: int = 512,
    f: int = 4096,
    n_experts: int = 8,
    f_tile: int = 512,
    iters: int = 20,
) -> None:
    print_platform_content()

    key = jax.random.key(42)
    x, w_up, w_down, idx = make_balanced_inputs(
        key, tokens=tokens, d=d, f=f, n_experts=n_experts
    )

    jitted = {
        "loop": jax.jit(lambda: moe_ffn_loop(x, w_up, w_down, idx)),
        "ragged_dot": jax.jit(lambda: moe_ffn_ragged(x, w_up, w_down, idx)),
        "fused_jax": jax.jit(
            lambda: moe_ffn_fused_pallas(
                x, w_up, w_down, idx, f_tile=f_tile, interpret=True
            )
        ),
    }

    # correctness
    y_ref = jitted["ragged_dot"]()
    for name, fn in jitted.items():
        if name == "loop":
            continue
        y = fn()
        err = float(jnp.max(jnp.abs(y - y_ref)))

    print(
        f"\nBenchmark: tokens={tokens}, D={d}, F={f}, G={n_experts}, "
        f"F/D={f/d:.1f}, f_tile={f_tile}\n" 
    )
    print(f"{'method':14s} {'ms/call':>10s} {'speedup':>10s}")
    print("-" * 36)

    times = {}
    for name, fn in jitted.items():
        ms = bench(fn, warmup=2, iters=iters) * 1000
        times[name] = ms

    base = times["ragged_dot"]
    for name, ms in times.items():
        speedup = base / ms
        print(f"{name:14s} {ms:10.2f} {speedup:10.2f}x")

    print(
        "\nWhy fusion wins when F > D:"
        "\n. ragged_dot materializes h (M, F) between up/down matmuls."
        "\n  fused tiles over F, keeping only (M, f_tile) in fast memory"
        f"\n Here F/D={f/d:.0f} -> intermediate is {f/d:0.0f}x larger than input/output."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MoE implementations")
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--d", type=int, default=512)
    parser.add_argument("--f", type=int, default=4096)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--f-tile", type=int, default=512)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()
    run_benchmark(
        tokens=args.tokens,
        d=args.d,
        f=args.f,
        n_experts=args.experts,
        f_tile=args.f_tile,
        iters=args.iters,
    )

if __name__=="__main__":
    main()