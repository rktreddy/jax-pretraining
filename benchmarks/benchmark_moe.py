"""Benchmark MoE FFN: jax.lax.ragged_dot vs fused Pallas kernel (v2).

The exercise: find a setting where fusing the up/down projections measurably
beats ragged_dot for F > D, and explain why.

Usage (Colab T4, the memory-bound setting):
    PYTHONPATH=. python benchmarks/benchmark_moe.py \
        --tokens 65536 --d 64 --f 1024 --experts 8 --dtype f16

CPU smoke test (interpret mode, timing meaningless):
    PYTHONPATH=. python benchmarks/benchmark_moe.py \
        --tokens 512 --d 32 --f 128 --experts 4 --block-m 32 --f-tile 32
"""

from __future__ import annotations

import argparse
import functools
import time

import jax
import jax.numpy as jnp

from kernels.fused_moe_pallas_v2 import moe_ffn_fused_pallas_v2
from moe.ragged_moe import moe_ffn_loop, moe_ffn_ragged

# (fp16/bf16 matmul FLOP/s, fp32 matmul FLOP/s, HBM bytes/s) - approximate peaks
_DEVICE_PEAKS: dict[str, tuple[float, float, float]] = {
    "t4": (65e12, 8.1e12, 300e9),
    "l4": (121e12, 30e12, 300e9),
    "v100": (125e12, 15.7e12, 900e9),
    "a100": (312e12, 19.5e12, 1.6e12),
    "h100": (990e12, 67e12, 3.35e12),
}

_DTYPES = {"f32": jnp.float32, "f16": jnp.float16, "bf16": jnp.bfloat16}


def device_peaks() -> tuple[str, tuple[float, float, float] | None]:
    dev = jax.devices()[0]
    kind = getattr(dev, "device_kind", str(dev)).lower()
    peaks = next((v for k, v in _DEVICE_PEAKS.items() if k in kind), None)
    return f"{dev.platform}:{dev.device_kind}", peaks


def make_balanced_inputs(key, *, tokens, d, f, n_experts, dtype):
    """Round-robin routing -> exactly tokens/n_experts per expert."""
    assert tokens % n_experts == 0
    keys = jax.random.split(key, 3)
    x = jax.random.normal(keys[0], (tokens, d), dtype=jnp.float32).astype(dtype)
    w_up = (jax.random.normal(keys[1], (n_experts, d, f)) * 0.05).astype(dtype)
    w_down = (jax.random.normal(keys[2], (n_experts, f, d)) * 0.05).astype(dtype)
    idx = jnp.tile(jnp.arange(n_experts), tokens // n_experts)
    return x, w_up, w_down, idx


def bench(fn, *args, warmup: int = 3, iters: int = 20) -> float:
    """Mean seconds per call over `iters` timed calls, after warmup."""
    for _ in range(warmup):
        fn(*args).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args).block_until_ready()
    return (time.perf_counter() - t0) / iters


def roofline(tokens, d, f, g, itemsize, peaks) -> dict:
    """Predicted times from first principles (see report for derivation)."""
    flops = 4 * tokens * d * f  # 2MDF up + 2MDF down
    unfused_elems = 2 * tokens * d + 2 * g * d * f + 2 * tokens * f  # X,Y + Wu,Wd + H roundtrip
    fused_elems = 2 * tokens * d + 2 * g * d * f  # H never leaves SRAM
    out = {
        "flops": flops,
        "unfused_mb": unfused_elems * itemsize / 1e6,
        "fused_mb": fused_elems * itemsize / 1e6,
        "traffic_ratio": unfused_elems / fused_elems,
    }
    if peaks:
        flops_lo, flops_f32, bw = peaks
        peak = flops_f32 if itemsize == 4 else flops_lo
        t_compute = flops / peak
        out["t_unfused_ms"] = max(t_compute, unfused_elems * itemsize / bw) * 1e3
        out["t_fused_ms"] = max(t_compute, fused_elems * itemsize / bw) * 1e3
        out["pred_speedup"] = out["t_unfused_ms"] / out["t_fused_ms"]
    return out


def run_benchmark(
    tokens: int,
    d: int,
    f: int,
    n_experts: int,
    block_m: int,
    f_tile: int,
    iters: int,
    dtype_name: str,
    interpret: bool | None,
) -> None:
    kind, peaks = device_peaks()
    dtype = _DTYPES[dtype_name]
    itemsize = jnp.dtype(dtype).itemsize
    if interpret is None:
        interpret = jax.devices()[0].platform == "cpu"

    print(f"Device: {kind} | dtype={dtype_name} | interpret={interpret}")
    if interpret:
        print("  (interpret mode: correctness only, timings are meaningless)")
    print(
        f"Shapes: M={tokens} D={d} F={f} G={n_experts} (F/D={f / d:.0f}) | "
        f"block_m={block_m} f_tile={f_tile}\n"
    )

    key = jax.random.key(42)
    x, w_up, w_down, idx = make_balanced_inputs(
        key, tokens=tokens, d=d, f=f, n_experts=n_experts, dtype=dtype
    )

    # IMPORTANT: arrays must be jit *arguments*, not closed-over constants -
    # zero-arg jitted lambdas get constant-folded by XLA and time nothing.
    fns = {
        "loop": jax.jit(moe_ffn_loop),
        "ragged_dot": jax.jit(moe_ffn_ragged),
        "fused_pallas": jax.jit(
            functools.partial(
                moe_ffn_fused_pallas_v2,
                block_m=block_m, f_tile=f_tile, interpret=interpret,
            )
        ),
    }
    args = (x, w_up, w_down, idx)

    # correctness vs fp32 loop reference
    ref = moe_ffn_loop(
        x.astype(jnp.float32), w_up.astype(jnp.float32), w_down.astype(jnp.float32), idx
    )
    tol = 1e-4 if itemsize == 4 else 3e-2
    for name, fn in fns.items():
        err = float(jnp.max(jnp.abs(fn(*args).astype(jnp.float32) - ref)))
        status = "ok" if err < tol else "MISMATCH"
        print(f"  correctness {name:14s} max_abs_err={err:.2e}  [{status}]")
    print()

    times = {name: bench(fn, *args, iters=iters) * 1e3 for name, fn in fns.items()}
    base = times["ragged_dot"]
    print(f"{'method':16s} {'ms/call':>10s} {'vs ragged_dot':>14s}")
    print("-" * 42)
    for name, ms in times.items():
        print(f"{name:16s} {ms:10.3f} {base / ms:13.2f}x")

    r = roofline(tokens, d, f, n_experts, itemsize, peaks)
    print(
        f"\nRoofline: {r['flops'] / 1e9:.1f} GFLOP | "
        f"HBM traffic {r['unfused_mb']:.0f} MB unfused vs {r['fused_mb']:.0f} MB fused "
        f"({r['traffic_ratio']:.1f}x less)"
    )
    if "pred_speedup" in r:
        print(
            f"Predicted: unfused {r['t_unfused_ms']:.2f} ms, fused {r['t_fused_ms']:.2f} ms "
            f"-> {r['pred_speedup']:.1f}x speedup"
            f"\nMeasured:  {base:.2f} ms vs {times['fused_pallas']:.2f} ms "
            f"-> {base / times['fused_pallas']:.1f}x"
        )
    print(
        "\nWhy fusion wins when F > D (and the op is memory-bound):"
        "\n  ragged_dot materializes H = gelu(X @ Wu), shape (M, F), in HBM between"
        "\n  the up and down matmuls - a 2*M*F-element round trip that is F/D times"
        "\n  larger than X and Y combined. The fused kernel tiles over F and keeps"
        "\n  each (block_m, f_tile) sliver of H in SRAM, deleting that traffic."
        "\n  When compute-bound (fp32, or large D), the saving hides behind the"
        "\n  compute wall and fusion cannot win - dtype and D pick the regime."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MoE FFN implementations")
    parser.add_argument("--tokens", type=int, default=65536)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--f", type=int, default=1024)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--block-m", type=int, default=128)
    parser.add_argument("--f-tile", type=int, default=128)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--dtype", choices=list(_DTYPES), default="f16")
    parser.add_argument(
        "--interpret", action=argparse.BooleanOptionalAction, default=None,
        help="Force Pallas interpret mode (default: auto - on for CPU, off for GPU)",
    )
    args = parser.parse_args()
    run_benchmark(
        tokens=args.tokens,
        d=args.d,
        f=args.f,
        n_experts=args.experts,
        block_m=args.block_m,
        f_tile=args.f_tile,
        iters=args.iters,
        dtype_name=args.dtype,
        interpret=args.interpret,
    )


if __name__ == "__main__":
    main()
