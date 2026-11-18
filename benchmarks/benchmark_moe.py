"""Benchmark MoE FFN: jax.lax.ragged_dot vs fused Pallas kernel (v2).

The exercise: find a setting where fusing the up/down projections measurably
beats ragged_dot for F > D, and explain why.

Usage (Colab T4, the memory-bound setting):
    PYTHONPATH=. python benchmarks/benchmark_moe.py \
        --tokens 65536 --d 64 --f 1024 --experts 8 --dtype f16

Sweep tile sizes and save results (TPU: include f_tile == F):
    PYTHONPATH=. python benchmarks/benchmark_moe.py \
        --sweep-block-m 64,128,256 --sweep-f-tile 128,1024 --json reports/bench.json

CPU smoke test (interpret mode, timing meaningless):
    PYTHONPATH=. python benchmarks/benchmark_moe.py \
        --tokens 512 --d 32 --f 128 --experts 4 --block-m 32 --f-tile 32
"""

from __future__ import annotations

import argparse
import functools
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from kernels.fused_moe_pallas_v2 import (
    moe_ffn_fused_pallas_tpu,
    moe_ffn_fused_pallas_v2,
    moe_ffn_fused_ragged_tpu,
    moe_ffn_fused_ragged_v2,
)
from moe.ragged_moe import moe_ffn_loop, moe_ffn_ragged

# (fp16/bf16 matmul FLOP/s, fp32 matmul FLOP/s, HBM bytes/s) - approximate peaks
# TPU entries are per core (one Pallas kernel runs on one core).
_DEVICE_PEAKS: dict[str, tuple[float, float, float]] = {
    "t4": (65e12, 8.1e12, 300e9),
    "l4": (121e12, 30e12, 300e9),
    "v100": (125e12, 15.7e12, 900e9),
    "a100": (312e12, 19.5e12, 1.6e12),
    "h100": (990e12, 67e12, 3.35e12),
    "tpu v2": (23e12, 3e12, 350e9),
    "tpu v3": (61e12, 4e12, 450e9),
    "v5 lite": (197e12, 12e12, 819e9),  # v5e reports as "TPU v5 lite"
}

_DTYPES = {"f32": jnp.float32, "f16": jnp.float16, "bf16": jnp.bfloat16}


def device_peaks() -> tuple[str, tuple[float, float, float] | None]:
    dev = jax.devices()[0]
    kind = getattr(dev, "device_kind", str(dev)).lower()
    peaks = next((v for k, v in _DEVICE_PEAKS.items() if k in kind), None)
    return f"{dev.platform}:{dev.device_kind}", peaks


def make_inputs(key, *, tokens, d, f, n_experts, dtype, skew=0.0):
    """skew=0: round-robin, exactly tokens/n_experts per expert.
    skew>0: expert 0 gets ~skew of all tokens, rest split the remainder."""
    assert tokens % n_experts == 0
    keys = jax.random.split(key, 4)
    x = jax.random.normal(keys[0], (tokens, d), dtype=jnp.float32).astype(dtype)
    w_up = (jax.random.normal(keys[1], (n_experts, d, f)) * 0.05).astype(dtype)
    w_down = (jax.random.normal(keys[2], (n_experts, f, d)) * 0.05).astype(dtype)
    if skew > 0:
        p = jnp.array([skew] + [(1 - skew) / (n_experts - 1)] * (n_experts - 1))
        idx = jax.random.choice(keys[3], n_experts, (tokens,), p=p)
    else:
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


def roofline(tokens, d, f, g, itemsize, peaks, *, block_m) -> dict:
    """Predicted times from first principles (see report for derivation).

    Two models for fused weight traffic, bracketing the truth:
      read-once: every expert's Wu/Wd cross HBM once (2*G*D*F elements).
      re-read:   every token block re-loads its expert's weights
                 ((M/block_m) * 2*D*F) - what both kernels literally issue.
    On GPU, L2 can absorb the re-reads when all weights fit (2*G*D*F*itemsize
    vs L2 size), pulling the truth toward read-once. On TPU there is no cache
    between HBM and VMEM: re-read is exact, unless f_tile == F, where the
    weight block index is constant across an expert's consecutive token
    blocks and the Pallas pipeline skips the copy -> read-once.
    The unfused baseline is modelled read-once (XLA's choice, not ours).
    """
    flops = 4 * tokens * d * f  # 2MDF up + 2MDF down
    n_blocks = -(-tokens // block_m)
    xy = 2 * tokens * d
    unfused_elems = xy + 2 * g * d * f + 2 * tokens * f  # X,Y + Wu,Wd + H roundtrip
    fused_elems = xy + 2 * g * d * f  # H never leaves SRAM
    fused_reread_elems = xy + n_blocks * 2 * d * f
    out = {
        "flops": flops,
        "unfused_mb": unfused_elems * itemsize / 1e6,
        "fused_mb": fused_elems * itemsize / 1e6,
        "fused_reread_mb": fused_reread_elems * itemsize / 1e6,
        "weights_mb": 2 * g * d * f * itemsize / 1e6,
        "traffic_ratio": unfused_elems / fused_elems,
    }
    if peaks:
        flops_lo, flops_f32, bw = peaks
        peak = flops_f32 if itemsize == 4 else flops_lo
        t_compute = flops / peak
        out["t_compute_ms"] = t_compute * 1e3
        out["t_unfused_ms"] = max(t_compute, unfused_elems * itemsize / bw) * 1e3
        out["t_fused_ms"] = max(t_compute, fused_elems * itemsize / bw) * 1e3
        out["t_fused_reread_ms"] = max(t_compute, fused_reread_elems * itemsize / bw) * 1e3
        out["pred_speedup"] = out["t_unfused_ms"] / out["t_fused_ms"]
        out["pred_speedup_reread"] = out["t_unfused_ms"] / out["t_fused_reread_ms"]
    return out


def _expected_model(platform: str, f: int, f_tile: int) -> str:
    """Which roofline model should hold for this config (see roofline)."""
    if platform == "tpu":
        return "read-once" if f_tile == f else "re-read"
    return "between (L2-dependent)"


def run_benchmark(
    tokens: int,
    d: int,
    f: int,
    n_experts: int,
    block_ms: list[int],
    f_tiles: list[int],
    iters: int,
    dtype_name: str,
    interpret: bool | None,
    skew: float = 0.0,
) -> dict:
    """Time baselines once, then the fused kernels at every (block_m, f_tile).

    Returns a JSON-serializable dict of the config, timings and predictions.
    """
    kind, peaks = device_peaks()
    platform = jax.devices()[0].platform
    if platform == "tpu" and dtype_name == "f16":
        print("TPU has no fp16 - switching to bf16")
        dtype_name = "bf16"
    dtype = _DTYPES[dtype_name]
    itemsize = jnp.dtype(dtype).itemsize
    if interpret is None:
        interpret = platform == "cpu"
    fused_impl = moe_ffn_fused_pallas_tpu if platform == "tpu" else moe_ffn_fused_pallas_v2
    ragged_impl = moe_ffn_fused_ragged_tpu if platform == "tpu" else moe_ffn_fused_ragged_v2

    print(f"Device: {kind} | dtype={dtype_name} | interpret={interpret} "
          f"| kernels={fused_impl.__name__},{ragged_impl.__name__}")
    if interpret:
        print("  (interpret mode: correctness only, timings are meaningless)")
    print(
        f"Shapes: M={tokens} D={d} F={f} G={n_experts} (F/D={f / d:.0f}) | "
        f"block_m={block_ms} f_tile={f_tiles} | skew={skew}\n"
    )

    key = jax.random.key(42)
    x, w_up, w_down, idx = make_inputs(
        key, tokens=tokens, d=d, f=f, n_experts=n_experts, dtype=dtype, skew=skew
    )

    # IMPORTANT: arrays must be jit *arguments*, not closed-over constants -
    # zero-arg jitted lambdas get constant-folded by XLA and time nothing.
    fns = {
        "loop": (jax.jit(moe_ffn_loop), None),
        "ragged_dot": (jax.jit(moe_ffn_ragged), None),
    }
    for block_m in block_ms:
        for f_tile in f_tiles:
            if f % f_tile:
                print(f"  skip block_m={block_m} f_tile={f_tile}: F % f_tile != 0")
                continue
            cfg = (block_m, f_tile)
            fns[f"fused_ragged bm={block_m} ft={f_tile}"] = (
                jax.jit(functools.partial(
                    ragged_impl, block_m=block_m, f_tile=f_tile, interpret=interpret,
                )),
                cfg,
            )
            # balanced-only kernel: asserts equal per-expert groups, so it can
            # only run under round-robin routing
            if skew == 0 and (tokens // n_experts) % block_m == 0:
                fns[f"fused_pallas bm={block_m} ft={f_tile}"] = (
                    jax.jit(functools.partial(
                        fused_impl, block_m=block_m, f_tile=f_tile, interpret=interpret,
                    )),
                    cfg,
                )
    args = (x, w_up, w_down, idx)

    # correctness vs fp32 loop reference; a kernel that fails to compile
    # (e.g. Mosaic tiling, Triton SMEM) is reported and dropped, not fatal
    ref = moe_ffn_loop(
        x.astype(jnp.float32), w_up.astype(jnp.float32), w_down.astype(jnp.float32), idx
    )
    tol = 1e-4 if itemsize == 4 else 3e-2
    errors = {}
    for name, (fn, _) in list(fns.items()):
        try:
            err = float(jnp.max(jnp.abs(fn(*args).astype(jnp.float32) - ref)))
        except Exception as e:  # noqa: BLE001 - report any lowering failure
            msg = str(e).strip().splitlines()[0][:200] if str(e).strip() else type(e).__name__
            print(f"  correctness {name:24s} FAILED: {msg}")
            errors[name] = msg
            del fns[name]
            continue
        status = "ok" if err < tol else "MISMATCH"
        print(f"  correctness {name:24s} max_abs_err={err:.2e}  [{status}]")
    print()

    times = {name: bench(fn, *args, iters=iters) * 1e3 for name, (fn, _) in fns.items()}
    base = times["ragged_dot"]
    print(f"{'method':26s} {'ms/call':>10s} {'vs ragged_dot':>14s}")
    print("-" * 52)
    for name, ms in times.items():
        print(f"{name:26s} {ms:10.3f} {base / ms:13.2f}x")

    rooflines = {bm: roofline(tokens, d, f, n_experts, itemsize, peaks, block_m=bm)
                 for bm in block_ms}
    r0 = rooflines[block_ms[0]]
    print(
        f"\nRoofline: {r0['flops'] / 1e9:.1f} GFLOP | unfused {r0['unfused_mb']:.0f} MB "
        f"| fused read-once {r0['fused_mb']:.0f} MB | all weights {r0['weights_mb']:.2f} MB"
    )
    rows = []
    if "pred_speedup" in r0:
        print(f"Compute floor {r0['t_compute_ms']:.3f} ms | unfused predicted "
              f"{r0['t_unfused_ms']:.3f} ms, measured {base:.3f} ms\n")
        print(f"{'fused kernel':26s} {'re-read MB':>10s} {'pred once':>10s} "
              f"{'pred reread':>11s} {'measured':>9s}  expected")
        print("-" * 86)
    for name, (_, cfg) in fns.items():
        if cfg is None:
            continue
        block_m, f_tile = cfg
        r = rooflines[block_m]
        row = {"name": name, "block_m": block_m, "f_tile": f_tile,
               "ms": times[name], "speedup": base / times[name],
               "fused_reread_mb": r["fused_reread_mb"],
               "expected_model": _expected_model(platform, f, f_tile)}
        if "pred_speedup" in r:
            row |= {"pred_speedup_once": r["pred_speedup"],
                    "pred_speedup_reread": r["pred_speedup_reread"]}
            print(f"{name:26s} {r['fused_reread_mb']:10.0f} {r['pred_speedup']:9.2f}x "
                  f"{r['pred_speedup_reread']:10.2f}x {row['speedup']:8.2f}x  "
                  f"{row['expected_model']}")
        rows.append(row)
    print(
        "\nWhy fusion wins when F > D (and the op is memory-bound):"
        "\n  ragged_dot materializes H = gelu(X @ Wu), shape (M, F), in HBM between"
        "\n  the up and down matmuls - a 2*M*F-element round trip that is F/D times"
        "\n  larger than X and Y combined. The fused kernel tiles over F and keeps"
        "\n  each (block_m, f_tile) sliver of H in SRAM, deleting that traffic."
        "\n  What it adds back: each token block re-loads its expert's weights, so"
        "\n  weight traffic scales with M/block_m unless a cache (GPU L2) or an"
        "\n  unchanged block index (TPU, f_tile == F) dedupes it."
        "\n  When compute-bound (fp32, or large D), the saving hides behind the"
        "\n  compute wall and fusion cannot win - dtype and D pick the regime."
    )
    return {
        "device": kind, "platform": platform, "dtype": dtype_name,
        "interpret": interpret, "tokens": tokens, "d": d, "f": f,
        "experts": n_experts, "skew": skew, "iters": iters,
        "times_ms": times, "kernels": rows, "errors": errors,
        "roofline": {str(bm): r for bm, r in rooflines.items()},
    }


def _int_list(s: str) -> list[int]:
    return [int(v) for v in s.split(",") if v]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MoE FFN implementations")
    parser.add_argument("--tokens", type=int, default=65536)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--f", type=int, default=1024)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--block-m", type=int, default=128)
    parser.add_argument("--f-tile", type=int, default=128)
    parser.add_argument(
        "--sweep-block-m", type=_int_list, default=None, metavar="BM,BM,...",
        help="Comma-separated block_m values to sweep (overrides --block-m)",
    )
    parser.add_argument(
        "--sweep-f-tile", type=_int_list, default=None, metavar="FT,FT,...",
        help="Comma-separated f_tile values to sweep (overrides --f-tile); "
        "on TPU include F itself to test the weights-fetched-once-per-expert case",
    )
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--dtype", choices=list(_DTYPES), default="f16")
    parser.add_argument(
        "--skew", type=float, default=0.0,
        help="Fraction of tokens routed to expert 0 (0 = balanced round-robin; "
        "skewed routing drops the balanced-only fused_pallas kernel)",
    )
    parser.add_argument(
        "--interpret", action=argparse.BooleanOptionalAction, default=None,
        help="Force Pallas interpret mode (default: auto - on for CPU, off for GPU)",
    )
    parser.add_argument("--json", type=Path, default=None, help="Write results as JSON here")
    args = parser.parse_args()
    result = run_benchmark(
        tokens=args.tokens,
        d=args.d,
        f=args.f,
        n_experts=args.experts,
        block_ms=args.sweep_block_m or [args.block_m],
        f_tiles=args.sweep_f_tile or [args.f_tile],
        iters=args.iters,
        dtype_name=args.dtype,
        interpret=args.interpret,
        skew=args.skew,
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
