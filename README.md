# jax-pretraining: addition transformer, Chinchilla laws, and a fused MoE kernel

Working through the exercise from [How to land a job at a frontier lab](https://vladfeinberg.com/2026/05/10/how-to-land-a-job-at-a-frontier-lab.html):
train a ~10M-parameter transformer to do 3-digit addition using only JAX/Flax/Optax,
derive Chinchilla scaling laws for it (dense vs MoE), and write a Pallas kernel that
beats `jax.lax.ragged_dot` by fusing the MoE up/down projections.

All code hand-written for the learning experience (the git history and the
[typo-class bug pattern it started from](#lessons) are the receipts).

## Results

| Exercise | Status | Result |
|---|---|---|
| ~10M transformer learns addition (Colab T4) | done | **96.9%** exact-match after 15k steps (~20 min) |
| Chinchilla laws, dense vs MoE | done | see below + [full report](reports/chinchilla_addition.md) |
| Fused Pallas MoE kernel | kernel done, validated | TPU benchmark pending (T4 cannot compile Pallas - Triton requires Ampere) |

### Scaling-law findings ([report](reports/chinchilla_addition.md))

18-run sweep (3 model sizes x 3 token budgets x {dense, top-1 MoE}), no curriculum,
loss on answer tokens only, `L(N, D) = a (N/N0)^-alpha + b (D/D0)^-beta` with E pinned to 0:

| arch | alpha (params) | beta (data) |
|---|---|---|
| dense | 1.49 | 2.33 |
| MoE (active params) | 1.91 | 2.14 |

1. **Dense and MoE scale the same when N counts active parameters.** Top-1 MoE buys
   ~3x total parameters at matched active-parameter loss - the canonical MoE claim,
   reproduced at toy scale.
2. **Exponents are 5-8x steeper than language-model Chinchilla** (0.34/0.28). Addition
   is deterministic and saturable: loss phase-transitions when the model masters a digit
   class instead of gliding down a power law. Fitting with the language exponents fixed
   fails outright (negative B, complex compute-optimal allocations).
3. **Allocation tilts toward parameters** (beta/(alpha+beta) ~ 0.6), unlike language
   Chinchilla's balanced split - synthetic data is free, model capacity is the bottleneck.

### Kernel design ([kernels/fused_moe_pallas_v2.py](kernels/fused_moe_pallas_v2.py))

`ragged_dot` materializes the hidden activation `H = gelu(X @ Wu)`, shape `(M, F)`, in
HBM between the up and down matmuls. For F > D that round trip is `F/D`x larger than the
input and output combined. The fused kernel tiles over F and keeps each
`(block_m, f_tile)` sliver of H in SRAM. Two variants, because the hardware execution
model shapes the kernel:

- **GPU (Triton)**: grid programs are parallel CTAs, so the F loop lives *inside* the
  kernel; weights stay in HBM (`memory_space=ANY`) and tiles are loaded manually.
- **TPU (Mosaic)**: grid steps run *sequentially*, so F is a third grid axis with a
  VMEM fp32 scratch accumulator (init at first visit, store at last).

Fusion only pays when the op is memory-bound: on a T4, that means fp16 and D ≲ 200
(critical intensity ~217 FLOPs/byte fp16 vs ~27 fp32). The benchmark prints its own
roofline prediction next to the measurement.

## Layout

```
addition_transformer/   model, data, training loop, checkpointing
chinchilla/             sweep, scaling-law fits, report generator
moe/                    routing + ragged_dot / loop MoE FFN reference
kernels/                fused Pallas kernels (GPU + TPU variants)
benchmarks/             MoE FFN benchmark with built-in roofline
tests/                  kernel correctness vs ragged_dot (interpret mode)
notebooks/              Colab notebook (T4 training end-to-end)
reports/                sweep results + generated scaling-law report
```

## Reproduce

```bash
uv sync
uv run pytest                                  # kernel correctness (CPU, interpret mode)
uv run python -m addition_transformer.train    # train the 10M model
uv run python -m chinchilla sweep --quick      # 18-run scaling sweep (hours on CPU)
uv run python -m chinchilla report             # regenerate reports/chinchilla_addition.md

# MoE kernel benchmark (memory-bound setting; needs Ampere+ GPU or TPU)
PYTHONPATH=. uv run python benchmarks/benchmark_moe.py --tokens 65536 --d 64 --f 1024 --dtype f16
```

On Colab: open `notebooks/addition_transformer_colab.ipynb`, pick a T4 runtime for
training (sections 0-4) or a TPU runtime for the kernel benchmark (run it with `%run`,
in-process - the TPU is single-process).

## Lessons

Things this exercise actually taught, mostly the hard way:

- JAX's purity cuts both ways: `state.apply_gradients(...)` with the return value
  dropped trains nothing, silently - a bug class PyTorch's mutability makes impossible,
  caused by the same design that makes `jit`/`vmap`/`ragged_dot` possible.
- Zero-argument jitted lambdas get **constant-folded by XLA** - the "benchmark" times a
  memory fetch. Arrays must be jit arguments.
- Curricula confound everything: they made "tokens seen" mean "syllabus position"
  (corrupting the scaling fits) and caused catastrophic forgetting of the rarest
  format (2-digit sums with a carry - `48 + 58` was the last prompt standing).
- Scaling-law fitting is as much methodology as data: unconstrained least squares
  happily returns a negative irreducible loss.
- Whether kernel fusion pays is a property of the chip's compute:bandwidth ratio,
  not the kernel: the same code flips from "cannot win" to "~3.5x predicted" between
  fp32 and fp16, and between TPU v2 and v5e.
